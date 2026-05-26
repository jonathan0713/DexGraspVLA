from typing import Dict
import torch
import numpy as np
import copy
import json
from controller.common.pytorch_util import dict_apply
from controller.common.streaming_replay_buffer import StreamingReplayBuffer
from controller.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from controller.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from controller.dataset.base_dataset import BaseImageDataset
import torch.nn.functional as F

class MaskImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(518, 518),
            use_grasp_xy_aux=False,
            use_pregrasp_delta_aux=False,
            use_pregrasp_joint_delta_aux=False,
            pregrasp_aux_max_joint_dist=0.5,
            pregrasp_aux_min_ratio=0.05,
            pregrasp_aux_max_ratio=0.8
            ):
        
        super().__init__()
        self.image_size = image_size
        self.use_grasp_xy_aux = use_grasp_xy_aux
        self.use_pregrasp_delta_aux = False
        self.use_pregrasp_joint_delta_aux = use_pregrasp_joint_delta_aux
        self.pregrasp_aux_max_joint_dist = float(pregrasp_aux_max_joint_dist)
        self.pregrasp_aux_min_ratio = float(pregrasp_aux_min_ratio)
        self.pregrasp_aux_max_ratio = float(pregrasp_aux_max_ratio)
        if self.use_grasp_xy_aux and self.use_pregrasp_joint_delta_aux:
            raise ValueError(
                "Only one auxiliary path can be active: disable use_grasp_xy_aux "
                "or use_pregrasp_joint_delta_aux."
            )
        if use_pregrasp_delta_aux:
            print(
                "Warning: use_pregrasp_delta_aux is deprecated and ignored. "
                "Use use_pregrasp_joint_delta_aux instead."
            )
        
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Initialize auxiliary loss data structures
        self.target_grasp_xy_per_episode = []  # List of dicts indexed by zarr_idx and episode_idx
        self.target_arm_joints_per_episode = []  # List of dicts indexed by zarr_idx and episode_idx
        self.pregrasp_mask_per_zarr = []  # List of per-frame masks indexed by zarr_idx
        self.frame_to_episode = []  # List of arrays indexed by zarr_idx, mapping frame idx to episode idx
        self.sampler_to_episode_map = []  # Maps sampler idx to (zarr_idx, episode_idx)
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['right_cam_img', 'rgbm', 'right_state', 'action'])
            self.replay_buffers.append(replay_buffer)
            
            # Process target poses and create auxiliary structures
            self._process_target_poses_for_zarr(
                replay_buffer, zarr_idx=len(self.replay_buffers)-1)
            
            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)
            self.train_masks.append(train_mask)
            
            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=dict(right_cam_img=n_obs_steps, rgbm=n_obs_steps))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        
    def _decode_target_pose(self, target_pose):
        if isinstance(target_pose, np.ndarray) and target_pose.shape == ():
            target_pose = target_pose.item()
        if isinstance(target_pose, bytes):
            target_pose = target_pose.decode("utf-8")
        if isinstance(target_pose, str):
            target_pose = json.loads(target_pose)
        return target_pose

    def _process_target_poses_for_zarr(self, replay_buffer, zarr_idx):
        """
        Load target poses from meta and build training-only auxiliary targets.
        """
        target_poses = replay_buffer.meta.get('target_poses', None)
        episode_ends = replay_buffer.episode_ends
        
        # Initialize storage for this zarr
        target_grasp_xy_dict = {}
        target_arm_joints_dict = {}
        frame_to_episode = np.zeros(len(replay_buffer), dtype=np.int32)
        pregrasp_mask = np.zeros(len(replay_buffer), dtype=np.float32)
        
        if target_poses is None:
            target_poses = []
        
        # Build frame_to_episode mapping and extract privileged auxiliary targets.
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        right_state_all = replay_buffer['right_state']
        
        for episode_idx in range(len(episode_ends)):
            episode_start = int(episode_starts[episode_idx])
            episode_end = int(episode_ends[episode_idx])
            episode_length = max(episode_end - episode_start, 0)
            
            # Map frames to episode
            frame_to_episode[episode_start:episode_end] = episode_idx
            
            target_arm_joints = None
            t_grasp = -1
            min_dist = float("inf")
            pregrasp_ratio = 0.0
            valid_aux_episode = False
            invalid_reasons = []

            # Extract targets from this episode's target pose.
            if episode_idx < len(target_poses):
                target_pose = target_poses[episode_idx]
                try:
                    target_pose = self._decode_target_pose(target_pose)

                    if isinstance(target_pose, dict) and 'cartesian_pose_xyzw' in target_pose:
                        cartesian_pose = target_pose['cartesian_pose_xyzw']
                        if isinstance(cartesian_pose, (list, tuple, np.ndarray)) and len(cartesian_pose) >= 2:
                            grasp_xy = np.array(cartesian_pose[:2], dtype=np.float32)
                            target_grasp_xy_dict[episode_idx] = grasp_xy
                    if isinstance(target_pose, dict) and 'arm_joints' in target_pose:
                        raw_arm_joints = target_pose['arm_joints']
                        if isinstance(raw_arm_joints, (list, tuple, np.ndarray)) and len(raw_arm_joints) >= 6:
                            target_arm_joints = np.array(raw_arm_joints[:6], dtype=np.float32)
                            target_arm_joints_dict[episode_idx] = target_arm_joints
                except Exception as e:
                    print(f"Warning: Failed to parse target_pose for episode {episode_idx}: {e}")
                    invalid_reasons.append(f"target_pose_parse_failed:{e}")
            else:
                invalid_reasons.append("missing_target_pose")

            if target_arm_joints is None:
                invalid_reasons.append("missing_or_invalid_arm_joints")

            if episode_length <= 0:
                invalid_reasons.append("empty_episode")

            if self.use_pregrasp_joint_delta_aux and target_arm_joints is not None and episode_length > 0:
                episode_states = np.asarray(
                    right_state_all[episode_start:episode_end, :6],
                    dtype=np.float32,
                )
                if episode_states.shape[0] == episode_length and episode_states.shape[1] == 6:
                    distances = np.linalg.norm(episode_states - target_arm_joints[None, :], axis=1)
                    t_grasp = int(np.argmin(distances))
                    min_dist = float(distances[t_grasp])
                    pregrasp_ratio = float((t_grasp + 1) / episode_length)

                    if min_dist > self.pregrasp_aux_max_joint_dist:
                        invalid_reasons.append(
                            f"min_dist>{self.pregrasp_aux_max_joint_dist}"
                        )
                    if pregrasp_ratio < self.pregrasp_aux_min_ratio:
                        invalid_reasons.append(
                            f"pregrasp_ratio<{self.pregrasp_aux_min_ratio}"
                        )
                    if pregrasp_ratio > self.pregrasp_aux_max_ratio:
                        invalid_reasons.append(
                            f"pregrasp_ratio>{self.pregrasp_aux_max_ratio}"
                        )

                    valid_aux_episode = len(invalid_reasons) == 0
                    if valid_aux_episode:
                        pregrasp_mask[episode_start:episode_start + t_grasp + 1] = 1.0
                else:
                    invalid_reasons.append("right_state_shape_invalid")

            if self.use_pregrasp_joint_delta_aux:
                reason = "none" if valid_aux_episode else ",".join(invalid_reasons)
                print(
                    "[PREGRASP_JOINT_DELTA_AUX DATASET] "
                    f"zarr_idx={zarr_idx} episode_idx={episode_idx} "
                    f"target_arm_joints={None if target_arm_joints is None else target_arm_joints.tolist()} "
                    f"episode_length={episode_length} "
                    f"t_grasp={t_grasp} "
                    f"min_dist={min_dist:.6f} "
                    f"pregrasp_ratio={pregrasp_ratio:.6f} "
                    f"valid_aux_episode={valid_aux_episode} "
                    f"invalid_reason={reason}"
                )
        
        self.target_grasp_xy_per_episode.append(target_grasp_xy_dict)
        self.target_arm_joints_per_episode.append(target_arm_joints_dict)
        self.pregrasp_mask_per_zarr.append(pregrasp_mask)
        self.frame_to_episode.append(frame_to_episode)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict(right_cam_img=self.n_obs_steps, rgbm=self.n_obs_steps))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

    def _process_mask_image_batch(self, images):
        """Process images in batch"""
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        mask = torch.from_numpy(images[..., 3:]).float() # [T, H, W, 1]
        
        # Process RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # Process mask
        mask = mask.permute(0, 3, 1, 2)  # [T, 1, H, W]
        mask = F.interpolate(
            mask,
            size=self.image_size,
            mode='nearest'
        )

        mask = (mask > 0.5).float()

        # Combine
        combined = torch.cat([rgb, mask], dim=1)  # [T, 4, H, W]
        return combined.numpy()

    def _process_image_batch(self, images):
        """Process images in batch"""
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
        # Process RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        return rgb.numpy()
    
    def _sample_to_data(self, sample, zarr_idx, frame_idx):
        """
        Convert sampler sample to data dict, including auxiliary targets if needed.
        
        Args:
            sample: output from SequenceSampler.sample_sequence
            zarr_idx: which zarr file this sample came from
            frame_idx: frame index in the global replay buffer for this zarr
        """
        right_state = sample['right_state'].astype(np.float32)
        T_slice = slice(self.n_obs_steps)

        # Process all images in batch
        mask_processed_frames = self._process_mask_image_batch(sample['rgbm'][T_slice])
        processed_frames = self._process_image_batch(sample['right_cam_img'][T_slice])

        data = {
            'obs': {
                'rgbm': mask_processed_frames,
                'right_cam_img': processed_frames,
                'right_state': right_state[T_slice]
            },
            'action': sample['action'].astype(np.float32)
        }
        
        # Add training-only auxiliary grasp workspace target.
        if self.use_grasp_xy_aux:
            # Get episode index for this frame
            episode_idx = int(self.frame_to_episode[zarr_idx][frame_idx])
            
            # Get target grasp XY for this episode.
            target_grasp_xy_dict = self.target_grasp_xy_per_episode[zarr_idx]
            
            if episode_idx in target_grasp_xy_dict:
                grasp_xy = target_grasp_xy_dict[episode_idx].astype(np.float32)
                grasp_xy_valid = 1.0
            else:
                # No target pose for this episode
                grasp_xy = np.zeros(2, dtype=np.float32)
                grasp_xy_valid = 0.0
            
            data['grasp_xy'] = grasp_xy
            data['grasp_xy_valid'] = np.array(grasp_xy_valid, dtype=np.float32)

        if self.use_pregrasp_joint_delta_aux:
            episode_idx = int(self.frame_to_episode[zarr_idx][frame_idx])
            target_arm_joints_dict = self.target_arm_joints_per_episode[zarr_idx]
            target_arm_joints = target_arm_joints_dict.get(
                episode_idx,
                np.zeros(6, dtype=np.float32),
            ).astype(np.float32)
            pregrasp_mask = self.pregrasp_mask_per_zarr[zarr_idx][frame_idx]

            data['target_arm_joints'] = target_arm_joints
            data['pregrasp_mask'] = np.array(pregrasp_mask, dtype=np.float32)
        
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        # Merge all data
        actions = []
        right_states = []
        for rb in self.replay_buffers:
            actions.append(rb['action'])
            right_states.append(rb['right_state'])
            
        data = {
            'action': np.concatenate(actions, axis=0),
            'right_state': np.concatenate(right_states, axis=0)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if self.use_grasp_xy_aux:
            grasp_xy = []
            for zarr_idx, train_mask in enumerate(self.train_masks):
                target_grasp_xy_dict = self.target_grasp_xy_per_episode[zarr_idx]
                train_episode_idxs = np.nonzero(train_mask)[0]
                for episode_idx in train_episode_idxs:
                    if int(episode_idx) in target_grasp_xy_dict:
                        grasp_xy.append(target_grasp_xy_dict[int(episode_idx)])

            if len(grasp_xy) == 0:
                print(
                    "Warning: use_grasp_xy_aux=True but no valid "
                    "target_pose['cartesian_pose_xyzw'] values were found in train episodes."
                )
                grasp_xy = np.zeros((1, 2), dtype=np.float32)
            else:
                grasp_xy = np.stack(grasp_xy, axis=0).astype(np.float32)

            mean = torch.from_numpy(grasp_xy.mean(axis=0)).float()
            std = torch.from_numpy(grasp_xy.std(axis=0)).float()
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            scale = 1.0 / std
            offset = -mean * scale
            input_stats_dict = {
                'min': torch.from_numpy(grasp_xy.min(axis=0)).float(),
                'max': torch.from_numpy(grasp_xy.max(axis=0)).float(),
                'mean': mean,
                'std': std
            }
            normalizer['grasp_xy'] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=input_stats_dict
            )
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler and get zarr index
        curr_idx = idx
        zarr_idx_local = 0
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                zarr_idx_local = i
                break
            curr_idx -= length
        
        # Get the sample from the appropriate sampler
        sampler = self.samplers[zarr_idx_local]
        sample = sampler.sample_sequence(curr_idx)
        
        # Get the frame index from the sampler indices
        buffer_start_idx, buffer_end_idx, _, _ = sampler.indices[curr_idx]
        # Use the start frame of the sequence as the key frame for auxiliary targets
        frame_idx = buffer_start_idx
        
        # Convert sample to data with auxiliary targets
        data = self._sample_to_data(sample, zarr_idx_local, frame_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return sum(self.sampler_lens)
