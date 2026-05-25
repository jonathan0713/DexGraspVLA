from typing import Dict
import torch
import numpy as np
import copy
import json
from controller.common.pytorch_util import dict_apply
from controller.common.streaming_replay_buffer import StreamingReplayBuffer
from controller.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from controller.model.common.normalizer import LinearNormalizer
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
            use_pregrasp_delta_aux=False
            ):
        
        super().__init__()
        self.image_size = image_size
        self.use_pregrasp_delta_aux = use_pregrasp_delta_aux
        
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Initialize auxiliary loss data structures
        self.target_arm_joints_per_episode = []  # List of dicts indexed by zarr_idx and episode_idx
        self.pregrasp_masks = []  # List of arrays indexed by zarr_idx, containing pregrasp masks for all frames
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
        
    def _process_target_poses_for_zarr(self, replay_buffer, zarr_idx):
        """
        Load target poses from meta and build pregrasp masks.
        """
        target_poses = replay_buffer.meta.get('target_poses', None)
        episode_ends = replay_buffer.episode_ends
        
        # Initialize storage for this zarr
        target_arm_joints_dict = {}
        pregrasp_mask = np.zeros(len(replay_buffer), dtype=np.float32)
        frame_to_episode = np.zeros(len(replay_buffer), dtype=np.int32)
        
        if target_poses is None or len(target_poses) == 0:
            # No target poses available
            self.target_arm_joints_per_episode.append(target_arm_joints_dict)
            self.pregrasp_masks.append(pregrasp_mask)
            self.frame_to_episode.append(frame_to_episode)
            return
        
        # Build frame_to_episode mapping and extract target arm joints
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        
        for episode_idx in range(len(episode_ends)):
            episode_start = episode_starts[episode_idx]
            episode_end = episode_ends[episode_idx]
            
            # Map frames to episode
            frame_to_episode[episode_start:episode_end] = episode_idx
            
            # Extract target arm joints from this episode's target pose
            if episode_idx < len(target_poses):
                target_pose = target_poses[episode_idx]
                try:
                    # Handle both string and dict format
                    if isinstance(target_pose, str):
                        target_pose = json.loads(target_pose)
                    
                    # Extract arm_joints
                    if isinstance(target_pose, dict) and 'arm_joints' in target_pose:
                        arm_joints = target_pose['arm_joints']
                        if isinstance(arm_joints, (list, np.ndarray)):
                            arm_joints = np.array(arm_joints[:6], dtype=np.float32)
                            target_arm_joints_dict[episode_idx] = arm_joints
                except Exception as e:
                    print(f"Warning: Failed to parse target_pose for episode {episode_idx}: {e}")
                    continue
        
        # Compute pregrasp masks based on distance to target arm joints
        if self.use_pregrasp_delta_aux:
            right_state = replay_buffer['right_state']  # [T, 8]
            
            for episode_idx in range(len(episode_ends)):
                if episode_idx not in target_arm_joints_dict:
                    continue
                
                episode_start = episode_starts[episode_idx]
                episode_end = episode_ends[episode_idx]
                
                target_joints = target_arm_joints_dict[episode_idx]  # [6]
                episode_right_state = right_state[episode_start:episode_end]  # [T, 8]
                episode_arm_joints = episode_right_state[:, :6]  # [T, 6]
                
                # Compute distance to target for each frame
                distances = np.linalg.norm(
                    episode_arm_joints - target_joints[np.newaxis, :],  # broadcast to [T, 6]
                    axis=1
                )  # [T]
                
                # Find closest frame to target
                t_grasp = np.argmin(distances)
                
                # Mark frames <= t_grasp as pregrasp
                pregrasp_mask[episode_start:episode_start+t_grasp+1] = 1.0
        
        self.target_arm_joints_per_episode.append(target_arm_joints_dict)
        self.pregrasp_masks.append(pregrasp_mask)
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
        
        # Add auxiliary targets
        if self.use_pregrasp_delta_aux:
            # Get episode index for this frame
            episode_idx = int(self.frame_to_episode[zarr_idx][frame_idx])
            
            # Get target arm joints and pregrasp mask for this frame
            target_arm_joints_dict = self.target_arm_joints_per_episode[zarr_idx]
            pregrasp_mask = self.pregrasp_masks[zarr_idx]
            
            if episode_idx in target_arm_joints_dict:
                target_arm_joints = target_arm_joints_dict[episode_idx].astype(np.float32)
                pregrasp_mask_value = float(pregrasp_mask[frame_idx])
            else:
                # No target pose for this episode
                target_arm_joints = np.zeros(6, dtype=np.float32)
                pregrasp_mask_value = 0.0
            
            data['target_arm_joints'] = target_arm_joints
            data['pregrasp_mask'] = np.array(pregrasp_mask_value, dtype=np.float32)
        
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