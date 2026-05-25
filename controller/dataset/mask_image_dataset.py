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
            goal_dim=7,
            goal_mode='joint',
            ):
        
        super().__init__()
        self.image_size = image_size
        
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        self.goal_conds = []
        self.frame_to_episodes = []
        self.goal_dim = int(goal_dim)
        self.goal_mode = goal_mode
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['right_cam_img', 'rgbm', 'right_state', 'action'])
            self.replay_buffers.append(replay_buffer)
            self.goal_conds.append(self._parse_episode_goal_conds(replay_buffer))
            self.frame_to_episodes.append(self._build_frame_to_episode(replay_buffer.episode_ends))
            
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

    def _build_frame_to_episode(self, episode_ends):
        episode_ends = np.asarray(episode_ends, dtype=np.int64)
        total_frames = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
        frame_to_episode = np.zeros(total_frames, dtype=np.int64)

        start = 0
        for episode_idx, end in enumerate(episode_ends):
            frame_to_episode[start:int(end)] = episode_idx
            start = int(end)
        return frame_to_episode

    def _parse_episode_goal_conds(self, replay_buffer):
        episode_ends = np.asarray(replay_buffer.episode_ends, dtype=np.int64)
        n_episodes = len(episode_ends)
        goals = np.zeros((n_episodes, self.goal_dim), dtype=np.float32)

        target_poses = replay_buffer.meta.get('target_poses', None)
        if target_poses is None:
            return goals

        if len(target_poses) != n_episodes:
            raise ValueError(
                f"meta/target_poses length {len(target_poses)} does not match "
                f"episode_ends length {n_episodes} in {replay_buffer.zarr_path}"
            )

        for episode_idx, raw_target_pose in enumerate(target_poses):
            goals[episode_idx] = self._target_pose_to_goal_cond(raw_target_pose)
        return goals

    def _target_pose_to_goal_cond(self, raw_target_pose):
        if isinstance(raw_target_pose, bytes):
            raw_target_pose = raw_target_pose.decode('utf-8')

        if isinstance(raw_target_pose, str):
            if raw_target_pose.strip() == "":
                return np.zeros((self.goal_dim,), dtype=np.float32)
            target_pose = json.loads(raw_target_pose)
        elif isinstance(raw_target_pose, np.ndarray) and raw_target_pose.shape == ():
            return self._target_pose_to_goal_cond(raw_target_pose.item())
        else:
            target_pose = raw_target_pose

        if not isinstance(target_pose, dict) or len(target_pose) == 0:
            return np.zeros((self.goal_dim,), dtype=np.float32)

        if self.goal_mode != 'joint':
            raise NotImplementedError(f"Unsupported goal_mode: {self.goal_mode}")

        arm_joints = target_pose.get('arm_joints', [])
        gripper_joints = target_pose.get('gripper_joints', [])
        assert len(arm_joints) == 6, f"Expected 6 arm_joints, got {len(arm_joints)}"
        assert len(gripper_joints) == 1, (
            f"Expected 1 gripper_joints value, got {len(gripper_joints)}"
        )

        goal_cond = np.asarray(list(arm_joints) + list(gripper_joints), dtype=np.float32)
        assert goal_cond.shape == (self.goal_dim,), (
            f"Expected goal_cond shape {(self.goal_dim,)}, got {goal_cond.shape}"
        )
        return goal_cond

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
    
    def _sample_to_data(self, sample, goal_cond=None):
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
        if goal_cond is not None:
            goal_cond = np.asarray(goal_cond, dtype=np.float32)
            assert goal_cond.shape == (self.goal_dim,), (
                f"Expected goal_cond shape {(self.goal_dim,)}, got {goal_cond.shape}"
            )
            data['goal_cond'] = goal_cond
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
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                buffer_start_idx = int(self.samplers[i].indices[curr_idx][0])
                episode_idx = int(self.frame_to_episodes[i][buffer_start_idx])
                goal_cond = self.goal_conds[i][episode_idx]
                break
            curr_idx -= length
            
        data = self._sample_to_data(sample, goal_cond=goal_cond)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return sum(self.sampler_lens)
