from typing import Dict
import torch
import numpy as np
import copy
from controller.common.pytorch_util import dict_apply
from controller.common.streaming_replay_buffer import StreamingReplayBuffer
from controller.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from controller.model.common.normalizer import LinearNormalizer
from controller.dataset.base_dataset import BaseImageDataset
import torch.nn.functional as F
from omegaconf import OmegaConf

class MaskImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            shape_meta,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(518, 518)
            ):
        super().__init__()
        self.image_size = image_size
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        
        # 動態解開 Hydra 物件
        if not isinstance(shape_meta, dict) and hasattr(shape_meta, '_dict'):
            shape_meta_dict = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            shape_meta_dict = shape_meta

        # 自動化檢測名稱與類型
        self.cam_types = {k: v['type'] for k, v in shape_meta_dict['obs'].items()}
        self.obs_keys = list(shape_meta_dict['obs'].keys())
        self.state_keys = [k for k, v in shape_meta_dict['obs'].items() if v['type'] == 'low_dim']
        self.action_key = 'action'
        rb_keys = self.obs_keys + [self.action_key]

        img_keys = [k for k, v in shape_meta_dict['obs'].items() if v['type'] in ['rgb', 'rgbm']]
        self.key_first_k = {k: n_obs_steps for k in img_keys}

        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        for zarr_path in zarr_paths:
            replay_buffer = StreamingReplayBuffer.copy_from_path(zarr_path, keys=rb_keys)
            self.replay_buffers.append(replay_buffer)
            
            val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(train_mask, max_train_episodes, seed)
            self.train_masks.append(train_mask)
            
            sampler = SequenceSampler(
                replay_buffer=replay_buffer, sequence_length=horizon,
                pad_before=pad_before, pad_after=pad_after,
                episode_mask=train_mask, key_first_k=self.key_first_k)
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            sampler = SequenceSampler(
                replay_buffer=replay_buffer, sequence_length=self.horizon,
                pad_before=self.pad_before, pad_after=self.pad_after,
                episode_mask=~self.train_masks[i], key_first_k=self.key_first_k)
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
        return val_set

    def _process_mask_image_batch(self, images):
        with torch.no_grad():
            # 🌟 修正維度炸彈：因為 45GB 的 Zarr 已經是 CHW (T, 4, 518, 518)
            # 直接在 dim=1 切割通道，絕對不會再切出 515 個怪東西
            rgb = torch.from_numpy(images[:, :3, :, :]).float()
            mask = torch.from_numpy(images[:, 3:, :, :]).float()
            
            # 移除原本的 permute，直接縮放
            rgb = F.interpolate(rgb / 255.0, size=self.image_size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=self.image_size, mode='nearest')
            mask = (mask > 0.5).float()
            
            # 🌟 加上 .copy() 切斷記憶體殘留
            return torch.cat([rgb, mask], dim=1).numpy().copy()

    def _process_image_batch(self, images):
        with torch.no_grad():
            # 🌟 修正維度炸彈：已經是 CHW (T, 3, 518, 518)
            rgb = torch.from_numpy(images).float()
            rgb = F.interpolate(rgb / 255.0, size=self.image_size, mode='bilinear', align_corners=False)
            # 🌟 加上 .copy() 切斷記憶體殘留
            return rgb.numpy().copy()
    
    def _sample_to_data(self, sample):
        T_slice = slice(self.n_obs_steps)
        data = {'obs': {}}
        
        for key in self.obs_keys:
            cam_type = self.cam_types[key] 
            if cam_type == 'rgbm':
                data['obs'][key] = self._process_mask_image_batch(sample[key][T_slice])
            elif cam_type == 'rgb':
                data['obs'][key] = self._process_image_batch(sample[key][T_slice])
            elif cam_type == 'low_dim':
                # 🌟 加上 .copy()
                data['obs'][key] = sample[key].astype(np.float32)[T_slice].copy()

        # 🌟 加上 .copy()
        data['action'] = sample[self.action_key].astype(np.float32).copy()
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': np.concatenate([rb[self.action_key] for rb in self.replay_buffers], axis=0)}
        for sk in self.state_keys:
            data[sk] = np.concatenate([rb[sk] for rb in self.replay_buffers], axis=0)
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length
            
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    def __len__(self):
        return sum(self.sampler_lens)