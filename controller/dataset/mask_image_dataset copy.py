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
        self.zarr_paths = zarr_paths  # 🌟 只存路徑
        self.image_size = image_size
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        
        # 🌟 徹底解開 Hydra 物件
        if not isinstance(shape_meta, dict) and hasattr(shape_meta, '_dict'):
            shape_meta_dict = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            shape_meta_dict = shape_meta

        self.cam_types = {k: v['type'] for k, v in shape_meta_dict['obs'].items()}
        self.obs_keys = list(shape_meta_dict['obs'].keys())
        self.state_keys = [k for k, v in shape_meta_dict['obs'].items() if v['type'] == 'low_dim']
        self.action_key = 'action'
        
        # 影像 keys 預處理
        img_keys = [k for k, v in shape_meta_dict['obs'].items() if v['type'] in ['rgb', 'rgbm']]
        self.key_first_k = {k: n_obs_steps for k in img_keys}
        
        # 🌟 延遲加載 (Lazy Init) 標記
        self.replay_buffers = None
        self.samplers = None
        self.sampler_lens = []

        # 為了計算長度，我們必須先快速掃描 Zarr 的 meta 資訊
        for path in zarr_paths:
            tmp_rb = StreamingReplayBuffer.copy_from_path(path, keys=[self.action_key])
            val_mask = get_val_mask(tmp_rb.n_episodes, val_ratio, seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(train_mask, max_train_episodes, seed)
            
            # 使用 Dummy Sampler 計算長度，避免在主進程載入資料
            tmp_sampler = SequenceSampler(
                replay_buffer=tmp_rb, sequence_length=horizon,
                pad_before=pad_before, pad_after=pad_after,
                episode_mask=train_mask)
            self.sampler_lens.append(len(tmp_sampler))

    def _lazy_init(self):
        """🌟 只有在 Worker 進程中第一次讀取資料時才會執行一次"""
        if self.replay_buffers is not None:
            return
            
        self.replay_buffers = []
        self.samplers = []
        rb_keys = self.obs_keys + [self.action_key]
        
        for i, path in enumerate(self.zarr_paths):
            rb = StreamingReplayBuffer.copy_from_path(path, keys=rb_keys)
            self.replay_buffers.append(rb)
            
            val_mask = get_val_mask(rb.n_episodes, self.val_ratio, self.seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(train_mask, self.max_train_episodes, self.seed)
            
            sampler = SequenceSampler(
                replay_buffer=rb, sequence_length=self.horizon,
                pad_before=self.pad_before, pad_after=self.pad_after,
                episode_mask=train_mask, key_first_k=self.key_first_k)
            self.samplers.append(sampler)

    def _process_mask_image_batch(self, images):
        with torch.no_grad():
            rgb = torch.from_numpy(images[..., :3]).float().permute(0, 3, 1, 2)
            mask = torch.from_numpy(images[..., 3:]).float().permute(0, 3, 1, 2)
            rgb = F.interpolate(rgb / 255.0, size=self.image_size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=self.image_size, mode='nearest')
            mask = (mask > 0.5).float()
            return torch.cat([rgb, mask], dim=1).numpy().copy()

    def _process_image_batch(self, images):
        with torch.no_grad():
            rgb = torch.from_numpy(images[..., :3]).float().permute(0, 3, 1, 2)
            rgb = F.interpolate(rgb / 255.0, size=self.image_size, mode='bilinear', align_corners=False)
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
                data['obs'][key] = sample[key].astype(np.float32)[T_slice].copy()

        data['action'] = sample[self.action_key].astype(np.float32).copy()
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._lazy_init() # 🌟 確保在 Worker 進程中初始化
        
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

    def get_normalizer(self, mode='limits', **kwargs):
        """🌟 Normalizer 需要單獨載入一次小資料來計算"""
        actions, states = [], {sk: [] for sk in self.state_keys}
        for path in self.zarr_paths:
            rb = StreamingReplayBuffer.copy_from_path(path, keys=[self.action_key] + self.state_keys)
            actions.append(rb[self.action_key])
            for sk in self.state_keys:
                states[sk].append(rb[sk])
            
        data = {'action': np.concatenate(actions, axis=0)}
        for sk in self.state_keys:
            data[sk] = np.concatenate(states[sk], axis=0)
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer