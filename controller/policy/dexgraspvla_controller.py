from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import inspect
from controller.model.common.normalizer import LinearNormalizer
from controller.policy.base_image_policy import BaseImagePolicy
from controller.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from controller.model.vision.obs_encoder import ObsEncoder
from scipy.optimize import linear_sum_assignment
import pickle


# Adapted from https://github.com/lucidrains/pi-zero-pytorch/blob/e82fced40e55023a0ded22ab3bda495964353253/pi_zero_pytorch/pi_zero.py#L216
def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

class DexGraspVLAController(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: ObsEncoder,
            num_inference_steps=None,
            # arch
            n_layer=7,
            n_head=8,
            p_drop_attn=0.1,
            use_attn_mask=False,
            start_ckpt_path=None,
            use_goal_token=False,
            goal_dim=7,
            goal_token_dim=None,
            goal_mlp_hidden_dim=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape, obs_part_length = obs_encoder.output_shape()
        n_emb = obs_shape[-1]
        obs_tokens = obs_shape[-2]
        self.use_goal_token = bool(use_goal_token)
        self.goal_dim = int(goal_dim)
        self.goal_token_dim = n_emb if goal_token_dim is None else int(goal_token_dim)
        self.goal_mlp_hidden_dim = (
            n_emb if goal_mlp_hidden_dim is None else int(goal_mlp_hidden_dim)
        )

        if self.use_goal_token:
            assert self.goal_token_dim == n_emb, (
                f"goal_token_dim must match policy token dim {n_emb}, "
                f"got {self.goal_token_dim}"
            )
            self.goal_encoder = nn.Sequential(
                nn.Linear(self.goal_dim, self.goal_mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.goal_mlp_hidden_dim, self.goal_token_dim),
            )
            self.null_goal_token = nn.Parameter(torch.zeros(1, 1, self.goal_token_dim))
        else:
            self.goal_encoder = None
            self.null_goal_token = None

        goal_token_count = 1 if self.use_goal_token else 0
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+goal_token_count+1, # obs/goal tokens + 1 token for time
            p_drop_attn=p_drop_attn,
            obs_part_length=obs_part_length,
            use_attn_mask=use_attn_mask
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.start_ckpt_path = start_ckpt_path
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def validate_goal_cond_shape(
            self,
            goal_cond: Optional[torch.Tensor],
            batch_size: int,
        ) -> None:
        if goal_cond is None:
            return
        assert goal_cond.shape == (batch_size, self.goal_dim), (
            f"Expected batched goal_cond shape {(batch_size, self.goal_dim)}, "
            f"got {goal_cond.shape}"
        )

    def append_goal_token(
            self,
            tokens: torch.Tensor,
            goal_cond: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if not self.use_goal_token:
            return tokens

        B = tokens.shape[0]
        if goal_cond is None:
            goal_token = self.null_goal_token.expand(B, -1, -1)
        else:
            if goal_cond.ndim != 2:
                raise ValueError(f"goal_cond must have shape [B, D], got {goal_cond.shape}")
            if goal_cond.shape != (B, self.goal_dim):
                raise ValueError(
                    f"goal_cond must have shape {(B, self.goal_dim)}, got {goal_cond.shape}"
                )
            goal_cond = goal_cond.to(device=tokens.device, dtype=tokens.dtype)
            goal_token = self.goal_encoder(goal_cond).unsqueeze(1)

        goal_token = goal_token.to(device=tokens.device, dtype=tokens.dtype)
        assert goal_token.shape == (B, 1, tokens.shape[-1])
        return torch.cat([tokens, goal_token], dim=1)
    
    # ========= inference  ============
    def conditional_sample(self, cond=None, gen_attn_map=True, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler
        B = cond.shape[0]

        trajectory = torch.randn(
            size=(B, self.action_horizon, self.action_dim), 
            dtype=self.dtype,
            device=self.device)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Store attention maps for all timesteps
        all_timestep_attention_maps = {}

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output, attention_maps = model(trajectory, t, cond, training=False, gen_attn_map=gen_attn_map)
            all_timestep_attention_maps[t.cpu().item()] = attention_maps

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                **kwargs
                ).prev_sample

        return trajectory, all_timestep_attention_maps

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            output_path: str = None,
            goal_cond: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        action_pred: predicted action
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]
        self.validate_goal_cond_shape(goal_cond, B)
        
        # process input
        obs_tokens = self.obs_encoder(nobs, training=False)
        obs_tokens = self.append_goal_token(obs_tokens, goal_cond=goal_cond)
        # (B, N, n_emb)
        
        # run sampling
        nsample, all_timestep_attention_maps = self.conditional_sample(
            cond=obs_tokens,
            gen_attn_map=True if output_path is not None else False,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        if output_path is not None:
            # Convert tensors in obs_dict to numpy arrays
            obs_dict_numpy = {}
            for k, v in obs_dict.items():
                if k in ['rgbm', 'right_cam_img']:
                    obs_dict_numpy[k] = np.clip(v.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                else:
                    obs_dict_numpy[k] = v.detach().cpu().numpy()
                obs_dict_numpy[k] = obs_dict_numpy[k][:2]

            save_dict = {
                'attention_maps': all_timestep_attention_maps,
                'obs_dict': obs_dict_numpy
            }

            with open(output_path, 'wb') as f:
                pickle.dump(save_dict, f)

        return action_pred

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            betas: Tuple[float, float],
        ) -> torch.optim.Optimizer:

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
        return optimizer

    def compute_loss(self, batch, training=True, goal_cond: Optional[torch.Tensor] = None):
        # normalize input
        assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        nobs = batch['obs']
        if goal_cond is None:
            goal_cond = batch.get('goal_cond', None)
        B = next(iter(nobs.values())).shape[0]
        self.validate_goal_cond_shape(goal_cond, B)
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions

        # process input
        obs_tokens = self.obs_encoder(nobs, training)
        obs_tokens = self.append_goal_token(obs_tokens, goal_cond=goal_cond)
        # (B, N, n_emb)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        assignment = noise_assignment(trajectory, noise)
        noise = noise[assignment]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # Predict the noise residual
        pred, _ = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens,
            training=training,
            gen_attn_map=False
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)

        return loss

    def forward(self, batch, training=True, goal_cond: Optional[torch.Tensor] = None):
        return self.compute_loss(batch, training, goal_cond=goal_cond)
