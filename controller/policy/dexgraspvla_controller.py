from typing import Dict, Tuple
import torch
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
            # auxiliary loss
            use_grasp_xy_aux=False,
            grasp_xy_aux_dim=2,
            grasp_xy_aux_hidden_dim=256,
            normalize_grasp_xy=True,
            use_pregrasp_delta_aux=False,
            use_pregrasp_joint_delta_aux=False,
            pregrasp_joint_delta_aux_dim=6,
            pregrasp_joint_delta_aux_hidden_dim=256,
            pregrasp_delta_aux_hidden_dim=256,
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
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+1, # obs tokens + 1 token for time
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
        
        # Training-only grasp XY auxiliary loss setup.
        self.use_grasp_xy_aux = use_grasp_xy_aux
        self.grasp_xy_aux_dim = grasp_xy_aux_dim
        self.normalize_grasp_xy = normalize_grasp_xy
        self.use_pregrasp_delta_aux = False
        self.use_pregrasp_joint_delta_aux = use_pregrasp_joint_delta_aux
        self.pregrasp_joint_delta_aux_dim = pregrasp_joint_delta_aux_dim
        if self.use_pregrasp_joint_delta_aux and self.pregrasp_joint_delta_aux_dim != 6:
            raise ValueError("pregrasp_joint_delta_aux_dim must be 6 for arm joint deltas.")
        if self.use_grasp_xy_aux and self.use_pregrasp_joint_delta_aux:
            raise ValueError(
                "Only one auxiliary path can be active: disable use_grasp_xy_aux "
                "or use_pregrasp_joint_delta_aux."
            )
        self._xy_aux_loss_batch_count = 0  # For debug logging
        self._pregrasp_joint_delta_aux_loss_batch_count = 0  # For debug logging
        self._current_epoch = 0
        if use_pregrasp_delta_aux:
            print(
                "Warning: use_pregrasp_delta_aux is deprecated and ignored. "
                "Use use_pregrasp_joint_delta_aux instead."
            )
        if use_grasp_xy_aux:
            # Auxiliary head: pool observation features -> predict grasp workspace XY.
            self.grasp_xy_aux_head = torch.nn.Sequential(
                torch.nn.Linear(n_emb, grasp_xy_aux_hidden_dim),
                torch.nn.LayerNorm(grasp_xy_aux_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(grasp_xy_aux_hidden_dim, grasp_xy_aux_dim)
            )
        if use_pregrasp_joint_delta_aux:
            # Training-only auxiliary head:
            # pool observation features -> predict current-arm to grasp-contact joint delta.
            self.pregrasp_joint_delta_aux_head = torch.nn.Sequential(
                torch.nn.Linear(n_emb, pregrasp_joint_delta_aux_hidden_dim),
                torch.nn.LayerNorm(pregrasp_joint_delta_aux_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(
                    pregrasp_joint_delta_aux_hidden_dim,
                    pregrasp_joint_delta_aux_dim,
                )
            )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
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

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        action_pred: predicted action
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs, training=False)
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

    def compute_loss(self, batch, training=True):
        # normalize input
        assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        nobs = batch['obs']
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions

        # process input
        obs_tokens = self.obs_encoder(nobs, training)
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
        
        # Compute training-only grasp XY auxiliary loss if enabled.
        if self.use_grasp_xy_aux and training:
            if 'grasp_xy' in batch:
                B = obs_tokens.shape[0]
                device = obs_tokens.device

                obs_summary = torch.mean(obs_tokens, dim=1)  # [B, n_emb]
                pred_grasp_xy_norm = self.grasp_xy_aux_head(obs_summary)  # [B, 2]

                target_grasp_xy_raw = batch['grasp_xy'].to(device)  # [B, 2]
                target_grasp_xy_norm = target_grasp_xy_raw
                if self.normalize_grasp_xy:
                    target_grasp_xy_norm = self.normalizer['grasp_xy'].normalize(target_grasp_xy_raw)

                mse_per_sample = torch.mean(
                    (pred_grasp_xy_norm - target_grasp_xy_norm) ** 2,
                    dim=-1
                )  # [B]

                if 'grasp_xy_valid' in batch:
                    valid_mask = batch['grasp_xy_valid'].to(device)
                    if valid_mask.dim() > 1:
                        valid_mask = valid_mask.squeeze(-1)
                else:
                    valid_mask = torch.ones(B, dtype=target_grasp_xy_norm.dtype, device=device)

                valid_sum = torch.sum(valid_mask)
                eps = 1e-6
                xy_aux_loss = torch.sum(valid_mask * mse_per_sample) / (valid_sum + eps)

                # First-epoch debug logging for the first few batches.
                if getattr(self, '_current_epoch', 0) == 0 and self._xy_aux_loss_batch_count < 3:
                    action_loss_val = loss.detach().item()
                    xy_aux_loss_val = xy_aux_loss.detach().item()
                    lambda_aux = getattr(self, '_lambda_grasp_xy_aux', 0.02)
                    weighted_xy_aux_loss_val = lambda_aux * xy_aux_loss_val
                    ratio_val = weighted_xy_aux_loss_val / max(action_loss_val, 1e-12)

                    if valid_sum.item() > 0:
                        valid_bool = valid_mask > 0.5
                        target_stats_tensor = target_grasp_xy_norm[valid_bool]
                        pred_stats_tensor = pred_grasp_xy_norm[valid_bool]
                    else:
                        target_stats_tensor = target_grasp_xy_norm
                        pred_stats_tensor = pred_grasp_xy_norm

                    def stats_str(name, tensor):
                        tensor = tensor.detach().float()
                        return (
                            f"{name}=[mean={tensor.mean(dim=0).cpu().numpy()}, "
                            f"std={tensor.std(dim=0, unbiased=False).cpu().numpy()}, "
                            f"min={tensor.min(dim=0).values.cpu().numpy()}, "
                            f"max={tensor.max(dim=0).values.cpu().numpy()}]"
                        )

                    def first_values(tensor, count=5):
                        return tensor[:count].detach().float().cpu().numpy().tolist()

                    grasp_xy_mean = None
                    grasp_xy_std = None
                    if self.normalize_grasp_xy and 'grasp_xy' in self.normalizer.params_dict:
                        grasp_xy_stats = self.normalizer['grasp_xy'].get_input_stats()
                        grasp_xy_mean = grasp_xy_stats['mean'].detach().float().cpu().numpy()
                        grasp_xy_std = grasp_xy_stats['std'].detach().float().cpu().numpy()

                    print(
                        f"[GRASP_XY_AUX DEBUG {self._xy_aux_loss_batch_count}] "
                        f"epoch={getattr(self, '_current_epoch', 0)}, "
                        f"grasp_xy_mean={grasp_xy_mean}, "
                        f"grasp_xy_std={grasp_xy_std}; "
                        f"first5_grasp_xy_raw={first_values(target_grasp_xy_raw)}, "
                        f"first5_grasp_xy_norm={first_values(target_grasp_xy_norm)}; "
                        f"action_loss={action_loss_val:.6f}, "
                        f"xy_aux_loss={xy_aux_loss_val:.6f}, "
                        f"lambda*xy_aux_loss={weighted_xy_aux_loss_val:.6f}, "
                        f"ratio={ratio_val:.6f}, "
                        f"total_loss={action_loss_val + weighted_xy_aux_loss_val:.6f}; "
                        f"grasp_xy_valid.sum()/batch_size={valid_sum.item():.1f}/{B}, "
                        f"normalized={self.normalize_grasp_xy}; "
                        f"{stats_str('target_grasp_xy_norm', target_stats_tensor)}; "
                        f"{stats_str('pred_grasp_xy_norm', pred_stats_tensor)}"
                    )
                    self._xy_aux_loss_batch_count += 1

                lambda_grasp_xy_aux = getattr(self, '_lambda_grasp_xy_aux', 0.02)
                loss = loss + lambda_grasp_xy_aux * xy_aux_loss

        # Compute training-only phase-gated pre-grasp joint delta auxiliary loss.
        if self.use_pregrasp_joint_delta_aux and training:
            if 'target_arm_joints' in batch and 'pregrasp_mask' in batch:
                B = obs_tokens.shape[0]
                device = obs_tokens.device

                obs_summary = torch.mean(obs_tokens, dim=1)  # [B, n_emb]
                pred_delta_arm = self.pregrasp_joint_delta_aux_head(obs_summary)  # [B, 6]

                target_arm_joints = batch['target_arm_joints'].to(device).float()  # [B, 6]
                current_arm_joints = batch['obs']['right_state'][:, 0, :6].to(device).float()
                delta_arm_to_grasp = target_arm_joints - current_arm_joints

                pregrasp_mask = batch['pregrasp_mask'].to(device).float()
                if pregrasp_mask.dim() > 1:
                    pregrasp_mask = pregrasp_mask.squeeze(-1)

                mse_per_sample = torch.mean(
                    (pred_delta_arm - delta_arm_to_grasp) ** 2,
                    dim=-1,
                )
                valid_sum = torch.sum(pregrasp_mask)
                eps = 1e-6
                joint_delta_aux_loss = (
                    torch.sum(pregrasp_mask * mse_per_sample) / (valid_sum + eps)
                )

                lambda_aux = getattr(self, '_lambda_pregrasp_joint_delta_aux', 0.02)

                if getattr(self, '_current_epoch', 0) == 0 and self._pregrasp_joint_delta_aux_loss_batch_count < 3:
                    action_loss_val = loss.detach().item()
                    joint_delta_aux_loss_val = joint_delta_aux_loss.detach().item()
                    weighted_aux_loss_val = lambda_aux * joint_delta_aux_loss_val
                    ratio_val = weighted_aux_loss_val / max(action_loss_val, 1e-12)

                    if valid_sum.item() > 0:
                        valid_bool = pregrasp_mask > 0.5
                        target_delta_stats_tensor = delta_arm_to_grasp[valid_bool]
                        pred_delta_stats_tensor = pred_delta_arm[valid_bool]
                    else:
                        target_delta_stats_tensor = delta_arm_to_grasp
                        pred_delta_stats_tensor = pred_delta_arm

                    def stats_str(name, tensor):
                        tensor = tensor.detach().float()
                        return (
                            f"{name}=[mean={tensor.mean(dim=0).cpu().numpy()}, "
                            f"std={tensor.std(dim=0, unbiased=False).cpu().numpy()}, "
                            f"min={tensor.min(dim=0).values.cpu().numpy()}, "
                            f"max={tensor.max(dim=0).values.cpu().numpy()}]"
                        )

                    def first_values(tensor, count=5):
                        return tensor[:count].detach().float().cpu().numpy().tolist()

                    print(
                        f"[PREGRASP_JOINT_DELTA_AUX DEBUG "
                        f"{self._pregrasp_joint_delta_aux_loss_batch_count}] "
                        f"epoch={getattr(self, '_current_epoch', 0)}, "
                        f"action_loss={action_loss_val:.6f}, "
                        f"joint_delta_aux_loss={joint_delta_aux_loss_val:.6f}, "
                        f"lambda*joint_delta_aux_loss={weighted_aux_loss_val:.6f}, "
                        f"ratio={ratio_val:.6f}, "
                        f"total_loss={action_loss_val + weighted_aux_loss_val:.6f}; "
                        f"pregrasp_mask.sum()/batch_size={valid_sum.item():.1f}/{B}; "
                        f"{stats_str('target_delta_arm', target_delta_stats_tensor)}; "
                        f"{stats_str('pred_delta_arm', pred_delta_stats_tensor)}; "
                        f"first5_target_arm_joints={first_values(target_arm_joints)}, "
                        f"first5_current_arm_joints={first_values(current_arm_joints)}, "
                        f"first5_delta_arm_to_grasp={first_values(delta_arm_to_grasp)}"
                    )
                    self._pregrasp_joint_delta_aux_loss_batch_count += 1

                loss = loss + lambda_aux * joint_delta_aux_loss
        
        return loss

    def forward(self, batch, training=True):
        return self.compute_loss(batch, training)
