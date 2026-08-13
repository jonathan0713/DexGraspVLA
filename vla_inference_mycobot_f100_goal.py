import argparse
import dill
import json
import os
import sys

import cv2
import hydra
import numpy as np
import rclpy
import torch
from omegaconf import open_dict

import vla_inference_mycobot_f100 as base


GOAL_DIM = int(os.environ.get("VLA_F100_GOAL_DIM", "7"))


def load_goal_policy(run_dir, ckpt_name):
    cfg_path = os.path.join(run_dir, ".hydra")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.abspath(cfg_path), version_base=None)
    cfg = hydra.compose(config_name="config")
    base.validate_policy_shape(cfg)

    with open_dict(cfg.policy):
        cfg.policy.use_goal_token = True
        cfg.policy.goal_dim = GOAL_DIM
        if "goal_token_dim" not in cfg.policy:
            cfg.policy.goal_token_dim = None
        if "goal_mlp_hidden_dim" not in cfg.policy:
            cfg.policy.goal_mlp_hidden_dim = None
    cfg.task.dataset = None

    from controller.workspace.train_dexgraspvla_controller_workspace import (
        TrainDexGraspVLAControllerWorkspace,
    )

    workspace = TrainDexGraspVLAControllerWorkspace(cfg)

    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    try:
        workspace.load_payload(
            payload,
            exclude_keys=("optimizer", "lr_scheduler"),
            include_keys=None,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load this checkpoint as a goal-conditioned model. "
            "The checkpoint and runtime architecture still disagree. Check that "
            "the checkpoint was trained with use_goal_token=True, goal_dim=7, "
            "and the same goal_mlp_hidden_dim."
        ) from exc

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()
    return policy


def parse_goal_values(text):
    values = [chunk for chunk in text.replace(",", " ").split() if chunk]
    try:
        goal = np.asarray([float(value) for value in values], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Goal contains a non-float value: {text!r}") from exc

    if goal.shape != (GOAL_DIM,):
        raise ValueError(f"Expected {GOAL_DIM} goal values, got {goal.shape[0]}")
    return goal


def parse_goal_json(text_or_path):
    if os.path.exists(text_or_path):
        with open(text_or_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(text_or_path)

    if not isinstance(payload, dict):
        raise ValueError("Goal JSON must be an object")

    arm_joints = payload.get("arm_joints", [])
    gripper_joints = payload.get("gripper_joints", [])
    if len(arm_joints) != len(base.ARM_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(base.ARM_JOINT_NAMES)} arm_joints, got {len(arm_joints)}"
        )
    if len(gripper_joints) != len(base.GRIPPER_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(base.GRIPPER_JOINT_NAMES)} gripper_joints, "
            f"got {len(gripper_joints)}"
        )
    return parse_goal_values(" ".join(map(str, list(arm_joints) + list(gripper_joints))))


def prompt_goal_cond():
    print("\nManual F100 joint-space grasp target is required.")
    print("Enter 7 values in this order:")
    for index, name in enumerate(base.ARM_JOINT_NAMES + base.GRIPPER_JOINT_NAMES):
        print(f"  [{index}] {name}")
    print("Example: 1.57 -0.418 -0.523 0.523 1.57 0.0 0.7")

    while True:
        text = input("goal_cond> ").strip()
        if not text:
            print("Please enter 7 numeric values.")
            continue
        try:
            return parse_goal_values(text)
        except ValueError as exc:
            print(f"Invalid goal: {exc}")


def resolve_goal_cond(args):
    if args.goal_cond:
        return parse_goal_values(args.goal_cond)
    if args.goal_json:
        return parse_goal_json(args.goal_json)

    env_goal = os.environ.get("VLA_F100_GOAL_COND")
    if env_goal:
        return parse_goal_values(env_goal)

    if not sys.stdin.isatty():
        raise ValueError(
            "No manual goal was provided. Use --goal-cond, --goal-json, or "
            "VLA_F100_GOAL_COND."
        )
    return prompt_goal_cond()


class GoalConditionedVLAEvaluator(base.VLA_IPCEvaluator):
    def __init__(self, policy, goal_cond):
        self.goal_cond_np = np.asarray(goal_cond, dtype=np.float32)
        if self.goal_cond_np.shape != (GOAL_DIM,):
            raise ValueError(
                f"Expected goal_cond shape {(GOAL_DIM,)}, got {self.goal_cond_np.shape}"
            )
        super().__init__(policy)
        self.get_logger().info(
            "Manual joint-space goal_cond loaded: "
            f"{np.array2string(self.goal_cond_np, precision=4, separator=', ')}"
        )

    def goal_cond_tensor(self):
        return torch.from_numpy(self.goal_cond_np).unsqueeze(0).to(self.device)

    def inference_worker(self, data_snapshot, generation):
        try:
            front_rgb = data_snapshot["front_rgb"].copy()
            wrist_rgb = data_snapshot["wrist_rgb"].copy()
            mask_uint8 = data_snapshot["mask"]

            mask_255 = (mask_uint8 > 0).astype(np.uint8) * 255

            if front_rgb.shape[:2] != base.IMG_SIZE:
                front_rgb = cv2.resize(front_rgb, base.IMG_SIZE)
                mask_255 = cv2.resize(mask_255, base.IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            rgbm = np.concatenate([front_rgb, mask_255[:, :, np.newaxis]], axis=-1)
            rgbm_tensor = (
                torch.from_numpy(rgbm)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )

            wrist_rgb = cv2.resize(wrist_rgb, base.IMG_SIZE)
            wrist_tensor = (
                torch.from_numpy(wrist_rgb)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )

            t_state = self.get_state_vector(data_snapshot["state"])

            obs = {
                "rgbm": rgbm_tensor,
                "right_cam_img": wrist_tensor,
                "right_state": t_state.unsqueeze(0).unsqueeze(0).to(self.device),
            }
            goal_cond = self.goal_cond_tensor()
            assert goal_cond.shape == (1, GOAL_DIM)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.policy.predict_action(obs, goal_cond=goal_cond)
                action_seq = output["action"] if isinstance(output, dict) else output
                full_traj = action_seq[0].cpu().numpy()

            with self.lock:
                if generation != self.reset_generation:
                    return

                start_idx = base.SKIP_FIRST_K
                skip_steps = self.steps_executed_during_inference
                chunk = full_traj[
                    start_idx + skip_steps : start_idx + skip_steps + base.CHUNK_SIZE
                ]
                candidate_actions = chunk if len(chunk) > 0 else full_traj[-1:]
                reference_action = (
                    self.last_executed_action
                    if self.last_executed_action is not None
                    else t_state.cpu().numpy()
                )
                self.action_buffer = self.stabilize_action_sequence(
                    candidate_actions,
                    reference_action,
                )
        except Exception as exc:
            self.get_logger().error(f"Goal-conditioned inference failed: {exc}")
        finally:
            with self.lock:
                self.active_inference_workers = max(0, self.active_inference_workers - 1)
                if generation == self.reset_generation:
                    self.is_inferencing = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="myCobot F100 VLA inference with a manually supplied goal_cond."
    )
    parser.add_argument(
        "--goal-cond",
        default=None,
        help="Seven joint-space goal values: j1 j2 j3 j4 j5 j6 gripper.",
    )
    parser.add_argument(
        "--goal-json",
        default=None,
        help=(
            "JSON string or file path containing arm_joints and gripper_joints. "
            "Only joint-space goal conditioning is supported here."
        ),
    )
    return parser.parse_args()


def main(args=None):
    parsed_args = parse_args() if args is None else args
    goal_cond = resolve_goal_cond(parsed_args)
    print(
        "Using manual goal_cond: "
        f"{np.array2string(goal_cond, precision=4, separator=', ')}"
    )

    rclpy.init(args=None)
    node = None
    try:
        policy = load_goal_policy(base.RUN_DIR, base.CKPT_NAME)
        node = GoalConditionedVLAEvaluator(policy, goal_cond)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
