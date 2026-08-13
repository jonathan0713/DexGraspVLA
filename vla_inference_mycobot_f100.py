import collections
import os
import sys
import threading
import time
from multiprocessing.connection import Listener

import cv2
import dill
import hydra
import numpy as np
import rclpy
import torch
import zmq
from rclpy.node import Node
from sensor_msgs.msg import JointState


sys.path.append(os.getcwd())

PROJECT_PATH = "/home/jonathan/Documents/ITRI_Project"
ROS_WS_PATH = f"{PROJECT_PATH}/ros2_ws"
DATA_COLLECTOR_PATH = f"{ROS_WS_PATH}/src/data_collector"

for path in (PROJECT_PATH, ROS_WS_PATH, DATA_COLLECTOR_PATH):
    if path not in sys.path:
        sys.path.append(path)

from mycobot_f100.vla_config_mycobot_f100 import config


# Defaults target the F100 run whose hydra config has right_state/action shape 8.
RUN_DIR = os.environ.get(
    "VLA_F100_RUN_DIR",
    # "data/outputs/2026.05.24/22.39_train_dexgraspvla_controller_grasp",
    "data/outputs/from_itri/special",
)
CKPT_NAME = os.environ.get("VLA_F100_CKPT_NAME", "latest.ckpt")

IMG_SIZE = (518, 518)
SMOOTHING_ALPHA = float(os.environ.get("VLA_F100_SMOOTHING_ALPHA", "1.0"))
CONTROL_FREQ = float(os.environ.get("VLA_F100_CONTROL_FREQ", "48"))
RESET_SIGNAL_PATH = os.environ.get(
    "VLA_F100_RESET_FILE",
    os.environ.get("VLA_INFERENCE_RESET_FILE", "/tmp/vla_inference_f100_reset.flag"),
)
CONTROL_SOCKET_PATH = os.environ.get(
    "VLA_F100_CONTROL_SOCKET",
    os.environ.get("VLA_INFERENCE_CONTROL_SOCKET", "/tmp/vla_inference_f100_control.sock"),
)
CAMERA_SOCKET_PATH = os.environ.get("VLA_F100_CAMERA_SOCKET", "/tmp/vla_cam_stream")
TRACKER_ENDPOINT = os.environ.get("VLA_F100_TRACKER_ENDPOINT", "tcp://127.0.0.1:5555")

SKIP_FIRST_K = int(os.environ.get("VLA_F100_SKIP_FIRST_K", "8"))
CHUNK_SIZE = int(os.environ.get("VLA_F100_CHUNK_SIZE", "64"))
TRIGGER_THRESHOLD = int(os.environ.get("VLA_F100_TRIGGER_THRESHOLD", "12"))

INTERPOLATE_ACTION_DISTANCE = float(os.environ.get("VLA_F100_INTERPOLATE_DISTANCE", "0.015"))
OUTLIER_ACTION_DISTANCE = float(os.environ.get("VLA_F100_OUTLIER_DISTANCE", "0.06"))
MAX_INTERPOLATION_STEPS = int(os.environ.get("VLA_F100_MAX_INTERPOLATION_STEPS", "12"))

GRIPPER_JOINT_VALUE_SCALES = {
    "gripper_controller": float(os.environ.get("VLA_F100_GRIPPER_SCALE", "1.0")),
}
DEBUG_GRIPPER = os.environ.get("VLA_F100_DEBUG_GRIPPER", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
DEBUG_GRIPPER_PRINT_INTERVAL = float(os.environ.get("VLA_F100_GRIPPER_PRINT_INTERVAL", "0.25"))

TOPIC_JOINT_STATE = config.topic_isaac_joint_states
TOPIC_JOINT_CMD = config.topic_joint_command

ARM_JOINT_NAMES = list(config.arm_joint_names)
GRIPPER_JOINT_NAMES = list(config.gripper_joint_names)
TARGET_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
STATE_DIM = len(TARGET_JOINT_NAMES) + 1
ACTION_DIM = STATE_DIM

GRIPPER_MIN = float(min(config.gripper_open_joints + config.gripper_close_joints))
GRIPPER_MAX = float(max(config.gripper_open_joints + config.gripper_close_joints))


class VLA_IPCEvaluator(Node):
    def __init__(self, policy):
        super().__init__("dexgrasp_vla_f100_ipc_evaluator")
        self.policy = policy
        self.device = policy.device

        self.latest_image_payload = None
        self.state_buffer = collections.deque(maxlen=60)
        self.latest_synced_snapshot = None

        self.img_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.snapshot_lock = threading.Lock()
        self.lock = threading.Lock()

        self.action_buffer = []
        self.is_inferencing = False
        self.active_inference_workers = 0
        self.steps_executed_during_inference = 0
        self.last_inferred_time = 0.0
        self.reset_generation = 0
        self.reset_signal_mtime = (
            os.path.getmtime(RESET_SIGNAL_PATH) if os.path.exists(RESET_SIGNAL_PATH) else 0.0
        )

        self.last_gripper_pose = np.array(config.gripper_open_joints, dtype=np.float32)
        self.latest_state_gripper_pose = None
        self.last_gripper_debug_print_time = 0.0
        self.smoothed_action = None
        self.last_executed_action = None

        threading.Thread(target=self.control_listener_worker, daemon=True).start()
        threading.Thread(target=self.camera_listener_worker, daemon=True).start()
        threading.Thread(target=self.continuous_tracker_and_sync_worker, daemon=True).start()

        self.pub_joint_cmd = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        qos = rclpy.qos.qos_profile_sensor_data
        self.sub_state = self.create_subscription(
            JointState,
            TOPIC_JOINT_STATE,
            self.state_callback,
            qos_profile=qos,
        )

        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_loop)
        self.get_logger().info(
            "F100 VLA ready | "
            f"joints={TARGET_JOINT_NAMES} | state_dim={STATE_DIM} | "
            f"skip={SKIP_FIRST_K} chunk={CHUNK_SIZE}"
        )

    def reset_runtime_state(self, reason):
        with self.lock:
            self.reset_generation += 1
            self.action_buffer.clear()
            self.is_inferencing = False
            self.active_inference_workers = 0
            self.steps_executed_during_inference = 0
            self.last_inferred_time = 0.0
            self.last_gripper_pose = np.array(config.gripper_open_joints, dtype=np.float32)
            self.smoothed_action = None
            self.last_executed_action = None
        with self.img_lock:
            self.latest_image_payload = None
        with self.state_lock:
            self.state_buffer.clear()
        with self.snapshot_lock:
            self.latest_synced_snapshot = None
        self.get_logger().warn(f"F100 VLA runtime reset: {reason}")

    def control_listener_worker(self):
        if os.path.exists(CONTROL_SOCKET_PATH):
            os.remove(CONTROL_SOCKET_PATH)

        listener = Listener(CONTROL_SOCKET_PATH, family="AF_UNIX", authkey=b"vla_control")
        self.get_logger().info(f"F100 VLA control socket ready: {CONTROL_SOCKET_PATH}")

        while True:
            conn = None
            try:
                conn = listener.accept()
                command = conn.recv()
                if command == "reset":
                    self.reset_runtime_state("control socket")
                    conn.send("OK")
                else:
                    conn.send(f"ERROR: unknown command {command!r}")
            except Exception as exc:
                self.get_logger().warn(
                    f"VLA control command failed: {exc}",
                    throttle_duration_sec=1.0,
                )
            finally:
                if conn is not None:
                    conn.close()

    def check_reset_signal(self):
        try:
            mtime = os.path.getmtime(RESET_SIGNAL_PATH)
        except OSError:
            return
        if mtime > self.reset_signal_mtime:
            self.reset_signal_mtime = mtime
            self.reset_runtime_state(RESET_SIGNAL_PATH)

    def camera_listener_worker(self):
        if os.path.exists(CAMERA_SOCKET_PATH):
            os.remove(CAMERA_SOCKET_PATH)

        listener = Listener(CAMERA_SOCKET_PATH, family="AF_UNIX", authkey=b"vla_cam")
        self.get_logger().info(f"F100 camera socket ready: {CAMERA_SOCKET_PATH}")

        while True:
            try:
                conn = listener.accept()
                while True:
                    payload = conn.recv()
                    while conn.poll():
                        payload = conn.recv()

                    with self.img_lock:
                        self.latest_image_payload = payload
            except Exception as exc:
                self.get_logger().warn(
                    f"Camera stream disconnected: {exc}",
                    throttle_duration_sec=1.0,
                )

    def state_callback(self, state_msg):
        arrive_time = time.time()
        with self.state_lock:
            self.state_buffer.append({"state": state_msg, "timestamp": arrive_time})
            if state_msg.name:
                positions_by_name = dict(zip(state_msg.name, state_msg.position))
                gripper_values = [
                    positions_by_name.get(name) for name in GRIPPER_JOINT_NAMES
                ]
                if all(value is not None for value in gripper_values):
                    self.latest_state_gripper_pose = np.array(
                        gripper_values,
                        dtype=np.float32,
                    )
            else:
                positions = np.asarray(state_msg.position, dtype=np.float32)
                if len(positions) >= len(TARGET_JOINT_NAMES):
                    self.latest_state_gripper_pose = positions[
                        len(ARM_JOINT_NAMES) : len(TARGET_JOINT_NAMES)
                    ].copy()

    def continuous_tracker_and_sync_worker(self):
        context = zmq.Context()
        req = context.socket(zmq.REQ)
        req.connect(TRACKER_ENDPOINT)

        try:
            req.send_multipart([b"reset", b""])
            req.recv()
            self.get_logger().info(f"Tracker connected: {TRACKER_ENDPOINT}")
        except Exception as exc:
            self.get_logger().error(f"Tracker initial reset failed: {exc}")

        while True:
            with self.img_lock:
                payload = self.latest_image_payload

            if payload is None:
                time.sleep(0.01)
                continue

            try:
                img_time = payload.get("timestamp", 0)
                front_rgb = payload["front"]
                wrist_rgb = payload["wrist"]

                bgr_img = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode(".jpg", bgr_img)
                req.send_multipart([b"cam1", img_encoded.tobytes()])

                mask_bytes = req.recv()
                mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(bgr_img.shape[:2])

                best_state = None
                min_diff = float("inf")
                with self.state_lock:
                    for state_data in self.state_buffer:
                        diff = abs(state_data["timestamp"] - img_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_state = state_data["state"]

                if best_state is None:
                    time.sleep(0.005)
                    continue

                if min_diff <= 0.1:
                    with self.snapshot_lock:
                        self.latest_synced_snapshot = {
                            "front_rgb": front_rgb,
                            "wrist_rgb": wrist_rgb,
                            "mask": mask_uint8,
                            "state": best_state,
                            "sync_diff": min_diff,
                            "system_time": time.time(),
                        }
                else:
                    self.get_logger().warn(
                        f"Image/state sync diff too large: {min_diff * 1000:.1f}ms",
                        throttle_duration_sec=1.0,
                    )
            except Exception as exc:
                self.get_logger().warn(f"Tracker/sync failed: {exc}", throttle_duration_sec=1.0)

            time.sleep(0.005)

    def get_state_vector(self, state_msg):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)

        if state_msg.name:
            positions_by_name = dict(zip(state_msg.name, state_msg.position))
            for i, name in enumerate(ARM_JOINT_NAMES):
                state_vec[i] = positions_by_name.get(name, 0.0)
            for i, name in enumerate(GRIPPER_JOINT_NAMES, start=len(ARM_JOINT_NAMES)):
                fallback = self.last_gripper_pose[i - len(ARM_JOINT_NAMES)]
                state_vec[i] = positions_by_name.get(name, fallback)
        else:
            positions = np.asarray(state_msg.position, dtype=np.float32)
            usable = min(len(positions), len(TARGET_JOINT_NAMES))
            state_vec[:usable] = positions[:usable]
            if usable < len(TARGET_JOINT_NAMES):
                state_vec[len(ARM_JOINT_NAMES) : len(TARGET_JOINT_NAMES)] = (
                    self.last_gripper_pose
                )

        state_vec[-1] = 0.0
        return torch.from_numpy(state_vec).float()

    def action_distance(self, previous_action, next_action):
        previous = np.asarray(previous_action, dtype=np.float32)[: len(ARM_JOINT_NAMES)]
        next_value = np.asarray(next_action, dtype=np.float32)[: len(ARM_JOINT_NAMES)]
        return float(np.linalg.norm(next_value - previous))

    def apply_gripper_joint_scaling(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        for joint_name, scale in GRIPPER_JOINT_VALUE_SCALES.items():
            if joint_name not in GRIPPER_JOINT_NAMES:
                continue
            action_index = len(ARM_JOINT_NAMES) + GRIPPER_JOINT_NAMES.index(joint_name)
            action[action_index] *= float(scale)
        return action

    def clamp_gripper(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        start = len(ARM_JOINT_NAMES)
        end = len(TARGET_JOINT_NAMES)
        action[start:end] = np.clip(action[start:end], GRIPPER_MIN, GRIPPER_MAX)
        return action

    def maybe_print_gripper_debug(
        self,
        model_action,
        scaled_action,
        final_action,
        previous_command,
    ):
        if not DEBUG_GRIPPER:
            return

        now = time.time()
        if now - self.last_gripper_debug_print_time < DEBUG_GRIPPER_PRINT_INTERVAL:
            return
        self.last_gripper_debug_print_time = now

        start = len(ARM_JOINT_NAMES)
        end = len(TARGET_JOINT_NAMES)
        with self.state_lock:
            state_gripper = (
                None
                if self.latest_state_gripper_pose is None
                else self.latest_state_gripper_pose.copy()
            )

        def fmt(values):
            if values is None:
                return "None"
            arr = np.asarray(values, dtype=np.float32)
            return np.array2string(arr, precision=4, separator=", ")

        print(
            "\n[F100 gripper debug] "
            f"names={GRIPPER_JOINT_NAMES} "
            f"raw={fmt(model_action[start:end])} "
            f"scaled={fmt(scaled_action[start:end])} "
            f"cmd={fmt(final_action[start:end])} "
            f"prev_cmd={fmt(previous_command)} "
            f"isaac_state={fmt(state_gripper)} "
            f"open={fmt(config.gripper_open_joints)} "
            f"close={fmt(config.gripper_close_joints)} "
            f"scale={GRIPPER_JOINT_VALUE_SCALES}"
        )

    def append_action_with_arm_interpolation(self, stabilized, previous_action, target_action):
        distance = self.action_distance(previous_action, target_action)
        inserted_count = 0

        if distance > INTERPOLATE_ACTION_DISTANCE:
            interp_segments = int(np.ceil(distance / INTERPOLATE_ACTION_DISTANCE))
            interp_segments = min(interp_segments, MAX_INTERPOLATION_STEPS)
            for step in range(1, interp_segments):
                alpha = step / interp_segments
                interpolated = target_action.copy()
                arm_count = len(ARM_JOINT_NAMES)
                interpolated[:arm_count] = (
                    previous_action[:arm_count]
                    + alpha * (target_action[:arm_count] - previous_action[:arm_count])
                )
                interpolated[arm_count : len(TARGET_JOINT_NAMES)] = previous_action[
                    arm_count : len(TARGET_JOINT_NAMES)
                ]
                stabilized.append(interpolated.astype(np.float32))
            inserted_count = max(0, interp_segments - 1)

        stabilized.append(target_action.astype(np.float32))
        return target_action, inserted_count

    def stabilize_action_sequence(self, action_sequence, reference_action):
        stabilized = []
        previous_action = (
            None if reference_action is None else np.asarray(reference_action, dtype=np.float32)
        )
        dropped_count = 0
        inserted_count = 0
        replaced_count = 0

        for raw_action in action_sequence:
            action = np.asarray(raw_action, dtype=np.float32)
            if action.shape[0] < len(TARGET_JOINT_NAMES):
                dropped_count += 1
                continue

            if previous_action is None:
                stabilized.append(action)
                previous_action = action
                continue

            distance = self.action_distance(previous_action, action)
            if distance > OUTLIER_ACTION_DISTANCE:
                midpoint = action.copy()
                arm_count = len(ARM_JOINT_NAMES)
                midpoint[:arm_count] = (
                    previous_action[:arm_count]
                    + 0.5 * (action[:arm_count] - previous_action[:arm_count])
                )
                midpoint[arm_count : len(TARGET_JOINT_NAMES)] = previous_action[
                    arm_count : len(TARGET_JOINT_NAMES)
                ]
                previous_action, added = self.append_action_with_arm_interpolation(
                    stabilized,
                    previous_action,
                    midpoint,
                )
                inserted_count += added
                replaced_count += 1
                continue

            previous_action, added = self.append_action_with_arm_interpolation(
                stabilized,
                previous_action,
                action,
            )
            inserted_count += added

        if dropped_count or inserted_count or replaced_count:
            self.get_logger().info(
                "Trajectory stabilized: "
                f"inserted={inserted_count}, replaced={replaced_count}, dropped={dropped_count}"
            )

        return stabilized

    def inference_worker(self, data_snapshot, generation):
        try:
            front_rgb = data_snapshot["front_rgb"].copy()
            wrist_rgb = data_snapshot["wrist_rgb"].copy()
            mask_uint8 = data_snapshot["mask"]

            mask_255 = (mask_uint8 > 0).astype(np.uint8) * 255

            if front_rgb.shape[:2] != IMG_SIZE:
                front_rgb = cv2.resize(front_rgb, IMG_SIZE)
                mask_255 = cv2.resize(mask_255, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

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

            wrist_rgb = cv2.resize(wrist_rgb, IMG_SIZE)
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

            batch = {
                "obs": {
                    "rgbm": rgbm_tensor,
                    "right_cam_img": wrist_tensor,
                    "right_state": t_state.unsqueeze(0).unsqueeze(0).to(self.device),
                }
            }

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.policy.predict_action(batch["obs"])
                action_seq = output["action"] if isinstance(output, dict) else output
                full_traj = action_seq[0].cpu().numpy()

            with self.lock:
                if generation != self.reset_generation:
                    return

                start_idx = SKIP_FIRST_K
                skip_steps = self.steps_executed_during_inference
                chunk = full_traj[start_idx + skip_steps : start_idx + skip_steps + CHUNK_SIZE]
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
            self.get_logger().error(f"Inference failed: {exc}")
        finally:
            with self.lock:
                self.active_inference_workers = max(0, self.active_inference_workers - 1)
                if generation == self.reset_generation:
                    self.is_inferencing = False

    def control_loop(self):
        action_to_execute = None
        action_generation = None
        self.check_reset_signal()

        with self.lock:
            if self.is_inferencing:
                self.steps_executed_during_inference += 1

            can_start_inference = (
                len(self.action_buffer) <= TRIGGER_THRESHOLD
                and not self.is_inferencing
                and self.active_inference_workers == 0
            )

            if can_start_inference:
                with self.snapshot_lock:
                    if self.latest_synced_snapshot is not None:
                        data_snapshot = self.latest_synced_snapshot.copy()
                        current_snapshot_time = data_snapshot.get("system_time", 0)
                        time_since_last_sync = time.time() - current_snapshot_time

                        if time_since_last_sync > 0.5:
                            self.get_logger().warn(
                                f"Snapshot stale ({time_since_last_sync:.2f}s); clearing actions",
                                throttle_duration_sec=1.0,
                            )
                            self.action_buffer.clear()
                        elif current_snapshot_time != self.last_inferred_time:
                            self.last_inferred_time = current_snapshot_time
                            self.is_inferencing = True
                            self.active_inference_workers += 1
                            self.steps_executed_during_inference = 0
                            generation = self.reset_generation
                            threading.Thread(
                                target=self.inference_worker,
                                args=(data_snapshot, generation),
                                daemon=True,
                            ).start()

            if len(self.action_buffer) > 0:
                action_to_execute = self.action_buffer.pop(0)
            elif self.smoothed_action is not None:
                action_to_execute = self.smoothed_action

            if action_to_execute is not None:
                action_generation = self.reset_generation

        if action_to_execute is None:
            return

        try:
            with self.lock:
                if action_generation != self.reset_generation:
                    return

            if self.smoothed_action is None:
                self.smoothed_action = action_to_execute
            else:
                self.smoothed_action = (
                    SMOOTHING_ALPHA * action_to_execute
                    + (1.0 - SMOOTHING_ALPHA) * self.smoothed_action
                )

            model_action = np.asarray(self.smoothed_action, dtype=np.float32).copy()
            if model_action.shape[0] < len(TARGET_JOINT_NAMES):
                self.get_logger().warn(
                    f"Model action dim too small: {model_action.shape[0]} < "
                    f"{len(TARGET_JOINT_NAMES)}",
                    throttle_duration_sec=1.0,
                )
                return

            previous_gripper_command = self.last_gripper_pose.copy()
            scaled_action = self.apply_gripper_joint_scaling(model_action)
            final_action = self.clamp_gripper(scaled_action)
            self.maybe_print_gripper_debug(
                model_action,
                scaled_action,
                final_action,
                previous_gripper_command,
            )
            self.last_gripper_pose = final_action[
                len(ARM_JOINT_NAMES) : len(TARGET_JOINT_NAMES)
            ]

            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = TARGET_JOINT_NAMES
            joint_msg.position = final_action[: len(TARGET_JOINT_NAMES)].tolist()
            self.pub_joint_cmd.publish(joint_msg)
            self.last_executed_action = final_action.copy()

            status = "infer" if self.is_inferencing else "run"
            print(
                f"[F100 VLA {status}] remaining={len(self.action_buffer):02d} "
                f"arm0={final_action[0]:.2f} grip={self.last_gripper_pose[0]:.2f}    ",
                end="\r",
            )
        except Exception as exc:
            self.get_logger().warn(f"Control loop failed: {exc}", throttle_duration_sec=1.0)


def _shape_dim(cfg, *keys):
    value = cfg
    for key in keys:
        value = value[key] if isinstance(value, dict) else getattr(value, key)
    if not isinstance(value, (str, bytes)):
        try:
            return int(value[0])
        except (TypeError, KeyError, IndexError):
            pass
    return int(value)


def validate_policy_shape(cfg):
    state_dim = _shape_dim(cfg, "task", "shape_meta", "obs", "right_state", "shape")
    action_dim = _shape_dim(cfg, "task", "shape_meta", "action", "shape")

    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            "Checkpoint shape does not match myCobot F100 runtime: "
            f"cfg state/action=({state_dim}, {action_dim}), "
            f"runtime state/action=({STATE_DIM}, {ACTION_DIM}). "
            "Use an F100 checkpoint or override VLA_F100_RUN_DIR/VLA_F100_CKPT_NAME."
        )


def load_policy(run_dir, ckpt_name):
    cfg_path = os.path.join(run_dir, ".hydra")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.abspath(cfg_path), version_base=None)
    cfg = hydra.compose(config_name="config")
    validate_policy_shape(cfg)
    cfg.task.dataset = None

    from controller.workspace.train_dexgraspvla_controller_workspace import (
        TrainDexGraspVLAControllerWorkspace,
    )

    workspace = TrainDexGraspVLAControllerWorkspace(cfg)

    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    workspace.load_payload(
        payload,
        exclude_keys=("optimizer", "lr_scheduler"),
        include_keys=None,
    )

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()
    return policy


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        policy = load_policy(RUN_DIR, CKPT_NAME)
        node = VLA_IPCEvaluator(policy)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
