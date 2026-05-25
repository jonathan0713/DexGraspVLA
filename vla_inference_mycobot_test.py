import sys
import os
import threading
import time
import numpy as np
import cv2
import torch
import hydra
import dill
from omegaconf import OmegaConf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import zmq 
from multiprocessing.connection import Listener
import collections

sys.path.append(os.getcwd())

PROJECT_PATH = "/home/jonathan/Documents/ITRI_Project"
ROS_WS_PATH = f"{PROJECT_PATH}/ros2_ws"
DATA_COLLECTOR_PATH = f"{ROS_WS_PATH}/src/data_collector"

for path in (PROJECT_PATH, ROS_WS_PATH, DATA_COLLECTOR_PATH):
    if path not in sys.path:
        sys.path.append(path)

from mycobot.vla_config_mycobot import config

# ================= 使用者設定區 =================
# RUN_DIR = "data/outputs/2026.04.27/12.26_train_dexgraspvla_controller_grasp"
# CKPT_NAME = "epoch=0118-train_loss=0.0013.ckpt"
RUN_DIR = "data/outputs/from_itri/0512_pro6000"
CKPT_NAME = "epoch=0125-train_loss=0.0009.ckpt"

# RUN_DIR = "data/outputs/2026.04.25/19.15_train_dexgraspvla_controller_grasp"
# CKPT_NAME = "epoch=0124-train_loss=0.003.ckpt"

IMG_SIZE = (518, 518)

# 🌟 修正 1：關閉 EMA 平滑！
# 讓模型最原始的預測直接輸出，不要讓舊座標拖累新座標
SMOOTHING_ALPHA = 1.0

# 頻率與 collect_data_mycobot.py 的 rendering_dt=1/30 對齊
CONTROL_FREQ = 32

CHUNK_SIZE = 16

# 軌跡穩定策略：
# - 相鄰手臂 action 距離超過 INTERPOLATE_ACTION_DISTANCE 時，自動補線性插值點
# - 超過 OUTLIER_ACTION_DISTANCE 時，視為離群點並替換成與上一步的中點
# - 手臂與夾爪分別用不同參數計算插值段數
INTERPOLATE_ACTION_DISTANCE = 0.015
OUTLIER_ACTION_DISTANCE = 100
ENABLE_EXTREME_OUTLIER_DROP = True
EXTREME_OUTLIER_DROP_WARMUP_SECONDS = 0.0
EXTREME_OUTLIER_ACTION_DISTANCE = 100
MAX_INTERPOLATION_STEPS = 12
GRIPPER_INTERPOLATE_ACTION_DISTANCE = 0.06
EXTREME_GRIPPER_OUTLIER_DISTANCE = 0.8
MAX_GRIPPER_INTERPOLATION_STEPS = 8

# 自動 action window 軌跡優化：
# 先用局部速度、加速度、方向反轉找不穩定 window，只在 window 內做 anchored smoothing。
ENABLE_ACTION_WINDOW_OPTIMIZATION = False
ACTION_WINDOW_ARM_VELOCITY_THRESHOLD = INTERPOLATE_ACTION_DISTANCE * 3.0
ACTION_WINDOW_ARM_ACCEL_THRESHOLD = INTERPOLATE_ACTION_DISTANCE * 1.5
ACTION_WINDOW_ARM_SPIKE_THRESHOLD = INTERPOLATE_ACTION_DISTANCE * 2.5
ACTION_WINDOW_GRIPPER_VELOCITY_THRESHOLD = GRIPPER_INTERPOLATE_ACTION_DISTANCE * 2.0
ACTION_WINDOW_GRIPPER_ACCEL_THRESHOLD = GRIPPER_INTERPOLATE_ACTION_DISTANCE * 1.5
ACTION_WINDOW_DIRECTION_FLIP_COSINE = -0.25
ACTION_WINDOW_MIN_DELTA = INTERPOLATE_ACTION_DISTANCE * 0.5
ACTION_WINDOW_PADDING = 2
ACTION_WINDOW_MIN_LENGTH = 2
ACTION_WINDOW_SMOOTHING_PASSES = 3
ACTION_WINDOW_RAW_BLEND = 0.35
ACTION_WINDOW_ENDPOINT_RAW_BLEND = 0.75

# 自動對齊策略：
# diffusion horizon 的前幾步有時會重複舊軌跡；不固定跳過 K 步，而是找出
# 預測軌跡中最接近目前已執行姿態的位置，從下一步接續。
ADAPTIVE_ALIGN_ARM_TOLERANCE = INTERPOLATE_ACTION_DISTANCE * 3.0
ADAPTIVE_ALIGN_GRIPPER_TOLERANCE = GRIPPER_INTERPOLATE_ACTION_DISTANCE * 2.0
ADAPTIVE_ALIGN_GRIPPER_WEIGHT = 0.25

# 夾爪指定關節縮放倍率；1.0 代表不縮放
GRIPPER_JOINT_VALUE_SCALES = {
    "index_J2": 1.0,
    "ring_J2": 1.0,
    "thumb_J2": 1.0,
}
# ==============================================

TOPIC_JOINT_STATE = config.topic_isaac_joint_states
TOPIC_JOINT_CMD = config.topic_joint_command

ARM_JOINT_NAMES = list(config.arm_joint_names)
GRIPPER_JOINT_NAMES = list(config.gripper_joint_names)
TARGET_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
STATE_DIM = len(TARGET_JOINT_NAMES) + 1
ACTION_DIM = STATE_DIM

class VLA_IPCEvaluator(Node):
    def __init__(self, policy):
        super().__init__('dexgrasp_vla_ipc_evaluator')
        self.policy = policy
        self.device = policy.device

        # 緩衝區設計
        self.latest_image_payload = None  
        self.state_buffer = collections.deque(maxlen=60) 
        self.latest_synced_snapshot = None 
        
        self.img_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.snapshot_lock = threading.Lock()
        
        # 啟動雙背景小精靈
        threading.Thread(target=self.camera_listener_worker, daemon=True).start()
        threading.Thread(target=self.continuous_tracker_and_sync_worker, daemon=True).start()

        self.action_buffer = []  
        self.is_inferencing = False  
        self.trigger_threshold = 0
        self.start_time = time.time()
        
        # 🌟 新增：記錄上一次送進去推論的畫面時間戳記 (防重複機制)
        self.last_inferred_time = 0.0 
        
        self.last_gripper_pose = np.array(config.gripper_open_joints, dtype=np.float32)
        self.lock = threading.Lock()
        self.smoothed_action = None
        self.last_executed_action = None

        self.pub_joint_cmd = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        qos = rclpy.qos.qos_profile_sensor_data
        self.sub_state = self.create_subscription(JointState, TOPIC_JOINT_STATE, self.state_callback, qos_profile=qos)

        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_loop)
        self.get_logger().info("✅ VLA (完美時序配對版) 就緒 | 軌跡自動對齊啟用")

    def enforce_monotonic_gripper_closure(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        gripper_start = len(ARM_JOINT_NAMES)
        gripper_end = len(TARGET_JOINT_NAMES)
        predicted_gripper = action[gripper_start:gripper_end]

        previous_abs = np.abs(self.last_gripper_pose)
        predicted_abs = np.abs(predicted_gripper)
        closing_mask = predicted_abs >= previous_abs

        monotonic_gripper = np.where(closing_mask, predicted_gripper, self.last_gripper_pose)
        action[gripper_start:gripper_end] = monotonic_gripper
        return action

    def apply_gripper_joint_scaling(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        for joint_name, scale in GRIPPER_JOINT_VALUE_SCALES.items():
            if joint_name not in GRIPPER_JOINT_NAMES:
                continue
            action_index = len(ARM_JOINT_NAMES) + GRIPPER_JOINT_NAMES.index(joint_name)
            action[action_index] *= float(scale)
        return action

    def action_distance(self, previous_action, next_action):
        previous = np.asarray(previous_action, dtype=np.float32)[:len(ARM_JOINT_NAMES)]
        next_value = np.asarray(next_action, dtype=np.float32)[:len(ARM_JOINT_NAMES)]
        return float(np.linalg.norm(next_value - previous))

    def gripper_distance(self, previous_action, next_action):
        start = len(ARM_JOINT_NAMES)
        end = len(TARGET_JOINT_NAMES)
        previous = np.asarray(previous_action, dtype=np.float32)[start:end]
        next_value = np.asarray(next_action, dtype=np.float32)[start:end]
        return float(np.linalg.norm(next_value - previous))

    def should_drop_extreme_outliers(self):
        elapsed = time.time() - self.start_time
        return ENABLE_EXTREME_OUTLIER_DROP and elapsed >= EXTREME_OUTLIER_DROP_WARMUP_SECONDS

    def select_adaptive_start_index(self, full_traj, reference_action):
        traj = np.asarray(full_traj, dtype=np.float32)
        if traj.ndim != 2 or len(traj) == 0:
            return 0, None

        target_dim = min(traj.shape[1], len(TARGET_JOINT_NAMES))
        if target_dim < len(ARM_JOINT_NAMES):
            return 0, None

        reference = np.asarray(reference_action, dtype=np.float32)
        if reference.shape[0] < target_dim:
            return 0, None

        arm_end = len(ARM_JOINT_NAMES)
        gripper_start = arm_end
        gripper_end = target_dim

        arm_distances = np.linalg.norm(
            traj[:, :arm_end] - reference[:arm_end],
            axis=1,
        )

        if gripper_end > gripper_start:
            gripper_distances = np.linalg.norm(
                traj[:, gripper_start:gripper_end] - reference[gripper_start:gripper_end],
                axis=1,
            )
        else:
            gripper_distances = np.zeros_like(arm_distances)

        score = arm_distances + ADAPTIVE_ALIGN_GRIPPER_WEIGHT * gripper_distances
        best_idx = int(np.argmin(score))
        best_arm_distance = float(arm_distances[best_idx])
        best_gripper_distance = float(gripper_distances[best_idx])
        is_aligned = (
            best_arm_distance <= ADAPTIVE_ALIGN_ARM_TOLERANCE
            and best_gripper_distance <= ADAPTIVE_ALIGN_GRIPPER_TOLERANCE
        )

        if not is_aligned:
            return 0, {
                "mode": "fallback",
                "matched_idx": best_idx,
                "arm_distance": best_arm_distance,
                "gripper_distance": best_gripper_distance,
            }

        start_idx = min(best_idx + 1, len(traj) - 1)
        return start_idx, {
            "mode": "aligned",
            "matched_idx": best_idx,
            "arm_distance": best_arm_distance,
            "gripper_distance": best_gripper_distance,
        }

    def detect_unstable_action_windows(self, action_sequence, reference_action):
        actions = np.asarray(action_sequence, dtype=np.float32)
        if (
            not ENABLE_ACTION_WINDOW_OPTIMIZATION
            or actions.ndim != 2
            or len(actions) < ACTION_WINDOW_MIN_LENGTH
        ):
            return [], {}

        target_dim = min(actions.shape[1], len(TARGET_JOINT_NAMES))
        if target_dim < len(ARM_JOINT_NAMES):
            return [], {}

        reference = np.asarray(reference_action, dtype=np.float32)
        if reference.shape[0] < target_dim:
            return [], {}

        points = np.vstack([
            reference[:target_dim],
            actions[:, :target_dim],
        ])
        arm_end = len(ARM_JOINT_NAMES)
        gripper_start = arm_end
        gripper_end = target_dim
        action_count = len(actions)
        unstable = np.zeros(action_count, dtype=bool)

        arm_delta = np.diff(points[:, :arm_end], axis=0)
        arm_velocity = np.linalg.norm(arm_delta, axis=1)
        high_arm_velocity = arm_velocity > ACTION_WINDOW_ARM_VELOCITY_THRESHOLD

        if gripper_end > gripper_start:
            gripper_delta = np.diff(points[:, gripper_start:gripper_end], axis=0)
            gripper_velocity = np.linalg.norm(gripper_delta, axis=1)
            high_gripper_velocity = gripper_velocity > ACTION_WINDOW_GRIPPER_VELOCITY_THRESHOLD
        else:
            gripper_delta = np.zeros((action_count, 0), dtype=np.float32)
            gripper_velocity = np.zeros(action_count, dtype=np.float32)
            high_gripper_velocity = np.zeros(action_count, dtype=bool)

        if action_count >= 2:
            arm_acceleration = np.linalg.norm(np.diff(arm_delta, axis=0), axis=1)
            high_arm_acceleration = arm_acceleration > ACTION_WINDOW_ARM_ACCEL_THRESHOLD
            unstable[1:] |= high_arm_acceleration

            if gripper_delta.shape[1] > 0:
                gripper_acceleration = np.linalg.norm(np.diff(gripper_delta, axis=0), axis=1)
                high_gripper_acceleration = gripper_acceleration > ACTION_WINDOW_GRIPPER_ACCEL_THRESHOLD
                unstable[1:] |= high_gripper_acceleration
            else:
                gripper_acceleration = np.zeros(action_count - 1, dtype=np.float32)

            previous_delta = arm_delta[:-1]
            next_delta = arm_delta[1:]
            previous_norm = np.linalg.norm(previous_delta, axis=1)
            next_norm = np.linalg.norm(next_delta, axis=1)
            denom = np.maximum(previous_norm * next_norm, 1e-6)
            cosine = np.sum(previous_delta * next_delta, axis=1) / denom
            direction_flip = (
                (cosine < ACTION_WINDOW_DIRECTION_FLIP_COSINE)
                & (previous_norm > ACTION_WINDOW_MIN_DELTA)
                & (next_norm > ACTION_WINDOW_MIN_DELTA)
            )
            unstable[:-1] |= direction_flip
            unstable[1:] |= direction_flip
        else:
            arm_acceleration = np.zeros(0, dtype=np.float32)
            gripper_acceleration = np.zeros(0, dtype=np.float32)
            direction_flip = np.zeros(0, dtype=bool)

        if action_count >= 2:
            arm_spike_error = np.linalg.norm(
                points[1:-1, :arm_end] - 0.5 * (points[:-2, :arm_end] + points[2:, :arm_end]),
                axis=1,
            )
            unstable[:-1] |= arm_spike_error > ACTION_WINDOW_ARM_SPIKE_THRESHOLD
        else:
            arm_spike_error = np.zeros(0, dtype=np.float32)

        windows = []
        idx = 0
        while idx < action_count:
            if not unstable[idx]:
                idx += 1
                continue

            end = idx + 1
            while end < action_count and unstable[end]:
                end += 1

            start = max(0, idx - ACTION_WINDOW_PADDING)
            padded_end = min(action_count, end + ACTION_WINDOW_PADDING)
            if padded_end - start >= ACTION_WINDOW_MIN_LENGTH:
                if windows and start <= windows[-1][1]:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], padded_end))
                else:
                    windows.append((start, padded_end))

            idx = end

        metrics = {
            "max_arm_velocity": float(np.max(arm_velocity)) if len(arm_velocity) else 0.0,
            "max_arm_acceleration": float(np.max(arm_acceleration)) if len(arm_acceleration) else 0.0,
            "max_gripper_velocity": float(np.max(gripper_velocity)) if len(gripper_velocity) else 0.0,
            "max_gripper_acceleration": float(np.max(gripper_acceleration)) if len(gripper_acceleration) else 0.0,
            "max_spike_error": float(np.max(arm_spike_error)) if len(arm_spike_error) else 0.0,
            "high_arm_velocity_count": int(np.sum(high_arm_velocity)) if len(high_arm_velocity) else 0,
            "high_gripper_velocity_count": int(np.sum(high_gripper_velocity)) if len(high_gripper_velocity) else 0,
            "direction_flips": int(np.sum(direction_flip)) if len(direction_flip) else 0,
        }
        return windows, metrics

    def smooth_action_window(self, actions, start, end, reference_action):
        optimized = actions.copy()
        target_dim = min(optimized.shape[1], len(TARGET_JOINT_NAMES))
        reference = np.asarray(reference_action, dtype=np.float32)
        raw = actions[:, :target_dim].copy()

        for _ in range(ACTION_WINDOW_SMOOTHING_PASSES):
            next_optimized = optimized.copy()
            for action_idx in range(start, end):
                previous_value = (
                    reference[:target_dim]
                    if action_idx == 0
                    else optimized[action_idx - 1, :target_dim]
                )
                next_value = (
                    optimized[action_idx + 1, :target_dim]
                    if action_idx + 1 < len(optimized)
                    else raw[action_idx]
                )
                neighbor_average = 0.5 * (previous_value + next_value)
                raw_blend = ACTION_WINDOW_RAW_BLEND
                if action_idx == start or action_idx == end - 1:
                    raw_blend = ACTION_WINDOW_ENDPOINT_RAW_BLEND

                next_optimized[action_idx, :target_dim] = (
                    raw_blend * raw[action_idx]
                    + (1.0 - raw_blend) * neighbor_average
                )

            optimized = next_optimized

        return optimized

    def optimize_unstable_action_windows(self, action_sequence, reference_action):
        actions = np.asarray(action_sequence, dtype=np.float32)
        if actions.ndim != 2 or len(actions) == 0:
            return action_sequence

        windows, metrics = self.detect_unstable_action_windows(actions, reference_action)
        if not windows:
            return actions

        optimized = actions.copy()
        for start, end in windows:
            optimized = self.smooth_action_window(optimized, start, end, reference_action)

        optimized_points = sum(end - start for start, end in windows)
        self.get_logger().info(
            "🪟 action window 優化: "
            f"{len(windows)} 段/{optimized_points} 點 | "
            f"v={metrics['max_arm_velocity']:.4f} | "
            f"a={metrics['max_arm_acceleration']:.4f} | "
            f"spike={metrics['max_spike_error']:.4f} | "
            f"v_hi={metrics['high_arm_velocity_count']} | "
            f"flip={metrics['direction_flips']}"
        )
        return optimized

    def interpolation_segments(self, distance, threshold, max_steps):
        if distance <= threshold:
            return 1
        return min(int(np.ceil(distance / threshold)), max_steps)

    def append_action_with_interpolation(self, stabilized, previous_action, target_action):
        arm_segments = self.interpolation_segments(
            self.action_distance(previous_action, target_action),
            INTERPOLATE_ACTION_DISTANCE,
            MAX_INTERPOLATION_STEPS,
        )
        gripper_segments = self.interpolation_segments(
            self.gripper_distance(previous_action, target_action),
            GRIPPER_INTERPOLATE_ACTION_DISTANCE,
            MAX_GRIPPER_INTERPOLATION_STEPS,
        )
        interp_segments = max(arm_segments, gripper_segments)

        for step in range(1, interp_segments):
            alpha = step / interp_segments
            interpolated = target_action.copy()
            interpolated[:len(TARGET_JOINT_NAMES)] = (
                previous_action[:len(TARGET_JOINT_NAMES)]
                + alpha
                * (target_action[:len(TARGET_JOINT_NAMES)] - previous_action[:len(TARGET_JOINT_NAMES)])
            )
            stabilized.append(interpolated.astype(np.float32))

        stabilized.append(target_action.astype(np.float32))
        return target_action, max(0, interp_segments - 1)

    def stabilize_action_sequence(self, action_sequence, reference_action):
        stabilized = []
        previous_action = None if reference_action is None else np.asarray(reference_action, dtype=np.float32)
        dropped_count = 0
        inserted_count = 0
        replaced_count = 0
        extreme_dropped_count = 0

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
            gripper_distance = self.gripper_distance(previous_action, action)
            if (
                self.should_drop_extreme_outliers()
                and (
                    distance > EXTREME_OUTLIER_ACTION_DISTANCE
                    or gripper_distance > EXTREME_GRIPPER_OUTLIER_DISTANCE
                )
            ):
                extreme_dropped_count += 1
                continue

            if distance > OUTLIER_ACTION_DISTANCE:
                midpoint = action.copy()
                midpoint[:len(ARM_JOINT_NAMES)] = (
                    previous_action[:len(ARM_JOINT_NAMES)]
                    + 0.5
                    * (action[:len(ARM_JOINT_NAMES)] - previous_action[:len(ARM_JOINT_NAMES)])
                )
                midpoint[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)] = (
                    previous_action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]
                    + 0.5
                    * (
                        action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]
                        - previous_action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]
                    )
                )
                previous_action, added = self.append_action_with_interpolation(
                    stabilized,
                    previous_action,
                    midpoint,
                )
                inserted_count += added
                replaced_count += 1
                continue

            previous_action, added = self.append_action_with_interpolation(
                stabilized,
                previous_action,
                action,
            )
            inserted_count += added

        if dropped_count or inserted_count or replaced_count or extreme_dropped_count:
            self.get_logger().info(
                f"🧩 軌跡穩定: 插值 {inserted_count} 點，替換離群 {replaced_count} 點，"
                f"刪除極端離群 {extreme_dropped_count} 點，刪除異常維度 {dropped_count} 點"
            )

        return stabilized

    def camera_listener_worker(self):
        socket_path = '/tmp/vla_cam_stream'
        if os.path.exists(socket_path): os.remove(socket_path) 
        listener = Listener(socket_path, family='AF_UNIX', authkey=b'vla_cam')
        
        while True:
            try:
                conn = listener.accept()
                while True:
                    payload = conn.recv()
                    while conn.poll(): payload = conn.recv()
                    
                    with self.img_lock:
                        self.latest_image_payload = payload
            except Exception: pass

    def state_callback(self, state_msg):
        # 🌟 核心修正：不信任 ROS Header，狀態一抵達立刻用系統絕對時間蓋章
        arrive_time = time.time()
        with self.state_lock:
            self.state_buffer.append({'state': state_msg, 'timestamp': arrive_time})

    def continuous_tracker_and_sync_worker(self):
        zmq_req_context = zmq.Context()
        zmq_req = zmq_req_context.socket(zmq.REQ)
        zmq_req.connect("tcp://127.0.0.1:5555")
        zmq_req.send_multipart([b"reset", b""])
        zmq_req.recv()
        self.get_logger().info("🔗 Tracker 連線成功，開始非同步追蹤與配對...")

        while True:
            with self.img_lock:
                payload = self.latest_image_payload
            
            if payload is None:
                time.sleep(0.01); continue
                
            img_time = payload.get('timestamp', 0)
            front_rgb = payload['front']
            wrist_rgb = payload['wrist']
            
            try:
                # 1. 把圖丟給 Cutie 算 Mask (此時 ROS 狀態有時間在網路上飛奔過來)
                bgr_img = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr_img)
                zmq_req.send_multipart([b"cam1", img_encoded.tobytes()])
                
                mask_bytes = zmq_req.recv()
                mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(bgr_img.shape[:2])
                
                # 2. Cutie 算完了！去翻找剛好對應這張圖時間的 ROS 狀態
                best_state = None
                min_diff = float('inf')
                
                with self.state_lock:
                    for st_data in self.state_buffer:
                        diff = abs(st_data['timestamp'] - img_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_state = st_data['state']
                
                # 3. 容許 100 毫秒的微小網路誤差，超過就警告
                if best_state is not None:
                    if min_diff <= 0.1:
                        with self.snapshot_lock:
                            self.latest_synced_snapshot = {
                                'front_rgb': front_rgb,
                                'wrist_rgb': wrist_rgb,
                                'mask': mask_uint8,  
                                'state': best_state, 
                                'sync_diff': min_diff,
                                'system_time': time.time()  # 🌟 新增：記錄這包資料產生的絕對時間
                            }
                    else:
                        print(f"⚠️ [警告] 影像與狀態時間錯位過大: {min_diff*1000:.1f}ms", end='\r')

            except Exception: pass
            time.sleep(0.005) 

    def get_state_vector(self, state_msg):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)

        if state_msg.name:
            positions_by_name = dict(zip(state_msg.name, state_msg.position))
            for i, name in enumerate(ARM_JOINT_NAMES):
                state_vec[i] = positions_by_name.get(name, 0.0)
            for i, name in enumerate(GRIPPER_JOINT_NAMES, start=len(ARM_JOINT_NAMES)):
                state_vec[i] = positions_by_name.get(name, self.last_gripper_pose[i - len(ARM_JOINT_NAMES)])
        else:
            positions = np.asarray(state_msg.position, dtype=np.float32)
            usable = min(len(positions), len(TARGET_JOINT_NAMES))
            state_vec[:usable] = positions[:usable]
            if usable < len(TARGET_JOINT_NAMES):
                state_vec[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)] = self.last_gripper_pose

        state_vec[-1] = 0.0
        return torch.from_numpy(state_vec).float()

    def inference_worker(self, data_snapshot):
        try:
            front_rgb = data_snapshot['front_rgb'].copy()
            wrist_rgb = data_snapshot['wrist_rgb'].copy()
            mask_uint8 = data_snapshot['mask']
            
            mask_255 = mask_uint8 * 255
            
            if front_rgb.shape[:2] != IMG_SIZE:
                front_rgb = cv2.resize(front_rgb, IMG_SIZE)
                mask_255 = cv2.resize(mask_255, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            rgbm = np.concatenate([front_rgb, mask_255[:, :, np.newaxis]], axis=-1) 
            rgbm_tensor = torch.from_numpy(rgbm).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            
            wrist_rgb = cv2.resize(wrist_rgb, IMG_SIZE)
            wrist_tensor = torch.from_numpy(wrist_rgb).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0 

            t_state = self.get_state_vector(data_snapshot['state'])

            batch = {'obs': {
                'rgbm': rgbm_tensor, 
                'right_cam_img': wrist_tensor,
                'right_state': t_state.unsqueeze(0).unsqueeze(0).to(self.device)
            }}

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.policy.predict_action(batch['obs'])
                action_seq = output['action'] if isinstance(output, dict) else output
                full_traj = action_seq[0].cpu().numpy()

            with self.lock:
                reference_action = (
                    self.last_executed_action
                    if self.last_executed_action is not None
                    else t_state.cpu().numpy()
                )
                start_idx, alignment = self.select_adaptive_start_index(
                    full_traj,
                    reference_action,
                )

                chunk = full_traj[start_idx : start_idx + CHUNK_SIZE]
                candidate_actions = chunk if len(chunk) > 0 else full_traj[-1:]
                self.action_buffer = self.stabilize_action_sequence(
                    candidate_actions,
                    reference_action,
                )
                if alignment is not None:
                    self.get_logger().info(
                        "🔎 軌跡對齊: "
                        f"{alignment['mode']} | start={start_idx} | "
                        f"match={alignment['matched_idx']} | "
                        f"arm={alignment['arm_distance']:.4f} | "
                        f"gripper={alignment['gripper_distance']:.4f}"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}")
        finally:
            with self.lock: self.is_inferencing = False

    def control_loop(self):
        action_to_execute = None

        with self.lock:
            if len(self.action_buffer) <= self.trigger_threshold and not self.is_inferencing:
                with self.snapshot_lock:
                    if self.latest_synced_snapshot is not None:
                        data_snapshot = self.latest_synced_snapshot.copy()
                        
                        # 🌟 取得資料的時間戳記
                        current_snapshot_time = data_snapshot.get('system_time', 0)
                        time_since_last_sync = time.time() - current_snapshot_time
                        
                        # 🌟 雙重防護機制開始
                        if time_since_last_sync > 0.5:
                            # 防護 1：資料太舊 (Tracker 斷線或網路延遲)
                            self.get_logger().warn(f"⚠️ 畫面過期 ({time_since_last_sync:.2f}s)，緊急煞車！", throttle_duration_sec=1.0)
                            self.action_buffer.clear()
                            
                        elif current_snapshot_time == self.last_inferred_time:
                            # 防護 2：資料沒過期，但這張畫面剛才已經推論過了 (防重複推論)
                            pass 
                            
                        else:
                            # 通過檢查：全新且新鮮的畫面！
                            self.last_inferred_time = current_snapshot_time # 標記為已使用
                            self.is_inferencing = True
                            threading.Thread(target=self.inference_worker, args=(data_snapshot,), daemon=True).start()

            if len(self.action_buffer) > 0:
                action_to_execute = self.action_buffer.pop(0)
            elif self.smoothed_action is not None:
                action_to_execute = self.smoothed_action

        if action_to_execute is not None:
            try:
                if self.smoothed_action is None:
                    self.smoothed_action = action_to_execute
                else:
                    self.smoothed_action = SMOOTHING_ALPHA * action_to_execute + (1.0 - SMOOTHING_ALPHA) * self.smoothed_action
                
                final_action = np.asarray(self.smoothed_action, dtype=np.float32)
                if final_action.shape[0] < len(TARGET_JOINT_NAMES):
                    self.get_logger().warn(
                        f"⚠️ 模型 action 維度不足: {final_action.shape[0]} < {len(TARGET_JOINT_NAMES)}",
                        throttle_duration_sec=1.0,
                    )
                    return

                final_action = self.apply_gripper_joint_scaling(final_action)
                final_action = self.enforce_monotonic_gripper_closure(final_action)
                self.last_gripper_pose = final_action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]

                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = TARGET_JOINT_NAMES
                joint_msg.position = final_action[:len(TARGET_JOINT_NAMES)].tolist()
                self.pub_joint_cmd.publish(joint_msg)
                self.last_executed_action = final_action.copy()

                status = "🤔 VLA 運算中" if self.is_inferencing else "✅ 執行中    "
                print(f"[{status}] 剩餘: {len(self.action_buffer):02d}步 | Arm[0]: {final_action[0]:.2f}    ", end='\r')

            except Exception as e:
                pass

def load_policy(run_dir, ckpt_name):
    cfg_path = os.path.join(run_dir, ".hydra")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.abspath(cfg_path), version_base=None)
    cfg = hydra.compose(config_name="config")
    cfg.task.dataset = None 
    
    from controller.workspace.train_dexgraspvla_controller_workspace import TrainDexGraspVLAControllerWorkspace
    workspace = TrainDexGraspVLAControllerWorkspace(cfg)
    
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    workspace.load_payload(
        payload,
        exclude_keys=("optimizer", "lr_scheduler"),
        include_keys=None
    )
    
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()
    return policy

def main(args=None):
    rclpy.init(args=args)
    try:
        policy = load_policy(RUN_DIR, CKPT_NAME)
        node = VLA_IPCEvaluator(policy)
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()
