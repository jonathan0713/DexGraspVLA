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
CKPT_NAME = "epoch=0160-train_loss=0.0006.ckpt"
# CKPT_NAME = "latest.ckpt"
# RUN_DIR = "data/outputs/2026.04.25/19.15_train_dexgraspvla_controller_grasp"
# CKPT_NAME = "epoch=0124-train_loss=0.003.ckpt"

IMG_SIZE = (518, 518)

# 🌟 修正 1：關閉 EMA 平滑！
# 讓模型最原始的預測直接輸出，不要讓舊座標拖累新座標
SMOOTHING_ALPHA = 1.0

# 頻率與 collect_data_mycobot.py 的 rendering_dt=1/30 對齊
CONTROL_FREQ = 48
RESET_SIGNAL_PATH = os.environ.get("VLA_INFERENCE_RESET_FILE", "/tmp/vla_inference_reset.flag")
CONTROL_SOCKET_PATH = os.environ.get("VLA_INFERENCE_CONTROL_SOCKET", "/tmp/vla_inference_control.sock")

# 🌟 修正 3：因為頻率變高了，我們把捨棄的步數稍微拉長
SKIP_FIRST_K = 16
CHUNK_SIZE = 64
# (在 30Hz 下，跳過前 5 步代表跳過 0.16 秒的猶豫期)

# 軌跡穩定策略：
# - 相鄰手臂 action 距離超過 INTERPOLATE_ACTION_DISTANCE 時，自動補線性插值點
# - 超過 OUTLIER_ACTION_DISTANCE 時，視為離群點並替換成與上一步的中點
# - 插值只補手臂 6 軸，夾爪 6 軸不做插值
INTERPOLATE_ACTION_DISTANCE = 0.015
OUTLIER_ACTION_DISTANCE = 0.1
MAX_INTERPOLATION_STEPS = 12

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

        self.action_buffer = []  
        self.is_inferencing = False  
        self.active_inference_workers = 0
        self.trigger_threshold = 12
        self.steps_executed_during_inference = 0
        
        # 🌟 新增：記錄上一次送進去推論的畫面時間戳記 (防重複機制)
        self.last_inferred_time = 0.0 
        self.reset_generation = 0
        self.reset_signal_mtime = os.path.getmtime(RESET_SIGNAL_PATH) if os.path.exists(RESET_SIGNAL_PATH) else 0.0
        
        self.last_gripper_pose = np.array(config.gripper_open_joints, dtype=np.float32)
        self.lock = threading.Lock()
        self.smoothed_action = None
        self.last_executed_action = None
        
        # 啟動背景 workers
        threading.Thread(target=self.control_listener_worker, daemon=True).start()
        threading.Thread(target=self.camera_listener_worker, daemon=True).start()
        threading.Thread(target=self.continuous_tracker_and_sync_worker, daemon=True).start()

        self.pub_joint_cmd = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        qos = rclpy.qos.qos_profile_sensor_data
        self.sub_state = self.create_subscription(JointState, TOPIC_JOINT_STATE, self.state_callback, qos_profile=qos)

        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_loop)
        self.get_logger().info(f"✅ VLA (完美時序配對版) 就緒 | 捨棄步數: {SKIP_FIRST_K}")

    def reset_runtime_state(self, reason):
        with self.lock:
            self.reset_generation += 1
            self.action_buffer.clear()
            self.is_inferencing = False
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
        self.get_logger().warn(f"🔄 VLA runtime reset: {reason}")

    def control_listener_worker(self):
        socket_path = CONTROL_SOCKET_PATH
        if os.path.exists(socket_path):
            os.remove(socket_path)
        listener = Listener(socket_path, family='AF_UNIX', authkey=b'vla_control')
        self.get_logger().info(f"🕹️ VLA control socket ready: {socket_path}")

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
            except Exception as e:
                self.get_logger().warn(f"VLA control command failed: {e}", throttle_duration_sec=1.0)
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

    def append_action_with_arm_interpolation(self, stabilized, previous_action, target_action):
        distance = self.action_distance(previous_action, target_action)
        inserted_count = 0

        if distance > INTERPOLATE_ACTION_DISTANCE:
            interp_segments = int(np.ceil(distance / INTERPOLATE_ACTION_DISTANCE))
            interp_segments = min(interp_segments, MAX_INTERPOLATION_STEPS)
            for step in range(1, interp_segments):
                alpha = step / interp_segments
                interpolated = target_action.copy()
                interpolated[:len(ARM_JOINT_NAMES)] = (
                    previous_action[:len(ARM_JOINT_NAMES)]
                    + alpha
                    * (target_action[:len(ARM_JOINT_NAMES)] - previous_action[:len(ARM_JOINT_NAMES)])
                )
                interpolated[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)] = (
                    previous_action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]
                )
                stabilized.append(interpolated.astype(np.float32))
            inserted_count = max(0, interp_segments - 1)

        stabilized.append(target_action.astype(np.float32))
        return target_action, inserted_count

    def stabilize_action_sequence(self, action_sequence, reference_action):
        stabilized = []
        previous_action = None if reference_action is None else np.asarray(reference_action, dtype=np.float32)
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
                midpoint[:len(ARM_JOINT_NAMES)] = (
                    previous_action[:len(ARM_JOINT_NAMES)]
                    + 0.5
                    * (action[:len(ARM_JOINT_NAMES)] - previous_action[:len(ARM_JOINT_NAMES)])
                )
                midpoint[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)] = (
                    previous_action[len(ARM_JOINT_NAMES):len(TARGET_JOINT_NAMES)]
                )
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
                f"🧩 軌跡穩定: 插值 {inserted_count} 點，替換離群 {replaced_count} 點，刪除 {dropped_count} 點"
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

    def inference_worker(self, data_snapshot, generation):
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
                    
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}")
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
                len(self.action_buffer) <= self.trigger_threshold
                and not self.is_inferencing
                and self.active_inference_workers == 0
            )

            if can_start_inference:
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

        if action_to_execute is not None:
            try:
                with self.lock:
                    if action_generation != self.reset_generation:
                        return

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
