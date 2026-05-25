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

# ================= 使用者設定區 =================
# RUN_DIR = "data/outputs/2026.04.27/12.26_train_dexgraspvla_controller_grasp"
# CKPT_NAME = "epoch=0118-train_loss=0.0013.ckpt"
RUN_DIR = "data/outputs/from_itri/0428_pro6000"
CKPT_NAME = "epoch=0120-train_loss=0.0014.ckpt"

# RUN_DIR = "data/outputs/2026.04.25/19.15_train_dexgraspvla_controller_grasp"
# CKPT_NAME = "epoch=0124-train_loss=0.003.ckpt"

IMG_SIZE = (518, 518)

# 🌟 修正 1：關閉 EMA 平滑！
# 讓模型最原始的預測直接輸出，不要讓舊座標拖累新座標
SMOOTHING_ALPHA = 1.0

# 🌟 修正 2：頻率必須與收集資料 (rendering_dt) 嚴格一致！
# 如果 collect_data 是 30Hz，這裡絕對不能是 16Hz
CONTROL_FREQ = 16

# 🌟 修正 3：因為頻率變高了，我們把捨棄的步數稍微拉長
SKIP_FIRST_K = 0
CHUNK_SIZE = 16
# (在 30Hz 下，跳過前 5 步代表跳過 0.16 秒的猶豫期)
# ==============================================

TOPIC_JOINT_STATE = "isaac_joint_states"
TOPIC_JOINT_CMD = "joint_command"

TARGET_JOINT_NAMES = [
    'shoulder_1_joint', 'shoulder_2_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

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
        self.steps_executed_during_inference = 0
        
        self.settle_start_time = None

        # 🌟 新增：記錄上一次送進去推論的畫面時間戳記 (防重複機制)
        self.last_inferred_time = 0.0 
        
        self.last_gripper_pose = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        self.lock = threading.Lock()
        self.smoothed_action = None

        self.pub_joint_cmd = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        qos = rclpy.qos.qos_profile_sensor_data
        self.sub_state = self.create_subscription(JointState, TOPIC_JOINT_STATE, self.state_callback, qos_profile=qos)

        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_loop)
        self.get_logger().info(f"✅ VLA (完美時序配對版) 就緒 | 捨棄步數: {SKIP_FIRST_K}")

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
        arm_joints = np.zeros(6, dtype=np.float32)
        for i, name in enumerate(TARGET_JOINT_NAMES):
            if name in state_msg.name:
                idx = state_msg.name.index(name)
                arm_joints[i] = state_msg.position[idx]

        gripper_vec = np.zeros(7, dtype=np.float32)
        gripper_vec[:6] = self.last_gripper_pose      
        gripper_vec[6] = 0.0                          
        return torch.from_numpy(np.concatenate([arm_joints, gripper_vec])).float()

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
                start_idx = SKIP_FIRST_K
                skip_steps = self.steps_executed_during_inference
                
                chunk = full_traj[start_idx + skip_steps : start_idx + skip_steps + CHUNK_SIZE]
                if len(chunk) > 0:
                    self.action_buffer = list(chunk)
                else:
                    self.action_buffer = list(full_traj[-1:])
                    
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}")
        finally:
            with self.lock: self.is_inferencing = False

    def control_loop(self):
        action_to_execute = None

        with self.lock:
            if self.is_inferencing:
                self.steps_executed_during_inference += 1

            # 當動作執行完畢，且沒有在推論時
            if len(self.action_buffer) <= self.trigger_threshold and not self.is_inferencing:
                
                # 🌟 新增：開始倒數 2 秒
                if self.settle_start_time is None:
                    self.settle_start_time = time.time()
                    print("🛑 動作耗盡，等待 2 秒讓系統與畫面完全同步...", end='\r')
                    
                # 🌟 確認已經靜止等待超過 2 秒
                elif time.time() - self.settle_start_time >= 2.0:
                    
                    with self.snapshot_lock:
                        if self.latest_synced_snapshot is not None:
                            data_snapshot = self.latest_synced_snapshot.copy()
                            current_snapshot_time = data_snapshot.get('system_time', 0)
                            time_since_last_sync = time.time() - current_snapshot_time
                            
                            if time_since_last_sync > 0.5:
                                self.get_logger().warn(f"⚠️ 畫面過期 ({time_since_last_sync:.2f}s)，緊急煞車！", throttle_duration_sec=1.0)
                                self.action_buffer.clear()
                                
                            elif current_snapshot_time == self.last_inferred_time:
                                pass 
                                
                            else:
                                # 通過檢查：靜止 2 秒後抓取到的完美同步畫面！
                                self.last_inferred_time = current_snapshot_time 
                                self.is_inferencing = True
                                self.steps_executed_during_inference = 0
                                self.settle_start_time = None # 觸發後重置計時器
                                threading.Thread(target=self.inference_worker, args=(data_snapshot,), daemon=True).start()
            else:
                # 如果還有動作在執行，重置計時器
                self.settle_start_time = None

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
                
                final_action = self.smoothed_action
                self.last_gripper_pose = final_action[6:12] 

                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = [
                    'shoulder_1_joint', 'shoulder_2_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
                    'index_J1', 'index_J2', 'ring_J1', 'ring_J2', 'thumb_J1', 'thumb_J2'
                ]
                joint_msg.position = final_action[:12].tolist()
                self.pub_joint_cmd.publish(joint_msg)

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
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
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