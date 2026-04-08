import sys
import os
import threading
import numpy as np
import cv2
import torch
import hydra
import dill
from omegaconf import OmegaConf
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# 確保能找到 controller 模組
sys.path.append(os.getcwd())

# ================= 使用者設定區 =================
# RUN_DIR = "data/outputs/2026.02.04/08.06_train_dexgraspvla_controller_grasp"
RUN_DIR = "data/outputs/2026.03.14/03.05_train_dexgraspvla_controller_grasp"
CKPT_NAME = "epoch=0118-train_loss=0.0012.ckpt"

TARGET_CLASS_ID = 4
IMG_SIZE = (518, 518)

# 夾爪參數
POS_OPEN  = np.array([1.5, 0.2, 0.0, -0.6, 0.0, 0.5])
POS_CLOSE = np.array([1.5, 0.6, 0.0, -1.2, 0.1, 1.0])

# 平滑係數 (0.3 = 推薦值)
SMOOTHING_ALPHA = 1.0
CONTROL_FREQ = 20
# ==============================================

# Topic 定義
TOPIC_CAM1_RGB = "/side_cam/rgb"
TOPIC_CAM1_SEG = "/side_cam/semantic_segmentation"
TOPIC_CAM2_RGB = "/wrist_cam_2/rgb"
TOPIC_CAM3_RGB = "/side_cam_2/rgb"
TOPIC_CAM3_SEG = "/side_cam_2/semantic_segmentation"
TOPIC_CAM4_RGB = "/global_cam/rgb"
TOPIC_JOINT_STATE = "/joint_states"
TOPIC_ARM_CMD = "/joint_command"
TOPIC_GRIPPER_CMD = "/H100_right"

class IsaacEvaluatorClean(Node):
    def __init__(self, policy, target_id):
        super().__init__('dexgrasp_evaluator_clean')
        self.policy = policy
        self.target_id = target_id
        self.bridge = CvBridge()
        self.device = policy.device

        self.action_buffer = []  
        self.chunk_size = 20    
        
        # --- 新增：非同步推論控制變數 ---
        self.is_inferencing = False  # 標記是否正在背景思考
        self.trigger_threshold = 10  # 當 buffer 剩下幾步時，提前啟動推論 (10步 = 500ms 緩衝)
        self.steps_executed_during_inference = 0 # 記錄推論期間，機器人走了幾步 (用於時間校正)
        
        self.obs_buffer = {}
        self.last_gripper_pose = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        self.lock = threading.Lock()
        self.smoothed_action = None

        self.pub_arm = self.create_publisher(JointState, TOPIC_ARM_CMD, 10)
        self.pub_gripper = self.create_publisher(JointState, TOPIC_GRIPPER_CMD, 10)

        qos = rclpy.qos.qos_profile_sensor_data
        
        self.sub_c1_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM1_RGB, qos_profile=qos)
        self.sub_c1_seg = message_filters.Subscriber(self, Image, TOPIC_CAM1_SEG, qos_profile=qos)
        self.sub_c2_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM2_RGB, qos_profile=qos)
        self.sub_c3_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM3_RGB, qos_profile=qos)
        self.sub_c3_seg = message_filters.Subscriber(self, Image, TOPIC_CAM3_SEG, qos_profile=qos)
        self.sub_c4_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM4_RGB, qos_profile=qos)
        self.sub_state = message_filters.Subscriber(self, JointState, TOPIC_JOINT_STATE, qos_profile=qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_c1_rgb, self.sub_c1_seg, self.sub_c2_rgb, 
             self.sub_c3_rgb, self.sub_c3_seg, self.sub_c4_rgb, self.sub_state],
            queue_size=15, slop=0.15
        )
        self.ts.registerCallback(self.sync_callback)

        period = 1.0 / CONTROL_FREQ
        self.timer = self.create_timer(period, self.control_loop)
        self.get_logger().info(f"✅ 模型就緒 (非同步無縫控制版) | Freq={CONTROL_FREQ}Hz")

    def sync_callback(self, c1_rgb, c1_seg, c2_rgb, c3_rgb, c3_seg, c4_rgb, state):
        with self.lock:
            self.obs_buffer['c1_rgb'] = c1_rgb
            self.obs_buffer['c1_seg'] = c1_seg
            self.obs_buffer['c2_rgb'] = c2_rgb
            self.obs_buffer['c3_rgb'] = c3_rgb
            self.obs_buffer['c3_seg'] = c3_seg
            self.obs_buffer['c4_rgb'] = c4_rgb
            self.obs_buffer['state'] = state

    def process_rgb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img).float() / 255.0

    def process_rgbm(self, rgb_msg, seg_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding='passthrough')
        
        if rgb.shape[:2] != IMG_SIZE: rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        if seg.shape[:2] != IMG_SIZE: seg = cv2.resize(seg, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        mask = np.zeros_like(seg, dtype=np.uint8)
        mask[seg == self.target_id] = 255 
        mask = mask[:, :, np.newaxis] 

        rgbm = np.concatenate([rgb, mask], axis=-1) 
        rgbm = np.moveaxis(rgbm, -1, 0) 
        return torch.from_numpy(rgbm).float() / 255.0

    def get_state_vector(self, state_msg):
        arm_joints = np.array(state_msg.position[:6]) 
        gripper_vec = np.zeros(7, dtype=np.float32)
        gripper_vec[:6] = self.last_gripper_pose      
        gripper_vec[6] = 0.0                          
        state_vec = np.concatenate([arm_joints, gripper_vec])
        return torch.from_numpy(state_vec).float()

    def inference_worker(self, data_snapshot):
        """獨立的背景推論執行緒：專門負責與 GPU 溝通"""
        try:
            # 1. 準備輸入 (使用擷取當下的畫面)
            t_rgbm = self.process_rgbm(data_snapshot['c1_rgb'], data_snapshot['c1_seg'])
            t_wrist = self.process_rgb(data_snapshot['c2_rgb'])
            t_state = self.get_state_vector(data_snapshot['state'])

            batch = {'obs': {
                'rgbm': t_rgbm.unsqueeze(0).unsqueeze(0).to(self.device),
                'right_cam_img': t_wrist.unsqueeze(0).unsqueeze(0).to(self.device),
                'right_state': t_state.unsqueeze(0).unsqueeze(0).to(self.device),
                'rgbm_aux': self.process_rgbm(data_snapshot['c3_rgb'], data_snapshot['c3_seg']).unsqueeze(0).unsqueeze(0).to(self.device),
                'aux_cam_img': self.process_rgb(data_snapshot['c4_rgb']).unsqueeze(0).unsqueeze(0).to(self.device)
            }}

            # 2. 推論 (保留 Tensor Cores 加速)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.policy.predict_action(batch['obs'])
                action_seq = output['action'] if isinstance(output, dict) else output
                chunk = action_seq[0, :self.chunk_size].cpu().numpy()

            # 3. 寫入 Buffer 與時間校正 (Time-Shift)
            with self.lock:
                # 計算：在 GPU 思考的這段時間，機器人已經偷跑了幾步？
                skip_steps = self.steps_executed_during_inference
                
                # 如果還沒超過 Chunk 長度，就把過去的動作剪掉，完美對齊當下時空
                if skip_steps < self.chunk_size:
                    self.action_buffer = list(chunk[skip_steps:])
                else:
                    self.action_buffer = list(chunk[-1:]) # 防呆：如果想太久，就直接接最後一步
                    
        except Exception as e:
            self.get_logger().error(f"Background Inference Fail: {e}")
        finally:
            with self.lock:
                self.is_inferencing = False # 釋放標記，允許下次推論

    def control_loop(self):
        """20Hz 的控制主迴圈：永遠不會被推論卡住"""
        action_to_execute = None

        with self.lock:
            if not self.obs_buffer: return
            
            # --- 邏輯判斷：是否需要提前啟動推論？ ---
            # 如果 Buffer 快見底了，且目前沒有人在推論，就叫背景小精靈開始算
            if len(self.action_buffer) <= self.trigger_threshold and not self.is_inferencing:
                self.is_inferencing = True
                self.steps_executed_during_inference = 0
                data_snapshot = self.obs_buffer.copy()
                
                # 啟動獨立執行緒
                threading.Thread(target=self.inference_worker, args=(data_snapshot,), daemon=True).start()

            # --- 執行階段：從 buffer 取出目前要執行的一步 ---
            if len(self.action_buffer) > 0:
                action_to_execute = self.action_buffer.pop(0)
                # 如果背景正在推論，記錄我們又往前走了一步
                if self.is_inferencing:
                    self.steps_executed_during_inference += 1

        # --- 實際發送控制命令 (不佔用 Lock 的時間) ---
        if action_to_execute is not None:
            try:
                if self.smoothed_action is None:
                    self.smoothed_action = action_to_execute
                else:
                    self.smoothed_action = SMOOTHING_ALPHA * action_to_execute + (1.0 - SMOOTHING_ALPHA) * self.smoothed_action
                
                final_action = self.smoothed_action

                target_arm = final_action[:6]
                target_gripper = final_action[6:12]
                self.last_gripper_pose = target_gripper 

                arm_msg = JointState()
                arm_msg.header.stamp = self.get_clock().now().to_msg()
                arm_msg.name = ['shoulder_1_joint', 'shoulder_2_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
                arm_msg.position = target_arm.tolist()
                self.pub_arm.publish(arm_msg)

                grip_msg = JointState()
                grip_msg.header.stamp = self.get_clock().now().to_msg()
                grip_msg.name = ["Revolute_1", "Revolute_4", "Revolute_2", 
                                 "Revolute_7", "Revolute_3", "Revolute_10"]
                grip_msg.position = target_gripper.tolist()
                self.pub_gripper.publish(grip_msg)

                # 簡單的進度顯示
                status = "🤔 思考中..." if self.is_inferencing else "✅ 執行中"
                print(f"[{status}] Buffer 剩餘: {len(self.action_buffer):02d} | Arm[0]: {target_arm[0]:.2f}", end='\r')

            except Exception as e:
                self.get_logger().error(f"Execution Fail: {e}")

def load_policy(run_dir, ckpt_name):
    cfg_path = os.path.join(run_dir, ".hydra")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.abspath(cfg_path), version_base=None)
    cfg = hydra.compose(config_name="config")
    cfg.task.dataset = None 
    
    from controller.workspace.train_dexgraspvla_controller_workspace import TrainDexGraspVLAControllerWorkspace
    workspace = TrainDexGraspVLAControllerWorkspace(cfg)
    
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    print(f"Loading checkpoint: {ckpt_path}")
    
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    if cfg.training.use_ema:
        print("Using EMA model.")
        policy = workspace.ema_model
    else:
        policy = workspace.model
    
    policy.eval()
    policy.cuda()
    return policy

def main(args=None):
    rclpy.init(args=args)
    try:
        print(f"Loading Policy from: {RUN_DIR}")
        policy = load_policy(RUN_DIR, CKPT_NAME)
        node = IsaacEvaluatorClean(policy, target_id=TARGET_CLASS_ID)
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}"); import traceback; traceback.print_exc()
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()