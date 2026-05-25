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
# 請填入您要測試的訓練資料夾
RUN_DIR = "data/outputs/2026.01.20/21.55_train_dexgraspvla_controller_grasp"
# RUN_DIR = "data/outputs/2026.01.20/18.58_train_dexgraspvla_controller_grasp"
# 請填入對應的 Checkpoint
CKPT_NAME = "epoch=0108-train_loss=0.002.ckpt"
# CKPT_NAME = "epoch=0067-train_loss=0.011.ckpt"

TARGET_CLASS_ID = 4
IMG_SIZE = (518, 518)

# 夾爪參數
POS_OPEN  = np.array([1.5, 0.2, 0.0, -0.6, 0.0, 0.5])
POS_CLOSE = np.array([1.5, 0.6, 0.0, -1.2, 0.1, 1.0])

# 平滑係數 (0.3 = 推薦值)
SMOOTHING_ALPHA = 1 
CONTROL_FREQ = 0.6
# ==============================================

# Topic 定義
TOPIC_CAM1_RGB = "/side_cam/rgb"
TOPIC_CAM1_SEG = "/side_cam/semantic_segmentation"
TOPIC_CAM2_RGB = "/wrist_cam_2/rgb"
TOPIC_CAM3_RGB = "/side_cam_2/rgb"
TOPIC_CAM3_SEG = "/side_cam_2/semantic_segmentation"
TOPIC_CAM4_RGB = "/global_cam/rgb"
TOPIC_JOINT_STATE = "/joint_states"
TOPIC_ARM_CMD = "/joint_commands"
TOPIC_GRIPPER_CMD = "/H100_right"

class IsaacEvaluatorClean(Node):
    def __init__(self, policy, target_id):
        super().__init__('dexgrasp_evaluator_clean')
        self.policy = policy
        self.target_id = target_id
        self.bridge = CvBridge()
        self.device = policy.device
        
        self.obs_buffer = {}
        self.lock = threading.Lock()
        self.last_gripper_val = 0.2 
        self.smoothed_action = None
        self.first_run = True 

        self.pub_arm = self.create_publisher(JointState, TOPIC_ARM_CMD, 10)
        self.pub_gripper = self.create_publisher(JointState, TOPIC_GRIPPER_CMD, 10)

        qos = rclpy.qos.qos_profile_sensor_data
        
        # 訂閱所有可能用到的相機
        self.sub_c1_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM1_RGB, qos_profile=qos)
        self.sub_c1_seg = message_filters.Subscriber(self, Image, TOPIC_CAM1_SEG, qos_profile=qos)
        self.sub_c2_rgb = message_filters.Subscriber(self, Image, TOPIC_CAM2_RGB, qos_profile=qos)
        # 為了相容多視角訓練，這裡預留介面，但如果模型只訓練了2視角，這裡的資料不會被使用
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
        self.get_logger().info(f"✅ 模型就緒 (Clean Mode) | Freq={CONTROL_FREQ}Hz")

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
        """處理純 RGB -> 歸一化到 0~1"""
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img).float() / 255.0

    def process_rgbm(self, rgb_msg, seg_msg):
        """處理 RGBM -> 歸一化到 0~1 (Mask 為 0/1)"""
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding='passthrough')
        
        if rgb.shape[:2] != IMG_SIZE: rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        if seg.shape[:2] != IMG_SIZE: seg = cv2.resize(seg, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        mask = np.zeros_like(seg, dtype=np.uint8)
        mask[seg == self.target_id] = 255 
        mask = mask[:, :, np.newaxis] 

        # 拼接成 4 Channel: [R, G, B, M]
        rgbm = np.concatenate([rgb, mask], axis=-1) 
        rgbm = np.moveaxis(rgbm, -1, 0) 
        
        # 轉 Tensor 並除以 255
        # RGB 變 0.0~1.0
        # Mask 變 0.0 或 1.0
        return torch.from_numpy(rgbm).float() / 255.0

    def get_state_vector(self, state_msg):
        arm_joints = np.array(state_msg.position[:6])
        gripper_vec = np.zeros(7, dtype=np.float32)
        gripper_vec[0] = self.last_gripper_val
        state_vec = np.concatenate([arm_joints, gripper_vec])
        return torch.from_numpy(state_vec).float()

    def control_loop(self):
        with self.lock:
            if not self.obs_buffer: return
            data = self.obs_buffer.copy()

        try:
            # 1. 準備基礎輸入 (2 Camera)
            # 如果您的模型是 2 Camera 訓練的，這樣就夠了
            # 如果是 4 Camera 訓練的，請取消下面註解
            
            t_rgbm = self.process_rgbm(data['c1_rgb'], data['c1_seg'])
            t_wrist = self.process_rgb(data['c2_rgb'])
            t_state = self.get_state_vector(data['state'])

            batch = {
                'obs': {
                    'rgbm': t_rgbm.unsqueeze(0).unsqueeze(0).to(self.device),          # (1, 1, 4, H, W)
                    'right_cam_img': t_wrist.unsqueeze(0).unsqueeze(0).to(self.device), # (1, 1, 3, H, W)
                    'right_state': t_state.unsqueeze(0).unsqueeze(0).to(self.device)    # (1, 1, 13)
                }
            }
            
            # --- 如果是多視角模型，請把這段加入 batch['obs'] ---
            t_rgbm_aux = self.process_rgbm(data['c3_rgb'], data['c3_seg'])
            t_global = self.process_rgb(data['c4_rgb'])
            batch['obs']['rgbm_aux'] = t_rgbm_aux.unsqueeze(0).unsqueeze(0).to(self.device)
            batch['obs']['aux_cam_img'] = t_global.unsqueeze(0).unsqueeze(0).to(self.device)
            # -----------------------------------------------

            # 2. 推論 (不需任何 Patch!)
            with torch.no_grad():
                output = self.policy.predict_action(batch['obs'])
                
                if isinstance(output, dict): action_seq = output['action']
                elif isinstance(output, torch.Tensor): action_seq = output
                else: raise TypeError(f"Unknown output type: {type(output)}")

                if self.first_run:
                    print(f"\n[DEBUG] Action Shape: {action_seq.shape}")
                    self.first_run = False

                if action_seq.ndim == 3: action_raw = action_seq[0, 0]
                elif action_seq.ndim == 2: action_raw = action_seq[0]
                else: raise ValueError("Bad shape")
                
                action_raw = action_raw.cpu().numpy()

            # 3. 平滑化
            if self.smoothed_action is None:
                self.smoothed_action = action_raw
            else:
                self.smoothed_action = SMOOTHING_ALPHA * action_raw + (1.0 - SMOOTHING_ALPHA) * self.smoothed_action
            
            final_action = self.smoothed_action

            # 4. 發布指令
            target_arm = final_action[:6]
            pred_gripper_val = final_action[6]
            self.last_gripper_val = pred_gripper_val

            arm_msg = JointState()
            arm_msg.header.stamp = self.get_clock().now().to_msg()
            arm_msg.name = ['shoulder_1_joint', 'shoulder_2_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            arm_msg.position = target_arm.tolist()
            self.pub_arm.publish(arm_msg)

            val_clamped = np.clip(pred_gripper_val, 0.2, 0.6)
            alpha = (val_clamped - 0.2) / (0.6 - 0.2)
            current_gripper_pose = (1 - alpha) * POS_OPEN + alpha * POS_CLOSE
            
            grip_msg = JointState()
            grip_msg.header.stamp = self.get_clock().now().to_msg()
            grip_msg.name = ["Revolute_1", "Revolute_4", "Revolute_2", 
                             "Revolute_7", "Revolute_3", "Revolute_10"]
            grip_msg.position = current_gripper_pose.tolist()
            self.pub_gripper.publish(grip_msg)

            print(f"Pred: Arm[0]={target_arm[0]:.2f} | GripVal={pred_gripper_val:.2f}", end='\r')

        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}")

def load_policy(run_dir, ckpt_name):
    # 這裡是最單純的載入，不需要 force_patch
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