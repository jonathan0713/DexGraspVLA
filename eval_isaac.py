import sys
import os
import threading
import numpy as np
import cv2
import torch
import hydra
import dill # pip install dill
from omegaconf import OmegaConf
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import torchvision.transforms as T
import torch.nn as nn

# 確保能找到 controller 模組
sys.path.append(os.getcwd())

# ================= 使用者設定區 =================
RUN_DIR = "data/outputs/2026.01.20/15.16_train_dexgraspvla_controller_grasp"
CKPT_NAME = "epoch=0055-train_loss=0.010.ckpt"
TARGET_CLASS_ID = 5 
IMG_SIZE = (518, 518)
POS_OPEN  = np.array([1.5, 0.2, 0.0, -0.6, 0.0, 0.5])
POS_CLOSE = np.array([1.5, 0.6, 0.0, -1.2, 0.1, 1.0])

# 【關鍵參數】平滑係數 (Smoothing Alpha)
# 範圍 0.0 ~ 1.0
# 1.0 = 完全不平滑 (原始輸出，反應最快但最抖)
# 0.1 = 非常平滑 (反應慢，像是在水裡移動)
# 建議值: 0.2 ~ 0.4
SMOOTHING_ALPHA = 0.3

CONTROL_FREQ = 3
# ==============================================

TOPIC_RGBM_RGB = "/side_cam/rgb"
TOPIC_RGBM_SEG = "/side_cam/semantic_segmentation"
TOPIC_WRIST_RGB = "/wrist_cam_2/rgb" 
TOPIC_JOINT_STATE = "/joint_states"
TOPIC_ARM_CMD = "/joint_commands"
TOPIC_GRIPPER_CMD = "/H100_right"

# --- 智能 Normalizer ---
class SmartNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean3', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std3', torch.tensor(std).view(1, 3, 1, 1))
        mean4 = list(mean) + [0.0]
        std4 = list(std) + [1.0]
        self.register_buffer('mean4', torch.tensor(mean4).view(1, 4, 1, 1))
        self.register_buffer('std4', torch.tensor(std4).view(1, 4, 1, 1))

    def forward(self, tensor):
        c = tensor.shape[1]
        if c == 3: return (tensor - self.mean3) / self.std3
        elif c == 4: return (tensor - self.mean4) / self.std4
        return tensor

class IsaacEvaluator(Node):
    def __init__(self, policy, target_id):
        super().__init__('dexgrasp_evaluator')
        self.policy = policy
        self.target_id = target_id
        self.bridge = CvBridge()
        self.device = policy.device
        
        self.obs_buffer = {}
        self.lock = threading.Lock()
        
        # 平滑化相關變數
        self.last_gripper_val = 0.2 
        self.smoothed_action = None # 用來儲存上一步的平滑結果
        self.first_run = True 

        self.pub_arm = self.create_publisher(JointState, TOPIC_ARM_CMD, 10)
        self.pub_gripper = self.create_publisher(JointState, TOPIC_GRIPPER_CMD, 10)

        qos = rclpy.qos.qos_profile_sensor_data
        self.sub_rgb = message_filters.Subscriber(self, Image, TOPIC_RGBM_RGB, qos_profile=qos)
        self.sub_seg = message_filters.Subscriber(self, Image, TOPIC_RGBM_SEG, qos_profile=qos)
        self.sub_wrist = message_filters.Subscriber(self, Image, TOPIC_WRIST_RGB, qos_profile=qos)
        self.sub_state = message_filters.Subscriber(self, JointState, TOPIC_JOINT_STATE, qos_profile=qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_seg, self.sub_wrist, self.sub_state],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # 【修改這裡】將 0.1 改為 1.0 / CONTROL_FREQ
        period = 1.0 / CONTROL_FREQ
        self.timer = self.create_timer(period, self.control_loop)
        
        self.get_logger().info(f"✅ 模型就緒！頻率: {CONTROL_FREQ}Hz | Alpha={SMOOTHING_ALPHA}...")

    def sync_callback(self, rgb_msg, seg_msg, wrist_msg, state_msg):
        with self.lock:
            self.obs_buffer['rgb'] = rgb_msg
            self.obs_buffer['seg'] = seg_msg
            self.obs_buffer['wrist'] = wrist_msg
            self.obs_buffer['state'] = state_msg

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

    def process_wrist(self, wrist_msg):
        img = self.bridge.imgmsg_to_cv2(wrist_msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != IMG_SIZE: img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img).float() / 255.0

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
            rgbm_tensor = self.process_rgbm(data['rgb'], data['seg'])
            wrist_tensor = self.process_wrist(data['wrist'])
            state_tensor = self.get_state_vector(data['state'])

            batch = {
                'obs': {
                    'rgbm': rgbm_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
                    'right_cam_img': wrist_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
                    'right_state': state_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                }
            }

            with torch.no_grad():
                output = self.policy.predict_action(batch['obs'])
                
                # Type Guard
                if isinstance(output, dict): action_seq = output['action']
                elif isinstance(output, torch.Tensor): action_seq = output
                else: raise TypeError(f"Unknown output type: {type(output)}")

                # Safe Parsing
                if action_seq.ndim == 3: action_raw = action_seq[0, 0]
                elif action_seq.ndim == 2: action_raw = action_seq[0]
                else: raise ValueError(f"Unexpected shape: {action_seq.shape}")

                action_raw = action_raw.cpu().numpy()

            # 【關鍵修改】平滑化邏輯 (EMA Filter)
            if self.smoothed_action is None:
                self.smoothed_action = action_raw
            else:
                # 公式: New = alpha * Raw + (1 - alpha) * Old
                self.smoothed_action = SMOOTHING_ALPHA * action_raw + (1.0 - SMOOTHING_ALPHA) * self.smoothed_action
            
            # 使用平滑後的動作
            final_action = self.smoothed_action

            target_arm = final_action[:6]
            pred_gripper_val = final_action[6]
            self.last_gripper_val = pred_gripper_val

            # Publish Arm
            arm_msg = JointState()
            arm_msg.header.stamp = self.get_clock().now().to_msg()
            arm_msg.name = ['shoulder_1_joint', 'shoulder_2_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            arm_msg.position = target_arm.tolist()
            self.pub_arm.publish(arm_msg)

            # Publish Gripper
            val_clamped = np.clip(pred_gripper_val, 0.2, 0.6)
            alpha = (val_clamped - 0.2) / (0.6 - 0.2)
            current_gripper_pose = (1 - alpha) * POS_OPEN + alpha * POS_CLOSE
            
            grip_msg = JointState()
            grip_msg.header.stamp = self.get_clock().now().to_msg()
            grip_msg.name = ["Revolute_1", "Revolute_4", "Revolute_2", 
                             "Revolute_7", "Revolute_3", "Revolute_10"]
            grip_msg.position = current_gripper_pose.tolist()
            self.pub_gripper.publish(grip_msg)

            print(f"Pred: Arm[0]={target_arm[0]:.2f} | GripVal={pred_gripper_val:.2f} ({'Close' if alpha>0.5 else 'Open'})", end='\r')

        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}")

# ... (保持 force_patch_obs_encoder 和 load_policy 與 V8 相同) ...
def force_patch_obs_encoder(policy):
    print(">>> 正在執行 V6 智能修補 (Smart Patching)...")
    try:
        if hasattr(policy, 'obs_encoder'): obs_encoder = policy.obs_encoder
        elif hasattr(policy, 'model') and hasattr(policy.model, 'obs_encoder'): obs_encoder = policy.model.obs_encoder
        else: return
        if not hasattr(obs_encoder, 'dino_transform'): return
        transform = obs_encoder.dino_transform
        def replace_normalize(t_list):
            for i, t in enumerate(t_list):
                if isinstance(t, T.Normalize):
                    print(f"   -> 替換 Normalize 為 SmartNormalize (Mean: {t.mean})")
                    t_list[i] = SmartNormalize(t.mean, t.std).to(policy.device)
                elif isinstance(t, T.Compose): replace_normalize(t.transforms)
        if isinstance(transform, T.Compose): replace_normalize(transform.transforms)
        elif isinstance(transform, T.Normalize): obs_encoder.dino_transform = SmartNormalize(transform.mean, transform.std).to(policy.device)
    except Exception as e: print(f"❌ 修補錯誤: {e}")

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
    else: policy = workspace.model
    policy.eval(); policy.cuda()
    force_patch_obs_encoder(policy)
    return policy

def main(args=None):
    rclpy.init(args=args)
    try:
        print(f"Loading Policy from: {RUN_DIR}")
        policy = load_policy(RUN_DIR, CKPT_NAME)
        node = IsaacEvaluator(policy, target_id=TARGET_CLASS_ID)
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}"); import traceback; traceback.print_exc()
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()