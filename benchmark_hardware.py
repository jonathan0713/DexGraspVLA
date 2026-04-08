import sys
import os
import torch
import numpy as np
import hydra
import dill

sys.path.append(os.getcwd())

# ================= 使用者設定區 =================
RUN_DIR = "data/outputs/2026.02.04/08.06_train_dexgraspvla_controller_grasp"
CKPT_NAME = "epoch=0122-train_loss=0.0010.ckpt"
IMG_SIZE = (518, 518)
DEVICE = "cuda"
# ==============================================

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
    policy.to(DEVICE)
    return policy

def create_dummy_batch():
    dummy_rgbm = torch.rand((1, 1, 4, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32).to(DEVICE)
    dummy_rgb = torch.rand((1, 1, 3, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32).to(DEVICE)
    dummy_state = torch.rand((1, 1, 13), dtype=torch.float32).to(DEVICE)

    return {'obs': {
        'rgbm': dummy_rgbm,
        'right_cam_img': dummy_rgb,
        'right_state': dummy_state,
        'rgbm_aux': dummy_rgbm.clone(),
        'aux_cam_img': dummy_rgb.clone()
    }}

def main():
    policy = load_policy(RUN_DIR, CKPT_NAME)
    batch = create_dummy_batch()

    print("\n" + "="*50)
    print(" 🛠️ 階段一：靜態絕對指標計算")
    print("="*50)
    
    # 1. 計算模型總參數 (靜態 VRAM 門檻)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"▶ 模型參數總量 (Parameters): {total_params / 1e6:.2f} M (百萬)")

    # 2. 計算單次輸入資料大小 (PCIe 頻寬門檻)
    total_bytes = 0
    for k, v in batch['obs'].items():
        total_bytes += v.element_size() * v.nelement()
    total_mb = total_bytes / (1024 * 1024)
    print(f"▶ 單次推論 Host-to-Device 傳輸量: {total_mb:.2f} MB")
    print(f"  (若要求 5Hz 控制頻率，硬體 PCIe 頻寬至少需負荷 {total_mb * 5:.2f} MB/s 的連續傳輸)")

    print("\n" + "="*50)
    print(" 🚀 階段二：動態 FLOPs 算力剖析 (請稍候...)")
    print("="*50)

    # 暖身
    with torch.no_grad():
        for _ in range(3):
            _ = policy.predict_action(batch['obs'])
    
    # 3. 使用 PyTorch Profiler 計算真實 FLOPs
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True # 關鍵：啟動 FLOPs 計算
        ) as prof:
            _ = policy.predict_action(batch['obs'])

    # 結算 FLOPs
    # 注意：這裡取得的是總 FLOPs
    events = prof.key_averages()
    total_flops = sum([e.flops for e in events if e.flops > 0])
    
    gflops = total_flops / 1e9
    
    print(f"▶ 單次推論運算量 (FLOPs): {gflops:.2f} GFLOPs")
    print(f"  (若要求 5Hz 控制頻率，硬體實時算力至少需 {gflops * 5:.2f} GFLOPs)")
    
    # 動態 VRAM 峰值 (包含 Tensor 暫存)
    vram_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    print(f"▶ 執行期最高 VRAM 佔用: {vram_peak:.2f} MB")
    
    print("\n💡【硬體適配公式】")
    print(f"當你挑選實機 GPU (如 Jetson 或獨立顯卡) 時，只要確認它的規格表：")
    print(f"1. FP32/FP16 算力 (TFLOPS) 大於 {(gflops * 5) / 1000:.4f} TFLOPS (建議抓 3~5 倍餘裕以防過熱降頻)")
    print(f"2. 實體 VRAM 容量大於 {vram_peak / 1024:.2f} GB")
    print("即可保證能流暢執行此模型！")

if __name__ == '__main__':
    main()