import sys
import os
import time
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

# --- 壓力測試設定 ---
NUM_WARMUP = 10     # 暖身次數 (讓 GPU 風扇轉起來，進入 P0 效能狀態)
NUM_TESTS = 50      # 連續高壓測試次數 (測穩定值)
USE_AMP = False     # 【關鍵】是否開啟混合精度(FP16/BF16)加速？你可以改成 True 試試看極限！
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
    print(" 🛠️ 階段一：理論絕對指標 (不受硬體影響)")
    print("="*50)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"▶ 模型參數總量: {total_params / 1e6:.2f} M")

    # 快閃 Profiler 只為取得 FLOPs
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
            _ = policy.predict_action(batch['obs'])
    events = prof.key_averages()
    total_flops = sum([e.flops for e in events if e.flops > 0])
    print(f"▶ 單次推論運算量: {total_flops / 1e9:.2f} GFLOPs")


    print("\n" + "="*50)
    print(f" 🚀 階段二：本機真實極限壓測 (AMP 加速: {'開啟' if USE_AMP else '關閉'})")
    print("="*50)
    
    # 1. 暖身 (Warmup) - 非常重要，把 GPU 喚醒
    print(f"正在暖身 GPU ({NUM_WARMUP} steps)...")
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
        for _ in range(NUM_WARMUP):
            _ = policy.predict_action(batch['obs'])
            torch.cuda.synchronize(DEVICE)

    # 2. 正式壓力測試 (無 Profiler 拖累)
    print(f"開始連續 {NUM_TESTS} 次高壓推論測試...")
    times = []
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
        for i in range(NUM_TESTS):
            torch.cuda.synchronize(DEVICE) # 確保之前的都做完了
            t_start = time.perf_counter()
            
            output = policy.predict_action(batch['obs'])
            
            torch.cuda.synchronize(DEVICE) # 確保這次的推論徹底算完
            t_end = time.perf_counter()
            
            times.append((t_end - t_start) * 1000)
            print(f"  Step {i+1:02d}: {times[-1]:.2f} ms", end='\r')
    print("\n")

    # 3. 結算真實數據
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    p99_ms = np.percentile(times, 99) # 99% 的推論都在多少毫秒內完成 (代表穩定度)
    max_hz = 1000.0 / avg_ms
    vram_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    print("📊【本機真實硬體效能報告】")
    print(f"▶ 平均推論時間: {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"▶ 最差延遲 (99th Pct): {p99_ms:.2f} ms (這決定了你實機上最卡的那一瞬間)")
    print(f"▶ 本機極限控制頻率: {max_hz:.2f} Hz")
    print(f"▶ 執行期最高 VRAM: {vram_peak:.2f} MB")
    
    print("\n💡 結論分析：")
    if max_hz >= 5.0:
        print(f"✅ 你的 RTX 5070 Ti 能夠穩定支撐 5Hz (目標) 的控制頻率！")
    else:
        print(f"❌ 你的 RTX 5070 Ti 目前極限只有 {max_hz:.2f} Hz，未達 5Hz 目標。")
        if not USE_AMP:
            print("👉 強烈建議：將腳本上方的 USE_AMP 改為 True，啟動 Tensor Cores 再測一次！")

if __name__ == '__main__':
    main()