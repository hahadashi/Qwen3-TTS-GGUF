import numpy as np
import soundfile as sf
import os

def check_divergence():
    # 路径配置
    output_dir = "output_verify"
    ref_path = os.path.join(output_dir, "verify_streaming_wrapper.wav") # 全量参考
    test_path = os.path.join(output_dir, "verify_stateful_stream.wav") # 流式验证
    
    if not os.path.exists(ref_path) or not os.path.exists(test_path):
        print("❌ 找不到音频文件，请先运行 81-Verify-Stateful-PyTorch.py")
        return

    # 加载音频
    ref, sr = sf.read(ref_path)
    test, _ = sf.read(test_path)
    
    print(f"📊 参考长度: {len(ref)} 采样点")
    print(f"📊 测试长度: {len(test)} 采样点")
    
    # 1. 寻找第一个不一致的点
    min_len = min(len(ref), len(test))
    diff = np.abs(ref[:min_len] - test[:min_len])
    
    # 定义判定阈值（浮点数微小误差通常在 1e-6 左右）
    threshold = 1e-5
    mismatches = np.where(diff > threshold)[0]
    
    if len(mismatches) == 0:
        print("\n✅ 在共有长度范围内，两个音频完全一致！")
    else:
        first_idx = mismatches[0]
        print(f"\n⚠️ 首次出现显著差异的位置: {first_idx}")
        print(f"   - 时间点: {first_idx / sr:.4f} 秒")
        print(f"   - 差异幅度: {diff[first_idx]:.6f}")
        
    # 2. 采样率与帧的对应关系
    # Qwen3-TTS 12Hz, SR=24000 -> 1 帧 = 2000 采样点 (或 1920 采样点，取决于内部配置)
    UPSAMPLE_RATE = 1920 # 根据你 codes 的 config
    frame_idx = first_idx // UPSAMPLE_RATE
    print(f"   - 对应代码帧索引: 约第 {frame_idx} 帧")

    # 3. 统计尾部长度
    if len(test) > len(ref):
        extra_len = len(test) - len(ref)
        print(f"\n📝 测试音频比参考音频长出: {extra_len} 采样点")
        print(f"   - 长出的时长: {extra_len / sr:.4f} 秒")
        print(f"   - 对应帧数: 约 {extra_len / UPSAMPLE_RATE:.2f} 帧")

if __name__ == "__main__":
    check_divergence()
