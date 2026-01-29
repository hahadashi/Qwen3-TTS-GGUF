import os
import sys
import torch
import numpy as np
import librosa
import onnxruntime as ort

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Qwen3-TTS"))

from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

# 配置
ONNX_PATH = os.path.join(PROJECT_ROOT, "model-base", "qwen3_tts_speaker_encoder.onnx")
CAPTURED_PATH = os.path.join(PROJECT_ROOT, "captured_encoder_outputs", "speaker_embedding.npy")
REF_AUDIO = os.path.join(PROJECT_ROOT, "output", "sample.wav")

def main():
    if not os.path.exists(ONNX_PATH):
        print(f"❌ 找不到 ONNX 模型: {ONNX_PATH}")
        return
    if not os.path.exists(CAPTURED_PATH):
        print(f"❌ 找不到捕获的数据: {CAPTURED_PATH}")
        return

    # 1. 加载官方捕获的数据 [1, 2048]
    official_emb = np.load(CAPTURED_PATH).flatten()
    print(f"载入官方捕获数据: Shape {official_emb.shape}")

    # 2. 准备 ONNX 输入 (Mel 谱图)
    print(f"正在处理参考音频并提取 Mel 谱图: {REF_AUDIO}")
    wav, _ = librosa.load(REF_AUDIO, sr=24000)
    
    with torch.no_grad():
        mels = mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        mels_np = mels.numpy().astype(np.float32)
    
    # 3. 运行 ONNX 推理
    print(f"载入 ONNX 模型: {ONNX_PATH}")
    session = ort.InferenceSession(ONNX_PATH)
    
    print("正在执行 ONNX 推理...")
    outputs = session.run(['spk_emb'], {'mels': mels_np})
    onnx_emb = outputs[0][0] # 拿走 Batch 维 -> [2048]
    
    print(f"ONNX 输出维度: {onnx_emb.shape}")

    # 4. 计算一致性
    dot_product = np.dot(official_emb, onnx_emb)
    cos_sim = dot_product / (np.linalg.norm(official_emb) * np.linalg.norm(onnx_emb))
    max_diff = np.max(np.abs(official_emb - onnx_emb))

    print("\n" + "="*40)
    print("Speaker Encoder (ONNX vs Official) 对比:")
    print(f"余弦相似度 (Cosine Similarity): {cos_sim:.8f}")
    print(f"最大绝对误差 (Max Abs Diff):    {max_diff:.8e}")
    print("="*40)

    if cos_sim > 0.9999:
        print("\n✅ 验证通过！Speaker Encoder ONNX 输出与官方高度一致。")
    else:
        print("\n❌ 验证失败！相似度过低。")

if __name__ == "__main__":
    main()
