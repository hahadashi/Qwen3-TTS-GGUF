import os
import sys
import numpy as np
import librosa
import onnxruntime as ort

# 配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model-base")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_encoder.onnx")
CAPTURED_PATH = os.path.join(PROJECT_ROOT, "captured_encoder_outputs", "tokenizer_audio_codes.npy")
REF_AUDIO = os.path.join(PROJECT_ROOT, "output", "sample.wav")

def main():
    if not os.path.exists(ONNX_PATH):
        print(f"❌ 找不到 ONNX 模型: {ONNX_PATH}")
        return
    if not os.path.exists(CAPTURED_PATH):
        print(f"❌ 找不到捕获的数据: {CAPTURED_PATH}")
        return

    # 1. 加载官方捕获的 Codec IDs
    official_codes = np.load(CAPTURED_PATH)
    print(f"载入官方捕获 ID: Shape {official_codes.shape}")

    # 2. 加载 16 层 Embedding 表 (零件)
    print("载入 16 层 Codec Embedding 表...")
    codec_tables = []
    for i in range(16):
        table_path = os.path.join(MODEL_DIR, f"codec_embedding_{i}_raw.npy")
        codec_tables.append(np.load(table_path))

    # 3. 运行 ONNX 推理获取 ONNX IDs
    session = ort.InferenceSession(ONNX_PATH)
    wav, _ = librosa.load(REF_AUDIO, sr=24000)
    input_values = wav.reshape(1, -1).astype(np.float32)
    print("正在执行 ONNX 推理获取 ID...")
    outputs = session.run(['audio_codes'], {'input_values': input_values})
    onnx_codes = outputs[0][0] # [T, 16]

    # 4. 对齐长度
    min_len = min(official_codes.shape[0], onnx_codes.shape[0])
    off_slice = official_codes[:min_len, :]
    onx_slice = onnx_codes[:min_len, :]

    # 5. 计算 Embedding (查表并累加)
    def compute_summed_embedding(codes):
        # codes: [T, 16]
        T = codes.shape[0]
        summed = np.zeros((T, 2048), dtype=np.float32)
        for q in range(16):
            # 将第 q 层的 ID 查表并加到总和
            summed += codec_tables[q][codes[:, q]]
        return summed

    print("正在计算 Embedding 空间映射...")
    off_embeds = compute_summed_embedding(off_slice)
    onx_embeds = compute_summed_embedding(onx_slice)

    # 6. 对比分析
    matches = np.equal(off_slice, onx_slice)
    id_match_rate = np.mean(matches)
    
    print("\n[逐帧比对结果]:")
    print(f"{'Frame':<8} | {'Cos Sim':<12} | {'Max Abs Diff':<14} | {'Matches'}")
    print("-" * 55)
    
    cos_sims = []
    for t in range(min_len):
        v1, v2 = off_embeds[t], onx_embeds[t]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_sims.append(sim)
        
        diff_frame = np.abs(v1 - v2)
        match_count = np.sum(matches[t])
        print(f"{t:<8} | {sim:.8f} | {np.max(diff_frame):.8e} | {match_count:>2}/16")
    
    avg_cos = np.mean(cos_sims)
    max_mae = np.max(np.abs(off_embeds - onx_embeds))

    print("\n" + "="*40)
    print("Codec Encoder 深度比对 (ID 空间 vs Embedding 空间):")
    print(f"对齐帧数: {min_len}")
    print(f"ID 空间: 整体 Token 匹配率: {id_match_rate * 100:.2f}%")
    print(f"Embed 空间: 平均余弦相似度: {avg_cos:.10f}")
    print(f"Embed 空间: 最大绝对误差 (MAE): {max_mae:.8e}")
    
    print("\n分量化器 ID 匹配率检视:")
    for q in range(16):
        q_rate = np.mean(matches[:, q])
        print(f"  Quantizer {q:2d}: {q_rate * 100:6.2f}%")
    print("="*40)

    if avg_cos > 0.9999:
        print("\n✅ 验证成功！虽然有极微小 ID 翻转，但 Embedding 空间高度对齐。")
    elif id_match_rate < 0.99:
        print("\n⚠️ 注意：ID 匹配率受限可能导致 Prompt 验证失败。")
        print("建议在最终验证时使用官方原始 ID 排除干扰。")

if __name__ == "__main__":
    main()
