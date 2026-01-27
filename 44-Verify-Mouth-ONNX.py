import os
import numpy as np
import onnxruntime as ort
import soundfile as sf

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_mouth")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")

def run_verification():
    print(f"Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    # 1. 加载捕获的数据
    input_codes = np.load(os.path.join(CAPTURED_DIR, "input_codes.npy")) # [B, T, Q]
    official_waveform = np.load(os.path.join(CAPTURED_DIR, "output_waveform.npy")) # [B, 1, T_wav]
    
    print(f"Input codes shape: {input_codes.shape}")
    print(f"Official waveform shape: {official_waveform.shape}")
    
    # 2. 运行 ONNX
    # 注意：导出脚本定义的输入名是 'audio_codes'
    onnx_out = sess.run(None, {'audio_codes': input_codes})[0] # [B, T_wav]
    
    # 3. 对齐处理
    # 官方输出是 [B, 1, T]，ONNX 输出是 [B, T]
    official_waveform = official_waveform.squeeze(1)
    
    # 长度对齐 (如果有微小差异)
    min_len = min(official_waveform.shape[1], onnx_out.shape[1])
    official_waveform = official_waveform[:, :min_len]
    onnx_out = onnx_out[:, :min_len]
    
    # 4. 数值比较
    official_flat = official_waveform.flatten()
    onnx_flat = onnx_out.flatten()
    
    mae = np.mean(np.abs(official_flat - onnx_flat))
    cos_sim = np.dot(official_flat, onnx_flat) / (np.linalg.norm(official_flat) * np.linalg.norm(onnx_flat) + 1e-9)
    
    print("\n--- 嘴巴 (Mouth/Codec Decoder) 对齐结果 ---")
    pass_mark = "✅" if mae < 1e-4 and cos_sim > 0.9999 else "⚠️"
    print(f"{pass_mark} MAE: {mae:.6f}")
    print(f"{pass_mark} Cosine Similarity: {cos_sim:.6f}")
    
    # 5. 保存 ONNX 输出 wav 供对比
    onnx_wav_path = os.path.join(CAPTURED_DIR, "onnx_out.wav")
    sf.write(onnx_wav_path, onnx_flat, 24000)
    print(f"\nONNX 输出音频已保存至: {onnx_wav_path}")
    print(f"官方 输出音频已保存至: {os.path.join(CAPTURED_DIR, 'official_out.wav')}")
    
    if cos_sim > 0.9999:
        print("\n结论: 嘴巴组件保真度极高，验证通过。")
    else:
        print("\n结论: 存在数值偏差，请检查 opset 或数据类型转换。")

if __name__ == "__main__":
    run_verification()
