import os
import json
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Inference-JSON-78")

def main():
    # 1. 路径和配置
    EXPORT_DIR = "./model-custom"
    # 使用合并版模型 (模型 64)
    model_path = os.path.join(EXPORT_DIR, "qwen3_tts_decoder.onnx")
    json_path = r"D:\qwen3-tts\output\sample.json"
    output_wav = "output_combined_json.wav"

    if not os.path.exists(model_path):
        logger.error(f"❌ 找不到模型: {model_path}")
        return
    if not os.path.exists(json_path):
        logger.error(f"❌ 找不到 JSON 文件: {json_path}")
        return

    # 2. 载入模型 (使用 DirectML)
    logger.info("📦 正在使用 DirectML 载入联合版 ONNX 模型...")
    # NOTE: 这里你可以根据需要切换为 CPUExecutionProvider 观察数值
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        logger.info("✅ 模型载入成功。")
    except Exception as e:
        logger.error(f"❌ 模型载入失败: {e}")
        return

    # 3. 读取并准备输入数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 假设 codes 形状是 [N, 16]
    codes = np.array(data["codes"], dtype=np.int64)
    N = codes.shape[0]
    B = 1
    Q = 16
    logger.info(f"📄 加载 codes 成功: {codes.shape}")

    # 输入 reshape 为 [1, N, 16]
    audio_codes = codes[np.newaxis, ...]

    # 4. 初始化“0记忆长度”的状态 (按用户要求)
    # 卷积状态: 长度使用 0
    pre_conv_history = np.zeros((B, 512, 0), dtype=np.float32)
    latent_buffer = np.zeros((B, 1024, 0), dtype=np.float32)
    conv_history = np.zeros((B, 1024, 0), dtype=np.float32)
    
    # KV Cache: 长度使用 0
    num_layers = 8
    num_heads = 16
    head_dim = 64
    past_kvs = {}
    for i in range(num_layers):
        past_kvs[f"past_key_{i}"] = np.zeros((B, num_heads, 0, head_dim), dtype=np.float32)
        past_kvs[f"past_value_{i}"] = np.zeros((B, num_heads, 0, head_dim), dtype=np.float32)

    is_last = np.array([1.0], dtype=np.float32) # 一次性跑完

    # 构建完整 feed_dict
    feed = {
        "audio_codes": audio_codes,
        "pre_conv_history": pre_conv_history,
        "latent_buffer": latent_buffer,
        "conv_history": conv_history,
        "is_last": is_last,
    }
    feed.update(past_kvs)

    # 5. 执行推理
    logger.info(f"🚀 开始执行推理 (N={N} 帧)...")
    try:
        start_time = time.perf_counter()
        outputs = session.run(None, feed)
        end_time = time.perf_counter()
        
        logger.info(f"✅ 推理完成，耗时: {(end_time - start_time)*1000:.2f} ms")
        
        # 6. 后处理波形
        # Index 0: final_wav, Index 1: valid_samples
        raw_wav = outputs[0][0]  # [num_samples]
        valid_len = int(outputs[1][0])
        
        # 根据 valid_samples 截取有效部分
        final_wav = raw_wav[:valid_len]
        
        # 7. 保存 WAV
        sr = 24000 # 假设采样率为 24k
        sf.write(output_wav, final_wav, sr)
        logger.info(f"💾 音频已保存至: {output_wav}")
        logger.info(f"📊 音频长度: {len(final_wav)} samples (Raw: {len(raw_wav)})")
        
    except Exception as e:
        logger.error(f"❌ 推理过程中崩溃: {e}")
        # 这里如果崩溃，大概率是因为模型内部 narrow 在 past_len=0 时不工作
        logger.info("💡 提示: 如果报错 'Invalid size or shape', 说明模型 64 内部 narrow 逻辑强制需要非 0 的初始历史内存。")

if __name__ == "__main__":
    main()
