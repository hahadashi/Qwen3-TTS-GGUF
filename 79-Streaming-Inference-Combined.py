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
logger = logging.getLogger("Streaming-79")

def main():
    # 1. 路径和配置
    EXPORT_DIR = "./model-base"
    model_path = os.path.join(EXPORT_DIR, "qwen3_tts_decoder.onnx")
    json_path = r"D:\qwen3-tts\output\sample.json"
    output_wav = "output_streaming_25.wav"
    CHUNK_SIZE = 2

    if not os.path.exists(model_path):
        logger.error(f"❌ 找不到模型: {model_path}")
        return
    if not os.path.exists(json_path):
        logger.error(f"❌ 找不到 JSON 文件: {json_path}")
        return

    # 2. 载入模型 (使用 DirectML)
    logger.info("📦 正在使用 DirectML 载入联合版 ONNX 模型...")
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        logger.info("✅ 模型载入成功。")
    except Exception as e:
        logger.error(f"❌ 模型载入失败: {e}")
        return

    # 3. 读取输入数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_codes = np.array(data["codes"], dtype=np.int64)
    all_codes = np.concatenate([all_codes, all_codes, all_codes], axis=0) # 扩充为 3 倍长度进行长序列测试
    total_frames = all_codes.shape[0]
    logger.info(f"📄 加载 codes 成功: {all_codes.shape}")

    # 4. 初始化状态 (0 记忆长度冷启动)
    B = 1
    pre_conv_history = np.zeros((B, 512, 0), dtype=np.float32)
    latent_buffer = np.zeros((B, 1024, 0), dtype=np.float32)
    conv_history = np.zeros((B, 1024, 0), dtype=np.float32)
    
    num_layers = 8
    num_heads = 16
    head_dim = 64
    past_kvs = {}
    for i in range(num_layers):
        past_kvs[f"past_key_{i}"] = np.zeros((B, num_heads, 0, head_dim), dtype=np.float32)
        past_kvs[f"past_value_{i}"] = np.zeros((B, num_heads, 0, head_dim), dtype=np.float32)

    # 5. 流式推理循环
    all_audio_chunks = []
    start_time_total = time.perf_counter()

    for i in range(50):
    
        for start_idx in range(0, total_frames, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_frames)
            is_last_chunk = (end_idx == total_frames)
            
            chunk_codes = all_codes[start_idx:end_idx]
            # Reshape to [1, N, 16]
            audio_codes_input = chunk_codes[np.newaxis, ...]
            
            logger.info(f"👉 处理块: {start_idx} 到 {end_idx} (N={end_idx-start_idx}, last={is_last_chunk})")
            
            # 构建 feed_dict
            feed = {
                "audio_codes": audio_codes_input,
                "pre_conv_history": pre_conv_history,
                "latent_buffer": latent_buffer,
                "conv_history": conv_history,
                "is_last": np.array([1.0 if is_last_chunk else 0.0], dtype=np.float32),
            }
            feed.update(past_kvs)
            
            # 推理
            outputs = session.run(None, feed)
            
            # 解包输出 (标准化索引：0:波形, 1:有效长度, 2:PreConvh, 3:Latent, 4:Convh, 5+:KV)
            final_wav_raw = outputs[0]
            valid_samples_val = int(outputs[1][0])
            
            # 使用 valid_samples 截取音频
            chunk_audio = final_wav_raw[0, :valid_samples_val]
            all_audio_chunks.append(chunk_audio)
            
            logger.info(f"   <- 产出音频长度: {len(chunk_audio)} samples (Raw: {final_wav_raw.shape[1]})")
            
            # 更新状态以便下一轮迭代
            pre_conv_history = outputs[2]
            latent_buffer = outputs[3]
            conv_history = outputs[4]
            
            # 更新 KV Cache (从索引 5 开始)
            next_kv_start_idx = 5
            for i in range(num_layers):
                past_kvs[f"past_key_{i}"] = outputs[next_kv_start_idx + i]
                past_kvs[f"past_value_{i}"] = outputs[next_kv_start_idx + num_layers + i]

    end_time_total = time.perf_counter()
    logger.info(f"✅ 流式推理完成！总耗时: {(end_time_total - start_time_total)*1000:.2f} ms")

    # 6. 合并并保存音频
    if all_audio_chunks:
        final_full_audio = np.concatenate(all_audio_chunks)
        sr = 24000
        sf.write(output_wav, final_full_audio, sr)
        logger.info(f"💾 音频已保存至: {output_wav}")
        logger.info(f"📊 总音频长度: {len(final_full_audio)} samples (约 {len(final_full_audio)/sr:.2f} 秒)")
    else:
        logger.warning("⚠️ 没有生成任何音频数据。")

if __name__ == "__main__":
    main()
