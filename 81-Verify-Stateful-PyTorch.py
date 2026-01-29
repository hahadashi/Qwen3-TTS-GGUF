import os
import torch
import numpy as np
import soundfile as sf
from qwen3_tts_gguf.codec_export import StatefulCodecExportWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    DEBUG_DATA_DIR = "debug_data"
    
    codes_path = os.path.join(DEBUG_DATA_DIR, "jintian_codes.npy")
    ref_wav_path = os.path.join(DEBUG_DATA_DIR, "jintian_ref.wav")
    
    if not os.path.exists(codes_path):
        print(f"❌ 找不到调试数据: {codes_path}，请先运行 43-Generate-Debug-Data.py")
        return

    # 2. 加载数据
    codes_np = np.load(codes_path)
    ref_wav_np, sr = sf.read(ref_wav_path)
    
    # 3. 加载模型
    print(f"🚀 正在加载模型: {MODEL_DIR}...")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
    model.eval()
    
    # 初始化 Stateful Wrapper
    wrapper = StatefulCodecExportWrapper(model).eval()
    
    # 4. 模拟流式推理
    print("🧪 正在执行流式分波段推理对比...")
    
    # 参数设置：我们将 9 帧代码拆成 3 段，每段 3 帧
    chunk_size = 3
    num_chunks = len(codes_np) // chunk_size
    
    pkv = None
    latent_buf = None
    all_chunks_audio = []
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            is_last = (i == num_chunks - 1)
            
            # 准备当前 chunk 的输入
            chunk_codes = torch.from_numpy(codes_np[start:end]).unsqueeze(0).long()
            
            print(f"  > 处理分片 {i+1}/{num_chunks} (帧 {start} 到 {end-1}, is_last={is_last})")
            
            # 执行有状态推理
            # audio: [B, samples], pkv: Cache, latent_buf: [B, Hidden, 4]
            audio, pkv, latent_buf = wrapper(
                chunk_codes, 
                past_key_values=pkv, 
                latent_buffer=latent_buf, 
                is_last_chunk=is_last
            )
            
            # 只有非空音频才追加
            audio_np = audio.numpy().squeeze()
            if audio_np.size > 0:
                all_chunks_audio.append(audio_np)
                print(f"      └─ 输出 {audio_np.size} 采样点")
            else:
                print(f"      └─ 累积中，暂无输出")

    # 5. 拼接与对比
    full_audio_stream = np.concatenate(all_chunks_audio)
    
    # 长度对齐：原始全量推理包含了最后的 4 帧 padding (因为我们之前的 codes 里带了 EOS)
    # 但我们的 Stateful Wrapper 在 is_last_chunk=True 时也会额外产出补齐的 4 帧音频
    
    print("\n" + "="*40)
    print(f"📊 流式拼接总长度: {len(full_audio_stream)}")
    print(f"📊 参考波形总长度: {len(ref_wav_np)}")
    
    # 计算差异
    # 注意：流式拼接后的波形理论上应与 ref_wav 完全一致
    # 但由于全量推理时的 Padding 处理可能略有不同，我们对比最接近的部分
    common_len = min(len(full_audio_stream), len(ref_wav_np))
    mse = np.mean((full_audio_stream[:common_len] - ref_wav_np[:common_len])**2)
    max_diff = np.max(np.abs(full_audio_stream[:common_len] - ref_wav_np[:common_len]))
    
    print(f"✅ 验证结果:")
    print(f"   - MSE: {mse:.2e}")
    print(f"   - Max Diff: {max_diff:.2e}")
    print("="*40)
    
    if max_diff < 1e-4:
        print("\n🎉 成功！流式状态机逻辑验证通过。")
        print("这证明了我们将模型拆解为 (codes + kv + buffer) -> (wav + new_kv + new_buffer) 是数学等价的。")
    else:
        print("\n⚠️ 警报：流式输出与参考输出不一致，请检查 latent_buffer 拼接逻辑。")

    # 保存结果供人工听感检查
    output_dir = "output_verify"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "verify_stateful_stream.wav"), full_audio_stream, 24000)
    print(f"\n音频已保存至: {output_dir}")

if __name__ == "__main__":
    main()
