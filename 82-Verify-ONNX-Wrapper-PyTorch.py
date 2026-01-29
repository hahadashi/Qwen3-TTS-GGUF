import torch
import numpy as np
import librosa
import os
from qwen3_tts_gguf.codec_export import StatefulCodecONNXWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

def main():
    model_path = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tokenizer_load_path = os.path.join(model_path, "speech_tokenizer") if os.path.exists(os.path.join(model_path, "speech_tokenizer")) else model_path
    device = "cpu"
    
    print(f"🚀 正在加载模型并初始化 ONNX Wrapper...")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_load_path).to(device)
    wrapper = StatefulCodecONNXWrapper(model).to(device)
    
    # 修正数据路径 (与 81 脚本保持一致)
    DEBUG_DATA_DIR = "debug_data"
    codes_ref = torch.from_numpy(np.load(os.path.join(DEBUG_DATA_DIR, "jintian_codes.npy"))).to(device)
    if codes_ref.dim() == 2:
        codes_ref = codes_ref.unsqueeze(0)
    
    # 计算参考波形
    import soundfile as sf
    wav_ref, _ = sf.read(os.path.join(DEBUG_DATA_DIR, "jintian_ref.wav"))
    
    B, T_frames, Q = codes_ref.shape
    chunk_size = 3
    
    # 初始化历史状态 (必须是 Tensor)
    is_last = torch.tensor([0.0], device=device)
    pre_conv_h = torch.zeros(B, 512, 0, device=device)
    latent_buf = torch.zeros(B, 1024, 0, device=device)
    conv_h = torch.zeros(B, 1024, 0, device=device)
    
    # KV 缓存：8层 * 2 (K,V) = 16 个 Tensor
    # 初始为空 Tensor [B, 16, 0, 64]
    pkv_list = []
    for _ in range(8):
        pkv_list.append(torch.zeros(B, 16, 0, 64, device=device)) # K
        pkv_list.append(torch.zeros(B, 16, 0, 64, device=device)) # V
        
    all_chunks = []
    
    print(f"🧪 正在执行无分支流式推理验证...")
    for i in range(0, T_frames, chunk_size):
        end_idx = min(i + chunk_size, T_frames)
        chunk_codes = codes_ref[:, i:end_idx, :]
        
        is_last_bool = (end_idx == T_frames)
        is_last.fill_(1.0 if is_last_bool else 0.0)
        
        # 调用无分支 Wrapper
        outputs = wrapper(
            chunk_codes,
            is_last,
            pre_conv_h,
            latent_buf,
            conv_h,
            *pkv_list
        )
        
        # 解包返回结果 (适配静态输出方案)
        chunk_wav_full = outputs[0]
        valid_len = int(outputs[1].item())
        pre_conv_h = outputs[2]
        latent_buf = outputs[3]
        conv_h = outputs[4]
        pkv_list = outputs[5:]
        
        # 根据 valid_len 截取有效部分
        chunk_wav = chunk_wav_full[:, :valid_len]
        
        if chunk_wav.shape[-1] > 0:
            print(f"  > 步长 {i//chunk_size}: 输出 {chunk_wav.shape[-1]} 采样点")
            all_chunks.append(chunk_wav.detach().cpu().numpy())
    
    # 合并结果
    wav_stream = np.concatenate(all_chunks, axis=-1)[0]
    
    print("\n" + "="*40)
    print(f"📊 流式拼接总长度: {len(wav_stream)}")
    print(f"📊 参考波形总长度: {len(wav_ref)}")
    
    common_len = min(len(wav_stream), len(wav_ref))
    mse = np.mean((wav_stream[:common_len] - wav_ref[:common_len])**2)
    max_diff = np.max(np.abs(wav_stream[:common_len] - wav_ref[:common_len]))
    
    print(f"✅ 验证结果:")
    print(f"   - MSE: {mse:.2e}")
    print(f"   - Max Diff: {max_diff:.2e}")
    print("="*40)
    
    if mse < 1e-6:
        print("🎉 恭喜！无分支版本逻辑验证成功，完全等价。")
    else:
        print("❌ 警告：无分支版本存在数值差异，请检查代码。")

if __name__ == "__main__":
    main()
