import os
import sys
import numpy as np
import onnxruntime as ort
import librosa
import torch
from transformers import AutoTokenizer

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model-base")
ONNX_ENC_PATH = os.path.join(PROJECT_ROOT, "model-base", "qwen3_tts_encoder.onnx")
ONNX_SPK_PATH = os.path.join(PROJECT_ROOT, "model-base", "qwen3_tts_speaker_encoder.onnx")
CAPTURED_PROMPT_PATH = os.path.join(PROJECT_ROOT, "captured_prompt_outputs", "prompt_inputs_embeds.npy")
REF_AUDIO = os.path.join(PROJECT_ROOT, "output", "sample.wav")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-Base")

# 官方协议 ID (Verified)
P = {
    "PAD": 2148, "BOS": 2149, "EOS": 2150, 
    "BOS_TOKEN": 151672, "EOS_TOKEN": 151673, "PAD_TOKEN": 151671,
    "THINK": 2154, "NOTHINK": 2155, "THINK_BOS": 2156, "THINK_EOS": 2157,
    "CHINESE": 2055
}

def mel_spectrogram_np(wav):
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    mels = mel_spectrogram(
        torch.from_numpy(wav).unsqueeze(0), 
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000
    ).transpose(1, 2).numpy().astype(np.float32)
    return mels

def main():
    print("🚀 启动修正版 Prompt Embedding 手工构造 (81 帧精准还原)...")

    # 1. 加载资产
    text_table = np.load(os.path.join(MODEL_DIR, "text_embedding_projected.npy"), mmap_mode='r')
    # 根据 model-base 文件夹列表，所有文件都有 _raw.npy 后缀
    codec_tables = [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}_raw.npy")) for i in range(16)]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    # 2. 推理 ONNX 组件获取输入特征
    sess_enc = ort.InferenceSession(ONNX_ENC_PATH)
    sess_spk = ort.InferenceSession(ONNX_SPK_PATH)
    wav, _ = librosa.load(REF_AUDIO, sr=24000)
    
    # Speaker Embedding
    mels = mel_spectrogram_np(wav)
    spk_emb = sess_spk.run(['spk_emb'], {'mels': mels})[0][0] # (2048,)
    
    # Codec Codes
    audio_codes = sess_enc.run(['audio_codes'], {'input_values': wav.reshape(1, -1)})[0][0] # (T, 16)
    T_audio = audio_codes.shape[0]

    # 3. 准备基础文本 IDs (需与 66 号脚本完全对齐)
    text = "你好！这是捕获官方提示词嵌入的测试。"
    ref_text = "你好，我是具有随机性的千问3-TTS，这是我的终极进化形态"
    input_ids = tokenizer.encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    ref_ids = tokenizer.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n", add_special_tokens=False)

    embed_blocks = []

    # --- [A] Header (前 3 帧): <|im_start|>assistant\n
    for i in range(3):
        embed_blocks.append(text_table[input_ids[i]])

    # --- [B] Prefill (第 3-8 帧, 共 6 帧)
    # 官方: codec_input_emebdding = [think, think_bos, lang, think_eos, spk_emb, pad, bos]
    # 取除 BOS 外的前 6 位，与 (Pad*5, Bos) 叠加
    codec_in = [
        codec_tables[0][P["THINK"]],
        codec_tables[0][P["THINK_BOS"]],
        codec_tables[0][P["CHINESE"]],
        codec_tables[0][P["THINK_EOS"]],
        spk_emb,
        codec_tables[0][P["PAD"]]
    ]
    text_bg = [
        text_table[P["PAD_TOKEN"]],
        text_table[P["PAD_TOKEN"]],
        text_table[P["PAD_TOKEN"]],
        text_table[P["PAD_TOKEN"]],
        text_table[P["PAD_TOKEN"]],
        text_table[P["BOS_TOKEN"]]
    ]
    for c, t in zip(codec_in, text_bg):
        embed_blocks.append(c + t)

    # --- [C] ICL (第 9-80 帧, 共 72 帧)
    # generate_icl_prompt 逻辑:
    # Text侧: ref_ids[3:-2] + input_ids[3:4] + tts_eos
    icl_text_ids = ref_ids[3:-2] + [input_ids[3]] + [P["EOS_TOKEN"]]
    
    # Codec侧: codec_bos + SUM(ref_audio_codes)
    def get_summed_codec(t_idx):
        s = np.zeros((2048,), dtype=np.float32)
        for q in range(16):
            s += codec_tables[q][audio_codes[t_idx, q]]
        return s

    icl_codec_0 = codec_tables[0][P["BOS"]]
    
    # ICL 区域长度为 72 (1位BOS + 71位特征)
    for i in range(72):
        # 文本侧补 Pad
        if i < len(icl_text_ids):
            v_t = text_table[icl_text_ids[i]]
        else:
            v_t = text_table[P["PAD_TOKEN"]]
            
        # 音频侧
        if i == 0:
            v_c = icl_codec_0
        else:
            v_c = get_summed_codec(i - 1)
            
        embed_blocks.append(v_t + v_c)

    final_prompt = np.stack(embed_blocks).reshape(1, -1, 2048).astype(np.float32)

    # 4. 对比验证
    official_prompt = np.load(CAPTURED_PROMPT_PATH)
    print("\n" + "="*40)
    print(f"验证分析:")
    print(f"手动构造总帧数: {final_prompt.shape[1]}")
    print(f"官方捕获总帧数: {official_prompt.shape[1]}")
    
    min_len = min(final_prompt.shape[1], official_prompt.shape[1])
    f_cut = final_prompt[0, :min_len]
    o_cut = official_prompt[0, :min_len]

    cos_sim = np.dot(f_cut.flatten(), o_cut.flatten()) / (np.linalg.norm(f_cut) * np.linalg.norm(o_cut))
    max_diff = np.max(np.abs(f_cut - o_cut))
    
    print(f"对比长度: {min_len}")
    print(f"余弦相似度: {cos_sim:.10f}")
    print(f"最大绝对误差: {max_diff:.8e}")

    if cos_sim > 0.9999 and final_prompt.shape[1] == official_prompt.shape[1]:
        print("\n✅ 完美成功！81 帧逻辑已完全闭合。")
    else:
        print("\n❌ 仍有不一致，详细检视关键帧:")
        # 检视 Header, Prefill 切换点, ICL 起始点
        check_points = [0, 2, 3, 8, 9, 80]
        for p in check_points:
            if p < min_len:
                d = np.max(np.abs(f_cut[p] - o_cut[p]))
                print(f"  帧 {p}: 误差 {d:.4e}")

if __name__ == "__main__":
    main()
