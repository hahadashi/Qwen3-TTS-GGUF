import os
import ctypes
import numpy as np
import soundfile as sf
import sounddevice as sd
import onnxruntime as ort
import time
import librosa
from transformers import AutoTokenizer
import qwen3_tts_gguf.llama as llama
from qwen3_tts_gguf import logger

# 强制使用 FP32 确保精度
os.environ["ORT_LOGGING_LEVEL"] = "3"

class Qwen3TTSClone:
    """
    Qwen3-TTS Base 模型离线语音克隆推理引擎 (101号脚本)
    特点: 整合 81 帧 ICL Prompt 构造，支持 ONNX 编码器 + GGUF 解码器
    """
    
    PROTOCOL = {
        "PAD": 2148, "BOS": 2149, "EOS": 2150, "BOS_TOKEN": 151672, "EOS_TOKEN": 151673,
        "THINK": 2154, "NOTHINK": 2155, "THINK_BOS": 2156, "THINK_EOS": 2157
    }

    def __init__(self, model_root="model-base", tokenizer_path="Qwen3-TTS-12Hz-1.7B-Base"):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_root)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        
        # 路径定义
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder.onnx"),
            "codec_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_encoder.onnx"),
            "spk_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_speaker_encoder.onnx"),
            "text_table": os.path.join(self.model_dir, "text_embedding_projected.npy"),
            "proj_w": os.path.join(self.model_dir, "proj_weight.npy"),
            "proj_b": os.path.join(self.model_dir, "proj_bias.npy")
        }
        
        print("🚀 [101-Clone] 正在初始化 Base 克隆引擎...")
        self.load_assets()
        self.init_engines()
        
    def load_assets(self):
        print("  - 正在加载权重矩阵与资产...")
        self.assets = {
            "text_table": np.load(self.paths["text_table"]),
            "proj": {
                "weight": np.load(self.paths["proj_w"]),
                "bias": np.load(self.paths["proj_b"])
            },
            "emb_tables": [],
            "emb_tables_1024": []
        }
        
        # 加载 16 层 Codec Embedding
        for i in range(16):
            # Base 导出时通常带有 _raw 后缀，这里做兼容
            base_name = f"codec_embedding_{i}_raw.npy"
            path = os.path.join(self.model_dir, base_name)
            if not os.path.exists(path):
                path = os.path.join(self.model_dir, f"codec_embedding_{i}.npy")
            
            emb_table = np.load(path)
            self.assets["emb_tables"].append(emb_table)
            
            # 预计算工匠投影 (2048 -> 1024)
            pw, pb = self.assets["proj"]["weight"], self.assets["proj"]["bias"]
            self.assets["emb_tables_1024"].append(emb_table @ pw.T + pb)
            
        self.assets["tts_pad"] = self.assets["text_table"][151671]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
        print("  ✅ 资产加载完成。")

    def init_engines(self):
        print("  - 正在挂载 GGUF/ONNX 算力矩阵...")
        # GGUF 引擎
        self.m_model = llama.load_model(self.paths["master_gguf"], n_gpu_layers=-1)
        self.c_model = llama.load_model(self.paths["craftsman_gguf"], n_gpu_layers=-1)
        
        m_params = llama.llama_context_default_params()
        m_params.n_ctx = 4096
        m_params.embeddings = True
        self.m_ctx = llama.llama_init_from_model(self.m_model, m_params)
        
        c_params = llama.llama_context_default_params()
        c_params.n_ctx = 512
        self.c_ctx = llama.llama_init_from_model(self.c_model, c_params)
        
        # ONNX 引擎 (编码与解码)
        self.codec_enc = ort.InferenceSession(self.paths["codec_enc_onnx"], providers=['CPUExecutionProvider'])
        self.spk_enc = ort.InferenceSession(self.paths["spk_enc_onnx"], providers=['CPUExecutionProvider'])
        self.mouth_sess = ort.InferenceSession(self.paths["mouth_onnx"], providers=['CPUExecutionProvider'])
        
        # Batch 初始化
        self.m_batch = llama.llama_batch_init(4096, 2048, 1)
        self.c_batch = llama.llama_batch_init(32, 1024, 1)
        print("  ✅ 引擎全线就绪。")

    def _extract_audio_features(self, ref_audio_path):
        """通过 ONNX 提取 Codec IDs 和 Speaker Embedding"""
        print(f"  - 正在分析参考音频: {os.path.basename(ref_audio_path)}")
        # 1. Codec Encoder (期望 24kHz)
        wav_24k, _ = librosa.load(ref_audio_path, sr=24000)
        c_out = self.codec_enc.run(['audio_codes'], {'input_values': wav_24k.reshape(1, -1).astype(np.float32)})
        codes = c_out[0][0] # [T, 16]
        
        # 2. Speaker Encoder (由 73 号脚本验证的标准参数: sr=24000, n_mels=128)
        # 使用 librosa 模拟官方 mel_spectrogram 逻辑
        spec = librosa.feature.melspectrogram(
            y=wav_24k, sr=24000, n_fft=1024, hop_length=256, win_length=1024, 
            n_mels=128, fmin=0.0, fmax=12000.0, center=False
        )
        mels = librosa.power_to_db(spec, ref=np.max) # 官方通常有 db 转换
        mels_input = mels.T[np.newaxis, ...].astype(np.float32) # [1, T, 128]
        
        # 注意: 输出节点名需与 73 号脚本保持一致 (spk_emb)
        s_out = self.spk_enc.run(['spk_emb'], {'mels': mels_input})
        spk_emb = s_out[0][0] # [2048]
        
        return codes, spk_emb

    def _construct_clone_prompt(self, text, ref_text, codes, spk_emb, lang_id=2055):
        """【核心】构造 81 帧语音克隆 Prompt"""
        print("  - 正在构造 81 帧 ICL Prompt...")
        p = self.PROTOCOL
        
        # 1. Header (3帧)
        header_ids = [151644, 77091, 198] # <|im_start|>assistant\n
        embeds = [self.assets["text_table"][tid] for tid in header_ids]
        
        # 2. Prefill (6帧)
        # 背景是 PAD(151671) 的投影，叠加音频侧的 ID 对应的向量
        prefill_layers = [p["THINK"], p["THINK_BOS"], lang_id, p["THINK_EOS"], 0, p["PAD"]]
        for cid in prefill_layers:
            v = self.assets["tts_pad"].copy()
            if cid != 0:
                v += self.assets["emb_tables"][0][cid]
            embeds.append(v)
        
        # 在特定位置（第 8 帧，即 embeds[7]）注入 Speaker Embedding
        embeds[7] += spk_emb
        
        # 3. ICL (72帧)
        # 组合 ID: 参考文本 ID + 目标文本 ID + EOS
        ref_ids = self.tokenizer.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n", add_special_tokens=False)
        target_ids = self.tokenizer.encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        
        # 官方 ICL Token 构造: ref[3:-2] + target[3:-5]
        icl_ids = ref_ids[3:-2] + target_ids[3:-5]
        icl_ids = icl_ids[:71] # 截断/对齐最大 71 个核心 Token
        icl_ids.append(151673) # 补一个 EOS
        
        # 【关键修复】如果文本 Token 比音频帧短，用 151671 (PAD) 填充，防止 Index Out Of Range
        num_audio_frames = min(72, codes.shape[0])
        if len(icl_ids) < num_audio_frames:
            icl_ids += [151671] * (num_audio_frames - len(icl_ids))
        
        # 逐帧叠加 16 层 Codec
        for t in range(num_audio_frames):
            # 获取文本背景 (已确保 t 不会溢出)
            v = self.assets["text_table"][icl_ids[t]].copy()
            # 叠加该帧的 16 层音频
            for q in range(16):
                v += self.assets["emb_tables"][q][codes[t, q]]
            embeds.append(v)
            
        return np.array(embeds).reshape(1, len(embeds), 2048).astype(np.float32)

    def _sample(self, logits, temp=1.0, top_p=1.0, top_k=50):
        if temp <= 1e-5: return np.argmax(logits)
        logits = logits / temp
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        if top_k > 0:
            indices = np.argsort(probs)[-top_k:]
            mask = np.ones_like(probs, dtype=bool); mask[indices] = False
            probs[mask] = 0; probs /= np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def clone_voice(self, text, ref_audio, ref_text, max_steps=400, verbose=True):
        start_time = time.time()
        
        # A. 提取参考特征
        codes, spk_emb = self._extract_audio_features(ref_audio)
        
        # B. 构造 Prompt
        prompt = self._construct_clone_prompt(text, ref_text, codes, spk_emb)
        print(f"  - Prompt Ready: {prompt.shape}")

        # C. 推理自回环 (与 41 号逻辑一致)
        llama.llama_memory_clear(llama.llama_get_memory(self.m_ctx), True)
        self.m_batch.n_tokens = prompt.shape[1]
        ctypes.memmove(self.m_batch.embd, prompt[0].ctypes.data, prompt[0].nbytes)
        for i in range(prompt.shape[1]):
            self.m_batch.pos[i] = self.m_batch.pos[prompt.shape[1]+i] = self.m_batch.pos[2*prompt.shape[1]+i] = i
            self.m_batch.pos[3*prompt.shape[1]+i] = 0
            self.m_batch.n_seq_id[i], self.m_batch.seq_id[i][0], self.m_batch.logits[i] = 1, 0, 1
        
        llama.llama_decode(self.m_ctx, self.m_batch)
        m_hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.m_ctx), shape=(prompt.shape[1], 2048))[-1].copy()
        m_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.m_ctx), shape=(prompt.shape[1], 3072))[-1].copy()
        
        cur_pos, all_codes = prompt.shape[1], []
        print("  - 正在启动生成的齿轮...")

        for s in range(max_steps):
            # Master 预测
            c0 = self._sample(m_logits)
            if c0 == 2150: break
            
            # Craftsman 辅助
            step_codes, step_emb_2048 = [c0], [self.assets["emb_tables"][0][c0].copy()]
            pw, pb = self.assets["proj"]["weight"], self.assets["proj"]["bias"]
            m_h_1024 = m_hidden @ pw.T + pb
            c_in = np.stack([m_h_1024, self.assets["emb_tables_1024"][0][c0]], axis=0)
            
            llama.llama_memory_clear(llama.llama_get_memory(self.c_ctx), True)
            self.c_batch.n_tokens = 2
            ctypes.memmove(self.c_batch.embd, c_in.ctypes.data, c_in.nbytes)
            for j in range(2):
                self.c_batch.pos[j], self.c_batch.n_seq_id[j], self.c_batch.seq_id[j][0], self.c_batch.logits[j] = j, 1, 0, (1 if j == 1 else 0)
            llama.llama_decode(self.c_ctx, self.c_batch)
            
            last_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.c_ctx), shape=(1, 30720))[0]
            for cs in range(1, 16):
                sl = last_logits[(cs-1)*2048 : (cs-1)*2048 + 2048]
                c = self._sample(sl)
                step_codes.append(c)
                step_emb_2048.append(self.assets["emb_tables"][cs][c].copy())
                if cs < 15:
                    self.c_batch.n_tokens, self.c_batch.pos[0], self.c_batch.logits[0] = 1, cs+1, 1
                    ctypes.memmove(self.c_batch.embd, self.assets["emb_tables_1024"][cs][c].ctypes.data, 4096)
                    llama.llama_decode(self.c_ctx, self.c_batch)
                    last_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.c_ctx), shape=(30720,))
            
            all_codes.append(step_codes)
            if verbose and s % 10 == 0: print(f"    [Step {s}] 生成中...", end="\r")
            
            # Master Feedback
            summed = np.sum(step_emb_2048, axis=0) + self.assets["tts_pad"].flatten()
            self.m_batch.n_tokens = 1
            ctypes.memmove(self.m_batch.embd, summed.ctypes.data, summed.nbytes)
            self.m_batch.pos[0] = self.m_batch.pos[1] = self.m_batch.pos[2] = cur_pos
            self.m_batch.pos[3], self.m_batch.logits[0], cur_pos = 0, 1, cur_pos + 1
            llama.llama_decode(self.m_ctx, self.m_batch)
            m_hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.m_ctx), shape=(1, 2048))[0].copy()
            m_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.m_ctx), shape=(1, 3072))[0].copy()

        # D. 音频渲染
        print(f"\n  - 生成步数: {len(all_codes)} | 耗时: {time.time() - start_time:.2f}s")
        audio = self.mouth_sess.run(None, {'audio_codes': np.array(all_codes)[np.newaxis, ...].astype(np.int64)})[0].squeeze()
        return audio

if __name__ == "__main__":
    engine = Qwen3TTSClone()
    
    TEXT = "你好！这是 Qwen3-TTS Base 模型的全离线语音克隆测试。所有的推理都在 GGUF 和 ONNX 上完成。"
    REF_AUDIO = "output/sample2.wav"
    REF_TEXT = "你好，我是千问3-TTS，很高兴遇见你，你今天过得好吗？"
    
    # 执行克隆
    wav = engine.clone_voice(TEXT, REF_AUDIO, REF_TEXT)
    
    # 保存结果
    out_path = "output/clone_result.wav"
    sf.write(out_path, wav, 24000)
    print(f"\n🎉 克隆成功！文件已存至: {out_path}")
    sd.play(wav, 24000); sd.wait()
