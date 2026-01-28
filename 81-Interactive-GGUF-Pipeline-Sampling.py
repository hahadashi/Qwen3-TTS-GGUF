import os
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
import time
from transformers import AutoTokenizer
import qwen3_tts_gguf.nano_llama as nano_llama
from qwen3_tts_gguf import logger

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Qwen3TTS:
    """
    交互式 Qwen3-TTS GGUF 合成引擎 (采样增强版)
    """
    def __init__(self, model_root="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_root)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        
        # 路径定义
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman_advanced.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder.onnx"),
            "master_head": os.path.join(self.model_dir, "codec_head_weight.npy"),
            "text_table": os.path.join(self.model_dir, "text_embedding_projected.npy"),
            "proj_pt": os.path.join(self.model_dir, "craftsman_hf/master_to_craftsman_proj.pt")
        }
        
        print("[Engine] 正在启动初始化流程...")
        self.load_assets()
        self.init_engines()
        
    def load_assets(self):
        """加载权重表与 Tokenizer"""
        print("  - 加载权重表与 Tokenizer...")
        self.assets = {
            "master_head": np.load(self.paths["master_head"]),
            "text_table": np.load(self.paths["text_table"]),
            "emb_tables": [np.load(os.path.join(self.model_dir, f"codec_embedding_{i}.npy")) for i in range(16)],
            "proj": torch.load(self.paths["proj_pt"], map_location="cpu")
        }
        self.assets["tts_pad"] = self.assets["text_table"][151671]
        
        # 预投影加速
        proj_w = self.assets["proj"]["weight"].float()
        proj_b = self.assets["proj"]["bias"].float()
        self.assets["emb_tables_1024"] = [
            F.linear(torch.from_numpy(t).float(), proj_w, proj_b).numpy() for t in self.assets["emb_tables"]
        ]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
        print("  ✅ 资产加载完成。")

    def init_engines(self):
        """初始化 GGUF 与 ONNX 引擎"""
        print("  - 正在通过 Vulkan 挂载 GPU 引擎...")
        self.m_model = nano_llama.load_model(self.paths["master_gguf"], n_gpu_layers=-1)
        self.c_model = nano_llama.load_model(self.paths["craftsman_gguf"], n_gpu_layers=-1)
        
        # 上下文初始化 (持久化)
        m_params = nano_llama.llama_context_default_params()
        m_params.n_ctx = 4096
        m_params.embeddings = True
        self.m_ctx = nano_llama.llama_init_from_model(self.m_model, m_params)
        
        c_params = nano_llama.llama_context_default_params()
        c_params.n_ctx = 512
        c_params.embeddings = False
        self.c_ctx = nano_llama.llama_init_from_model(self.c_model, c_params)
        
        # 口腔解码器
        self.mouth_sess = ort.InferenceSession(self.paths["mouth_onnx"], providers=['CPUExecutionProvider'])
        
        # Batch 初始化
        self.m_batch = nano_llama.llama_batch_init(4096, 2048, 1)
        self.c_batch = nano_llama.llama_batch_init(32, 1024, 1)
        print("  ✅ 引擎初始化成功。环境已就绪。")

    def _sample(self, logits, temperature=1.0, top_p=1.0, top_k=0):
        """
        基于 NumPy 的采样函数
        """
        # 1. Temperature
        if temperature <= 1e-5:
            return np.argmax(logits)
        logits = logits / temperature
        
        # 2. Softmax
        # 数值稳定性处理
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        # 3. Top-K
        if top_k > 0 and top_k < len(probs):
            # 将小于 Top-K 阈值的概率置零
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.ones_like(probs, dtype=bool)
            mask[top_k_indices] = False
            probs[mask] = 0.0
            probs = probs / np.sum(probs) # 归一化
            
        # 4. Top-P (Nucleus)
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # 找到累积概率超过 top_p 的截止点
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            # 包含正好超过阈值的那个词，索引需加1
            cutoff_index = min(cutoff_index + 1, len(probs))
            
            # 保留前 cutoff_index 个，其余置零
            keep_indices = sorted_indices[:cutoff_index]
            mask = np.ones_like(probs, dtype=bool)
            mask[keep_indices] = False
            probs[mask] = 0.0
            probs = probs / np.sum(probs) # 归一化
            
        # 5. Random Choice
        return np.random.choice(len(probs), p=probs)

    def synthesize(self, text, speaker_id=3065, max_steps=250, verbose=False, 
                   do_sample=True, temperature=0.7, top_p=0.9, top_k=20):
        """
        全动态合成入口 (支持采样参数)
        """
        if verbose: print(f"\n[Synthesizer] 目标文本: {text}")
        start_time = time.time()
        
        # 1. 编译 Prompt
        p_start = time.time()
        prompt_embeds = self._construct_prompt(text, speaker_id)
        p_time = time.time() - p_start
        
        # 2. 推理
        sampling_config = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        all_codes, perf_stats = self._execute_inference(prompt_embeds, max_steps, verbose, sampling_config)
        
        # 3. 渲染
        r_start = time.time()
        audio_data = self._render_audio(all_codes)
        r_time = time.time() - r_start
        
        total_time = time.time() - start_time
        audio_dur = len(audio_data) / 24000.0
        rtf = total_time / audio_dur if audio_dur > 0 else 0
        
        if verbose:
            print("-" * 40)
            print(f"性能分析报告 (音频长度: {audio_dur:.2f}s)")
            print(f"  1. Prompt 编译: {p_time:.4f}s")
            print(f"  2. 大师 Prefill: {perf_stats['prefill_time']:.4f}s")
            print(f"  3. 自回环总计: {perf_stats['loop_time']:.4f}s")
            print(f"  4. 嘴巴渲染: {r_time:.4f}s")
            print("-" * 40)
            print(f"总耗时: {total_time:.4f}s | RTF: {rtf:.4f}")
        else:
            print(f"[Done] RTF: {rtf:.4f}")
            
        return audio_data

    # --- 内部组件 ---

    def _construct_prompt(self, text, spk_id):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        seq = [ (151644, 0), (77091, 0), (198, 0), (151671, 2154), (151671, 2156), (151671, 2055), (151671, 2157), (151671, spk_id), (151672, 2148) ]
        for tid in ids: seq.append((tid, 2148))
        seq.append((151673, 2148))
        seq.append((151671, 2149)) # Codec BOS
        
        embeds = []
        for tid, cid in seq:
            v = self.assets["text_table"][tid] + (self.assets["emb_tables"][0][cid] if cid != 0 else 0)
            embeds.append(v)
        return np.array(embeds).reshape(1, len(seq), 2048).astype(np.float32)

    def _execute_inference(self, prompt, max_steps, verbose, sampling_config):
        # 清理大师记忆
        nano_llama.llama_memory_clear(nano_llama.llama_get_memory(self.m_ctx), True)
        
        stats = {"master_time": 0, "craftsman_time": 0, "feedback_time": 0}
        
        conf = sampling_config
        # 提取参数
        do_sample = conf.get("do_sample", True)
        temp = conf.get("temperature", 0.7)
        top_p = conf.get("top_p", 0.9)
        top_k = conf.get("top_k", 20)
        
        # Prefill Master
        pre_start = time.time()
        n_p = prompt.shape[1]
        self.m_batch.n_tokens = n_p
        ctypes.memmove(self.m_batch.embd, np.ascontiguousarray(prompt[0]).ctypes.data, prompt[0].nbytes)
        for i in range(n_p):
            self.m_batch.pos[i], self.m_batch.pos[n_p+i], self.m_batch.pos[2*n_p+i], self.m_batch.pos[3*n_p+i] = i, i, i, 0
            self.m_batch.n_seq_id[i], self.m_batch.seq_id[i][0], self.m_batch.logits[i] = 1, 0, 1
        nano_llama.llama_decode(self.m_ctx, self.m_batch)
        
        m_hidden = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(self.m_ctx), shape=(n_p, 2048))[-1].copy()
        cur_pos, all_codes = n_p, []
        stats["prefill_time"] = time.time() - pre_start
        
        # Loop
        loop_start = time.time()
        for step_idx in range(max_steps):
            # 1. 大师预测
            m_logits = m_hidden @ self.assets["master_head"].T
            
            # Master Sampling
            if do_sample:
                code_0 = self._sample(m_logits, temp, top_p, top_k)
            else:
                code_0 = np.argmax(m_logits)
            
            # 日志 (Optional)
            # logger.info(f'{code_0=}')
            
            if code_0 == 2150: 
                if verbose: print(f"  └─ 步数 {step_idx}: 获得 EOS 信号，结束生成。")
                break
            
            # Craftsman
            step_codes, step_emb_2048 = [code_0], [self.assets["emb_tables"][0][code_0].copy()]
            proj_assets = self.assets["proj"]
            m_h_1024 = m_hidden @ proj_assets["weight"].float().numpy().T + proj_assets["bias"].float().numpy()
            c_in = np.stack([m_h_1024, self.assets["emb_tables_1024"][0][code_0]], axis=0)
            
            nano_llama.llama_memory_clear(nano_llama.llama_get_memory(self.c_ctx), True)
            self.c_batch.n_tokens = 2
            ctypes.memmove(self.c_batch.embd, c_in.ctypes.data, c_in.nbytes)
            for j in range(2):
                self.c_batch.pos[j], self.c_batch.n_seq_id[j], self.c_batch.seq_id[j][0], self.c_batch.logits[j] = j, 1, 0, (1 if j == 1 else 0)
            nano_llama.llama_decode(self.c_ctx, self.c_batch)
            
            last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(self.c_ctx), shape=(1, 30720))[0]
            
            for cs in range(1, 16):
                # Craftsman Sampling
                # 根据官方逻辑, subtalker 也可以 sample。我们统一使用一套参数。
                logits_slice = last_logits[(cs-1)*2048 : (cs-1)*2048 + 2048]
                
                if do_sample:
                    c = self._sample(logits_slice, temp, top_p, top_k)
                else:
                    c = np.argmax(logits_slice)
                    
                step_codes.append(c)
                step_emb_2048.append(self.assets["emb_tables"][cs][c].copy())
                
                if cs < 15:
                    self.c_batch.n_tokens, self.c_batch.pos[0], self.c_batch.logits[0] = 1, cs+1, 1
                    ctypes.memmove(self.c_batch.embd, self.assets["emb_tables_1024"][cs][c].ctypes.data, 4096)
                    nano_llama.llama_decode(self.c_ctx, self.c_batch)
                    last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(self.c_ctx), shape=(30720,))
            
            all_codes.append(step_codes)
            
            # Feedback
            summed = np.sum(step_emb_2048, axis=0) + self.assets["tts_pad"].flatten()
            self.m_batch.n_tokens = 1
            ctypes.memmove(self.m_batch.embd, summed.ctypes.data, summed.nbytes)
            self.m_batch.pos[0] = self.m_batch.pos[1] = self.m_batch.pos[2] = cur_pos
            self.m_batch.pos[3], self.m_batch.logits[0], cur_pos = 0, 1, cur_pos + 1
            nano_llama.llama_decode(self.m_ctx, self.m_batch)
            m_hidden = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(self.m_ctx), shape=(1, 2048))[0].copy()
        
        else:
            print(f"  ⚠️ 熔断预警: 推理达到上限 {max_steps} 步仍未停止，已强行熔断。")
            
        stats["loop_time"] = time.time() - loop_start
        return all_codes, stats

    def _render_audio(self, codes):
        if not codes: return np.array([])
        c_in = np.array(codes)[np.newaxis, ...].astype(np.int64)
        return self.mouth_sess.run(None, {'audio_codes': c_in})[0].squeeze()

    def __del__(self):
        try:
            nano_llama.llama_batch_free(self.m_batch)
            nano_llama.llama_batch_free(self.c_batch)
            nano_llama.llama_free(self.m_ctx)
            nano_llama.llama_free(self.c_ctx)
            nano_llama.llama_model_free(self.m_model)
            nano_llama.llama_model_free(self.c_model)
        except: pass

if __name__ == "__main__":
    tts = Qwen3TTS()
    TARGET_TEXT = "你好，我是具有随机性的 Qwen3-TTS。如果不信，你可以多让我说几次同样的话。"
    SPEAKER_ID = 3065
    MAX_STEPS = 400
    
    # 实验 1: 0.7 温度
    print("\n>>> 实验 1: Temperature 0.7 (标准)")
    wav = tts.synthesize(TARGET_TEXT, speaker_id=SPEAKER_ID, max_steps=MAX_STEPS, verbose=True, 
                         temperature=0.7, top_p=0.8, top_k=20)
    sf.write("output/sample_t07.wav", wav, 24000)

    # 实验 2: 1.0 温度 (更活泼)
    print("\n>>> 实验 2: Temperature 1.0 (活泼)")
    wav = tts.synthesize(TARGET_TEXT, speaker_id=SPEAKER_ID, max_steps=MAX_STEPS, verbose=True, 
                         temperature=1.0, top_p=0.9, top_k=40)
    sf.write("output/sample_t10.wav", wav, 24000)
    
    # 实验 3: Greedy
    print("\n>>> 实验 3: Greedy (固定)")
    wav = tts.synthesize(TARGET_TEXT, speaker_id=SPEAKER_ID, max_steps=MAX_STEPS, verbose=True, 
                         do_sample=False)
    sf.write("output/sample_greedy.wav", wav, 24000)
    
    print("\n✅ 所有采样实验完成。请检查 output 目录。")
