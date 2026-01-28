import os
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
import time
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 路径与配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(SAVE_DIR, exist_ok=True)

MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MOUTH_ONNX = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")

EOS_TOKEN_ID = 2150

def load_assets():
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    assets["prefill_input"] = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
    assets["trailing_text"] = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
    assets["tts_pad"] = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)

    # 预计算工匠投影表
    proj_w = assets["proj"]["weight"].float()
    proj_b = assets["proj"]["bias"].float()
    assets["emb_tables_1024"] = [F.linear(torch.from_numpy(t).float(), proj_w, proj_b).numpy() for t in assets["emb_tables"]]
    return assets

def apply_projection(hidden_2048, proj_assets):
    """投影维度: 2048 -> 1024 (工匠专用)"""
    w = proj_assets["weight"].float().numpy()
    b = proj_assets["bias"].float().numpy()
    return hidden_2048 @ w.T + b

def run_performance_test():
    print("=== [77] Qwen3-TTS GGUF 精简版推理性能测试 ===")
    
    assets = load_assets()
    
    # 初始化模型
    m_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=-1)
    m_ctx_params = nano_llama.llama_context_default_params()
    m_ctx_params.n_ctx = 4096
    m_ctx_params.embeddings = True
    m_ctx = nano_llama.llama_init_from_model(m_model, m_ctx_params)
    
    c_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=-1)
    c_ctx_params = nano_llama.llama_context_default_params()
    c_ctx_params.n_ctx = 512
    c_ctx_params.embeddings = True
    c_ctx = nano_llama.llama_init_from_model(c_model, c_ctx_params)
    
    sess_opts = ort.SessionOptions()
    mouth_sess = ort.InferenceSession(MOUTH_ONNX, sess_opts, providers=['CPUExecutionProvider'])
    
    m_batch = nano_llama.llama_batch_init(4096, 2048, 1)
    c_batch = nano_llama.llama_batch_init(32, 1024, 1)

    print("\n--- 开始端到端生成 ---")
    start_time = time.time()
    
    # 1. Prefill
    prefill_input = assets["prefill_input"]
    n_tokens = prefill_input.shape[1]
    m_batch.n_tokens = n_tokens
    ctypes.memmove(m_batch.embd, np.ascontiguousarray(prefill_input[0]).ctypes.data, prefill_input[0].nbytes)
    for i in range(n_tokens):
        m_batch.pos[i] = i
        m_batch.pos[n_tokens+i] = i
        m_batch.pos[2*n_tokens+i] = i
        m_batch.pos[3*n_tokens+i] = 0
        m_batch.n_seq_id[i] = 1
        m_batch.seq_id[i][0] = 0
        m_batch.logits[i] = 1 if i == n_tokens - 1 else 0
    nano_llama.llama_decode(m_ctx, m_batch)
    
    m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
    m_hidden_last = np.ctypeslib.as_array(m_out_ptr, shape=(n_tokens, 2048))[-1].copy()
    current_m_pos = n_tokens
    
    all_codes = []
    
    # 2. 自回归循环
    loop_start = time.time()
    for step_idx in range(50):
        # Master Predict
        m_logits = m_hidden_last @ assets["master_head"].T
        code_0 = np.argmax(m_logits)
        if code_0 == EOS_TOKEN_ID: break
            
        step_codes = [code_0]
        step_emb_2048 = [assets["emb_tables"][0][code_0].copy()]
        
        # 大师隐藏层实时投影到 1024 维
        m_hidden_last_1024 = apply_projection(m_hidden_last, assets["proj"])
        
        # Craftsman Predict: [M_1024, C0_1024] -> [2, 1024]
        c_in_1024 = np.stack([m_hidden_last_1024, assets["emb_tables_1024"][0][code_0]], axis=0)
        nano_llama.llama_memory_clear(nano_llama.llama_get_memory(c_ctx), True)
        
        c_batch.n_tokens = 2
        ctypes.memmove(c_batch.embd, c_in_1024.ctypes.data, c_in_1024.nbytes)
        for i in range(2):
            c_batch.pos[i], c_batch.n_seq_id[i], c_batch.seq_id[i][0], c_batch.logits[i] = i, 1, 0, (1 if i == 1 else 0)
        nano_llama.llama_decode(c_ctx, c_batch)
        
        last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(2, 30720))[1]
        for c_step in range(1, 16):
            code = np.argmax(last_logits[(c_step-1)*2048 : (c_step-1)*2048 + 2048])
            step_codes.append(code)
            step_emb_2048.append(assets["emb_tables"][c_step][code].copy())
            if c_step < 15:
                next_in_1024 = assets["emb_tables_1024"][c_step][code]
                c_batch.n_tokens = 1
                c_batch.pos[0] = c_step + 1
                ctypes.memmove(c_batch.embd, next_in_1024.ctypes.data, next_in_1024.nbytes)
                c_batch.logits[0] = 1
                nano_llama.llama_decode(c_ctx, c_batch)
                last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))

        all_codes.append(step_codes)
        
        # Feedback
        summed = np.sum(step_emb_2048, axis=0)
        summed += assets["trailing_text"][0, step_idx] if step_idx < assets["trailing_text"].shape[1] else assets["tts_pad"].flatten()
        m_batch.n_tokens = 1
        ctypes.memmove(m_batch.embd, summed.ctypes.data, summed.nbytes)
        m_batch.pos[0], m_batch.pos[1], m_batch.pos[2], m_batch.pos[3] = current_m_pos, current_m_pos, current_m_pos, 0
        m_batch.logits[0] = 1
        nano_llama.llama_decode(m_ctx, m_batch)
        current_m_pos += 1
        m_hidden_last = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(m_ctx), shape=(1, 2048))[0].copy()

    loop_end = time.time()
    
    # 3. Decode
    decode_start = time.time()
    codes_np = np.array(all_codes)[np.newaxis, ...].astype(np.int64)
    audio_data = mouth_sess.run(None, {'audio_codes': codes_np})[0].squeeze()
    decode_end = time.time()
    
    total_end = time.time()
    
    # 保存结果
    W_PATH = os.path.join(SAVE_DIR, "final_speed_test.wav")
    sf.write(W_PATH, audio_data, 24000)
    
    # 统计
    n_frames = len(all_codes)
    n_tokens_total = n_frames * 16
    audio_dur = len(audio_data) / 24000.0
    
    print("\n--- 性能报告 ---")
    print(f"音频长度: {audio_dur:.2f} 秒")
    print(f"生成帧数: {n_frames}")
    print(f"总离散分码: {n_tokens_total}")
    print(f"Prefill 耗时: {loop_start - start_time:.4f} 秒")
    print(f"自回归耗时: {loop_end - loop_start:.4f} 秒 ({n_frames / (loop_end - loop_start):.2f} 帧/秒)")
    print(f"解码器耗时: {decode_end - decode_start:.4f} 秒")
    print(f"总端到端耗时: {total_end - start_time:.4f} 秒")
    print(f"实时率 (RTF): {(total_end - start_time) / audio_dur:.4f}")
    print(f"✅ 结果已保存: {W_PATH}")

    # 清理
    nano_llama.llama_batch_free(m_batch)
    nano_llama.llama_batch_free(c_batch)
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(m_model)
    nano_llama.llama_model_free(c_model)

if __name__ == "__main__":
    run_performance_test()
