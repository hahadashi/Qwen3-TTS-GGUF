import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

# 模型与路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_steps")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_talker.gguf")

def compare_embeddings(official, gguf_out, step_name):
    official = official.flatten()
    gguf_out = gguf_out.flatten()
    
    # 检查长度
    if len(official) != len(gguf_out):
        print(f"  [{step_name}] ❌ 维度不匹配: Official {len(official)} vs GGUF {len(gguf_out)}")
        return
    
    # 计算 MAE
    mae = np.mean(np.abs(official - gguf_out))
    # 计算余弦相似度
    cos_sim = np.dot(official, gguf_out) / (np.linalg.norm(official) * np.linalg.norm(gguf_out))
    
    # 判定通过标准
    pass_mark = "✅" if mae < 1e-3 and cos_sim > 0.999 else "⚠️"
    print(f"  {pass_mark} [{step_name}] MAE: {mae:.6f}, CosSim: {cos_sim:.6f}")
    return mae, cos_sim

def run_verification():
    # 1. 初始化模型与上下文
    print(f"正在加载 GGUF 模型: {MODEL_PATH}")
    model = nano_llama.load_model(MODEL_PATH, n_gpu_layers=0) # 建议 CPU 模式以获得更稳健的数值对比
    if not model: return
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.embeddings = True # 必须开启才能提取隐藏层
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    
    n_embd = nano_llama.llama_model_n_embd(model)
    
    # 获取捕获的步数
    if not os.path.exists(CAPTURED_DIR):
        print(f"❌ 未找到捕获目录: {CAPTURED_DIR}")
        return
    
    file_list = os.listdir(CAPTURED_DIR)
    steps = sorted([int(f.split('_')[1]) for f in file_list if f.startswith("step_") and "input_embeds" in f])
    
    if not steps:
        print("❌ 目录中没有捕获步数据")
        return

    print(f"开始对齐验证，总计 {len(steps)} 步 (数据来源: {CAPTURED_DIR})\n")
    
    current_pos = 0 # 维护全局位置偏移
    
    for step_idx in steps:
        input_file = os.path.join(CAPTURED_DIR, f"step_{step_idx}_input_embeds.npy")
        output_file = os.path.join(CAPTURED_DIR, f"step_{step_idx}_output_hidden.npy")
        
        # 加载官方数据
        inputs_embeds = np.load(input_file).astype(np.float32)
        official_hidden = np.load(output_file).astype(np.float32)
        
        n_tokens = inputs_embeds.shape[1]
        
        # 2. 准备 Batch
        # llama_batch_init(n_tokens, dim, n_seq_max)
        # 注意：由于使用了 IMRoPE，batch 大小通常需要扩展以预留多维度 pos 槽位
        batch = nano_llama.llama_batch_init(n_tokens * 4, n_embd, 1)
        batch.n_tokens = n_tokens
        
        # 注入 Embedding
        full_embd = np.ascontiguousarray(inputs_embeds[0])
        ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
        
        # 填充位置 (M-RoPE 要求 3 个维度的位置 ID)
        for k in range(n_tokens):
            pos = current_pos + k
            batch.pos[k] = pos
            batch.pos[n_tokens + k] = pos
            batch.pos[2 * n_tokens + k] = pos
            batch.pos[3 * n_tokens + k] = 0 # Padding Dim
            
            batch.n_seq_id[k] = 1
            batch.seq_id[k][0] = 0
            # 我们需要获取这一步生成的最后一个 token 的 hidden state
            # 标记为 1 时，llama.cpp 才会计算该位置的 output
            batch.logits[k] = 1 if k == n_tokens - 1 else 0
            
        # 3. 推理 (Decode)
        ret = nano_llama.llama_decode(ctx, batch)
        if ret != 0:
            print(f"  ❌ Step {step_idx}: llama_decode 失败 (代码 {ret})")
            nano_llama.llama_batch_free(batch)
            break
            
        # 4. 获取 GGUF 的输出 Hidden State
        out_ptr = nano_llama.llama_get_embeddings(ctx)
        
        # 处理 llama.cpp 可能返回全量 batch embedding 的情况
        # 如果 embeddings=True，llama.cpp 无论 logits 设为什么，都可能返回全量输入 token 的 embedding
        # 我们取最后一个 token 的 output
        gguf_full_hidden = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
        gguf_last_hidden = gguf_full_hidden[-1].copy()
        
        # 5. 跟官方输出对比 (取序列最后一个 token)
        compare_embeddings(official_hidden[0, -1], gguf_last_hidden, f"Step {step_idx:02}")
        
        # 更新位置偏移 (Prefill 完之后步进)
        current_pos += n_tokens
        
        # 清理 Batch
        nano_llama.llama_batch_free(batch)

    # 最终清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    print("\n对齐验证完成。")

if __name__ == "__main__":
    run_verification()
