import os
import ctypes
import numpy as np
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MASTER_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_PATH = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")

def test_joint_loading():
    print("=== [Test 73] 大师与工匠联合加载与推理测试 ===\n")

    # 1. 检查文件
    for p in [MASTER_PATH, CRAFTSMAN_PATH]:
        if not os.path.exists(p):
            print(f"❌ 找不到模型文件: {p}")
            return

    # 2. 同时加载两个模型
    print(f"Loading Master: {os.path.basename(MASTER_PATH)}")
    master_model = nano_llama.load_model(MASTER_PATH, n_gpu_layers=0) # 先用 CPU
    
    print(f"Loading Craftsman: {os.path.basename(CRAFTSMAN_PATH)}")
    craftsman_model = nano_llama.load_model(CRAFTSMAN_PATH, n_gpu_layers=0)
    
    if not master_model or not craftsman_model:
        print("❌ 模型加载失败。")
        return

    # 3. 初始化两个 Context (独立上下文)
    print("\n正在初始化 Context...")
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 512
    ctx_params.embeddings = True
    
    master_ctx = nano_llama.llama_init_from_model(master_model, ctx_params)
    craftsman_ctx = nano_llama.llama_init_from_model(craftsman_model, ctx_params)
    
    if not master_ctx or not craftsman_ctx:
        print("❌ Context 初始化失败。")
        return

    # 4. 大师推理测试 (2048-dim input)
    print("\n--- [Phase 1] 大师组件推理 ---")
    n_tokens = 1
    master_n_embd = nano_llama.llama_model_n_embd(master_model)
    print(f"Master n_embd: {master_n_embd}")
    
    master_batch = nano_llama.llama_batch_init(n_tokens, master_n_embd, 1)
    master_batch.n_tokens = n_tokens
    
    # 填充随机数据
    rng_data = np.random.randn(n_tokens, master_n_embd).astype(np.float32)
    ctypes.memmove(master_batch.embd, rng_data.ctypes.data, rng_data.nbytes)
    
    master_batch.pos[0] = 0
    master_batch.n_seq_id[0] = 1
    master_batch.seq_id[0][0] = 0
    master_batch.logits[0] = 1
    
    ret = nano_llama.llama_decode(master_ctx, master_batch)
    if ret == 0:
        hidden_ptr = nano_llama.llama_get_embeddings(master_ctx)
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(master_n_embd,))
        print(f"✅ 大师推理成功！输出隐藏层均值: {np.mean(hidden):.6f}")
    else:
        print(f"❌ 大师推理失败: {ret}")

    # 5. 工匠推理测试 (1024-dim input)
    print("\n--- [Phase 2] 工匠组件推理 ---")
    craft_n_embd = nano_llama.llama_model_n_embd(craftsman_model)
    craft_n_vocab = nano_llama.llama_vocab_n_tokens(nano_llama.llama_model_get_vocab(craftsman_model))
    print(f"Craftsman n_embd: {craft_n_embd}, n_vocab: {craft_n_vocab}")
    
    craft_batch = nano_llama.llama_batch_init(n_tokens, craft_n_embd, 1)
    craft_batch.n_tokens = n_tokens
    
    rng_data_c = np.random.randn(n_tokens, craft_n_embd).astype(np.float32)
    ctypes.memmove(craft_batch.embd, rng_data_c.ctypes.data, rng_data_c.nbytes)
    
    craft_batch.pos[0] = 0
    craft_batch.n_seq_id[0] = 1
    craft_batch.seq_id[0][0] = 0
    craft_batch.logits[0] = 1
    
    ret = nano_llama.llama_decode(craftsman_ctx, craft_batch)
    if ret == 0:
        # 获取 Logits
        logits_ptr = nano_llama.llama_get_logits(craftsman_ctx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(craft_n_vocab,))
        pred_id = np.argmax(logits)
        print(f"✅ 工匠推理成功！预测 Token ID: {pred_id}, 置信度: {logits[pred_id]:.4f}")
    else:
        print(f"❌ 工匠推理失败: {ret}")

    # 清理
    print("\n正在释放资源...")
    nano_llama.llama_batch_free(master_batch)
    nano_llama.llama_batch_free(craft_batch)
    nano_llama.llama_free(master_ctx)
    nano_llama.llama_free(craftsman_ctx)
    nano_llama.llama_model_free(master_model)
    nano_llama.llama_model_free(craftsman_model)
    print("DONE.")

if __name__ == "__main__":
    test_joint_loading()
