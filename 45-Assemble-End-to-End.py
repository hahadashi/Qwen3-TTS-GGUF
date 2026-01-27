import os
import sys
import ctypes
import numpy as np
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")

def run_assembly():
    print(f"--- 组装联合测试 (Assembly Test) ---")
    
    # 1. 加载捕获的数据
    print("加载捕获数据...")
    try:
        prefill_input_embeds = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
        official_prefill_hidden = np.load(os.path.join(CAPTURED_DIR, "prefill_output_hidden.npy")).astype(np.float32)
        official_craftsman_input = np.load(os.path.join(CAPTURED_DIR, "craftsman_step_0_input_2048.npy")).astype(np.float32)
        official_final_codes = np.load(os.path.join(CAPTURED_DIR, "master_step_0_result_codes.npy"))
    except FileNotFoundError as e:
        print(f"❌ 缺少捕获文件: {e}")
        return

    # 提取 official last_id_hidden (raw embedding)
    # craftsman_input = [past_hidden, last_id_hidden] (Dim: [B, 2, 2048])
    # 注意：Official Capture 并没有保证 dim=2 是 cat 顺序，需检查 modeling 代码
    # modeling: torch.cat((past_hidden, last_id_hidden), dim=1)
    # past_hidden 通常是 [B, 1, 2048]. so index 0 is past_hidden, index 1 is last_id_hidden.
    
    official_last_id_hidden = official_craftsman_input[:, 1:2, :] # [B, 1, 2048]
    
    # 2. 运行 Master GGUF
    print(f"\n[Master GGUF] Loading {GGUF_PATH}...")
    model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    if not model: return
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.embeddings = True
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(model)
    
    # 准备 GGUF Batch (Prefill)
    n_tokens = prefill_input_embeds.shape[1]
    batch = nano_llama.llama_batch_init(n_tokens * 4, n_embd, 1)
    batch.n_tokens = n_tokens
    
    full_embd = np.ascontiguousarray(prefill_input_embeds[0])
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    current_pos = 0 # 初始位置
    for k in range(n_tokens):
        pos = current_pos + k
        # M-RoPE 填充 (Mocking position layout, assuming pure text prompt or handled similarly)
        # Prefill 阶段通常是 Prompt，包含 Vision 吗？ "今天天气不错" 是纯文本。
        # 纯文本时 pos 是一样的。
        batch.pos[k] = pos
        batch.pos[n_tokens + k] = pos
        batch.pos[2 * n_tokens + k] = pos
        batch.pos[3 * n_tokens + k] = 0
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        batch.logits[k] = 1 if k == n_tokens - 1 else 0 # 只计算最后一个 token 的 output
    
    print(f"[Master GGUF] Running Prefill ({n_tokens} tokens)...")
    ret = nano_llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"❌ GGUF 推理失败: {ret}")
        return
        
    out_ptr = nano_llama.llama_get_embeddings(ctx)
    gguf_full_hidden = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
    gguf_last_hidden = gguf_full_hidden[-1].copy() # [2048]
    gguf_past_hidden = gguf_last_hidden.reshape(1, 1, n_embd) # [1, 1, 2048]
    
    # 验证 Master Output
    # official_prefill_hidden [1, 1, 2048]
    diff_master = np.abs(official_prefill_hidden - gguf_past_hidden).mean()
    cos_master = np.dot(official_prefill_hidden.flatten(), gguf_past_hidden.flatten()) / (
        np.linalg.norm(official_prefill_hidden) * np.linalg.norm(gguf_past_hidden)
    )
    
    print(f"  [Check 1] GGUF Output vs Official Output")
    pass_mark_1 = "✅" if diff_master < 1e-3 and cos_master > 0.999 else "⚠️"
    print(f"  {pass_mark_1} MAE: {diff_master:.6f}, CosSim: {cos_master:.6f}")
    
    # 3. 组装输入 (Glue)
    # Hybrid Input = GGUF Output + Official Last ID Embedding
    hybrid_craftsman_input = np.concatenate([gguf_past_hidden, official_last_id_hidden], axis=1) # [1, 2, 2048]
    
    # 4. 运行 Craftsman ONNX
    # 4. 运行 Craftsman ONNX (Step 0)
    print(f"\n[Craftsman ONNX] Loading {ONNX_PATH}...")
    
    # Load Heads
    heads_path = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")
    print(f"[Craftsman ONNX] Loading Heads from {heads_path}...")
    if not os.path.exists(heads_path):
        print(f"❌ Heads file missing")
        return
    predictor_heads = np.load(heads_path) # [15, 3072, 2048]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # Load all Embedding Tables (Codec 0-15)
    # Codec 0 is Master Output / Craftsman Input Last ID
    # Codec 1-15 are Craftsman Prediction Output embeddings
    
    print(f"[Craftsman ONNX] Loading Embedding Tables...")
    embedding_tables = {}
    for i in range(16):
        path = os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")
        if not os.path.exists(path):
            print(f"❌ Missing embedding table: {path}")
            return
        embedding_tables[i] = np.load(path) # [N, 2048]

    # Prepare Empty Pasts (10 tensors) for Step 0
    current_pasts = {f"past_{i}": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(10)}
    
    # 获取 last_id_hidden 对应的 code
    # official_craftsman_input 是 [gguf_past, last_id_hidden]
    # 我们需要找到 last_id_hidden 是哪个 code 产生的。
    # 从 capture 的 master_step_0_input_ids.npy 获取
    master_step_0_input_ids = np.load(os.path.join(CAPTURED_DIR, "master_step_0_input_ids.npy"))
    # 这通常是 Step 0 的 target，或者输入的 code。
    # 在 T-T-S 场景，Generate 阶段的 input_ids 是刚刚生成的 token。
    input_code_idx = master_step_0_input_ids[0, 0] # 标量
    print(f"  Input Code Index (Master): {input_code_idx}")
    
    # 验证 last_id_hidden 与 embedding_table[0]
    # gguf output 用的 prefill output.
    # glue input part 2 should match embedding_table[0][input_code_idx]
    
    last_id_embed = embedding_tables[0][input_code_idx].reshape(1, 1, 2048)
    
    # Recalculate hybrid input using pure embeddings for consistency if desired, 
    # but let's stick to using GGUF output for the first part to prove GGUF linkage.
    # hybrid_craftsman_input = [gguf_past_hidden, last_id_embed]
    
    # 开始 15 步自回归
    print(f"[Craftsman ONNX] Running 15-Step Autoregressive Loop...")
    
    generated_codes = []
    generated_embeddings = [] # 用于 Sum
    
    # 添加 last_id_hidden 到 Sum 列表 (对应 Group 0)
    generated_embeddings.append(last_id_embed)
    
    # 初始 Input
    current_input_embeds = np.concatenate([gguf_past_hidden, last_id_embed], axis=1) # [1, 2, 2048]
    
    for step in range(15):
        # 1. Run ONNX
        inputs = {'inputs_embeds': current_input_embeds}
        inputs.update(current_pasts)
        
        outputs = sess.run(None, inputs)
        onnx_hidden = outputs[0] # [1, 1, 2048]
        pasts_out = outputs[1:]
        
        # Update Pasts
        for i in range(5):
            current_pasts[f"past_{i*2}"] = pasts_out[i*2]
            current_pasts[f"past_{i*2+1}"] = pasts_out[i*2+1]
            
        # 2. Predict Code
        # predictor_heads[step] 对应 prediction of Group (step+1)
        head_weight = predictor_heads[step] 
        logits = onnx_hidden[0, -1] @ head_weight.T
        code = np.argmax(logits)
        generated_codes.append(code)
        
        # 3. Look up Embedding for NEXT step Input
        # Code Group (step+1) produced 'code'.
        # We need Embedding Table for Group (step+1) -> codec_embedding_{step+1}.npy
        # Wait, from my analysis of export script:
        # get_input_embeddings()[0] -> Group 1 -> codec_embedding_1.npy
        # So step 0 (predicts Group 1) uses codec_embedding_1.
        
        embed_tbl = embedding_tables[step + 1]
        code_embed = embed_tbl[code].reshape(1, 1, 2048)
        
        # 4. Accumulate for Sum
        generated_embeddings.append(code_embed)
        
        # 5. Prepare Input for Next Step
        # Next step input is just the code embedding of the current prediction
        current_input_embeds = code_embed

    print(f"  Generated Codes: {generated_codes}")
    print(f"  Official Codes (Preds): {official_final_codes[0, 1:]}")
    
    match_codes = np.array_equal(generated_codes, official_final_codes[0, 1:])
    print(f"  { '✅' if match_codes else '❌' } Codes Match")

    # 6. Verify Summation (Input to Master Backbone)
    print(f"[Assembly] Verifying Inputs for Next Master Step...")
    # Sum all 16 embeddings (Last ID + 15 Generated)
    summed_embeds = np.sum(generated_embeddings, axis=0) # [1, 1, 2048]
    
    # Load Official Backbone Input
    try:
        official_backbone_input = np.load(os.path.join(CAPTURED_DIR, "master_step_0_backbone_input.npy"))
    except FileNotFoundError:
        print("❌ Missing master_step_0_backbone_input.npy")
        return

    diff_sum = np.mean(np.abs(summed_embeds - official_backbone_input))
    cos_sum = np.dot(summed_embeds.flatten(), official_backbone_input.flatten()) / (
        np.linalg.norm(summed_embeds) * np.linalg.norm(official_backbone_input)
    )
    
    print(f"  Summed Embeds vs Official Backbone Input")
    pass_mark_sum = "✅" if diff_sum < 1e-3 and cos_sum > 0.999 else "⚠️"
    print(f"  {pass_mark_sum} MAE: {diff_sum:.6f}, CosSim: {cos_sum:.6f}")

    if match_codes and cos_sum > 0.999:
        print("\n结论: GGUF 大师 + ONNX 工匠 全链路自回归组装测试通过！")
    else:
        print("\n结论: 组装测试未完全通过。")
    
    if match and diff_master < 0.1: # Relaxed GGUF check due to BF16 diff
        print("\n结论: GGUF 大师 + ONNX 工匠 组装测试通过！")
    else:
        print("\n结论: 组装测试存在差异。")
    
    # Cleanup
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)

if __name__ == "__main__":
    run_assembly()
