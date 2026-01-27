import os
import sys
import ctypes
import numpy as np
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
FULL_GEN_DIR = os.path.join(PROJECT_ROOT, "captured_full_gen")

GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")
HEADS_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")

EOS_TOKEN_ID = 2150 # Based on logs

def run_full_pipeline():
    print("--- 全链路自回归流水线 (Full Pipeline Loop) ---")
    
    # 1. Load Data & Models
    # ---------------------
    print("[1/5] Loading Assets...")
    
    # Embeddings
    embedding_tables = {}
    for i in range(16):
        embedding_tables[i] = np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy"))
        
    master_head_weight = np.load(MASTER_HEAD_PATH) # [Vocab, Hidden] or [Hidden, Vocab]? Check shape.
    # Export was: codec_head.weight -> [2152, 2048] usually.
    # Linear layer weight is [Out, In]. So [Vocab, Hidden].
    print(f"  Master Head Shape: {master_head_weight.shape}")
    
    predictor_heads = np.load(HEADS_PATH)
    
    # Inputs (Start State)
    try:
        prefill_input = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
        trailing_text = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
        tts_pad = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)
    except FileNotFoundError as e:
        print(f"❌ Missing captured input: {e}")
        return

    # Official Truth
    try:
        official_codes = np.load(os.path.join(FULL_GEN_DIR, "full_generated_codes.npy"))
        print(f"  Official Truth Loaded: {len(official_codes)} steps")
    except:
        print("⚠️ Official truth not found, running in blind mode.")
        official_codes = None

    # Load Models
    print("[2/5] Initializing Engines...")
    # GGUF
    gguf_model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    if not gguf_model: return
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 4096 # Sufficient context
    ctx_params.embeddings = True
    gguf_ctx = nano_llama.llama_init_from_model(gguf_model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(gguf_model)
    
    # ONNX
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 2. Prefill Phase (GGUF)
    # -----------------------
    print("\n[3/5] Running Prefill...")
    n_tokens = prefill_input.shape[1]
    batch = nano_llama.llama_batch_init(4096, n_embd, 1) # Alloc large batch
    
    # Fill Batch
    batch.n_tokens = n_tokens
    full_embd = np.ascontiguousarray(prefill_input[0])
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    current_pos = 0
    for k in range(n_tokens):
        # Position Logic (Simple 1D for Text)
        pos = current_pos + k
        batch.pos[k] = pos
        batch.pos[n_tokens + k] = pos
        batch.pos[2 * n_tokens + k] = pos
        batch.pos[3 * n_tokens + k] = 0
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        batch.logits[k] = 1 if k == n_tokens - 1 else 0
    
    if nano_llama.llama_decode(gguf_ctx, batch) != 0:
        print("❌ Prefill Failed")
        return

    current_pos += n_tokens
    
    # Get initial hidden state
    out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
    result_embd = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
    current_hidden = result_embd[-1].copy().reshape(1, 1, n_embd) # [1, 1, 2048]
    
    # 3. Generation Loop
    # ------------------
    print(f"\n[4/5] Entering Generation Loop (Max 50 steps)...")
    
    generated_steps = []
    
    for step_idx in range(50):
        # A. Master Generate (Last ID)
        # linear(hidden) -> logits
        # master_head_weight: [Vocab, Hidden]. hidden: [1, 1, 2048]
        # logits = hidden @ weight.T
        master_logits = current_hidden[0] @ master_head_weight.T # [1, Vocab]
        last_id = np.argmax(master_logits)
        
        # Check EOS
        if last_id == EOS_TOKEN_ID:
            print(f"🛑 EOS Detected at step {step_idx}. Stopping.")
            break
            
        # Get Embedding for Last ID (Codec 0)
        last_id_embed = embedding_tables[0][last_id].reshape(1, 1, n_embd)
        
        # B. Craftsman Generate (15 Codes)
        craftsman_input_embeds = np.concatenate([current_hidden, last_id_embed], axis=1) # [1, 2, 2048]
        
        # Reset Craftsman Pasts
        current_pasts = {f"past_{i}": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(10)}
        
        step_codes = [last_id]
        step_embeds = [last_id_embed] # For sum
        
        for c_step in range(15):
            # ONNX Inference
            inputs = {'inputs_embeds': craftsman_input_embeds}
            inputs.update(current_pasts)
            outputs = onnx_sess.run(None, inputs)
            
            onnx_hidden = outputs[0]
            pasts = outputs[1:]
            
            # Update Pasts
            for i in range(5):
                current_pasts[f"past_{2*i}"] = pasts[2*i]
                current_pasts[f"past_{2*i+1}"] = pasts[2*i+1]
                
            # Predict Code (Group c_step+1)
            head_w = predictor_heads[c_step]
            logits = onnx_hidden[0, -1] @ head_w.T
            code = np.argmax(logits)
            
            step_codes.append(code)
            
            # Prepare next input (Embedding of this code)
            code_embed = embedding_tables[c_step + 1][code].reshape(1, 1, n_embd)
            step_embeds.append(code_embed)
            craftsman_input_embeds = code_embed

        generated_steps.append(step_codes)
        print(f"  Step {step_idx:02}: Codes {step_codes}")
        
        # C. Glue for Next Master Step
        # Sum 16 embeddings
        summed = np.sum(step_embeds, axis=0) # [1, 1, 2048]
        
        # Add Trailing/Pad
        if step_idx < trailing_text.shape[1]:
            addition = trailing_text[:, step_idx].reshape(1, 1, n_embd)
            summed += addition
        else:
            summed += tts_pad
            
        # D. Master Inference (Next Token)
        # Prepare Batch (1 token)
        batch.n_tokens = 1
        full_embd = np.ascontiguousarray(summed[0])
        ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
        
        pos = current_pos
        current_pos += 1
        
        batch.pos[0] = pos
        batch.pos[1] = pos
        batch.pos[2] = pos
        batch.pos[3] = 0 
        batch.logits[0] = 1 # We need output
        
        if nano_llama.llama_decode(gguf_ctx, batch) != 0:
            print(f"❌ Master Decode Failed at step {step_idx}")
            break
            
        # Update current_hidden for next loop
        out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
        result_embd = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd))
        current_hidden = result_embd[0].reshape(1, 1, n_embd)

    # 4. Verify Results
    # -----------------
    print("\n[5/5] Verifying...")
    final_generated = np.array(generated_steps)
    print(f"  Generated Shape: {final_generated.shape}")
    
    if official_codes is not None:
        print(f"  Official  Shape: {official_codes.shape}")
        
        min_len = min(len(final_generated), len(official_codes))
        match = np.array_equal(final_generated[:min_len], official_codes[:min_len])
        
        if match:
             print("\n✅ PERFECT MATCH! Full Pipeline Verified.")
        else:
             print("\n❌ Mismatch Detected.")
             # Show First Diff
             diff = final_generated[:min_len] != official_codes[:min_len]
             first_diff_idx = np.where(diff.any(axis=1))[0][0]
             print(f"  First mismatch at Step {first_diff_idx}")
             print(f"  Mine: {final_generated[first_diff_idx]}")
             print(f"  Auth: {official_codes[first_diff_idx]}")

    # Cleanup
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(gguf_ctx)
    nano_llama.llama_model_free(gguf_model)

if __name__ == "__main__":
    run_full_pipeline()
