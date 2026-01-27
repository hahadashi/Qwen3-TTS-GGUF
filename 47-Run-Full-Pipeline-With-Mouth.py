import os
import sys
import ctypes
import numpy as np
import soundfile as sf
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_audio")
os.makedirs(SAVE_DIR, exist_ok=True)

GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
ONNX_PREDICTOR_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")
ONNX_DECODER_PATH = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")
HEADS_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")

EOS_TOKEN_ID = 2150 

def run_pipeline_with_mouth():
    print("--- 全链路语音合成 (Full Pipeline with Mouth) ---")
    
    # 1. Load Data & Models
    print("[1/6] Loading Assets...")
    
    embedding_tables = {}
    for i in range(16):
        embedding_tables[i] = np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy"))
        
    master_head_weight = np.load(MASTER_HEAD_PATH)
    predictor_heads = np.load(HEADS_PATH)
    
    # Inputs
    try:
        prefill_input = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
        trailing_text = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
        tts_pad = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)
    except FileNotFoundError as e:
        print(f"❌ Missing captured input: {e}")
        return

    # Init Models
    # GGUF
    print("[2/6] Init Master (GGUF)...")
    gguf_model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 4096 
    ctx_params.embeddings = True
    gguf_ctx = nano_llama.llama_init_from_model(gguf_model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(gguf_model) # 2048
    
    # ONNX Predictor
    print("[3/6] Init Craftsman (ONNX)...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    predictor_sess = ort.InferenceSession(ONNX_PREDICTOR_PATH, sess_opts, providers=['CPUExecutionProvider'])
    
    # ONNX Decoder
    print("[4/6] Init Mouth (ONNX)...")
    decoder_sess = ort.InferenceSession(ONNX_DECODER_PATH, sess_opts, providers=['CPUExecutionProvider'])

    # -----------------------
    # PIPELINE START
    # -----------------------
    
    # Prefill
    print("\n[5/6] Generating Codes...")
    n_tokens = prefill_input.shape[1]
    batch = nano_llama.llama_batch_init(4096, n_embd, 1)
    
    batch.n_tokens = n_tokens
    full_embd = np.ascontiguousarray(prefill_input[0])
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    current_pos = 0
    for k in range(n_tokens):
        pos = current_pos + k
        batch.pos[k] = pos
        batch.pos[n_tokens + k] = pos # Temporal, H, W positions (Simplified as 1D+Zero for text)
        batch.pos[2 * n_tokens + k] = pos
        batch.pos[3 * n_tokens + k] = 0
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        batch.logits[k] = 1 if k == n_tokens - 1 else 0
    
    nano_llama.llama_decode(gguf_ctx, batch)
    current_pos += n_tokens
    
    out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
    result_embd = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
    current_hidden = result_embd[-1].copy().reshape(1, 1, n_embd)
    
    # Generation (Accumulate Codes)
    full_code_sequence = []
    
    for step_idx in range(50):
        # A. Master Predict
        master_logits = current_hidden[0] @ master_head_weight.T
        last_id = np.argmax(master_logits)
        
        if last_id == EOS_TOKEN_ID:
            print(f"🛑 EOS Detected at step {step_idx}. Generation Finished.")
            break
            
        last_id_embed = embedding_tables[0][last_id].reshape(1, 1, n_embd)
        
        # B. Craftsman Predict
        craftsman_input = np.concatenate([current_hidden, last_id_embed], axis=1)
        current_pasts = {f"past_{i}": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(10)}
        
        step_codes = [last_id]
        step_embeds = [last_id_embed]
        
        for c_step in range(15):
            inputs = {'inputs_embeds': craftsman_input}
            inputs.update(current_pasts)
            outputs = predictor_sess.run(None, inputs)
            
            onnx_hidden = outputs[0]
            pasts = outputs[1:]
            
            for i in range(5):
                current_pasts[f"past_{2*i}"] = pasts[2*i]
                current_pasts[f"past_{2*i+1}"] = pasts[2*i+1]
                
            head_w = predictor_heads[c_step]
            logits = onnx_hidden[0, -1] @ head_w.T
            code = np.argmax(logits)
            
            step_codes.append(code)
            
            code_embed = embedding_tables[c_step + 1][code].reshape(1, 1, n_embd)
            step_embeds.append(code_embed)
            craftsman_input = code_embed
            
        full_code_sequence.append(step_codes)
        
        # C. Glue
        summed = np.sum(step_embeds, axis=0) # [1, 1, 2048]
        if step_idx < trailing_text.shape[1]:
            summed += trailing_text[:, step_idx].reshape(1, 1, n_embd)
        else:
            summed += tts_pad
            
        # D. Master Next
        batch.n_tokens = 1
        full_embd = np.ascontiguousarray(summed[0])
        ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
        
        batch.pos[0] = current_pos
        batch.pos[1] = current_pos
        batch.pos[2] = current_pos
        batch.pos[3] = 0 
        batch.logits[0] = 1
        
        nano_llama.llama_decode(gguf_ctx, batch)
        current_pos += 1
        
        out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
        result_embd = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd))
        current_hidden = result_embd[0].reshape(1, 1, n_embd)

    # Decode Audio
    print(f"\n[6/6] Decoding Audio with Mouth (Mouth)...")
    if len(full_code_sequence) == 0:
        print("❌ No codes generated.")
        return
        
    # Shape: [N_Steps, 16] -> Transpose/Reshape as needed for Decoder
    # Decoder expects [1, N_Frames] where Frame is 1 code? No.
    # Check 44-Verify-Mouth: inputs={'input_ids': codes} where codes is [1, T] 
    # But wait, decoder takes sequence of codes.
    # Qwen3TTSDecoder forward(self, input_ids): input_ids [B, T] ? 
    # Actually Qwen3TTSDecoder expects input_ids [B, T] where T is timesteps.
    # But wait, we have 16 layers.
    # speech_tokenizer.model.decoder(codes) -> codes shape [1, 16, T] ??
    # Verify logic in 44-Verify-Mouth-ONNX.py or 12-Export-Codec-Decoder.py
    
    # From 12-Export: dummy_input = torch.randint(0, 1024, (1, 128)) ? 
    # No, speech_tokenizer is causal/conv1d?
    # Let's check 44 codes shape. official_codes shape was (1, 16, 212).
    # So we need to transpose: [Steps, 16] -> [16, Steps] -> [1, 16, Steps]
    
    codes_np = np.array(full_code_sequence) # [Steps, 16]
    print(f"  Generated Codes Shape: {codes_np.shape}")
    
    # Decoder expects [1, Steps, 16] inferred from error (dim 2 must be 16)
    codes_input = codes_np[np.newaxis, ...].astype(np.int64) # [1, Steps, 16]
    
    print(f"  Decoder Input Shape: {codes_input.shape}")
    
    audio_out = decoder_sess.run(None, {'audio_codes': codes_input})[0] # [1, 1, Samples] ???
    # Check 44: audio outputs shape. 
    # Official output usually [1, 1, Samples] or [1, Samples]
    
    audio_data = audio_out.squeeze()
    
    SAVE_PATH = os.path.join(SAVE_DIR, "onnx_full.wav")
    sf.write(SAVE_PATH, audio_data, 24000)
    print(f"✅ 全链路合成音频已保存: {SAVE_PATH}")
    
    # Cleanup
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(gguf_ctx)
    nano_llama.llama_model_free(gguf_model)

if __name__ == "__main__":
    run_pipeline_with_mouth()
