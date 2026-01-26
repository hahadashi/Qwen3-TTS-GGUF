
import sys
import os
import ctypes
import numpy as np
import torch
from safetensors.torch import load_file

# =========================================================================
# Configuration
# =========================================================================
DLL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\qwen3_tts_gguf\bin"
GGUF_MODEL_PATH = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-F16.gguf"
PT_MODEL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice"

# =========================================================================
# Part 1: PyTorch Inference (Simulation/Partial Loading for speed and stability)
# =========================================================================
# Note: Loading the full custom model class might require complex dependencies.
# We will manually simulate the Embed -> Projection -> First Layer (approx) or just check dimensions and initial projection.
# However, to compare logits, we need the full transformer.
# Let's try to import the actual model class if possible, otherwise we warn.

def get_pytorch_output(input_ids):
    print("\n--- [PyTorch] Loading Original Talker Components ---")
    sys.path.append(os.path.join(os.getcwd(), "Qwen3-TTS"))
    
    try:
        # Try to load weights manually to avoid full class dependency hell if unnecessary
        # But for full logical verification, we really need the class. 
        # Let's try a minimal approach: Load safetensors and manually inspect/run if possible, 
        # or use the class if imports work.
        
        # Given the user environment, let's try importing the class first.
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerConfig, Qwen3TTSTalkerTextPreTrainedModel, Qwen3TTSTalkerAttention
        # Wait, the main model is Qwen3TTSForConditionalGeneration which contains Talker.
        # But we exported just the Talker to GGUF.
        
        # Let's verify if we can just load the Talker weights into a Qwen2VL-ish structure 
        # or if we must use the original code.
        # Ideally, we used 21-Export to standard HF Qwen2VL format.
        # So we can try loading that HF model using transformers!
        # That would be the "Reference HF" implementation.
        
        print("Loading exported HF model (from script 21 output) using transformers...")
        from transformers import AutoModelForCausalLM, AutoConfig
        
        hf_export_dir = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-HF"
        if not os.path.exists(hf_export_dir):
            print(f"Error: Exported HF model not found at {hf_export_dir}. Run 21-Export-Talker-HF.py first.")
            return None

        # Load with transformers (Qwen2VL structure)
        # Note: We need to trust remote code or ensure Qwen2VL is supported. 
        # Qwen3-TTS exported config claims "Qwen3VLForConditionalGeneration"
        # Since we might not have that class registered in transformers local, 
        # this might fall back to Qwen2VL or error.
        
        # Fallback: Load the original weights and manually project to verify the GGUF input
        print("Falling back to manual weight inspection of Original Safetensors...")
        weights = load_file(os.path.join(PT_MODEL_DIR, "model.safetensors"))
        
        # 1. Get Embeddings and Projection
        token_emb = weights["talker.model.text_embedding.weight"] # [vocab, 2048]
        fc1_w = weights["talker.text_projection.linear_fc1.weight"]
        fc1_b = weights["talker.text_projection.linear_fc1.bias"]
        fc2_w = weights["talker.text_projection.linear_fc2.weight"]
        fc2_b = weights["talker.text_projection.linear_fc2.bias"]
        
        # 2. Run Projection
        print("Running manual Text Projection...")
        # Inputs
        x_ids = torch.tensor(input_ids).long()
        x_emb = torch.nn.functional.embedding(x_ids, token_emb)
        
        # Projection: fc2(silu(fc1(x)))
        h = torch.nn.functional.linear(x_emb, fc1_w, fc1_b)
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, fc2_w, fc2_b)
        
        print(f"Projected Embedding Mean: {h.float().mean().item():.6f}")
        print(f"Projected Embedding Std:  {h.float().std().item():.6f}")
        return h.detach().float().numpy() # Return the embeddings for comparison?
        
        # NOTE: Comparing full logits requires running 28 layers of Transformers. 
        # Without a running PyTorch model instance, this is hard.
        # We will stop at the projected embedding (the GGUF Input) and verifying the Head (GGUF Output) structure.
        
    except Exception as e:
        print(f"PyTorch Load Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# =========================================================================
# Part 2: GGUF Inference (DLL)
# =========================================================================
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

class llama_model_params(ctypes.Structure):
    _fields_ = [("devices", ctypes.c_void_p), ("tensor_buft_overrides", ctypes.c_void_p), ("n_gpu_layers", ctypes.c_int32), ("split_mode", ctypes.c_int32), ("main_gpu", ctypes.c_int32), ("tensor_split", ctypes.POINTER(ctypes.c_float)), ("progress_callback", ctypes.c_void_p), ("progress_callback_user_data", ctypes.c_void_p), ("kv_overrides", ctypes.c_void_p), ("vocab_only", ctypes.c_bool), ("use_mmap", ctypes.c_bool), ("use_direct_io", ctypes.c_bool), ("use_mlock", ctypes.c_bool), ("check_tensors", ctypes.c_bool), ("use_extra_bufts", ctypes.c_bool), ("no_host", ctypes.c_bool), ("no_alloc", ctypes.c_bool)]

class llama_context_params(ctypes.Structure):
    _fields_ = [("n_ctx", ctypes.c_uint32), ("n_batch", ctypes.c_uint32), ("n_ubatch", ctypes.c_uint32), ("n_seq_max", ctypes.c_uint32), ("n_threads", ctypes.c_int32), ("n_threads_batch", ctypes.c_int32), ("rope_scaling_type", ctypes.c_int32), ("pooling_type", ctypes.c_int32), ("attention_type", ctypes.c_int32), ("flash_attn_type", ctypes.c_int32), ("rope_freq_base", ctypes.c_float), ("rope_freq_scale", ctypes.c_float), ("yarn_ext_factor", ctypes.c_float), ("yarn_attn_factor", ctypes.c_float), ("yarn_beta_fast", ctypes.c_float), ("yarn_beta_slow", ctypes.c_float), ("yarn_orig_ctx", ctypes.c_uint32), ("defrag_thold", ctypes.c_float), ("cb_eval", ctypes.c_void_p), ("cb_eval_user_data", ctypes.c_void_p), ("type_k", ctypes.c_int32), ("type_v", ctypes.c_int32), ("abort_callback", ctypes.c_void_p), ("abort_callback_data", ctypes.c_void_p), ("embeddings", ctypes.c_bool), ("offload_kqv", ctypes.c_bool), ("no_perf", ctypes.c_bool), ("op_offload", ctypes.c_bool), ("swa_full", ctypes.c_bool), ("kv_unified", ctypes.c_bool), ("samplers", ctypes.c_void_p), ("n_samplers", ctypes.c_size_t)]

def get_gguf_logits(input_ids):
    print("\n--- [GGUF] Running Inference via DLL ---")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(DLL_DIR)
    
    try:
        llama = ctypes.CDLL(os.path.join(DLL_DIR, "llama.dll"))
        ggml = ctypes.CDLL(os.path.join(DLL_DIR, "ggml.dll"))
        ggml.ggml_backend_load_all()
        llama.llama_backend_init()
    except Exception as e:
        print(f"DLL Load Error: {e}")
        return None


    # Initialize backend
    print("Initializing backends...")
    original_cwd = os.getcwd()
    os.chdir(DLL_DIR)
    try:
        if hasattr(ggml, 'ggml_backend_load_all'):
            ggml.ggml_backend_load_all()
        if hasattr(llama, 'llama_backend_init'):
            llama.llama_backend_init()
    except Exception as e:
        print(f"Backend Init Warning: {e}")
    finally:
        os.chdir(original_cwd)

    # Define minimal signatures
    llama.llama_model_default_params.restype = llama_model_params
    llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama.llama_model_load_from_file.restype = ctypes.c_void_p
    llama.llama_model_free.argtypes = [ctypes.c_void_p]
    
    llama.llama_context_default_params.restype = llama_context_params
    llama.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama.llama_init_from_model.restype = ctypes.c_void_p
    llama.llama_free.argtypes = [ctypes.c_void_p]
    
    llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama.llama_batch_init.restype = llama_batch
    llama.llama_batch_free.argtypes = [llama_batch]
    
    llama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama.llama_decode.restype = ctypes.c_int32
    
    llama.llama_get_logits.argtypes = [ctypes.c_void_p]
    llama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Load
    m_params = llama.llama_model_default_params()
    m_params.n_gpu_layers = 99
    c_params = llama.llama_context_default_params()
    c_params.n_ctx = 2048
    c_params.n_batch = 512
    
    model = llama.llama_model_load_from_file(GGUF_MODEL_PATH.encode("utf-8"), m_params)
    if not model: 
        print("Failed to load GGUF model.")
        return None
    ctx = llama.llama_init_from_model(model, c_params)
    if not ctx: 
        print("Failed to create context.")
        return None

    # Batch
    n_tokens = len(input_ids)
    batch = llama.llama_batch_init(n_tokens, 0, 1)
    batch.n_tokens = n_tokens
    for i, t_id in enumerate(input_ids):
        batch.token[i] = t_id
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = (i == n_tokens - 1)
        
    # Decode
    ret = llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"Decode failed: {ret}")
        return None

    # Logits
    logits_ptr = llama.llama_get_logits(ctx)
    # We padded vocab to 151936
    logits = np.ctypeslib.as_array(logits_ptr, shape=(151936,)).copy()
    
    llama.llama_batch_free(batch)
    llama.llama_free(ctx)
    llama.llama_model_free(model)
    
    return logits

def main():
    test_input = [1, 2055, 123] # Example inputs
    print(f"Test Input: {test_input}")
    
    gguf_logits = get_gguf_logits(test_input)
    pt_proj_emb = get_pytorch_output(test_input)
    
    if gguf_logits is not None:
        print("\n[GGUF Logits Analysis]")
        print(f"Shape: {gguf_logits.shape}")
        print(f"Mean: {np.mean(gguf_logits):.4f}")
        print(f"Max:  {np.max(gguf_logits):.4f}")
        
    if pt_proj_emb is not None:
        pass
        
if __name__ == "__main__":
    main()
