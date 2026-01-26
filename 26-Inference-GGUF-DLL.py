import sys
import os
import ctypes
import numpy as np

# =========================================================================
# Configuration
# =========================================================================
DLL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\qwen3_tts_gguf\bin"
MODEL_PATH = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-F16.gguf"

# =========================================================================
# Llama.cpp Ctypes Definitions
# =========================================================================
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

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

def main():
    print(f"--- [Script 26] GGUF Inference via DLL ---")
    
    # 1. Load Dynamic Libraries
    print(f"Loading DLLs from {DLL_DIR}...")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(DLL_DIR)
    
    try:
        ggml_base = ctypes.CDLL(os.path.join(DLL_DIR, "ggml-base.dll"))
        ggml = ctypes.CDLL(os.path.join(DLL_DIR, "ggml.dll"))
        llama = ctypes.CDLL(os.path.join(DLL_DIR, "llama.dll"))
        print("DLLs loaded successfully.")
    except Exception as e:
        print(f"Error loading DLLs: {e}")
        return

    # Define function signatures for ggml
    ggml.ggml_backend_load_all.argtypes = []
    ggml.ggml_backend_load_all.restype = None

    # Define function signatures for llama
    llama.llama_backend_init.argtypes = []
    llama.llama_backend_init.restype = None
    
    llama.llama_backend_free.argtypes = []
    llama.llama_backend_free.restype = None
    
    llama.llama_model_default_params.argtypes = []
    llama.llama_model_default_params.restype = llama_model_params
    
    llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama.llama_model_load_from_file.restype = ctypes.c_void_p
    
    llama.llama_model_free.argtypes = [ctypes.c_void_p]
    llama.llama_model_free.restype = None
    
    llama.llama_context_default_params.argtypes = []
    llama.llama_context_default_params.restype = llama_context_params
    
    llama.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama.llama_init_from_model.restype = ctypes.c_void_p
    
    llama.llama_free.argtypes = [ctypes.c_void_p]
    llama.llama_free.restype = None
    
    llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama.llama_batch_init.restype = llama_batch
    
    llama.llama_batch_free.argtypes = [llama_batch]
    llama.llama_batch_free.restype = None
    
    llama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama.llama_decode.restype = ctypes.c_int32
    
    llama.llama_get_logits.argtypes = [ctypes.c_void_p]
    llama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Initialize backend
    print("Initializing backends...")
    original_cwd = os.getcwd()
    os.chdir(DLL_DIR)
    try:
        ggml.ggml_backend_load_all()
        llama.llama_backend_init()
    finally:
        os.chdir(original_cwd)

    # 2. Setup Parameters
    print("Setting parameters...")
    m_params = llama.llama_model_default_params()
    m_params.n_gpu_layers = 99
    
    c_params = llama.llama_context_default_params()
    c_params.n_ctx = 512
    c_params.n_batch = 512
    
    # 3. Load Model
    print(f"Loading model: {MODEL_PATH}")
    model = llama.llama_model_load_from_file(MODEL_PATH.encode("utf-8"), m_params)
    if not model:
        print("Error: Failed to load model.")
        return

    # 4. Create Context
    ctx = llama.llama_init_from_model(model, c_params)
    if not ctx:
        print("Error: Failed to create context.")
        return

    # 5. Tokenize Input
    test_tokens = [1, 256, 123]
    n_tokens = len(test_tokens)
    
    # 6. Prepare Batch
    batch = llama.llama_batch_init(n_tokens, 0, 1)
    batch.n_tokens = n_tokens  # Crucial: Set the active token count
    for i, t_id in enumerate(test_tokens):
        batch.token[i] = t_id
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = (i == n_tokens - 1)

    # 7. Decode
    print(f"Decoding {n_tokens} tokens...")
    ret = llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"Error: llama_decode failed with code {ret}")
        return

    # 8. Get Logits
    logits_ptr = llama.llama_get_logits(ctx)
    vocab_size = 151936 
    logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()
    
    print("\n[Inference Result]")
    print(f"Logits shape: {logits.shape}")
    print(f"Top 5 Logit values: {logits[:5]}")
    print(f"Max Logit: {np.max(logits)} at index {np.argmax(logits)}")

    # 9. Cleanup
    llama.llama_batch_free(batch)
    llama.llama_free(ctx)
    llama.llama_model_free(model)
    llama.llama_backend_free()
    
    print("\n✅ Script 26 complete! DLL-based inference verified.")

if __name__ == "__main__":
    main()
