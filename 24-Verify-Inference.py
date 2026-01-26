import torch
import numpy as np
from llama_cpp import Llama
import os

# Configuration
GGUF_PATH = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-F16.gguf"
SOURCE_MODEL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice"

def verify_inference():
    if not os.path.exists(GGUF_PATH):
        print(f"Error: GGUF file not found at {GGUF_PATH}")
        return

    print(f"--- [Script 4] Verifying Inference with llama-cpp-python ---")
    
    # 1. Load the model
    # We use n_gpu_layers=-1 to use GPU if available, or 0 for CPU
    print(f"Loading GGUF from {GGUF_PATH}...")
    llm = Llama(model_path=GGUF_PATH, n_gpu_layers=0, verbose=False)
    
    # 2. Prepare dummy input (batch_size=1, seq_len=1)
    # Token IDs for 'Hello' or similar, but for Talker we care about the codec ID space (0-3071)
    test_token_id = 100 
    
    print(f"Running inference with token ID: {test_token_id}...")
    # We use the raw llama_decode logic or just the __call__ for simplicity if it's treated as a text model
    # Note: llama-cpp-python's high-level API might try to use the tokenizer. 
    # For custom codec models, we might need to use the low-level API.
    
    # High level test (treat as completion)
    # output = llm("Hello", max_tokens=5)
    # print("High-level Output:", output)
    
    # Low level test (better for Talker hidden states)
    tokens = [test_token_id]
    llm.eval(tokens)
    
    # Extract hidden states (logits are easier to check first)
    logits = llm._logits
    print("Logits Shape (from llama-cpp):", np.array(logits).shape)
    print("Top 5 Logits:", np.array(logits)[0, :5])

    print("\n✅ Inference verification successful (Model reachable and producing logits).")

if __name__ == "__main__":
    verify_inference()
