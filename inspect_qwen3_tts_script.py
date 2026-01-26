import json
import os
from safetensors import safe_open

model_dir = r"c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice"
config_path = os.path.join(model_dir, "config.json")
model_path = os.path.join(model_dir, "model.safetensors")

print("--- Comprehensive Model Inspection ---")

try:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        
        def print_shape(name):
            if name in f.keys():
                print(f"{name}: {f.get_tensor(name).shape}")
            else:
                print(f"{name}: NOT FOUND")

        # 1. Verify GQA (Grouped Query Attention) and Head Dim
        print("\n[1] Attention Structure (Layer 0)")
        print_shape("talker.model.layers.0.self_attn.q_proj.weight")
        print_shape("talker.model.layers.0.self_attn.k_proj.weight")
        print_shape("talker.model.layers.0.self_attn.v_proj.weight")
        print_shape("talker.model.layers.0.self_attn.o_proj.weight")

        # 2. Verify MLP structure (SwiGLU/SilU)
        print("\n[2] MLP Structure (Layer 0)")
        print_shape("talker.model.layers.0.mlp.gate_proj.weight")
        print_shape("talker.model.layers.0.mlp.up_proj.weight")
        print_shape("talker.model.layers.0.mlp.down_proj.weight")

        # 3. Check for Code Predictor / MTP (Multi-Token Prediction)
        print("\n[3] Code Predictor / MTP Components")
        mtp_keys = [k for k in keys if "talker.code_predictor" in k or ".mtp." in k]
        print(f"Total MTP/Predictor Tensors: {len(mtp_keys)}")
        if mtp_keys:
            # Check the heads for different codebooks
            heads = [k for k in mtp_keys if "lm_head" in k or "head" in k]
            print(f"Prediction Heads: {heads[:10]}")
            if heads:
                print_shape(heads[0])

        # 4. Speaker Encoder
        print("\n[4] Speaker Encoder")
        spk_keys = [k for k in keys if "spk_encoder" in k]
        print(f"Total Speaker Encoder Tensors: {len(spk_keys)}")
        if spk_keys:
             # Check for ECAPA-TDNN or similar structure
             print(f"Example Speaker Tensors: {spk_keys[:5]}")
             print_shape(spk_keys[0])

        # 5. Embeddings
        print("\n[5] Embeddings")
        print_shape("talker.model.text_embedding.weight")
        print_shape("talker.model.codec_embedding.weight")

except Exception as e:
    print(f"Error: {e}")
