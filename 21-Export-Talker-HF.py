import torch
import os
import json
import shutil

# Configuration
SOURCE_MODEL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_HF_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-HF"
os.makedirs(OUTPUT_HF_DIR, exist_ok=True)

def export_talker_hf():
    print(f"[1/4] Loading Qwen3-TTS config and weights...")
    config_path = os.path.join(SOURCE_MODEL_DIR, "config.json")
    model_path = os.path.join(SOURCE_MODEL_DIR, "model.safetensors")
    
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    
    # 1. Prepare Qwen3-VL configuration for Talker
    talker_config_base = full_config.get("talker_config", {})
    
    # Based on Experience/02-Qwen3-Talker-GGUF-Export.md
    hf_config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "hidden_size": talker_config_base.get("hidden_size", 2048),
        "intermediate_size": talker_config_base.get("intermediate_size", 6144),
        "num_attention_heads": talker_config_base.get("num_attention_heads", 16),
        "num_hidden_layers": talker_config_base.get("num_hidden_layers", 28),
        "num_key_value_heads": talker_config_base.get("num_key_value_heads", 8),
        "max_position_embeddings": talker_config_base.get("max_position_embeddings", 32768),
        "rms_norm_eps": talker_config_base.get("rms_norm_eps", 1e-06),
        "rope_theta": talker_config_base.get("rope_theta", 1000000.0),
        "vocab_size": 151936, # Standard Qwen2.5/3 vocab size for GGUF compatibility
        "tie_word_embeddings": False,
        "use_cache": True,
        "rope_scaling": {
            "mrope_section": [24, 20, 20],
            "type": "mrope"
        },
        "vision_config": {
            "deepstack_visual_indexes": [1, 2, 3] # Required by some Qwen3-VL scripts to avoid errors
        }
    }
    
    hf_config["hidden_act"] = "silu"

    print(f"[2/4] Saving config.json to {OUTPUT_HF_DIR}...")
    with open(os.path.join(OUTPUT_HF_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # 2. Extract Talker weights
    print(f"[3/4] Extracting weights from safetensors...")
    from safetensors import safe_open
    from safetensors.torch import save_file

    talker_weights = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Standard Transformer Layers
            if key.startswith("talker.model.layers.") or key.startswith("talker.model.norm."):
                # talker.model.layers.0.self_attn.k_norm.weight -> model.layers.0.self_attn.k_norm.weight
                new_key = key.replace("talker.model.", "model.")
                talker_weights[new_key] = f.get_tensor(key)
            
            # Heads - Qwen3-VL logic in conversion script maps 'lm_head.weight'
            elif key.startswith("talker.codec_head."):
                # Based on experience document: pad to 151936
                weight = f.get_tensor(key)
                hidden_size = weight.shape[1]
                padded_weight = torch.zeros((151936, hidden_size), dtype=weight.dtype)
                padded_weight[:weight.shape[0], :] = weight
                talker_weights["lm_head.weight"] = padded_weight
            
            # Embeddings
            elif key == "talker.model.codec_embedding.weight":
                # Similarly pad embeddings
                weight = f.get_tensor(key)
                hidden_size = weight.shape[1]
                padded_weight = torch.zeros((151936, hidden_size), dtype=weight.dtype)
                padded_weight[:weight.shape[0], :] = weight
                talker_weights["model.embed_tokens.weight"] = padded_weight

    print(f"Extracted {len(talker_weights)} tensors.")
    
    # 3. Save as safetensors in HF directory
    print(f"[4/4] Saving weights to {OUTPUT_HF_DIR}...")
    save_file(talker_weights, os.path.join(OUTPUT_HF_DIR, "model.safetensors"))
    
    # 4. Copy tokenizer and other required files
    print(f"Copying tokenizer files to {OUTPUT_HF_DIR}...")
    files_to_copy = [
        "tokenizer.json", 
        "tokenizer_config.json", 
        "vocab.json", 
        "merges.txt", 
        "generation_config.json"
    ]
    for filename in files_to_copy:
        src = os.path.join(SOURCE_MODEL_DIR, filename)
        dst = os.path.join(OUTPUT_HF_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Copied {filename}")
        else:
            # Also check subdirectories? Qwen3-TTS usually has them in the root of the checkpoint
            pass
    
    print("✅ Done! Talker exported to HF format (Qwen3-VL style).")

if __name__ == "__main__":
    export_talker_hf()
