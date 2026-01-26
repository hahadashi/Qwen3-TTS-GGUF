import os
import sys
import torch
import json
import shutil
from safetensors.torch import save_file
from safetensors import safe_open

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = PROJECT_ROOT # qwen_tts is in the root
sys.path.insert(0, SOURCE_DIR)

# 配置
SOURCE_MODEL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Bare-Master"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_bare_master():
    print(f"Starting Bare Master extraction to {OUTPUT_DIR}...")
    
    config_path = os.path.join(SOURCE_MODEL_DIR, "config.json")
    model_path = os.path.join(SOURCE_MODEL_DIR, "model.safetensors")
    
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    
    # 1. 构造 Bare Master Config
    # 我们保留 talker_config，但移除 code_predictor_config 以示主权
    bare_config = full_config.copy()
    talker_cfg = bare_config['talker_config'].copy()
    if 'code_predictor_config' in talker_cfg:
        del talker_cfg['code_predictor_config']
    
    # 关键：更新 vocab_size 为 151936 (padded)
    talker_cfg['vocab_size'] = 151936
    bare_config['talker_config'] = talker_cfg
    
    # 保存 config
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(bare_config, f, indent=2)
    print("Saved modified config.json")
    
    # 2. 提取并 Padding 权重
    bare_weights = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        # LLM 骨干 (保留 talker.model 前缀)
        for key in f.keys():
            if key.startswith("talker.model."):
                bare_weights[key] = f.get_tensor(key)
            elif key == "talker.text_projection.weight" or key == "talker.text_projection.bias":
                bare_weights[key] = f.get_tensor(key)
        
        # Padded Head: 从 3072 扩充到 151936
        head_weight = f.get_tensor("talker.codec_head.weight")
        hidden_size = head_weight.shape[1]
        padded_head = torch.zeros((151936, hidden_size), dtype=head_weight.dtype)
        padded_head[:3072, :] = head_weight
        bare_weights["talker.codec_head.weight"] = padded_head
        
        # 还要把 text_embedding 搬过来（如果它不是在 talker.model 下的话，实际上它在 talker.model.text_embedding）
        # 刚才循环已经涵盖了 talker.model.*
        
    print(f"Extracted {len(bare_weights)} tensors (with logit padding).")
    save_file(bare_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
    print("Saved model.safetensors")
    
    # 3. 复制模型定义代码 (可选，但方便以后加载)
    CODE_SRC = os.path.join(SOURCE_DIR, "qwen_tts", "core", "models")
    for fname in ["modeling_qwen3_tts.py", "configuration_qwen3_tts.py"]:
        shutil.copy(os.path.join(CODE_SRC, fname), os.path.join(OUTPUT_DIR, fname))
        print(f"Copied {fname}")

    print("\n✅ Bare Master saved successfully to model/Bare-Master")

if __name__ == "__main__":
    save_bare_master()
