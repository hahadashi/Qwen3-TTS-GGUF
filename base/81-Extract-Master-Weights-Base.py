import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def extract_master_weights_base():
    """从 Base 模型中提取 LLM Master 权重"""

    # 路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-Base")
    ORIGINAL_WEIGHTS = os.path.join(ORIGINAL_MODEL_PATH, "model.safetensors")
    ORIGINAL_CONFIG = os.path.join(ORIGINAL_MODEL_PATH, "config.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model-base", "hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model.safetensors")
    OUTPUT_CONFIG = os.path.join(OUTPUT_DIR, "config.json")

    print(f"--- 正在提取 Base 大师模型权重 ---")
    print(f"源路径: {ORIGINAL_WEIGHTS}")

    # 1. 读取原始配置
    with open(ORIGINAL_CONFIG, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    # 提取 talker_config
    master_config = original_config['talker_config'].copy()

    # 纯音频词表方案：只用 codec_embedding (3072)
    master_config['vocab_size'] = 3072
    master_config['_original_model_type'] = 'qwen3_tts_talker'
    master_config['_extracted_from'] = 'Qwen3-TTS-12Hz-1.7B-Base'
    master_config['_actual_model_type'] = 'qwen3_tts_talker'
    master_config['_codec_only_vocab'] = True

    # 伪装成 Qwen3-VL 架构
    master_config['architectures'] = ['Qwen3VLForConditionalGeneration']
    master_config['model_type'] = 'qwen3_vl'

    # 2. 提取权重
    master_weights = {}
    codec_head_weight = None

    with safe_open(ORIGINAL_WEIGHTS, framework="pt", device="cpu") as f:
        for key in f.keys():
            # LLM Backbone: talker.model.*
            if key.startswith("talker.model."):
                new_key = key.replace("talker.model.", "")
                tensor = f.get_tensor(key)

                # 特殊处理：Codec Embedding (0-3071)
                if key == "talker.model.codec_embedding.weight":
                    master_weights["embed_tokens.weight"] = tensor
                    print(f"  ✓ 提取并更名: {key} -> embed_tokens.weight")
                    continue

                # 跳过 Text Embedding (纯音频方案不需要)
                if key == "talker.model.text_embedding.weight":
                    continue

                master_weights[new_key] = tensor

            # Codec Head: talker.codec_head.weight
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  ✓ 提取 Codec Head: {key}")

    # 3. 创建 lm_head
    if codec_head_weight is not None:
        master_weights["lm_head.weight"] = codec_head_weight.contiguous()
        print(f"  ✓ 设置 lm_head 为实时的 Codec Head")
    else:
        print("❌ 错误: 未能在模型中找到 talker.codec_head.weight")
        return

    # 4. 保存
    save_file(master_weights, OUTPUT_WEIGHTS)
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 提取完成! 权重保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_master_weights_base()
