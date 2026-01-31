"""
从原始的 Qwen3-TTS 模型中提取"大师"(LLM Backbone)的权重
将大师权重保存为独立的 safetensors 文件

纯音频词表方案 (Codec-Only Vocab):
- 只使用 codec_embedding [3072, 2048] 作为 embedding
- vocab_size = 3072 (codec tokens only)
- lm_head = codec_head.T [2048, 3072] (无冗余)
- 输入/输出都是 codec token IDs (0-3071)

优势:
- 更小的模型 (lm_head 从 [2048, 155008] 减少到 [2048, 3072])
- 更快的推理 (logits 计算量减少 98%)
- 更简洁的架构 (无需文本 tokenizer)

"""
import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from export_config import MODEL_DIR, EXPORT_DIR

def extract_master_weights():
    """提取大师权重到独立文件"""

    # 路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_MODEL_PATH = MODEL_DIR
    ORIGINAL_WEIGHTS = os.path.join(ORIGINAL_MODEL_PATH, "model.safetensors")
    ORIGINAL_CONFIG = os.path.join(ORIGINAL_MODEL_PATH, "config.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(EXPORT_DIR, "hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model.safetensors")
    OUTPUT_CONFIG = os.path.join(OUTPUT_DIR, "config.json")

    print(f"--- 正在提取大师模型权重 ---")
    print(f"正在从以下路径加载原始权重: {ORIGINAL_WEIGHTS}")

    # 1. 读取原始配置
    with open(ORIGINAL_CONFIG, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    # 提取 talker_config 作为大师的配置
    master_config = original_config['talker_config'].copy()

    # 纯音频词表方案：只用 codec_embedding
    master_config['vocab_size'] = 3072  # codec_vocab_size
    master_config['_original_model_type'] = 'qwen3_tts_talker'
    master_config['_extracted_from'] = 'Qwen3-TTS-12Hz-1.7B-CustomVoice'
    master_config['_actual_model_type'] = 'qwen3_tts_talker'
    master_config['_codec_only_vocab'] = True  # 标记为纯音频词表

    # 伪装成 Qwen3-VL 架构（llama.cpp 已支持 IMRoPE）
    master_config['architectures'] = ['Qwen3VLForConditionalGeneration']
    master_config['model_type'] = 'qwen3_vl'

    # 2. 提取大师权重 (talker.model.* -> 无前缀)
    master_weights = {}
    codec_head_weight = None

    with safe_open(ORIGINAL_WEIGHTS, framework="pt", device="cpu") as f:
        total_keys = len(list(f.keys()))
        print(f"原始模型中的总权重键数: {total_keys}")

        for key in f.keys():
            # 大师 backbone 权重: talker.model.*
            if key.startswith("talker.model."):
                # 变成无前缀格式 (如 layers.0.xxx)，这样 transformers 才能直接加载
                new_key = key.replace("talker.model.", "")
                tensor = f.get_tensor(key)

                # 特殊处理：只提取 codec_embedding (纯音频方案)
                if key == "talker.model.codec_embedding.weight":
                    print(f"  已提取 codec embedding: {key}, 形状: {tensor.shape}")
                    # 保存为 embed_tokens (标准 HF 命名)
                    master_weights["embed_tokens.weight"] = tensor
                    print(f"  -> 保存为 embed_tokens.weight [3072, 2048]")
                    continue

                # 跳过 text_embedding (纯音频方案不需要)
                if key == "talker.model.text_embedding.weight":
                    print(f"  跳过文本 embedding: {key}, 形状: {tensor.shape} (纯音频方案不需要)")
                    continue

                master_weights[new_key] = tensor
                print(f"  已提取: {key} -> {new_key}, 形状: {tensor.shape}")

            # Codec head 权重: talker.codec_head.weight
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  已提取: {key}, 形状: {codec_head_weight.shape}")

            # 跳过其他权重 (code_predictor, speaker_encoder 等)
            else:
                print(f"  已跳过: {key}")

    print(f"\n成功提取 {len(master_weights)} 个大师模型权重")

    # 3. 创建 lm_head (纯音频方案：直接用 codec_head)
    if codec_head_weight is not None:
        print(f"\n--- 正在创建 lm_head (纯音频方案) ---")
        print(f"Codec head 形状: {codec_head_weight.shape}")  # [3072, 2048]

        # 直接使用原权重作为 lm_head，遵循 HF 标准形状 [3072, 2048]
        lm_head = codec_head_weight.contiguous()
        print(f"lm_head 形状: {lm_head.shape}")
        print(f"  - 直接使用 codec_head (无需转置或填充)")
        print(f"  - 词表大小: {lm_head.shape[0]} (仅限 codec tokens)")

        # 保存为 lm_head.weight (HF 标准命名)
        master_weights["lm_head.weight"] = lm_head
        print(f"已添加 lm_head.weight 用于纯音频推理 (标准 HF 形状)")
    else:
        print("警告: 未找到 codec_head，模型将缺少 lm_head!")

    # 4. 保存大师权重
    save_file(master_weights, OUTPUT_WEIGHTS)
    print(f"大师模型权重已保存至: {OUTPUT_WEIGHTS}")

    # 5. 保存大师配置
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    print(f"大师模型配置已保存至: {OUTPUT_CONFIG}")

    # 6. 生成模型信息
    print(f"\n--- 正在创建元数据 (Metadata) ---")
    metadata = {
        "model_type": "Qwen3TTSTalkerModel",
        "architecture": "Master-only (LLM Backbone) with Codec-Only Vocab",
        "source_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "extraction_date": "2026-01-27",
        "vocab_strategy": "codec_only",
        "components": {
            "master": "LLM Backbone (28 layers, 2048 hidden)",
            "embed_tokens": "Codec embedding [3072, 2048] (codec tokens only, no text)",
            "lm_head": "Codec head transpose [2048, 3072] (direct, no padding)",
            "vocab_size": 3072,
            "note": "Input/Output are codec token IDs (0-3071)"
        },
        "usage": {
            "gguf_conversion": "python 84-Convert-Master-to-GGUF.py",
            "load_with": "from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel",
            "config_class": "from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig",
            "llamacpp_compatible": "伪装成 Qwen3-VL，纯音频词表，无文本 tokenizer",
            "inference": "Input codec token IDs directly, output codec token IDs"
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存至: {metadata_path}")

    print("\n✅ 提取完成!")
    print(f"大师模型已保存至: {OUTPUT_DIR}")
    print(f"  - model.safetensors ({os.path.getsize(OUTPUT_WEIGHTS) / 1024**3:.2f} GB)")
    print(f"    └─ embed_tokens [3072, 2048] (仅音频)")
    print(f"    └─ lm_head [2048, 3072] (仅音频)")
    print(f"    └─ 28 层 + norm")
    print(f"  - config.json (vocab_size=3072)")
    print(f"  - metadata.json")

if __name__ == "__main__":
    extract_master_weights()
