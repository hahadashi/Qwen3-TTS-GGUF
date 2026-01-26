"""
在不修改 convert_hf_to_gguf.py 的情况下，将大师模型转换为 GGUF

策略：
1. 创建一个伪装的配置文件，让转换器认为这是 Qwen3VLForConditionalGeneration
2. 修改模型文件的结构，添加必要的标识
"""
import os
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

def prepare_for_conversion():
    """准备大师模型以供转换"""

    PROJECT_ROOT = Path(__file__).parent
    MASTER_MODEL = PROJECT_ROOT / "Standalone-Bare-Master"
    CONVERSION_READY = PROJECT_ROOT / "Standalone-Bare-Master-For-Conversion"

    print("=== Preparing Master Model for GGUF Conversion ===\n")

    # 1. 读取大师配置
    with open(MASTER_MODEL / "config.json", "r", encoding="utf-8") as f:
        master_config = json.load(f)

    print(f"Master config loaded:")
    print(f"  - Hidden size: {master_config['hidden_size']}")
    print(f"  - Layers: {master_config['num_hidden_layers']}")
    print(f"  - Vocab size: {master_config['vocab_size']}")

    # 2. 创建 Qwen3-VL 兼容的配置
    qwen3vl_config = {
        # 架构标识（关键！）
        "architectures": ["Qwen3VLForConditionalGeneration"],

        # 基础配置（来自大师模型）
        "hidden_size": master_config["hidden_size"],
        "num_hidden_layers": master_config["num_hidden_layers"],
        "num_attention_heads": master_config["num_attention_heads"],
        "num_key_value_heads": master_config["num_key_value_heads"],
        "head_dim": master_config["head_dim"],
        "vocab_size": master_config["vocab_size"],

        # RoPE 配置（关键！）
        "rope_theta": master_config["rope_theta"],
        "rope_scaling": master_config["rope_scaling"],

        # RMS Norm
        "rms_norm_eps": master_config["rms_norm_eps"],

        # FFN 配置
        "intermediate_size": master_config["intermediate_size"],
        "hidden_act": master_config["hidden_act"],

        # 注意力配置
        "attention_dropout": master_config.get("attention_dropout", 0),
        "attention_bias": master_config.get("attention_bias", False),

        # 位置编码
        "max_position_embeddings": master_config["max_position_embeddings"],

        # Qwen3-VL 特定配置（必须添加）
        "vision_config": {
            # 空的 vision_config 表示没有视觉部分
            # deepstack_visual_indexes 为空列表，deepstack_layers = 0
            "deepstack_visual_indexes": []
        },

        # 其他必要字段
        "model_type": "qwen3_vl",
        "use_cache": True,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",

        # 保留原始元数据
        "_original_model": "qwen3_tts_talker",
        "_converted_from": "Standalone-Bare-Master",
    }

    # 3. 创建转换目录
    CONVERSION_READY.mkdir(exist_ok=True)

    # 4. 保存 Qwen3-VL 兼容配置
    config_path = CONVERSION_READY / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(qwen3vl_config, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Created Qwen3-VL compatible config: {config_path}")

    # 5. 复制模型权重（无需修改权重名称）
    src_weights = MASTER_MODEL / "model.safetensors"
    dst_weights = CONVERSION_READY / "model.safetensors"

    if src_weights.exists():
        shutil.copy2(src_weights, dst_weights)
        print(f"[OK] Copied model weights: {dst_weights}")
        print(f"  Size: {dst_weights.stat().st_size / 1024**3:.2f} GB")
    else:
        print(f"[X] Source weights not found: {src_weights}")
        return False

    # 6. 复制 codec_head（作为参考，不用于转换）
    src_codec = MASTER_MODEL / "codec_head.safetensors"
    dst_codec = CONVERSION_READY / "codec_head.safetensors"

    if src_codec.exists():
        shutil.copy2(src_codec, dst_codec)
        print(f"[OK] Copied codec_head: {dst_codec}")

    # 7. 创建 generation_config.json（可选）
    generation_config = {
        "max_length": 2048,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    gen_config_path = CONVERSION_READY / "generation_config.json"
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)
    print(f"[OK] Created generation_config.json")

    # 8. 创建 README
    readme_content = """# Master Model - Ready for GGUF Conversion

This directory contains the master model prepared for conversion to GGUF format using llama.cpp's convert_hf_to_gguf.py.

## Conversion Instructions

```bash
python convert_hf_to_gguf.py \\
    --model . \\
    --outfile ../qwen3-tts-master-f16.gguf \\
    --outtype f16
```

For quantized versions:
```bash
python convert_hf_to_gguf.py \\
    --model . \\
    --outfile ../qwen3-tts-master-q4_k_m.gguf \\
    --outtype q4_k_m
```

## Notes

- This model is disguised as Qwen3VLForConditionalGeneration to work with llama.cpp's converter
- The vision_config is empty (no visual components)
- MRoPE sections: [24, 20, 20]
- All weights are in model.safetensors

## Original Model

- Source: Standalone-Bare-Master
- Architecture: Qwen3-VL based (28 layers, 2048 hidden)
- Vocab: 3072 (codec tokens)
"""
    readme_path = CONVERSION_READY / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[OK] Created README.md")

    print("\n" + "="*60)
    print("[SUCCESS] Preparation Complete!")
    print("="*60)
    print(f"\nModel is ready for conversion at: {CONVERSION_READY}")
    print("\nNext steps:")
    print("1. Navigate to llama.cpp directory")
    print("2. Run conversion script:")
    print(f"   python convert_hf_to_gguf.py --model {CONVERSION_READY} --outfile qwen3-tts-master-f16.gguf --outtype f16")
    print("="*60)

    return True

if __name__ == "__main__":
    success = prepare_for_conversion()
    exit(0 if success else 1)
