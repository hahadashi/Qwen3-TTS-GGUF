"""
从原始 Qwen3-TTS 模型中直接提取集成的大师模型

目标：
1. 使用 text_embedding (151K) 作为输入 embedding
2. 将 codec_head 扩展并集成到主模型文件中
3. 输出：单个 model.safetensors 文件，包含所有权重

这样模型就是标准的 LLM 格式：输入 token ID → 输出 logits
"""
import os
import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

def extract_integrated_master():
    """提取集成的大师模型"""

    PROJECT_ROOT = Path(__file__).parent
    ORIGINAL_MODEL = PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    OUTPUT_DIR = PROJECT_ROOT / "Integrated-Master-Model"

    print("="*70)
    print("Extracting Integrated Master Model")
    print("="*70)

    # 1. 读取配置
    with open(ORIGINAL_MODEL / "config.json", "r", encoding="utf-8") as f:
        original_config = json.load(f)

    talker_config = original_config['talker_config']

    print("\n[Config]")
    print(f"  Codec vocab: {talker_config['vocab_size']}")
    print(f"  Text vocab: {talker_config['text_vocab_size']}")
    print(f"  Hidden size: {talker_config['hidden_size']}")
    print(f"  Layers: {talker_config['num_hidden_layers']}")

    # 2. 创建改造后的配置
    transformed_config = talker_config.copy()
    transformed_config["vocab_size"] = transformed_config["text_vocab_size"]  # 151936
    transformed_config["_original_codec_vocab_size"] = talker_config["vocab_size"]

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 保存配置
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(transformed_config, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Config saved: {config_path}")
    print(f"  vocab_size: {transformed_config['vocab_size']}")

    # 3. 提取权重
    weights_path = ORIGINAL_MODEL / "model.safetensors"
    print(f"\n[Extracting] From: {weights_path}")

    master_weights = {}
    codec_head_weight = None

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            # 提取大师 backbone
            if key.startswith("talker.model."):
                new_key = key.replace("talker.model.", "")
                master_weights[new_key] = f.get_tensor(key)

            # 提取 codec_head
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  [OK] codec_head: {codec_head_weight.shape}")

    print(f"  [OK] Extracted {len(master_weights)} tensors")

    # 4. 处理 embedding：使用 text_embedding 作为 token_embd
    if "text_embedding.weight" in master_weights:
        master_weights["token_embd.weight"] = master_weights.pop("text_embedding.weight")
        print(f"\n[Transform] text_embedding -> token_embd")
        print(f"  Shape: {master_weights['token_embd.weight'].shape}")

    # 移除 codec_embedding
    if "codec_embedding.weight" in master_weights:
        del master_weights["codec_embedding.weight"]
        print(f"[Removed] codec_embedding.weight")

    # 5. 扩展并集成 codec_head
    print(f"\n[Extending] codec_head")
    print(f"  Original: [3072, 2048]")

    original_vocab = codec_head_weight.shape[0]  # 3072
    new_vocab = transformed_config["vocab_size"]  # 151936
    hidden_size = codec_head_weight.shape[1]  # 2048

    # 扩展
    extended_head = torch.zeros(new_vocab, hidden_size, dtype=codec_head_weight.dtype)
    extended_head[:original_vocab] = codec_head_weight

    print(f"  Extended: [151936, 2048]")
    print(f"  - [0:3072]: preserved")
    print(f"  - [3072:151936]: zero-initialized")

    # 集成到主模型
    master_weights["output.weight"] = extended_head
    print(f"\n[Integrated] output.weight -> master weights")

    # 6. 保存集成的模型
    output_path = OUTPUT_DIR / "model.safetensors"
    temp_path = OUTPUT_DIR / "model_temp.safetensors"

    save_file(master_weights, temp_path)

    # 移动到最终位置
    import shutil
    shutil.move(str(temp_path), str(output_path))

    print(f"\n[Saved] {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024**3:.2f} GB")
    print(f"  Tensors: {len(master_weights)}")

    # 7. 验证
    print("\n[Verification]")
    with safe_open(output_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # 检查输入 embedding
        if "token_embd.weight" in keys:
            emb = f.get_tensor("token_embd.weight")
            print(f"  [OK] token_embd.weight: {emb.shape}")

        # 检查输出层
        if "output.weight" in keys:
            out = f.get_tensor("output.weight")
            print(f"  [OK] output.weight: {out.shape}")

        # 检查 codec_embedding 已移除
        if "codec_embedding.weight" not in keys:
            print(f"  [OK] codec_embedding.weight removed")

    # 8. 创建 README
    readme = """# Integrated Master Model

Master model extracted from Qwen3-TTS and transformed to standard LLM format.

## Model Structure

**Input**: Token ID (0-151935)
  ↓
**Embedding**: token_embd.weight [151936, 2048]
  ↓
**LLM Backbone**: 28x Transformer Layers (Qwen3-VL architecture)
  ↓
**Output**: output.weight [151936, 2048]
  ↓
**Output**: Logits [151936]

## Key Features

- Single file: `model.safetensors` contains all weights
- Input: Text token IDs (151936 vocab)
- Output: Logits for 151936 tokens
- First 3072 tokens: Trained codec head weights
- Tokens 3072-151935: Zero-initialized (can be fine-tuned)

## Usage

This is a standard LLM model:
- Load with HuggingFace: `AutoModel.from_pretrained()`
- Convert to GGUF: `python convert_hf_to_gguf.py --model .`
- Inference: Pass token IDs, get logits directly

## Architecture

- Base: Qwen3-VL (28 layers, 2048 hidden)
- Attention: 16 heads, 8 KV heads (GQA)
- RoPE: Multi-modal with sections [24, 20, 20]
- Vocab: 151936 (text embedding)
"""
    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "="*70)
    print("[SUCCESS] Integrated Master Model Extracted!")
    print("="*70)
    print(f"\nLocation: {OUTPUT_DIR}")
    print("\nModel files:")
    print(f"  - model.safetensors ({output_path.stat().st_size / 1024**3:.2f} GB)")
    print(f"  - config.json")
    print(f"  - README.md")
    print("\nCapabilities:")
    print("  - Input: token_id (0-151935)")
    print("  - Output: logits [151936]")
    print("  - Single file, ready for conversion")
    print("="*70)

if __name__ == "__main__":
    extract_integrated_master()
