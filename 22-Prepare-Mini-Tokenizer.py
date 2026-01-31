"""
从零构造适配 3072 词表的迷你 Tokenizer。
完全不依赖外部模型文件，直接在脚本内生成所有配置。
"""
import os
import json

from export_config import EXPORT_DIR

def create_zero_tokenizer():
    TARGET_DIR = os.path.join(EXPORT_DIR, "hf")
    
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    print(f"--- 正在从零构造迷你分词器 (3072 词表) ---")
    print(f"输出目录: {TARGET_DIR}")

    # 1. 构造极简词表 (vocab.json)
    # 只要包含 BPE merge 所需的字符和特殊字符即可。
    # 缺失的 ID 会被转换脚本自动填充为 [PAD]
    print("[1/4] 正在生成极简词表...")
    vocab = {}
    
    # 填充特殊 token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    for i, tok in enumerate(special_tokens):
        vocab[tok] = i
        
    # 为 BPE 合并规则保留 a, b, ab
    vocab["a"] = 97
    vocab["b"] = 98
    vocab["ab"] = 99
    
    # 我们随便填充一点，凑够 10 个
    for i in range(3, 10):
        vocab[f"<t_{i}>"] = i

    with open(os.path.join(TARGET_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  ✓ 已保存 vocab.json (大小: {len(vocab)})")

    # 2. 构造 BPE 合并规则 (merges.txt)
    print("[2/4] 正在生成极简 merges.txt...")
    with open(os.path.join(TARGET_DIR, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("a b\n")
    print(f"  ✓ 已保存 merges.txt")

    # 3. 构造配置 (tokenizer_config.json)
    print("[3/4] 正在生成 tokenizer_config.json...")
    tokenizer_config = {
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "special": True},
            "1": {"content": "<|im_start|>", "lstrip": False, "normalized": False, "rstrip": False, "special": True},
            "2": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "special": True}
        },
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "model_max_length": 32768,
        "tokenizer_class": "Qwen2Tokenizer",
        "model_type": "qwen2"
    }
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已保存 tokenizer_config.json")

    # 4. 构造完整结构 (tokenizer.json)
    print("[4/4] 正在生成 tokenizer.json...")
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "<|im_end|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["a b"]
        }
    }
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    print(f"  ✓ 已保存 tokenizer.json")

    print(f"\n✅ 零基分词器创建完成！")

if __name__ == "__main__":
    create_zero_tokenizer()
