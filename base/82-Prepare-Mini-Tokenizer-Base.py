import os
import json

def create_base_mini_tokenizer():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "model-base", "hf")
    
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    print(f"--- 正在为 Base 模型构造迷你分词器 ---")

    # 1. 词表
    vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "a": 97, "b": 98, "ab": 99}
    for i in range(3, 10): vocab[f"<t_{i}>"] = i

    # 2. 保存文件
    with open(os.path.join(TARGET_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    
    with open(os.path.join(TARGET_DIR, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("a b\n")

    tokenizer_config = {
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "special": True},
            "1": {"content": "<|im_start|>", "special": True},
            "2": {"content": "<|im_end|>", "special": True}
        },
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", 
        "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "model_max_length": 32768, "tokenizer_class": "Qwen2Tokenizer"
    }
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    tokenizer_json = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "<|im_end|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": ["a b"]}
    }
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f)

    print(f"✅ 迷你分词器已生成于: {TARGET_DIR}")

if __name__ == "__main__":
    create_base_mini_tokenizer()
