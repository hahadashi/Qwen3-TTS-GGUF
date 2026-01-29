import os
import json

def create_craftsman_tokenizer_base():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "model-base", "craftsman_hf")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"--- 正在为 Base 工匠组件构造假分词器 ---")
    
    vocab = {"a": 0, "b": 1, "ab": 2}
    for i in range(3, 10):
        vocab[f"<t_{i}>"] = i
        
    tokenizer_json = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "a", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 1, "content": "b", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
            {"id": 2, "content": "ab", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False}
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
    
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)
        
    print(f"✅ Base 工匠假分词器构造完成: {TARGET_DIR}")

if __name__ == "__main__":
    create_craftsman_tokenizer_base()
