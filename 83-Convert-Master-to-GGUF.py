"""
83-Convert-Master-to-GGUF.py (80 Series - No Tokenizer Edition)
使用 Monkey Patch 动态修改 Llama.cpp 官方转换脚本，
实现 3072 词表大师模型的“无 Tokenizer”极简转换。

工作流：
1. 内存中重整权重：为 Tensor 添加 model. 前缀，转置 lm_head，对齐 HF 标准。
2. 动态补丁：强制 TextModel.set_vocab 走 _set_vocab_none 路径，跳过所有词表校验。
3. 内联调用：直接运行转换器的 main 函数，不产生持久化的中间冗余文件。
"""
import os
import sys
import shutil
import json
import torch
from safetensors.torch import save_file, safe_open

# 1. 环境准备：强行优先加载本地库
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GG_PATH = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf")

# 必须将本地库路径插到最前面，确保覆盖环境中的旧版 gguf 库
if GG_PATH not in sys.path:
    sys.path.insert(0, GG_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # 优先尝试从本地包导入
    import convert_hf_to_gguf as cvt
except ImportError as e:
    print(f"❌ Error: {e}")
    print("❌ Error: Trying alternative import...")
    try:
        import qwen3_tts_gguf.convert_hf_to_gguf as cvt
    except ImportError as e2:
        print(f"❌ Error: {e2}")
        sys.exit(1)

import inspect

# 2. Monkey Patch: 核心逻辑 - 强制绕过 Tokenizer
def patched_set_vocab(self):
    print(f"      [Monkey Patch] Activated for {self.__class__.__name__}! Forcing None behavior...")
    self._set_vocab_none()

# 通用批量注入：自动遍历所有模型类并覆盖 set_vocab
print("[Monkey Patch] Scanning and patching all model classes...")
patch_count = 0
for name, cls in inspect.getmembers(cvt):
    if inspect.isclass(cls) and hasattr(cls, 'set_vocab'):
        # 确保是该模块定义的类或相关的 TextModel 子类
        if issubclass(cls, cvt.ModelBase):
             cls.set_vocab = patched_set_vocab
             patch_count += 1
print(f"  → Patched {patch_count} classes.")

def create_dummy_tokenizer_files(target_dir):
    """创建极简的 Tokenizer 文件以满足 AutoTokenizer.from_pretrained 的文件存在性检查"""
    print("  → Creating dummy tokenizer files to fool AutoTokenizer...")
    
    # vocab.json
    vocab = {"<|endoftext|>": 0, "dummy": 1}
    with open(os.path.join(target_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
        
    # merges.txt
    with open(os.path.join(target_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        
    # tokenizer_config.json (模仿 Qwen2)
    tokenizer_config = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "model_max_length": 32768,
        "tokenizer_class": "Qwen2Tokenizer",
        "added_tokens_decoder": {}
    }
    with open(os.path.join(target_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f)
        
    # tokenizer.json (可选，但为了保险起见)
    tokenizer_json = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    }
    with open(os.path.join(target_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f)

def convert_master_80():
    BARE_MASTER_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master")
    OUTPUT_GGUF = os.path.join(PROJECT_ROOT, "master-codec-only-3072-f16.gguf")
    TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_conv_83")
    
    print(f"--- 80 Series GGUF Conversion (Systematic Bypass) ---")
    
    if not os.path.exists(BARE_MASTER_DIR):
        print(f"❌ Error: {BARE_MASTER_DIR} not found. Please run script 81 first.")
        return

    # A. 权重内存重载与标准化
    print("[1/3] Preparing standardized weights in memory...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # 【新增】创建 Dummy 文件
    create_dummy_tokenizer_files(TEMP_DIR)

    src_weights_path = os.path.join(BARE_MASTER_DIR, "model.safetensors")
    new_weights = {}

    with safe_open(src_weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # 逻辑 1: 处理 lm_head 转置 (从 [2048, 3072] 转回 [3072, 2048])
            if key == "lm_head":
                print(f"  → Transposing lm_head: {tensor.shape} -> [3072, 2048]")
                new_weights["lm_head.weight"] = tensor.t().contiguous()
            
            # 逻辑 2: 处理命名对齐 (添加 model. 前缀和 .weight 后缀)
            elif key == "embed_tokens":
                new_weights["model.embed_tokens.weight"] = tensor
            else:
                # 81 脚本输出如 layers.0.xxx，我们需要 model.layers.0.xxx.weight
                new_key = key if key.startswith("model.") else f"model.{key}"
                if not new_key.endswith(".weight") and "norm" not in key: 
                    # norm 层通常在 Qwen2 中本来就叫 model.norm.weight
                    new_key += ".weight"
                new_weights[new_key] = tensor

    # 保存临时标准化权重文件，供官网脚本加载
    save_file(new_weights, os.path.join(TEMP_DIR, "model.safetensors"))

    # B. 配置处理
    with open(os.path.join(BARE_MASTER_DIR, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 强制核心参数一致
    config["vocab_size"] = 3072
    # 为了让脚本认出它是 Qwen3VL 并不去翻找视觉组件，这里保持原样
    with open(os.path.join(TEMP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # C. 内联调用转换核心
    print("[2/3] Calling GGUF Converter (Monkey Patched)...")
    
    # 构造模拟命令行参数
    old_args = sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py",
        TEMP_DIR,
        "--outfile", OUTPUT_GGUF,
        "--outtype", "f16"
    ]

    # 这个 main 是从 convert_hf_to_gguf 导入的
    cvt.main()
    print(f"\n✅ SUCCESS: GGUF generated at {OUTPUT_GGUF}")

    # 还原参数并清理
    sys.argv = old_args 
    print("[3/3] Cleaning up temporary conversion files...")
    if os.path.exists(TEMP_DIR):
         shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    convert_master_80()
