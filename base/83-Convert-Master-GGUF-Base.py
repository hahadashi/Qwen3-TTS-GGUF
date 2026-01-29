import sys
import os
import json
from pathlib import Path

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf")
HF_MODEL_DIR = os.path.join(PROJECT_ROOT, "model-base", "hf")
GGUF_OUT_FILE = os.path.join(PROJECT_ROOT, "model-base", "qwen3_tts_talker.gguf")

if CONVERT_LIB_DIR not in sys.path:
    sys.path.insert(0, CONVERT_LIB_DIR)

# 导入转换库
from convert_hf_to_gguf import TextModel, ModelBase
import convert_hf_to_gguf

def patched_get_vocab_base_pre(self, tokenizer) -> str:
    print(f"💉 [补丁] 绕过哈希检查，强制返回 'qwen2'。")
    return "qwen2"

def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    print(f"💉 [补丁] 强制从 {dir_model / 'config.json'} 加载配置。")
    with open(dir_model / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 兼容多种配置层级
    if "llm_config" in config:
        config["text_config"] = config["llm_config"]
    if "lm_config" in config:
        config["text_config"] = config["lm_config"]
    if "talker_config" in config:
        config["text_config"] = config["talker_config"]
        
    return config

def main():
    print(f"--- 正在执行 Base 模型 GGUF 转换 ---")
    
    # 应用补丁
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    ModelBase.load_hparams = staticmethod(patched_load_hparams)
    
    original_index_tensors = ModelBase.index_tensors
    def patched_index_tensors(self, *args, **kwargs):
        tensors = original_index_tensors(self, *args, **kwargs)
        new_tensors = {}
        for name, data_gen in tensors.items():
            if name.startswith("lm_head") or name.startswith("model."):
                new_tensors[name] = data_gen
            else:
                new_tensors[f"model.{name}"] = data_gen
        return new_tensors
    ModelBase.index_tensors = patched_index_tensors

    # 模拟参数执行
    sys.argv = [
        "convert_hf_to_gguf.py",
        HF_MODEL_DIR,
        "--outfile", GGUF_OUT_FILE,
        "--outtype", "f16"
    ]
    
    print(f"正在调用转换器...")
    convert_hf_to_gguf.main()
    print(f"✅ GGUF 转换完成！输出: {GGUF_OUT_FILE}")

if __name__ == "__main__":
    main()
