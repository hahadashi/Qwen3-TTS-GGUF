import sys
import os
import json
from pathlib import Path

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf")

if CONVERT_LIB_DIR not in sys.path:
    sys.path.insert(0, CONVERT_LIB_DIR)

# 导入转换库
try:
    import convert_hf_to_gguf
    from convert_hf_to_gguf import TextModel, ModelBase
except ImportError as e:
    print(f"❌ Error importing convert_hf_to_gguf: {e}")
    sys.exit(1)

# 定义补丁函数
def patched_get_vocab_base_pre(self, tokenizer) -> str:
    print(f"💉 [补丁] 拦截并强制返回 'qwen2'。")
    return "qwen2"

def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    print(f"💉 [补丁] 强制加载 {dir_model / 'config.json'}。")
    with open(dir_model / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def main():
    CRAFTSMAN_DIR = os.path.join(PROJECT_ROOT, "model-base", "craftsman_hf")
    GGUF_OUT = os.path.join(PROJECT_ROOT, "model-base", "qwen3_tts_craftsman.gguf")

    print(f"--- 正在执行 Base 工匠 GGUF 转换 ---")
    print(f"源路径: {CRAFTSMAN_DIR}")

    # 应用补丁
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    ModelBase.load_hparams = staticmethod(patched_load_hparams)
    
    # 动态映射权重名
    original_index_tensors = ModelBase.index_tensors
    def patched_index_tensors(self, *args, **kwargs):
        tensors = original_index_tensors(self, *args, **kwargs)
        new_tensors = {}
        for name, data_gen in tensors.items():
            # 工匠模型已经带了 embed_tokens 和 lm_head，我们需要给 Backbone 加上 model. 前缀
            if name.startswith("lm_head") or name.startswith("output.") or name.startswith("token_embd.") or name.startswith("model."):
                new_tensors[name] = data_gen
            else:
                new_tensors[f"model.{name}"] = data_gen
        return new_tensors
    ModelBase.index_tensors = patched_index_tensors

    # 模拟命令行参数 (F16 转换)
    sys.argv = [
        "convert_hf_to_gguf.py",
        CRAFTSMAN_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    print(f"正在调用 convert_hf_to_gguf.main()...")
    convert_hf_to_gguf.main()
    print(f"✅ Base 工匠 GGUF 转换完成！输出: {GGUF_OUT}")

if __name__ == "__main__":
    main()
