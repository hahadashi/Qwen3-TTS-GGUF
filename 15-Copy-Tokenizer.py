"""
15-Copy-Tokenizer.py
复制文本 Tokenizer 到 model 文件夹，方便模型部署和管理。
"""
import os
import shutil
from pathlib import Path

from export_config import MODEL_DIR, EXPORT_DIR

def main():
    # 1. 检查源路径
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        return

    # 2. 目标文件
    TARGET_FILE = os.path.join(EXPORT_DIR, "tokenizer.json")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    print(f"   源路径: {MODEL_DIR}")
    print(f"   目标文件: {TARGET_FILE}")

    # [核心增强] 生成单一 tokenizer.json 以便被 tokenizers 库加载
    try:
        from transformers import AutoTokenizer
        print("   正在将官方 Tokenizer 转换为单一格式 (tokenizer.json)...")
        # 直接从源目录加载官方分词器
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        
        # 核心：直接使用 Rust 后端的 save 方法，只导出单一的 tokenizer.json
        if hasattr(hf_tokenizer, "backend_tokenizer"):
            hf_tokenizer.backend_tokenizer.save(TARGET_FILE)
            file_size = os.path.getsize(TARGET_FILE) / 1024 / 1024
            print(f"   ✅ 已合成单体 tokenizer.json ({file_size:.2f} MB)。")
        else:
            # 降级方案：保存到临时目录再移动
            temp_dir = os.path.join(EXPORT_DIR, "_temp_tok")
            hf_tokenizer.save_pretrained(temp_dir)
            src_json = os.path.join(temp_dir, "tokenizer.json")
            if os.path.exists(src_json):
                shutil.move(src_json, TARGET_FILE)
                print("   ✅ 已通过降级方案合成单体 tokenizer.json。")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except ImportError:
        print("   ⚠️  未安装 transformers，无法合成单体 tokenizer.json。")
    except Exception as e:
        print(f"   ⚠️  Tokenizer 转换失败: {e}")


if __name__ == "__main__":
    main()
