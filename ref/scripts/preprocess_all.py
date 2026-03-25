#!/usr/bin/env python
"""
一键预处理脚本

预处理 Qwen3-TTS 模型的 Embeddings 和 Tokenizer，加速推理启动。

用法:
    python scripts/preprocess_all.py --model_path /path/to/model --output_dir ./preprocessed

输出文件:
    - text_embedding_projected.npy: 投影后的文本 embedding [151936, 2048]
    - codec_embedding_0.npy: codec_0 embedding [3072, 2048]
    - codec_embedding_1~15.npy: codec_1~15 embeddings [2048, 1024]
    - tokenizer.json: HuggingFace tokenizer
    - special_tokens.json: 特殊 token ID 映射
    - preprocess_config.json: 预处理配置摘要
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="预处理 Qwen3-TTS 模型资产",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # CPU 预处理
    python scripts/preprocess_all.py --model_path ./Qwen3-TTS-12Hz-1.7B-Base

    # GPU 预处理
    python scripts/preprocess_all.py --model_path ./Qwen3-TTS-12Hz-1.7B-Base --device cuda

    # 指定输出目录
    python scripts/preprocess_all.py --model_path ./model --output_dir ./preprocessed
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Qwen3-TTS 模型路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed",
        help="输出目录 (默认: ./preprocessed)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="设备 (默认: cpu)"
    )
    parser.add_argument(
        "--skip_text",
        action="store_true",
        help="跳过 text embedding 预处理"
    )
    parser.add_argument(
        "--skip_codec",
        action="store_true",
        help="跳过 codec embedding 预处理"
    )
    parser.add_argument(
        "--skip_tokenizer",
        action="store_true",
        help="跳过 tokenizer 预处理"
    )

    args = parser.parse_args()

    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 导入预处理器
    try:
        from qwen3_tts_wrapper.assets.preprocessor import (
            TextEmbeddingPreprocessor,
            CodecEmbeddingPreprocessor,
            TokenizerPreprocessor,
        )
    except ImportError as e:
        print(f"错误: 无法导入预处理器: {e}")
        sys.exit(1)

    # 加载模型
    print("=" * 60)
    print("Qwen3-TTS 资产预处理")
    print("=" * 60)
    print(f"\n模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")

    print("\n[1/4] 加载模型...")
    try:
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            args.model_path,
            local_files_only=True,
            device_map=args.device,
        )
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        sys.exit(1)

    # 预处理 Text Embedding
    if not args.skip_text:
        print("\n[2/4] 预处理 Text Embedding...")
        try:
            text_preprocessor = TextEmbeddingPreprocessor(model.talker)
            text_path = os.path.join(args.output_dir, "text_embedding_projected.npy")
            text_preprocessor.save(text_path)
            print("  ✓ Text Embedding 预处理完成")
        except Exception as e:
            print(f"  ✗ Text Embedding 预处理失败: {e}")
    else:
        print("\n[2/4] 跳过 Text Embedding 预处理")

    # 预处理 Codec Embeddings
    if not args.skip_codec:
        print("\n[3/4] 预处理 Codec Embeddings...")
        try:
            codec_preprocessor = CodecEmbeddingPreprocessor(model)
            codec_preprocessor.save_all(args.output_dir)
            print("  ✓ Codec Embeddings 预处理完成")
        except Exception as e:
            print(f"  ✗ Codec Embeddings 预处理失败: {e}")
    else:
        print("\n[3/4] 跳过 Codec Embeddings 预处理")

    # 预处理 Tokenizer
    if not args.skip_tokenizer:
        print("\n[4/4] 预处理 Tokenizer...")
        try:
            tokenizer_preprocessor = TokenizerPreprocessor(args.model_path)
            tokenizer_preprocessor.save_tokenizer(
                os.path.join(args.output_dir, "tokenizer.json")
            )
            tokenizer_preprocessor.save_special_tokens(
                os.path.join(args.output_dir, "special_tokens.json")
            )
            print("  ✓ Tokenizer 预处理完成")
        except Exception as e:
            print(f"  ✗ Tokenizer 预处理失败: {e}")
    else:
        print("\n[4/4] 跳过 Tokenizer 预处理")

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)

    print(f"\n输出目录: {os.path.abspath(args.output_dir)}")
    print("\n生成的文件:")

    import numpy as np

    total_size = 0
    for f in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, f)
        size = os.path.getsize(filepath)
        total_size += size

        if f.endswith('.npy'):
            try:
                data = np.load(filepath)
                print(f"  {f}")
                print(f"    shape: {data.shape}, dtype: {data.dtype}, size: {size/1024/1024:.2f} MB")
            except:
                print(f"  {f}: {size/1024/1024:.2f} MB")
        else:
            print(f"  {f}: {size/1024:.2f} KB")

    print(f"\n总大小: {total_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
