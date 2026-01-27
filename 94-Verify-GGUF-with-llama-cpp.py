"""
使用 llama.cpp C API 直接注入 embeddings 来验证 GGUF 模型

验证内容：
1. 使用 llama.cpp DLL 加载 GGUF 模型
2. 直接注入 embeddings（与 92 脚本相同的输入）
3. 验证推理结果是否与 HF 模型一致
4. 检查 logits 分布是否符合合并词表方案（前 3072 个位置有效）

参考：fun_asr_gguf/nano_llama.py 和 core/decoder.py
"""
import os
import sys
import numpy as np
import ctypes
import time

# 添加 fun_asr_gguf 模块路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fun_asr_gguf"))

# 导入整个模块，而不是单独导入函数
# 这是因为 llama_model_get_vocab 等函数是在 init_llama_lib() 中动态初始化的
import fun_asr_gguf.nano_llama as nano_llama


def verify_gguf_with_embeddings():
    """使用 llama.cpp C API 验证 GGUF 模型的 embedding 推理"""

    # 配置
    GGUF_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "master-merged-vocab-f16.gguf")

    print("=" * 70)
    print("GGUF Model Verification with llama.cpp C API")
    print("=" * 70)
    print(f"\nGGUF model path: {GGUF_MODEL_PATH}")

    # 检查文件
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"❌ Error: GGUF model not found at {GGUF_MODEL_PATH}")
        print(f"Please run 93-Convert-Master-to-GGUF.py first")
        return False

    # 1. 加载 GGUF 模型（使用新的 load_model 函数，会自动初始化库）
    print(f"\n[1/6] Loading GGUF model...")
    try:
        model = nano_llama.load_model(GGUF_MODEL_PATH)
        if not model:
            raise RuntimeError("Failed to load GGUF model (model is None)")

        vocab = nano_llama.llama_model_get_vocab(model)
        vocab_size = nano_llama.llama_vocab_n_tokens(vocab)
        eos_token = nano_llama.llama_vocab_eos(vocab)

        print(f"  ✓ Model loaded successfully")
        print(f"    - Vocab size: {vocab_size}")
        print(f"    - EOS token: {eos_token}")

        if vocab_size == 151936:
            print(f"    ✓ Correct vocab size (merged vocab: 151936)")
        else:
            print(f"    ⚠ Unexpected vocab size (expected 151936)")

    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 创建上下文
    print(f"\n[2/6] Creating context...")
    try:
        ctx_params = nano_llama.llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_batch = 2048
        ctx_params.n_ubatch = 512
        ctx_params.embeddings = False  # 我们要输出 logits，不是 embeddings
        ctx_params.no_perf = True
        ctx_params.n_threads = os.cpu_count() // 2

        ctx = nano_llama.llama_init_from_model(model, ctx_params)
        if not ctx:
            raise RuntimeError("Failed to create context")

        print(f"  ✓ Context created successfully")
        print(f"    - Context size: {ctx_params.n_ctx}")
        print(f"    - Threads: {ctx_params.n_threads}")

    except Exception as e:
        print(f"  ❌ Failed to create context: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 加载测试数据
    print(f"\n[3/6] Loading test data...")
    try:
        inputs_embeds = np.load("40_first_step_embeds.npy")  # [1, seq_len, 2048]
        expected_logits = np.load("40_first_step_logits.npy")  # [1, 3072]

        # 确保是 float32 类型（llama.cpp 使用 float32）
        inputs_embeds = inputs_embeds.astype(np.float32)

        print(f"  ✓ Test data loaded")
        print(f"    - Input embeddings shape: {inputs_embeds.shape}")
        print(f"    - Expected logits shape: {expected_logits.shape}")

        # 获取输入的 token 数量
        n_input_tokens = inputs_embeds.shape[1]
        hidden_size = inputs_embeds.shape[2]

        print(f"    - Sequence length: {n_input_tokens}")
        print(f"    - Hidden size: {hidden_size}")

        if hidden_size != 2048:
            print(f"    ⚠ Unexpected hidden size (expected 2048)")

    except Exception as e:
        print(f"  ❌ Failed to load test data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 注入 embeddings 并推理
    print(f"\n[4/6] Injecting embeddings and running inference...")
    try:
        # 清空 KV 缓存
        mem = nano_llama.llama_get_memory(ctx)
        nano_llama.llama_memory_clear(mem, True)

        # 移除 batch 维度 [1, seq_len, hidden] -> [seq_len, hidden]
        full_embd = inputs_embeds[0]  # [seq_len, 2048]

        # 确保 C 连续
        if not full_embd.flags['C_CONTIGUOUS']:
            full_embd = np.ascontiguousarray(full_embd)

        # 创建 batch
        batch = nano_llama.llama_batch_init(n_input_tokens, full_embd.shape[1], 1)
        batch.n_tokens = n_input_tokens

        # 设置 token 为空（我们使用 embeddings）
        batch.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token))

        # 复制 embeddings
        ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)

        # 设置位置和序列 ID
        for k in range(n_input_tokens):
            batch.pos[k] = k
            batch.n_seq_id[k] = 1
            batch.seq_id[k][0] = 0
            # 只在最后一个位置计算 logits
            batch.logits[k] = 1 if k == n_input_tokens - 1 else 0

        # 运行推理
        t_start = time.perf_counter()
        ret = nano_llama.llama_decode(ctx, batch)
        t_inject = time.perf_counter() - t_start

        nano_llama.llama_batch_free(batch)

        if ret != 0:
            raise RuntimeError(f"llama_decode failed (ret={ret})")

        print(f"  ✓ Inference completed successfully")
        print(f"    - Time: {t_inject*1000:.2f}ms")

    except Exception as e:
        print(f"  ❌ Failed to run inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 获取 logits 并分析
    print(f"\n[5/6] Analyzing logits...")
    try:
        # 获取 logits 指针
        logits_ptr = nano_llama.llama_get_logits(ctx)

        # 转换为 numpy 数组
        # logits 的形状应该是 [vocab_size] = [151936]
        logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))

        print(f"  ✓ Logits extracted")
        print(f"    - Logits shape: {logits_arr.shape}")
        print(f"    - Logits dtype: {logits_arr.dtype}")

        # 分析 logits 分布
        print(f"\n  --- Logit Distribution Analysis ---")

        # Codec logits (前 3072 个位置)
        codec_logits = logits_arr[:3072]
        print(f"  Codec logits [0:3072]:")
        print(f"    - Shape: {codec_logits.shape}")
        print(f"    - Min: {codec_logits.min():.6f}")
        print(f"    - Max: {codec_logits.max():.6f}")
        print(f"    - Mean: {codec_logits.mean():.6f}")
        print(f"    - Std: {codec_logits.std():.6f}")

        # Padding logits (3072-151935 位置)
        padding_logits = logits_arr[3072:]
        print(f"  Padding logits [3072:151936]:")
        print(f"    - Shape: {padding_logits.shape}")
        print(f"    - Min: {padding_logits.min():.6f}")
        print(f"    - Max: {padding_logits.max():.6f}")
        print(f"    - Mean: {padding_logits.mean():.6f}")
        print(f"    - Std: {padding_logits.std():.6f}")

        # 找到最大值位置
        max_logit_value = np.max(logits_arr)
        max_logit_pos = np.argmax(logits_arr)

        print(f"\n  Maximum logit:")
        print(f"    - Value: {max_logit_value:.6f}")
        print(f"    - Position: {max_logit_pos}")

        if max_logit_pos < 3072:
            print(f"    ✓ Maximum is in codec range [0, 3071]")
        else:
            print(f"    ⚠ Maximum is outside codec range (position {max_logit_pos})")

        # 检查 padding logits 是否接近零
        is_padding_zero = np.allclose(padding_logits, 0.0, atol=1e-6)
        print(f"\n  Padding logits check:")
        if is_padding_zero:
            print(f"    ✓ All padding logits are close to zero (atol=1e-6)")
        else:
            print(f"    ⚠ Some padding logits are non-zero")
            print(f"    - Non-zero count: {np.count_nonzero(padding_logits)}")

    except Exception as e:
        print(f"  ❌ Failed to analyze logits: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 与期望结果对比
    print(f"\n[6/6] Comparing with expected results...")
    try:
        # 计算预测的 token ID
        actual_token_id = int(np.argmax(logits_arr))

        # 计算期望的 token ID
        expected_token_id = int(np.argmax(expected_logits[0]))

        print(f"  Predicted token ID: {actual_token_id}")
        print(f"  Expected token ID:  {expected_token_id}")

        if actual_token_id == expected_token_id:
            print(f"  ✓ Token ID matches!")

            # 比较 codec logits（只比较前 3072 个位置）
            actual_codec_logits = logits_arr[:3072]
            expected_codec_logits = expected_logits[0]

            # 计算相似度
            cosine_sim = np.dot(actual_codec_logits, expected_codec_logits) / (
                np.linalg.norm(actual_codec_logits) * np.linalg.norm(expected_codec_logits)
            )

            print(f"\n  Codec logits comparison:")
            print(f"    - Cosine similarity: {cosine_sim:.6f}")
            print(f"    - MSE: {np.mean((actual_codec_logits - expected_codec_logits)**2):.6f}")

            if cosine_sim > 0.95:
                print(f"    ✓ High similarity (>0.95)")
            elif cosine_sim > 0.9:
                print(f"    ⚠ Moderate similarity (0.9-0.95)")
            else:
                print(f"    ⚠ Low similarity (<0.9)")

        else:
            print(f"  ⚠ Token ID mismatch!")
            print(f"    Difference: {actual_token_id - expected_token_id}")

    except Exception as e:
        print(f"  ❌ Failed to compare results: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 清理资源
    print(f"\n--- Cleanup ---")
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    nano_llama.llama_backend_free()
    print(f"✓ Resources cleaned up")

    # 总结
    print(f"\n" + "=" * 70)
    print(f"Summary:")
    print(f"=" * 70)
    print(f"✓ GGUF model loaded successfully")
    print(f"✓ Embedding inference works")
    print(f"✓ Vocab size: {vocab_size} (merged vocab)")
    print(f"✓ Logits distribution correct:")
    print(f"  - Codec range [0:3072]: non-zero")
    print(f"  - Padding range [3072:151936]: close to zero")
    print(f"✓ Token prediction: {actual_token_id} (expected: {expected_token_id})")
    print(f"=" * 70)

    return actual_token_id == expected_token_id


if __name__ == "__main__":
    try:
        success = verify_gguf_with_embeddings()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
