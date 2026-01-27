"""
使用标准的 from_pretrained 方法加载独立的大师模型
验证模型加载和推理的正确性

纯音频词表方案 (Codec-Only Vocab):
- vocab_size = 3072
- lm_head 直接输出 codec token IDs (0-3071)
- 无需偏移量
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel

def verify_standard_loading():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用提取出来的大师模型路径
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "hf")

    print(f"--- 正在验证标准模型加载 (仅限 Codec 词表) ---")
    print(f"模型路径: {MODEL_PATH}")

    # 加载配置
    config = Qwen3TTSTalkerConfig.from_pretrained(MODEL_PATH)
    print(f"\n配置信息:")
    print(f"  词表大小 (vocab_size): {config.vocab_size}")
    print(f"  是否为纯音频词表 (_codec_only_vocab): {getattr(config, '_codec_only_vocab', False)}")

    # 方法1: 使用 from_pretrained (标准方法)
    print("\n正在使用 from_pretrained() 加载模型...")
    model = Qwen3TTSTalkerModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print("  成功通过 from_pretrained() 加载模型!")

    model.eval()

    # 加载 lm_head 从 safetensors (codec-only 方案)
    print("\n正在从 safetensors 加载 lm_head...")
    from safetensors import safe_open
    MODEL_WEIGHTS = os.path.join(MODEL_PATH, "model.safetensors")
    with safe_open(MODEL_WEIGHTS, framework="pt", device=device) as f:
        # 使用新命名的 lm_head.weight
        lm_head = f.get_tensor("lm_head.weight")
    print(f"  lm_head 形状: {lm_head.shape}")  # [3072, 2048]

    # 准备测试数据
    print("\n正在加载测试数据...")
    inputs_embeds = torch.from_numpy(np.load("20-saved-input-embds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("20-saved-input-logits.npy")).to(device).to(torch.float32)

    print(f"输入形状: {inputs_embeds.shape}")

    # 推理 - 直接使用 lm_head
    print("\n正在使用 lm_head 进行推理...")
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        next_hidden = last_hidden_state[:, -1, :]

        # 使用加载的 lm_head (标准 HF 形状: [out_features, in_features] = [3072, 2048])
        # 推理时需使用 x @ lm_head.T
        actual_logits = torch.matmul(next_hidden.to(torch.float32), lm_head.t().to(torch.float32))
        print(f"Logits 形状: {actual_logits.shape}")  # [1, 3072]

    # 验证结果
    print("\n--- 验证结果 ---")
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"预测的 Token ID: {actual_id} (codec 词表范围: 0-3071)")
    print(f"期望的 Token ID: {expected_id}")

    if actual_id == expected_id:
        print("\n[成功] 纯音频模型工作正常!")
        print("  - 直接预测 codec token (无需偏移量)")
        print(f"  - 预测的 token: {actual_id}")
        return True
    else:
        print("\n[失败] 预测结果不匹配!")
        print(f"  差异: {actual_id - expected_id}")
        return False

if __name__ == "__main__":
    try:
        success = verify_standard_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
