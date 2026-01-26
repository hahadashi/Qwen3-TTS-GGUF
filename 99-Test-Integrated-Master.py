"""
使用 bare_master 的模型定义加载集成模型并进行推理验证
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel
from safetensors import safe_open

def test_integrated_master():
    """加载集成模型并测试推理"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = PROJECT_ROOT / "Integrated-Master-Model"

    print("="*70)
    print("Testing Integrated Master Model Inference")
    print("="*70)

    # 1. 加载配置
    print("\n[1] Loading config...")
    config = Qwen3TTSTalkerConfig.from_pretrained(MODEL_DIR)
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")

    # 2. 创建模型
    print("\n[2] Creating model...")
    model = Qwen3TTSTalkerModel(config).to(DEVICE).to(torch.bfloat16)
    model.eval()

    # 3. 加载权重
    print("\n[3] Loading weights...")
    model_path = MODEL_DIR / "model.safetensors"

    with safe_open(model_path, framework="pt", device=DEVICE) as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # 加载到模型
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    print(f"  [OK] Weights loaded")

    # 4. 准备测试数据
    print("\n[4] Preparing test data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(DEVICE).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(DEVICE).to(torch.float32)

    print(f"  Input shape: {inputs_embeds.shape}")
    print(f"  Expected token ID: {torch.argmax(expected_logits).item()}")

    # 5. 推理
    print("\n[5] Running inference...")
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=DEVICE)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 取最后一个 token
        next_hidden = last_hidden_state[:, -1, :]

        # 应用 output 层
        # 从 state_dict 中获取 output.weight
        output_weight = state_dict["output.weight"].to(torch.float32)

        # 计算 logits
        actual_logits = torch.matmul(next_hidden.to(torch.float32), output_weight.T)

    # 6. 对比结果
    print("\n[6] Comparing results...")
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"  Predicted token ID: {actual_id}")
    print(f"  Expected token ID:  {expected_id}")

    # 只比较前 3072 个 logits（原始 codec 范围）
    actual_logits_codec = actual_logits[0, :3072]
    expected_logits_codec = expected_logits[0]

    diff = torch.abs(actual_logits_codec - expected_logits_codec)
    max_diff = torch.max(diff).item()
    print(f"  Max logit diff (codec range): {max_diff:.6f}")

    # 7. 结论
    print("\n" + "="*70)
    if actual_id == expected_id:
        print("[SUCCESS] Integrated model works correctly!")
        print(f"  Predicted: {actual_id}")
        print(f"  Expected:  {expected_id}")
        print(f"  Match: YES")
    else:
        print("[FAILURE] Prediction mismatch!")
        print(f"  Predicted: {actual_id}")
        print(f"  Expected:  {expected_id}")
        print(f"  Match: NO")
    print("="*70)

    return actual_id == expected_id

if __name__ == "__main__":
    try:
        success = test_integrated_master()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
