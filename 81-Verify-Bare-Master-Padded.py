import os
import sys
import torch
import numpy as np
from safetensors import safe_open

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel, Qwen3TTSTalkerConfig

def verify_bare_master():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    WEIGHTS_PATH = os.path.join(MODEL_PATH, "model.safetensors")
    CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
    
    import json
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    
    talker_config = Qwen3TTSTalkerConfig(**full_config['talker_config'])
    
    print(f"Loading Bare Master weights from {WEIGHTS_PATH}...")
    # 1. 构造剥离后的模型
    master = Qwen3TTSTalkerModel(talker_config).to(device).to(torch.bfloat16)
    
    # 2. 手动加载权重 (只加载 talker.model 部分)
    with safe_open(WEIGHTS_PATH, framework="pt", device=device) as f:
        state_dict = {}
        for key in f.keys():
            if key.startswith("talker.model."):
                new_key = key.replace("talker.model.", "")
                state_dict[new_key] = f.get_tensor(key)
        master.load_state_dict(state_dict, strict=True)
    
    # 3. 构造 Padded Codec Head
    # 原始输出层是 talker.codec_head (Linear: Hidden -> 3072)
    original_head_weight = None
    with safe_open(WEIGHTS_PATH, framework="pt", device=device) as f:
        original_head_weight = f.get_tensor("talker.codec_head.weight")
    
    # 我们要补齐到 151936
    PADDED_VOCAB_SIZE = 151936
    HIDDEN_SIZE = original_head_weight.shape[1]
    print(f"Padding lm_head: 3072 -> {PADDED_VOCAB_SIZE}")
    
    padded_head = torch.nn.Linear(HIDDEN_SIZE, PADDED_VOCAB_SIZE, bias=False).to(device).to(torch.bfloat16)
    with torch.no_grad():
        padded_head.weight.zero_()
        padded_head.weight[:3072, :] = original_head_weight
    
    # 4. 加载拦截的数据
    print("Loading intercepted data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)
    
    print(f"Input embeds shape: {inputs_embeds.shape}") # Should be [1, 14, 2048]
    
    # 5. 运行推理 (模拟 Master 的前向过程)
    master.eval()
    with torch.no_grad():
        # Talker Model Forward
        # 注意：这里需要传入默认的 attention_mask，否则 get_rope_index 可能会挂
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        # 模拟 modeling_qwen3_tts.py 中的 forward 逻辑
        # 因为我们直接调 talker.model.forward，它会自己计算 position_ids
        output = master(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state # [B, T, Hidden]
        
        # 取最后一个位置推测第一个 token
        next_hidden = last_hidden_state[:, -1, :] # [B, Hidden]
        
        # 通过 Padded Head 获取 Logits
        actual_logits = padded_head(next_hidden).to(torch.float32) # [B, 151936]
        
    # 6. 对比验证
    print("\n--- Comparison Results ---")
    # 检查前 3072 维
    slice_actual = actual_logits[0, :3072]
    slice_expected = expected_logits[0]
    
    diff = torch.abs(slice_actual - slice_expected)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Max difference in first 3072 logits: {max_diff:.6f}")
    print(f"Mean difference in first 3072 logits: {mean_diff:.6f}")
    
    # 检查填充部分是否为 0
    padding_sum = torch.sum(torch.abs(actual_logits[0, 3072:])).item()
    print(f"Sum of absolute values in padding zone (3072-151936): {padding_sum:.6f}")
    
    # 检查预测 ID
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()
    print(f"Actual predicted ID: {actual_id}")
    print(f"Expected predicted ID: {expected_id}")
    
    if actual_id == expected_id and max_diff < 1e-3 and padding_sum == 0:
        print("\n✅ Verification SUCCESS! Bare Master (Padded) matches baseline perfectly.")
    else:
        print("\n❌ Verification FAILED!")

if __name__ == "__main__":
    verify_bare_master()
