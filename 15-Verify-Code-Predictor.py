
import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add source path
PROJECT_ROOT = Path(__file__).parent
SOURCE_DIR = PROJECT_ROOT / "Qwen3-TTS"
sys.path.append(str(SOURCE_DIR))

try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

# Configuration
MODEL_PATH = PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
ONNX_PATH = PROJECT_ROOT / "model" / "qwen3_tts_predictor.onnx"
HEADS_PATH = PROJECT_ROOT / "model" / "qwen3_tts_predictor_heads.npy"

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def verify_predictor():
    print(f"--- Verification: Code Predictor (ONNX vs PyTorch) ---")
    
    # 1. Load PyTorch Model
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    full_model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu"
    )
    predictor_pt = full_model.talker.code_predictor
    predictor_pt.eval()
    
    # 2. Load ONNX model
    print(f"Loading ONNX model from {ONNX_PATH}...")
    ort_session = ort.InferenceSession(str(ONNX_PATH))
    
    # 3. Load Heads
    print(f"Loading Heads from {HEADS_PATH}...")
    heads_weights = np.load(HEADS_PATH) # [15, Vocab, Hidden]
    
    # 4. Prepare Inputs
    batch_size = 1
    input_dim = 2048
    hidden_size = 1024
    head_dim = 128
    num_kv_heads = 8
    num_layers = 5
    
    # 我们模拟一次“增量解码”过程：
    # 步骤 A: 初始输入 (past_seq=0) -> 得到 hidden_state 和 present_kv
    # 步骤 B: 拿 present_kv 作为下一次的 past_kv (past_seq=1) -> 验证一致性
    
    print("\n[Test 1] Initial Forward (past_seq=0)")
    dummy_input_pt = torch.randn(batch_size, 1, input_dim)
    dummy_past_pt = [torch.zeros(batch_size, num_kv_heads, 0, head_dim) for _ in range(10)]
    
    # PyTorch Run
    with torch.no_grad():
        # 我们用模型内部的逻辑流程进行对比
        # A. Projection
        h_proj = predictor_pt.small_to_mtp_projection(dummy_input_pt)
        # B. Backbone
        outputs_pt = predictor_pt.model(
            inputs_embeds=h_proj,
            past_key_values=None, # 初始为空
            use_cache=True,
            return_dict=False
        )
        last_hidden_pt = outputs_pt[0].numpy()
        present_kv_pt = outputs_pt[1]
        
    # ONNX Run
    onnx_inputs = {"inputs_embeds": dummy_input_pt.numpy()}
    for i in range(10):
        onnx_inputs[f"past_{i}"] = dummy_past_pt[i].numpy()
        
    onnx_outputs = ort_session.run(None, onnx_inputs)
    last_hidden_onnx = onnx_outputs[0]
    present_kv_onnx = onnx_outputs[1:]
    
    # Compare Hidden
    diff_hidden = np.abs(last_hidden_pt - last_hidden_onnx).max()
    print(f"  Hidden State Max Diff: {diff_hidden:.6e}")
    
    # Compare Heads (Heads 并不在 ONNX 里，我们手动乘一下)
    # predictor_pt.lm_head[0] 对应 Code 1 的预测
    logits_pt = predictor_pt.lm_head[0](outputs_pt[0]).detach().numpy()
    logits_onnx = last_hidden_onnx @ heads_weights[0].T
    diff_logits = np.abs(logits_pt - logits_onnx).max()
    print(f"  Logits (Head 0) Max Diff: {diff_logits:.6e}")
    
    if diff_hidden < 1e-4:
        print("✅ Numerical match confirmed for initial forward!")
    else:
        print("❌ Numerical discrepancy detected!")

    print("\n[Test 2] Incremental Forward (past_seq=1)")
    # 使用第一步产生的 present_kv 作为第二步的 past_kv
    dummy_input_next_pt = torch.randn(batch_size, 1, input_dim)
    
    # PyTorch Next
    with torch.no_grad():
        h_proj_next = predictor_pt.small_to_mtp_projection(dummy_input_next_pt)
        outputs_next_pt = predictor_pt.model(
            inputs_embeds=h_proj_next,
            past_key_values=present_kv_pt,
            use_cache=True,
            return_dict=False
        )
        last_hidden_next_pt = outputs_next_pt[0].numpy()
        
    # ONNX Next
    onnx_inputs_next = {"inputs_embeds": dummy_input_next_pt.numpy()}
    for i in range(10):
        onnx_inputs_next[f"past_{i}"] = present_kv_onnx[i] # 传回刚才得到的 KV
        
    onnx_outputs_next = ort_session.run(None, onnx_inputs_next)
    last_hidden_next_onnx = onnx_outputs_next[0]
    
    diff_next = np.abs(last_hidden_next_pt - last_hidden_next_onnx).max()
    print(f"  Next Hidden State Max Diff: {diff_next:.6e}")
    
    if diff_next < 1e-4:
        print("✅ Numerical match confirmed for incremental forward (KV Cache works)!")
    else:
        print("❌ KV Cache logic might be broken in ONNX!")

if __name__ == "__main__":
    verify_predictor()
