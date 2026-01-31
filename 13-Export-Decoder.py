"""
85-Export-Stateful-ONNX.py
Qwen3-TTS Stateful Codec Decoder ONNX 导出脚本。

核心经验参考：Experience/01-Qwen3-Code-Predictor-Export.md
- 强制 dynamo=False 使用经典 JIT 路径
- 显式参数签名避免 TreeSpec 校验失败
- Dummy Input 维度必须与模型权重对齐
"""
import os
import torch
import numpy as np
from qwen3_tts_gguf.codec_export import StatefulCodecONNXWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

from export_config import MODEL_DIR, EXPORT_DIR

def main():
    # 1. 配置
    ONNX_FILENAME = "qwen3_tts_decoder_stateful.onnx"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    device = "cpu"
    
    # 2. 加载模型
    print("🚀 正在加载模型...")
    tokenizer_load_path = os.path.join(MODEL_DIR, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_DIR, "speech_tokenizer")) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_load_path).to(device)
    wrapper = StatefulCodecONNXWrapper(model).to(device)
    wrapper.eval()
    
    # 3. 获取关键配置
    num_layers = wrapper.num_layers
    # 关键：head_dim 必须从配置中直接读取，而非公式推导！
    # 模型配置中 hidden_size=512, num_heads=16, 但 head_dim=64 (硬编码)
    cfg = wrapper.decoder.config
    num_heads = cfg.num_key_value_heads if hasattr(cfg, 'num_key_value_heads') else cfg.num_attention_heads
    head_dim = cfg.head_dim  # 直接读取，避免维度陷阱
    
    print(f"   num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}")

    
    # 4. 创建 Dummy Inputs (显式签名)
    B = 1
    N = 3  # 每跳 3 帧
    Q = 16  # 量化器数量
    
    dummy_audio_codes = torch.zeros(B, N, Q, dtype=torch.long, device=device)
    dummy_is_last = torch.tensor([0.0], device=device)
    dummy_pre_conv_h = torch.zeros(B, 512, 0, device=device) 
    dummy_latent_buf = torch.zeros(B, 1024, 0, device=device)
    dummy_conv_h = torch.zeros(B, 1024, 0, device=device)
    
    # KV Cache: num_layers 个 K + num_layers 个 V
    # Shape: [B, num_heads, past_seq, head_dim]
    dummy_kv = []
    for _ in range(num_layers):
        dummy_kv.append(torch.zeros(B, num_heads, 0, head_dim, device=device))  # K
    for _ in range(num_layers):
        dummy_kv.append(torch.zeros(B, num_heads, 0, head_dim, device=device))  # V

    
    # 打包成元组
    dummy_inputs = (
        dummy_audio_codes,
        dummy_is_last,
        dummy_pre_conv_h,
        dummy_latent_buf,
        dummy_conv_h,
        *dummy_kv
    )
    
    # 5. 定义输入输出名称 (显式签名)
    input_names = [
        "audio_codes",
        "is_last",
        "pre_conv_history",
        "latent_buffer",
        "conv_history",
    ]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
    for i in range(num_layers):
        input_names.append(f"past_value_{i}")
    
    output_names = [
        "final_wav",
        "valid_samples",
        "next_pre_conv_history",
        "next_latent_buffer",
        "next_conv_history",
    ]
    for i in range(num_layers):
        output_names.append(f"next_key_{i}")
    for i in range(num_layers):
        output_names.append(f"next_value_{i}")
    
    # 6. 定义动态维度 (关键)
    dynamic_axes = {
        "audio_codes": {1: "num_frames"},  # N
        "pre_conv_history": {2: "pre_conv_len"},
        "latent_buffer": {2: "latent_len"},
        "conv_history": {2: "conv_len"},
        "final_wav": {1: "wav_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {2: f"past_seq_{i}"}
        dynamic_axes[f"past_value_{i}"] = {2: f"past_seq_{i}"}
        dynamic_axes[f"next_key_{i}"] = {2: f"next_seq_{i}"}
        dynamic_axes[f"next_value_{i}"] = {2: f"next_seq_{i}"}
    
    # 7. 导出 FP32 模型
    onnx_path_fp32 = os.path.join(EXPORT_DIR, ONNX_FILENAME)
    print(f"📦 正在导出 FP32 ONNX 到: {onnx_path_fp32}")
    print(f"   使用经典 JIT 路径 (dynamo=False)...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path_fp32,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,  # 关键：强制使用经典 JIT 路径
        )

    fp32_size = os.path.getsize(onnx_path_fp32) / 1024 / 1024
    print(f"✅ ONNX 导出成功！文件大小: {fp32_size:.2f} MB")

    # 8. 验证模型
    print("\n🔍 正在验证 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(onnx_path_fp32)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 模型校验通过！")

if __name__ == "__main__":
    main()
