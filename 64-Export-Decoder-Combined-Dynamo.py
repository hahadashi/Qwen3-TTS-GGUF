import os
import torch
import torch.onnx
import logging
from qwen3_tts_gguf.export.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
from qwen3_tts_gguf.export.codec_export import StatefulDecoderDynamoCombined
from export_config import MODEL_DIR, EXPORT_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Export-Combined")

def main():
    # 1. 配置路径
    ONNX_FILENAME = "qwen3_tts_decoder.onnx"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, ONNX_FILENAME)
    
    device = "cpu"
    
    # 2. 加载模型
    logger.info("🚀 正在加载模型...")
    tokenizer_load_path = os.path.join(MODEL_DIR, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_DIR, "speech_tokenizer")) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_load_path).to(device)
    
    # 【CRITICAL】应用 Transformer 导出经验中的 Patch
    decoder_config = model.config.decoder_config
    decoder_config._attn_implementation = "eager"
    decoder_config.head_dim = decoder_config.hidden_size // decoder_config.num_attention_heads

    
    # 包装全量模型
    decoder_combined = StatefulDecoderDynamoCombined(model.decoder).to(device).eval()
    
    # 3. 准备 Dummy Inputs
    B, N = 1, 3
    Q = model.config.decoder_config.num_quantizers # 8
    
    dummy_codes = torch.zeros(B, N, Q, dtype=torch.long)
    dummy_pre_conv_hist = torch.zeros(B, 512, 0)
    dummy_latent_buffer = torch.zeros(B, 1024, 0)
    dummy_conv_history = torch.zeros(B, 1024, 0)
    dummy_is_last = torch.tensor([0.0])
    
    # KV Cache 维度
    num_layers = decoder_combined.num_layers
    num_heads = model.config.decoder_config.num_key_value_heads
    head_dim = 64
    init_kv_len = 0
    dummy_kv = [torch.zeros(B, num_heads, init_kv_len, head_dim) for _ in range(num_layers * 2)]
    
    # 4. 定义动态维度 (Modern Dynamo Style)
    batch = torch.export.Dim("batch", min=1, max=8)
    num_frames = torch.export.Dim("num_frames", min=1, max=1024)
    past_seq = torch.export.Dim("past_seq", min=0, max=72)
    pre_conv_seq = torch.export.Dim("pre_conv_seq", min=0, max=2)
    latent_seq = torch.export.Dim("latent_seq", min=0, max=4)
    conv_seq = torch.export.Dim("conv_seq", min=0, max=4)
    
    # 匹配 forward(audio_codes, pre_conv_hist, latent_buffer, conv_history, is_last, *past_kv_flat)
    dynamic_shapes = (
        {0: batch, 1: num_frames},      # audio_codes
        {0: batch, 2: pre_conv_seq},    # pre_conv_history
        {0: batch, 2: latent_seq},      # latent_buffer
        {0: batch, 2: conv_seq},        # conv_history
        None,                           # is_last
        tuple([{0: batch, 2: past_seq}] * (num_layers * 2)) # *past_kv_flat
    )
    
    input_names = ["audio_codes", "pre_conv_history", "latent_buffer", "conv_history", "is_last"]
    output_names = ["final_wav", "valid_samples", "next_pre_conv_hist", "next_latent_buf", "next_conv_hist"]
    
    for i in range(num_layers): input_names.append(f"past_key_{i}")
    for i in range(num_layers): input_names.append(f"past_value_{i}")
    for i in range(num_layers): output_names.append(f"next_key_{i}")
    for i in range(num_layers): output_names.append(f"next_value_{i}")

    # 5. 执行联合导出
    logger.info(f"📦 正在使用 dynamo=True 执行 Decoder 全量联合导出: {onnx_path}")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder_combined,
                (dummy_codes, dummy_pre_conv_hist, dummy_latent_buffer, dummy_conv_history, dummy_is_last, *dummy_kv),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                opset_version=18,
                dynamo=True,
            )
        
        file_size = os.path.getsize(onnx_path) / 1024 / 1024
        logger.info(f"✅ 联合 ONNX 导出成功！文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
