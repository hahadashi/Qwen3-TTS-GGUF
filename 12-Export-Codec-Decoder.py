import os
import sys
import torch
from qwen3_tts_gguf import logger
from qwen3_tts_gguf.codec_export import CodecExportWrapper
from qwen3_tts_gguf.modeling_tokenizer import Qwen3TTSTokenizerV2Model

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    OUTPUT_DIR = r'./model'
    ONNX_PATH = os.path.join(OUTPUT_DIR, 'qwen3_tts_decoder.onnx')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 路径检测逻辑
    logger.info(f"正在加载模型: {MODEL_DIR}")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    if load_path != MODEL_DIR:
        logger.info(f"检测到 speech_tokenizer 子目录，使用: {load_path}")
    
    # 2. 加载 PyTorch 模型
    try:
        model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 3. 准备导出包装器 (Decoder)
    logger.info("准备 ONNX 导出包装器 (Decoder)...")
    decoder_wrapper = CodecExportWrapper(model).eval()
    
    # 4. Dummy Input
    num_quantizers = model.config.decoder_config.num_quantizers
    # Shape: [Batch=1, Time=100, NumQuantizers]
    dummy_input = torch.randint(0, 1024, (1, 100, num_quantizers), dtype=torch.long)
    
    # 5. Export
    logger.info(f"开始导出 Decoder ONNX 到: {ONNX_PATH}")
    try:
        torch.onnx.export(
            decoder_wrapper,
            (dummy_input,),
            ONNX_PATH,
            input_names=['audio_codes'],
            output_names=['audio_values'],
            dynamic_axes={
                'audio_codes': {0: 'batch_size', 1: 'sequence_length'},
                'audio_values': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=18, # 最佳实践：使用较新 Opset
            do_constant_folding=True
        )
        logger.info("✅ Decoder ONNX 导出成功！")
    except Exception as e:
        logger.error(f"❌ Decoder ONNX 导出失败: {e}")
        return

    # 6. 验证
    logger.info("验证导出结果...")
    try:
        import onnxruntime
        import numpy as np
        
        sess = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        
        with torch.no_grad():
            torch_out = decoder_wrapper(dummy_input).numpy()
            
        onnx_out = sess.run(None, {'audio_codes': dummy_input.numpy()})[0]
        
        if np.allclose(torch_out, onnx_out, atol=1e-4):
            logger.info("数值验证通过！(PyTorch vs ONNX)")
        else:
            diff = np.max(np.abs(torch_out - onnx_out))
            logger.warning(f"数值验证存在差异，最大误差: {diff:.6f}")
            
    except ImportError:
        logger.warning("未安装 onnxruntime，跳过验证。")

if __name__ == "__main__":
    main()
