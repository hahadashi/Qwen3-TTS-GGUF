# coding=utf-8
import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from pathlib import Path
from export_config import EXPORT_DIR

def convert_to_fp16(input_path):
    output_path = input_path.replace(".fp32.onnx", ".fp16.onnx")
    print(f"\n[FP16] Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        model = onnx.load(input_path)
        # 使用 ORT Transformers 转换以获得更好的 DML 兼容性
        # 保留 LayerNorm 和 Softmax 为 FP32 对稳定性至关重要
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=False,
            min_positive_val=1e-7,
            max_finite_val=65504,
            op_block_list=['LayerNormalization', 'Softmax', 'Range'] 
        )
        onnx.save(model_fp16, output_path)
        print(f"   ✅ [成功] 已保存 FP16 模型。")
    except Exception as e:
        print(f"   ❌ [失败] FP16 转换错误: {e}")

def convert_to_int8(input_path):
    output_path = input_path.replace(".fp32.onnx", ".int8.onnx")
    print(f"\n[INT8] Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            op_types_to_quantize=["MatMul", "Attention", "Conv"], # 权重量化的核心
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"   ✅ [成功] 已保存 INT8 模型。")
    except Exception as e:
        print(f"   ❌ [失败] INT8 量化错误: {e}")

def convert_to_int4(input_path):
    output_path = input_path.replace(".fp32.onnx", ".int4.onnx")
    print(f"\n[INT4] Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        quantizer = MatMulNBitsQuantizer(
            model=input_path,
            block_size=128,
            is_symmetric=False,
            accuracy_level=None
        )
        quantizer.process()
        quantizer.model.save_model_to_file(output_path)
        print(f"   ✅ [成功] 已保存 INT4 模型。")
    except Exception as e:
        print(f"   ❌ [失败] INT4 量化错误: {e}")

def main():
    print("--- 正在开始针对 Qwen3-TTS ONNX 模型的批量量化/转换 ---")
    
    export_path = Path(EXPORT_DIR)
    if not export_path.exists():
        print(f"❌ 错误: 目录 {export_path} 不存在。")
        return

    # 定义目标文件列表
    targets = [
        "qwen3_tts_codec_encoder.fp32.onnx",
        "qwen3_tts_speaker_encoder.fp32.onnx",
        "qwen3_tts_decoder.fp32.onnx"
    ]
    
    for filename in targets:
        model_path = str(export_path / filename)
        
        if not os.path.exists(model_path):
            print(f"\n⚠️ 跳过: 找不到基准 FP32 模型: {filename}")
            continue
            
        print(f"\n>>> 处理模型: {filename}")
        
        # 1. 转换为 FP16 
        convert_to_fp16(model_path)
        
        # # 2. 动态量化为 INT8 
        convert_to_int8(model_path)
        
        # # 3. 权重量化为 INT4 
        # convert_to_int4(model_path)

    print("\n--- 所有转换工作已完成 ---")

if __name__ == "__main__":
    main()
