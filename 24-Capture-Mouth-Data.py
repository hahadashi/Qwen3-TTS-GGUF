import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_mouth")
os.makedirs(SAVE_DIR, exist_ok=True)

class MouthCapture:
    def __init__(self):
        self.captured = False

capture = MouthCapture()

def decoder_post_hook(module, input, output):
    """
    拦截分词器解码器 (Mouth) 的输入和输出。
    input[0]: audio_codes [Batch, Q, Time] (注意 Qwen3 内部 transpose 了)
    但我们要比对的是导出的 API，导出包装器输入为 [B, T, Q]。
    """
    if capture.captured:
        return
    
    # input[0] 是传递给 decoder 的 codes
    # 官方模型内部在调用 decoder 前会进行 transpose(1, 2)
    # 所以这里的 input[0] 形状应该是 [B, Q, T]
    codes = input[0].detach().cpu()
    
    # 转换回导出包装器期望的格式: [B, T, Q]
    export_format_codes = codes.transpose(1, 2).to(torch.int64) # 强制转长整型
    
    # output 是波形: [B, 1, T_wav]
    waveform = output.detach().info() if hasattr(output, 'info') else output # 防止 ModelOutput
    if isinstance(waveform, (list, tuple)):
        waveform = waveform[0]
    elif hasattr(waveform, 'audio_values'):
        waveform = waveform.audio_values
        
    waveform_np = waveform.detach().cpu().to(torch.float32).numpy()
    
    np.save(os.path.join(SAVE_DIR, "input_codes.npy"), export_format_codes.numpy())
    np.save(os.path.join(SAVE_DIR, "output_waveform.npy"), waveform_np)
    
    # 同时也保存一个 wav 方便试听
    sf.write(os.path.join(SAVE_DIR, "official_out.wav"), waveform_np.flatten(), 24000)
    
    print(f"[CAPTURE] Mouth Data Saved. Codes Shape: {export_format_codes.shape}, Waveform Shape: {waveform_np.shape}")
    capture.captured = True

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行嘴巴数据捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        
        # 准确定位分词器的解码器
        # tts.model.speech_tokenizer.model.decoder
        decoder = tts.model.speech_tokenizer.model.decoder
        decoder.register_forward_hook(decoder_post_hook)
        
        # 固定随机性
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        print("开始推理「你好，这是嘴巴组件的验证测试」...")
        with torch.no_grad():
            tts.generate_custom_voice(
                text="你好，这是嘴巴组件的验证测试",
                speaker="Vivian",
                language="Chinese",
                **deterministic_kwargs
            )
        
        if capture.captured:
            print(f"\n✅ 嘴巴数据捕获完成！保存在: {SAVE_DIR}")
        else:
            print(f"\n❌ 捕获失败，Hook 未触发。")
            
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
