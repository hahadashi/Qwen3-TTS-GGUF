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
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_audio")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行音频捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        
        # 固定参数
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        print("开始推理「今天天气不错」...")
        with torch.no_grad():
            audio_out = tts.generate_custom_voice(
                text="今天天气不错",
                speaker="Vivian",
                language="Chinese",
                **deterministic_kwargs
            )
            
        print(f"  Result Type: {type(audio_out)}")
        
        if isinstance(audio_out, tuple):
            print(f"  Result Tuple Len: {len(audio_out)}")
            # Usually (audio, sr) or (audio, something)
            audio_out = audio_out[0]
            
        if isinstance(audio_out, list):
             print(f"  Result List Len: {len(audio_out)}")
             if len(audio_out) > 0:
                 audio_out = audio_out[0]
        
        if isinstance(audio_out, torch.Tensor):
            audio_out = audio_out.detach().cpu().numpy()
            
        print(f"  Final Audio Shape: {audio_out.shape}, Dtype: {audio_out.dtype}")
        
        SAVE_PATH = os.path.join(SAVE_DIR, "official_full.wav")
        sf.write(SAVE_PATH, audio_out, 24000)
        print(f"✅ 官方音频已保存: {SAVE_PATH} (Length: {audio_out.shape})")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import pip
    try:
        import soundfile
    except ImportError:
        print("Installing soundfile...")
        pip.main(['install', 'soundfile'])
        
    main()
