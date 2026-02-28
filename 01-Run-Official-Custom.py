import os
import sys
import time
import torch
import random
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path
from export_config import MODEL_CUSTOM as MODEL_DIR

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))

from qwen_tts import Qwen3TTSModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def play_audio(file_path):
    print(f"正在播放 {file_path}...")
    # Use powershell to play audio
    try:
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"], check=True)
    except Exception as e:
        print(f"播放音频失败: {e}")

def main():
    # 使用 GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 检查模型文件
    if not os.path.exists(MODEL_DIR):
        print(f"错误：找不到模型路径：{MODEL_DIR}")
        return

    print(f"正在从 {MODEL_DIR} 加载模型")
    
    try:
        print("开始加载模型...")
        
        # 定义数据类型
        dtype = torch.bfloat16
        
        # 载入模型
        t_load_start = time.time()
        set_seed(47)
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_DIR,
            device_map=device,
            dtype=dtype,
        )
        t_load_end = time.time()
        load_time = t_load_end - t_load_start
        print(f"模型加载完成，耗时 {load_time:.4f} 秒。")
        
        # 连续生成多个音频
        for i in range(9):
            print(f"\n--- 正在生成第 {i+1} 个音频 ---")
            t_infer_start = time.time()
            wavs, sr = tts.generate_custom_voice(
                text="今天天气好。",
                language="Chinese",
                speaker='Vivian',
                instruct="", 
                temperature=0.6, 
                subtalker_temperature=0.6, 
            )
            t_infer_end = time.time()
            infer_time = t_infer_end - t_infer_start
            print(f"推理完成，耗时 {infer_time:.4f} 秒。")

            # 保存并播放音频
            output_file = f"output/custom_{i+1}.wav"
            sf.write(output_file, wavs[0], sr)
            print(f"音频已保存至: {output_file}")
            play_audio(output_file)

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
