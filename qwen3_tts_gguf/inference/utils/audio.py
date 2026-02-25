"""
audio.py - 音频预处理工具类
职责：使用 pydub 提供万能格式转换，确保输入编码器的音频符合 24000Hz Mono 16bit 规范。
"""
import os
import io
import numpy as np
from pathlib import Path
from typing import Union, Optional
from pydub import AudioSegment
from .. import logger

def preprocess_audio(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    预处理音频文件。
    1. 检查是否存在。
    2. 使用 pydub 加载（支持 mp3, opus, m4a 等）。
    3. 强制转换为 24000Hz, Mono, 16bit。
    4. 返回 numpy float32 数组。
    """
    p = Path(path)
    if not p.exists():
        logger.error(f"❌ 音频文件不存在: {p}")
        return None
        
    try:
        # 1. 加载音频
        audio = AudioSegment.from_file(str(p))
        
        # 2. 标准化参数 (24kHz, Mono, 16-bit)
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        # 3. 转换为 numpy float32 (librosa 风格: [-1, 1])
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        # 16-bit PCM max is 32768
        samples /= 32768.0
        
        if len(samples) == 0:
            logger.warning(f"⚠️ 音频文件为空: {p}")
            return None
            
        return samples
    except Exception as e:
        logger.error(f"❌ 音频预处理失败 ({p.name}): {e}")
        return None

def save_temp_wav(samples: np.ndarray, sr=24000) -> str:
    """将内存中的 samples 保存为临时 wav 文件供 librosa 加载（如果需要）"""
    import soundfile as sf
    import tempfile
    temp_p = tempfile.mktemp(suffix=".wav")
    sf.write(temp_p, samples, sr)
    return temp_p
