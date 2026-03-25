"""
配置类定义
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSConfig:
    """TTS合成配置

    Attributes:
        max_steps: 最大生成步数
        temperature: 采样温度 (Talker)
        sub_temperature: 子采样温度 (Predictor)
        top_k: Top-K采样参数
        top_p: Top-P采样参数
        repetition_penalty: 重复惩罚
        seed: 随机种子 (可选)
        streaming: 是否启用流式模式
    """
    max_steps: int = 400
    temperature: float = 0.7
    sub_temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    streaming: bool = True


@dataclass
class EngineConfig:
    """引擎配置

    Attributes:
        device: 计算设备 (cuda/cpu/npu)
        dtype: 数据类型
        low_cpu_mem_usage: 是否使用低内存模式
    """
    device: str = "cuda"
    dtype: str = "float16"
    low_cpu_mem_usage: bool = False
