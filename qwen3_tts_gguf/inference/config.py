from dataclasses import dataclass

@dataclass
class TTSConfig:
    """
    TTS 推理全链路控制参数封装。
    包含 Talker (生成语义特征) 和 Predictor (生成声学码) 两个阶段的独立采样控制。
    """
    # --- 大师控制 (Talker / Semantic Stage) ---
    do_sample: bool = True           # 是否开启随机采样。False 则使用 Greedy Search，结果稳定但机械。
    temperature: float = 0.8         # 采样温度。值越大越随机(情感起伏大)，过大可能崩字；值越小越严谨。
    top_p: float = 1.0               # 核采样阈值。只从累积概率达到 p 的 Token 中采样。
    top_k: int = 50                  # 候选集大小。采样时只考虑概率最高的前 k 个 Token。
    
    # --- 工匠控制 (Predictor / Acoustic Stage) ---
    sub_do_sample: bool = True      # 工匠阶段通常建议 False，使用确定性生成或低温度生成以保证音频稳定。
    sub_temperature: float = 0.8     # 工匠阶段的温度。调低可以减少语速抖动和电音感。
    sub_top_p: float = 1.0           # 工匠阶段的 Top-P。
    sub_top_k: int = 50              # 工匠阶段的 Top-K。
    
    # --- 全局生成控制 ---
    max_steps: int = 300             # 最大生成步数。决定了单次合成最长的持续时间。
