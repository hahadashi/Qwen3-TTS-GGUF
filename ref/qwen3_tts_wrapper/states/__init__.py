"""
状态类模块

提供流式推理所需的各种状态类。
支持 DynamicCache (GQA 兼容) 和传统 Tuple KV Cache。
"""

from .state_classes import (
    # V1 状态类 (传统 Tuple KV Cache)
    TalkerStateV1,
    PredictorStateV1,

    # V2 状态类 (DynamicCache, GQA 兼容)
    TalkerState,
    PredictorState,
    PredictorStateV2,  # 别名

    # Decoder 状态
    DecoderState,

    # 完整流式状态
    StreamingState,

    # 状态初始化器
    StateInitializer,

    # 连续合成状态
    ContinuousSynthesisState,
)

__all__ = [
    # V1
    "TalkerStateV1",
    "PredictorStateV1",

    # V2 (推荐)
    "TalkerState",
    "PredictorState",
    "PredictorStateV2",

    # Decoder
    "DecoderState",

    # 完整状态
    "StreamingState",

    # 初始化器
    "StateInitializer",

    # 连续合成
    "ContinuousSynthesisState",
]
