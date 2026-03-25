"""
状态类定义

定义流式推理所需的各种状态类。
支持 DynamicCache (GQA 兼容) 和传统 Tuple KV Cache。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch

# HuggingFace DynamicCache
try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None


# ========== V1 状态类 (传统 Tuple KV Cache) ==========

@dataclass
class TalkerStateV1:
    """Talker 流式状态 V1 (传统 Tuple KV Cache)

    Talker 组件自回归生成 codec_0 时需要维护的状态。

    Attributes:
        past_key_values: KV Cache (28层 × 2: key/value)
                        形状: num_layers × 2 × [batch, heads, seq_len, head_dim]
        past_codes: 已生成的 codec_0 序列 [batch, seq_len]
        batch_size: 批次大小
    """
    past_key_values: Optional[Tuple] = None
    past_codes: Optional[torch.Tensor] = None
    batch_size: int = 1

    def get_kv_cache_shape(self) -> Optional[Tuple]:
        """获取 KV Cache 的形状信息"""
        if self.past_key_values is None:
            return None
        num_layers = len(self.past_key_values) // 2
        return (
            num_layers,
            self.batch_size,
            self.past_key_values[0].shape[1],  # num_heads
            self.past_key_values[0].shape[2],  # seq_len
            self.past_key_values[0].shape[3]   # head_dim
        )


@dataclass
class PredictorStateV1:
    """Predictor 流式状态 V1 (传统 Tuple KV Cache)

    Predictor 组件生成 codec_1~15 时需要维护的状态。

    Attributes:
        past_key_values: KV Cache (5层 × 2)
        past_codes: 已生成的 codes 序列
        batch_size: 批次大小
    """
    past_key_values: Optional[Tuple] = None
    past_codes: Optional[torch.Tensor] = None
    batch_size: int = 1


# ========== V2 状态类 (DynamicCache, GQA 兼容) ==========

@dataclass
class TalkerState:
    """Talker 流式状态 V2 (DynamicCache, GQA 兼容)

    使用 HuggingFace DynamicCache 自动处理 GQA。

    Attributes:
        past_key_values: DynamicCache 对象 (自动适配 GQA)
        position: 当前位置
        generated_codes: 已生成的 codes 列表
        batch_size: 批次大小
    """
    past_key_values: Optional['DynamicCache'] = None
    position: int = 0
    generated_codes: List[int] = field(default_factory=list)
    batch_size: int = 1

    def get_cache_length(self) -> int:
        """获取 cache 序列长度"""
        if self.past_key_values is None:
            return 0
        if hasattr(self.past_key_values, 'get_seq_length'):
            return self.past_key_values.get_seq_length(layer_idx=0)
        return 0


@dataclass
class PredictorState:
    """Predictor 流式状态 V2 (DynamicCache, GQA 兼容)

    使用 HuggingFace DynamicCache 自动处理 GQA。

    Attributes:
        past_key_values: DynamicCache 对象
        position: 当前位置
        batch_size: 批次大小
    """
    past_key_values: Optional['DynamicCache'] = None
    position: int = 0
    batch_size: int = 1

    def get_cache_length(self) -> int:
        """获取 cache 序列长度"""
        if self.past_key_values is None:
            return 0
        if hasattr(self.past_key_values, 'get_seq_length'):
            return self.past_key_values.get_seq_length(layer_idx=0)
        return 0


# 别名，保持兼容
PredictorStateV2 = PredictorState


# ========== 连续合成状态 ==========

@dataclass
class ContinuousSynthesisState:
    """连续合成状态

    用于在多次合成之间传递状态，实现连续语音合成。

    参考 GGUF 的 final_state 机制:
    - 保存 Talker 的 KV Cache 和位置信息
    - 保存 Decoder 的完整状态
    - 可序列化以便存储和恢复

    使用示例:
    ```python
    # 第一次合成
    for output in engine.generate_stream(prompt1, config):
        if output.is_last:
            final_state = output.final_state

    # 第二次合成 (从上一次状态继续)
    for output in engine.generate_stream(prompt2, config, initial_state=final_state):
        ...
    ```
    """

    # ========== Talker 状态 ==========
    talker_position: int = 0                          # Talker 当前位置
    talker_generated_codes: List[int] = field(default_factory=list)  # 已生成的 codec_0

    # ========== Decoder 状态 ==========
    decoder_pre_conv_history: Optional[torch.Tensor] = None  # [B, 512, history]
    decoder_latent_buffer: Optional[torch.Tensor] = None     # [B, 1024, history]
    decoder_conv_history: Optional[torch.Tensor] = None      # [B, 1024, history]

    # ========== 元数据 ==========
    speaker_embedding: Optional[torch.Tensor] = None  # 说话人嵌入 [1, hidden]
    sample_rate: int = 24000                          # 采样率

    def to(self, device: torch.device) -> 'ContinuousSynthesisState':
        """移动状态到指定设备"""
        return ContinuousSynthesisState(
            talker_position=self.talker_position,
            talker_generated_codes=self.talker_generated_codes.copy(),
            decoder_pre_conv_history=self.decoder_pre_conv_history.to(device) if self.decoder_pre_conv_history is not None else None,
            decoder_latent_buffer=self.decoder_latent_buffer.to(device) if self.decoder_latent_buffer is not None else None,
            decoder_conv_history=self.decoder_conv_history.to(device) if self.decoder_conv_history is not None else None,
            speaker_embedding=self.speaker_embedding.to(device) if self.speaker_embedding is not None else None,
            sample_rate=self.sample_rate,
        )

    def to_cpu(self) -> 'ContinuousSynthesisState':
        """移动状态到 CPU (用于存储)"""
        return self.to(torch.device('cpu'))

    @classmethod
    def from_streaming_state(cls, state: 'StreamingState') -> 'ContinuousSynthesisState':
        """从 StreamingState 创建 ContinuousSynthesisState

        注意: Talker 的 KV Cache (DynamicCache) 无法直接序列化，
        需要在重新开始时重新 prefill。
        这里只保存可移植的状态信息。
        """
        # 提取 Decoder 状态 (StatefulDecoderState 的属性)
        decoder_pre_conv = None
        decoder_latent = None
        decoder_conv = None
        decoder_position = 0
        decoder_skip = 0

        if state.decoder_state is not None:
            ds = state.decoder_state
            if hasattr(ds, 'pre_conv_history') and ds.pre_conv_history is not None:
                decoder_pre_conv = ds.pre_conv_history.clone()
            if hasattr(ds, 'latent_audio') and ds.latent_audio is not None:
                decoder_latent = ds.latent_audio.clone()
            if hasattr(ds, 'position'):
                decoder_position = ds.position
            if hasattr(ds, 'skip_samples'):
                decoder_skip = ds.skip_samples

        return cls(
            talker_position=state.text_pool_index,
            talker_generated_codes=[],  # codes 在 StreamingState.generated_codes 中是 Tensor
            decoder_pre_conv_history=decoder_pre_conv,
            decoder_latent_buffer=decoder_latent,
            decoder_conv_history=decoder_conv,  # StatefulDecoderState 没有 conv_history
        )

    def to_decoder_state(self, batch_size: int = 1, device: torch.device = None) -> 'DecoderState':
        """转换为 DecoderState

        注意: StatefulDecoderState 和 DecoderState 有不同的属性结构。
        这里只设置 pre_conv_history，其他状态由 Decoder 内部管理。
        """
        from .state_classes import DecoderState
        device = device or torch.device('cpu')

        return DecoderState(
            pre_conv_history=self.decoder_pre_conv_history if self.decoder_pre_conv_history is not None
                else torch.zeros(batch_size, 512, 0, device=device),
            latent_buffer=self.decoder_latent_buffer if self.decoder_latent_buffer is not None
                else torch.zeros(batch_size, 1024, 0, device=device),
            conv_history=self.decoder_conv_history if self.decoder_conv_history is not None
                else torch.zeros(batch_size, 1024, 0, device=device),
            past_key_values=None,  # KV Cache 需要重新构建
            batch_size=batch_size,
        )


# ========== Decoder 状态 ==========

@dataclass
class DecoderState:
    """Decoder 流式状态

    与 ONNX 导出的状态定义完全对应。
    Decoder 将 audio codes 解码为音频波形时需要维护的状态。

    三段式架构状态:
    - Part 1 (Pre-Conv): pre_conv_history
    - Part 2 (Transformer): past_key_values
    - Part 3 (Upsample): latent_buffer, conv_history

    Attributes:
        pre_conv_history: RVQ 解码后的预卷积历史 [B, 512, history], history ∈ [0, 2]
        latent_buffer: Transformer 输出的缓冲 [B, 1024, history], history ∈ [0, 4]
        conv_history: 上采样卷积链的历史 [B, 1024, history], history ∈ [0, 4]
        past_key_values: Transformer 层的 KV Cache
                         num_layers × 2 × [B, heads, seq, dim]
        batch_size: 批次大小
    """
    pre_conv_history: torch.Tensor
    latent_buffer: torch.Tensor
    conv_history: torch.Tensor
    past_key_values: Optional[Tuple] = None
    batch_size: int = 1

    def get_kv_cache_shape(self) -> Optional[Tuple]:
        """获取 KV Cache 的形状信息"""
        if self.past_key_values is None:
            return None
        num_layers = len(self.past_key_values) // 2
        return (
            num_layers,
            self.batch_size,
            self.past_key_values[0].shape[1],  # num_heads
            self.past_key_values[0].shape[2],  # seq_len
            self.past_key_values[0].shape[3]   # head_dim
        )

    def to(self, device: torch.device) -> 'DecoderState':
        """移动状态到指定设备"""
        return DecoderState(
            pre_conv_history=self.pre_conv_history.to(device),
            latent_buffer=self.latent_buffer.to(device),
            conv_history=self.conv_history.to(device),
            past_key_values=self.past_key_values,  # KV Cache 会在推理时自动处理
            batch_size=self.batch_size,
        )


# ========== 完整流式状态 ==========

@dataclass
class StreamingState:
    """完整流式状态容器

    管理整个流式推理流程的状态。

    Attributes:
        talker_state: Talker 组件状态
        predictor_state: Predictor 组件状态 (每帧重置)
        decoder_state: Decoder 组件状态
        text_pool_index: 文本池当前索引
        generated_codes: 已生成的 codes 列表
        finished: 是否已完成
    """
    talker_state: Optional[TalkerState] = None
    predictor_state: Optional[PredictorState] = None
    decoder_state: Optional[DecoderState] = None
    text_pool_index: int = 0
    generated_codes: List[torch.Tensor] = field(default_factory=list)
    finished: bool = False


# ========== 状态初始化器 ==========

class StateInitializer:
    """状态初始化器

    提供统一的状态初始化接口。
    """

    def __init__(self, device: torch.device = None):
        """
        Args:
            device: 默认设备
        """
        self.device = device or torch.device('cpu')

    def init_talker_state(self, batch_size: int = 1) -> TalkerState:
        """初始化 Talker 状态"""
        return TalkerState(
            past_key_values=DynamicCache() if HAS_DYNAMIC_CACHE else None,
            position=0,
            generated_codes=[],
            batch_size=batch_size,
        )

    def init_predictor_state(self, batch_size: int = 1) -> PredictorState:
        """初始化 Predictor 状态"""
        return PredictorState(
            past_key_values=DynamicCache() if HAS_DYNAMIC_CACHE else None,
            position=0,
            batch_size=batch_size,
        )

    def init_decoder_state(
        self,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> DecoderState:
        """初始化 Decoder 状态

        Args:
            batch_size: 批次大小
            device: 设备
            dtype: 数据类型

        Returns:
            DecoderState: 初始化的状态
        """
        device = device or self.device

        return DecoderState(
            pre_conv_history=torch.zeros(batch_size, 512, 0, device=device, dtype=dtype),
            latent_buffer=torch.zeros(batch_size, 1024, 0, device=device, dtype=dtype),
            conv_history=torch.zeros(batch_size, 1024, 0, device=device, dtype=dtype),
            past_key_values=None,
            batch_size=batch_size,
        )

    def init_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> StreamingState:
        """初始化完整流式状态

        Args:
            batch_size: 批次大小
            device: 设备
            dtype: Decoder 状态的数据类型

        Returns:
            StreamingState: 初始化的完整状态
        """
        return StreamingState(
            talker_state=self.init_talker_state(batch_size),
            predictor_state=None,  # 每帧重置
            decoder_state=self.init_decoder_state(batch_size, device, dtype),
            text_pool_index=0,
            generated_codes=[],
            finished=False,
        )
