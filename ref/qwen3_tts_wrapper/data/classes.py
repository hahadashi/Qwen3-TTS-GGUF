"""
数据类定义

定义流式推理中使用的所有数据类。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import torch


@dataclass
class VoiceAnchor:
    """音色锚点

    用于存储音色相关数据，支持Base模型的参考音频模式。
    """
    # ========== Base模型所需 ==========
    speaker_embedding: torch.Tensor        # [1, hidden] 说话人嵌入
    reference_codes: torch.Tensor          # [T, 16] 参考音频codes

    # ========== ICL模式所需 ==========
    ref_text: Optional[str] = None         # 参考文本（可选）

    # ========== 元数据 ==========
    name: str = "default"
    lang: str = "zh"

    @classmethod
    def from_audio(
        cls,
        audio: torch.Tensor,
        engine: 'Qwen3TTSWrapperEngine',
        ref_text: Optional[str] = None
    ) -> 'VoiceAnchor':
        """从音频创建音色锚点"""
        # 编码音频
        codes, speaker_emb = engine.encode_reference_audio(audio)

        return cls(
            speaker_embedding=speaker_emb,
            reference_codes=codes,
            ref_text=ref_text
        )


@dataclass
class TTSConfig:
    """TTS生成配置"""
    # ========== 采样参数 ==========
    temperature: float = 1.0               # 温度参数
    top_k: int = 0                         # Top-K采样
    top_p: float = 1.0                     # Top-P采样
    min_p: float = 0.0                     # Min-P采样
    repeat_penalty: float = 1.0            # 重复惩罚

    # ========== 生成控制 ==========
    max_steps: int = 500                   # 最大生成步数
    seed: Optional[int] = None             # 随机种子

    # ========== 流式控制 ==========
    chunk_size: int = 6                    # 音频块大小(帧数)
    enable_streaming: bool = True          # 启用流式

    # ========== 解码控制 ==========
    decode_per_frame: bool = True          # 每帧解码


@dataclass
class PromptData:
    """构建好的Prompt数据"""

    # ========== Prefill阶段输入 ==========
    prefill_embeds: torch.Tensor           # [1, prefill_len, hidden]
    prefill_attention_mask: torch.Tensor  # [1, prefill_len]
    prefill_position_ids: torch.Tensor    # [1, prefill_len, 4]

    # ========== Trailing Text (步内注入) ==========
    trailing_text_embeds: torch.Tensor     # [1, trail_len, hidden]

    # ========== 元数据 ==========
    text: str                              # 原始文本
    text_ids: List[int]                    # 文本token IDs
    speaker_embedding: torch.Tensor        # [1, hidden]

    # ========== ICL模式参考数据 ==========
    ref_codes: Optional[torch.Tensor]      # [ref_len, 16]
    ref_text: Optional[str]                # 参考文本

    @property
    def prefill_len(self) -> int:
        """Prefill序列长度"""
        return self.prefill_embeds.shape[1]

    @property
    def trail_len(self) -> int:
        """Trailing文本长度"""
        return self.trailing_text_embeds.shape[1]


@dataclass
class StreamState:
    """流式推理的完整状态"""

    # ========== Talker状态 ==========
    talker_kv_cache: Optional[Tuple]       # KV Cache (num_layers × 2 × [B, heads, seq, dim])
    position: int = 0                       # 当前位置
    last_hidden: Optional[torch.Tensor] = None  # [1, hidden] Talker最后的hidden state

    # ========== Decoder状态 ==========
    decoder_state: Optional['DecoderState'] = None  # Decoder流式状态

    # ========== 文本池状态 ==========
    text_position: int = 0                  # trailing_text当前索引
    text_pool_pos: int = 0                  # 音频反馈用的池位置

    # ========== 已生成数据 ==========
    generated_codes: List[torch.Tensor] = field(default_factory=list)
    last_audio_embeddings: Optional[List[torch.Tensor]] = None  # 16层embeddings

    # ========== 控制标志 ==========
    finished: bool = False

    def increment_position(self) -> int:
        """递增位置并返回新位置"""
        self.position += 1
        return self.position


@dataclass
class GeneratorOutput:
    """单步生成输出"""

    # ========== 生成结果 ==========
    codes: Optional[torch.Tensor]           # [1, 16] 当前帧codes
    audio_chunk: Optional[torch.Tensor]     # [samples] 解码的音频块

    # ========== 状态信息 ==========
    is_last: bool                           # 是否最后一帧
    position: int = 0                       # 当前位置
    state: Optional[StreamState] = None     # 更新后的状态

    # ========== 连续合成支持 ==========
    final_state: Optional['StreamingState'] = None  # 最终状态 (仅 is_last=True 时有效)

    # ========== 时间信息 (可选) ==========
    step: int = 0                           # 当前步数
    latency_ms: float = 0.0                 # 本步延迟


@dataclass
class TTSResult:
    """TTS合成结果 (GGUF风格)

    参考 GGUF TTSResult 实现，包含完整的合成结果。
    """

    # ========== 主要输出 ==========
    audio: torch.Tensor                     # [samples] 音频波形
    text: str                               # 合成文本

    # ========== 中间结果 ==========
    codes: Optional[List[torch.Tensor]] = None  # 所有帧的codes [frames, 16]
    summed_embeds: Optional[List[torch.Tensor]] = None  # 音频反馈embeddings

    # ========== 性能统计 ==========
    timing: Optional[object] = None         # Timing对象，包含各阶段耗时

    # ========== 元数据 ==========
    sample_rate: int = 24000                # 采样率
    duration_ms: float = 0.0                # 音频时长(ms)

    def __post_init__(self):
        """自动计算音频时长"""
        if self.audio is not None and self.audio.numel() > 0:
            self.duration_ms = self.audio.shape[0] / self.sample_rate * 1000
