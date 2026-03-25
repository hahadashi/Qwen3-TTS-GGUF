"""
Qwen3-TTS PyTorch Wrapper Package

提供基于原生 PyTorch 模型的流式推理 Wrapper 实现。

核心组件:
- StreamingEngine: 流式推理引擎
- TalkerWrapperV2, PredictorWrapperV2, DecoderWrapperV2: 核心组件
- CodecEncoderWrapper, SpeakerEncoderWrapper: 音频编码器封装
- AssetsManager, Sampler, PromptBuilder: 辅助组件
"""

__version__ = "0.3.0"

# ==================== 数据类 ====================
from .data import (
    VoiceAnchor,
    TTSConfig,
    PromptData,
    StreamState,
    GeneratorOutput,
)

# ==================== 状态类 ====================
from .states import (
    TalkerState,
    PredictorState,
    PredictorStateV2,
    DecoderState,
    StreamingState,
    StateInitializer,
    ContinuousSynthesisState,
)

# ==================== V2 Wrappers ====================
from .wrappers import (
    TalkerWrapperV2,
    PredictorWrapperV2,
    DecoderWrapperV2,
    StatefulDecoderWrapper,
    CodecEncoderWrapper,
    SpeakerEncoderWrapper,
    create_codec_encoder,
    create_speaker_encoder,
    # V3 Decoder (GGUF Compatible)
    DecoderWrapperV3,
    DecoderStateV3,
    create_decoder_v3,
)

# ==================== 流式组件 ====================
from .streaming import (
    StreamingEngine,
    StreamConfig,
    AssetsManager,
    Sampler,
    PromptBuilder,
)


# ==================== 便捷函数 ====================

def create_encoders(model):
    """
    便捷函数：创建所有 encoder wrappers

    Args:
        model: Qwen3TTSModel 或 Qwen3TTSForConditionalGeneration

    Returns:
        (codec_encoder, speaker_encoder): Encoder wrapper 实例

    Example:
        >>> from qwen3_tts import Qwen3TTSModel
        >>> from qwen3_tts_wrapper import create_encoders
        >>>
        >>> model = Qwen3TTSModel.from_pretrained(model_path)
        >>> codec_enc, speaker_enc = create_encoders(model)
        >>>
        >>> # 提取 speaker embedding
        >>> spk_emb = speaker_enc.encode("path/to/audio.wav")
        >>>
        >>> # 提取 codec codes
        >>> codes = codec_enc.encode(audio)
    """
    return (
        create_codec_encoder(model),
        create_speaker_encoder(model),
    )


def encode_audio(model, audio, return_numpy=True):
    """
    便捷函数：编码音频为 codec codes

    Args:
        model: Qwen3TTSModel 或 Qwen3TTSForConditionalGeneration
        audio: 输入音频 (文件路径、numpy 数组或 torch.Tensor)
        return_numpy: 是否返回 numpy 数组

    Returns:
        codes: [T, 16] 或 [B, T, 16] codec codes
    """
    encoder = create_codec_encoder(model)
    return encoder.encode(audio, return_numpy=return_numpy)


def extract_speaker_embedding(model, audio, sr=24000, return_numpy=True):
    """
    便捷函数：提取说话人嵌入

    Args:
        model: Qwen3TTSModel 或 Qwen3TTSForConditionalGeneration
        audio: 输入音频 (文件路径、numpy 数组或 torch.Tensor)
        sr: 音频采样率 (仅用于 numpy/tensor 输入验证)
        return_numpy: 是否返回 numpy 数组

    Returns:
        spk_emb: [2048] 说话人嵌入向量
    """
    encoder = create_speaker_encoder(model)
    return encoder.encode(audio, sr=sr, return_numpy=return_numpy)


__all__ = [
    # 引擎
    "StreamingEngine",
    "StreamConfig",

    # 数据类
    "VoiceAnchor",
    "PromptData",
    "GeneratorOutput",
    "TTSConfig",
    "StreamState",

    # V2 Wrappers
    "TalkerWrapperV2",
    "PredictorWrapperV2",
    "DecoderWrapperV2",
    "StatefulDecoderWrapper",

    # V3 Decoder (GGUF Compatible)
    "DecoderWrapperV3",
    "DecoderStateV3",
    "create_decoder_v3",

    # Encoder Wrappers
    "CodecEncoderWrapper",
    "SpeakerEncoderWrapper",
    "create_codec_encoder",
    "create_speaker_encoder",
    "create_encoders",
    "encode_audio",
    "extract_speaker_embedding",

    # 流式组件
    "AssetsManager",
    "Sampler",
    "PromptBuilder",

    # 状态类
    "TalkerState",
    "PredictorState",
    "PredictorStateV2",
    "DecoderState",
    "StreamingState",
    "StateInitializer",
    "ContinuousSynthesisState",
]
