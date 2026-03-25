"""
Wrapper 组件模块

提供各组件的封装，支持流式推理。

组件列表:
- TalkerWrapperV2: Talker 组件封装 (GQA 兼容)
- PredictorWrapperV2: Predictor 组件封装 (GQA 兼容)
- DecoderWrapperV2 / StatefulDecoderWrapper: Decoder 组件封装 (状态化)
- DecoderWrapperV3: Decoder 组件封装 (GGUF 完全兼容)
- CodecEncoderWrapper: Codec Encoder 组件封装 (音频编码)
- SpeakerEncoderWrapper: Speaker Encoder 组件封装 (说话人嵌入)
"""

# V2 Wrappers (GQA 兼容)
from .talker_v2 import TalkerWrapperV2, TalkerState as TalkerStateV2
from .predictor_v2 import PredictorWrapperV2, PredictorStateV2
from .decoder_v2 import (
    StatefulDecoderWrapper,
    StatefulDecoderState,
    DecoderWrapperV2,
)
from .codec_encoder import CodecEncoderWrapper, create_codec_encoder
from .speaker_encoder import SpeakerEncoderWrapper, create_speaker_encoder

# V3 Decoder (GGUF 完全兼容)
from .decoder_v3 import (
    DecoderWrapperV3,
    DecoderStateV3,
    create_decoder_v3,
)

__all__ = [
    "TalkerWrapperV2",
    "TalkerStateV2",
    "PredictorWrapperV2",
    "PredictorStateV2",
    "DecoderWrapperV2",
    "StatefulDecoderWrapper",
    "StatefulDecoderState",
    "DecoderWrapperV3",
    "DecoderStateV3",
    "create_decoder_v3",
    "CodecEncoderWrapper",
    "create_codec_encoder",
    "SpeakerEncoderWrapper",
    "create_speaker_encoder",
]
