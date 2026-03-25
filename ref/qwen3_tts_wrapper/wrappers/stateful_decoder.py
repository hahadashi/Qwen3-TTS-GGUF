"""
Overlap-based Stateful Decoder Wrapper - 基于 left_context overlap 的状态化解码器

对标 GGUF StatefulDecoder 的行为:
- 维护 codes 历史作为 left_context
- 支持 ref_codes 预解码 (初始化历史)
- chunk 间连续平滑

关键区别:
- GGUF: 使用显式状态 (KV cache + conv history + latent buffer)
- 本实现: 使用 left_context overlap 达到相同效果 (对标 Native API chunked_decode)

作者: Claude
日期: 2026-03-25
"""

import torch
from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class OverlapDecoderState:
    """状态化解码器状态"""
    code_history: List[torch.Tensor] = field(default_factory=list)  # 历史帧 codes
    total_frames: int = 0  # 已解码的总帧数


class OverlapDecoderWrapper:
    """
    PyTorch 状态化解码器封装 (基于 left_context overlap)

    对标 GGUF StatefulDecoder 的行为:
    - 维护 codes 历史作为 left_context
    - 支持 ref_codes 预解码 (初始化历史)
    - chunk 间连续平滑

    注意: GGUF 使用显式状态 (KV cache + conv history + latent buffer)
          本实现使用 left_context overlap 达到相同效果
    """

    # 对标 Native API chunked_decode 默认值
    LEFT_CONTEXT_SIZE = 25
    # 每帧对应的音频样本数 (total_upsample)
    SAMPLES_PER_FRAME = 1920

    def __init__(self, decoder_model, device: str = "cpu"):
        self.decoder = decoder_model
        self.device = device
        self.state = OverlapDecoderState()
        self.samples_per_frame = getattr(decoder_model, 'total_upsample', 1920)

    def warmup(self, ref_codes: torch.Tensor) -> None:
        """预解码 ref_codes 初始化历史 (对标 GGUF final_state)"""
        if ref_codes.dim() == 3:
            ref_codes = ref_codes.squeeze(0)

        n = min(ref_codes.shape[0], self.LEFT_CONTEXT_SIZE)
        self.state.code_history = []

        for i in range(ref_codes.shape[0] - n, ref_codes.shape[0]):
            self.state.code_history.append(ref_codes[i].clone())

    def decode_chunk(
        self,
        chunk_codes: Union[torch.Tensor, List[torch.Tensor]],
        is_final: bool = False
    ) -> torch.Tensor:
        """
        解码一个 chunk (对标 GGUF StatefulDecoder._decode)

        Args:
            chunk_codes: [N, 16] 或 List[[16]] 当前 chunk 的 codes
            is_final: 是否最后一块

        Returns:
            audio: [samples] 解码的音频波形
        """
        # 1. 规范化 chunk_codes 为 tensor
        if isinstance(chunk_codes, list):
            chunk_list = []
            for c in chunk_codes:
                if c.dim() > 1:
                    chunk_list.append(c.squeeze(0))
                else:
                    chunk_list.append(c)
            chunk_tensor = torch.stack(chunk_list)  # [N, 16]
        else:
            chunk_tensor = chunk_codes
            if chunk_tensor.dim() == 3:
                chunk_tensor = chunk_tensor.squeeze(0)

        # 2. 拼接 left_context + current_chunk
        context_frames = 0
        if self.state.code_history:
            context = torch.stack(self.state.code_history)
            codes_to_decode = torch.cat([context, chunk_tensor], dim=0)
            context_frames = len(self.state.code_history)
        else:
            codes_to_decode = chunk_tensor

        # 3. 调用 decoder forward
        codes_input = codes_to_decode.unsqueeze(0).transpose(1, 2).to(self.device)
        with torch.no_grad():
            wav = self.decoder(codes_input)  # [1, 1, samples]

        # 4. 裁剪 left_context 音频
        if context_frames > 0:
            trim_samples = context_frames * self.samples_per_frame
            if wav.shape[-1] > trim_samples:
                audio = wav[0, 0, trim_samples:]
            else:
                audio = wav[0, 0, :]
        else:
            audio = wav[0, 0, :]

        # 5. 更新历史
        chunk_frames = chunk_tensor.shape[0]
        for i in range(chunk_frames):
            self.state.code_history.append(chunk_tensor[i].clone().cpu())

        if len(self.state.code_history) > self.LEFT_CONTEXT_SIZE:
            self.state.code_history = self.state.code_history[-self.LEFT_CONTEXT_SIZE:]

        self.state.total_frames += chunk_frames
        return audio

    def reset(self) -> None:
        """重置历史状态"""
        self.state = OverlapDecoderState()

    @property
    def history_length(self) -> int:
        return len(self.state.code_history)
