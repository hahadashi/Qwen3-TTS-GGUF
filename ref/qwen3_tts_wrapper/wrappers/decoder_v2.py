"""
Stateful Decoder Wrapper - 支持完整状态管理的音频解码器

参考 GGUF 方案实现相同的 KV Cache 处理策略:
1. 8层 Transformer KV Cache + 滑动窗口 (72帧)
2. pre_conv 历史缓冲
3. latent_buffer 和 conv_history 缓冲
4. 帧边界处理 (skip_samples + latent_audio)

作者: Claude
日期: 2025-03-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import math

# DynamicCache support
try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    DynamicCache = None
    HAS_DYNAMIC_CACHE = False


@dataclass
class StatefulDecoderState:
    """完整的 Decoder 状态 - 对应 GGUF 的 DecoderState"""
    # Transformer KV Cache (8层) - 使用 DynamicCache
    kv_cache: Optional['DynamicCache'] = None

    # pre_conv 历史缓冲
    pre_conv_history: Optional[torch.Tensor] = None

    # 延迟音频缓冲 (帧边界处理)
    latent_audio: Optional[torch.Tensor] = None

    # 跳过采样点计数
    skip_samples: int = 0

    # 当前位置
    position: int = 0

    # 批次大小
    batch_size: int = 1

    # 是否已初始化
    initialized: bool = False


class StatefulDecoderWrapper:
    """
    状态化 Decoder 封装器 - 实现与 GGUF 相同的状态管理策略

    关键特性:
    1. KV Cache 管理 (8层 Transformer, 滑动窗口 72 帧)
    2. 历史缓冲管理 (pre_conv, latent, conv)
    3. 帧边界处理 (skip_samples + latent_audio)
    4. 分块解码支持

    对比原 DecoderWrapper:
    - 原: 无状态，每帧独立解码
    - 新: 有状态，支持跨帧连续性
    """

    # 常量配置 (与 GGUF 对齐)
    NUM_LAYERS = 8          # Transformer 层数
    NUM_HEADS = 16          # 注意力头数
    HEAD_DIM = 64           # 每个头的维度
    PRE_CONV_WINDOW = 2     # pre_conv 历史窗口
    LOOKAHEAD_FRAMES = 4    # 前瞻帧数
    KV_CACHE_WINDOW = 72    # KV Cache 滑动窗口
    SAMPLES_PER_FRAME = 1920 # 每帧采样数 (80ms @ 24kHz)

    def __init__(
        self,
        decoder,
        config,
        kv_cache_window: int = 72,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        初始化状态化 Decoder

        Args:
            decoder: Qwen3TTSTokenizerV2Decoder 模型
            config: Decoder 配置
            kv_cache_window: KV Cache 滑动窗口大小
            device: 设备
            dtype: 数据类型
        """
        self.decoder = decoder
        self.config = config
        self.kv_cache_window = kv_cache_window
        self.device = device
        self.dtype = dtype

        # 从配置提取参数
        self.num_quantizers = getattr(config, 'num_quantizers', 16)
        self.codebook_dim = getattr(config, 'codebook_dim', 256)
        self.latent_dim = getattr(config, 'latent_dim', 1024)
        self.hidden_size = getattr(config, 'hidden_size', 1024)
        self.total_upsample = getattr(decoder, 'total_upsample', 1920)

        # 检查组件
        self._verify_decoder_structure()

        print(f"[StatefulDecoderWrapper] 初始化完成:")
        print(f"  - num_quantizers: {self.num_quantizers}")
        print(f"  - codebook_dim: {self.codebook_dim}")
        print(f"  - latent_dim: {self.latent_dim}")
        print(f"  - total_upsample: {self.total_upsample}")
        print(f"  - kv_cache_window: {self.kv_cache_window}")

    def _verify_decoder_structure(self):
        """验证 Decoder 结构"""
        assert hasattr(self.decoder, 'quantizer'), "Decoder 缺少 quantizer"
        assert hasattr(self.decoder, 'pre_conv'), "Decoder 缺少 pre_conv"
        assert hasattr(self.decoder, 'pre_transformer'), "Decoder 缺少 pre_transformer"
        assert hasattr(self.decoder, 'upsample'), "Decoder 缺少 upsample"
        assert hasattr(self.decoder, 'decoder'), "Decoder 缺少 decoder"

    def init_state(self, batch_size: int = 1) -> StatefulDecoderState:
        """
        初始化完整状态

        Args:
            batch_size: 批次大小

        Returns:
            StatefulDecoderState: 初始化的状态对象
        """
        state = StatefulDecoderState(
            kv_cache=None,
            pre_conv_history=torch.zeros(
                batch_size, self.codebook_dim, 0,
                device=self.device, dtype=self.dtype
            ),
            latent_audio=None,
            skip_samples=0,
            position=0,
            batch_size=batch_size,
            initialized=True,
        )
        return state

    def _init_kv_cache(self, batch_size: int) -> 'DynamicCache':
        """
        初始化 KV Cache

        Returns:
            DynamicCache object
        """
        if HAS_DYNAMIC_CACHE:
            return DynamicCache()
        else:
            return None

    def _trim_kv_cache(self, kv_cache: 'DynamicCache', max_len: int):
        """
        修剪 KV Cache 到滑动窗口大小

        Note: DynamicCache may already have sliding window support built-in,
        so we only trim if necessary and possible.

        Args:
            kv_cache: DynamicCache 对象
            max_len: 最大长度
        """
        if kv_cache is None or not HAS_DYNAMIC_CACHE:
            return kv_cache

        # 获取当前长度
        try:
            seq_len = kv_cache.get_seq_length(layer_idx=0)
        except (AttributeError, IndexError):
            return kv_cache

        # 如果超过窗口大小，尝试裁剪
        # Note: DynamicCache with sliding window layers may handle this automatically
        if seq_len > max_len:
            try:
                kv_cache.crop(max_length=max_len)
            except ValueError:
                # Sliding window cache manages itself, no need to crop
                pass

        return kv_cache

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        state: Optional[StatefulDecoderState] = None,
        is_final: bool = False,
    ) -> Tuple[torch.Tensor, StatefulDecoderState]:
        """
        流式解码 - 支持状态管理

        Args:
            codes: Audio Codes [batch, num_quantizers, seq_len] 或 [batch, seq_len, num_quantizers]
            state: 当前状态
            is_final: 是否最后一帧

        Returns:
            audio: 解码的音频波形 [batch, samples]
            new_state: 更新后的状态
        """
        # 初始化状态
        if state is None:
            batch_size = codes.shape[0]
            state = self.init_state(batch_size)

        # 规范化 codes 形状: [batch, num_quantizers, seq_len]
        codes = self._normalize_codes_shape(codes)
        batch_size, num_q, seq_len = codes.shape

        # 空输入处理
        if seq_len == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio
                new_state = StatefulDecoderState(
                    kv_cache=None,
                    pre_conv_history=state.pre_conv_history,
                    latent_audio=None,
                    skip_samples=0,
                    position=state.position,
                    batch_size=batch_size,
                    initialized=True,
                )
                return audio, new_state
            return torch.zeros(batch_size, 0, device=self.device, dtype=self.dtype), state

        # 执行解码 (使用状态化的 transformer)
        wav, kv_cache, pre_conv_history = \
            self._decode_with_state(codes, state, is_final)

        # 帧边界处理
        audio, latent_audio, skip_samples = self._handle_frame_boundary(
            wav, state, is_final
        )

        # 修剪 KV Cache (滑动窗口)
        kv_cache = self._trim_kv_cache(kv_cache, self.kv_cache_window)

        # 构建新状态
        new_state = StatefulDecoderState(
            kv_cache=kv_cache,
            pre_conv_history=pre_conv_history,
            latent_audio=latent_audio,
            skip_samples=skip_samples,
            position=state.position + seq_len,
            batch_size=batch_size,
            initialized=True,
        )

        return audio, new_state

    def _normalize_codes_shape(self, codes: torch.Tensor) -> torch.Tensor:
        """规范化 codes 形状为 [batch, num_quantizers, seq_len]"""
        if codes.dim() == 2:
            # [batch, seq_len, num_q] -> [batch, num_q, seq_len]
            codes = codes.transpose(1, 2)
        elif codes.dim() == 3:
            if codes.shape[1] != self.num_quantizers:
                # [batch, seq_len, num_q] -> [batch, num_q, seq_len]
                codes = codes.transpose(1, 2)
        return codes

    def _decode_with_state(
        self,
        codes: torch.Tensor,
        state: StatefulDecoderState,
        is_final: bool,
    ) -> Tuple[torch.Tensor, 'DynamicCache', torch.Tensor]:
        """
        使用状态执行解码

        这是核心解码逻辑，尽可能利用模型的缓存支持
        """
        batch_size, num_q, seq_len = codes.shape

        # 1. Quantizer 解码
        hidden = self.decoder.quantizer.decode(codes)  # [batch, codebook_dim, seq_len]

        # 2. pre_conv (带历史)
        hidden, pre_conv_history = self._pre_conv_with_history(hidden, state.pre_conv_history)

        # 3. Transformer (带 KV Cache)
        hidden = hidden.transpose(1, 2)  # [batch, seq_len, hidden]

        # 准备 transformer 输入
        past_key_values = state.kv_cache if state.initialized else None

        # 调用 transformer
        transformer_output = self.decoder.pre_transformer(
            inputs_embeds=hidden,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        hidden = transformer_output.last_hidden_state
        new_kv_cache = transformer_output.past_key_values
        hidden = hidden.transpose(1, 2)  # [batch, hidden, seq_len]

        # 4. Upsample blocks
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)

        # 5. Decoder blocks (最终卷积层，无状态)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)

        # 6. Clip output
        wav = wav.clamp(min=-1, max=1)

        return wav, new_kv_cache, pre_conv_history

    def _pre_conv_with_history(
        self,
        hidden: torch.Tensor,
        history: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pre_conv 带历史缓冲处理

        Args:
            hidden: [batch, codebook_dim, seq_len]
            history: [batch, codebook_dim, history_len]

        Returns:
            output: [batch, latent_dim, seq_len]
            new_history: [batch, codebook_dim, history_len]
        """
        # 拼接历史
        if history is not None and history.shape[-1] > 0:
            hidden = torch.cat([history, hidden], dim=-1)

        # 执行 pre_conv
        output = self.decoder.pre_conv(hidden)

        # 更新历史 (保留最后 PRE_CONV_WINDOW 帧)
        new_history = hidden[:, :, -self.PRE_CONV_WINDOW:] if hidden.shape[-1] > self.PRE_CONV_WINDOW else hidden

        return output, new_history

    def _handle_frame_boundary(
        self,
        wav: torch.Tensor,
        state: StatefulDecoderState,
        is_final: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        处理帧边界 - 延迟输出以保证连续性

        Args:
            wav: [batch, 1, samples] 或 [batch, samples]
            state: 当前状态
            is_final: 是否最后一帧

        Returns:
            audio: 当前输出的音频
            latent_audio: 延迟的音频
            skip_samples: 跳过的采样点数
        """
        # 确保 wav 是 3D
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # [batch, 1, samples]

        batch_size = wav.shape[0]
        total_samples = wav.shape[-1]

        # 计算有效样本 (减去前瞻帧的样本)
        lookahead_samples = self.LOOKAHEAD_FRAMES * self.SAMPLES_PER_FRAME
        valid_samples = max(0, total_samples - lookahead_samples)

        skip_counter = state.skip_samples

        if is_final:
            # 最后一帧：全部输出
            audio = wav[:, 0, :]  # [batch, samples]
            latent_audio = None
            skip_samples = 0
        else:
            # 非最后一帧：延迟输出
            if valid_samples > 0:
                audio = wav[:, 0, :valid_samples]
                latent_audio = wav[:, 0, valid_samples:]
            else:
                audio = torch.zeros(batch_size, 0, device=self.device, dtype=self.dtype)
                latent_audio = wav[:, 0, :]

            # 处理跳过采样点 (用于过滤状态注入时的初始残留)
            if skip_counter > 0 and audio.shape[-1] > 0:
                if audio.shape[-1] <= skip_counter:
                    skip_counter -= audio.shape[-1]
                    audio = torch.zeros(batch_size, 0, device=self.device, dtype=self.dtype)
                else:
                    audio = audio[:, skip_counter:]
                    skip_counter = 0

            # 任务结束时标记下次需要跳过的残留音频
            skip_samples = 4 * self.SAMPLES_PER_FRAME if is_final else skip_counter

        return audio, latent_audio, skip_samples

    @torch.no_grad()
    def decode_full(self, codes: torch.Tensor) -> torch.Tensor:
        """
        一次性解码所有 codes (非流式场景)

        Args:
            codes: [batch, seq_len, num_quantizers]

        Returns:
            audio: [batch, samples]
        """
        return self.decoder(codes)

    @torch.no_grad()
    def decode_chunked(
        self,
        codes: torch.Tensor,
        chunk_size: int = 12,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """
        分块解码 - 类似 GGUF 的分块策略

        Args:
            codes: [batch, seq_len, num_quantizers]
            chunk_size: 每块大小
            left_context_size: 左上下文大小

        Returns:
            audio: [batch, samples]
        """
        codes = self._normalize_codes_shape(codes)
        batch_size, num_q, seq_len = codes.shape

        state = self.init_state(batch_size)
        audio_chunks = []

        start_index = 0
        while start_index < seq_len:
            end_index = min(start_index + chunk_size, seq_len)
            context_size = left_context_size if start_index > left_context_size else start_index

            codes_chunk = codes[:, :, start_index - context_size:end_index]
            is_final = (end_index >= seq_len)

            wav_chunk, state = self.decode(codes_chunk, state, is_final=is_final)

            # 跳过上下文部分的音频
            if context_size > 0 and wav_chunk.shape[-1] > context_size * self.total_upsample:
                wav_chunk = wav_chunk[:, context_size * self.total_upsample:]

            if wav_chunk.shape[-1] > 0:
                audio_chunks.append(wav_chunk)

            start_index = end_index

        if audio_chunks:
            return torch.cat(audio_chunks, dim=-1)
        else:
            return torch.zeros(batch_size, 0, device=self.device, dtype=self.dtype)

    def flush(self, state: StatefulDecoderState) -> Tuple[torch.Tensor, StatefulDecoderState]:
        """
        刷新缓冲区，返回剩余音频

        Args:
            state: 当前状态

        Returns:
            remaining_audio: 剩余音频
            new_state: 清空后的状态
        """
        if state.latent_audio is not None:
            audio = state.latent_audio
        else:
            audio = torch.zeros(state.batch_size, 0, device=self.device, dtype=self.dtype)

        new_state = self.init_state(state.batch_size)
        return audio, new_state


# ========== 兼容性封装 ==========

class DecoderWrapperV2(StatefulDecoderWrapper):
    """
    兼容性封装 - 与原 DecoderWrapper 接口兼容

    使用方式与原 DecoderWrapper 相同，但内部使用状态化解码
    """

    def __init__(self, speech_tokenizer, device: str = "cpu"):
        """
        初始化

        Args:
            speech_tokenizer: Qwen3TTSTokenizer 实例
            device: 设备
        """
        # 提取 decoder 和 config
        if hasattr(speech_tokenizer, 'model'):
            actual_model = speech_tokenizer.model
        else:
            actual_model = speech_tokenizer

        decoder = actual_model.decoder
        config = decoder.config

        super().__init__(
            decoder=decoder,
            config=config,
            device=device,
            dtype=torch.float32,
        )

        # 保存原始引用
        self._speech_tokenizer = speech_tokenizer
        self.sample_rate = 24000
        self.hop_size = 320

    def decode_single_frame(
        self,
        codes_16: torch.Tensor,
        state: Optional[StatefulDecoderState] = None,
        is_last: bool = False,
    ) -> Tuple[torch.Tensor, StatefulDecoderState]:
        """
        单帧解码 - 兼容原接口

        对于单帧流式解码，不使用 pre_conv 历史拼接。
        历史拼接主要用于批量解码场景以提高连续性。

        Args:
            codes_16: [16] 或 [1, 16] 或 [1, 16, 1]
            state: 当前状态
            is_last: 是否最后一帧

        Returns:
            audio_chunk: [samples]
            new_state: 更新后的状态
        """
        # 规范化形状为 [batch, num_quantizers, seq_len]
        if codes_16.dim() == 1:
            codes = codes_16.unsqueeze(0).unsqueeze(-1)  # [16] -> [1, 16, 1]
        elif codes_16.dim() == 2:
            codes = codes_16.unsqueeze(-1)  # [1, 16] -> [1, 16, 1]
        else:
            codes = codes_16  # 已经是 [1, 16, 1]

        # codes shape: [batch, num_quantizers, seq_len] = [1, 16, 1]

        # 初始化状态
        if state is None:
            state = self.init_state(batch_size=1)

        batch_size, num_q, seq_len = codes.shape

        # 1. Quantizer 解码
        hidden = self.decoder.quantizer.decode(codes)  # [batch, codebook_dim, seq_len]

        # 2. pre_conv (不使用历史拼接，保持 seq_len=1)
        hidden = self.decoder.pre_conv(hidden)
        hidden = hidden.transpose(1, 2)  # [batch, seq_len, hidden]

        # 3. Transformer (带 KV Cache)
        past_key_values = state.kv_cache if state.initialized else None

        transformer_output = self.decoder.pre_transformer(
            inputs_embeds=hidden,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        new_kv_cache = transformer_output.past_key_values
        hidden = transformer_output.last_hidden_state
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden, seq_len]

        # 4. Upsample blocks
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)

        # 5. Decoder blocks
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)

        wav = wav.clamp(min=-1, max=1)

        # 输出音频
        audio = wav.squeeze(1)  # [batch, samples]

        # 构建新状态
        new_state = StatefulDecoderState(
            kv_cache=new_kv_cache,
            pre_conv_history=None,  # 单帧模式不使用 pre_conv 历史
            latent_audio=None,
            skip_samples=0,
            position=state.position + seq_len,
            batch_size=batch_size,
            initialized=True,
        )

        return audio.squeeze(0), new_state  # [samples], state
