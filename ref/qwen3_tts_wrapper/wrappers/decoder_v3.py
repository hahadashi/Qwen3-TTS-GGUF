"""
Stateful Decoder Wrapper V3 - GGUF-Compatible Architecture

完全对标 GGUF 方案实现的 Decoder Wrapper:
1. 3-Part 架构 (Part1: PreConv, Part2: Transformer, Part3: Upsample)
2. Flattened KV Cache 格式 (与 ONNX 兼容)
3. 3D latent_buffer 和 conv_history (存储 hidden states, not audio)
4. valid_samples 输出 (精确控制帧边界)

作者: Claude
日期: 2026-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


# ==============================================================================
# Decoder State V3 - GGUF Compatible Format
# ==============================================================================

@dataclass
class DecoderStateV3:
    """
    Decoder 状态 V3 - 完全对标 GGUF 格式

    关键变化:
    - KV Cache: 使用扁平化列表格式 (ONNX 友好)
    - latent_buffer: [B, 1024, 0-4] 存储 hidden states, NOT audio
    - conv_history: [B, 1024, 0-4] 新增卷积历史缓冲
    """
    # KV Cache (扁平化列表格式, 每层独立 key/value)
    past_keys: List[torch.Tensor] = field(default_factory=list)      # [num_layers] of [B, heads, seq, head_dim]
    past_values: List[torch.Tensor] = field(default_factory=list)    # [num_layers] of [B, heads, seq, head_dim]

    # History buffers (3D tensors)
    pre_conv_history: Optional[torch.Tensor] = None    # [B, 512, 0-2]
    latent_buffer: Optional[torch.Tensor] = None       # [B, 1024, 0-4] - HIDDEN STATES
    conv_history: Optional[torch.Tensor] = None        # [B, 1024, 0-4]

    # Position tracking
    position: int = 0
    batch_size: int = 1

    # 是否已初始化
    initialized: bool = False


# ==============================================================================
# Part 1: PreConv Module
# ==============================================================================

class DecoderPart1PreConv(nn.Module):
    """
    Qwen3-TTS Decoder Part 1: 特征提取与预卷积 (RVQ + Pre-Conv)

    完全对标 GGUF 实现:
    1. Quantizer decode: [B, N, Q] -> [B, Dim, N]
    2. 拼接 pre_conv_history
    3. 执行 pre_conv
    4. 切片得到当前输出
    5. 更新历史 (保留最后 2 帧)
    """

    PRE_CONV_HISTORY_WINDOW = 2

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.quantizer = decoder.quantizer
        self.pre_conv = decoder.pre_conv

    def forward(
        self,
        audio_codes: torch.Tensor,
        pre_conv_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_codes: [B, N, Q] where N=seq_len, Q=num_quantizers
            pre_conv_history: [B, 512, hist_len] where hist_len in [0, 2]

        Returns:
            hidden: [B, N, 1024] - 当前帧的 hidden states
            next_pre_conv_hist: [B, 512, 2] - 更新后的历史
        """
        # 1. Transpose codes: [B, N, Q] -> [B, Q, N]
        codes = audio_codes.transpose(1, 2)

        # 2. Quantizer decode: [B, Q, N] -> [B, 512, N]
        quantized = self.quantizer.decode(codes)

        # 3. 拼接历史: [B, 512, hist_len] + [B, 512, N] -> [B, 512, hist_len+N]
        if pre_conv_history is not None and pre_conv_history.shape[-1] > 0:
            quant_full = torch.cat([pre_conv_history, quantized], dim=-1)
        else:
            quant_full = quantized

        # 4. 执行 pre_conv: [B, 512, hist_len+N] -> [B, 1024, hist_len+N]
        hidden_all = self.pre_conv(quant_full)

        # 5. 切片得到当前输出
        hist_len = pre_conv_history.size(-1) if pre_conv_history is not None else 0
        hidden = hidden_all[:, :, hist_len:]  # [B, 1024, N]

        # 6. 转置: [B, 1024, N] -> [B, N, 1024]
        hidden = hidden.transpose(1, 2)

        # 7. 更新历史 (保留最后 PRE_CONV_HISTORY_WINDOW 帧)
        # quant_full: [B, 512, hist_len+N]
        if quant_full.shape[-1] >= self.PRE_CONV_HISTORY_WINDOW:
            next_pre_conv_hist = quant_full[:, :, -self.PRE_CONV_HISTORY_WINDOW:]
        else:
            next_pre_conv_hist = quant_full

        return hidden, next_pre_conv_hist


# ==============================================================================
# Part 2: Transformer Module
# ==============================================================================

class TraceableKVStack:
    """
    可被 Dynamo 追踪的滑动窗口 KV 缓存容器

    完全对标 GGUF 的 TraceableKVStack 实现。
    """

    def __init__(
        self,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        window_size: int,
    ):
        self.key_cache = keys
        self.value_cache = values
        self.window_size = window_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新指定层的 KV cache 并应用滑动窗口裁剪

        Args:
            key_states: [B, heads, N, head_dim]
            value_states: [B, heads, N, head_dim]
            layer_idx: 层索引

        Returns:
            updated_keys: [B, heads, seq, head_dim]
            updated_values: [B, heads, seq, head_dim]
        """
        # 拼接历史
        k_combined = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v_combined = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # 滑动窗口裁剪
        self.key_cache[layer_idx] = k_combined[:, :, -self.window_size:, :]
        self.value_cache[layer_idx] = v_combined[:, :, -self.window_size:, :]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """获取当前序列长度"""
        return self.key_cache[layer_idx].size(2)

    def __len__(self) -> int:
        return len(self.key_cache)


class DecoderPart2Transformer(nn.Module):
    """
    Qwen3-TTS Decoder Part 2: Transformer 骨干

    完全对标 GGUF 实现:
    1. Input projection
    2. RoPE with position_ids
    3. Attention mask (causal + sliding window)
    4. 8 transformer layers with TraceableKVStack
    5. Output projection
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.trans = decoder.pre_transformer
        self.num_layers = self.trans.config.num_hidden_layers
        self.window_size = getattr(self.trans.config, 'sliding_window', 72)

        # 获取 attention 配置
        self.num_attention_heads = getattr(self.trans.config, 'num_attention_heads', 16)
        self.num_key_value_heads = getattr(self.trans.config, 'num_key_value_heads', 8)
        self.head_dim = getattr(self.trans.config, 'head_dim', 64)

    def forward(
        self,
        hidden: torch.Tensor,
        *past_kv_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden: [B, N, 1024] - PreConv 输出
            *past_kv_flat: 扁平化的 KV cache (num_layers * 2 个张量)

        Returns:
            new_hidden: [B, N, 1024] - Transformer 输出
            *next_kv_flat: 更新后的 KV cache (扁平化)
        """
        B, N, H_dim = hidden.shape
        device = hidden.device

        # 1. 解析 KV cache
        keys_in = list(past_kv_flat[:self.num_layers])
        values_in = list(past_kv_flat[self.num_layers:])

        kv_stack = TraceableKVStack(keys_in, values_in, self.window_size)

        past_len = kv_stack.get_seq_length()
        total_len = past_len + N

        # 2. Input projection (如果有)
        if hasattr(self.trans, 'input_proj'):
            h = self.trans.input_proj(hidden)
        else:
            h = hidden

        # 3. Position IDs
        position_ids = torch.arange(past_len, total_len, device=device).unsqueeze(0)

        # 4. RoPE (如果有)
        if hasattr(self.trans, 'rotary_emb'):
            pos_embeddings = self.trans.rotary_emb(h, position_ids)
        else:
            pos_embeddings = None

        # 5. Attention Mask (因果 + 滑动窗口)
        # 创建 causal mask + sliding window mask
        q_idx = torch.arange(N, device=device).unsqueeze(1)  # [N, 1]
        full_k_idx = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]
        k_idx = full_k_idx[:, -self.window_size:]  # [1, window_size]

        # mask 条件: (k_idx <= past_len + q_idx) AND (k_idx > past_len + q_idx - window_size)
        mask_cond = (k_idx <= (past_len + q_idx)) & (k_idx > (past_len + q_idx - self.window_size))
        attn_mask = torch.where(mask_cond, 0.0, -10000.0)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, window_size]

        # 6. Transformer Layers
        for layer_idx, layer in enumerate(self.trans.layers):
            h = layer(
                h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=kv_stack,
                use_cache=True,
                position_embeddings=pos_embeddings,
            )

        # 7. Layer Norm (如果有)
        if hasattr(self.trans, 'norm'):
            h = self.trans.norm(h)

        # 8. Output projection (如果有)
        if hasattr(self.trans, 'output_proj'):
            new_hidden = self.trans.output_proj(h).transpose(1, 2)  # [B, 1024, N]
        else:
            new_hidden = h.transpose(1, 2)  # [B, 1024, N]

        # 9. 返回输出 + 更新后的 KV cache (扁平化)
        return (new_hidden,) + tuple(kv_stack.key_cache) + tuple(kv_stack.value_cache)


# ==============================================================================
# Part 3: Upsample Module
# ==============================================================================

class DecoderPart3Upsample(nn.Module):
    """
    Qwen3-TTS Decoder Part 3: 上采样与波形生成

    完全对标 GGUF 实现:
    1. 拼接 latent_buffer
    2. 计算可结算帧数 (num_finalize)
    3. 运行 upsample + decoder blocks
    4. 计算 valid_samples
    5. 更新 latent_buffer 和 conv_history
    """

    SAMPLES_PER_FRAME = 1920  # 80ms @ 24kHz
    LOOKAHEAD_FRAMES = 4      # 前瞻帧数
    CONV_HISTORY_WINDOW = 4   # 卷积历史窗口

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.upsample = decoder.upsample
        self.decoder = decoder.decoder
        self.samples_per_frame = decoder.total_upsample

    def forward(
        self,
        new_hidden: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            new_hidden: [B, 1024, N] - Transformer 输出
            latent_buffer: [B, 1024, buf_len] - 累积的 hidden states
            conv_history: [B, 1024, hist_len] - 卷积历史
            is_last: [1] - 是否最后一帧

        Returns:
            final_wav: [B, samples] - 解码的音频
            valid_samples: [1] - 有效采样点数
            next_latent_buf: [B, 1024, 4] - 更新后的 latent buffer
            next_conv_hist: [B, 1024, 4] - 更新后的卷积历史
        """
        device = new_hidden.device

        # 1. 拼接 Latent: [B, 1024, buf_len] + [B, 1024, N] -> [B, 1024, buf_len+N]
        if latent_buffer is not None and latent_buffer.shape[-1] > 0:
            accumulated = torch.cat([latent_buffer, new_hidden], dim=-1)
        else:
            accumulated = new_hidden

        # 2. 确定可结算帧数 (对标 GGUF 公式)
        total_acc_t = torch.zeros(1, device=device, dtype=torch.long) + accumulated.size(2)
        lookahead_t = torch.zeros(1, device=device, dtype=torch.long) + self.LOOKAHEAD_FRAMES

        total_acc_f = total_acc_t.to(torch.float32)
        lookahead_f = lookahead_t.to(torch.float32)

        # num_finalize = is_last ? total_frames : max(0, total_frames - lookahead)
        num_finalize_f = is_last * total_acc_f + (1.0 - is_last) * torch.clamp(total_acc_f - lookahead_f, min=0.0)
        num_finalize = num_finalize_f.to(torch.long)
        num_finalize_idx = num_finalize[0]

        # 3. 运行卷积链
        if conv_history is not None and conv_history.shape[-1] > 0:
            conv_chain_input = torch.cat([conv_history, accumulated], dim=-1)
        else:
            conv_chain_input = accumulated

        curr = conv_chain_input
        for blocks in self.upsample:
            for block in blocks:
                curr = block(curr)
        for block in self.decoder:
            curr = block(curr)

        wav = curr.squeeze(1).clamp(min=-1, max=1)  # [B, samples]

        # 4. 计算有效输出 (核心修复: 仅切左, 结尾依赖 valid_samples)
        upsample_factor = self.samples_per_frame
        conv_hist_len = conv_history.size(2) if conv_history is not None else 0
        start_samples_idx = conv_hist_len * upsample_factor

        valid_samples = (num_finalize * upsample_factor).view(1)
        final_wav = wav[:, start_samples_idx:]  # [B, samples-start]

        # 5. 更新 latent_buffer (保留最后 LOOKAHEAD_FRAMES 帧)
        next_latent_buf = accumulated[:, :, -self.LOOKAHEAD_FRAMES:]

        # 6. 更新 conv_history (对标 GGUF 的 gather 操作)
        B, C = accumulated.size(0), accumulated.size(1)
        indices = torch.arange(self.CONV_HISTORY_WINDOW, device=device, dtype=torch.long)
        target_indices = (num_finalize_idx - self.CONV_HISTORY_WINDOW) + indices
        gather_indices = torch.clamp(target_indices, min=0).unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        next_conv_hist = torch.gather(accumulated, 2, gather_indices)

        return final_wav, valid_samples, next_latent_buf, next_conv_hist


# ==============================================================================
# Main Decoder Wrapper V3
# ==============================================================================

class DecoderWrapperV3(nn.Module):
    """
    Stateful Decoder Wrapper V3 - 完全对标 GGUF 架构

    关键特性:
    1. 3-Part 架构 (PreConv, Transformer, Upsample)
    2. Flattened KV Cache 格式 (与 ONNX 兼容)
    3. 3D latent_buffer 和 conv_history
    4. valid_samples 输出

    使用方式:
        >>> wrapper = DecoderWrapperV3(speech_tokenizer)
        >>> state = wrapper.init_state()
        >>>
        >>> # 单帧解码
        >>> audio_chunk, state = wrapper.decode_single_frame(codes, state, is_last=False)
        >>>
        >>> # 多帧解码
        >>> audio_chunk, state = wrapper.decode(codes, state, is_last=False)
    """

    def __init__(self, speech_tokenizer, device: str = "cpu"):
        super().__init__()
        self.device = device

        # 提取 decoder - 支持多种输入格式
        # 格式 1: Qwen3TTSModel -> model.model.speech_tokenizer.model.decoder
        # 格式 2: Qwen3TTSForConditionalGeneration -> speech_tokenizer.model.decoder
        # 格式 3: Qwen3TTSTokenizer -> model.decoder
        # 格式 4: Qwen3TTSTokenizerV2Model -> model.decoder
        # 格式 5: 直接传入 decoder 模型
        if hasattr(speech_tokenizer, 'model') and hasattr(speech_tokenizer.model, 'speech_tokenizer'):
            # Qwen3TTSModel
            actual_model = speech_tokenizer.model.speech_tokenizer.model
        elif hasattr(speech_tokenizer, 'speech_tokenizer'):
            # Qwen3TTSForConditionalGeneration
            actual_model = speech_tokenizer.speech_tokenizer.model
        elif hasattr(speech_tokenizer, 'model') and hasattr(speech_tokenizer.model, 'decoder'):
            # Qwen3TTSTokenizerV2Model
            actual_model = speech_tokenizer.model
        elif hasattr(speech_tokenizer, 'decoder'):
            # 直接传入 decoder
            actual_model = speech_tokenizer
        else:
            raise ValueError(f"无法识别的模型类型: {type(speech_tokenizer)}")

        self.decoder = actual_model.decoder
        self.config = self.decoder.config

        # 获取配置
        self.num_quantizers = getattr(self.config, 'num_quantizers', 16)
        self.codebook_dim = getattr(self.config, 'codebook_dim', 512)
        self.latent_dim = getattr(self.config, 'latent_dim', 1024)
        self.hidden_size = getattr(self.config, 'hidden_size', 1024)
        self.total_upsample = getattr(self.decoder, 'total_upsample', 1920)
        self.num_layers = getattr(self.config, 'num_hidden_layers', 8)

        # 创建 3 个 Part
        self.part1 = DecoderPart1PreConv(self.decoder)
        self.part2 = DecoderPart2Transformer(self.decoder)
        self.part3 = DecoderPart3Upsample(self.decoder)

        # 常量
        self.SAMPLES_PER_FRAME = 1920
        self.PRE_CONV_HISTORY_WINDOW = 2
        self.LOOKAHEAD_FRAMES = 4
        self.CONV_HISTORY_WINDOW = 4

        # 评估模式
        self.eval()

        print(f"[DecoderWrapperV3] 初始化完成:")
        print(f"  - num_quantizers: {self.num_quantizers}")
        print(f"  - codebook_dim: {self.codebook_dim}")
        print(f"  - latent_dim: {self.latent_dim}")
        print(f"  - total_upsample: {self.total_upsample}")
        print(f"  - num_layers: {self.num_layers}")

    def init_state(self, batch_size: int = 1) -> DecoderStateV3:
        """
        初始化状态

        Returns:
            DecoderStateV3: 初始化的状态
        """
        # 初始化空的 KV cache
        past_keys = []
        past_values = []

        # 每层初始化为空 tensor [B, heads, 0, head_dim]
        # Decoder 使用 num_key_value_heads (可能是 16, 不是 GQA)
        num_kv_heads = getattr(self.config, 'num_key_value_heads',
                               getattr(self.config, 'num_attention_heads', 16))
        head_dim = getattr(self.config, 'head_dim', 64)

        for _ in range(self.num_layers):
            past_keys.append(torch.zeros(
                batch_size, num_kv_heads, 0, head_dim,
                device=self.device, dtype=torch.float32
            ))
            past_values.append(torch.zeros(
                batch_size, num_kv_heads, 0, head_dim,
                device=self.device, dtype=torch.float32
            ))

        state = DecoderStateV3(
            past_keys=past_keys,
            past_values=past_values,
            pre_conv_history=torch.zeros(
                batch_size, self.codebook_dim, 0,
                device=self.device, dtype=torch.float32
            ),
            latent_buffer=torch.zeros(
                batch_size, self.latent_dim, 0,
                device=self.device, dtype=torch.float32
            ),
            conv_history=torch.zeros(
                batch_size, self.latent_dim, 0,
                device=self.device, dtype=torch.float32
            ),
            position=0,
            batch_size=batch_size,
            initialized=True,
        )

        return state

    @torch.no_grad()
    def decode_single_frame(
        self,
        codes_16: torch.Tensor,
        state: Optional[DecoderStateV3] = None,
        is_last: bool = False,
    ) -> Tuple[torch.Tensor, DecoderStateV3]:
        """
        单帧解码 - 流式推理主接口

        Args:
            codes_16: [16] 或 [1, 16] 或 [1, 16, 1] - 单帧 codec codes
                [16] = 1 帧，16 个 quantizer 值
                [1, 16] = batch=1，1 帧，16 个 quantizer 值
                [1, 16, 1] = batch=1，seq_len=1，16 个 quantizer
            state: 当前状态
            is_last: 是否最后一帧

        Returns:
            audio_chunk: [samples] - 解码的音频
            new_state: 更新后的状态
        """
        # 规范化形状为 [batch, num_quantizers, seq_len] = [B, Q, N]
        # 注意: 输入 codes_16 是 [Q] 或 [B, Q] 格式，需要转置为 [B, Q, N]
        # Part1 会再转置为 [B, N, Q] 用于 quantizer.decode()
        if codes_16.dim() == 1:
            codes = codes_16.reshape(1, 1, -1)  # [16] -> [1, 1, 16]
        elif codes_16.dim() == 2:
            codes = codes_16.unsqueeze(1)  # [1, 16] -> [1, 1, 16]
        else:
            codes = codes_16  # 已经是 [1, 1, 16] 或类似格式

        # codes shape: [batch=1, num_quantizers=16, seq_len=1]

        # 初始化状态
        if state is None:
            state = self.init_state(batch_size=1)

        # is_last 转为 tensor
        is_last_tensor = torch.tensor([1.0 if is_last else 0.0], device=self.device)

        # 准备 KV cache (扁平化)
        past_kv_flat = tuple(state.past_keys) + tuple(state.past_values)

        # Part 1: PreConv
        hidden, next_pre_conv_hist = self.part1(codes, state.pre_conv_history)
        # hidden: [1, 1, 1024]
        # next_pre_conv_hist: [1, 512, 1] (因为 seq_len=1)

        # Part 2: Transformer
        trans_outputs = self.part2(hidden, *past_kv_flat)
        new_hidden = trans_outputs[0]
        # new_hidden: [1, 1024, 1]

        # 提取更新后的 KV cache
        next_kv_flat = trans_outputs[1:]
        next_keys = list(next_kv_flat[:self.num_layers])
        next_values = list(next_kv_flat[self.num_layers:])

        # Part 3: Upsample
        final_wav, valid_samples, next_latent_buf, next_conv_hist = self.part3(
            new_hidden, state.latent_buffer, state.conv_history, is_last_tensor
        )
        # final_wav: [1, samples] 或 [1, 0]
        # valid_samples: [1]

        # 根据 valid_samples 裁剪输出
        valid_len = valid_samples.item()
        if valid_len > 0 and final_wav.shape[-1] >= valid_len:
            audio_chunk = final_wav[:, :valid_len]  # [1, valid_len]
        else:
            audio_chunk = final_wav[:, :0]  # [1, 0] - empty when valid_samples=0

        # 构建新状态
        new_state = DecoderStateV3(
            past_keys=next_keys,
            past_values=next_values,
            pre_conv_history=next_pre_conv_hist,
            latent_buffer=next_latent_buf,
            conv_history=next_conv_hist,
            position=state.position + 1,
            batch_size=state.batch_size,
            initialized=True,
        )

        return audio_chunk.squeeze(0), new_state  # [samples], state

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        state: Optional[DecoderStateV3] = None,
        is_last: bool = False,
    ) -> Tuple[torch.Tensor, DecoderStateV3]:
        """
        多帧解码 - 批量解码接口

        Args:
            codes: [B, N, Q] 或 [B, Q, N] - codec codes
            state: 当前状态
            is_last: 是否最后一帧

        Returns:
            audio: [B, samples] - 解码的音频
            new_state: 更新后的状态
        """
        # 规范化形状为 [batch, num_quantizers, seq_len]
        if codes.dim() == 3:
            if codes.shape[1] != self.num_quantizers:
                # [B, N, Q] -> [B, Q, N]
                codes = codes.transpose(1, 2)
        # codes: [B, Q, N]

        # 初始化状态
        if state is None:
            state = self.init_state(batch_size=codes.shape[0])

        # is_last 转为 tensor
        is_last_tensor = torch.tensor([1.0 if is_last else 0.0], device=self.device)

        # 准备 KV cache (扁平化)
        past_kv_flat = tuple(state.past_keys) + tuple(state.past_values)

        # Part 1: PreConv
        hidden, next_pre_conv_hist = self.part1(codes, state.pre_conv_history)

        # Part 2: Transformer
        trans_outputs = self.part2(hidden, *past_kv_flat)
        new_hidden = trans_outputs[0]
        next_kv_flat = trans_outputs[1:]
        next_keys = list(next_kv_flat[:self.num_layers])
        next_values = list(next_kv_flat[self.num_layers:])

        # Part 3: Upsample
        final_wav, valid_samples, next_latent_buf, next_conv_hist = self.part3(
            new_hidden, state.latent_buffer, state.conv_history, is_last_tensor
        )

        # 根据 valid_samples 裁剪输出
        valid_len = valid_samples.item()
        if valid_len > 0 and final_wav.shape[-1] >= valid_len:
            audio = final_wav[:, :valid_len]
        else:
            audio = final_wav[:, :0]  # Empty when valid_samples=0

        # 构建新状态
        new_state = DecoderStateV3(
            past_keys=next_keys,
            past_values=next_values,
            pre_conv_history=next_pre_conv_hist,
            latent_buffer=next_latent_buf,
            conv_history=next_conv_hist,
            position=state.position + codes.shape[-1],
            batch_size=state.batch_size,
            initialized=True,
        )

        return audio, new_state

    @torch.no_grad()
    def flush(self, state: DecoderStateV3) -> Tuple[torch.Tensor, DecoderStateV3]:
        """
        刷新缓冲区 - 输出所有剩余音频

        Args:
            state: 当前状态

        Returns:
            remaining_audio: [B, samples] - 剩余音频
            new_state: 清空后的状态
        """
        # 标记为最后一帧来触发完整输出
        # 使用 is_last=True 来输出 latent_buffer 中的所有剩余音频
        is_last_tensor = torch.tensor([1.0], device=self.device)

        # 直接调用 Part3 输出剩余的 latent_buffer
        if state.latent_buffer is not None and state.latent_buffer.shape[-1] > 0:
            # 创建空的 transformer 输出 (只用于触发 Part3)
            B = state.batch_size
            empty_hidden = torch.zeros(B, self.latent_dim, 0, device=self.device)

            final_wav, valid_samples, _, _ = self.part3(
                empty_hidden, state.latent_buffer, state.conv_history, is_last_tensor
            )

            # 根据 valid_samples 裁剪输出
            valid_len = valid_samples.item()
            if valid_len > 0 and final_wav.shape[-1] >= valid_len:
                remaining_audio = final_wav[:, :valid_len]
            else:
                remaining_audio = final_wav
        else:
            remaining_audio = torch.zeros(state.batch_size, 0, device=self.device)

        # 清空状态
        new_state = self.init_state(state.batch_size)
        return remaining_audio, new_state


# ==============================================================================
# 便捷创建函数
# ==============================================================================

def create_decoder_v3(speech_tokenizer, device: str = "cpu") -> DecoderWrapperV3:
    """
    便捷函数: 创建 DecoderWrapperV3

    Args:
        speech_tokenizer: Qwen3TTSTokenizer 实例
        device: 设备

    Returns:
        DecoderWrapperV3 实例
    """
    return DecoderWrapperV3(speech_tokenizer, device=device)
