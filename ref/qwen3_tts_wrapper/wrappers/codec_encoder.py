"""
CodecEncoder Wrapper - PyTorch 方案的音频编码器封装

对标 GGUF 方案的 CodecEncoder (ONNX 版)，提供简洁的音频编码接口。

设计思路：
1. 接受 Qwen3TTSModel 或 Qwen3TTSTokenizerV2Model
2. 提供 encode(audio) -> codes 简洁接口
3. 统一输出格式 [T, 16] 或 [B, T, 16]
4. 自动处理 padding_mask 和 tensor 转换

Author: Claude
Date: 2026-03-23
"""

import numpy as np
import torch
from typing import Union, Optional


class CodecEncoderWrapper:
    """
    PyTorch Codec Encoder 封装类

    对标 GGUF 的 CodecEncoder (qwen3_tts_gguf/inference/encoder.py)，
    提供相同的简洁接口。

    用法:
        from qwen3_tts import Qwen3TTSModel
        from qwen3_tts_wrapper.wrappers import CodecEncoderWrapper

        model = Qwen3TTSModel.from_pretrained(model_path)
        encoder = CodecEncoderWrapper(model)

        # 编码音频
        audio = np.random.randn(24000).astype(np.float32)  # 1秒 @ 24kHz
        codes = encoder.encode(audio)  # [T, 16]
    """

    def __init__(self, model):
        """
        初始化 Codec Encoder

        Args:
            model: Qwen3TTSModel 或 Qwen3TTSTokenizerV2Model
        """
        self._setup_model(model)
        self._validate_model()
        self.eval()

    def _setup_model(self, model):
        """解包并提取 speech_tokenizer"""
        # 情况1: Qwen3TTSModel (封装类)
        if hasattr(model, 'model') and hasattr(model.model, 'speech_tokenizer'):
            self.tokenizer = model.model.speech_tokenizer
            self.model_type = "Qwen3TTSModel"
        # 情况2: Qwen3TTSTokenizerV2Model (直接使用)
        elif hasattr(model, 'encoder') and hasattr(model, 'config'):
            self.tokenizer = model
            self.model_type = "Qwen3TTSTokenizerV2Model"
        # 情况3: Qwen3TTSForConditionalGeneration (底层模型)
        elif hasattr(model, 'speech_tokenizer'):
            self.tokenizer = model.speech_tokenizer
            self.model_type = "Qwen3TTSForConditionalGeneration"
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                f"Expected Qwen3TTSModel, Qwen3TTSForConditionalGeneration, "
                f"or Qwen3TTSTokenizerV2Model."
            )

        # 获取内部模型 (用于 encode 调用)
        if hasattr(self.tokenizer, 'model'):
            self._encode_model = self.tokenizer.model
        elif hasattr(self.tokenizer, 'encode'):
            self._encode_model = self.tokenizer
        else:
            raise ValueError("Cannot find encode method in tokenizer")

    def _validate_model(self):
        """验证模型配置"""
        if not hasattr(self._encode_model, 'encode'):
            raise ValueError("Model does not have encode method")

        # 检查 encoder_valid_num_quantizers
        if hasattr(self._encode_model, 'config'):
            config = self._encode_model.config
            if hasattr(config, 'encoder_valid_num_quantizers'):
                self._num_quantizers = config.encoder_valid_num_quantizers
            else:
                self._num_quantizers = 16  # 默认 16 层 RVQ
        else:
            self._num_quantizers = 16

    def eval(self):
        """设置模型为评估模式"""
        if hasattr(self.tokenizer, 'eval'):
            self.tokenizer.eval()
        if hasattr(self._encode_model, 'eval'):
            self._encode_model.eval()

    @torch.no_grad()
    def encode(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        编码音频为 codec codes

        Args:
            audio: 输入音频
                - np.ndarray: [samples] 或 [B, samples]
                - torch.Tensor: [samples] 或 [B, samples]
            return_numpy: 是否返回 numpy 数组 (True) 或 torch.Tensor (False)

        Returns:
            codes: 编码后的 codes
                - 单批次: [T, 16] int64
                - 多批次: [B, T, 16] int64
        """
        # ========== 1. 输入处理 ==========
        input_tensor, batch_mode = self._prepare_input(audio)

        # ========== 2. 创建 padding_mask ==========
        # padding_mask: True 表示有效音频, False 表示 padding
        # 我们的输入全是有效音频，所以全 True
        padding_mask = torch.ones(input_tensor.shape[:2], dtype=torch.bool, device=input_tensor.device)

        # ========== 3. 调用 encode ==========
        result = self._encode_model.encode(
            input_values=input_tensor,
            padding_mask=padding_mask,
            return_dict=False
        )

        # ========== 4. 提取并转换结果 ==========
        codes = self._extract_codes(result)

        # ========== 5. 返回 ==========
        if return_numpy:
            return codes.cpu().numpy().astype(np.int64)
        return codes

    def _prepare_input(self, audio: Union[np.ndarray, torch.Tensor]) -> tuple[torch.Tensor, bool]:
        """
        准备输入 tensor

        Returns:
            (input_tensor, batch_mode): (tensor, is_batched)
        """
        # 转换为 tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # 确保是 float32
        if audio.dtype != torch.float32:
            audio = audio.float()

        # 检查维度
        batch_mode = audio.ndim == 2

        if not batch_mode:
            # [samples] -> [1, samples]
            audio = audio.unsqueeze(0)
        # else: [B, samples] 保持不变

        return audio, batch_mode

    def _extract_codes(self, result: tuple) -> torch.Tensor:
        """
        从 encode 结果中提取 codes

        Args:
            result: encode 返回的 tuple (audio_codes, ...)

        Returns:
            codes: [B, T, 16] 或 [T, 16]
        """
        # result 是 (audio_codes, ...)
        # audio_codes 是 List[torch.Tensor]，每个形状 [T, num_quantizers]
        audio_codes_list = result[0]

        # 提取第一个 batch
        if len(audio_codes_list) == 1:
            codes = audio_codes_list[0]  # [T, 16]
        else:
            # 多批次情况 - 堆叠
            codes = torch.stack(audio_codes_list, dim=0)  # [B, T, 16]

        return codes

    @property
    def num_quantizers(self) -> int:
        """返回 quantizer 数量"""
        return self._num_quantizers

    @property
    def device(self) -> torch.device:
        """返回模型设备"""
        if hasattr(self.tokenizer, 'device'):
            return self.tokenizer.device
        return next(self.tokenizer.parameters()).device


def create_codec_encoder(model) -> CodecEncoderWrapper:
    """
    工厂函数：创建 CodecEncoderWrapper

    Args:
        model: Qwen3TTSModel 或 Qwen3TTSTokenizerV2Model

    Returns:
        CodecEncoderWrapper 实例
    """
    return CodecEncoderWrapper(model)
