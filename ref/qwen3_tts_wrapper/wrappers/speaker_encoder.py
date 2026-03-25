"""
SpeakerEncoder Wrapper - PyTorch 方案的说话人编码器封装

对标 GGUF 方案的 SpeakerEncoder (ONNX 版)，提供简洁的说话人嵌入提取接口。

设计思路：
1. 接受 Qwen3TTSModel 或 Qwen3TTSSpeakerEncoder
2. 提供 encode(audio) -> spk_emb 简洁接口
3. 统一输出格式 [2048] float32
4. 自动处理音频加载、重采样、Mel 谱图提取

Author: Claude
Date: 2026-03-23
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple


class SpeakerEncoderWrapper:
    """
    PyTorch Speaker Encoder 封装类

    对标 GGUF 的 SpeakerEncoder (ONNX 版)，提供相同的简洁接口。

    用法:
        from qwen3_tts import Qwen3TTSModel
        from qwen3_tts_wrapper.wrappers import SpeakerEncoderWrapper

        model = Qwen3TTSModel.from_pretrained(model_path)
        encoder = SpeakerEncoderWrapper(model)

        # 从文件提取
        spk_emb = encoder.encode("path/to/audio.wav")  # [2048]

        # 从 numpy 提取
        audio = np.random.randn(24000).astype(np.float32)  # 1秒 @ 24kHz
        spk_emb = encoder.encode(audio)
    """

    # Mel 谱图参数 (与 model.extract_speaker_embedding 一致)
    MEL_N_FFT = 1024
    MEL_NUM_MELS = 128
    MEL_SAMPLING_RATE = 24000
    MEL_HOP_SIZE = 256
    MEL_WIN_SIZE = 1024
    MEL_FMIN = 0
    MEL_FMAX = 12000

    def __init__(self, model):
        """
        初始化 Speaker Encoder

        Args:
            model: Qwen3TTSModel 或 Qwen3TTSSpeakerEncoder
        """
        self._setup_model(model)
        self._validate_model()
        self.eval()

    def _setup_model(self, model):
        """解包并提取 speaker_encoder"""
        # 情况1: Qwen3TTSModel (封装类)
        if hasattr(model, 'model') and hasattr(model.model, 'speaker_encoder'):
            self.speaker_encoder = model.model.speaker_encoder
            self.model_type = "Qwen3TTSModel"
            self.device = getattr(model, 'device', torch.device('cpu'))
            self.dtype = getattr(model, 'dtype', torch.float32)
        # 情况2: Qwen3TTSForConditionalGeneration (底层模型)
        elif hasattr(model, 'speaker_encoder'):
            self.speaker_encoder = model.speaker_encoder
            self.model_type = "Qwen3TTSForConditionalGeneration"
            self.device = getattr(model, 'device', torch.device('cpu'))
            self.dtype = getattr(model, 'dtype', torch.float32)
        # 情况3: Qwen3TTSSpeakerEncoder (直接使用)
        elif hasattr(model, 'blocks') and hasattr(model, 'asp'):
            self.speaker_encoder = model
            self.model_type = "Qwen3TTSSpeakerEncoder"
            self.device = next(model.parameters()).device
            self.dtype = next(model.parameters()).dtype
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                f"Expected Qwen3TTSModel, Qwen3TTSForConditionalGeneration, "
                f"or Qwen3TTSSpeakerEncoder."
            )

        # 导入 mel_spectrogram 函数 (延迟导入)
        try:
            from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
            self._mel_spectrogram = mel_spectrogram
        except ImportError:
            # 尝试相对导入
            from ....Qwen3_TTS_main.qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
            self._mel_spectrogram = mel_spectrogram

    def _validate_model(self):
        """验证模型配置"""
        if not hasattr(self.speaker_encoder, 'forward'):
            raise ValueError("Model does not have forward method")

        # 获取输出维度
        if hasattr(self.speaker_encoder, 'config'):
            config = self.speaker_encoder.config
            if hasattr(config, 'enc_dim'):
                self._embedding_dim = config.enc_dim
            else:
                self._embedding_dim = 2048  # 默认 2048
        else:
            self._embedding_dim = 2048

    def eval(self):
        """设置模型为评估模式"""
        if hasattr(self.speaker_encoder, 'eval'):
            self.speaker_encoder.eval()

    @torch.no_grad()
    def encode(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sr: int = 24000,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        提取说话人嵌入

        Args:
            audio: 输入音频
                - str: wav 文件路径
                - np.ndarray: [samples] float32
                - torch.Tensor: [samples] float32
            sr: 音频采样率 (仅用于 np.ndarray/tensor 输入验证)
            return_numpy: 是否返回 numpy 数组 (True) 或 torch.Tensor (False)

        Returns:
            spk_emb: 说话人嵌入向量 [2048] float32
        """
        # ========== 1. 加载/准备音频 ==========
        audio_tensor = self._prepare_input(audio, sr)

        # ========== 2. 提取 Mel 谱图 ==========
        mels = self._extract_mel(audio_tensor)
        # mels: [1, T, 128]

        # ========== 3. Speaker Encoder 推理 ==========
        spk_emb = self.speaker_encoder(mels)
        # spk_emb: [1, 2048] 或 [2048]

        # ========== 4. 确保输出格式 ==========
        if spk_emb.ndim == 2:
            spk_emb = spk_emb[0]  # [1, 2048] -> [2048]

        # ========== 5. 返回 ==========
        if return_numpy:
            return spk_emb.cpu().numpy().astype(np.float32)
        return spk_emb

    def _prepare_input(self, audio: Union[str, np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        """
        准备输入 tensor

        Returns:
            audio_tensor: [1, samples] float32
        """
        # 1. 加载音频文件
        if isinstance(audio, str):
            audio_tensor, file_sr = self._load_audio(audio)
            # 重采样到 24kHz (如果需要)
            if file_sr != self.MEL_SAMPLING_RATE:
                audio_tensor = self._resample(audio_tensor, file_sr, self.MEL_SAMPLING_RATE)
        else:
            # 2. 转换为 tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio)
            else:
                audio_tensor = audio

            # 验证采样率
            if sr != self.MEL_SAMPLING_RATE:
                raise ValueError(
                    f"Audio sampling rate must be {self.MEL_SAMPLING_RATE}Hz, "
                    f"got {sr}Hz. Please resample or provide sr parameter."
                )

        # 3. 确保是 float32
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()

        # 4. 确保是 2D: [samples] -> [1, samples]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim > 2:
            raise ValueError(f"Audio must be 1D or 2D, got shape {audio_tensor.shape}")

        # 5. 移动到设备
        audio_tensor = audio_tensor.to(self.device)

        return audio_tensor

    def _extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        从音频提取 Mel 谱图

        Args:
            audio: [1, samples] float32, 24kHz

        Returns:
            mels: [1, T, 128] Mel 谱图
        """
        mels = self._mel_spectrogram(
            audio,
            n_fft=self.MEL_N_FFT,
            num_mels=self.MEL_NUM_MELS,
            sampling_rate=self.MEL_SAMPLING_RATE,
            hop_size=self.MEL_HOP_SIZE,
            win_size=self.MEL_WIN_SIZE,
            fmin=self.MEL_FMIN,
            fmax=self.MEL_FMAX,
            center=False,
        ).transpose(1, 2)  # [1, 128, T] -> [1, T, 128]

        return mels

    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件

        Returns:
            (audio, sr): (tensor [samples], sampling_rate)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 尝试使用 scipy
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)

            # 转换为 float32 [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

            # 立体声转单声道
            if len(audio.shape) == 2:
                audio = audio.mean(axis=1)

            return torch.from_numpy(audio), sr

        except Exception as e:
            # 尝试使用 soundfile
            try:
                import soundfile as sf
                audio, sr = sf.read(audio_path, dtype='float32')

                # 立体声转单声道
                if len(audio.shape) == 2:
                    audio = audio.mean(axis=1)

                return torch.from_numpy(audio), sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio: {e}, {e2}")

    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        重采样音频

        Args:
            audio: [samples] tensor
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            resampled_audio: [samples] tensor
        """
        import scipy.signal

        audio_np = audio.cpu().numpy()
        num_samples = int(len(audio_np) * target_sr / orig_sr)
        audio_resampled = scipy.signal.resample(audio_np, num_samples)
        return torch.from_numpy(audio_resampled).to(audio.device)

    @property
    def embedding_dim(self) -> int:
        """返回 embedding 维度"""
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        """返回模型设备"""
        return self._device

    @device.setter
    def device(self, value: torch.device):
        """设置模型设备"""
        self._device = value


def create_speaker_encoder(model) -> SpeakerEncoderWrapper:
    """
    工厂函数：创建 SpeakerEncoderWrapper

    Args:
        model: Qwen3TTSModel 或 Qwen3TTSSpeakerEncoder

    Returns:
        SpeakerEncoderWrapper 实例
    """
    return SpeakerEncoderWrapper(model)
