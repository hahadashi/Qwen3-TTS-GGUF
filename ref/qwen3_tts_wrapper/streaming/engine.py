"""
StreamingEngine - 统一流式推理引擎 (V2 + V3 Decoder)

基于 V2 Wrappers 实现，支持 GQA 模型。
使用 V3 Decoder 实现 GGUF 兼容的音频解码。

核心功能:
1. 整合 TalkerWrapperV2、PredictorWrapperV2、DecoderWrapperV3
2. 支持预处理资产加载
3. 统一的流式生成接口
4. 音频反馈机制
"""

import torch
from typing import Optional
from dataclasses import dataclass
import time
from datetime import datetime

# 日志辅助函数：带时间戳
def _log(msg: str = ""):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

# V2 Wrappers
from ..wrappers.talker_v2 import TalkerWrapperV2
from ..wrappers.predictor_v2 import PredictorWrapperV2
# V3 Decoder (GGUF Compatible)
from ..wrappers.decoder_v3 import DecoderWrapperV3, DecoderStateV3

# 状态类
from ..states import StateInitializer

# 数据类
from ..data import PromptData, VoiceAnchor

# 资产管理
from .assets import AssetsManager
from .sampler import Sampler
from .prompt_builder import PromptBuilder


@dataclass
class StreamConfig:
    """
    流式生成配置 - 双采样器架构

    参考 GGUF 方案，Talker 和 Predictor 使用独立的采样器:
    - Talker: 完整采样参数 + 惩罚参数，控制语义生成
    - Predictor: 简洁采样参数，保持声学稳定

    默认值对标 Native API (qwen3_tts_model.py)
    """

    # ==================== Talker 采样参数 (大师阶段) ====================
    # 控制 codec_0 生成，决定语音的内容和节奏

    temperature: float = 0.9         # 采样温度 (Native API 默认 0.9)
    top_k: int = 50                  # 候选集大小 (Native API 默认 50)
    top_p: float = 1.0               # 核采样阈值 (Native API 默认 1.0)
    min_p: float = 0.0               # Min-P 阈值

    # 惩罚参数 (防止复读机，增加多样性)
    repeat_penalty: float = 1.05     # 重复惩罚 (Native API 默认 1.05)
    frequency_penalty: float = 0.0   # 频率惩罚
    presence_penalty: float = 0.0    # 存在惩罚
    penalty_last_n: int = 128        # 惩罚窗口大小

    seed: Optional[int] = None       # Talker 随机种子 (Native API 默认 None)

    # EOS 豁免 (不希望因为生成过 EOS 而降低结尾概率)
    eos_exempt_from_penalty: bool = True

    # ==================== EOS Boosting 参数 ====================
    # 帮助模型更容易采样 EOS token，实现自然的停止

    enable_eos_boost: bool = False     # 禁用 EOS boosting (依赖模型自然产生 EOS)
    max_eos_boost: float = 3.0         # 最大 EOS boost 值 (logit space)

    # ==================== Predictor 采样参数 (工匠阶段) ====================
    # 控制 codec_1~15 生成，决定音频的细节和音色
    # 注意: Predictor 不使用惩罚参数，以保持声音稳定

    sub_temperature: float = 0.9     # Predictor 温度 (Native API 默认 0.9)
    sub_top_k: int = 50              # Predictor Top-K (Native API 默认 50)
    sub_top_p: float = 1.0           # Predictor Top-P (Native API 默认 1.0)
    sub_seed: Optional[int] = None   # Predictor 随机种子 (Native API 默认 None)

    # ==================== 生成参数 ====================

    max_frames: int = 2048           # 最大生成帧数 (Native API 默认 2048)

    # ==================== 音频反馈参数 ====================

    enable_audio_feedback: bool = True   # 启用音频反馈 (已测试，无 Early EOS 问题)
    feedback_delay: int = 0          # GGUF: 从第一帧就开始音频反馈
    audio_feedback_scale: float = 1.0  # GGUF: scale=1.0

    # ==================== 解码参数 ====================

    decode_audio: bool = True        # 是否解码音频
    chunk_size: int = 12             # 累积多少帧后解码 (12帧 = 960ms)
    streaming: bool = True           # True=流式解码, False=整段一次性解码
    parallel_decode: bool = True     # 启用并行解码 (解码与生成并行执行)

    # ==================== 音频归一化参数 ====================

    normalize_audio: bool = False    # 是否归一化音频 (匹配 Native API 输出水平)
    target_rms: float = 0.1          # 目标 RMS 水平 (Native API 典型值)


class StreamingEngine:
    """
    统一流式推理引擎 (V2 + V3 Decoder)

    使用 V2 Wrappers 实现 GQA 兼容的流式 TTS 推理。
    使用 V3 Decoder 实现 GGUF 兼容的音频解码。

    架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      StreamingEngine                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
    │  │ TalkerWrapperV2│  │PredictorWrapper│  │ DecoderWrapper │     │
    │  │   (GQA 兼容)    │  │      V2        │  │     V3         │     │
    │  │                │  │                │  │ (GGUF Compat)  │     │
    │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │
    │          │                   │                   │              │
    │          └───────────────────┼───────────────────┘              │
    │                              │                                  │
    │                    ┌─────────▼─────────┐                        │
    │                    │   AssetsManager   │                        │
    │                    │  (预处理资产加载)  │                        │
    │                    └───────────────────┘                        │
    └─────────────────────────────────────────────────────────────────┘

    流程:
    1. Prefill: 处理 Prompt，初始化 KV Cache
    2. Decode: 逐帧生成 codec_0
    3. Predict: 生成 codec_1~15
    4. Decode Audio: 解码为音频波形 (V3 3-Part architecture)
    5. Audio Feedback: 反馈到下一帧 (可选)
    """

    def __init__(
        self,
        model,  # Qwen3TTSModel or Qwen3TTSForConditionalGeneration
        device: str = "cpu",
        assets_dir: Optional[str] = None,
        debug: bool = False,
        dtype: str = "auto",
    ):
        """
        初始化 StreamingEngine

        Args:
            model: Qwen3TTSModel 实例或 Qwen3TTSForConditionalGeneration
            device: 设备
            assets_dir: 预处理资产目录 (可选)
            debug: 是否启用调试输出 (默认 False, 启用会影响性能)
            dtype: 推理精度 ("auto"=GPU自动fp16/CPU fp32, "float32", "float16", "bfloat16")
        """
        self.device = device
        self.debug = debug

        # 推理精度
        if dtype == "auto":
            self.use_autocast = (device != "cpu")
            self.autocast_dtype = torch.float16
        elif dtype == "float16":
            self.use_autocast = True
            self.autocast_dtype = torch.float16
        elif dtype == "bfloat16":
            self.use_autocast = True
            self.autocast_dtype = torch.bfloat16
        else:  # float32
            self.use_autocast = False
            self.autocast_dtype = torch.float32

        # 处理不同类型的模型输入
        # Qwen3TTSModel 包装类有 .model 属性
        if hasattr(model, 'model') and hasattr(model.model, 'talker'):
            # Qwen3TTSModel wrapper
            self.wrapper_model = model
            self.model = model.model
        else:
            # 直接是 Qwen3TTSForConditionalGeneration
            self.wrapper_model = None
            self.model = model

        # 提取子模型
        self.talker_model = self.model.talker
        self.predictor_model = self.model.talker.code_predictor
        # speech_tokenizer.model 是 Qwen3TTSTokenizerV2Model，包含 .decoder
        # 传递整个 model 给 DecoderWrapperV3，它会提取 .decoder
        self.decoder_model = self.model.speech_tokenizer.model

        # 创建 V2 Wrappers
        self.talker = TalkerWrapperV2(self.talker_model)
        # CRITICAL: 传递 Talker 的 codec_embedding 给 Predictor
        # 用于 code_0 的音频反馈 (vocab=3072 vs Predictor vocab=2048)
        talker_codec_emb = self.talker_model.model.codec_embedding
        self.predictor = PredictorWrapperV2(self.predictor_model, talker_codec_embedding=talker_codec_emb)
        # V3 Decoder (GGUF Compatible - 3-Part architecture)
        self.decoder = DecoderWrapperV3(self.decoder_model, device=device)

        # 资产管理器 - 需要原始包装类
        if self.wrapper_model is not None:
            self.assets = AssetsManager(self.wrapper_model)
        else:
            self.assets = AssetsManager(self.model)

        # 状态初始化器
        self.state_initializer = StateInitializer(self.model)

        # 配置
        self.config = self.model.config
        self.talker_config = self.model.config.talker_config

        # 特殊 token IDs
        self.codec_bos_id = getattr(self.talker_config, 'codec_bos_id', 2149)
        self.codec_eos_id = getattr(self.talker_config, 'codec_eos_token_id', 2150)
        self.codec_pad_id = getattr(self.talker_config, 'codec_pad_id', 2148)

        # 采样器 (默认配置)
        self.default_sampler = Sampler(temperature=0.7)

        # ========== Encoder Wrappers (可选使用) ==========
        # 提供简化的音频编码接口，与 GGUF 方案对齐
        from ..wrappers import CodecEncoderWrapper, SpeakerEncoderWrapper

        # 使用原始模型创建 wrappers (如果是 wrapper_model，使用它；否则用 model)
        ref_model = self.wrapper_model if self.wrapper_model is not None else self.model
        self.codec_encoder = CodecEncoderWrapper(ref_model)
        self.speaker_encoder = SpeakerEncoderWrapper(ref_model)

        _log(f"[StreamingEngine] 初始化完成")
        _log(f"  - TalkerWrapperV2: OK")
        _log(f"  - PredictorWrapperV2: OK")
        _log(f"  - DecoderWrapperV3 (GGUF Compatible): OK")
        _log(f"  - CodecEncoderWrapper: OK")
        _log(f"  - SpeakerEncoderWrapper: OK")
        _log(f"  - codec_bos_id: {self.codec_bos_id}")
        _log(f"  - codec_eos_id: {self.codec_eos_id}")

    # ========== 语言ID映射 ==========

    def _get_language_id(self, lang: str) -> Optional[int]:
        """
        将语言代码映射到 language_id

        Args:
            lang: 语言代码 (如 "zh", "chinese", "auto")

        Returns:
            language_id: 用于 think mode 的 ID，None 表示 auto/nothink mode
        """
        if lang.lower() == "auto":
            return None

        # 语言代码映射
        lang_map = {
            "zh": "chinese",
            "cn": "chinese",
            "chinese": "chinese",
            "en": "english",
            "english": "english",
        }

        normalized_lang = lang_map.get(lang.lower(), lang.lower())

        # 从 config 获取 language_id
        codec_language_id = self.assets.codec_language_id
        if codec_language_id and normalized_lang in codec_language_id:
            return codec_language_id[normalized_lang]

        # 未找到，使用 auto mode
        return None

    # ========== GGUF 风格接口 ==========

    def _decode_chunk_buffer(
        self,
        chunk_buffer: list,
        ref_codes: Optional[torch.Tensor] = None,
        use_ref_context: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        批量解码 chunk buffer 中的 codes

        Args:
            chunk_buffer: List[torch.Tensor] 累积的 codes
                - streaming=True: List[[1, 16]] (单个 codec frame)
                - streaming=False: List[[1, 16]] 累积后再解码
            ref_codes: Optional[torch.Tensor] 参考音频的 codes [ref_len, 16]
                用于提供解码器上下文 (和 Native API 一样)
            use_ref_context: 是否使用参考 codes 作为上下文
                - True (非流式): 连接 ref + gen 一起解码，然后移除 ref 部分
                - False (流式): 只解码生成的 codes (避免重复处理 ref)

        Returns:
            audio: 解码的音频 [samples]
        """
        if len(chunk_buffer) == 0:
            return None

        # chunk_buffer contains tensors of shape [1, 16]
        # Concatenate to get [N, 16]
        generated_codes = torch.cat(chunk_buffer, dim=0)  # [N, 16]

        # 非流式模式: Native API 方式 - 连接参考 codes + 生成的 codes 一起解码
        # 这样解码器的卷积层可以利用参考音频的上下文
        if use_ref_context and ref_codes is not None and ref_codes.numel() > 0:
            # ref_codes: [ref_len, 16], generated_codes: [N, 16]
            codes_to_decode = torch.cat([ref_codes, generated_codes], dim=0)  # [ref_len + N, 16]
            ref_len = ref_codes.shape[0]
            total_len = codes_to_decode.shape[0]
            should_trim_ref = True
        else:
            # 流式模式: 只解码生成的 codes
            # (避免在每个 chunk 中重复处理 ref，导致上下文不一致)
            codes_to_decode = generated_codes
            ref_len = 0
            total_len = generated_codes.shape[0]
            should_trim_ref = False

        # 使用 speech_tokenizer.decode() 高级 API（和 Native API 完全一样）
        with torch.no_grad():
            # speech_tokenizer.decode() 期望 dict 格式: {"audio_codes": codes}
            # 返回: (wavs: List[np.ndarray], sample_rate: int)
            encoded = {"audio_codes": codes_to_decode}
            wavs, sr = self.model.speech_tokenizer.decode(encoded)
            full_audio = torch.from_numpy(wavs[0]).to(self.device)  # [samples]

            # 如果使用了参考 codes，移除对应的音频部分 (和 Native API 一样)
            if should_trim_ref and ref_len > 0:
                # 按比例移除参考部分的音频
                cut = int(ref_len / max(total_len, 1) * full_audio.shape[0])
                audio = full_audio[cut:]
            else:
                audio = full_audio

        return audio  # [samples]

    def _sample_codec_0(self, logits: torch.Tensor, sampler) -> int:
        """
        采样 codec_0

        解码阶段只允许采样:
        - 0 ~ 2047: 正常 codec tokens
        - 2150 (EOS): 结束标记

        不允许采样:
        - 2148 (PAD): 填充标记
        - 2149 (BOS): 开始标记 (只在 prefill 中使用)

        Args:
            logits: [vocab_size] 预测 logits
            sampler: 采样器 (Sampler 或 EnhancedSampler)

        Returns:
            codec_0: 采样的 token ID
        """
        # 限制有效 token 范围 (0 ~ 2047 + EOS)
        # GGUF 参考: limit_start=0, limit_end=2048, allow_tokens={EOS}
        valid_range = set(range(2048)) | {self.codec_eos_id}  # 只有 EOS，不包括 BOS/PAD
        return sampler.sample(logits, allowed_tokens=valid_range)

    def _get_text_vector(
        self,
        trailing_text_embeds: Optional[torch.Tensor],
        text_pool_index: int,
    ) -> torch.Tensor:
        """
        获取文本向量 (GGUF风格: 耗尽后使用tts_pad)

        GGUF参考 (talker.py):
            if step_idx < len(trailing_text_pool):
                text_vec = trailing_text_pool[step_idx]
            else:
                text_vec = tts_pad  # 文本耗尽，使用Pad填充

        Args:
            trailing_text_embeds: [1, text_len, hidden] 或 None (GGUF风格无trailing)
            text_pool_index: 当前索引

        Returns:
            text_vec: [1, hidden]
        """
        # GGUF 风格: trailing_text_embeds 可能为 None
        if trailing_text_embeds is None:
            return self.assets.tts_pad.unsqueeze(0)

        text_len = trailing_text_embeds.shape[1]
        if text_pool_index < text_len:
            # 文本未耗尽，使用当前索引的文本
            return trailing_text_embeds[:, text_pool_index, :]
        else:
            # GGUF风格: 文本耗尽，使用tts_pad
            return self.assets.tts_pad.unsqueeze(0)

    # ========== GGUF 风格接口 ==========

    def clone(
        self,
        text: str,
        voice: VoiceAnchor,
        config: Optional[StreamConfig] = None,
    ) -> "TTSResult":
        """
        克隆模式合成 (GGUF 风格入口)

        参考 GGUF TTSStream.clone() 实现。
        调用链:
            clone() → _run_engine_loop() → _run_engine_loop_gen()

        Args:
            text: 要合成的文本
            voice: 音色锚点 (包含 speaker_embedding 和 reference_codes)
            config: 生成配置

        Returns:
            TTSResult: 包含 audio, codes, timing 等信息
        """
        from ..data import PromptData

        config = config or StreamConfig()

        # 获取 tokenizer (从包装模型或直接从模型)
        if self.wrapper_model is not None and hasattr(self.wrapper_model, 'processor'):
            tokenizer = self.wrapper_model.processor
        elif hasattr(self.model, 'processor'):
            tokenizer = self.model.processor
        else:
            raise ValueError("无法获取 tokenizer，请确保模型有 processor 属性")

        # 映射语言到 language_id
        lang_id = self._get_language_id(voice.lang)

        prompt_builder = PromptBuilder(tokenizer, self.assets)
        prompt_data = prompt_builder.build_clone_prompt(text, voice, lang_id=lang_id)

        # 执行生成循环
        timing, all_codes, summed_embeds, audio = self._run_engine_loop(
            prompt_data, config
        )

        # 返回结果
        from ..data import TTSResult
        return TTSResult(
            audio=audio,
            text=text,
            codes=all_codes,
            timing=timing,
        )

    def _run_engine_loop(
        self,
        prompt_data: "PromptData",
        config: StreamConfig,
    ):
        """
        运行生成循环 (GGUF 风格外层循环)

        对应 GGUF: stream.py:_run_engine_loop()
        - 管理统计计时
        - 累积 chunk_buffer
        - 批量解码音频

        Args:
            prompt_data: 构建好的 Prompt 数据
            config: 生成配置

        Returns:
            timing: 性能计时统计
            all_codes: 所有生成的 codes
            summed_embeds: 所有音频反馈 embeddings
            audio: 解码后的音频
        """
        timing_data = {
            "prompt_time": 0.0,
            "prefill_time": 0.0,
            "talker_loop_times": [],
            "predictor_loop_times": [],
            "chunk_gen_times": [],
            "decoder_compute_times": [],
            "total_steps": 0,
        }

        all_codes = []
        summed_embeds = []
        chunk_buffer = []
        audio_chunks = []

        streaming = config.streaming
        decode_audio = config.decode_audio
        chunk_size = config.chunk_size
        step_count = 0

        # ========== 流式模式: 初始化 StatefulDecoderWrapper ==========
        stateful_decoder = None
        async_decoder = None
        if streaming and decode_audio:
            from ..wrappers.stateful_decoder import OverlapDecoderWrapper
            # 获取 PyTorch 原始 decoder 模型
            raw_decoder = self.decoder_model.decoder  # Qwen3TTSTokenizerV2Decoder
            stateful_decoder = OverlapDecoderWrapper(raw_decoder, device=self.device)
            # 预解码 ref_codes 初始化历史 (对标 GGUF final_state)
            if prompt_data.ref_codes is not None and prompt_data.ref_codes.numel() > 0:
                stateful_decoder.warmup(prompt_data.ref_codes)
            _log(f"  [OverlapDecoder] 初始化完成, 历史帧: {stateful_decoder.history_length}")

            # ========== 并行解码: 初始化 AsyncAudioDecoder ==========
            if config.parallel_decode:
                from .async_decoder import AsyncAudioDecoder

                # 包装解码函数 (闭包捕获 stateful_decoder)
                def decode_fn(chunk_codes):
                    chunk_tensor = torch.cat(chunk_codes, dim=0)
                    return stateful_decoder.decode_chunk(chunk_tensor, is_final=False)

                async_decoder = AsyncAudioDecoder(
                    decode_fn=decode_fn,
                    chunk_size=chunk_size,
                )
                _log(f"  [AsyncDecoder] 并行解码已启用, chunk_size={chunk_size}")

        # ========== 记录起始时间 ==========
        timing_data["loop_start"] = time.time()

        # ========== 运行生成器循环 ==========
        last_chunk_time = time.time()

        for step_codes, audio_summed, loop_timing in self._run_engine_loop_gen(
            prompt_data, config, timing_data
        ):
            step_count += 1
            all_codes.append(step_codes)
            summed_embeds.append(audio_summed)

            # 累积到 chunk_buffer
            chunk_buffer.append(step_codes)

            # 并行模式: 提交 frame 到异步解码器
            if async_decoder is not None:
                async_decoder.submit_frame(step_codes)

            # GGUF 风格：简洁判断
            # - streaming=False: 跳过，继续累积
            # - buffer 未满: 跳过，继续累积
            # - 只有 streaming=True 且 buffer 满了才执行解码
            if not streaming or len(chunk_buffer) < chunk_size:
                continue

            # 解码当前 chunk
            if decode_audio:
                chunk_gen_time = time.time() - last_chunk_time
                timing_data["chunk_gen_times"].append(chunk_gen_time)
                last_chunk_time = time.time()

                if async_decoder is not None:
                    # 并行模式: 非阻塞提交解码任务，继续生成
                    async_decoder.submit_chunk()
                    # 尝试收集已完成的音频
                    try:
                        audio = async_decoder.get_audio(timeout=0)
                        if audio is not None and audio.numel() > 0:
                            audio_chunks.append(audio)
                            chunk_idx = len(audio_chunks)
                            total_audio_ms = sum(a.shape[0] for a in audio_chunks) / 24.0
                            _log(f"  [Async] chunk {chunk_idx}: {chunk_size} frames, "
                                  f"audio {audio.shape[0]/24000:.3f}s, "
                                  f"gen_time {chunk_gen_time:.3f}s, "
                                  f"cumulative_audio {total_audio_ms:.0f}ms")
                    except RuntimeError as e:
                        _log(f"  [Async] 解码错误: {e}")
                elif stateful_decoder is not None:
                    # 串行流式模式: 使用 StatefulDecoderWrapper (left_context overlap)
                    chunk_tensor = torch.cat(chunk_buffer, dim=0)  # [N, 16]
                    audio_chunk = stateful_decoder.decode_chunk(
                        chunk_tensor, is_final=False
                    )
                    if audio_chunk.numel() > 0:
                        audio_chunks.append(audio_chunk)
                        # 流式日志: 每个 chunk 的解码时间和累计音频时长
                        chunk_idx = len(audio_chunks)
                        total_audio_ms = sum(a.shape[0] for a in audio_chunks) / 24.0
                        _log(f"  [Stream] chunk {chunk_idx}: {len(chunk_buffer)} frames, "
                              f"audio {audio_chunk.shape[0]/24000:.3f}s, "
                              f"gen_time {chunk_gen_time:.3f}s, "
                              f"cumulative_audio {total_audio_ms:.0f}ms")
                else:
                    # 非流式模式: 使用旧的 ref context 方式
                    use_ref = (len(audio_chunks) == 0)
                    audio_chunk = self._decode_chunk_buffer(
                        chunk_buffer,
                        ref_codes=prompt_data.ref_codes,
                        use_ref_context=use_ref,
                    )
                    if audio_chunk is not None and audio_chunk.numel() > 0:
                        audio_chunks.append(audio_chunk.flatten())
            chunk_buffer = []

        # 最终解码（循环结束后，一次性解码剩余的所有帧）
        timing_data["chunk_gen_times"].append(time.time() - last_chunk_time)

        if decode_audio:
            # ========== 并行模式: 刷新并收集所有音频 ==========
            if async_decoder is not None:
                # 提交剩余 codes
                if chunk_buffer:
                    async_decoder.submit_chunk()

                # 等待所有解码完成并收集音频
                audio = async_decoder.flush()
                async_decoder.shutdown()

                if audio.numel() > 0:
                    _log(f"  [Async] 完成: 总音频 {audio.shape[0]/24000:.3f}s")
                else:
                    audio = torch.tensor([], device=self.device)

            # ========== 串行流式模式 ==========
            elif stateful_decoder is not None:
                # 解码剩余 chunk
                if chunk_buffer:
                    chunk_tensor = torch.cat(chunk_buffer, dim=0)
                    final_audio = stateful_decoder.decode_chunk(
                        chunk_tensor, is_final=True
                    )
                    if final_audio.numel() > 0:
                        audio_chunks.append(final_audio)
                        chunk_idx = len(audio_chunks)
                        total_audio_ms = sum(a.shape[0] for a in audio_chunks) / 24.0
                        _log(f"  [Stream] chunk {chunk_idx} (final): {len(chunk_buffer)} frames, "
                              f"audio {final_audio.shape[0]/24000:.3f}s, "
                              f"cumulative_audio {total_audio_ms:.0f}ms")

                # 合并音频
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=0)
                else:
                    audio = torch.tensor([], device=self.device)

            # ========== 非流式模式 ==========
            else:
                if chunk_buffer:
                    remaining_audio = self._decode_chunk_buffer(
                        chunk_buffer,
                        ref_codes=prompt_data.ref_codes,
                        use_ref_context=True,
                    )
                    if remaining_audio is not None and remaining_audio.numel() > 0:
                        audio_chunks.append(remaining_audio.flatten())

                # 合并音频
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=0)
                else:
                    audio = torch.tensor([], device=self.device)
        else:
            audio = torch.tensor([], device=self.device)

        # Audio normalization (可选)
        if config.normalize_audio and audio.numel() > 0:
            # 目标 RMS (基于 Native API 的典型值)
            target_rms = config.target_rms
            current_rms = torch.sqrt(torch.mean(audio ** 2))
            if current_rms > 0:
                audio = audio * (target_rms / current_rms)
                # 防止削波
                audio = torch.clamp(audio, min=-1.0, max=1.0)

        timing_data["total_steps"] = step_count

        # 转换 timing 为简单对象
        class Timing:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)

        timing = Timing(timing_data)

        return timing, all_codes, summed_embeds, audio

    def _run_engine_loop_gen(
        self,
        prompt_data: "PromptData",
        config: StreamConfig,
        timing_data: dict,
    ):
        """
        生成器循环 (GGUF 风格核心双层循环)

        对应 GGUF: stream.py:_run_engine_loop_gen()

        循环结构:
        ┌─────────────────────────────────────────────────────────────────────┐
        │                    for step in range(max_frames):  【第1层循环】          │
        │                                                                      │
        │   ┌────────────────────────────────────────────────────────────────┐ │
        │   │ Talker Stage:                                                  │ │
        │   │   decode_step(fused_embed) → logits                              │ │
        │   │   sampler.sample(logits) → codec_0                                │ │
        │   │   sampler.accept(codec_0)  ← 关键: 更新历史                           │ │
        │   │   if codec_0 == EOS: break                                     │ │
        │   └────────────────────────────────────────────────────────────────┘ │
        │                              │                                       │
        │                              ▼                                       │
        │   ┌────────────────────────────────────────────────────────────────┐ │
        │   │ Predictor Stage: predict_frame()  【第2层循环】                │ │
        │   │   for cs in range(1, 16):  ← 遍历 codec_1~15              │ │
        │   │       sampler.sample() → code                                    │ │
        │   │       ctx.decode()  ← 单步推理                                  │ │
        │   │   返回: codes_16 [16], embeddings_16 [16]                        │ │
        │   └────────────────────────────────────────────────────────────────┘ │
        │                              │                                       │
        │                              ▼                                       │
        │   ┌────────────────────────────────────────────────────────────────┐ │
        │   │ Feedback Stage:                                                │ │
        │   │   audio_summed = sum(embeddings_16)                             │ │
        │   │   fused_embed = build_feedback(audio_summed, text_vec)          │ │
        │   └────────────────────────────────────────────────────────────────┘ │
        │                                                                      │
        │   yield step_codes, audio_summed, timing                            │
        └─────────────────────────────────────────────────────────────────────┘

        Yields:
            step_codes: [16] 该帧的完整 codes
            audio_summed: [hidden] 16层 embedding 求和
            timing: dict 计时信息
        """
        # ========== 0. 创建双采样器 (参考 GGUF 方案) ==========
        if config.eos_exempt_from_penalty:
            exempt_tokens = {self.codec_bos_id, self.codec_eos_id, self.codec_pad_id}
        else:
            exempt_tokens = set()

        # Talker 采样器 (大师阶段) - 完整参数
        from .sampler import Sampler
        talker_sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            repeat_penalty=config.repeat_penalty,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            penalty_last_n=config.penalty_last_n,
            seed=config.seed,
            exempt_tokens=exempt_tokens,
        )

        # Predictor 采样器 (工匠阶段) - 简洁参数，无惩罚
        predictor_sampler = Sampler(
            temperature=config.sub_temperature,
            top_k=config.sub_top_k,
            top_p=config.sub_top_p,
            seed=config.sub_seed,
            repeat_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # ========== 1. Prefill 阶段 ==========
        prefill_start = time.time()

        # GGUF 风格: prefill 返回 logits，直接用于第一次采样
        with torch.autocast(device_type=self.device, enabled=self.use_autocast, dtype=self.autocast_dtype):
            initial_logits, last_hidden, talker_state = self.talker.prefill(
                inputs_embeds=prompt_data.prefill_embeds,
                attention_mask=prompt_data.prefill_attention_mask,
                position_ids=prompt_data.prefill_position_ids,
            )
        timing_data["prefill_time"] = (time.time() - prefill_start) * 1000  # ms

        # ========== 2. 初始化状态 ==========
        text_pool_index = 0  # 追踪文本池索引

        # ========== 3. 初始音频反馈 (零向量) ==========
        fused_embed = torch.zeros(
            1, self.assets.hidden_size,
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )

        # ========== 4. 生成循环 ==========
        _autocast_device = self.device
        for step in range(config.max_frames):
            # ----- 4.1 Talker Stage (大师决策) -----
            # 第一步: 使用 prefill 的 logits，后续步骤使用 decode_step_simple
            if step == 0:
                logits = initial_logits
                hidden = last_hidden
            else:
                with torch.autocast(device_type=_autocast_device, enabled=self.use_autocast, dtype=self.autocast_dtype):
                    logits, talker_state, hidden = self.talker.decode_step_simple(
                        fused_embed=fused_embed,
                        state=talker_state,
                    )

            # ----- 4.1.5 EOS Logit Boosting -----
            # 注意: GGUF 和 Native API 都不使用 EOS boosting
            # 它们依赖模型自然产生 EOS
            # 如果 EOS 不被采样，应该调查 logits 为什么不同
            # 而不是强行提升 EOS logit
            if config.enable_eos_boost:
                if step >= 5 and step % 10 == 0:
                    eos_logit = logits[0, self.codec_eos_id].item()
                    max_logit = logits[0].max().item()
                    _log(f"    [Step {step} EOS Debug] logit={eos_logit:.2f}, max={max_logit:.2f}")

            # ----- 4.2 采样 codec_0 -----
            # 确保使用 float32 进行采样，避免 float16 精度问题
            codec_0 = self._sample_codec_0(logits[0].float(), talker_sampler)

            # Debug: 打印 codec_0 值和 EOS 概率 (仅 debug 模式)
            if self.debug and step < 5:
                eos_logit = logits[0, self.codec_eos_id].item()
                max_logit = logits[0].max().item()
                probs = torch.softmax(logits[0], dim=-1)
                eos_prob = probs[self.codec_eos_id].item()
                top_5_probs, top_5_indices = torch.topk(probs[:2048], 5)
                _log(f"  [Step {step}] codec_0={codec_0}, EOS prob={eos_prob:.6f}, EOS logit={eos_logit:.4f}, max_logit={max_logit:.4f}")
                _log(f"    Top 5 (0-2047): {[(idx.item(), f'{p.item():.4f}') for idx, p in zip(top_5_indices, top_5_probs)]}")
                eos_rank = (logits[0] >= logits[0, self.codec_eos_id]).sum().item()
                _log(f"    EOS rank (after boost): {eos_rank} / {logits.shape[-1]}")

            # ----- 4.3 更新采样器历史 (关键!) -----
            # GGUF 参考: stream.py:304 - talker_sampler.accept(code_0)
            talker_sampler.accept(codec_0)

            # ----- 4.4 检查 EOS -----
            if codec_0 == self.codec_eos_id:
                break

            # ----- 4.5 Predictor Stage (工匠预测) -----
            predictor_start = time.time()
            with torch.autocast(device_type=_autocast_device, enabled=self.use_autocast, dtype=self.autocast_dtype):
                codes_16, embeddings_16 = self.predictor.predict_frame(
                    master_hidden=hidden,
                    code_0=codec_0,
                    sampler=predictor_sampler,
                )
            timing_data["predictor_loop_times"].append(
                (time.time() - predictor_start) * 1000
            )

            # ----- 4.6 Feedback Stage (音频反馈) -----
            feedback_start = time.time()
            audio_summed = torch.stack(embeddings_16).sum(dim=0)  # [1, hidden]

            # GGUF 风格: audio_summed + text_vec
            # GGUF 参考 talker.py:89-95:
            #   if step_idx < len(trailing_text_pool):
            #       text_vec = trailing_text_pool[step_idx]
            #   else:
            #       text_vec = tts_pad
            #   fused_embed = audio_embed + text_vec
            #   step_idx += 1
            if config.enable_audio_feedback and step >= config.feedback_delay:
                # 获取文本向量 (当前步的文本)
                text_vec = self._get_text_vector(
                    prompt_data.trailing_text_embeds,
                    text_pool_index,  # 使用当前索引
                )
                # GGUF 风格: 直接相加，不缩放
                fused_embed = audio_summed + text_vec
            else:
                # 使用文本向量
                fused_embed = self._get_text_vector(
                    prompt_data.trailing_text_embeds,
                    text_pool_index,
                )

            # 递增文本池索引 (GGUF: step_idx += 1 在 fusion 之后)
            text_pool_index += 1

            timing_data["talker_loop_times"].append(
                (time.time() - feedback_start) * 1000
            )

            # ----- 4.7 Yield 输出 -----
            yield codes_16, audio_summed, timing_data

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cpu",
        assets_dir: Optional[str] = None,
        debug: bool = False,
        dtype: str = "auto",
    ) -> "StreamingEngine":
        """
        从预训练模型创建 StreamingEngine

        Args:
            model_path: 模型路径
            device: 设备
            assets_dir: 预处理资产目录
            debug: 是否启用调试输出
            dtype: 推理精度 ("auto"=GPU自动fp16/CPU fp32, "float32", "float16", "bfloat16")

        Returns:
            engine: StreamingEngine 实例
        """
        from qwen_tts import Qwen3TTSModel

        _log(f"[StreamingEngine] 加载模型: {model_path}")
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=device,
        )

        return cls(model=model, device=device, assets_dir=assets_dir, debug=debug, dtype=dtype)
