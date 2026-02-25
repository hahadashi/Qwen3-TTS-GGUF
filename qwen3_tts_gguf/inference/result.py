"""
result.py - 合成结果与身份锚点统一类
核心职责：
1. 承载 TTS 合成结果 (音频、元数据、性能统计)。
2. 作为 Voice Identity 锚点提供克隆所需的特征。
3. 提供音频播放、保存以及 JSON 持久化能力。
"""
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union
from .constants import SAMPLE_RATE, map_speaker, map_language
from . import logger

@dataclass
class Timing:
    """性能耗时统计"""
    prompt_time: float = 0.0
    prefill_time: float = 0.0
    talker_loop_time: float = 0.0
    predictor_loop_time: float = 0.0
    decoder_render_time: float = 0.0
    total_steps: int = 0

    @property
    def total_inference_time(self) -> float:
        return (self.prompt_time + self.prefill_time + 
                self.talker_loop_time + self.predictor_loop_time + 
                self.decoder_render_time)

    @property
    def inference_only_time(self) -> float:
        """核心推理耗时 (不包含最终的解码渲染)"""
        return (self.prompt_time + self.prefill_time + 
                self.talker_loop_time + self.predictor_loop_time)

@dataclass
class LoopOutput:
    """推理内核循环的输出封装"""
    all_codes: List[List[int]]     # 所有生成的 Codec IDs
    summed_embeds: List[np.ndarray] # 叠加特征序列
    timing: Timing                  # 性能统计对象

@dataclass
class TTSResult:
    """TTS 合成结果 (同时也是音色锚点)"""
    # 核心特征 (锚点要素)
    text: str                               # 文字内容
    spk_emb: np.ndarray                     # 全局音色向量 (2048)
    text_ids: List[int]                     # 文本 Token IDs
    codes: np.ndarray                       # 音频 Codec IDs (T, 16)
    summed_embeds: Optional[List[np.ndarray]] = None # 音频叠加特征 (T, 2048) - 可选
    info: str = ""                          # 备注信息 (如音色描述)
    
    # 产出附件 (可选)
    audio: Optional[np.ndarray] = None      # 音频波形 (PCM float32)
    stats: Optional[Timing] = None          # 性能统计信息

    @property
    def is_valid_anchor(self) -> bool:
        """是否具有作为 Voice 锚点的必要特征"""
        return self.codes is not None and self.spk_emb is not None

    @property
    def duration(self) -> float:
        """音频时长 (s)"""
        if self.audio is not None:
            return len(self.audio) / SAMPLE_RATE
        return 0.0
    
    @property
    def rtf(self) -> float:
        """实时因子 (Real-Time Factor) - 基于核心推理耗时计算"""
        if self.duration == 0 or self.stats is None: return 0.0
        return self.stats.inference_only_time / self.duration

    def decode(self, decoder):
        """
        [渲染器注入] 将自己的 codes 解码为音频。
        适用于从 JSON 加载后丢失原始音频的场景。
        
        Args:
            decoder: 具备 .decode(codes, is_final=True) 接口的对象 (如 engine.decoder)
        """
        if self.codes is None:
            logger.error("❌ 无法进行音频解码: Codes 为空。")
            return None
            
        t0 = time.time()
        # 调用注入的解码器
        try:
            self.audio = decoder.decode(self.codes, is_final=True)
            render_time = time.time() - t0
            
            # 统计职责下放：如果统计对象存在，则更新它
            if self.stats:
                self.stats.decoder_render_time = render_time
                
            return self.audio
        except Exception as e:
            logger.error(f"❌ 音频解码失败: {e}")
            return None

    # --- IO 能力 ---

    def play(self, blocking: bool = True):
        """播放音频结果"""
        if self.audio is None or len(self.audio) == 0:
            if self.codes is not None:
                logger.warning("⚠️ 此结果当前无音频数据，但检测到 Codec 特征。请先调用 .decode(engine.decoder) 进行解码渲染。")
            else:
                logger.warning("⚠️ 此结果不包含音频数据，且无可用特征。")
            return
        import sounddevice as sd
        sd.play(self.audio, samplerate=24000, blocking=blocking)

    def save(self, path: str, **kwargs):
        """统一保存方法，根据后缀名自动选择 wav 或 json"""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            self.save_wav(path)
        elif ext == ".json":
            self.save_json(path, **kwargs)
        else:
            logger.error(f"❌ 不支持的保存格式: {ext}。请使用 .wav 或 .json")

    def save_wav(self, path: str):
        """保存为 WAV 文件"""
        if self.audio is None:
            logger.error("❌ No audio data to save.")
            return
        import soundfile as sf
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        sf.write(path, self.audio, SAMPLE_RATE)
        logger.info(f"💾 Audio saved to: {path}")

    # --- 持久化能力 ---

    def save_json(self, path: str, include_audio: bool = False, include_embeds: bool = False, light: bool = False, info: Optional[str] = None):
        """将特征锚点保存到 JSON"""
        if info is not None:
            self.info = info
            
        if not self.is_valid_anchor:
            logger.warning("⚠️ Result is incomplete, cannot save as anchor.")
            return
        
        data = {
            "text": self.text,
            "info": self.info,
            "text_ids": self.text_ids,
            "codes": self.codes.tolist(),
            "spk_emb": self.spk_emb.tolist(),
        }

        if include_embeds and self.summed_embeds is not None:
            data["summed_embeds"] = [e.tolist() for e in self.summed_embeds]
        
        if include_audio and self.audio is not None:
            data["audio"] = self.audio.tolist()
        
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None if light else 2)
        
        logger.info(f"💾 Voice JSON saved to: {path} (Light: {light})")

    @classmethod
    def from_json(cls, path: str):
        """从 JSON 加载锚点 (兼容性更强)"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Identity file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in file '{path}': {e}")

        # Use .get() for robustness against missing keys
        # For spk_emb and codes, if they are critical for an anchor,
        # we might want to raise an error if they are missing or provide a default that makes sense.
        # Here, we'll make them optional for loading, but `is_valid_anchor` will check their presence.
        spk_emb_data = data.get("spk_emb")
        codes_data = data.get("codes")
        text_ids_data = data.get("text_ids")

        res = cls(
            text=data.get("text", ""),
            info=data.get("info", ""),
            text_ids=text_ids_data if text_ids_data is not None else [], # Default to empty list if missing
            spk_emb=np.array(spk_emb_data, dtype=np.float32) if spk_emb_data is not None else None,
            codes=np.array(codes_data, dtype=np.int64) if codes_data is not None else None,
        )

        if "summed_embeds" in data:
            res.summed_embeds = [np.array(e, dtype=np.float32) for e in data["summed_embeds"]]
        
        if "audio" in data:
            res.audio = np.array(data["audio"], dtype=np.float32)
            
        return res

    def print_stats(self):
        """打印性能报告报告"""
        if self.stats is None:
            print("No performance stats available for this result.")
            return
            
        s = self.stats
        print("-" * 40)
        print(f"性能分析报告 (音频长度: {self.duration:.2f}s | 文本长度: {len(self.text)})")
        print(f"  1. Prompt 编译: {s.prompt_time:.4f}s")
        print(f"  2. Talker Prefill: {s.prefill_time:.4f}s")
        print(f"  3. 自回环总计: {s.talker_loop_time + s.predictor_loop_time:.4f}s")
        print(f"     └─ 大师 (Talker): {s.talker_loop_time:.4f}s")
        print(f"     └─ 工匠 (Predictor): {s.predictor_loop_time:.4f}s")
        print(f"  4. 解码渲染 (Decoder): {s.decoder_render_time:.4f}s")
        print("-" * 40)
        print(f"核心推理耗时: {s.inference_only_time:.2f}s | RTF (Core): {self.rtf:.2f}")
        print(f"全链路总响应: {s.total_inference_time:.2f}s")
