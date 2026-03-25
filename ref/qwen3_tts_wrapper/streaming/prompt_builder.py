"""
Prompt构建器 - 统一三种模式的Prompt构建

职责:
1. 构建ICL模式Prompt (Clone)
2. 构建Custom模式Prompt
3. 构建Design模式Prompt
4. 处理text+audio融合
"""

import torch
from typing import Optional, Tuple

from ..data import PromptData, VoiceAnchor
from .assets import AssetsManager


class PromptBuilder:
    """
    Prompt构建器 - 统一三种模式的Prompt构建
    """

    def __init__(self, tokenizer, assets: AssetsManager):
        """
        初始化PromptBuilder

        Args:
            tokenizer: 分词器
            assets: 资产管理器
        """
        self.tokenizer = tokenizer
        self.assets = assets
        self.config = assets.config

        # 获取实际的tokenizer (处理Qwen3TTSProcessor)
        if hasattr(tokenizer, 'tokenizer'):
            self._tokenizer = tokenizer.tokenizer
        else:
            self._tokenizer = tokenizer

    def _encode_text(self, text: str) -> list:
        """编码文本为token IDs

        Args:
            text: 输入文本

        Returns:
            token_ids: token ID列表
        """
        return self._tokenizer.encode(text)

    # ========== 公共接口 ==========

    def build_clone_prompt(
        self,
        text: str,
        voice: VoiceAnchor,
        lang_id: Optional[int] = None
    ) -> PromptData:
        """
        构建Clone模式Prompt (ICL模式) - GGUF兼容版本

        Prompt结构 (GGUF风格):
        ┌─────────────────────────────────────────────────────────────┐
        │ [prefix] + [body: text+audio融合]                           │
        │                                                              │
        │ prefix:   role_tokens + THINK_tokens + speaker + TTS_BOS   │
        │ body:     text_pool + audio_pool (逐位相加)                 │
        └─────────────────────────────────────────────────────────────┘

        Args:
            text: 目标文本
            voice: 音色锚点
            lang_id: 语言ID (可选)

        Returns:
            PromptData: 构建好的Prompt数据
        """
        # 1. Tokenize目标文本
        text_ids = self._encode_text(text)

        # 2. 构建GGUF风格prefix (包含role, THINK, speaker)
        prefix_embeds = self._build_prefix_gguf_style(
            speaker_embedding=voice.speaker_embedding,
            lang_id=lang_id
        )

        # 3. 构建GGUF风格ICL body
        body_embeds, trailing_embeds = self._build_icl_body_gguf_style(
            ref_codes=voice.reference_codes,
            ref_text=voice.ref_text,
            target_text_ids=text_ids
        )

        # 4. 拼接完整prefill
        prefill_embeds = torch.cat([
            prefix_embeds,    # [1, prefix_len, hidden]
            body_embeds,      # [1, body_len, hidden]
        ], dim=1)  # [1, total_len, hidden]

        # 5. 构建attention_mask和4D position_ids
        total_len = prefill_embeds.shape[1]
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=prefill_embeds.device)

        # 4D position_ids: [4, batch, seq_len]
        # 模型处理: position_ids[0] → text_position_ids, position_ids[1:] → 3个attention group
        # Native API (2D) 自动扩展为 [pos, pos, pos]
        # 所以这里需要 [pos, pos, pos, pos] 才能匹配 Native API
        positions = torch.arange(total_len, dtype=torch.long, device=prefill_embeds.device)
        position_ids = torch.stack([
            positions,    # text_position_ids (被模型提取为 position_ids[0])
            positions,    # attention group 0
            positions,    # attention group 1
            positions,    # attention group 2
        ], dim=0).unsqueeze(1)  # [4, 1, seq_len]

        return PromptData(
            prefill_embeds=prefill_embeds,
            prefill_attention_mask=attention_mask,
            prefill_position_ids=position_ids,
            trailing_text_embeds=trailing_embeds,
            text=text,
            text_ids=text_ids,
            speaker_embedding=voice.speaker_embedding,
            ref_codes=voice.reference_codes,
            ref_text=voice.ref_text
        )

    def build_custom_prompt(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        instruct: Optional[str] = None,
        lang_id: Optional[int] = None
    ) -> PromptData:
        """
        构建Custom模式Prompt

        Args:
            text: 目标文本
            speaker_embedding: 说话人嵌入
            instruct: 指令文本 (可选)
            lang_id: 语言ID (可选)

        Returns:
            PromptData: 构建好的Prompt数据
        """
        # TODO: 实现Custom模式
        raise NotImplementedError("Custom模式暂未实现")

    # ========== 内部方法 ==========

    def _build_prefix_gguf_style(
        self,
        speaker_embedding: Optional[torch.Tensor] = None,
        lang_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        构建 GGUF 风格的 prefix embeddings

        GGUF 结构:
        1. Role tokens: <|im_start|>assistant\n (3 tokens)
        2. THINK tokens: [THINK, THINK_BOS, lang_id, THINK_EOS] (4 tokens)
        3. Speaker: tts_pad + spk_emb (1 token)
        4. TTS_BOS + codec_pad: text_emb[TTS_BOS] + codec_emb[PAD] (1 token)

        Returns:
            prefix_embeds: [1, prefix_len, hidden]
        """
        prefix_embs = []
        tts_pad = self.assets.tts_pad

        # 1. Role tokens: <|im_start|>assistant\n
        role_text = "<|im_start|>assistant\n"
        role_ids = self._encode_text(role_text)
        for tid in role_ids:
            prefix_embs.append(self.assets.text_embeddings[tid])

        # 2. THINK tokens
        # 从 config 读取正确的 token IDs
        talker_cfg = self.assets.config.talker_config
        THINK_ID = talker_cfg.codec_think_id          # 2154
        NOHINK_ID = talker_cfg.codec_nothink_id       # 2155
        THINK_BOS_ID = talker_cfg.codec_think_bos_id  # 2156
        THINK_EOS_ID = talker_cfg.codec_think_eos_id  # 2157

        if lang_id is not None:
            # think mode: [THINK, THINK_BOS, lang_id, THINK_EOS] → 4 tokens
            think_tokens = [THINK_ID, THINK_BOS_ID, lang_id, THINK_EOS_ID]
        else:
            # nothink mode (Auto): [NOHINK, THINK_BOS, THINK_EOS] → 3 tokens
            think_tokens = [NOHINK_ID, THINK_BOS_ID, THINK_EOS_ID]

        # 使用 embedding layer 调用方式 (与 Native API 一致)
        if hasattr(self.assets, 'talker_codec_embedding') and self.assets.talker_codec_embedding is not None:
            codec_emb_layer = self.assets.talker_codec_embedding
            # 批量获取 embeddings，与 Native API 方式一致
            think_token_tensor = torch.tensor([think_tokens], dtype=torch.long)
            think_embeds = codec_emb_layer(think_token_tensor)[0]  # [num_tokens, hidden]
            for i, tid in enumerate(think_tokens):
                if tid < codec_emb_layer.weight.shape[0]:
                    prefix_embs.append(tts_pad + think_embeds[i])
                else:
                    prefix_embs.append(tts_pad)
        else:
            # Fallback: 使用 text_embedding (不完全正确)
            for tid in think_tokens:
                if tid < self.assets.text_embeddings.shape[0]:
                    prefix_embs.append(tts_pad + self.assets.text_embeddings[tid])
                else:
                    prefix_embs.append(tts_pad)

        # 3. Speaker embedding
        if speaker_embedding is not None:
            # 确保 speaker_embedding 是 1D
            if speaker_embedding.dim() > 1:
                speaker_embedding = speaker_embedding.squeeze()
            prefix_embs.append(tts_pad + speaker_embedding)

        # 4. TTS_BOS + codec_pad
        tts_bos_id = self.assets.tts_bos_id if self.assets.tts_bos_id else 151672
        codec_pad_id = self.assets.codec_pad_id

        if tts_bos_id < self.assets.text_embeddings.shape[0]:
            tts_bos_emb = self.assets.text_embeddings[tts_bos_id]
        else:
            tts_bos_emb = tts_pad

        # codec_pad 需要从 Talker 的 codec_embedding 获取
        if hasattr(self.assets, 'talker_codec_embedding') and self.assets.talker_codec_embedding is not None:
            codec_pad_emb = self.assets.talker_codec_embedding.weight[codec_pad_id]
        else:
            # Fallback: 使用 codec_embeddings 的平均值
            codec_pad_emb = self.assets.codec_embeddings[:, codec_pad_id, :].mean(dim=0)

        prefix_embs.append(tts_bos_emb + codec_pad_emb)

        return torch.stack(prefix_embs).unsqueeze(0)  # [1, prefix_len, hidden]

    def _build_prefix(self) -> torch.Tensor:
        """
        构建prefix embeddings (旧版本，保留兼容性)

        Returns:
            prefix_embeds: [1, 3, hidden]
        """
        tokens = [
            self.assets.tts_bos_id if self.assets.tts_bos_id is not None else self.assets.codec_bos_id,
            self.assets.tts_bos_id if self.assets.tts_bos_id is not None else self.assets.codec_bos_id,  # text_bos
            self.assets.codec_bos_id,
        ]

        embs = []
        for t in tokens:
            if t < self.assets.text_embeddings.shape[0]:
                embs.append(self.assets.text_embeddings[t])
            else:
                # 使用tts_pad作为fallback
                embs.append(self.assets.tts_pad)

        return torch.stack(embs).unsqueeze(0)  # [1, 3, hidden]

    def _build_icl_body_gguf_style(
        self,
        ref_codes: torch.Tensor,
        ref_text: Optional[str],
        target_text_ids: list,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        构建 GGUF 风格的 ICL body

        GGUF ICL 结构:
        1. text_pool = (ref_text_ids + target_text_ids) + [TTS_EOS]
        2. audio_pool = [BOS] + [sum of 16-layer embeddings for each frame]
        3. ICL fusion: text_pool[:a_len] + audio_pool (逐位相加)
        4. trailing = text_pool[a_len:] (剩余文本，包含目标文本)

        Returns:
            body_embeds: [1, body_len, hidden]
            trailing_embeds: [1, trailing_len, hidden] or None
        """
        ref_len = ref_codes.shape[0]
        hidden_dim = self.assets.tts_pad.shape[0]

        # 1. 构建 text_pool (GGUF: ref_text + target_text + TTS_EOS)
        if ref_text:
            ref_text_ids = self._encode_text(ref_text)
        else:
            ref_text_ids = []

        # GGUF: text_pool = (ref_text + target_text) + TTS_EOS
        tts_eos_id = self.assets.tts_eos_id if self.assets.tts_eos_id else 151673
        text_pool_ids = ref_text_ids + target_text_ids + [tts_eos_id]
        text_pool = self.assets.text_embeddings[text_pool_ids]

        # 2. 构建 audio_pool
        # audio_pool = [BOS] + [sum of 16-layer embeddings for each frame]
        audio_pool = []

        # BOS token - 使用 Talker 的 codec_embedding
        codec_bos_id = self.assets.codec_bos_id
        if hasattr(self.assets, 'talker_codec_embedding') and self.assets.talker_codec_embedding is not None:
            audio_pool.append(self.assets.talker_codec_embedding.weight[codec_bos_id])
        else:
            # Fallback
            audio_pool.append(self.assets.codec_embeddings[0, codec_bos_id, :])

        # 每帧的 16 层 embedding 求和
        # CRITICAL: 不同层使用不同的 embedding 表 (对标 Native API)
        # - Layer 0: talker_codec_embedding (vocab=3072)
        # - Layer 1-15: predictor.codec_embeddings[q-1] (vocab=2048)
        for t in range(ref_len):
            frame_sum = torch.zeros(hidden_dim)
            for q in range(16):
                code = ref_codes[t, q].item()
                if q == 0:
                    # Layer 0: 使用 Talker 的 codec_embedding (vocab=3072)
                    if hasattr(self.assets, 'talker_codec_embedding') and self.assets.talker_codec_embedding is not None:
                        frame_sum += self.assets.talker_codec_embedding.weight[code]
                    else:
                        # Fallback: 不应该发生
                        print(f"Warning: talker_codec_embedding not available for layer 0")
                else:
                    # Layer 1-15: 使用 Predictor 的 codec_embeddings (vocab=2048)
                    # assets.codec_embeddings[q] 包含 layer q 的 embedding (q=1~15)
                    if code < self.assets.codec_embeddings.shape[1]:
                        frame_sum += self.assets.codec_embeddings[q, code, :]
                    else:
                        # code 超出 vocab 范围，使用 fallback
                        print(f"Warning: code {code} out of range for layer {q}")

            audio_pool.append(frame_sum)

        audio_pool = torch.stack(audio_pool)

        # 3. ICL fusion (GGUF 风格)
        # GGUF: body = text_pool[:a_len] + audio_pool
        # trailing = text_pool[a_len:] (仅在 text_len > audio_len 时有剩余)
        t_len, a_len = len(text_pool), len(audio_pool)

        if t_len > a_len:
            # 文本比音频长，有剩余文本需要作为 trailing
            body = text_pool[:a_len] + audio_pool
            trailing = text_pool[a_len:]
            trailing_embeds = trailing.unsqueeze(0) if len(trailing) > 0 else None
            print(f"  [Prompt Debug] text_len({t_len}) > audio_len({a_len}), trailing has {len(trailing)} tokens")
        else:
            # 文本比音频短或相等，所有文本已融合在 body 中
            # GGUF: trailing = None (不设置)
            pad_seq = self.assets.tts_pad.unsqueeze(0).expand(a_len - t_len, -1)
            text_padded = torch.cat([text_pool, pad_seq], dim=0)
            body = text_padded + audio_pool
            trailing_embeds = None  # GGUF 风格: 当 text_len <= audio_len 时，trailing 为 None
            print(f"  [Prompt Debug] text_len({t_len}) <= audio_len({a_len}), trailing=None (GGUF style)")

        return body.unsqueeze(0), trailing_embeds

    def _build_trailing_text(self, text_ids: list) -> torch.Tensor:
        """
        构建trailing text embeddings

        Args:
            text_ids: 文本token IDs

        Returns:
            trailing_embeds: [1, text_len, hidden]
        """
        embs = self.assets.text_embeddings[text_ids]
        return embs.unsqueeze(0)  # [1, text_len, hidden]

    def _build_4d_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构建4维位置编码

        格式: [pos, pos, pos, 0]
        - 前3层: 正常位置编码
        - 第4层: 全零

        Args:
            seq_len: 序列长度
            device: 设备

        Returns:
            position_ids: [1, seq_len, 4]
        """
        positions = torch.arange(seq_len, device=device)
        return torch.stack([
            positions,          # Layer 0
            positions,          # Layer 1
            positions,          # Layer 2
            torch.zeros_like(positions),  # Layer 3
        ], dim=-1).unsqueeze(0)  # [1, seq_len, 4]
