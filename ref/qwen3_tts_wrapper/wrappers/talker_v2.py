"""
Talker Wrapper组件 - GQA兼容版本

基于 HuggingFace GQA 处理机制的重新实现。

关键发现:
1. 12Hz 模型使用 GQA (num_kv_heads=8, num_attn_heads=16)
2. KV Cache 使用 DynamicCache 类，形状为 [batch, 8, seq, head_dim]
3. 模型内部处理 repeat_kv 扩展
4. 需要正确传递 cache_position
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

# HuggingFace Cache
try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None


@dataclass
class TalkerState:
    """Talker 状态 (GQA兼容)"""
    past_key_values: Optional['DynamicCache']  # DynamicCache 对象
    position: int                               # 当前位置
    generated_codes: List[int]                  # 已生成的codes
    batch_size: int = 1

    def __post_init__(self):
        if self.generated_codes is None:
            self.generated_codes = []


class TalkerWrapperV2:
    """
    Talker Wrapper V2 - GQA 兼容实现

    关键改进:
    1. 使用 DynamicCache 管理KV Cache (HuggingFace标准)
    2. 正确处理 cache_position
    3. 让模型内部处理 position_embeddings
    """

    def __init__(self, talker_model):
        """
        初始化 Talker Wrapper

        Args:
            talker_model: Qwen3TTSTalkerForConditionalGeneration 实例
        """
        self.model = talker_model
        self.backbone = talker_model.model  # Qwen3TTSTalkerModel
        self.config = talker_model.config

        # 从 config 提取关键参数
        self.num_layers = getattr(self.config, 'num_hidden_layers', 28)
        self.num_attention_heads = getattr(self.config, 'num_attention_heads', 16)
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', 8)
        self.hidden_size = getattr(self.config, 'hidden_size', 2048)
        self.head_dim = self.hidden_size // self.num_attention_heads

        # GQA 比例
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Token IDs
        self.codec_bos_id = getattr(self.config, 'codec_bos_id', 2149)
        self.codec_eos_id = getattr(self.config, 'codec_eos_token_id', 2150)
        self.codec_pad_id = getattr(self.config, 'codec_pad_id', 2148)

        print(f"[TalkerWrapperV2] 初始化完成:")
        print(f"  - num_attention_heads: {self.num_attention_heads}")
        print(f"  - num_key_value_heads: {self.num_key_value_heads}")
        print(f"  - GQA groups: {self.num_key_value_groups}")
        print(f"  - hidden_size: {self.hidden_size}")
        print(f"  - num_layers: {self.num_layers}")

    @property
    def device(self) -> torch.device:
        """返回模型设备"""
        return next(self.model.parameters()).device

    def init_state(self, batch_size: int = 1) -> TalkerState:
        """
        初始化状态

        Returns:
            TalkerState: 初始化的状态
        """
        return TalkerState(
            past_key_values=None,
            position=0,
            generated_codes=[],
            batch_size=batch_size
        )

    @torch.no_grad()
    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, 'TalkerState']:
        """
        Prefill 阶段 - 处理初始 prompt

        Args:
            inputs_embeds: [batch, seq_len, hidden] 初始 embeddings
            attention_mask: [batch, seq_len] 注意力掩码
            position_ids: [4, batch, seq_len] 或 [3, batch, seq_len] 4D位置编码
            speaker_embedding: [batch, spk_dim] 说话人嵌入

        Returns:
            logits: [batch, vocab_size] codec_0 预测 logits (用于第一次采样)
            last_hidden: [batch, hidden] 最后位置的 hidden state
            state: 更新后的状态
        """
        batch_size = inputs_embeds.shape[0]

        # 初始化 DynamicCache
        if HAS_DYNAMIC_CACHE:
            past_key_values = DynamicCache()
        else:
            past_key_values = None

        # 调用 backbone forward
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # 获取最后 hidden state
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]

        # 计算 logits (直接通过 codec_head，与 GGUF 一致)
        # CRITICAL FIX: 不要使用 text_projection，它是用于文本输入的
        # codec-only 输入应该直接通过 codec_head
        logits = self.model.codec_head(hidden_states)[:, -1, :]  # [batch, vocab_size]

        # 获取 cache 长度
        if outputs.past_key_values is not None and hasattr(outputs.past_key_values, 'get_seq_length'):
            cache_len = outputs.past_key_values.get_seq_length(layer_idx=0)
        else:
            cache_len = inputs_embeds.shape[1]

        # 创建状态
        state = TalkerState(
            past_key_values=outputs.past_key_values,
            position=cache_len,
            generated_codes=[],
            batch_size=batch_size
        )

        return logits, last_hidden, state

    @torch.no_grad()
    def decode_step(
        self,
        inputs_embeds: torch.Tensor,
        state: TalkerState,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TalkerState]:
        """
        单步解码 - GQA 兼容版本

        关键: 让 HuggingFace 模型内部处理 GQA 的 KV Cache

        Args:
            inputs_embeds: [batch, 1, hidden] 单步输入
            state: 当前状态
            attention_mask: [batch, cache_len+1] 注意力掩码
            position_ids: [4, batch, 1] 或 [3, batch, 1] 4D位置编码

        Returns:
            logits: [batch, vocab_size] codec_0 预测 logits
            new_state: 更新后的状态
            last_hidden: [batch, hidden] 最后 hidden state
        """
        # 调用 backbone forward
        # 关键: 使用 state.past_key_values (DynamicCache)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=state.past_key_values,  # DynamicCache 对象
            use_cache=True,
            return_dict=True,
        )

        # 获取 hidden states
        hidden_states = outputs.last_hidden_state  # [batch, 1, hidden]

        # 计算 logits (直接通过 codec_head，与 GGUF 一致)
        # CRITICAL FIX: 不要使用 text_projection，它是用于文本输入的
        # codec-only 输入应该直接通过 codec_head
        logits = self.model.codec_head(hidden_states)[:, -1, :]  # [batch, vocab_size]

        # 获取 last_hidden
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]

        # 更新状态
        new_state = TalkerState(
            past_key_values=outputs.past_key_values,  # 更新后的 DynamicCache
            position=state.position + 1,
            generated_codes=state.generated_codes,
            batch_size=state.batch_size
        )

        return logits, new_state, last_hidden

    @torch.no_grad()
    def decode_step_simple(
        self,
        fused_embed: torch.Tensor,
        state: TalkerState,
    ) -> Tuple[torch.Tensor, TalkerState, torch.Tensor]:
        """
        简化的单步解码 - 自动处理位置编码和attention_mask

        这个版本让模型内部处理:
        - cache_position 计算
        - position_embeddings 生成
        - attention_mask 构造

        Args:
            fused_embed: [batch, hidden] 融合后的 embedding
            state: 当前状态

        Returns:
            logits: [batch, vocab_size]
            new_state: 更新后的状态
            last_hidden: [batch, hidden]
        """
        # 添加时间维度
        inputs_embeds = fused_embed.unsqueeze(1)  # [batch, 1, hidden]

        # 构造 4D 位置编码
        # 格式: [4, batch, 1] = [[text_pos], [pos], [pos], [pos]]
        # Native API 参考: 2D position_ids 自动扩展为 [pos, pos, pos]
        # 模型处理: position_ids[0] → text_position_ids, position_ids[1:] → 3个 attention group
        position = state.position
        batch_size = inputs_embeds.shape[0]

        position_ids = torch.zeros(4, batch_size, 1, dtype=torch.long, device=inputs_embeds.device)
        position_ids[0] = position  # text position
        position_ids[1] = position  # temporal
        position_ids[2] = position  # height
        position_ids[3] = position  # width (NOT 0!)

        # Debug: 打印位置信息
        if position < 55:  # 只打印前几步
            print(f"    [Decode Debug] position={position}, position_ids=[{position}, {position}, {position}, 0]")

        # 不传递 attention_mask，让 HuggingFace 内部处理
        # 当使用 DynamicCache 时，模型会自动处理因果注意力
        logits, new_state, last_hidden = self.decode_step(
            inputs_embeds=inputs_embeds,
            state=state,
            position_ids=position_ids,
            attention_mask=None,  # 让模型内部处理
        )

        return logits, new_state, last_hidden

    def debug_cache(self, state: TalkerState, stage: str = ""):
        """调试 KV Cache 状态"""
        print(f"\n[{stage}] Cache Debug:")
        if state.past_key_values is None:
            print("  Cache: None")
            return

        if hasattr(state.past_key_values, 'get_seq_length'):
            seq_len = state.past_key_values.get_seq_length(layer_idx=0)
            print(f"  DynamicCache seq_len: {seq_len}")

            # 检查第一层的 K/V 形状
            if hasattr(state.past_key_values, 'key_cache') and len(state.past_key_values.key_cache) > 0:
                k = state.past_key_values.key_cache[0]
                v = state.past_key_values.value_cache[0]
                print(f"  Layer 0 K shape: {k.shape}")  # 期望: [batch, num_kv_heads, seq, head_dim]
                print(f"  Layer 0 V shape: {v.shape}")
                print(f"  num_kv_heads (from shape): {k.shape[1]}")
        else:
            print(f"  Tuple Cache, {len(state.past_key_values)} layers")
