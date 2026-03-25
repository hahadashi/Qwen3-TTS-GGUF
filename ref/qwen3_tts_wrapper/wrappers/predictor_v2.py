"""
Predictor Wrapper V2 - GQA 兼容版本

封装Predictor模型，负责生成codec_1~15。

关键改进:
1. 使用 DynamicCache 管理 KV Cache
2. 正确调用 Predictor 的 forward 方法
3. 支持逐帧生成 (流式)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None


@dataclass
class PredictorStateV2:
    """Predictor 状态 (GQA兼容)"""
    past_key_values: Optional['DynamicCache']
    position: int
    batch_size: int = 1


class PredictorWrapperV2:
    """
    Predictor Wrapper V2 - GQA 兼容实现

    Predictor 是 5 层 Transformer，负责基于 codec_0 生成 codec_1~15。

    模型结构:
    - Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
      - model (backbone): Qwen3TTSTalkerCodePredictorModel (5层)
      - model.codec_embedding: ModuleList[15] - 每层一个 Embedding
      - lm_head: ModuleList[15] - 每层一个 Linear
      - small_to_mtp_projection: Linear(2048->1024) for 1.7B
    """

    def __init__(self, code_predictor, talker_codec_embedding=None):
        """
        初始化 Predictor Wrapper

        Args:
            code_predictor: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration 实例
            talker_codec_embedding: Talker 的 codec_embedding (用于 code_0 的音频反馈)
        """
        self.model = code_predictor
        self.backbone = code_predictor.model  # Qwen3TTSTalkerCodePredictorModel
        self.config = code_predictor.config

        # 关键配置
        self.num_layers = getattr(self.config, 'num_hidden_layers', 5)
        self.num_code_groups = getattr(self.config, 'num_code_groups', 16)  # 16 codebooks
        self.hidden_size = getattr(self.config, 'hidden_size', 1024)
        self.vocab_size = getattr(self.config, 'vocab_size', 2048)

        # 投影层 (1.7B 模型特有)
        self.projection = getattr(code_predictor, 'small_to_mtp_projection', None)
        if self.projection is None:
            self.projection = torch.nn.Identity()

        # 获取 codec_embedding 列表
        self.codec_embeddings = self.backbone.codec_embedding  # ModuleList[15]

        # 获取 lm_head 列表
        self.lm_heads = code_predictor.lm_head  # ModuleList[15]

        # Talker 的 codec_embedding (用于 code_0 的音频反馈)
        # 注意: code_0 来自 Talker 采样，vocab_size=3072
        # 而 Predictor 的 codec_embeddings vocab_size=2048
        self.talker_codec_embedding = talker_codec_embedding

        print(f"[PredictorWrapperV2] 初始化完成:")
        print(f"  - num_layers: {self.num_layers}")
        print(f"  - num_code_groups: {self.num_code_groups}")
        print(f"  - hidden_size: {self.hidden_size}")
        print(f"  - vocab_size: {self.vocab_size}")
        print(f"  - has_projection: {not isinstance(self.projection, torch.nn.Identity)}")
        print(f"  - has_talker_codec_embedding: {talker_codec_embedding is not None}")

    @property
    def device(self) -> torch.device:
        """返回模型设备"""
        return next(self.model.parameters()).device

    def init_state(self, batch_size: int = 1) -> PredictorStateV2:
        """初始化状态"""
        return PredictorStateV2(
            past_key_values=None,
            position=0,
            batch_size=batch_size
        )

    def get_codec_embedding(self, layer: int, code: int) -> torch.Tensor:
        """
        获取指定层的 codec embedding

        Args:
            layer: 层索引 (0-14, 对应 codec_1~15)
            code: code 值

        Returns:
            embedding: [hidden_size]
        """
        code_tensor = torch.tensor([code], dtype=torch.long, device=self.device)
        return self.codec_embeddings[layer](code_tensor).squeeze(0)  # [hidden]

    @torch.no_grad()
    def prefill(
        self,
        master_hidden: torch.Tensor,
        code_0: int,
    ) -> Tuple[torch.Tensor, PredictorStateV2]:
        """
        Prefill 阶段 - 使用 master_hidden 和 code_0 初始化

        Args:
            master_hidden: [batch, talker_hidden] Talker 的 hidden state
            code_0: 第 0 层 code

        Returns:
            logits: [batch, vocab_size] codec_1 的预测 logits
            state: 更新后的状态
        """
        batch_size = master_hidden.shape[0]

        # 1. 获取 codec_0 的 embedding (2048-dim, talker hidden size)
        # CRITICAL: code_0 来自 Talker，使用 Talker 的 codec_embedding (vocab=3072)
        code_0_tensor = torch.tensor([code_0], dtype=torch.long, device=self.device)
        if self.talker_codec_embedding is not None:
            code_0_embed = self.talker_codec_embedding(code_0_tensor)  # [1, talker_hidden]
        else:
            code_0_embed = self.codec_embeddings[0](code_0_tensor)  # [1, talker_hidden]

        # 2. 拼接输入: [master_hidden, code_0_embed]
        # master_hidden: [batch, talker_hidden] -> [batch, 1, talker_hidden]
        # code_0_embed: [1, talker_hidden] -> [batch, 1, talker_hidden]
        master_batch = master_hidden.unsqueeze(1)  # [batch, 1, talker_hidden]
        code_0_batch = code_0_embed.expand(batch_size, -1).unsqueeze(1)  # [batch, 1, talker_hidden]

        inputs_embeds = torch.cat([
            master_batch,    # [batch, 1, talker_hidden]
            code_0_batch     # [batch, 1, talker_hidden]
        ], dim=1)  # [batch, 2, talker_hidden]

        # 3. 应用投影 (2048 -> 1024 for 1.7B)
        inputs_embeds = self.projection(inputs_embeds)  # [batch, 2, predictor_hidden]

        # 4. 初始化 DynamicCache
        if HAS_DYNAMIC_CACHE:
            past_key_values = DynamicCache()
        else:
            past_key_values = None

        # 5. 调用 backbone
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # 6. 获取 hidden states 并通过 lm_head[0] 得到 logits
        hidden_states = outputs.last_hidden_state  # [batch, 2, hidden]
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]

        # 使用第 0 个 lm_head 预测 codec_1
        logits = self.lm_heads[0](last_hidden)  # [batch, vocab_size]

        # 7. 获取 cache 长度
        if outputs.past_key_values is not None and hasattr(outputs.past_key_values, 'get_seq_length'):
            cache_len = outputs.past_key_values.get_seq_length(layer_idx=0)
        else:
            cache_len = inputs_embeds.shape[1]

        state = PredictorStateV2(
            past_key_values=outputs.past_key_values,
            position=cache_len,
            batch_size=batch_size
        )

        return logits, state

    @torch.no_grad()
    def decode_step(
        self,
        prev_code: int,
        layer_idx: int,
        state: PredictorStateV2,
    ) -> Tuple[torch.Tensor, PredictorStateV2]:
        """
        单步解码 - 生成下一层的 code

        Args:
            prev_code: 上一层采样的 code (codec_{layer_idx})
            layer_idx: 当前层索引 (1-14, 对应生成 codec_{layer_idx+1})
                      例如: layer_idx=1 表示生成 codec_2，使用 codec_1 的 embedding
            state: 当前状态

        Returns:
            logits: [batch, vocab_size] 当前层的预测 logits
            new_state: 更新后的状态
        """
        batch_size = state.batch_size

        # 1. 获取上一层 code 的 embedding (2048-dim, talker hidden size)
        # CRITICAL FIX: When generating codec_{layer_idx+1}, we need to use codec_embeddings[layer_idx-1]
        # which is the embedding for the PREVIOUS code (codec_{layer_idx})
        # Example: When layer_idx=1 (generating codec_2), use codec_embeddings[0] (for codec_1)
        prev_code_tensor = torch.tensor([prev_code], dtype=torch.long, device=self.device)
        prev_embed = self.codec_embeddings[layer_idx - 1](prev_code_tensor)  # [1, talker_hidden]
        inputs_embeds = prev_embed.expand(batch_size, -1).unsqueeze(1)  # [batch, 1, talker_hidden]

        # 2. 应用投影 (2048 -> 1024 for 1.7B)
        inputs_embeds = self.projection(inputs_embeds)  # [batch, 1, predictor_hidden]

        # 3. 调用 backbone (使用 KV Cache)
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # 4. 获取 hidden states
        hidden_states = outputs.last_hidden_state  # [batch, 1, hidden]
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]

        # 5. 通过 lm_head[layer_idx] 得到 logits
        logits = self.lm_heads[layer_idx](last_hidden)  # [batch, vocab_size]

        # 6. 更新状态
        new_state = PredictorStateV2(
            past_key_values=outputs.past_key_values,
            position=state.position + 1,
            batch_size=batch_size
        )

        return logits, new_state

    @torch.no_grad()
    def predict_frame(
        self,
        master_hidden: torch.Tensor,
        code_0: int,
        temperature: float = 0.7,
        sampler=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        逐帧生成完整的 16 层 codes

        实现要点:
        1. 投影层: 1.7B 需要 2048->1024
        2. Prefill: 使用 master_hidden + code_0_embed 初始化
        3. 循环生成 codec_1~15
        4. 收集 embeddings 用于音频反馈

        Args:
            master_hidden: [batch, talker_hidden] Talker 的 hidden state
            code_0: 已采样的第 0 层 code
            temperature: 采样温度 (仅当 sampler=None 时使用)
            sampler: 采样器 (可选，推荐使用 Predictor 专用采样器)

        Returns:
            codes_16: [batch, 16] 完整的 16 层 codes
            embeddings_16: List[16] of [batch, talker_hidden] 16 层的 embeddings (用于反馈)
        """
        batch_size = master_hidden.shape[0]
        device = master_hidden.device

        # 获取 talker hidden size (embedding dim)
        # codec_embeddings use talker hidden size (2048), not predictor hidden size (1024)
        talker_hidden_size = master_hidden.shape[-1]

        # 初始化
        codes = [code_0]
        embeddings = []

        # 获取 code_0 的 embedding
        # CRITICAL FIX: code_0 来自 Talker，必须使用 Talker 的 codec_embedding (vocab=3072)
        # 而不是 Predictor 的 codec_embeddings (vocab=2048)
        code_0_tensor = torch.tensor([code_0], dtype=torch.long, device=device)

        if self.talker_codec_embedding is not None:
            # 正确: 使用 Talker 的 codec_embedding
            code_0_embed = self.talker_codec_embedding(code_0_tensor).expand(batch_size, -1)
        else:
            # 回退: 使用 Predictor 的 codec_embeddings[0] (可能有错误)
            # 如果 code_0 > 2047 会出错
            code_0_embed = self.codec_embeddings[0](code_0_tensor).expand(batch_size, -1)

        embeddings.append(code_0_embed)  # [batch, talker_hidden]

        # Debug: 打印第一个 embedding 的统计
        if len(embeddings) == 1:
            print(f"    [Embed Debug] code_0={code_0}, emb shape={code_0_embed.shape}")
            print(f"      emb[0] stats: mean={code_0_embed[0].mean().item():.6f}, std={code_0_embed[0].std().item():.6f}")
            print(f"      emb[0] first 5: {code_0_embed[0, :5].tolist()}")

        # 1. Prefill: 生成 codec_1
        logits_1, state = self.prefill(master_hidden, code_0)

        # 采样 codec_1
        if sampler is not None:
            code_1 = sampler.sample(logits_1[0])
            # TODO: Investigate if accept() should be called for predictor
            # sampler.accept(code_1)
        elif temperature > 0:
            probs = torch.softmax(logits_1 / temperature, dim=-1)
            code_1 = torch.multinomial(probs[0], num_samples=1).item()
        else:
            code_1 = torch.argmax(logits_1[0]).item()

        codes.append(code_1)

        # 获取 codec_1 的 embedding (用于音频反馈)
        code_1_embed = self.codec_embeddings[0](
            torch.tensor([code_1], device=device)
        ).expand(batch_size, -1)
        embeddings.append(code_1_embed)  # [batch, talker_hidden]

        # 2. 循环生成 codec_2~15
        for layer_idx in range(1, 15):  # layer_idx: 1-14
            # 单步解码
            logits, state = self.decode_step(
                prev_code=codes[-1],  # 上一层的 code
                layer_idx=layer_idx,
                state=state
            )

            # 采样
            if sampler is not None:
                code = sampler.sample(logits[0])
                # TODO: Investigate if accept() should be called for predictor
                # sampler.accept(code)
            elif temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                code = torch.multinomial(probs[0], num_samples=1).item()
            else:
                code = torch.argmax(logits[0]).item()

            codes.append(code)

            # 获取 embedding
            code_embed = self.codec_embeddings[layer_idx](
                torch.tensor([code], device=device)
            ).expand(batch_size, -1)
            embeddings.append(code_embed)

        # 3. 汇总结果
        codes_16 = torch.tensor([codes], dtype=torch.long, device=device)  # [1, 16]

        return codes_16, embeddings

    def debug_cache(self, state: PredictorStateV2, stage: str = ""):
        """调试 KV Cache 状态"""
        print(f"\n[{stage}] Predictor Cache Debug:")
        if state.past_key_values is None:
            print("  Cache: None")
            return

        if hasattr(state.past_key_values, 'get_seq_length'):
            seq_len = state.past_key_values.get_seq_length(layer_idx=0)
            print(f"  DynamicCache seq_len: {seq_len}")

            if hasattr(state.past_key_values, 'key_cache') and len(state.past_key_values.key_cache) > 0:
                k = state.past_key_values.key_cache[0]
                v = state.past_key_values.value_cache[0]
                print(f"  Layer 0 K shape: {k.shape}")
                print(f"  Layer 0 V shape: {v.shape}")
        else:
            print(f"  Tuple Cache, {len(state.past_key_values)} layers")
