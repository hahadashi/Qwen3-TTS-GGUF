"""
采样器 - 对标 GGUF 的 LlamaSampler

完整实现，包含:
1. Temperature 采样
2. Top-K 采样
3. Top-P 采样
4. Min-P 采样
5. Repeat Penalty (重复惩罚)
6. Frequency Penalty (频率惩罚)
7. Presence Penalty (存在惩罚)
8. Vocab Range Limiting (词汇表范围限制)
9. Token 豁免 (EOS/BOS/PAD 不受惩罚)
"""

import torch
from typing import Optional, Set, List, Deque
from collections import deque
import random
import numpy as np


class Sampler:
    """
    采样器 - 完整对标 GGUF 的 LlamaSampler

    支持 Talker 和 Predictor 双采样器架构:
    - Talker: 完整参数 + 惩罚参数，控制语义生成
    - Predictor: 简洁参数，无惩罚，保持声学稳定

    参考 GGUF/llama.cpp 实现:
    - repeat_penalty: 对重复 token 进行对数惩罚
    - frequency_penalty: 基于 token 出现次数的惩罚
    - presence_penalty: 基于 token 是否出现的惩罚
    - penalty_last_n: 惩罚窗口大小
    - exempt_tokens: 豁免惩罚的 token (如 EOS, BOS, PAD)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: Optional[int] = None,
        limit_start: int = 0,
        limit_end: int = -1,
        exempt_tokens: Optional[Set[int]] = None,
    ):
        """
        初始化采样器

        Args:
            temperature: 温度参数，控制随机性 (>0)
            top_k: Top-K 采样，只保留概率最大的 K 个 token (0=禁用)
            top_p: Top-P 采样，核采样 (1.0=禁用)
            min_p: Min-P 采样，相对于最大概率的阈值 (0=禁用)
            repeat_penalty: 重复惩罚 (1.0=禁用, >1 惩罚重复)
            frequency_penalty: 频率惩罚 (0=禁用, >0 惩罚高频)
            presence_penalty: 存在惩罚 (0=禁用, >0 惩罚已出现)
            penalty_last_n: 惩罚窗口大小，只考虑最近 N 个 token
            seed: 随机种子，用于可复现性
            limit_start: 词汇表起始限制
            limit_end: 词汇表结束限制 (-1=不限制)
            exempt_tokens: 豁免惩罚的 token 集合 (如 EOS, BOS, PAD)
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.penalty_last_n = penalty_last_n
        self.seed = seed
        self.limit_start = limit_start
        self.limit_end = limit_end
        self.exempt_tokens = exempt_tokens or set()

        # Token 历史 (用于重复惩罚)
        self._token_history: Deque[int] = deque(maxlen=penalty_last_n if penalty_last_n > 0 else 64)

        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def reset(self):
        """重置采样器状态（清除历史）"""
        self._token_history.clear()

    def accept(self, token: int):
        """
        记录已接受的 token（用于惩罚历史）

        在生成循环中调用此方法来记录采样的 token，
        以便后续采样时应用重复惩罚。

        Args:
            token: 已采样/接受的 token ID
        """
        self._token_history.append(token)

    def _apply_penalties(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用重复/频率/存在惩罚

        参考 GGUF/llama.cpp 实现:
        - repeat_penalty: 如果 token 在历史中出现，对其 logits 除以 penalty
        - frequency_penalty: 基于 token 出现次数的线性惩罚
        - presence_penalty: 如果 token 出现过，固定惩罚值
        - exempt_tokens: 豁免惩罚的 token (如 EOS, BOS, PAD)

        Args:
            logits: 原始 logits [vocab_size]

        Returns:
            logits: 惩罚后的 logits
        """
        if len(self._token_history) == 0:
            return logits

        logits = logits.clone()

        # 统计历史 token 频率
        history_list = list(self._token_history)
        token_counts = {}
        for t in history_list:
            token_counts[t] = token_counts.get(t, 0) + 1

        # 应用惩罚
        for token, count in token_counts.items():
            if token >= logits.shape[0]:
                continue

            # 跳过豁免 token (如 EOS, BOS, PAD)
            if token in self.exempt_tokens:
                continue

            # 1. Repeat Penalty (对数惩罚)
            if self.repeat_penalty != 1.0:
                if logits[token] > 0:
                    logits[token] = logits[token] / self.repeat_penalty
                else:
                    logits[token] = logits[token] * self.repeat_penalty

            # 2. Frequency Penalty (线性惩罚)
            if self.frequency_penalty > 0:
                logits[token] -= self.frequency_penalty * count

            # 3. Presence Penalty (固定惩罚)
            if self.presence_penalty > 0:
                logits[token] -= self.presence_penalty

        return logits

    def _apply_vocab_limit(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用词汇表范围限制

        Args:
            logits: 原始 logits [vocab_size]

        Returns:
            logits: 限制后的 logits
        """
        if self.limit_start == 0 and self.limit_end == -1:
            return logits

        logits = logits.clone()
        vocab_size = logits.shape[0]

        end = self.limit_end if self.limit_end > 0 else vocab_size

        # 将范围外的 token 设为 -inf
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
        mask[self.limit_start:end] = True
        logits[~mask] = float('-inf')

        return logits

    def sample(
        self,
        logits: torch.Tensor,  # [vocab_size]
        allowed_tokens: Optional[Set[int]] = None,
    ) -> int:
        """
        从 logits 采样

        流程:
        1. 应用词汇表限制
        2. 应用重复惩罚
        3. 限制 allowed_tokens
        4. 保护 exempt_tokens (不被 Top-K 过滤)
        5. 应用 temperature
        6. 应用 min_p
        7. 应用 top_k (保护 exempt_tokens)
        8. 应用 top_p
        9. 采样

        Args:
            logits: 预测 logits [vocab_size]
            allowed_tokens: 允许的 token 集合

        Returns:
            token: 采样的 token ID
        """
        logits = logits.clone()

        # 1. 词汇表范围限制
        logits = self._apply_vocab_limit(logits)

        # 2. 应用重复/频率/存在惩罚
        logits = self._apply_penalties(logits)

        # 3. 限制 allowed_tokens
        if allowed_tokens is not None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for token in allowed_tokens:
                if token < logits.shape[0]:
                    mask[token] = True
            logits[~mask] = float('-inf')

        # 4. Temperature
        if self.temperature > 0:
            logits = logits / self.temperature

        # 5. Min-P
        if self.min_p > 0:
            probs = torch.softmax(logits, dim=-1)
            top_prob = probs.max()
            min_p_threshold = top_prob * self.min_p
            logits[probs < min_p_threshold] = float('-inf')

        # 6. Top-K (保护 exempt_tokens，确保它们在 top-k 中)
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            # 应用 Top-K
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)

            # CRITICAL: 确保 exempt_tokens 始终在 top-k 中
            # 如果 exempt_token 不在 top-k 中，替换最弱的 top-k token
            for token in self.exempt_tokens:
                if token < logits.shape[0]:
                    # 检查 token 是否已经在 top-k 中
                    if logits[token] == float('-inf'):
                        # token 不在 top-k 中，找到最弱的 top-k token 并替换
                        # 找到 top-k 中最小的 logit
                        min_top_k_value = top_k_logits.min()
                        if min_top_k_value > float('-inf'):
                            # 将 exempt_token 的 logit 设置为比最弱的 top-k 稍高一点点
                            # 这样它就会被包含在采样中
                            logits[token] = min_top_k_value + 1e-4

        # 7. Top-P
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # 8. 采样
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()

        # 注意: 不在这里调用 accept()，由调用者决定是否记录
        # GGUF 行为: sample() 后需显式调用 accept()

        return token

    def sample_batch(
        self,
        logits: torch.Tensor,  # [batch_size, vocab_size]
        allowed_tokens: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        """
        批量采样

        Args:
            logits: 预测 logits [batch_size, vocab_size]
            allowed_tokens: 允许的 token 集合

        Returns:
            tokens: [batch_size] 采样的 token IDs
        """
        batch_size = logits.shape[0]
        tokens = []
        for i in range(batch_size):
            token = self.sample(logits[i], allowed_tokens)
            tokens.append(token)
        return torch.tensor(tokens, device=logits.device)

    def filter_logits(
        self,
        logits: torch.Tensor,
        allowed_tokens: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        """
        过滤 logits（不采样，只应用过滤）

        Args:
            logits: 原始 logits
            allowed_tokens: 允许的 token 集合

        Returns:
            filtered_logits: 过滤后的 logits
        """
        logits = logits.clone()

        # 词汇表限制
        logits = self._apply_vocab_limit(logits)

        # 重复惩罚
        logits = self._apply_penalties(logits)

        # Temperature
        if self.temperature > 0:
            logits = logits / self.temperature

        # Min-P
        if self.min_p > 0:
            probs = torch.softmax(logits, dim=-1)
            top_prob = probs.max()
            min_p_threshold = top_prob * self.min_p
            logits[probs < min_p_threshold] = float('-inf')

        # Top-K
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)

        # Top-P
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Allowed tokens
        if allowed_tokens is not None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for token in allowed_tokens:
                if token < logits.shape[0]:
                    mask[token] = True
            logits[~mask] = float('-inf')

        return logits

    @property
    def history(self) -> List[int]:
        """获取当前 token 历史"""
        return list(self._token_history)

    @property
    def history_size(self) -> int:
        """获取当前历史大小"""
        return len(self._token_history)


# 向后兼容别名
EnhancedSampler = Sampler
