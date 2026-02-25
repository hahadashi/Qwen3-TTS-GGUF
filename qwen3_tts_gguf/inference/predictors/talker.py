"""
predictors/talker.py - 大师预测器 (Talker)
负责 Qwen3-TTS 的主体 LLM 推理。
管理 llama.cpp 上下文、KV Cache 和步数。
"""
import ctypes
import numpy as np
import time
from .. import llama

from ..constants import PROTOCOL

class TalkerPredictor:
    """
    封装大师模型 (Talker) 的推理行为。
    """
    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets
        
        self.n_ctx = 4096 # 默认上下文大小
        self.cur_pos = 0
        
        # 预分配单步位置 Buffer (3 pos + 1 zero)
        self.pos_step_buffer = np.zeros(4, dtype=np.int32)
        
    def clear_memory(self):
        """完全清空 KV Cache"""
        self.ctx.clear_kv_cache()
        self.cur_pos = 0

    def prefill(self, prompt_embeds: np.ndarray, seq_id: int = 0) -> np.ndarray:
        """
        全量推入初始 Prompt 或新文本批次。
        Args:
            prompt_embeds: [Batch=1, Seq, Hidden=2048]
            seq_id: 分配给此批次的序列 ID (用于精确删除)
        Returns:
            hidden: [Batch, Hidden]
        """
        n_p = prompt_embeds.shape[1]
        
        # 2. 构造 Qwen3 专用的位置编码 (3层Pos + 1层Zero)
        # stride = n_p
        pos_base = np.arange(self.cur_pos, self.cur_pos + n_p, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_p, dtype=np.int32)])
        
        # 使用 LlamaBatch 的高阶接口注入数据
        self.batch.set_embd(prompt_embeds[0], pos=pos_arr, seq_id=seq_id)
            
        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Master Prefill Decode failed with status {llama_status} at pos {self.cur_pos}")

        # 提取最后一个位置的输出
        hidden_ptr = self.ctx.get_embeddings()
        
        # [OPTIMIZATION] 移除 Logits Copy
        # logits_ptr = self.ctx.get_logits()
        # logits = np.ctypeslib.as_array(logits_ptr, shape=(n_p, 3072))[-1].copy()
        
        # 直接返回 C++ 内存的 View (切片操作本身就是 View)
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(n_p, 2048))[-1]
        
        self.cur_pos += n_p
        return hidden

    def decode_step(self, feedback_embed: np.ndarray, seq_id: int = 0) -> np.ndarray:
        """
        单步预测下一帧。
        Args:
            feedback_embed: [Hidden=2048] 汇总后的上一帧反馈向量
            seq_id: 序列 ID
        Returns:
            hidden: [1, Hidden]
        """
        # 检查上下文是否已满
        if self.cur_pos >= self.n_ctx - 1:
            raise IndexError(f"Master context overflow: {self.cur_pos} >= {self.n_ctx}")
            
        # 构造成 (1, 2048) 注入
        if feedback_embed.ndim == 1:
            feedback_embed = feedback_embed.reshape(1, -1)
            
        # 构造 Qwen3 单步位置编码 (N=1, 3层Pos + 1层Zero)
        # [OPTIMIZATION] 使用预分配 Buffer
        self.pos_step_buffer[0:3] = self.cur_pos
        # self.pos_step_buffer[3] = 0 # 已经是0了
        
        # 使用 LlamaBatch 的高阶接口注入数据
        self.batch.set_embd(feedback_embed, pos=self.pos_step_buffer, seq_id=seq_id)
        
        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Master Step Decode failed at pos {self.cur_pos}")

        # 提取预测结果
        hidden_ptr = self.ctx.get_embeddings()
        

        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(1, 2048))[0]
        
        self.cur_pos += 1
        return hidden
