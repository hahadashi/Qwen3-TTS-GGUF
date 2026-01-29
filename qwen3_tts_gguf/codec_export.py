import torch
import torch.nn as nn
from . import logger

# 引入本地模型定义
try:
    from .tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
except ImportError:
    Qwen3TTSTokenizerV2Model = None
    logger.warning("Could not import Qwen3TTSTokenizerV2Model from .modeling_tokenizer")

class CodecEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频编码器导出包装类。
    
    【核心修复 V3】
    针对 JIT Trace 报 `unordered_map` 的问题，这通常是因为 Transformers 的子模块
    （MimiEncoder 或 Quantizer）内部返回了 ModelOutput 对象（类似字典）。
    
    本类采取以下策略：
    1. 强制覆盖 config.return_dict = False。
    2. 对于 Quantizer 这种可能不遵守 config 的组件，尝试通过属性访问手动剥离对象。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.config = model.config
        
        # 1. 强力覆写配置
        self.config.return_dict = False
        self.config.use_cache = False
        
        # 2. 提取核心模块 (MimiModel 内部组件)
        self.mimi_model = model.encoder
        self.cnn_encoder = self.mimi_model.encoder 
        self.transformer = self.mimi_model.encoder_transformer
        self.downsample = self.mimi_model.downsample
        self.quantizer = self.mimi_model.quantizer
        
        # 3. 锁定为评估模式，并递归禁用 return_dict/use_cache
        self.eval()
        for m in self.modules():
            if hasattr(m, 'config'):
                m.config.return_dict = False
                m.config.use_cache = False

    def forward(self, input_values):
        """
        Args:
            input_values (torch.FloatTensor): Shape [Batch, Time]
        Returns:
            audio_codes (torch.LongTensor): Shape [Batch, Time, Q]
        """
        # 1. Input: [B, T] -> [B, 1, T]
        x = input_values.unsqueeze(1)
        
        # 2. Step 1: CNN Encoder (SEANet)
        # Returns [B, Hidden, T_inner]
        cnn_out = self.cnn_encoder(x)
        if isinstance(cnn_out, (tuple, list)):
            hidden_states = cnn_out[0]
        else:
            hidden_states = cnn_out

        # 3. Step 2: Transformer Encoder
        # Transformer expects [B, T_inner, Hidden]
        hidden_states = hidden_states.transpose(1, 2)
        
        # return_dict=False is critical here
        trans_out = self.transformer(hidden_states, return_dict=False)
        
        if isinstance(trans_out, (tuple, list)):
            hidden_states = trans_out[0]
        else:
            hidden_states = trans_out
            
        # 4. Step 3: Downsample (if configured)
        # Downsample expects [B, Hidden, T_inner]
        hidden_states = hidden_states.transpose(1, 2)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        # 5. Step 4: Quantizer (RVQ)
        # MimiResidualVectorQuantizer.encode returns [Q, B, T]
        codes = self.quantizer.encode(hidden_states)
        
        # Defense against potential tuple/ModelOutput (though RVQ usually returns Tensor)
        if isinstance(codes, (tuple, list)):
            codes = codes[0]
        elif hasattr(codes, 'audio_codes'):
            codes = codes.audio_codes

        # 6. Qwen3 Slice Logic: Just take the valid number of quantizers
        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:valid_q, :, :] # Slice first dim [Q]
        
        # 7. Transpose to final format: [Q, B, T] -> [B, T, Q]
        codes = codes.transpose(0, 1).transpose(1, 2)
        
        return codes

class CodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器导出包装类 (保持不变)。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        # Input: [B, T, Q] -> [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        wav = self.decoder(codes)
        # Output: [B, 1, T] -> [B, T]
        return wav.squeeze(1)

class StreamingCodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器流式验证包装类。
    内部模拟逐帧/逐片推理，手动管理状态，但对外接口保持为 (audio_codes)。
    用于验证有状态拆解逻辑是否与原版等价。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        """
        验证逻辑：
        1. 内部模拟逐帧调用 Transformer 并传递 KV Cache。
        2. 卷积层暂时保持全序列处理（因为卷积状态的管理逻辑在下一阶段）。
        """
        # 1. 映射为向量 [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        hidden = self.decoder.quantizer.decode(codes)
        
        # 2. 初步卷积处理 [B, Hidden, T] -> [B, T, Hidden]
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        
        # 3. 核心：模拟流式 Transformer 缓存机制
        from transformers.cache_utils import DynamicCache
        B, T, H = hidden.shape
        past_key_values = DynamicCache()
        all_hidden = []
        
        for i in range(T):
            # 逐帧取特征
            frame_hidden = hidden[:, i:i+1, :]
            # position_ids 会在 pre_transformer 内部根据 past_key_values 长度自动计算
            outputs = self.decoder.pre_transformer(
                inputs_embeds=frame_hidden,
                past_key_values=past_key_values,
                use_cache=True
            )
            all_hidden.append(outputs.last_hidden_state)
            past_key_values = outputs.past_key_values
            
        # 合并 Transformer 的输出 [B, T, Hidden]
        hidden = torch.cat(all_hidden, dim=1)
        
        # 4. 后续卷积与上采样流水线
        hidden = hidden.permute(0, 2, 1) # [B, Hidden, T]
        
        # 依次通过 DecoderDecoderBlock
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
            
        # 返回波形 [B, T_audio]
        return wav.squeeze(1).clamp(min=-1, max=1)

class StatefulCodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 终极有状态流式包装器。
    它管理：
    1. Transformer 的 KV Cache。
    2. 卷积层的 4 帧前瞻 Buffer（需要累积够 4 帧有效特征才开始输出）。
    3. 终止时的自动 Padding 和冲刷。
    
    关键设计：latent_buffer 存储的是真实的隐藏层特征，首次调用时为 None，
    需要累积够 4 帧后才能开始正确输出音频。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes, past_key_values=None, latent_buffer=None, is_last_chunk=False):
        """
        Args:
            audio_codes: [B, N, Q] 本次输入的新代码
            past_key_values: Transformer 的缓存对象 (DynamicCache)
            latent_buffer: [B, Hidden, M] 上轮累积的特征 (M <= 4，首次为 None)
            is_last_chunk: 是否是最后一段，用于控制 flush 逻辑
        Returns:
            wav: [B, samples] 当前对应的音频 (可能为空 tensor)
            next_pkv: 更新后的 KV 缓存
            next_latent: 更新后的特征缓存
        """
        B, N, Q = audio_codes.shape
        device = audio_codes.device
        latent_dim = self.config.decoder_config.latent_dim
        
        # 1. 码本解码与预处理
        codes = audio_codes.transpose(1, 2)
        hidden = self.decoder.quantizer.decode(codes)
        # PreConv 处理 [B, Hidden, N] -> [B, N, Hidden]
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        
        # 2. Transformer 有状态推理 (只算 N 个新帧)
        outputs = self.decoder.pre_transformer(
            inputs_embeds=hidden,
            past_key_values=past_key_values,
            use_cache=True
        )
        new_hidden = outputs.last_hidden_state # [B, N, Hidden]
        next_pkv = outputs.past_key_values
        
        # 3. 卷积前瞻对齐逻辑
        # 转换维度为 [B, Hidden, N]
        new_hidden_t = new_hidden.transpose(1, 2)
        
        # 累积特征：将新特征追加到 buffer
        if latent_buffer is None:
            accumulated = new_hidden_t
        else:
            accumulated = torch.cat([latent_buffer, new_hidden_t], dim=-1)
        
        total_frames = accumulated.shape[-1]
        
        # 判断是否有足够的帧来产出音频
        # 需要至少 5 帧才能产出 1 帧音频（4 帧前瞻 + 1 帧当前）
        if is_last_chunk:
            # 最后一段：直接输出所有累积的特征
            # 原始卷积层是 Causal 设计，会自动处理边界 padding，无需手动追加
            conv_input = accumulated
            next_latent = None
        elif total_frames <= 4:
            # 累积不足：不输出音频，继续累积
            return torch.zeros(B, 0, device=device), next_pkv, accumulated
        else:
            # 正常流式：保留最后 4 帧，输出前面的
            next_latent = accumulated[:, :, -4:].clone()
            conv_input = accumulated[:, :, :-4]

        # 4. 进入卷积/上采样链条
        curr = conv_input
        # 4.1 Upsample 层 (DecoderDecoderBlock)
        for blocks in self.decoder.upsample:
            for block in blocks:
                curr = block(curr)
        
        # 4.2 最终解码层
        wav = curr
        for block in self.decoder.decoder:
            wav = block(wav)
            
        # [B, 1, samples] -> [B, samples]
        audio_values = wav.squeeze(1).clamp(min=-1, max=1)
        
        return audio_values, next_pkv, next_latent

class SpeakerEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 说话人编码器导出包装类。
    """
    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()

    def forward(self, mels):
        """
        Args:
            mels (torch.FloatTensor): [Batch, Seq, MelDim(128)]
        Returns:
            spk_emb (torch.FloatTensor): [Batch, 512]
        """
        return self.speaker_encoder(mels)

class MTPStepExportWrapper(nn.Module):
    """
    Qwen3-TTS MTP (Code Predictor) 单步导出包装类。
    因为 MTP 有 15 个不同的 Head 和 Embedding，我们将它们全部包含进来。
    """
    def __init__(self, code_predictor):
        super().__init__()
        self.code_predictor_model = code_predictor.model
        self.lm_heads = code_predictor.lm_head # ModuleList of 15 Linears
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.eval()

    def forward(self, inputs_embeds, step_idx, past_key_values_list=None):
        """
        Args:
            inputs_embeds: [Batch, Seq, Hidden]
            step_idx: int (0 to 14)
            past_key_values_list: Optional list/tuple of past KV
        """
        # 1. Project if needed
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        
        # 2. Model Forward
        # 注意：这里我们简化处理，假设一次跑一步或者一段。
        # 为了 ONNX 导出方便，我们可能需要更具体的 KV Cache 处理。
        # 但 MTP 总共只有 15 步，且序列很短（总共 16 2?），不带 KV Cache 跑 full self-attn 也可以。
        # 考虑到复杂度，这里先实现一个全序列版本的，推理时我们可以拼接序列。
        
        outputs = self.code_predictor_model(
            inputs_embeds=hidden_states,
            past_key_values=None, # 简化：不使用 KV Cache 避免 ONNX 复杂化
            use_cache=False
        )
        
        last_hidden = outputs.last_hidden_state[:, -1:, :]
        
        # 3. Apply the specific head for this step
        # 由于 ONNX 不支持动态索引 ModuleList，我们需要用一种 trick 或者导出多个模型。
        # 或者使用一个逻辑合并所有的 Head。
        # 这里为了导出一个模型，我们写一个 loop (ONNX 会展开它或者我们可以用 index_select)
        
        # 假设 step_idx 是 tensor
        all_logits = []
        for head in self.lm_heads:
            all_logits.append(head(last_hidden))
        
        # [15, B, 1, Vocab] -> [B, 1, Vocab]
        stacked_logits = torch.stack(all_logits, dim=0)
        # 使用 step_idx 选取对应的 logits
        logits = stacked_logits[step_idx] 
        
        return logits
