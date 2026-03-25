"""
资产管理器 - 统一管理模型中的embeddings和特殊token

对应GGUF版本的assets.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AssetsManager:
    """
    资产管理器 - 统一管理模型embeddings和特殊token

    职责:
    1. 提取codec_embeddings (16层)
    2. 提取text_embeddings
    3. 提取特殊token embeddings
    4. 提取投影层 (1.7B模型)
    """

    def __init__(self, model: 'Qwen3TTSModel'):
        """
        初始化资产管理器

        Args:
            model: Qwen3TTSModel实例
        """
        self.model = model
        self.base_model = model.model
        self.config = model.model.config
        self.device = model.model.device

        self._extract_assets()

    def _extract_assets(self):
        """提取所有资产"""
        cfg = self.config.talker_config

        # 1. 特殊token IDs
        self.codec_bos_id = cfg.codec_bos_id
        self.codec_eos_id = cfg.codec_eos_token_id
        self.codec_pad_id = cfg.codec_pad_id
        self.tts_bos_id = getattr(cfg, 'tts_bos_token_id', None)
        self.tts_eos_id = getattr(cfg, 'tts_eos_token_id', None)

        # 1.5 语言ID映射 (用于确定 think/nothink mode)
        self.codec_language_id = getattr(cfg, 'codec_language_id', {}) or {}

        # 2. Talker codec_embedding (vocab=3072) - 用于 THINK tokens 和 audio feedback
        self.talker_codec_embedding = self._get_talker_codec_embedding()

        # 3. Predictor codec_embeddings [16, vocab_size, hidden] (vocab=2048)
        self.codec_embeddings = self._get_codec_embeddings()

        # 4. Text embeddings [vocab_size, hidden] - 使用 text_embedding 而非 codec_embedding
        self.text_embeddings = self._get_text_embeddings()

        # 5. 特殊token embeddings
        self.tts_pad = self._get_tts_pad()
        self.tts_bos_embed = self._get_tts_bos()

        # 6. 投影层 (1.7B)
        self.projection_layer = self._get_projection_layer()

        print(f"AssetsManager initialized:")
        print(f"  - talker_codec_embedding: {self.talker_codec_embedding.weight.shape if self.talker_codec_embedding else None}")
        print(f"  - codec_embeddings: {self.codec_embeddings.shape}")
        print(f"  - text_embeddings: {self.text_embeddings.shape}")
        print(f"  - codec_bos_id: {self.codec_bos_id}")
        print(f"  - codec_eos_id: {self.codec_eos_id}")
        print(f"  - codec_pad_id: {self.codec_pad_id}")
        print(f"  - projection_layer: {self.projection_layer is not None}")

    def _get_talker_codec_embedding(self) -> Optional[nn.Embedding]:
        """
        获取 Talker 的 codec_embedding (vocab=3072)

        用于:
        1. THINK token 嵌入
        2. code_0 的 audio feedback
        3. ICL body 中的 audio pool

        Returns:
            talker_codec_embedding: Embedding 层 [3072, hidden]
        """
        talker = self.base_model.talker
        if hasattr(talker, 'model') and hasattr(talker.model, 'codec_embedding'):
            return talker.model.codec_embedding
        return None

    def _get_codec_embeddings(self) -> torch.Tensor:
        """
        获取16层codec embeddings (用于 Predictor 的 audio feedback)

        CRITICAL: 这些 embeddings 来自 talker.code_predictor.model.codec_embedding
        而不是 tokenizer 的 quantizer codebook！

        GGUF 参考:
        - codec_embedding_0.npy: talker.model.codec_embedding.weight
        - codec_embedding_1~15.npy: talker.code_predictor.model.codec_embedding.{i}.weight

        Returns:
            codec_embeddings: [16, vocab_size, hidden]
        """
        talker = self.base_model.talker

        # 正确来源: talker.code_predictor.model.codec_embedding
        if hasattr(talker, 'code_predictor') and hasattr(talker.code_predictor, 'model'):
            predictor = talker.code_predictor.model
            if hasattr(predictor, 'codec_embedding'):
                # codec_embedding 是 nn.ModuleList，包含 15 个 Embedding 层
                # 索引 0-14 对应 codec_1~15
                embeddings_list = []
                for i in range(15):
                    if hasattr(predictor.codec_embedding[i], 'weight'):
                        weight = predictor.codec_embedding[i].weight
                        embeddings_list.append(weight)
                    else:
                        break

                if len(embeddings_list) == 15:
                    # 添加第 0 层 (talker 的 codec_embedding，vocab=3072)
                    # 但注意：对于 audio feedback，code_0 使用 talker_codec_embedding
                    # codec_embeddings[0] 应该对应 codec_1，所以我们需要:
                    # - embeddings_list[0] = codec_1 embedding
                    # - ...
                    # - embeddings_list[14] = codec_15 embedding

                    # 我们需要 16 层，但只有 15 个
                    # GGUF 方案中 codec_embedding_0 是单独处理的 (talker_codec_embedding)
                    # 所以这里的 codec_embeddings 应该是 codec_1~15 对应的 embedding

                    # 堆叠为 [15, vocab_size, hidden]
                    result = torch.stack(embeddings_list, dim=0)

                    # 为了兼容，添加一个 placeholder 作为索引 0
                    # 实际使用时，code_0 使用 talker_codec_embedding
                    placeholder = torch.zeros(
                        1, result.shape[1], result.shape[2],
                        device=result.device, dtype=result.dtype
                    )
                    result = torch.cat([placeholder, result], dim=0)

                    print(f"  codec_embeddings shape: {result.shape}")
                    print(f"    - codec_0: placeholder (use talker_codec_embedding)")
                    print(f"    - codec_1~15: from code_predictor, vocab={result.shape[1]}")

                    return result

        # 回退: 尝试从 tokenizer 获取 (可能不正确)
        print("Warning: Using tokenizer codebook for codec_embeddings (may not be correct for audio feedback)")

        # 获取speech_tokenizer
        if hasattr(self.base_model.speech_tokenizer, 'model'):
            tokenizer_model = self.base_model.speech_tokenizer.model
        else:
            tokenizer_model = self.base_model.speech_tokenizer

        # 根据tokenizer类型获取embeddings
        if hasattr(tokenizer_model, 'vq') and hasattr(tokenizer_model.vq, 'codebook'):
            # V1 tokenizer (25Hz)
            codebook = tokenizer_model.vq.codebook  # [1024, 16, hidden]
            # 转换为 [16, 1024, hidden]
            return codebook.transpose(0, 1)
        elif hasattr(tokenizer_model, 'encoder') and hasattr(tokenizer_model.encoder, 'quantizer'):
            # V2 tokenizer (12Hz) - Mimi架构
            quantizer = tokenizer_model.encoder.quantizer
            embeddings_list = []

            # 首先从semantic RVQ获取 (1层)
            if hasattr(quantizer, 'semantic_residual_vector_quantizer'):
                semantic_vq = quantizer.semantic_residual_vector_quantizer
                if hasattr(semantic_vq, 'layers'):
                    for layer in semantic_vq.layers:
                        if hasattr(layer, 'codebook') and hasattr(layer.codebook, 'embed'):
                            # embed是 [codebook_size, codebook_dim]
                            weight = layer.codebook.embed  # [2048, 256]
                            # 扩展到hidden_size维度 (2048)
                            if weight.shape[1] < self.config.talker_config.hidden_size:
                                padded = torch.zeros(
                                    weight.shape[0],
                                    self.config.talker_config.hidden_size,
                                    device=weight.device,
                                    dtype=weight.dtype
                                )
                                padded[:, :weight.shape[1]] = weight
                                weight = padded
                            embeddings_list.append(weight)

            # 然后从acoustic RVQ获取 (最多15层，总共16层)
            if hasattr(quantizer, 'acoustic_residual_vector_quantizer'):
                acoustic_vq = quantizer.acoustic_residual_vector_quantizer
                if hasattr(acoustic_vq, 'layers'):
                    for layer in acoustic_vq.layers[:15]:  # 只取前15层
                        if hasattr(layer, 'codebook') and hasattr(layer.codebook, 'embed'):
                            weight = layer.codebook.embed  # [2048, 256]
                            # 扩展到hidden_size
                            if weight.shape[1] < self.config.talker_config.hidden_size:
                                padded = torch.zeros(
                                    weight.shape[0],
                                    self.config.talker_config.hidden_size,
                                    device=weight.device,
                                    dtype=weight.dtype
                                )
                                padded[:, :weight.shape[1]] = weight
                                weight = padded
                            embeddings_list.append(weight)

            # 堆叠为 [16, vocab_size, hidden]
            if len(embeddings_list) >= 16:
                return torch.stack(embeddings_list[:16], dim=0)
            elif len(embeddings_list) > 0:
                # 如果不足16层，复制最后一层
                while len(embeddings_list) < 16:
                    embeddings_list.append(embeddings_list[-1])
                return torch.stack(embeddings_list, dim=0)
            else:
                # 完全没有找到，使用随机初始化
                print("Warning: Could not extract codec embeddings, using random initialization")
                return torch.randn(
                    16,
                    self.config.talker_config.codec_vocab_size,
                    self.config.talker_config.hidden_size,
                    device=self.device
                )

        raise NotImplementedError(f"Unknown tokenizer structure: {type(tokenizer_model)}")

    def _get_text_embeddings(self) -> torch.Tensor:
        """
        获取文本嵌入（经过text_projection投影）

        对于12Hz模型，需要使用text_embedding + text_projection MLP
        而不是直接使用codec_embedding（只有3072个条目）

        Returns:
            text_embeddings: [vocab_size, hidden]
        """
        # 检查是否有text_embedding层
        if not hasattr(self.base_model.talker.model, 'text_embedding'):
            # 回退到codec_embedding（25Hz模型）
            return self.base_model.talker.get_input_embeddings().weight

        # 获取原始text_embedding
        raw_text_embed = self.base_model.talker.model.text_embedding.weight
        print(f"  Raw text_embedding shape: {raw_text_embed.shape}")

        # 检查是否有text_projection MLP
        if hasattr(self.base_model.talker, 'text_projection'):
            proj = self.base_model.talker.text_projection
            if hasattr(proj, 'linear_fc1') and hasattr(proj, 'linear_fc2'):
                # 应用投影: fc2(silu(fc1(x)))
                with torch.no_grad():
                    h = torch.nn.functional.linear(
                        raw_text_embed,
                        proj.linear_fc1.weight,
                        proj.linear_fc1.bias
                    )
                    h = torch.nn.functional.silu(h)
                    projected = torch.nn.functional.linear(
                        h,
                        proj.linear_fc2.weight,
                        proj.linear_fc2.bias
                    )
                print(f"  Projected text_embedding shape: {projected.shape}")
                return projected

        # 没有投影层，直接返回原始embedding
        return raw_text_embed

    def _get_tts_pad(self) -> torch.Tensor:
        """
        获取tts_pad embedding

        GGUF 实现: text_table[151671]
        这是一个学习到的 PAD 向量，而不是零向量

        Returns:
            tts_pad: [hidden]
        """
        # 使用 text_embeddings[151671] (与 GGUF 一致)
        return self.text_embeddings[151671]

    def _get_tts_bos(self) -> torch.Tensor:
        """
        获取tts_bos embedding

        Returns:
            tts_bos: [hidden]
        """
        vocab_size = self.codec_embeddings.shape[1]

        if self.tts_bos_id is not None and self.tts_bos_id < self.text_embeddings.shape[0]:
            return self.text_embeddings[self.tts_bos_id]
        # 如果没有专门的tts_bos或ID超出范围，使用零向量
        return torch.zeros(self.config.talker_config.hidden_size, device=self.device)

    def _get_projection_layer(self) -> Optional[nn.Module]:
        """
        获取投影层 (1.7B: 2048→1024)

        Returns:
            projection_layer: 线性投影层，或None
        """
        talker_hidden = self.config.talker_config.hidden_size

        # 尝试从code_predictor获取hidden_size
        if hasattr(self.base_model.talker, 'code_predictor') and \
           hasattr(self.base_model.talker.code_predictor, 'config'):
            predictor_hidden = getattr(
                self.base_model.talker.code_predictor.config,
                'hidden_size',
                talker_hidden
            )
        else:
            predictor_hidden = talker_hidden

        # 如果大小不同，说明需要投影
        if talker_hidden != predictor_hidden:
            # 尝试从模型中获取投影层
            if hasattr(self.base_model.talker.code_predictor, 'small_to_mtp_projection'):
                return self.base_model.talker.code_predictor.small_to_mtp_projection
            else:
                # 创建一个未初始化的线性层
                # 权重需要从其他位置加载或训练
                return nn.Linear(talker_hidden, predictor_hidden)

        return None

    def get_codec_embedding(self, layer: int, code: int) -> torch.Tensor:
        """
        获取指定层code的embedding

        Args:
            layer: 层索引 (0-15)
            code: code值

        Returns:
            embedding: [hidden]
        """
        return self.codec_embeddings[layer, code, :]  # [hidden]

    @property
    def hidden_size(self) -> int:
        """获取hidden size"""
        return self.text_embeddings.shape[1]

    @property
    def vocab_size(self) -> int:
        """获取codec vocab size"""
        return self.codec_embeddings.shape[1]

    @property
    def num_codec_layers(self) -> int:
        """获取codec层数"""
        return self.codec_embeddings.shape[0]
