# Qwen3-TTS Talker LLM GGUF 导出经验总结

## 1. 目标
将 Qwen3-TTS 的 Talker LLM 组件（基于 Qwen2.5/Qwen3 架构）导出为 GGUF 格式，并在 `llama.cpp` 环境下运行。

## 2. 核心挑战与解决方案

### 2.1 QK-Norm (Query/Key Normalization) 支持
*   **问题**：Qwen3-TTS 在 Attention 层使用了 QK-Norm。最初尝试将架构伪装为 `qwen3` 时，转换脚本 `convert_hf_to_gguf.py` 报错，提示缺少 `ATTN_K_NORM` 张量映射。
*   **分析**：通过查阅 `llama.cpp` 源码（`src/models/qwen3vl.cpp` 和 `src/llama-arch.cpp`），发现 `LLM_ARCH_QWEN3VL` (qwen3vl) 架构已原生支持 QK-Norm。
*   **改进**：
    *   在 `20-Prepare-Talker-HF.py` 中将架构伪装从 `Qwen3ForCausalLM` 改为 `Qwen3VLForConditionalGeneration`。
    *   修改 `convert_hf_to_gguf.py` 的 `modify_tensors` 方法，显式添加对 `.self_attn.q_norm.weight` 和 `.self_attn.k_norm.weight` 的映射支持。

### 2.2 mROPE (Multi-axis Rotary Positional Embedding)
*   **知识点**：Qwen3 系列引入了 mROPE 处理多维输入（如时间、高度、宽度）。Qwen3-TTS 虽然是音频模型，但其配置中仍包含 `mrope_section: [24, 20, 20]`。
*   **解决方案**：
    *   使用 `qwen3vl` 架构会自动触发转换脚本中的 `mrope_section` 处理逻辑。
    *   通过 `GGUFReader` 验证，最终生成的元数据 `qwen3vl.rope.dimension_sections` 为 `[24, 20, 20, 0]`，满足 `llama.cpp` 的要求。

### 2.3 词表大小与权重补齐
*   **问题**：原始模型的 `lm_head` 形状与代码预期的 `VOCAB_SIZE (151936)` 往往不完全吻合。
*   **解决方案**：在导出 `safetensors` 时，手动对 `lm_head.weight` 进行补零（Padding），确保其形状为 `[151936, hidden_size]`，保证了 GGUF 文件的兼容性。

### 2.4 配置覆盖失效 (AutoConfig 干扰)
*   **现象**：有时即使修改了本地的 `config.json`，转换脚本依然会按原始架构（如 Mistral 或 Llama）进行逻辑处理。
*   **原因**：`convert_hf_to_gguf.py` 内部使用 `AutoConfig.from_pretrained` 读取配置，它可能会识别出原始架构并忽略我们的手动覆盖。
*   **解决方案**：在脚本的 `load_hparams` 方法中增加逻辑，强制直接读取本地 `config.json` 字典，跳过 `AutoConfig` 的自动推断。

## 3. 验证结果
经测试，生成的 `Qwen3-LLM-1.7B-F16.gguf` 拥有正确的元数据：
- `general.architecture`: `qwen3vl`
- `qwen3vl.n_deepstack_layers`: `3` (根据配置中的 deepstack indexes 算出)
- `qwen3vl.rope.dimension_sections`: `[24, 20, 20, 0]`

这些配置确保了模型在具备相应后端的 `llama.cpp` 中能被正确识别并构建计算图。
