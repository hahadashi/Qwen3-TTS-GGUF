# llama.cpp 推理大师模型 - 变更计划

## 📋 分析总结

大师模型**本质上就是 Qwen3-VL 架构**，所以 llama.cpp 的现有代码可以基本复用！

### 🔍 大师模型特征

```json
{
  "architecture": "Qwen3-VL",
  "hidden_size": 2048,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 3072,           // Codec token vocab
  "text_vocab_size": 151936,     // Text token vocab (可能不使用)
  "rope_theta": 1000000,
  "rope_scaling": {
    "interleaved": true,
    "mrope_section": [24, 20, 20],  // Qwen3-VL 的 MRoPE!
    "rope_type": "default"
  }
}
```

### ✅ 好消息

1. **架构已支持**: llama.cpp 已有 `LLM_ARCH_QWEN3VL`
2. **MRoPE 已实现**: `ggml_rope_multi()` 支持多模态 RoPE
3. **Q/K Norm 已支持**: 代码中已有 `attn_q_norm` 和 `attn_k_norm`

---

## 📝 需要的修改

### 方案 A: 最小改动 (推荐)

直接复用 `LLM_ARCH_QWEN3VL`，只需修改转换脚本。

#### 1. 创建大师模型转换器

在 `convert_hf_to_gguf.py` 中添加：

```python
@ModelBase.register("Qwen3TTSTalkerModel")
class Qwen3TTSMasterModel(Qwen3VLTextModel):
    """Qwen3-TTS Master (LLM Backbone) - 基于 Qwen3-VL 架构"""

    model_arch = gguf.MODEL_ARCH.QWEN3VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # 大师模型没有视觉部分，设置 deepstack_layers = 0
        self.gguf_writer.add_num_deepstack_layers(0)

        # 设置 MRoPE sections
        mrope_section = self.hparams.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])
        self.gguf_writer.add_rope_dimension_sections(mrope_section)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # 权重名称已经没有前缀了（提取时已去掉 talker.model.）
        # 直接返回即可
        yield name, data_torch
```

#### 2. 转换大师模型到 GGUF

```bash
python convert_hf_to_gguf.py \
    --model Standalone-Bare-Master \
    --outfile qwen3-tts-master-v2-q8_0.gguf \
    --outtype q8_0
```

#### 3. 使用 llama.cpp 推理

```bash
./llama-cli \
    -m qwen3-tts-master-v2-q8_0.gguf \
    -p "你的输入" \
    -c 2048 \
    -n 512
```

---

### 方案 B: 独立架构 (更清晰)

如果想要完全独立，可以添加新架构 `LLM_ARCH_QWEN3TTS`。

#### 1. 添加架构枚举

修改 `src/llama-arch.cpp`:

```cpp
{ LLM_ARCH_QWEN3TTS, "qwen3tts" },
```

#### 2. 修改 `src/llama-hparams.h`

```cpp
enum llm_arch {
    ...
    LLM_ARCH_QWEN3TTS,
    ...
};
```

#### 3. 修改 `src/models/models.h`

```cpp
extern llm_build_qwen3tts;  // 复用 qwen3vl 的实现
```

#### 4. 修改 `src/llama-model.cpp`

添加参数解析：

```cpp
case LLM_ARCH_QWEN3TTS:
    {
        ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        // 识别模型大小
        switch (hparams.n_layer) {
            case 28: type = LLM_TYPE_1_7B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } break;
```

添加张量加载：

```cpp
case LLM_ARCH_QWEN3TTS:
    {
        // 与 QWEN3VL 完全相同
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head}, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        }
    } break;
```

添加 RoPE 类型：

```cpp
case LLM_ARCH_QWEN3TTS:
    return LLAMA_ROPE_TYPE_IMROPE;
```

添加构建器：

```cpp
case LLM_ARCH_QWEN3TTS:
    {
        llm = std::make_unique<llm_build_qwen3tts>(*this, params);
    } break;
```

#### 5. 复用 Qwen3-VL 的构建器

修改 `src/models/qwen3vl.cpp`:

```cpp
llm_build_qwen3tts::llm_build_qwen3tts(const llama_model & model, const llm_graph_params & params)
    : llm_build_qwen3vl(model, params) {
    // 完全相同，直接复用
}
```

或者添加别名：

```cpp
using llm_build_qwen3tts = llm_build_qwen3vl;
```

#### 6. 修改转换脚本

```python
@ModelBase.register("Qwen3TTSTalkerModel")
class Qwen3TTSMasterModel(Qwen3VLTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3TTS  # 使用新架构

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_num_deepstack_layers(0)
        mrope_section = self.hparams.get("rope_scaling", {}).get("mrope_section", [24, 20, 20])
        self.gguf_writer.add_rope_dimension_sections(mrope_section)
```

---

## 🎯 推荐方案

**推荐使用方案 A (最小改动)**：

### 优点
1. ✅ 改动最小，风险最低
2. ✅ llama.cpp 已完全支持 Qwen3-VL
3. ✅ 大师模型就是 Qwen3-VL 架构
4. ✅ 不需要修改 C++ 代码

### 缺点
1. ⚠️ 模型类型显示为 `qwen3vl` 而不是 `qwen3tts` (但不影响功能)

### 实施步骤

1. ✅ 提取大师模型 (已完成)
2. ⏳ 修改 `convert_hf_to_gguf.py` 添加转换支持
3. ⏳ 转换为 GGUF 格式
4. ⏳ 测试推理

---

## 🔧 额外注意事项

### 1. 词表大小

大师模型有两个 embedding：
- `codec_embedding`: 3072 vocab (用于音频 codec tokens)
- `text_embedding`: 151936 vocab (可能不需要)

转换时需要确定使用哪个作为主词表。

### 2. Codec Head

Codec Head 是一个独立的线性层 (2048 -> 3072)，不在大师模型中。

在 llama.cpp 中，这相当于 `output` 层。需要：
- 要么将 codec_head 权重合并到 GGUF
- 要么在推理时单独处理

### 3. 输入格式

大师模型接受 `inputs_embeds` (预嵌入的向量)，而不是原始的 `input_ids`。

使用 llama.cpp 推理时，需要：
- 要么提供 token IDs 作为输入
- 要么修改推理流程支持预嵌入输入

---

## 📊 对比总结

| 特性 | 大师模型 | Qwen3-VL | 兼容性 |
|------|-------------------|----------|--------|
| 架构 | 基于 Qwen3-VL | Qwen3-VL | ✅ 完全兼容 |
| MRoPE | [24, 20, 20] | [24, 20, 20] | ✅ 完全兼容 |
| Q/K Norm | ✅ | ✅ | ✅ 完全兼容 |
| 层数 | 28 | 28 | ✅ 完全兼容 |
| 隐藏层 | 2048 | 2048 | ✅ 完全兼容 |
| 注意力头 | 16/8 (GQA) | 16/8 (GQA) | ✅ 完全兼容 |
| Vision | ❌ 无 | ✅ 有 | ⚠️ 需设置 deepstack=0 |

---

## ✅ 结论

**大师模型可以直接使用 llama.cpp 的 Qwen3-VL 支持！**

只需要：
1. 修改转换脚本，识别大师模型配置
2. 设置 `deepstack_layers = 0` (无视觉部分)
3. 转换为 GGUF 格式
4. 直接使用 llama.cpp 推理

不需要修改任何 C++ 代码！🎉
