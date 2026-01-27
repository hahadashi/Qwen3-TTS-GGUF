# Qwen3-TTS Code Predictor ONNX 导出经验总结

在将 Qwen3-TTS 的 `Code Predictor`（工匠模块）导出为 ONNX 的过程中，我们克服了一系列由于架构设计、库版本代差以及导出器特性引起的“坑”。以下是核心经验总结：

## 1. KV Cache 的 JIT 兼容性改造
**挑战**：Transformers 库默认使用 `DynamicCache` 这种 Python 对象来管理键值缓存。然而，`torch.jit.trace`（导出 ONNX 的必备步骤）无法识别非张量（Non-tensor）对象的内部状态更新，导致导出的模型丢失增量推理能力。

**对策**：
- **自定义 JITCache**：手写了一个轻量级的 `JITCache` 类，模拟官方接口。
- **显式拼接**：在 `.update()` 方法中使用 `torch.cat` 进行物理拼接，确保 JIT 能够录制下缓存增长的计算图。

## 2. 隐藏的维度陷阱 (head_dim: 128)
**挑战**：按照常规公式 `hidden_size (1024) / num_heads (16)`，逻辑推导出的 `head_dim` 应该是 64。但实际运行中模型抛出维度不匹配错误。

**对策**：
- **逆推权重**：通过 `inspect_model_dims.py` 直接查看 `k_proj` 等层的维度，确认模型硬编码了 `head_dim = 128`。
- **参数对齐**：在导出脚本中手动指定 `head_dim = 128`，解决了 Dummy Input 与模型真实权重之间的冲突。

## 3. 绕过新版导出器 (Dynamo) 的不稳定性
**挑战**：PyTorch 2.4+ 默认的 Dynamo 导出路径在处理 SDPA Attention 或复杂的 Python 条件分支（如布尔值转换为 Tensor）时极易崩溃，抛出嵌套 Trace 错误。

**对策**：
- **强制降级**：在 `torch.onnx.export` 中显式设置 `dynamo=False`。
- **回归 JIT**：利用经典的基于 TorchScript 的导出路径，虽然被标记为 Deprecated，但对于带有大量“Hack”代码的科研模型，它的兼容性和鲁棒性仍然是目前的最优解。

## 4. 显式参数签名 (Signature) 机制
**挑战**：使用 `*args` 或 `**kwargs` 等模糊方式传递 KV Cache 时，ONNX 导出器无法为 `dynamic_axes`（动态维度）找到对应的参数名，导致 `TreeSpec` 校验失败。

**对策**：
- **静态展开**：在 Wrapper 层显式写出所有 11 个输入参数（1个 Embeds + 10个 KV 层级 Tensor）。
- **精准映射**：这种“笨办法”让 `dynamic_axes` 能够精准地根据名字为每个 Cache 维度打上 `past_seq` 标签，确保了模型在不同序列长度下的通用性。

## 5. 导出流程最佳实践
1. **先诊断后导出**：用脚本打印每一层的 Shape。
2. **权重剥离**：将 LM Heads 这种巨大的线性层导出为 NPY，不仅减小了 ONNX 的体积，还方便在 llama.cpp 或其他后端进行灵活查表。
3. **闭环验证**：导出后必须针对 Prefill（初始推理）和 Incremental（增量推理）两种场景进行数值比对，确信 KV Cache 的每一位都完全对齐。
