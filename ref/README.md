# Qwen3-TTS PyTorch CPU 推理方案

基于 PyTorch 的 Qwen3-TTS CPU 推理实现，支持流式语音合成。

## 特性

- ✅ 支持 CPU 推理（无需 GPU）
- ✅ 流式音频生成（实时回调）
- ✅ 音色克隆（Speaker Embedding + Reference Codes）
- ✅ 12Hz V2 Tokenizer 支持
- ✅ 与官方原生 API 结果一致

## 系统要求

- Python >= 3.10
- 操作系统：Windows / Linux
- 内存：建议 8GB+
- 磁盘：约 4GB（模型文件）

## 目录结构

```
pytorch_cpu/
├── qwen3_tts_wrapper/          # Wrapper 核心模块
│   ├── __init__.py
│   ├── engine.py               # 主引擎类
│   ├── stream.py               # 流式会话类
│   ├── config.py               # 配置类
│   ├── states/                 # 状态管理
│   │   ├── __init__.py
│   │   └── state_classes.py    # 状态类定义
│   └── wrappers/               # 组件封装
│       ├── __init__.py
│       ├── talker.py           # Talker (codec_0 生成)
│       ├── predictor.py        # Predictor (codec_1~15 生成)
│       ├── decoder.py          # Decoder (音频解码)
│       ├── codec_encoder.py    # Codec Encoder (音频编码)
│       └── speaker_encoder.py  # Speaker Encoder (说话人编码)
├── tests/                      # 单元测试
│   ├── test_*.py
│   └── __init__.py
├── examples/                   # 示例代码
│   ├── basic_usage.py          # 基础用法示例
│   └── __init__.py
├── test_e2e.py                 # 端到端测试脚本
└── README.md                   # 本文档
```

## 依赖安装

### 1. 安装核心依赖

```bash
pip install torch>=2.10 torchaudio>=2.10 torchvision
pip install transformers==4.57.6 accelerate==1.12.0
pip install librosa soundfile sentencepiece einops
```

### 2. 安装可选依赖

```bash
# 音频播放支持（可选）
pip install sounddevice pyaudio pygame

# 测试框架
pip install pytest pytest-cov
```

### 完整 requirements.txt

```
# 核心依赖
torch>=2.10
torchaudio>=2.10
transformers==4.57.6
accelerate==1.12.0
librosa
soundfile
sentencepiece
einops

# 可选依赖
sounddevice    # 音频播放
pyaudio        # 音频播放（Windows）
pygame         # 音频播放
pytest         # 测试框架
pytest-cov     # 测试覆盖率
```

## 模型准备

### 下载模型

模型目录应包含以下文件：

```
Qwen3-TTS-12Hz-1.7B-Base/
├── model.safetensors          # 模型权重（~3.6GB）
├── config.json                # 模型配置
├── generation_config.json     # 生成配置
├── tokenizer_config.json      # 分词器配置
├── vocab.json                 # 词汇表
├── merges.txt                 # BPE merges
├── special_tokens_map.json    # 特殊 token
└── speech_tokenizer/          # 语音分词器（必需！）
    ├── config.json
    ├── decoder.safetensors
    └── encoder.safetensors
```

**注意**：`speech_tokenizer/` 目录是必需的，缺少会导致解码失败。

### 推荐下载方式

```bash
# 使用 Hugging Face Hub
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

或从 ModelScope 下载（国内推荐）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='./models')"
```

## 快速开始

### 1. 基础使用

```python
import torch
from qwen_tts import Qwen3TTSModel
from qwen3_tts_wrapper import Qwen3TTSWrapperEngine

# 初始化引擎
engine = Qwen3TTSWrapperEngine(
    model_path="path/to/Qwen3-TTS-12Hz-1.7B-Base",
    device="cpu",
    dtype="float32"
)

# 创建流式会话
stream = engine.create_stream()

# 设置音色（需要预先准备参考音频）
import soundfile as sf
ref_audio, sr = sf.read("reference.wav")

# 提取 speaker embedding 和 codes
speaker_embedding = engine.full_model.model.extract_speaker_embedding(
    ref_audio.numpy(), 24000
)
ref_audio_tensor = torch.from_numpy(ref_audio).unsqueeze(0).to(engine.device)
reference_codes = engine.codec_encoder.encode(ref_audio_tensor)

stream.set_voice(
    speaker_embedding=speaker_embedding,
    reference_codes=reference_codes
)

# 合成语音
def audio_callback(chunk, is_last):
    print(f"收到音频块: {chunk.shape}, is_last={is_last}")

audio = stream.synthesize_stream(
    text="今天天气怎么样",
    audio_callback=audio_callback
)

# 保存结果
sf.write("output.wav", audio.cpu().numpy(), 24000)
```

### 2. 使用原生 API（推荐用于简单场景）

```python
from qwen_tts import Qwen3TTSModel

# 加载模型
model = Qwen3TTSModel.from_pretrained(
    "path/to/Qwen3-TTS-12Hz-1.7B-Base",
    local_files_only=True
)

# 生成语音
wavs, sr = model.generate_voice_clone(
    text="今天天气怎么样",
    ref_audio="reference.wav",
    x_vector_only_mode=True  # 只使用 speaker embedding
)

# 保存结果
import soundfile as sf
sf.write("output.wav", wavs[0], sr)
```

## 端到端测试

运行完整的端到端测试：

```bash
cd pytorch_cpu

# 测试 Wrapper 引擎
python test_e2e.py --mode wrapper

# 测试原生 API
python test_e2e.py --mode native

# 两者都测试
python test_e2e.py --mode both
```

测试会：
1. 加载 Qwen3-TTS 模型
2. 加载参考音频（Vivian 音色）
3. 合成语音："今天天气怎么样"
4. 保存结果到 `output/e2e_wrapper_weather.wav` 或 `output/e2e_native_weather.wav`

### 测试参数

可以在 `test_e2e.py` 中修改以下参数：

```python
model_path = "path/to/Qwen3-TTS-12Hz-1.7B-Base"  # 模型路径
reference_audio_path = "path/to/reference.wav"    # 参考音频
text = "今天天气怎么样"                            # 合成文本
output_path = "output/weather.wav"                # 输出路径
```

## 单元测试

运行单元测试：

```bash
cd pytorch_cpu
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=qwen3_tts_wrapper --cov-report=html
```

## 常见问题

### 1. AttributeError: 'Qwen3TTSTalkerModel' object has no attribute 'embed_tokens'

**原因**：模型初始化时 `embed_tokens` 属性未被正确设置。

**解决**：确保在 `engine.py` 的 `_load_model()` 后调用：

```python
self.base_model.talker.model.set_input_embeddings(
    self.base_model.talker.model.codec_embedding
)
```

### 2. RuntimeError: Tensors must have same number of dimensions

**原因**：reference_codes 形状不正确，应为 `[T, 16]` 而不是 `[T, 1, 16]`。

**解决**：在 `stream.py` 中进行形状转换：

```python
ref_code_tensor = self.reference_codes.squeeze(1)  # [T, 1, 16] -> [T, 16]
```

### 3. SoX could not be found!

**原因**：librosa 依赖 SoX 但未安装。

**解决**：这个警告可以忽略，不影响功能。如需消除警告，可安装 SoX：

- Windows: 下载 [SoX 二进制包](http://sox.sourceforge.net/)
- Linux: `sudo apt-get install sox` 或 `sudo yum install sox`

### 4. flash-attn is not installed

**原因**：Flash Attention 未安装（CPU 模式不需要）。

**解决**：这个警告可以忽略。如需安装 Flash Attention（需要 GPU）：

```bash
pip install flash-attn --no-build-isolation
```

## 性能优化

### CPU 模式优化

1. **使用 OMP_NUM_THREADS** 控制线程数：

```bash
export OMP_NUM_THREADS=4  # Linux/Mac
set OMP_NUM_THREADS=4     # Windows
```

2. **使用 MKL-DNN**（如果可用）：

```bash
pip install intel-openmp
```

3. **调整批次大小**：对于长文本，考虑分段处理。

### 内存优化

```python
# 使用低内存模式
engine = Qwen3TTSWrapperEngine(
    model_path="path/to/model",
    low_cpu_mem_usage=True  # 启用低内存模式
)
```

## 架构说明

### 组件关系

```
Qwen3TTSWrapperEngine (引擎)
    ├── TalkerWrapper (codec_0 生成器)
    │   └── Qwen3TTSTalkerModel (28层 Transformer)
    ├── PredictorWrapper (codec_1~15 生成器)
    │   └── Qwen3TTSTalkerCodePredictorModel (5层 Transformer)
    ├── DecoderWrapper (音频解码器)
    │   └── Qwen3TTSTokenizerDecoder (8层 + 上采样)
    ├── CodecEncoderWrapper (音频编码器)
    │   └── Qwen3TTSTokenizerEncoder (RVQ 编码)
    └── SpeakerEncoderWrapper (说话人编码器)
        └── Qwen3TTSSpeakerEncoder (Mel -> Embedding)
```

### 数据流

```
文本 → Tokenizer → Talker → codec_0
                              ↓
                           Predictor → codec_1~15
                              ↓
                           Decoder → 音频波形
```

## 配置说明

### TTSConfig 参数

```python
class TTSConfig:
    temperature: float = 0.7        # 采样温度
    top_k: int = 50                 # Top-K 采样
    top_p: float = 0.9              # Top-P (nucleus) 采样
    max_steps: int = 400            # 最大生成步数
    sub_temperature: float = 0.5    # Sub-talker 温度
    seed: Optional[int] = None      # 随机种子
```

## 参考资源

- [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 论文](https://arxiv.org/abs/2411.13136)
- [ModelScope 模型页](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

## 许可证

本项目基于 Apache 2.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0 (2025-03-21)

- ✅ 初始版本发布
- ✅ 支持 CPU 推理
- ✅ 流式音频生成
- ✅ 音色克隆
- ✅ 端到端测试通过
