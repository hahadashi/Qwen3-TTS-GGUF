# Qwen3-TTS 快速上手指南

## 5 分钟快速开始

### 步骤 1: 安装依赖

```bash
# 安装核心依赖
pip install torch torchaudio transformers accelerate librosa soundfile sentencepiece einops
```

### 步骤 2: 下载模型

**方式 A: 使用 Hugging Face Hub（推荐）**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

**方式 B: 使用 ModelScope（国内推荐）**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='./models')"
```

### 步骤 3: 运行测试

```bash
cd pytorch_cpu
python test_e2e.py --mode both
```

测试完成后，音频将保存到 `output/e2e_wrapper_weather.wav`

## 基础代码示例

```python
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# 1. 加载模型
model = Qwen3TTSModel.from_pretrained(
    "path/to/Qwen3-TTS-12Hz-1.7B-Base",
    local_files_only=True
)

# 2. 生成语音（使用参考音频进行音色克隆）
wavs, sr = model.generate_voice_clone(
    text="今天天气怎么样",
    ref_audio="reference.wav",  # 参考音频文件
    x_vector_only_mode=True     # 仅使用 speaker embedding 模式
)

# 3. 保存结果
sf.write("output.wav", wavs[0], sr)
print(f"音频已保存到 output.wav，时长 {len(wavs[0])/sr:.2f} 秒")
```

## 使用 Wrapper 引擎（流式生成）

```python
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen3_tts_wrapper import Qwen3TTSWrapperEngine

# 1. 初始化引擎
engine = Qwen3TTSWrapperEngine(
    model_path="path/to/Qwen3-TTS-12Hz-1.7B-Base",
    device="cpu",
    dtype="float32"
)

# 2. 准备参考音频
ref_audio, sr = sf.read("reference.wav")
ref_audio_tensor = torch.from_numpy(ref_audio).float()

# 3. 提取音色特征
speaker_embedding = engine.full_model.model.extract_speaker_embedding(
    ref_audio, 24000
)
ref_audio_tensor = ref_audio_tensor.unsqueeze(0)
reference_codes = engine.codec_encoder.encode(ref_audio_tensor)

# 4. 创建流式会话并设置音色
stream = engine.create_stream()
stream.set_voice(
    speaker_embedding=speaker_embedding,
    reference_codes=reference_codes
)

# 5. 流式合成
def audio_callback(chunk, is_last):
    print(f"收到音频块: {chunk.shape[0]} 样本, 最后={is_last}")

audio = stream.synthesize_stream(
    text="今天天气怎么样",
    audio_callback=audio_callback
)

# 6. 保存结果
sf.write("output_stream.wav", audio.cpu().numpy(), 24000)
```

## 常见问题

**Q: 如何获取参考音频？**

A: 任意一段清晰的中文语音录音即可，建议：
- 时长: 3-10 秒
- 采样率: 24kHz (会自动重采样)
- 格式: WAV 单声道

**Q: CPU 推理速度如何？**

A: 典型速度约为 0.05-0.1x 实时速度（即生成 1 秒音频需要 10-20 秒）

**Q: 如何使用 GPU？**

A: 将 `device="cpu"` 改为 `device="cuda"`，并安装 CUDA 版本的 PyTorch

**Q: 支持哪些语言？**

A: 主要支持中文，也支持英文（Auto 模式会自动检测）

## 下一步

- 阅读完整文档: `README.md`
- 查看示例代码: `examples/basic_usage.py`
- 运行单元测试: `pytest tests/ -v`
