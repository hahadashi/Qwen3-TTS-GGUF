# Qwen3-TTS GGUF

用 llama.cpp 跑的 Qwen3-TTS，支持 Vulkan GPU 加速。

## 性能表现

在我的 RTX 5050 上的实测数据：

- **RTX 5050 (独显)**: RTF 0.5 (实时率，1秒音频只需0.5秒生成)
- **CPU**: RTF 2.8
- **集显**: RTF 1.6

RTF < 1 就表示生成速度比实时播放还快。

## 当前功能状态

✅ **可用**：
- 用预置音色对文本进行语音合成
- 支持中英日韩等多种语言
- 支持多个预置说话人（Vivian、Serena、UncleFu 等）

❌ **暂不可用**：
- Encoder（用于音频编码和声音克隆）的导出目前还有问题
- 自定义声音克隆功能暂时无法使用

## 快速开始

### 1. 准备原始模型

把官方的 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 模型放到项目根目录。

### 2. 按顺序运行转换脚本

#### 阶段一：准备组件（可选）

这些脚本导出模型的不同组件，但 Encoder 还不能用：

```bash
# 11 - 导出 Encoder（暂不可用，非声音克隆也用不着）
python 11-Export-Encoder.py

# 12 - 导出 Decoder（嘴巴，用于合成声音）
python 12-Export-Decoder.py

# 13 - 导出 Embeddings
python 13-Export-Embeddings.py
```

#### 阶段二：提取大师模型（Talker）

大师模型是 LLM backbone，负责生成语音骨架：

```bash
# 21 - 从原始模型中提取大师权重
python 21-Extract-Master-Weights.py

# 22 - 准备 Tokenizer
python 22-Prepare-Mini-Tokenizer.py

# 23 - 转换为 GGUF 格式
python 23-Convert-Master-GGUF.py
```

#### 阶段三：提取工匠模型（Craftsman/Predictor）

工匠模型负责预测多层 codec codes，为骨架补充血肉：

```bash
# 31 - 提取工匠模型权重
python 31-Extract-Craftsman--Weights.py

# 32 - 准备工匠 Tokenizer
python 32-Prepare-Craftsman-Tokenizer.py

# 33 - 转换为 GGUF 格式
python 33-Convert-Craftsman-GGUF.py
```

### 3. 运行推理

#### 批量推理模式

```bash
python 41-Inference.py
```

这会生成完整的音频文件并保存到 `output/` 目录。

#### 流式推理模式

```bash
python 42-Inference-Streaming.py
```

支持边生成边播放，可以分段发送和播放音频。

## 推理使用示例

```python
from qwen3_tts_gguf import Qwen3TTS

# 初始化引擎
tts = Qwen3TTS()

# 合成语音
audio = tts.synthesize(
    text="你好，我是千问3-TTS",
    speaker_id="vivian",      # 或用数字 ID，如 3065
    language="chinese",       # 支持 chinese/english/japanese/korean 等
    max_steps=400,            # 最大生成步数
    temperature=0.9,          # 大师采样温度
    subtalker_temperature=0.9 # 工匠采样温度
)

# 保存音频
import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### 可用说话人列表

- **vivian** (3065)
- **serena** (3066)
- **uncle_fu** (3010)
- **ryan** (3061)
- **aiden** (2861)
- **ono_anna** (2873)
- **sohee** (2864)
- **eric** (2875)
- **dylan** (2878)

### 支持的语言

- **chinese** (2055) - 中文
- **english** (2050) - 英语
- **japanese** (2058) - 日语
- **korean** (2064) - 韩语
- **german** (2053) - 德语
- **spanish** (2054) - 西班牙语
- **french** (2061) - 法语
- **russian** (2069) - 俄语
- **beijing_dialect** (2074) - 北京话
- **sichuan_dialect** (2062) - 四川话

## 模型架构说明

这个项目把 Qwen3-TTS 拆成了几个独立部分，可以用人的器官来理解：

1. **左耳朵（文本 Tokenizer）**: 把你输入的文字转换成 tokens，让模型能读懂
2. **右耳朵（音频 Encoder）**: 把参考音频编码成音色特征（❌ 暂时还没弄好）
3. **大脑（Talker/大师）**: 1.42B 参数的 LLM backbone（28层，2048隐藏层），负责理解文本和生成语音骨架
4. **双手（Craftsman/工匠）**: 142M 参数的代码预测器（8层），负责生成多层 codec codes，给骨架补充血肉
5. **嘴巴（Decoder）**: ONNX 模型，把 codec codes 解码成最终音频

目前**右耳朵还没修好**，所以只能用预制音色，不能克隆声音。但左耳朵、大脑、双手、嘴巴都配合得很好了！

## 推理引擎

不同的部分用不同的推理引擎：

- **耳朵**: ONNX Runtime
- **大师和工匠**: llama.cpp（大师伪装成 Qwen3-VL，工匠伪装成 Qwen3，才能被 llama.cpp 识别）
- **嘴巴**: ONNX Runtime

llama.cpp 支持 Vulkan GPU 加速，这就是为什么在 RTX 5050 上能达到 RTF 0.5 的原因。

## GPU 加速

llama.cpp 会自动检测并使用可用的 GPU。如果想强制使用特定设备，可以设置环境变量：

```python
# 禁用 Vulkan（强制 CPU）
os.environ["VK_ICD_FILENAMES"] = "none"

# 强制使用集显
os.environ["GGML_VK_VISIBLE_DEVICES"] = "1"

# 禁用 FP16 计算（某些集显有精度问题）
os.environ["GGML_VK_DISABLE_F16"] = "1"
```

## 文件结构

```
qwen3-tts/
├── 01-Run-Official-Inference.py      # 测试官方模型是否工作
├── 11-Export-Encoder.py              # 导出 Encoder（暂不可用）
├── 12-Export-Decoder.py              # 导出 Decoder
├── 13-Export-Embeddings.py           # 导出所有 embedding 表
├── 21-Extract-Master-Weights.py      # 提取大师模型权重
├── 22-Prepare-Mini-Tokenizer.py      # 准备大师 tokenizer
├── 23-Convert-Master-GGUF.py         # 转换大师为 GGUF
├── 31-Extract-Craftsman--Weights.py  # 提取工匠模型权重
├── 32-Prepare-Craftsman-Tokenizer.py # 准备工匠 tokenizer
├── 33-Convert-Craftsman-GGUF.py      # 转换工匠为 GGUF
├── 41-Inference.py                   # 批量推理脚本
├── 42-Inference-Streaming.py         # 流式推理脚本
├── model/                            # 转换后的模型文件
│   ├── qwen3_tts_talker.gguf        # 大师模型
│   ├── qwen3_tts_craftsman.gguf     # 工匠模型
│   ├── qwen3_tts_decoder.onnx       # 解码器
│   └── codec_embedding_*.npy        # Codec embedding 表
└── qwen3_tts_gguf/                  # 核心推理和一些转换代码
```

## 常见问题

**Q: 为什么不用官方的 PyTorch 推理？**

A: 官方实现需要大量显存，而 llama.cpp + GGUF 格式可以用更少的资源运行，并且支持各种硬件加速。

**Q: 流式推理和批量推理有什么区别？**

A: 流式推理可以边生成边播放，降低首字延迟；批量推理生成完整音频后再播放。流式推理更适合实时对话场景。

**Q: 如何调整生成质量？**

A: 调整 `temperature` 参数。越低越保守但可能重复，越高越多样但可能不稳定。推荐 0.7-1.0 之间。

## 相关资源

- [Qwen3-TTS 官方技术报告](./Qwen3-TTS%20Technical%20Report.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
