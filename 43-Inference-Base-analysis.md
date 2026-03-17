# 43-Inference-Base.py 实现细节与底层调用

## 1. 文件概述

`43-Inference-Base.py` 是一个 TTS 语音克隆脚本，使用 Qwen3-TTS 基座模型进行流式语音合成。

### 主要功能
- 从 JSON 文件加载音色（无损克隆）
- 使用流式模式进行推理（边推边播）
- 保存结果为 WAV 和 JSON 格式

---

## 2. 核心调用流程

```
43-Inference-Base.py
└── TTSEngine(model_dir="model-base", onnx_provider="CUDA")
    ├── create_stream()
    │   └── TTSStream()
    │       ├── TalkerPredictor  (大师模型，生成第 0 层码本)
    │       ├── Predictor        (工匠模型，预测 Q1-Q15 码本)
    │       └── DecoderProxy     (解码器代理，多进程)
    ├── stream.set_voice(REF_JSON)
    │   └── _set_voice_from_json()
    │       └── TTSResult.from_json()
    └── stream.clone(text, language, config)
        └── _run_engine_loop()
            ├── Talker.prefill()      → Prompt 注入
            ├── Talker.decode_step()  → 融合反馈
            ├── Predictor.predict_frame() → 16 层码本生成
            └── Decoder.decode()      → 波形合成
```

---

## 3. 底层组件详解

### 3.1 TTSEngine (`qwen3_tts_gguf/inference/engine.py`)

| 组件 | 作用 |
|------|------|
| `talker_model` | LlamaModel，加载 `qwen3_tts_talker.q5_k.gguf` |
| `predictor_model` | LlamaModel，加载 `qwen3_tts_predictor.q8_0.gguf` |
| `decoder` | DecoderProxy，多进程解码器 |
| `codec_encoder` | CodecEncoder，音频→声学码 |
| `speaker_encoder` | SpeakerEncoder，音频→说话人嵌入 |

**关键方法：**
- `create_stream()` → 工厂方法创建 `TTSStream`
- `decode(codes, **kwargs)` → 委托给 `DecoderProxy.decode()`

**初始化流程：**
```python
# 1. 资产加载 (Tokenizer, Assets)
self.assets = AssetsManager(str(self.model_dir))
self.tokenizer = Tokenizer.from_file(str(self.paths['tokenizer']))

# 2. 音频及说话人编码器 (CPU 轻量型)
self.codec_encoder = CodecEncoder(...)
self.speaker_encoder = SpeakerEncoder(...)

# 3. 异步拉起解码器进程 (并行点 1)
self.decoder = DecoderProxy(...)

# 4. 模型引擎初始化 (并行点 2)
self._init_llama_engines()  # Talker & Predictor GGUF 模型
```

---

### 3.2 TTSStream (`qwen3_tts_gguf/inference/stream.py`)

**核心属性：**
```python
self.talker = TalkerPredictor(...)   # 大师推理
self.predictor = Predictor(...)       # 工匠推理
self.prompt_builder = PromptBuilder(...)  # Prompt 组装
self.voice = TTSResult                # 音色锚点
self.decoder = DecoderProxy           # 解码器代理
```

**`clone()` 方法流程：**
1. 检查 `voice` 是否已设定（音色锚点）
2. 构建 Clone Prompt：`PromptBuilder.build_clone_prompt()`
3. 运行推理循环：`_run_engine_loop()`
4. 后处理返回：`_post_process()` → `TTSResult`

---

### 3.3 推理循环 (`_run_engine_loop`)

```python
def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig):
    streaming = cfg.streaming
    chunk_size = self.engine.chunk_size
    all_codes = []
    chunk_buffer = []
    current_task_id = f"task_{self.task_counter}"

    for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
        all_codes.append(step_codes)
        chunk_buffer.append(step_codes)

        if not streaming or len(chunk_buffer) < chunk_size:
            continue

        # 解码 chunk (流式输出)
        state = self.voice.final_state if (...) else None
        self.decoder.decode(
            np.array(chunk_buffer),
            task_id=current_task_id,
            is_final=False,
            stream=streaming,
            state=state
        )
        chunk_buffer = []

    # 最后一个 chunk
    decode_result = self.decoder.decode(
        np.array(chunk_buffer),
        task_id=current_task_id,
        is_final=True,
        stream=streaming,
        state=state
    )
```

---

### 3.4 TalkerPredictor (`inference/talker.py`)

**两阶段推理：**

#### 1. Prefill: 注入初始 Prompt
```python
def prefill(self, pdata, seq_id=0):
    prompt_embeds = pdata.embd  # 文本+音色嵌入
    n_p = prompt_embeds.shape[1]

    # 构造 Qwen3 专用的位置编码 (3 层 Pos + 1 层 Zero)
    pos_base = np.arange(self.cur_pos, self.cur_pos + n_p, dtype=np.int32)
    pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_p, dtype=np.int32)])

    # 注入数据
    self.batch.set_embd(prompt_embeds[0], pos=pos_arr, seq_id=seq_id)
    llama_status = self.ctx.decode(self.batch)

    # 获取最后一个位置的隐层输出
    hidden = np.ctypeslib.as_array(hidden_ptr, shape=(n_p, hidden_dim))[-1].copy()
    self.cur_pos += n_p
    return hidden
```

#### 2. Decode Step: 融合音频反馈
```python
def decode_step(self, audio_embed, seq_id=0):
    # [音频特征 + 文本特征] 融合
    if self.step_idx < len(self.trailing_text_pool):
        text_vec = self.trailing_text_pool[self.step_idx]
    else:
        text_vec = self.assets.tts_pad  # Pad 填充

    fused_embed = audio_embed + text_vec

    # 构造单步位置编码 (3 层 Pos + 1 层 Zero)
    self.pos_step_buffer[0:3] = self.cur_pos
    self.batch.set_embd(fused_embed, pos=self.pos_step_buffer, seq_id=seq_id)

    self.ctx.decode(self.batch)
    self.cur_pos += 1
```

---

### 3.5 Predictor (`inference/predictor.py`)

**阶梯式生成 16 层码本：**

```python
def predict_frame(self, master_hidden, code_0, sampler):
    # 1. 维度投影 (1.7B 模型需要 2048→1024)
    if proj is not None:
        m_h_1024 = master_hidden @ proj["weight"].T + proj["bias"]
    else:
        m_h_1024 = master_hidden

    step_codes = [code_0]
    step_embeds_raw = [get_codec_embedding(0, code_0)]

    # 2. Prefill 工匠 (Master Hidden + Code_0 Embedding)
    c_in = np.stack([m_h_1024, get_codec_embedding_1024(0, code_0)], axis=0)
    self.batch.set_embd(c_in, pos=0, seq_id=0)
    self.ctx.decode(self.batch)

    # 3. 阶梯式生成 Q1→Q15
    for cs in range(1, 16):
        # 利用原生采样器的 range limit 功能
        start_offset = (cs-1) * 2048
        end_offset = cs * 2048

        token_id = sampler.sample(self.ctx, limit_start=start_offset, limit_end=end_offset)
        c = token_id - start_offset  # 转换为相对 code

        step_codes.append(c)
        step_embeds_raw.append(get_codec_embedding(cs, c))

        if cs < 15:
            emb_1024 = get_codec_embedding_1024(cs, c)
            self.batch.set_embd(emb_1024, pos=cs + 1, seq_id=0)
            self.ctx.decode(self.batch)

    return step_codes, step_embeds_raw
```

---

### 3.6 DecoderProxy (`inference/proxy.py`)

**多进程架构：**
```
主进程 (TTSEngine)
    │
    ├─ Queue → DecoderWorker (子进程)
    │            └─ StatefulDecoder (ONNX Runtime)
    │
    └─ Queue → SpeakerWorker (子进程)
                 └─ pyaudio 播放
```

**流式解码逻辑：**
```python
def decode(self, input, task_id, is_final, stream, state):
    # 预处理
    if isinstance(input, TTSResult):
        codes = input.codes
        state = state or input.final_state
    else:
        codes = input

    # 初始化状态
    self.results[task_id] = []
    self.events[task_id] = threading.Event()

    # 发送请求到 Worker
    req = DecodeRequest(task_id, msg_type, codes, is_final, state)
    self.codes_q.put(req)

    # 流式包立即返回
    if stream and not is_final:
        return np.array([], dtype=np.float32)

    # 阻塞等待完成
    self.events[task_id].wait(timeout=30.0)

    # 从消息列表提取音频碎片
    responses = self.results.get(task_id, [])
    result = DecodeResult(responses=responses)

    # 清理资源
    del self.results[task_id]
    del self.events[task_id]

    return result
```

---

### 3.7 StatefulDecoder (`inference/decoder.py`)

**ONNX 推理核心：**

```python
def _decode(self, audio_codes, state, is_final):
    # 输入规范化 [N, 16] → [1, N, 16]
    if audio_codes.ndim == 2:
        audio_codes = audio_codes[np.newaxis, ...]

    # 构建 ONNX 输入 feed dict
    feed = {
        "audio_codes": audio_codes.astype(np.int64),
        "is_last": np.array([1.0 if is_final else 0.0], dtype=self.dtype),
        "pre_conv_history": state.pre_conv_history,
        "latent_buffer": state.latent_buffer,
        "conv_history": state.conv_history,
    }

    # KV Cache 解包 (k0, v0, k1, v1 ...)
    for i in range(NUM_LAYERS):
        feed[f"past_key_{i}"] = state.kv_cache[2*i]
        feed[f"past_value_{i}"] = state.kv_cache[2*i + 1]

    # 执行 ONNX 推理
    outputs = self.sess.run(self.output_names, feed)

    # 提取音频
    final_wav = outputs[0]        # [1, num_samples]
    valid_samples = int(outputs[1][0])

    if is_final:
        audio = final_wav[0]  # 全量提取
    else:
        audio = final_wav[0, :valid_samples]  # 取有效部分

    # 构建新状态
    new_state = self._build_state_from_outputs(outputs)
    new_state.skip_samples = 4 * 1920 if is_final else skip_counter

    return audio.astype(np.float32), new_state
```

**状态结构 (`DecoderState`):**
```python
class DecoderState:
    kv_cache: List[np.ndarray]      # 8 层 KV Cache (k0,v0,k1,v1...)
                                    # 每层：[1, 16, 72, 64] float16
    pre_conv_history: np.ndarray    # 前卷积历史 [1, 512, 4] float16
    latent_buffer: np.ndarray       # 潜在缓冲区 [1, 1024, 4] float16
    conv_history: np.ndarray        # 卷积历史 [1, 1024, 4] float16
    skip_samples: int               # 跳过采样点数 (流式静音消除)
    latent_audio: np.ndarray        # 尾部残留音频
```

---

### 3.8 DecoderWorker (`inference/workers/decoder.py`)

**子进程工作函数：**

```python
def decoder_worker_proc(codes_queue, pcm_queue, decoder_onnx_path, ...):
    decoder = StatefulDecoder(decoder_onnx_path, ...)
    sessions = {}  # {task_id: DecoderState}

    while True:
        req = codes_queue.get()
        if req is None: break  # 毒丸

        if req.msg_type in ["DECODE", "DECODE_CHUNK"]:
            handle_decode_task(req, decoder, sessions, pcm_queue)

def handle_decode_task(req, decoder, sessions, response_queue):
    codes_all = np.array(req.codes, dtype=np.int64).reshape(-1, 16)

    # 获取或初始化会话
    session = sessions.get(req.task_id, DecoderSession())
    current_state = session.state or req.state

    is_task_final = req.is_final or (req.msg_type == "DECODE")

    # 执行解码
    audio, new_state = decoder.decode(codes_all, state=current_state, is_final=is_task_final)

    # 回传音频
    response_queue.put(DecoderResponse(
        msg_type="AUDIO",
        task_id=req.task_id,
        index=session.index,
        audio=audio,
        compute_time=dt
    ))

    # 更新会话状态
    session.state = new_state
    session.index += 1
    sessions[req.task_id] = session

    # 结束信号
    if is_task_final:
        response_queue.put(DecoderResponse(
            msg_type="FINISH",
            task_id=req.task_id,
            state=new_state
        ))
        del sessions[req.task_id]
```

---

## 4. 数据流总结

```
文本 + 音色 JSON
    ↓
PromptBuilder → 构建 Prompt (文本 Token + 音色 Embedding)
    ↓
Talker.prefill() → 注入初始上下文 (KV Cache 填充)
    ↓
┌───────────────────────────────────────┐
│ 自回归循环 (max_steps=400)            │
│  1. Talker 采样 Code_0                │
│     - LlamaSampler.sample()           │
│     - 限制范围 [0, 2048)              │
│  2. Predictor 生成 Code_1~Code_15     │
│     - 阶梯式采样，每层限制不同范围    │
│     - (cs-1)*2048 ~ cs*2048          │
│  3. 16 层 Embedding 叠加 → 反馈 Talker │
│     - summed_vec = sum(step_embeds)   │
│  4. yield step_codes (每步 1 帧)       │
└───────────────────────────────────────┘
    ↓
chunk_buffer 累积 (chunk_size=12)
    ↓
Decoder.decode() → DecoderProxy
    ↓
    ├─ codes_q.put(DecodeRequest) → DecoderWorker
    │                                 ↓
    │                           StatefulDecoder
    │                                 ↓
    │                           ONNX Runtime (CUDA/DML)
    │                                 ↓
    │                           音频 PCM → pcm_queue
    │
    └─ 主线程等待 events[task_id].wait()
         ↓
       DecodeResult(audio, final_state, chunk_compute_times)
    ↓
TTSResult(audio, codes, stats, final_state)
    ↓
保存 WAV / JSON
```

---

## 5. 关键配置参数

### TTSConfig
```python
config = TTSConfig(
    max_steps=400,          # 最大生成步数
    temperature=0.6,        # Talker 采样温度
    sub_temperature=0.6,    # Predictor 采样温度
    seed=42,                # Talker 随机种子
    sub_seed=45,            # Predictor 随机种子
    streaming=True,         # 流式输出
)
```

### 模型常量
```python
# Decoder 常量 (decoder.py)
NUM_LAYERS = 8
NUM_HEADS = 16
HEAD_DIM = 64
SAMPLES_PER_FRAME = 1920
KV_CACHE_WINDOW = 72

# 协议 Token (constants.py)
PROTOCOL = {
    "EOS": 2048,   # 结束符
    "BOS": 2049,   # 开始符
    "PAD": 2050,   # 填充符
}
```

---

## 6. 文件路径结构

```
model-base/
├── qwen3_tts_talker.q5_k.gguf       # 大师模型 (Q5_K 量化)
├── qwen3_tts_predictor.q8_0.gguf    # 工匠模型 (Q8_0 量化)
├── qwen3_tts_decoder.fp16.onnx      # 解码器 (FP16 ONNX)
├── qwen3_tts_codec_encoder.fp16.onnx # 编码器 (可选)
├── qwen3_tts_speaker_encoder.fp16.onnx # 说话人编码器 (可选)
└── tokenizer.json                   # 分词器
```

---

## 7. 原生模型与拆分子模型的关联

### 7.1 原生 Qwen3-TTS 模型架构

原生 Qwen3-TTS 是一个端到端的语音合成模型，其内部结构如下：

```
Qwen3TTSForConditionalGeneration
├── model
│   ├── talker                    # 大师模块 (LLM Backbone)
│   │   ├── text_embedding        # 文本嵌入层
│   │   ├── codec_embedding       # 音频码本嵌入层 (Code 0)
│   │   ├── layers                # 28 层 Transformer
│   │   └── codec_head            # 码本预测头 (输出 Code_0)
│   │
│   ├── code_predictor            # 工匠模块
│   │   ├── small_to_mtp_projection  # 维度投影 (1.7B 专用)
│   │   ├── model
│   │   │   ├── codec_embedding   # 音频码本嵌入层 (Code 1-15)
│   │   │   ├── layers            # 8 层 Transformer
│   │   │   └── norm              # 层归一化
│   │   └── lm_head               # 多段 LM 头 (Code 1-15 输出)
│   │
│   ├── speaker_encoder           # 说话人编码器
│   │   └── ...                   # 从 Mel 提取 Speaker Embedding
│   │
│   └── decoder                   # 解码器 (Stateful Codec)
│       ├── pre_conv_net          # 前卷积网络
│       ├── transformer_layers    # 8 层因果 Transformer
│       └── conv_out              # 输出卷积层
│
└── speech_tokenizer              # 语音分词器 (Encoder + Decoder)
```

---

### 7.2 拆分子模型对照表

| 子模型 | 来源 (原生模型路径) | 功能 | 输入 | 输出 |
|--------|---------------------|------|------|------|
| **Talker (GGUF)** | `talker.model.*` + `talker.codec_head.weight` | 文本理解 + 语音骨架生成 (Code_0) | 文本 Token + 音色 Embedding | Code_0 + Hidden State |
| **Predictor (GGUF)** | `talker.code_predictor.*` | 细节补充 (Code_1~Code_15) | Master Hidden + Code_0 | Code_1~Code_15 |
| **Decoder (ONNX)** | `decoder` | 波形合成 | Code_0~Code_15 + KV Cache | 音频 PCM |
| **Codec Encoder (ONNX)** | `speech_tokenizer.encoder` | 音频→声学码 | 音频波形 (24kHz) | Code_0~Code_15 |
| **Speaker Encoder (ONNX)** | `speaker_encoder` | 音频→说话人特征 | Mel 谱图 | Speaker Embedding |
| **Embeddings** | `talker.model.text_embedding`<br>`talker.model.codec_embedding`<br>`talker.code_predictor.model.codec_embedding` | 静态查找表 | Token ID | Embedding 向量 |

---

### 7.3 详细拆分说明

#### 7.3.1 Talker (大师) - 28 层 LLM Backbone

**原生路径:** `talker.model.*`

**提取的权重:**
```python
# 从 model.safetensors 中提取
"talker.model.text_embedding.weight"  → 跳过 (有独立投影)
"talker.model.codec_embedding.weight"  → "embed_tokens.weight"
"talker.model.layers.*"                → "layers.*"
"talker.model.norm.weight"             → "norm.weight"
"talker.codec_head.weight"             → "lm_head.weight"
```

**架构伪装:**
- 原始架构：`Qwen3TTSForConditionalGeneration` 的子模块
- GGUF 架构：`Qwen3VLForConditionalGeneration` (支持 QK-Norm 和 mROPE)

**关键配置修改:**
```python
# 21-Extract-Talker-Weights.py
talker_config['vocab_size'] = 3072  # codec_vocab_size (纯音频词表)
talker_config['architectures'] = ['Qwen3VLForConditionalGeneration']
talker_config['model_type'] = 'qwen3_vl'
```

**功能:**
1. 接收文本 Token 和音色 Embedding 的融合 Prompt
2. 自回归生成第 0 层码本 (Code_0)
3. 输出 Hidden State 给 Predictor

---

#### 7.3.2 Predictor (工匠) - 8 层小模型

**原生路径:** `talker.code_predictor.*`

**提取的权重:**
```python
# 31-Extract-Predictor-Weights.py
"talker.code_predictor.model.codec_embedding.{0-15}.weight"  → 拼接为 "embed_tokens.weight"
"talker.code_predictor.model.layers.{0-7}"                   → "layers.*"
"talker.code_predictor.model.norm.weight"                    → "norm.weight"
"talker.code_predictor.lm_head.{0-15}.weight"                → 拼接为 "lm_head.weight"
```

**1.7B 专用投影:**
```python
# 投影层权重 (1024 → 2048)
"talker.code_predictor.small_to_mtp_projection.weight"  → 保存为 proj_weight.npy
"talker.code_predictor.small_to_mtp_projection.bias"    → 保存为 proj_bias.npy
```

**架构配置:**
```python
# 自动根据权重探测
config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 1024,          # 探测自 embed_tokens
    "num_hidden_layers": 8,       # 探测自 layers
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 32768,          # 16 层 × 2048
}
```

**功能:**
1. 接收 Talker 的 Hidden State (2048 维)
2. 投影到 1024 维 (1.7B 模式)
3. 接收 Code_0 Embedding
4. 自回归生成 Code_1~Code_15

---

#### 7.3.3 Decoder (解码器) - 8 层因果 ConvNet

**原生路径:** `decoder`

**导出脚本:** `13-Export-Decoder.py`

**输入/输出签名:**
```python
# 输入
audio_codes:          [B, N, 16] int64     # 音频码本
pre_conv_history:     [B, 512, H] float16  # 前卷积历史
latent_buffer:        [B, 1024, H] float16 # 潜在缓冲
conv_history:         [B, 1024, H] float16 # 卷积历史
is_last:              [1] float32          # 是否结束
past_key_{0-7}:       [B, 16, S, 64] float16
past_value_{0-7}:     [B, 16, S, 64] float16

# 输出
final_wav:                   [B, samples] float32   # 完整波形
valid_samples:               [1] int32              # 有效样本数
next_pre_conv_history:       [B, 512, H'] float16
next_latent_buffer:          [B, 1024, H'] float16
next_conv_history:           [B, 1024, H'] float16
next_key_{0-7}:              [B, 16, S', 64] float16
next_value_{0-7}:            [B, 16, S', 64] float16
```

**导出优化:**
```python
# 强制 eager 模式，避免 DML 不稳定
model.config.decoder_config._attn_implementation = "eager"
model.config.decoder_config.head_dim = 64
```

**功能:**
1. 接收 16 层码本作为输入
2. 通过 8 层因果 Transformer 生成音频特征
3. 通过卷积层上采样为波形 (1920 样本/帧)
4. 维护 KV Cache 支持流式推理

---

#### 7.3.4 Codec Encoder (编码器)

**原生路径:** `speech_tokenizer.encoder`

**导出脚本:** `11-Export-Codec-Encoder.py`

**输入/输出:**
```python
# 输入
input_values: [B, T] float32  # 24kHz 音频波形

# 输出
audio_codes: [B, N, 16] int64  # 16 层码本
```

**功能:**
1. 从参考音频提取声学码本
2. 用于音色克隆场景

---

#### 7.3.5 Speaker Encoder (说话人编码器)

**原生路径:** `speaker_encoder`

**导出脚本:** `12-Export-Speaker-Encoder.py`

**输入/输出:**
```python
# 输入
mels: [B, T, 128] float32  # Mel 谱图

# 输出
spk_emb: [B, D] float32  # 说话人嵌入
```

**功能:**
1. 从参考音频提取说话人特征
2. 用于音色克隆和音色设计

---

#### 7.3.6 Embeddings (静态查找表)

**导出脚本:** `14-Export-Embeddings.py`

**导出的资产:**
```
embeddings/
├── text_embedding_projected.npy   # 文本嵌入 (已投影到 2048)
├── codec_embedding_0.npy          # Code_0 嵌入表 [3072, 2048]
├── codec_embedding_1.npy          # Code_1 嵌入表 [2048, 1024]
├── codec_embedding_2.npy          # Code_2 嵌入表 [2048, 1024]
├── ...
├── codec_embedding_15.npy         # Code_15 嵌入表 [2048, 1024]
├── proj_weight.npy                # 投影层权重 [2048, 1024] (1.7B)
└── proj_bias.npy                  # 投影层偏置 [1024] (1.7B)
```

**投影计算 (1.7B):**
```python
# 文本嵌入投影 (fc1 -> silu -> fc2)
h = silu(text_embed @ w1.T + b1)
projected = h @ w2.T + b2

# 码本嵌入预投影 (Predictor 用)
proj_emb = raw_emb @ proj_w.T + proj_b
```

---

### 7.4 原生模型 vs 拆分模型的推理对比

#### 原生推理流程 (PyTorch)
```python
# 原生 Qwen3TTSForConditionalGeneration
outputs = model(
    input_ids=text_ids,
    ref_audio=ref_audio,      # 参考音频
    ref_text=ref_text,        # 参考文本
    streaming=True
)
audio = outputs.audio
```

#### 拆分推理流程 (GGUF + ONNX)
```python
# 1. 特征提取 (可选，用于克隆)
codes = codec_encoder(ref_audio)
spk_emb = speaker_encoder(mels)

# 2. 初始化 Stream
stream.set_voice(ref_json)  # 包含 codes + spk_emb + final_state

# 3. 推理循环
for step in range(max_steps):
    # 3.1 Talker 生成 Code_0
    code_0 = talker_sampler.sample(talker_ctx)

    # 3.2 Predictor 生成 Code_1~15
    step_codes, step_embeds = predictor.predict_frame(
        master_hidden, code_0, sampler
    )

    # 3.3 反馈给 Talker
    summed_embed = sum(step_embeds)
    master_hidden = talker.decode_step(summed_embed)

    # 3.4 累积并解码
    chunk_buffer.append(step_codes)
    if len(chunk_buffer) >= chunk_size:
        audio = decoder.decode(chunk_buffer, state)

# 4. 保存结果
result.save("output.wav")
```

---

### 7.5 拆分优势

| 方面 | 原生模型 | 拆分模型 |
|------|----------|----------|
| **显存占用** | 完整加载 ~4GB+ | 按需加载 ~1.8GB |
| **推理后端** | PyTorch + CUDA | llama.cpp + ONNX Runtime |
| **加速选项** | CUDA | CUDA / DML / Vulkan / CPU |
| **量化支持** | FP16 | Q5_K (Talker) / Q8_0 (Predictor) / FP16 (Decoder) |
| **流式优化** | 有限 | 首包延迟 ~300ms |
| **跨平台** | CUDA 依赖 | 支持 AMD/Intel/ARM |
| **模块化** | 黑盒 | 可独立替换/升级组件 |

---

*生成时间：2026-03-17*

---

## 8. 子模型输入输出规格详解

### 8.1 PromptBuilder 输入输出

**作用:** 构造进入 Talker 的初始 Prompt Embedding

**输入模式:**

| 模式 | 参数组合 | 用途 |
|------|----------|------|
| 音色设计 | `text + instruct` | 根据文本指令设计音色 |
| 自定义音色 | `text + speaker` | 使用预设说话人 ID/名称 |
| 声音克隆 | `text + voice(codes, spk_emb, text)` | 克隆参考音频的音色 |

**输出结构 (PromptData):**

```python
class PromptData:
    embd: np.ndarray        # [1, seq_len, 2048] float32 - Talker 的初始输入
    text: str               # 目标合成文本
    text_ids: List[int]     # 文本 Token IDs
    spk_emb: np.ndarray     # [2048] float32 - 说话人嵌入
    trailing_text_embd:     # [1, T_remain, 2048] float32 - 待步内注入的文本池
    compile_time: float     # 构造耗时 (秒)
```

**Prompt 结构 (内存布局):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Prefix (固定前缀)                                                │
├─────────────────────────────────────────────────────────────────┤
│  [指令块] (可选) → user instruction                             │
│  [@assistant\n] → 角色标识                                      │
│  [tts_pad + think_token]                                        │
│  [tts_pad + think_bos]                                          │
│  [tts_pad + lang_id] (可选，语言标识)                            │
│  [tts_pad + think_eos]                                          │
│  [tts_pad + spk_emb] (可选，说话人嵌入)                          │
│  [tts_bos + codec_pad]                                          │
├─────────────────────────────────────────────────────────────────┤
│  Body (可变主体)                                                │
├─────────────────────────────────────────────────────────────────┤
│  模式 A (非流式): 全量文本 + codec_pad + tts_eos + codec_bos     │
│  模式 B (流式): 首个文本 token + codec_bos，其余进 trailing       │
│  模式 C (ICL 克隆): 文本与参考音频按位融合                        │
└─────────────────────────────────────────────────────────────────┘
```

---

### 8.2 TTSResult 数据结构详解

**作用:** TTS 合成结果的统一承载对象，同时作为音色克隆的身份锚点

```python
@dataclass
class TTSResult:
    # ============= 核心特征（音色锚点必备） =============
    text: str                       # 合成文字内容
    text_ids: List[int]             # 文本 Token IDs
    codes: np.ndarray               # [T, 16] int64 - 音频 Codec IDs
    spk_emb: np.ndarray             # [2048] float32 - 全局音色向量

    # ============= 记忆持久（克隆用） =============
    ref_codes: np.ndarray           # [T_ref, 16] int64 - 参考音频码本
    final_state: DecoderState       # 解码器最终状态（不序列化）

    # ============= 选填元数据 =============
    info: str                       # 备注信息（音色描述等）
    summed_embeds: List[np.ndarray] # T × [2048] float32 - 叠加特征序列

    # ============= 产出附件 =============
    audio: np.ndarray               # [N] float32 - 音频波形 (24kHz)
    stats: Timing                   # 性能统计对象
```

**派生属性:**

```python
@property
def is_valid_anchor(self) -> bool:
    """是否具有作为 Voice 锚点的必要条件"""
    return len(self.codes) > 0 and self.spk_emb is not None

@property
def duration(self) -> float:
    """音频时长（秒）= len(audio) / 24000"""

@property
def rtf(self) -> float:
    """实时因子 = 核心推理耗时 / 音频时长"""
```

**JSON 序列化格式 (音色文件):**

```json
{
  "info": "音色描述",
  "text": "参考文本",
  "text_ids": [123, 456, ...],
  "codes": [[c0_0, c0_1, ..., c0_15], [c1_0, ...], ...],
  "ref_codes": [[...], ...],
  "spk_emb": "<Base64 编码的 float32 数组，长度 2048>"
}
```

---

### 8.3 DecoderState 结构详解

**作用:** 解码器的状态快照，支持流式推理的断点续传

```python
@dataclass
class DecoderState:
    # KV Cache (8 层，每层 K+V)
    kv_cache: List[np.ndarray]      # 16 × [1, 16, 72, 64] float16
                                    # 布局：[k0, v0, k1, v1, ..., k7, v7]

    # 卷积历史 (跨 chunk 持久化)
    pre_conv_history: np.ndarray    # [1, 512, H] float16 - 前卷积历史
    latent_buffer: np.ndarray       # [1, 1024, H] float16 - 潜在缓冲区
    conv_history: np.ndarray        # [1, 1024, H] float16 - 卷积历史

    # 流式对齐
    skip_samples: int               # 跳过采样点数 (流式静音消除)
    latent_audio: np.ndarray        # 尾部残留音频 (流式结束时取出)
```

**常量定义:**

```python
NUM_LAYERS = 8          # Transformer 层数
NUM_HEADS = 16          # 注意力头数
HEAD_DIM = 64           # 头维度
KV_CACHE_WINDOW = 72    # KV Cache 滑动窗口大小
SAMPLES_PER_FRAME = 1920 # 每帧生成的音频样本数 (24kHz / 12.5Hz)
```

---

### 8.4 Timing 性能统计结构

**作用:** 记录推理全链路各阶段的耗时

```python
@dataclass
class Timing:
    # 一次性阶段
    prompt_time: float              # Prompt 构造耗时
    prefill_time: float             # Talker 预填充耗时

    # 循环阶段 (列表记录每步耗时)
    talker_loop_times: List[float]      # Talker 每步解码耗时列表
    predictor_loop_times: List[float]   # Predictor 每步预测耗时列表
    chunk_gen_times: List[float]        # 每 chunk 码本生成耗时
    decoder_compute_times: List[float]  # 每 chunk 解码耗时

    total_steps: int                # 总生成步数
```

**派生统计:**

```python
@property
def first_audio_latency(self) -> float:
    """全链路首音延迟 = 首 chunk 攒码时间 + 首包解码时间"""
    return self.first_chunk_latency + self.first_decode_latency

@property
def total_inference_time(self) -> float:
    """全链路总耗时 = prompt + prefill + talker + predictor + decoder"""

@property
def inference_only_time(self) -> float:
    """核心推理耗时（不含解码渲染）= prompt + prefill + talker + predictor"""
```

---

### 8.5 各组件输入输出速查表

| 组件 | 输入 | 输出 | 张量形状 |
|------|------|------|----------|
| **PromptBuilder** | `text, speaker, codes` | `PromptData.embd` | `[1, seq, 2048]` |
| **Talker.prefill()** | `PromptData.embd` | `hidden_state` | `[2048]` |
| **Talker.decode_step()** | `fused_embed(audio+text)` | `hidden_state` | `[2048]` |
| **Talker.sample()** | `logits` | `code_0` | `int64 (0-2047)` |
| **Predictor.predict_frame()** | `master_hidden + code_0` | `step_codes + embeds` | `16 × int64, 16 × [2048]` |
| **StatefulDecoder.decode()** | `codes [N, 16] + state` | `audio + new_state` | `[N × 1920] + DecoderState` |
| **CodecEncoder** | `audio [T]` | `codes [N, 16]` | `int64` |
| **SpeakerEncoder** | `mels [T, 128]` | `spk_emb [D]` | `float32` |

---

### 8.6 跨进程通信协议

**DecodeRequest (主进程 → DecoderWorker):**

```python
@dataclass
class DecodeRequest:
    task_id: Union[str, int]
    msg_type: str       # "DECODE" | "DECODE_CHUNK" | "STOP" | "RESET"
    codes: np.ndarray   # [N, 16] int64
    is_final: bool
    state: DecoderState # 初始状态注入
```

**DecoderResponse (DecoderWorker → 主进程):**

```python
@dataclass
class DecoderResponse:
    task_id: Union[str, int]
    msg_type: str       # "AUDIO" | "FINISH" | "READY" | "ERROR"
    index: int          # 片段序号
    audio: np.ndarray   # [N] float32
    compute_time: float # 解码耗时
    state: DecoderState # 最终状态 (FINISH 消息)
    recv_time: float    # Proxy 接收时间戳
```

---

### 8.7 完整数据流 (端到端)

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. 用户输入                                                       │
│    - text: "你好，这是测试文本"                                   │
│    - voice: ref.json (包含 codes, spk_emb)                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. PromptBuilder.build_clone_prompt()                            │
│    输出：PromptData.embd [1, seq, 2048]                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. TalkerPredictor.prefill()                                     │
│    注入 Prompt 到 KV Cache                                        │
│    输出：last_hidden [2048]                                      │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. 自回归循环 (max_steps=400)                                    │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │ 4.1 Talker.decode_step(fused_embed)                     │   │
│    │     输出：new_hidden [2048]                              │   │
│    │                                                          │   │
│    │ 4.2 Talker.sample() → code_0 ∈ [0, 2047)               │   │
│    │                                                          │   │
│    │ 4.3 Predictor.predict_frame(hidden, code_0)             │   │
│    │     输出：codes_1_to_15 [15], embeds [15 × 2048]        │   │
│    │                                                          │   │
│    │ 4.4 summed_embed = Σ(embeds)                            │   │
│    │                                                          │   │
│    │ 4.5 yield step_codes [16] → chunk_buffer                │   │
│    └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. 累积 chunk (chunk_size=12)                                    │
│    chunk_buffer: [12 × 16] int64                                 │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 6. DecoderProxy.decode()                                         │
│    → 跨进程发送到 DecoderWorker                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 7. StatefulDecoder._decode() (ONNX Runtime)                     │
│    输入：codes [12, 16] + state                                  │
│    输出：audio [12 × 1920 = 23040] + new_state                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 8. TTSResult                                                      │
│    - audio: [N] float32                                          │
│    - codes: [T, 16] int64                                        │
│    - spk_emb: [2048] float32                                     │
│    - final_state: DecoderState                                   │
│    - stats: Timing                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. 总结

本项目成功将原生 Qwen3-TTS 端到端模型拆分为以下独立可运行的子模块：

1. **Talker (GGUF)** - 28 层大师模型，负责文本理解和语音骨架生成
2. **Predictor (GGUF)** - 8 层工匠模型，负责细节码本补充
3. **Decoder (ONNX)** - 8 层因果 ConvNet，负责波形合成
4. **Codec Encoder (ONNX)** - 音频到声学码的编码器
5. **Speaker Encoder (ONNX)** - 说话人特征提取器
6. **Embeddings (NPY)** - 静态查找表资产

拆分后的架构支持：
- **跨平台推理**: CUDA / DML / Vulkan / CPU
- **量化压缩**: Q5_K (Talker) / Q8_0 (Predictor) / FP16 (Decoder)
- **流式优化**: 首包延迟 ~300ms，RTF < 0.1
- **模块化升级**: 可独立替换/优化任一组件

所有子模型的输入输出规格已在本文档中详细记录，便于后续维护和扩展。
