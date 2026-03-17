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

*生成时间：2026-03-17*
