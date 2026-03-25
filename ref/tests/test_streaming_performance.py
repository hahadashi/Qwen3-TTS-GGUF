"""
stream=True vs stream=False 性能对比测试

测量指标:
1. 首音频延迟 (First Audio Latency): 从调用开始到第一个音频块返回的时间
2. RTF (Real-Time Factor): 总生成时间 / 音频时长
3. 各 chunk 的生成时间分布
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Qwen3-TTS-main"))

import time
import torch
import numpy as np
import wave

from qwen_tts import Qwen3TTSModel
from qwen3_tts_wrapper.streaming import StreamingEngine, StreamConfig
from qwen3_tts_wrapper.data import VoiceAnchor

# ========== 配置 ==========
MODEL_PATH = 'D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base'
REF_AUDIO = 'D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav'
REF_TEXT = '你好，我是千问，你今天过得好吗？'
TEXT = '今天天气怎么样'
SEED = 42
OUTPUT_DIR = str(project_root / 'output')

# ========== 加载模型 ==========
print('=' * 70)
print('Loading model...')
print('=' * 70)

model = Qwen3TTSModel.from_pretrained(MODEL_PATH, local_files_only=True, device_map='cpu')

# 提取 voice features
voice_items = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False
)
voice_item = voice_items[0]

voice = VoiceAnchor(
    speaker_embedding=voice_item.ref_spk_embedding.unsqueeze(0),
    reference_codes=voice_item.ref_code,
    ref_text=REF_TEXT + " ",
    name='Vivian',
    lang='zh',
)

engine = StreamingEngine(model=model, device='cpu')

# ========== 辅助函数 ==========
def format_time(seconds):
    """格式化时间"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"

def calculate_rtf(gen_time, audio_duration):
    """计算 RTF"""
    if audio_duration > 0:
        return gen_time / audio_duration
    return 0

# ========== 1. stream=False (非流式) ==========
print('\n' + '=' * 70)
print('1. stream=False 模式测试')
print('=' * 70)

torch.manual_seed(SEED)
np.random.seed(SEED)

config_false = StreamConfig(
    max_frames=100,
    temperature=0.9,
    top_k=50,
    repeat_penalty=1.5,
    streaming=False,
    seed=SEED,
    decode_audio=True,
)

print(f'\n[START] 调用 clone() ...')
t_start_false = time.time()

result_false = engine.clone(text=TEXT, voice=voice, config=config_false)

t_end_false = time.time()
gen_time_false = t_end_false - t_start_false

audio_false = result_false.audio
duration_false = len(audio_false) / 24000
rtf_false = calculate_rtf(gen_time_false, duration_false)

print(f'\n[RESULT] stream=False:')
print(f'  总生成时间:     {format_time(gen_time_false)}')
print(f'  音频时长:       {format_time(duration_false)}')
print(f'  RTF:            {rtf_false:.3f}x')
print(f'  首音频延迟:     {format_time(gen_time_false)} (整段生成完成后返回)')
print(f'  生成帧数:       {len(result_false.codes)}')

# ========== 2. stream=True (流式) ==========
print('\n' + '=' * 70)
print('2. stream=True 模式测试')
print('=' * 70)

torch.manual_seed(SEED)
np.random.seed(SEED)

config_true = StreamConfig(
    max_frames=100,
    temperature=0.9,
    top_k=50,
    repeat_penalty=1.5,
    streaming=True,
    chunk_size=12,  # 每 12 帧解码一次
    seed=SEED,
    decode_audio=True,
)

# 使用流式 API 手动测量每个 chunk
print(f'\n[START] 调用 clone() ...')
t_start_true = time.time()

# 需要修改 engine 来支持 yield 音频
# 这里我们用 timing 信息来估算

result_true = engine.clone(text=TEXT, voice=voice, config=config_true)

t_end_true = time.time()
gen_time_true = t_end_true - t_start_true

audio_true = result_true.audio
duration_true = len(audio_true) / 24000
rtf_true = calculate_rtf(gen_time_true, duration_true)

# 从 timing 获取详细信息
timing = result_true.timing

print(f'\n[RESULT] stream=True:')
print(f'  总生成时间:     {format_time(gen_time_true)}')
print(f'  音频时长:       {format_time(duration_true)}')
print(f'  RTF:            {rtf_true:.3f}x')
print(f'  生成帧数:       {len(result_true.codes)}')

# 详细 timing 分析
if hasattr(timing, 'prefill_time'):
    print(f'\n[详细 Timing]:')
    print(f'  Prefill 时间:   {format_time(timing.prefill_time / 1000)}')

if hasattr(timing, 'talker_loop_times') and timing.talker_loop_times:
    avg_talker = np.mean(timing.talker_loop_times)
    print(f'  Talker 平均:    {format_time(avg_talker / 1000)} ({len(timing.talker_loop_times)} steps)')

if hasattr(timing, 'predictor_loop_times') and timing.predictor_loop_times:
    avg_predictor = np.mean(timing.predictor_loop_times)
    print(f'  Predictor 平均: {format_time(avg_predictor / 1000)} ({len(timing.predictor_loop_times)} steps)')

if hasattr(timing, 'chunk_gen_times') and timing.chunk_gen_times:
    print(f'\n[Chunk 生成时间]:')
    for i, ct in enumerate(timing.chunk_gen_times):
        print(f'  Chunk {i+1}: {format_time(ct)}')

# 估算首音频延迟
# stream=True 模式下，第一个 chunk (12帧) 解码完成后就返回音频
# 12帧 = 12 * 80ms = 960ms 音频
first_chunk_audio_duration = 12 * 0.08  # 秒
if hasattr(timing, 'chunk_gen_times') and timing.chunk_gen_times:
    # 首音频延迟 ≈ prefill + 第一个 chunk 的生成时间
    prefill_time = timing.prefill_time / 1000 if hasattr(timing, 'prefill_time') else 0
    first_chunk_gen_time = timing.chunk_gen_times[0] if timing.chunk_gen_times else 0
    estimated_first_latency = prefill_time + first_chunk_gen_time
    print(f'\n[首音频延迟估算]:')
    print(f'  Prefill:        {format_time(prefill_time)}')
    print(f'  Chunk 1 生成:   {format_time(first_chunk_gen_time)}')
    print(f'  估算首音频延迟: {format_time(estimated_first_latency)}')
    print(f'  首音频时长:     {format_time(first_chunk_audio_duration)}')

# ========== 3. 对比汇总 ==========
print('\n' + '=' * 70)
print('3. 性能对比汇总')
print('=' * 70)

print(f'\n{"指标":<20} {"stream=False":<15} {"stream=True":<15} {"差异"}')
print('-' * 65)
print(f'{"总生成时间":<20} {format_time(gen_time_false):<15} {format_time(gen_time_true):<15} {format_time(gen_time_true - gen_time_false)}')
print(f'{"音频时长":<20} {format_time(duration_false):<15} {format_time(duration_true):<15} {"-"}')
print(f'{"RTF":<20} {rtf_false:<15.3f} {rtf_true:<15.3f} {rtf_true - rtf_false:+.3f}')

# 首音频延迟对比
print(f'\n{"首音频延迟对比":<20}')
print('-' * 65)
print(f'  stream=False: {format_time(gen_time_false)} (需等待整段生成)')
if hasattr(timing, 'chunk_gen_times') and timing.chunk_gen_times:
    print(f'  stream=True:  {format_time(estimated_first_latency)} (第一个 chunk 返回)')
    latency_improvement = gen_time_false - estimated_first_latency
    print(f'  延迟降低:     {format_time(latency_improvement)} ({100*latency_improvement/gen_time_false:.1f}%)')

# Codes 一致性
codes_false = result_false.codes
codes_true = result_true.codes
codes_match = len(codes_false) == len(codes_true)
if codes_match:
    for i, (c1, c2) in enumerate(zip(codes_false, codes_true)):
        if not torch.equal(c1, c2):
            codes_match = False
            break

print(f'\n{"Codes 一致性":<20} {"OK - 一致" if codes_match else "X - 不一致"}')

# ========== 4. 保存音频 ==========
print('\n' + '=' * 70)
print('保存音频文件')
print('=' * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# stream=False
path_false = os.path.join(OUTPUT_DIR, 'perf_stream_false.wav')
audio_np = (audio_false.numpy() * 32767).astype(np.int16)
with wave.open(path_false, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio_np.tobytes())
print(f'  stream=False: {path_false}')

# stream=True
path_true = os.path.join(OUTPUT_DIR, 'perf_stream_true.wav')
audio_np = (audio_true.numpy() * 32767).astype(np.int16)
with wave.open(path_true, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio_np.tobytes())
print(f'  stream=True:  {path_true}')

print('\n完成!')
