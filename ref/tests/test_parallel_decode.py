"""
并行音频解码测试

对比串行解码 vs 并行解码的性能和音频质量
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

# ========== 配置 ==========
MODEL_PATH = 'D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base'
REF_AUDIO = 'D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav'
REF_TEXT = '你好，我是千问，你今天过得好吗？'
TEXT = '今天天气怎么样，我想出门走走'
SEED = 42
DEVICE = 'cpu'

def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"

# ========== 加载模型 ==========
print('=' * 70)
print('并行音频解码测试')
print('=' * 70)

from qwen_tts import Qwen3TTSModel
from qwen3_tts_wrapper.streaming import StreamingEngine, StreamConfig
from qwen3_tts_wrapper.data import VoiceAnchor

print('\n[Loading PyTorch model...]')
t_load_start = time.time()
model = Qwen3TTSModel.from_pretrained(MODEL_PATH, local_files_only=True, device_map=DEVICE)
t_load_end = time.time()
print(f'  Model load time: {format_time(t_load_end - t_load_start)}')
print(f'  Device: {DEVICE}')

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

# 输出目录
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# ========== 测试 1: 串行解码 (baseline) ==========
print('\n' + '=' * 70)
print('[Test 1] 串行解码 (parallel_decode=False)')
print('=' * 70)

engine1 = StreamingEngine(model=model, device=DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)

config1 = StreamConfig(
    max_frames=100,
    temperature=0.9,
    top_k=50,
    repeat_penalty=1.5,
    streaming=True,
    parallel_decode=False,  # 串行
    chunk_size=12,
    seed=SEED,
    decode_audio=True,
)

t_start1 = time.time()
result1 = engine1.clone(text=TEXT, voice=voice, config=config1)
t_end1 = time.time()
serial_time = t_end1 - t_start1

print(f'\n[RESULT] 串行解码:')
print(f'  总时间:       {format_time(serial_time)}')
print(f'  音频时长:     {format_time(len(result1.audio) / 24000)}')
print(f'  RTF:          {serial_time / (len(result1.audio) / 24000):.3f}x')
print(f'  生成帧数:     {len(result1.codes)}')

# 保存音频
import soundfile as sf
serial_path = output_dir / "serial_decode.wav"
sf.write(serial_path, result1.audio.numpy(), 24000)
print(f'  保存到:       {serial_path}')

# ========== 测试 2: 并行解码 ==========
print('\n' + '=' * 70)
print('[Test 2] 并行解码 (parallel_decode=True)')
print('=' * 70)

engine2 = StreamingEngine(model=model, device=DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)

config2 = StreamConfig(
    max_frames=100,
    temperature=0.9,
    top_k=50,
    repeat_penalty=1.5,
    streaming=True,
    parallel_decode=True,  # 并行
    chunk_size=12,
    seed=SEED,
    decode_audio=True,
)

t_start2 = time.time()
result2 = engine2.clone(text=TEXT, voice=voice, config=config2)
t_end2 = time.time()
parallel_time = t_end2 - t_start2

print(f'\n[RESULT] 并行解码:')
print(f'  总时间:       {format_time(parallel_time)}')
print(f'  音频时长:     {format_time(len(result2.audio) / 24000)}')
print(f'  RTF:          {parallel_time / (len(result2.audio) / 24000):.3f}x')
print(f'  生成帧数:     {len(result2.codes)}')

# 保存音频
parallel_path = output_dir / "parallel_decode.wav"
sf.write(parallel_path, result2.audio.numpy(), 24000)
print(f'  保存到:       {parallel_path}')

# ========== 对比结果 ==========
print('\n' + '=' * 70)
print('[对比结果]')
print('=' * 70)
print(f'  串行时间:     {format_time(serial_time)}')
print(f'  并行时间:     {format_time(parallel_time)}')
if parallel_time > 0:
    speedup = serial_time / parallel_time
    saved = serial_time - parallel_time
    print(f'  加速比:       {speedup:.2f}x')
    print(f'  节省时间:     {format_time(saved)} ({100*saved/serial_time:.1f}%)')

# 检查音频相似度
if result1.audio.shape == result2.audio.shape:
    diff = torch.abs(result1.audio - result2.audio).mean().item()
    max_diff = torch.abs(result1.audio - result2.audio).max().item()
    print(f'\n  音频差异 (MAE): {diff:.6f}')
    print(f'  音频差异 (Max): {max_diff:.6f}')
    if diff < 1e-5:
        print('  [OK] 音频基本相同')
    else:
        print('  [WARN] 音频有差异 (可能是浮点精度或线程调度导致)')
else:
    print(f'\n  音频长度不同: {result1.audio.shape} vs {result2.audio.shape}')

print('\n' + '=' * 70)
print('Done!')
print('=' * 70)
