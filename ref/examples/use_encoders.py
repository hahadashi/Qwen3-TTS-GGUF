"""
使用 Encoder Wrappers 示例

展示三种使用 CodecEncoderWrapper 和 SpeakerEncoderWrapper 的方式：
1. 原接口方式 (保持向后兼容)
2. 便捷函数方式 (推荐用于新代码)
3. StreamingEngine 方式 (推荐用于流式推理)

Author: Claude
Date: 2026-03-23
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

qwen_tts_path = project_root / "Qwen3-TTS-main"
if qwen_tts_path.exists():
    sys.path.insert(0, str(qwen_tts_path))

from qwen_tts import Qwen3TTSModel
from qwen3_tts_wrapper import (
    StreamingEngine,
    StreamConfig,
    VoiceAnchor,
    # 便捷函数
    create_encoders,
    encode_audio,
    extract_speaker_embedding,
    # Wrappers
    CodecEncoderWrapper,
    SpeakerEncoderWrapper,
)


# ============================================================================
# 方式 1: 原接口 (保持向后兼容)
# ============================================================================

def example_1_original_interface():
    """方式 1: 使用原接口 create_voice_clone_prompt"""
    print("\n" + "=" * 60)
    print("方式 1: 原接口 (create_voice_clone_prompt)")
    print("=" * 60)

    model_path = "D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio_path = "D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav"

    # 加载模型
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="cpu",
    )

    # 使用原接口提取特征 (一次性获取 speaker embedding 和 codec codes)
    voice_clone_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text="你好，我是千问",
        x_vector_only_mode=False,
    )

    voice_item = voice_clone_items[0]
    speaker_embedding = voice_item.ref_spk_embedding  # [2048]
    ref_codes = voice_item.ref_code                   # [T, 16]

    print(f"Speaker embedding: {speaker_embedding.shape}")
    print(f"Reference codes: {ref_codes.shape}")

    return model, speaker_embedding, ref_codes


# ============================================================================
# 方式 2: 便捷函数 (推荐用于新代码)
# ============================================================================

def example_2_convenience_functions():
    """方式 2: 使用便捷函数"""
    print("\n" + "=" * 60)
    print("方式 2: 便捷函数 (create_encoders, encode_audio, extract_speaker_embedding)")
    print("=" * 60)

    model_path = "D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio_path = "D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav"

    # 加载模型
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="cpu",
    )

    # 方式 2a: 使用 create_encoders 创建 wrappers
    print("\n[方式 2a] 使用 create_encoders:")
    codec_enc, speaker_enc = create_encoders(model)

    # 分别提取特征
    spk_emb = speaker_enc.encode(ref_audio_path)  # 直接从文件路径
    print(f"  Speaker embedding: {spk_emb.shape}")

    # 加载音频用于 codec 编码
    from scipy.io import wavfile
    sr, audio = wavfile.read(ref_audio_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    codes = codec_enc.encode(audio)
    print(f"  Reference codes: {codes.shape}")

    # 方式 2b: 使用顶层便捷函数
    print("\n[方式 2b] 使用顶层便捷函数:")
    spk_emb2 = extract_speaker_embedding(model, ref_audio_path)
    print(f"  Speaker embedding: {spk_emb2.shape}")

    codes2 = encode_audio(model, audio)
    print(f"  Reference codes: {codes2.shape}")

    return model, spk_emb, codes


# ============================================================================
# 方式 3: StreamingEngine (推荐用于流式推理)
# ============================================================================

def example_3_via_engine():
    """方式 3: 通过 StreamingEngine 访问 encoders"""
    print("\n" + "=" * 60)
    print("方式 3: 通过 StreamingEngine 访问 encoders")
    print("=" * 60)

    model_path = "D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio_path = "D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav"

    # 加载模型并创建引擎
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="cpu",
    )

    engine = StreamingEngine(model=model, device="cpu")

    # 通过引擎访问 encoders
    print("\n引擎中的 encoders:")
    print(f"  codec_encoder: {type(engine.codec_encoder).__name__}")
    print(f"  speaker_encoder: {type(engine.speaker_encoder).__name__}")

    # 提取特征
    spk_emb = engine.speaker_encoder.encode(ref_audio_path)
    print(f"\nSpeaker embedding: {spk_emb.shape}")

    from scipy.io import wavfile
    sr, audio = wavfile.read(ref_audio_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    codes = engine.codec_encoder.encode(audio)
    print(f"Reference codes: {codes.shape}")

    return engine, spk_emb, codes


# ============================================================================
# 完整示例: 使用 encoders + 流式合成
# ============================================================================

def example_complete_tts():
    """完整示例: 使用 encoders 提取特征并进行流式合成"""
    print("\n" + "=" * 60)
    print("完整示例: Encoders + 流式合成")
    print("=" * 60)

    model_path = "D:/claude/Qwen3-TTS-GGUF/models/Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio_path = "D:/claude/Qwen3-TTS-GGUF-TEST/output/elaborate/Vivian.wav"
    output_path = "output/encoder_example.wav"

    # 加载模型并创建引擎
    print("\n[1/5] 加载模型...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="cpu",
    )
    engine = StreamingEngine(model=model, device="cpu")

    # 使用引擎的 encoders 提取特征
    print("\n[2/5] 提取参考音频特征...")
    spk_emb = engine.speaker_encoder.encode(ref_audio_path)

    from scipy.io import wavfile
    sr, audio = wavfile.read(ref_audio_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    ref_codes = engine.codec_encoder.encode(audio)

    print(f"  Speaker embedding: {spk_emb.shape}")
    print(f"  Reference codes: {ref_codes.shape}")

    # 创建 VoiceAnchor
    print("\n[3/5] 创建 VoiceAnchor...")
    voice = VoiceAnchor(
        speaker_embedding=torch.from_numpy(spk_emb).unsqueeze(0),
        reference_codes=torch.from_numpy(ref_codes),
        ref_text="你好，我是千问",
        name="Vivian",
        lang="zh",
    )

    # 配置合成参数
    print("\n[4/5] 配置合成参数...")
    config = StreamConfig(
        max_frames=50,
        chunk_size=12,
        decode_audio=True,
        enable_audio_feedback=True,
    )

    # 执行合成
    print("\n[5/5] 执行流式合成...")
    text = "今天天气真不错，适合出去走走。"
    result = engine.clone(text=text, voice=voice, config=config)

    # 保存音频
    if result.audio is not None and result.audio.numel() > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio_np = result.audio.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype('int16')
        wavfile.write(output_path, 24000, audio_int16)

        audio_duration = result.audio.numel() / 24000
        print(f"\n合成完成!")
        print(f"  输出文件: {output_path}")
        print(f"  音频时长: {audio_duration:.2f}s")
        print(f"  生成帧数: {len(result.codes) if result.codes else 0}")

        return True
    else:
        print("\n警告: 没有生成音频!")
        return False


# ============================================================================
# 对比不同方式的适用场景
# ============================================================================

def print_usage_comparison():
    """打印不同方式的适用场景对比"""
    print("\n" + "=" * 60)
    print("不同方式适用场景对比")
    print("=" * 60)

    print("""
┌─────────────────┬────────────────────────────────────────────────┐
│ 方式            │ 适用场景                                        │
├─────────────────┼────────────────────────────────────────────────┤
│ 方式 1: 原接口  │ - 现有代码无需修改                              │
│                 │ - 需要同时获取 speaker embedding 和 codec codes  │
│                 │ - 使用 Qwen3TTSModel 的其他功能                 │
├─────────────────┼────────────────────────────────────────────────┤
│ 方式 2: 便捷函数│ - 新代码 (推荐)                                 │
│                 │ - 只需要 speaker embedding 或 codec codes       │
│                 │ - 简洁的 API，与 GGUF 方案对齐                  │
├─────────────────┼────────────────────────────────────────────────┤
│ 方式 3: Engine  │ - 流式推理场景 (推荐)                            │
│                 │ - 需要使用 StreamingEngine 进行合成              │
│                 │ - 统一管理所有组件                              │
└─────────────────┴────────────────────────────────────────────────┘

推荐用法:
- 纯特征提取: 使用方式 2 (便捷函数)
- 流式合成:    使用方式 3 (StreamingEngine)
- 现有代码:    保持方式 1 (原接口)
""")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("Encoder Wrappers 使用示例")
    print("=" * 60)

    # 打印使用对比
    print_usage_comparison()

    # 运行示例 (用户可以选择注释掉不需要的示例)

    # 示例 1: 原接口
    example_1_original_interface()

    # 示例 2: 便捷函数
    example_2_convenience_functions()

    # 示例 3: StreamingEngine
    example_3_via_engine()

    # 完整示例: 合成
    # example_complete_tts()  # 取消注释以运行完整合成

    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
