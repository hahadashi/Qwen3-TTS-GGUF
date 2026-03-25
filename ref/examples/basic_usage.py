"""
基本使用示例

演示Qwen3TTSWrapperEngine的基本使用方法。
"""

import torch
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_tts_wrapper import Qwen3TTSWrapperEngine, TTSConfig


def basic_usage_example():
    """基本使用示例"""
    print("=" * 60)
    print("Qwen3-TTS Wrapper 基本使用示例")
    print("=" * 60)

    # 1. 初始化引擎
    print("\n1. 初始化引擎...")
    print("   注意: 需要提供有效的模型路径")

    model_path = "D:/models/Qwen3-TTS-12Hz-1.7B-Base"  # 请修改为实际路径

    # 注释掉实际加载，使用模拟说明
    print(f"   model_path = {model_path}")
    print(f"   device = 'cuda' (如果GPU可用) 或 'cpu'")
    print(f"   dtype = 'float16'")

    # engine = Qwen3TTSWrapperEngine(
    #     model_path=model_path,
    #     device="cuda",
    #     dtype="float16"
    # )

    print("\n2. 打印引擎信息...")
    print(f"   模型类型: {model_path.split('-')[-1]}")  # Base/CustomVoice/VoiceDesign
    print(f"   模型大小: 1.7B")

    # print(engine)

    print("\n3. 创建流式会话...")
    # stream = engine.create_stream()

    print("   stream = engine.create_stream()")

    print("\n4. 设置音色...")
    print("   方式1: 使用参考音频 (Base模型)")
    print("   方式2: 使用预设音色 (CustomVoice模型)")
    print("   方式3: 使用音色描述 (VoiceDesign模型)")

    print("\n5. 合成语音...")
    text = "你好，我是千问语音助手，很高兴为您服务！"

    print(f"   text = '{text}'")

    # 非流式合成
    print("\n   非流式合成:")
    print("   audio = stream.synthesize(text=text)")

    # 流式合成
    print("\n   流式合成:")
    print("   audio = stream.synthesize_stream(")
    print("       text=text,")
    print("       audio_callback=lambda chunk, is_last: print(f'生成音频块: {chunk.shape}')")
    print("   )")

    print("\n6. 保存结果...")
    print("   import torchaudio")
    print("   torchaudio.save('output.wav', audio.unsqueeze(0), 24000)")

    print("\n" + "=" * 60)
    print("示例代码执行完毕")
    print("=" * 60)


def advanced_usage_example():
    """高级使用示例"""
    print("\n" + "=" * 60)
    print("高级使用示例")
    print("=" * 60)

    print("\n1. 自定义生成配置...")

    config = TTSConfig(
        max_steps=500,           # 最大生成步数
        temperature=0.8,         # 采样温度 (越高越随机)
        sub_temperature=0.8,     # 子采样温度
        top_k=50,                # Top-K采样
        top_p=0.9,               # Top-P采样
        repetition_penalty=1.1,  # 重复惩罚
        seed=42,                 # 随机种子 (可复现)
        streaming=True           # 启用流式模式
    )

    print(f"   config = TTSConfig(")
    print(f"       max_steps={config.max_steps},")
    print(f"       temperature={config.temperature},")
    print(f"       sub_temperature={config.sub_temperature},")
    print(f"       top_k={config.top_k},")
    print(f"       top_p={config.top_p},")
    print(f"       repetition_penalty={config.repetition_penalty},")
    print(f"       seed={config.seed},")
    print(f"       streaming={config.streaming}")
    print(f"   )")

    print("\n2. 使用自定义配置合成...")
    print("   audio = stream.synthesize(text=text, config=config)")

    print("\n3. 编码参考音频...")
    print("   reference_audio = torchaudio.load('reference.wav')[0]")
    print("   codes, speaker_emb = engine.encode_reference_audio(reference_audio)")
    print("   stream.set_voice(")
    print("       speaker_embedding=speaker_emb,")
    print("       reference_codes=codes")
    print("   )")

    print("\n4. 批量合成...")
    print("   texts = ['第一句话', '第二句话', '第三句话']")
    print("   for text in texts:")
    print("       audio = stream.synthesize(text=text)")
    print("       # 保存或播放")

    print("\n" + "=" * 60)


def callback_example():
    """回调函数示例"""
    print("\n" + "=" * 60)
    print("回调函数使用示例")
    print("=" * 60)

    print("\n1. 音频块回调 (实时播放)...")

    def audio_callback(chunk, is_last):
        """每生成一个音频块就播放"""
        print(f"   收到音频块: {chunk.shape}, is_last={is_last}")
        # 在实际使用中，可以在这里播放音频
        # import sounddevice as sd
        # sd.play(chunk.cpu().numpy(), 24000)

    print("   def audio_callback(chunk, is_last):")
    print("       print(f'收到音频块: {chunk.shape}, is_last={is_last}')")
    print("       # 播放音频...")
    print("")
    print("   audio = stream.synthesize_stream(")
    print("       text=text,")
    print("       audio_callback=audio_callback")
    print("   )")

    print("\n2. Code生成进度回调...")

    def code_callback(stage, step, codes):
        """显示生成进度"""
        if codes is None:
            print(f"   [{stage}] 开始生成...")
        else:
            print(f"   [{stage}] 已生成 {step} 帧, codes.shape={codes.shape}")

    print("   def code_callback(stage, step, codes):")
    print("       if codes is None:")
    print("           print(f'[{stage}] 开始生成...')")
    print("       else:")
    print("           print(f'[{stage}] 已生成 {step} 帧')")
    print("")
    print("   audio = stream.synthesize_stream(")
    print("       text=text,")
    print("       code_callback=code_callback")
    print("   )")

    print("\n" + "=" * 60)


def model_selection_example():
    """模型选择示例"""
    print("\n" + "=" * 60)
    print("不同模型的使用")
    print("=" * 60)

    print("\n1. Base模型 (声音克隆)...")
    print("   - 需要参考音频")
    print("   - 支持最高精度的声音克隆")
    print("   - 有Speaker Encoder组件")
    print("")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path='.../Qwen3-TTS-12Hz-1.7B-Base',")
    print("       device='cuda'")
    print("   )")
    print("   # 设置参考音频")
    print("   ref_audio, sr = torchaudio.load('reference.wav')")
    print("   codes, spk_emb = engine.encode_reference_audio(ref_audio)")
    print("   stream.set_voice(speaker_embedding=spk_emb, reference_codes=codes)")

    print("\n2. CustomVoice模型 (预设音色)...")
    print("   - 使用预设说话人ID")
    print("   - 可选的风格指令控制")
    print("   - 无Speaker Encoder组件")
    print("")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path='.../Qwen3-TTS-12Hz-1.7B-CustomVoice',")
    print("       device='cuda'")
    print("   )")
    print("   # 使用预设音色")
    print("   stream.set_builtin_voice('Vivian')")

    print("\n3. VoiceDesign模型 (音色设计)...")
    print("   - 通过自然语言描述创造音色")
    print("   - 无Speaker Encoder组件")
    print("")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path='.../Qwen3-TTS-12Hz-1.7B-VoiceDesign',")
    print("       device='cuda'")
    print("   )")

    print("\n" + "=" * 60)


def device_selection_example():
    """设备选择示例"""
    print("\n" + "=" * 60)
    print("设备选择示例")
    print("=" * 60)

    print("\n1. CUDA (NVIDIA GPU)...")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path=model_path,")
    print("       device='cuda',      # 使用CUDA")
    print("       dtype='float16'     # GPU推荐float16")
    print("   )")

    print("\n2. CPU...")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path=model_path,")
    print("       device='cpu',")
    print("       dtype='float32'     # CPU推荐float32")
    print("   )")

    print("\n3. 检测设备可用性...")
    print("   import torch")
    print("   if torch.cuda.is_available():")
    print("       device = 'cuda'")
    print("       dtype = 'float16'")
    print("   else:")
    print("       device = 'cpu'")
    print("       dtype = 'float32'")
    print("")
    print("   engine = Qwen3TTSWrapperEngine(")
    print("       model_path=model_path,")
    print("       device=device,")
    print("       dtype=dtype")
    print("   )")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    basic_usage_example()
    advanced_usage_example()
    callback_example()
    model_selection_example()
    device_selection_example()

    print("\n所有示例演示完毕！")
    print("\n注意: 实际使用时请:")
    print("1. 修改模型路径为实际路径")
    print("2. 确保已安装 qwen_tts 包")
    print("3. 取消注释相关代码行")
