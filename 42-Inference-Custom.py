"""
42-Inference-Custom.py - Qwen3-TTS 精品音色推理脚本 (Engine 版)
"""
import time 
from qwen3_tts_gguf.engine import TTSEngine

def main():
    
    # 1. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    # 注意：如果您的定制模型放在 model-custom 目录，请指定
    engine = TTSEngine(model_dir="model-custom")
    stream = engine.create_stream()

    # 2. 配置参数
    # 官方支持音色: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    TARGET_TEXT = "你好，我是千问，你今天过得好吗？"
    SPEAKER = "Vivian"
    INSTRUCT = "用温柔的语气说"

    # 3. 进行推理 (调用 custom 模式)
    print(f"🎭 正在合成...")
    result = stream.custom(
        text=TARGET_TEXT,
        speaker=SPEAKER,
        instruct=INSTRUCT,
        language='chinese',
        streaming=True,
        verbose=True
    )

    stream.join()
    time.sleep(0.5)
    
    print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
    result.save("./output/custom.wav")
    result.save("./output/custom.json")

    engine.shutdown()

if __name__ == "__main__":
    main()
