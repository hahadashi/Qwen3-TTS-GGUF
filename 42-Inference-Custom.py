"""
42-Inference-Custom.py - Qwen3-TTS 精品音色推理脚本 (Engine 版)
"""
import time 
from qwen3_tts_gguf.inference.engine import TTSEngine

def main():
    
    # 1. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    # 注意：如果您的定制模型放在 model-custom 目录，请指定
    engine = TTSEngine(model_dir="model-custom")
    stream = engine.create_stream()

    # 2. 配置参数
    # 官方支持音色: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    TARGET_TEXT = "很多人对大脑有个天大的误解，觉得这玩意儿是拿来思考的，其实大脑最重要的工作压根不是理解世界，而是防止自己被一惊一乍的世界给吓死，所以它真正的功能是，预测。"
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
