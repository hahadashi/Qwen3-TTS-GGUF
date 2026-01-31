"""
41-Inference-Base.py - Qwen3-TTS Base 模型语音克隆脚本 (Engine 版)
调用 TTSEngine 以确保 Prompt 构造协议的标准性。
"""
import time
from qwen3_tts_gguf import TTSEngine, TTSConfig

def main():

    # 初始化引擎
    print("🚀 [Base-Clone] 正在初始化 TTS 引擎...")
    engine = TTSEngine(model_dir="model-base")
    stream = engine.create_stream()
    
    # 设置音色锚点

    # 读取音频文件，需要编码为 Code，是有损克隆
    # REF_AUDIO = "output/sample.wav"                
    # REF_TEXT = "你好，我是千问，你今天过得好吗？"
    # stream.set_voice(REF_AUDIO, REF_TEXT)

    # 从 json 读取 code，无需从 wav 编码，可以无损克隆
    REF_JSON = "output/sample.json"           
    stream.set_voice(REF_JSON)
    
    # if stream.voice:
    #     print(f"🔊 正在还原并播放参考音频: {REF_AUDIO} ...")
    #     stream.voice.decode(engine.decoder)
    #     stream.voice.play(blocking=True)
    

    # 流式模式下，clone 依然会返回完整 result，但播放是并发进行的
    print(f"\n🎙️  [2/2] 开始流式推理 (边推边播)...")
    target_text = "你真是太棒啦！"
    config = TTSConfig(temperature=0.5)
    result = stream.clone(
        text=target_text,
        streaming=True,
        verbose=True
    )
    result.print_stats()
    
    print("⏳ 等待流式播放完成 (使用 Event 同步)...")
    stream.join()
    time.sleep(0.5)

    result.save("./output/clone_result.wav")     # 保存为音频
    result.save("./output/clone_result.json")    # 保存为json，内含无损的音频code

    engine.shutdown()

if __name__ == "__main__":
    main()
