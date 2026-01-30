"""
100-Test-Quick-Load.py - 验证单行代码开流定调
"""
import os
import sys

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine

def test_quick_load():
    print("\n" + "="*50)
    print("🧪 测试: 单行代码 [流初始化 + 音色载入] 验证")
    print("="*50)

    # 1. 初始化引擎
    engine = TTSEngine(model_dir="model")
    
    # 我们使用之前测试生成的 JSON 文件
    JSON_PATH = "output/identity_vivian_light.json"
    if not os.path.exists(JSON_PATH):
        print(f"❌ 找不到测试所需的 JSON 文件: {JSON_PATH}")
        return

    # 2. 极致简化：一键开流并设置音色
    print(f"\n🚀 正在通过 voice_path 参数创建流...")
    stream = engine.create_stream(voice_path=JSON_PATH)
    
    # 验证身份已自动锁定
    print(f"✅ 流已就绪！")
    print(f"   ├─ 当前音色文字: '{stream.voice.text}'")
    
    # 3. 直接合成
    print(f"\n🎤 正在合成新文本...")
    res = stream.tts("这句合成代码现在非常简洁，对吧？")
    res.play()
    res.print_stats()

if __name__ == "__main__":
    test_quick_load()
