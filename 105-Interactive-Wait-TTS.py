"""
105-Interactive-Wait-TTS.py - 交互式“全转完再播”终端 (增强指令版)
"""
import os
import sys
import time

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import TTSConfig
from qwen3_tts_gguf.constants import SPEAKER_MAP, LANGUAGE_MAP

def print_help():
    print("\n" + "-"*40)
    print("🛠️  可用指令系统 (非流式模式):")
    print("  /speakers          查看内置人名列表")
    print("  /languages         查看支持语言列表")
    print("  /voice <人名> <语言> <文本>")
    print("                     即时合成并激活新音色锚点")
    print("  /load <路径>       加载 JSON 存档")
    print("  /save <路径>       保存当前音色锚点")
    print("-" * 10)
    print("  /info              配置预览")
    print("  /temp <值>         采样温度控制")
    print("  /steps <值>        最大生成步数控制")
    print("  /reset             重置推理记忆")
    print("  /help              显示帮助")
    print("-" * 40)

def interactive_wait_session():
    print("\n" + "="*60)
    print("🎤 Qwen3-TTS 交互式终端 (非流式：全转完再放)")
    print("="*60)

    # 1. 引擎初始化
    engine = TTSEngine(verbose=True)
    
    # 2. 默认音色
    JSON_PATH = "output/vivian.json"
    if os.path.exists(JSON_PATH):
        print(f"\n✅ 自动加载默认音色: {JSON_PATH}")
        stream = engine.create_stream(voice_path=JSON_PATH)
    else:
        stream = engine.create_stream()
        print(f"\nℹ️ 暂无初始音色，您可以输入普通文字合成，或通过指令设定。")

    # 3. 非流式配置
    cfg = TTSConfig(stream_play=False, max_steps=300)

    print_help()

    try:
        while True:
            raw_input = input("\n👉 请输入文本或指令: ").strip()
            
            if not raw_input:
                continue

            # --- 指令处理 ---
            if raw_input.startswith('/'):
                parts = raw_input.split(maxsplit=3)
                cmd = parts[0].lower()
                
                if cmd == '/help':
                    print_help()
                elif cmd == '/info':
                    print("\n[当前状态]")
                    print(f"  - 采样温度: {cfg.temperature}")
                    print(f"  - 步数上限: {cfg.max_steps}")
                    print(f"  - 当前音色: {stream.voice.info if stream.voice else '未设置'}")
                elif cmd == '/speakers':
                    print("\n🎙️ 可用说话人:")
                    for name in SPEAKER_MAP.keys(): print(f"  - {name}")
                elif cmd == '/languages':
                    print("\n🌏 支持语言:")
                    for lang in LANGUAGE_MAP.keys(): print(f"  - {lang}")
                elif cmd == '/voice' and len(parts) >= 4:
                    spk, lang, v_text = parts[1], parts[2], parts[3]
                    print(f"⌛ 正在生成音色锚点...")
                    stream.set_voice_from_speaker(spk, v_text, language=lang, config=cfg, verbose=True)
                    print(f"✅ 音色已激活。")
                elif cmd == '/load' and len(parts) > 1:
                    stream.set_voice_from_json(parts[1])
                    print(f"✅ 已载入存档。")
                elif cmd == '/save':
                    p = parts[1] if len(parts) > 1 else f"output/voice_{int(time.time())}.json"
                    stream.voice.save_json(p)
                    print(f"✅ 已保存至: {p}")
                elif cmd == '/temp' and len(parts) > 1:
                    cfg.temperature = float(parts[1])
                    print(f"✅ 温度设为: {cfg.temperature}")
                elif cmd == '/steps' and len(parts) > 1:
                    cfg.max_steps = int(parts[1])
                elif cmd == '/reset':
                    stream.master.clear_memory()
                    print("✅ 已重置。")
                elif cmd in ['/exit', '/q']:
                    break
                else:
                    print(f"❓ 未知或参数不足: {cmd}")
                continue

            # --- 文本合成处理 ---
            if raw_input.lower() in ['exit', 'q', 'quit', '退出']:
                break
            
            print(f"⏳ 正在计算推理...")
            res = stream.tts(raw_input, config=cfg, verbose=False)
            res.print_stats()
            print("🔊 正在播放...")
            res.play()
            
    except KeyboardInterrupt:
        print("\n\n🛑 正在退出...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        print("⏳ 回收资源...")
        try: engine.shutdown()
        except: pass
        print("✅ 退出完毕。")

if __name__ == "__main__":
    interactive_wait_session()
