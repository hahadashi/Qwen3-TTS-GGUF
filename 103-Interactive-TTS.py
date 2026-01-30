"""
103-Interactive-TTS.py - 交互式流式语音合成终端 (增强指令版)
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
    print("🛠️  可用指令系统:")
    print("  /speakers          列出所有内置说话人")
    print("  /languages         列出所有支持的语言")
    print("  /voice <人名> <语言> <文本> ")
    print("                     使用指定 ID 合成一段语音并设为当前音色锚点")
    print("  /load <路径>       从 JSON 文件载入音色锚点")
    print("  /save <路径>       将当前音色锚点保存到 JSON 文件")
    print("-" * 10)
    print("  /info              查看当前推理配置与音色信息")
    print("  /temp <值>         设置采样温度 (0.1 - 2.0)")
    print("  /chunk <值>        设置分块大小 (5 - 100)")
    print("  /steps <值>        设置步数上限 (50 - 1000)")
    print("  /reset             手动清空推理记忆")
    print("  /help              显示此帮助")
    print("  /exit, /q          退出程序")
    print("-" * 40)

def interactive_session():
    print("\n" + "="*60)
    print("🎤 Qwen3-TTS 交互式流式合成终端")
    print("="*60)

    # 1. 引擎初始化
    engine = TTSEngine(verbose=True)
    
    # 2. 初始音色加载 (尝试加载 vivian)
    JSON_PATH = "output/vivian.json"
    if os.path.exists(JSON_PATH):
        print(f"\n✅ 自动加载默认音色: {JSON_PATH}")
        stream = engine.create_stream(voice_path=JSON_PATH)
    else:
        stream = engine.create_stream()
        print(f"\nℹ️ 未设置初始音色，您可以输入文字进行原生生成，或使用 /voice 指令设定。")

    # 3. 推理配置
    cfg = TTSConfig(stream_play=True, mouth_chunk_size=12, max_steps=300)

    print_help()

    try:
        while True:
            raw_input = input("\n👉 请输入文本或指令: ").strip()
            
            if not raw_input:
                continue
            
            # --- 指令处理逻辑 ---
            if raw_input.startswith('/'):
                parts = raw_input.split(maxsplit=3)
                cmd = parts[0].lower()
                
                if cmd == '/help':
                    print_help()
                elif cmd == '/info':
                    print("\n[当前状态]")
                    print(f"  - 采样温度: {cfg.temperature}")
                    print(f"  - 分块大小: {cfg.mouth_chunk_size}")
                    print(f"  - 步数上限: {cfg.max_steps}")
                    print(f"  - 当前音色: {stream.voice.info if stream.voice else '未设置'}")
                elif cmd == '/speakers':
                    print("\n🎙️ 可用说话人列表:")
                    for idx, (name, sid) in enumerate(SPEAKER_MAP.items()):
                        print(f"  {idx+1:2}. {name:15} (ID: {sid})")
                elif cmd == '/languages':
                    print("\n🌏 支持语言列表:")
                    for idx, (lang, lid) in enumerate(LANGUAGE_MAP.items()):
                        print(f"  {idx+1:2}. {lang:15} (ID: {lid})")
                elif cmd == '/voice':
                    if len(parts) < 4:
                        print("❌ 用法: /voice <说话人> <语言> <文本>")
                        continue
                    spk, lang, v_text = parts[1], parts[2], parts[3]
                    print(f"⌛ 正在使用 [{spk}] 的 [{lang}] 语音合成音色锚点...")
                    # 现在直接使用当前 stream_play 配置，设定的同时即可听到预览
                    res = stream.set_voice_from_speaker(spk, v_text, language=lang, config=cfg, verbose=True)
                    print(f"✅ 音色已设定并激活。")
                elif cmd == '/load':
                    if len(parts) < 2:
                        print("❌ 用法: /load <路径>")
                        continue
                    p = parts[1]
                    if os.path.exists(p):
                        stream.set_voice_from_json(p)
                        print(f"✅ 已从 {p} 加载音色。")
                    else: print(f"❌ 找不到文件: {p}")
                elif cmd == '/save':
                    if not stream.voice:
                        print("❌ 当前尚未设置音色，无法保存。")
                        continue
                    p = parts[1] if len(parts) > 1 else f"output/saved_voice_{int(time.time())}.json"
                    stream.voice.save_json(p)
                    print(f"✅ 已保存至: {p}")
                elif cmd == '/temp' and len(parts) > 1:
                    cfg.temperature = float(parts[1])
                    print(f"✅ 温度: {cfg.temperature}")
                elif cmd == '/chunk' and len(parts) > 1:
                    cfg.mouth_chunk_size = int(parts[1])
                    print(f"✅ 分块: {cfg.mouth_chunk_size}")
                elif cmd == '/steps' and len(parts) > 1:
                    cfg.max_steps = int(parts[1])
                    print(f"✅ 步数: {cfg.max_steps}")
                elif cmd == '/reset':
                    stream.master.clear_memory()
                    print("✅ 记忆已重置。")
                elif cmd in ['/exit', '/q', '/quit']:
                    break
                else:
                    print(f"❓ 未知指令: {cmd}。输入 /help 查看列表。")
                continue

            # --- 普通文本合成 ---
            if raw_input.lower() in ['exit', 'q', 'quit', '退出']:
                break
            
            print(f"🚀 正在合成...")
            t_0 = time.time()
            res = stream.tts(raw_input, config=cfg, verbose=True)
            print(f"\n✨ 完成! [时长: {res.duration:.2f}s | 耗时: {time.time()-t_0:.2f}s | RTF: {res.rtf:.2f}]")
            
    except KeyboardInterrupt:
        print("\n\n🛑 检测到中断信号，正在退出...")
    except Exception as e:
        print(f"\n❌ [Terminal] 错误: {e}")
    finally:
        print("⏳ 释放资源...")
        try: engine.shutdown()
        except: pass
        print("✅ 再见。")

if __name__ == "__main__":
    interactive_session()
