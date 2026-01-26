import os
import torch
import soundfile as sf
import time
from qwen_tts import Qwen3TTSModel

# 配置：模型路径和输出文件
MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
EMBEDDING_FILE = "31_intercepted_embedding.pt"
OUTPUT_WAV_ORIGINAL = "31_output_original.wav"
OUTPUT_WAV_INJECTED = "31_output_injected.wav"

def main():
    # 自动选择设备 (CUDA 或 CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 自动选择数据类型 (CPU不支持bfloat16则用float32)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print("正在加载模型...")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=dtype,
    )

    # 保存原始的 generate 函数，方便后续恢复或调用
    original_generate = tts.model.talker.generate

    # 1. 第一阶段：拦截并保存 Embedding (Intercept and Save)
    print("\n--- 第一阶段：拦截与保存 ---")
    
    # 定义拦截钩子函数
    def save_hook(*args, **kwargs):
        # 尝试从参数中获取 inputs_embeds
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is not None:
            print(f"  [Hook] 成功拦截 inputs_embeds，形状为: {inputs_embeds.shape}")
            # 保存到文件
            torch.save(inputs_embeds, EMBEDDING_FILE)
            print(f"  [Hook] 已保存到 {EMBEDDING_FILE}")
        
        # 继续执行原始的 generate，保证流程不中断
        return original_generate(*args, **kwargs)

    # 挂载钩子：替换模型原本的 generate 方法
    tts.model.talker.generate = save_hook
    
    # 运行一次推理
    print("  [Phase 1] 开始执行正常推理...")
    wavs, sr = tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        instruct="",
    )
    # 保存原始音频，用作对比
    sf.write(OUTPUT_WAV_ORIGINAL, wavs[0], sr)
    print(f"  [Phase 1] 原始音频已保存到 {OUTPUT_WAV_ORIGINAL}")


    # 2. 第二阶段：读取并注入 Embedding (Load and Inject)
    print("\n--- 第二阶段：读取与注入 ---")
    
    # 检查文件是否存在
    if not os.path.exists(EMBEDDING_FILE):
        print("错误: 找不到保存的 Embedding 文件！")
        return

    # 定义注入钩子函数
    def inject_hook(*args, **kwargs):
        print(f"  [Hook] 正在从 {EMBEDDING_FILE} 加载 Embedding...")
        loaded_embeds = torch.load(EMBEDDING_FILE, map_location=device)
        
        # 形状检查 (仅供调试查看)
        current_embeds = kwargs.get('inputs_embeds')
        if current_embeds is not None:
             print(f"  [Hook] 原始计算出的形状: {current_embeds.shape}, 读取的形状: {loaded_embeds.shape}")
        
        # 核心操作：偷梁换柱！
        # 将参数里的 inputs_embeds 强制替换成我们保存的那个
        kwargs['inputs_embeds'] = loaded_embeds.to(dtype).to(device)
        print("  [Hook] 已将 inputs_embeds 替换为加载的数据，忽略本次的文本输入！")
        
        # 用伪造的数据去调用原始生成
        return original_generate(*args, **kwargs)

    # 挂载注入钩子
    tts.model.talker.generate = inject_hook

    # 再次运行推理 
    # 注意：虽然这里输入了 text="今天天气好"，但实际上在 hook 里会被我们注入的 Embedding 覆盖。
    # 这里的输入主要是为了让代码流程能跑通，进入到 generate 函数。
    print("  [Phase 2] 开始执行注入推理...")
    wavs_injected, sr = tts.generate_custom_voice(
        text="今天很不好", 
        speaker="Vivian",
        instruct="",
    )
    sf.write(OUTPUT_WAV_INJECTED, wavs_injected[0], sr)
    print(f"  [Phase 2] 注入后的音频已保存到 {OUTPUT_WAV_INJECTED}")

    # 恢复环境 (把 generate 函数还回去)
    tts.model.talker.generate = original_generate

    print("\n验证完成！")
    print(f"请对比收听 {OUTPUT_WAV_INJECTED} 和 {OUTPUT_WAV_ORIGINAL}，它们应该听起来完全一样。")

if __name__ == "__main__":
    main()
