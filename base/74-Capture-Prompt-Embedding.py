import os
import sys
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_prompt_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

class CaptureState:
    def __init__(self):
        self.captured = False

state = CaptureState()

def talker_model_forward_hook(module, input, output):
    """
    捕获送入 Talker (LLM Master) 的 inputs_embeds
    Qwen3TTSTalkerModel.forward(self, input_ids, ..., inputs_embeds, ...)
    """
    if state.captured:
        return
    
    # 检查 kwargs 中的 inputs_embeds
    # 注意：在 generate 的 prefill 阶段，这里会传入完整的 prompt embedding
    # 如果是 forward 挂载，input 是个 tuple: (input_ids, attention_mask, ...)
    # 我们更倾向于直接捕获输入中的 inputs_embeds
    
    # 因为 forward 签名复杂，我们可以通过 inspect 或者简单的索引尝试
    # 实际上，inputs_embeds 通常是作为 keyword argument 传入的，或者在 input tuple 的特定位置
    pass

# 更直接的方法：在 Qwen3TTSTalkerModel.forward 入口截获
# 或者使用 Monkey Patch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-Base")
    REF_AUDIO = os.path.abspath("output/sample.wav")
    
    if not os.path.exists(REF_AUDIO):
        print(f"❌ 找不到参考音频: {REF_AUDIO}")
        return

    print(f"载入 Base 模型以捕获 Prompt Embedding (设备: {device})...")
    
    try:
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            device_map=device, 
            torch_dtype=torch.float32 # 统一精度
        )
        
        talker_model = tts.model.talker.model
        original_forward = talker_model.forward
        
        def patched_forward(*args, **kwargs):
            if not state.captured:
                # 提取 inputs_embeds
                inputs_embeds = kwargs.get("inputs_embeds", None)
                if inputs_embeds is None and len(args) > 4:
                    # 尝试从位置参数获取 (按照建模文件的定义顺序)
                    # forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, ...)
                    inputs_embeds = args[4]
                
                if inputs_embeds is not None:
                    # 捕获第一个遇到的（即 prefill 的完整 prompt）
                    data = inputs_embeds.detach().to(torch.float32).cpu().numpy()
                    np.save(os.path.join(SAVE_DIR, "prompt_inputs_embeds.npy"), data)
                    print(f"[CAPTURE] Talker inputs_embeds captured! Shape: {data.shape}")
                    state.captured = True
            
            return original_forward(*args, **kwargs)
            
        talker_model.forward = patched_forward
        print("✅ 已对 tts.model.talker.model.forward 应用 Monkey Patch")

        # 执行推理
        text = "你好！这是捕获官方提示词嵌入的测试。"
        ref_text = "你好，我是具有随机性的千问3-TTS，这是我的终极进化形态"
        
        print(f"开始推理...")
        with torch.no_grad():
            tts.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=REF_AUDIO,
                ref_text=ref_text,
                do_sample=False
            )
        
        print(f"\n✅ 数据捕获完成！")
        print(f"文件保存在: {SAVE_DIR}")
        
    except Exception as e:
        print(f"❌ 捕获失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
