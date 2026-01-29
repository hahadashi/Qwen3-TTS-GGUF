import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_encoder_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

def speaker_encoder_hook(module, input, output):
    """
    捕获 Speaker Encoder 的输出 (X-vector / Speaker Embedding)
    """
    # output 通常是 (B, D)
    file_name = os.path.join(SAVE_DIR, "speaker_embedding.npy")
    np.save(file_name, output.detach().cpu().to(torch.float32).numpy())
    print(f"[CAPTURE] Speaker Encoder: Saved output (Shape: {output.shape}) -> {file_name}")

class GlobalCounter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1

counter = GlobalCounter()

def speech_tokenizer_enc_hook(module, input, output):
    """
    捕获 Speech Tokenizer Encoder 的中间层输出
    """
    # 获取模块名称
    module_name = str(type(module).__name__)
    
    features = None
    if isinstance(output, torch.Tensor):
        features = output
    elif hasattr(output, 'last_hidden_state'):
        features = output.last_hidden_state
    elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        features = output[0]

    if features is not None:
        idx = counter.count
        file_name = os.path.join(SAVE_DIR, f"tokenizer_layer_{idx}_{module_name}.npy")
        np.save(file_name, features.detach().cpu().to(torch.float32).numpy())
        print(f"[CAPTURE] Tokenizer Layer {idx} ({module_name}): Saved features (Shape: {features.shape})")
        counter.increment()

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-Base")
    REF_AUDIO = os.path.abspath("output/sample.wav")
    
    if not os.path.exists(REF_AUDIO):
        print(f"❌ 找不到参考音频: {REF_AUDIO}")
        return

    print(f"载入 Base 模型进行插桩 (设备: {device})...")
    dtype = torch.float32 # 强制使用 float32 以便与 ONNX 对比
    
    try:
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            device_map=device, 
            dtype=dtype
        )
        
        # 1. 插装 Speaker Encoder
        # model.speaker_encoder
        if tts.model.speaker_encoder is not None:
            handle_spk = tts.model.speaker_encoder.register_forward_hook(speaker_encoder_hook)
            print("✅ 已为 Speaker Encoder 注册 Hook")
        else:
            print("⚠️ 模型不包含 Speaker Encoder (该 Base 模型可能不是预期的可克隆版本)")

        # 2. 插装 Speech Tokenizer Encoder
        # tts.model.speech_tokenizer 是 Qwen3TTSTokenizer
        # tts.model.speech_tokenizer.model 是 Qwen3TTSTokenizerV2Model
        # tts.model.speech_tokenizer.model.encoder 是 Qwen3TTSTokenizerV2Encoder (MimiModel)
        tokenizer_model = tts.model.speech_tokenizer.model
        # 2. 强力捕获最终 Code (Monkey Patch)
        original_encode = tts.model.speech_tokenizer.model.encode
        def patched_encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            # result 通常是 [audio_codes] 或 Qwen3TTSTokenizerV2EncoderOutput
            codes = None
            if hasattr(result, 'audio_codes'):
                codes = result.audio_codes[0].detach().cpu().numpy()
            elif isinstance(result, (list, tuple)):
                codes = result[0][0].detach().cpu().numpy()
            
            if codes is not None:
                np.save(os.path.join(SAVE_DIR, "tokenizer_audio_codes.npy"), codes)
                print(f"[CAPTURE] Patched Encode: Saved audio_codes (Shape: {codes.shape})")
            return result
            
        tts.model.speech_tokenizer.model.encode = patched_encode
        print("✅ 已对 Tokenizer.model.encode 应用 Monkey Patch")

        # 执行推理
        text = "你好！这是捕获音频编码器输出的测试。"
        ref_text = "你好，我是具有随机性的千问3-TTS，这是我的终极进化形态" # 示例参考文本
        
        print(f"开始推理: 「{text}」...")
        with torch.no_grad():
            tts.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=REF_AUDIO,
                ref_text=ref_text,
                do_sample=False # 固定结果便于分析
            )
        
        print(f"\n✅ 数据捕获完成！")
        print(f"文件保存在: {SAVE_DIR}")
        
    except Exception as e:
        print(f"❌ 捕获失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 移除 Hook
        if 'handle_spk' in locals(): handle_spk.remove()
        if 'handle_tok' in locals(): handle_tok.remove()

if __name__ == "__main__":
    main()
