import os
import sys
import torch
import numpy as np
import collections
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_full_gen")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

class CaptureState:
    def __init__(self):
        self.master_steps = 0
        self.generated_code_sequences = [] # List of [16] codes per master step
        self.master_hidden_states = []
        self.captured_head = False

state = CaptureState()

def talker_forward_post_hook(module, input, output):
    """
    Hook Talker forward to capture generation steps.
    """
    # output.hidden_states[1] contains (input_ids, sequences) -> [B, 16]
    # Check if it is generation phase
    if output.hidden_states is not None:
         # Capture Codec IDs
         codec_ids = output.hidden_states[1]
         if codec_ids is not None:
             codes = codec_ids.detach().cpu().numpy()
             state.generated_code_sequences.append(codes)
             
             # Capture Master Output (Hidden State for Next Step)
             # output.past_hidden -> [B, 1, Hidden]
             if output.past_hidden is not None:
                 hid = output.past_hidden.detach().cpu().to(torch.float32).numpy()
                 state.master_hidden_states.append(hid)
                 
             state.master_steps += 1

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行全量生成捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        talker = tts.model.talker
        
        # 1. Export Legacy Head (if not tied)
        # We need to check if talker.codec_head is tied to talker.model.codec_embedding
        codec_head_weight = talker.codec_head.weight
        codec_embed_weight = talker.model.codec_embedding.weight
        
        is_tied = torch.equal(codec_head_weight, codec_embed_weight)
        print(f"Checking Weight Tying: Codec Head vs Codec Embedding -> {is_tied}")
        
        if not is_tied:
            print("⚠️ Weights are NOT TIED. Exporting Codec Head...")
            np.save(os.path.join(MODEL_DIR, "codec_head_weight.npy"), codec_head_weight.detach().cpu().to(torch.float32).numpy())
        else:
            print("✅ Weights are TIED. We can use codec_embedding_0.npy as head.")

        # 2. Hook for Generation
        talker.register_forward_hook(talker_forward_post_hook)
        
        # 3. Run Generation
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        print("开始推理「今天天气不错」...")
        with torch.no_grad():
            audio = tts.generate_custom_voice(
                text="今天天气不错",
                speaker="Vivian",
                language="Chinese",
                **deterministic_kwargs
            )
            
        print(f"\n生成完成。捕获了 {len(state.generated_code_sequences)} 个 Master 步骤。")
        
        if len(state.generated_code_sequences) > 0:
            final_codes = np.concatenate(state.generated_code_sequences, axis=0)
            np.save(os.path.join(SAVE_DIR, "full_generated_codes.npy"), final_codes)
            print(f"✅ 全量 Codes 已保存: {SAVE_DIR}\\full_generated_codes.npy")
            
            # Save Hidden States
            if len(state.master_hidden_states) > 0:
                # Stack them: [Steps, 1, Hidden]
                final_hiddens = np.concatenate(state.master_hidden_states, axis=0)
                np.save(os.path.join(SAVE_DIR, "full_master_hidden_states.npy"), final_hiddens)
                print(f"✅ Master Hidden States 已保存: {SAVE_DIR}\\full_master_hidden_states.npy (Shape: {final_hiddens.shape})")
            
            # Print first few for verification
            print(f"First 2 steps codes: \n{final_codes[:2]}")
            
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
