
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add source path
PROJECT_ROOT = Path(__file__).parent
SOURCE_DIR = PROJECT_ROOT / "Qwen3-TTS"
sys.path.append(str(SOURCE_DIR))

try:
    import qwen_tts.core.models.modeling_qwen3_tts as modeling_mod
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

# Configuration
MODEL_PATH = PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = PROJECT_ROOT / "model"
ONNX_PATH = OUTPUT_DIR / "Qwen3-Code-Predictor.onnx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [Patch] Force bypass Cache type check
modeling_mod.Cache = object
try:
    import transformers.cache_utils as cache_utils
    cache_utils.Cache = object
except:
    pass

class JITCache:
    """A Cache-like object that JIT can handle during tracing."""
    def __init__(self, past_list):
        self.past_key_values = past_list
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        old_k, old_v = self.past_key_values[layer_idx]
        new_k = torch.cat([old_k, key_states], dim=2)
        new_v = torch.cat([old_v, value_states], dim=2)
        self.past_key_values[layer_idx] = (new_k, new_v)
        return new_k, new_v
    def get_seq_length(self, layer_idx=0):
        return self.past_key_values[0][0].shape[2]

class CodePredictorOnnxWrapper(torch.nn.Module):
    def __init__(self, predictor_model):
        super().__init__()
        self.predictor = predictor_model
        self.predictor.config.return_dict = False
        if hasattr(self.predictor, 'model'):
            self.predictor.model.config.return_dict = False

    def forward(self, inputs_embeds, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        # 1. Reconstruct Cache
        past_list = [(p0, p1), (p2, p3), (p4, p5), (p6, p7), (p8, p9)]
        past_key_values = JITCache(past_list)
        
        # 2. Projection
        hidden_states = self.predictor.small_to_mtp_projection(inputs_embeds)
        
        # 3. Cache Position
        past_len = p0.shape[2]
        current_len = inputs_embeds.shape[1]
        cache_position = torch.arange(past_len, past_len + current_len, device=inputs_embeds.device)
        
        # 4. Mask
        batch_size = inputs_embeds.shape[0]
        total_len = past_len + current_len
        attention_mask_dict = {
            "full_attention": torch.zeros((batch_size, 1, current_len, total_len), 
                                         dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        }
        
        # 5. Forward
        outputs = self.predictor.model(
            inputs_embeds=hidden_states,
            past_key_values=past_key_values, # Model calls .update() on our JITCache
            use_cache=True,
            return_dict=False,
            cache_position=cache_position,
            attention_mask=attention_mask_dict
        )
        
        last_hidden_state = outputs[0]
        # output[1] here is our JITCache object after updates
        
        # 6. Flatten
        present_flat = []
        for k, v in past_key_values.past_key_values:
            present_flat.append(k)
            present_flat.append(v)
            
        return (last_hidden_state, *present_flat)

def export_code_predictor():
    print(f"[1/4] Loading model...")
    full_model = Qwen3TTSForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")
    predictor = full_model.talker.code_predictor
    
    print("[2/4] Saving Heads...")
    heads_weights = [head.weight.detach().numpy() for head in predictor.lm_head]
    np.save(OUTPUT_DIR / "code_predictor_heads.npy", np.stack(heads_weights))
    
    print("[3/4] Preparing Wrapper...")
    wrapper = CodePredictorOnnxWrapper(predictor).eval()
    
    num_kv_heads = 8
    head_dim = 128
    dummy_input = torch.randn(1, 1, 2048)
    dummy_past = [torch.randn(1, num_kv_heads, 0, head_dim) for _ in range(10)]
    
    input_names = ["inputs_embeds"] + [f"past_{i}" for i in range(10)]
    output_names = ["hidden_states"] + [f"present_{i}" for i in range(10)]
    dynamic_axes = {"inputs_embeds": {0: "batch", 1: "seq"}}
    for i in range(10):
        dynamic_axes[f"past_{i}"] = {0: "batch", 2: "past_seq"}

    print(f"[4/4] Exporting to {ONNX_PATH}...")
    try:
        torch.onnx.export(
            wrapper,
            (dummy_input, *dummy_past),
            str(ONNX_PATH),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False
        )
        print("✅ Success!")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    export_code_predictor()
