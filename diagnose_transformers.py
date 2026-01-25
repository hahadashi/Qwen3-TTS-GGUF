import inspect
import transformers
from transformers import modeling_utils

print(f"Transformers version: {transformers.__version__}")

# Check 1: modeling_attn_mask_utils
try:
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    print("\nFound `_create_4d_causal_attention_mask` in `transformers.modeling_attn_mask_utils`")
    print(f"Signature: {inspect.signature(_create_4d_causal_attention_mask)}")
except ImportError:
    print("\n`_create_4d_causal_attention_mask` NOT found in `transformers.modeling_attn_mask_utils`")

try:
    from transformers.modeling_attn_mask_utils import create_causal_mask
    print("\nFound `create_causal_mask` in `transformers.modeling_attn_mask_utils`")
except ImportError:
    print("\n`create_causal_mask` NOT found in `transformers.modeling_attn_mask_utils`")


# Check 2: generation.masking_utils (Old location?)
try:
    from transformers.generation.masking_utils import create_causal_mask
    print("\nFound `create_causal_mask` in `transformers.generation.masking_utils`")
except ImportError:
    print("\n`create_causal_mask` NOT found in `transformers.generation.masking_utils`")

# Check 3: Check modeling_utils
print(f"\nScanning `transformers.modeling_utils` for 'mask' functions:")
for name in dir(modeling_utils):
    if 'mask' in name.lower():
        print(f" - {name}")

