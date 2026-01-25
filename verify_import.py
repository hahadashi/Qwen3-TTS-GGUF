try:
    from transformers.masking_utils import create_causal_mask
    print("SUCCESS: Imported create_causal_mask from transformers.masking_utils")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
