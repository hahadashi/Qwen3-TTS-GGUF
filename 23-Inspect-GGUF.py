import sys
import os

# Try to find gguf package from llama.cpp
sys.path.append(r"c:\Users\Haujet\Desktop\qwen3-tts\ref\llama.cpp")
try:
    import gguf
except ImportError:
    print("Error: Could not import gguf package. Ensure ref/llama.cpp/gguf-py is in path.")
    sys.exit(1)

GGUF_PATH = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-F16.gguf"

def inspect_gguf():
    if not os.path.exists(GGUF_PATH):
        print(f"Error: GGUF file not found at {GGUF_PATH}")
        return

    print(f"--- Inspecting GGUF: {GGUF_PATH} ---")
    reader = gguf.GGUFReader(GGUF_PATH)
    
    print("\n[1] Metadata:")
    for key in reader.fields:
        field = reader.fields[key]
        if field.types and field.types[0] == gguf.GGUFValueType.STRING:
             val = str(field.parts[-1].tobytes().decode('utf-8'))
        elif field.types and field.types[0] == gguf.GGUFValueType.ARRAY:
             # For arrays, we might need to look at all parts or the correct part
             val = [x for x in field.parts[-1].tolist()] if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
        else:
             val = field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
        print(f"  {key}: {val}")
        
    print("\n[2] Tensors (First 10):")
    for i, tensor in enumerate(reader.tensors):
        if i >= 10: break
        print(f"  {tensor.name}: {tensor.shape}, {tensor.tensor_type}")
    
    print(f"\nTotal Tensors: {len(reader.tensors)}")
    print("\n✅ GGUF inspection complete.")

if __name__ == "__main__":
    inspect_gguf()
