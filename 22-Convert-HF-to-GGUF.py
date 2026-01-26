import os
import subprocess
import sys

# Configuration
CONVERT_SCRIPT = r"c:\Users\Haujet\Desktop\qwen3-tts\qwen3_tts_gguf\convert_hf_to_gguf.py"
MODEL_DIR = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-HF"
GGUF_OUT = r"c:\Users\Haujet\Desktop\qwen3-tts\model\Qwen3-Talker-F16.gguf"

def convert_to_gguf():
    print(f"--- Converting {MODEL_DIR} to GGUF ---")
    
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: Could not find llama.cpp conversion script at {CONVERT_SCRIPT}")
        return

    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        MODEL_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    # We may need --verbose if it fails, but for now we keep it simple
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # We use run and check status
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Standard Output:")
        print(result.stdout)
        print("\n✅ Success! GGUF saved to:", GGUF_OUT)
    except subprocess.CalledProcessError as e:
        print("\n❌ Conversion failed!")
        print("Error Output:")
        print(e.stderr)
        print(e.stdout)

if __name__ == "__main__":
    convert_to_gguf()
