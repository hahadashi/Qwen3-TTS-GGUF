import os
import shutil
import json
import struct
import numpy as np

# =========================================================================
# 配置部分
# =========================================================================
SOURCE_MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
OUTPUT_HF_DIR = r'./model/Talker-HF-Temp'
MODEL_FILENAME = 'model.safetensors'

def streaming_extract_talker(source_dir, output_dir):
    source_path = os.path.join(source_dir, MODEL_FILENAME)
    config_path = os.path.join(source_dir, "config.json")
    
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found.")
        return False

    print(f"[*] Starting streaming extraction from {source_path}...")
    
    with open(source_path, "rb") as f_src:
        # 1. 读取 Safetensors Header
        h_size_bytes = f_src.read(8)
        h_size = struct.unpack("<Q", h_size_bytes)[0]
        header = json.loads(f_src.read(h_size).decode("utf-8"))
        
        full_keys = header.keys()
        
        # 2. 识别前缀并规划映射
        # 我们要提取 talker.model -> model, talker.codec_head -> lm_head
        prefix = "talker."
        tasks = []
        for key in sorted(full_keys):
            if not key.startswith(prefix):
                continue
            
            new_key = key[len(prefix):]
            final_key = None
            
            # 基础映射逻辑
            if new_key == "model.text_embedding.weight":
                final_key = "model.embed_tokens.weight"
            elif new_key == "codec_head.weight":
                final_key = "lm_head.weight"
            elif new_key.startswith("model.layers.") or new_key == "model.norm.weight":
                final_key = new_key
            
            if final_key:
                tasks.append((key, final_key, header[key]))

        if not tasks:
            print("Error: No talker weights found. Check if the prefix 'talker.' is correct.")
            return False

        print(f"[*] Identified {len(tasks)} tensors to extract.")

        # 3. 构造并保存伪装后的 Qwen3 Config
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        
        talker_config = full_config.get("talker_config", {})
        
        # 注入/覆盖关键参数，使用 Qwen3-VL 架构以支持 QK Norm 和 mROPE
        talker_config["architectures"] = ["Qwen3VLForConditionalGeneration"]
        talker_config["model_type"] = "qwen3_vl"
        # 词表统一设为 text_vocab_size (151936)，我们会对 lm_head 进行补齐
        VOCAB_SIZE = 151936
        talker_config["vocab_size"] = VOCAB_SIZE
        
        # 确保 mROPE 参数到位
        if "rope_scaling" not in talker_config:
             talker_config["rope_scaling"] = {
                 "mrope_section": [24, 20, 20],
                 "rope_type": "default",
                 "type": "default"
             }

        with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(talker_config, f, indent=2)
        print("[+] Config saved (spoofed as qwen3).")

        # 4. 复制 Tokenizer 文件
        for file in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
            src = os.path.join(source_dir, file)
            dst = os.path.join(output_dir, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
        print("[+] Tokenizer files copied.")

        # 5. 流式写入新 Safetensors
        target_file = os.path.join(output_dir, "model.safetensors")
        target_header = {"__metadata__": {"format": "pt"}}
        
        # 预计算 Offset
        current_offset = 0
        for src_key, final_key, info in tasks:
            shape = info["shape"]
            # 如果是 lm_head，需要补齐到 VOCAB_SIZE
            if final_key == "lm_head.weight":
                shape = [VOCAB_SIZE, shape[1]]
            
            num_elements = 1
            for s in shape: num_elements *= s
            size = num_elements * 4  # F32 = 4 bytes
            
            target_header[final_key] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + size]
            }
            current_offset += size
            
        header_json = json.dumps(target_header).encode("utf-8")
        header_padding = (8 - (len(header_json) % 8)) % 8
        header_json += b' ' * header_padding
        
        print(f"[*] Streaming weights to {target_file} (padding lm_head to {VOCAB_SIZE})...")
        with open(target_file, "wb") as f_dst:
            f_dst.write(struct.pack("<Q", len(header_json)))
            f_dst.write(header_json)
            
            for src_key, final_key, info in tasks:
                start_src = 8 + h_size + info["data_offsets"][0]
                end_src = 8 + h_size + info["data_offsets"][1]
                
                f_src.seek(start_src)
                raw_data = f_src.read(end_src - start_src)
                
                # 处理数据转换并写入
                dt = info["dtype"]
                if dt == "BF16":
                    arr_u16 = np.frombuffer(raw_data, dtype=np.uint16)
                    arr_f32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
                elif dt == "F16":
                    arr_f32 = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
                else:
                    arr_f32 = np.frombuffer(raw_data, dtype=np.float32)
                
                if final_key == "lm_head.weight":
                    # 补齐逻辑
                    old_vocab_size = info["shape"][0]
                    hidden_size = info["shape"][1]
                    # 创建完整的补齐矩阵
                    full_arr = np.zeros((VOCAB_SIZE, hidden_size), dtype=np.float32)
                    full_arr[:old_vocab_size, :] = arr_f32.reshape(old_vocab_size, hidden_size)
                    f_dst.write(full_arr.tobytes())
                else:
                    f_dst.write(arr_f32.tobytes())
        
        print(f"\n[!] Success! Talker-HF-Temp is ready at {output_dir}")
        return True

if __name__ == "__main__":
    streaming_extract_talker(SOURCE_MODEL_DIR, OUTPUT_HF_DIR)
