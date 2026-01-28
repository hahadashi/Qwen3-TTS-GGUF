import os
import torch
import numpy as np
from transformers import AutoTokenizer

# 1. 配置与路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")

# 核心权重表
TEXT_TABLE_PATH = os.path.join(MODEL_DIR, "text_embedding_projected.npy")
CODEC_TABLE_0_PATH = os.path.join(MODEL_DIR, "codec_embedding_0.npy")

# 常量定义
C_PAD = 2148
C_THINK = 2154
C_TBOS = 2156
C_CHINESE = 2055
C_TEOS = 2157
C_VIVIAN = 3065
C_AUDIO_BOS = 2148 
C_TTS_EOS = 2148
C_CODEC_BOS = 2149

T_IM_START = 151644
T_ASSISTANT = 77091
T_NL = 198
T_THINK_T = 151671 # 'Thought' token
T_AUDIO_BOS = 151672
T_TTS_EOS = 151673

def construct_dynamic_input(text, speaker_id=C_VIVIAN):
    print(f"--- 正在为文本 '{text}' 动态构造输入 Embedding ---")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    content_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Tokenizer IDs: {content_ids}")
    
    # 构造 ID 序列 (Text_ID, Codec0_ID)
    # 按照 48 号脚本总结的结构
    sequence = [
        (T_IM_START, 0),         # 0: <|im_start|>
        (T_ASSISTANT, 0),        # 1: assistant
        (T_NL, 0),               # 2: \n
        (T_THINK_T, C_THINK),    # 3: Think
        (T_THINK_T, C_TBOS),     # 4: Think_BOS
        (T_THINK_T, C_CHINESE),  # 5: Lang (Chinese)
        (T_THINK_T, C_TEOS),     # 6: Think_EOS
        (T_THINK_T, speaker_id), # 7: Speaker
        (T_AUDIO_BOS, C_AUDIO_BOS),# 8: Audio_BOS
    ]
    
    # 注入文本内容
    for tid in content_ids:
        sequence.append((tid, C_PAD))
        
    # 封口
    sequence.append((T_TTS_EOS, C_TTS_EOS))   # TTS_EOS
    sequence.append((T_THINK_T, C_CODEC_BOS)) # Codec_BOS
    
    print(f"  构造序列总长度: {len(sequence)}")
    
    # 加载表并相加
    text_table = np.load(TEXT_TABLE_PATH)
    codec_table = np.load(CODEC_TABLE_0_PATH)
    
    embed_list = []
    for t_id, c_id in sequence:
        t_vec = text_table[t_id]
        c_vec = codec_table[c_id] if c_id != 0 else np.zeros(2048, dtype=np.float32)
        embed_list.append(t_vec + c_vec)
        
    constructed_embeds = np.array(embed_list).reshape(1, len(sequence), 2048)
    return constructed_embeds

def verify_against_official(constructed):
    print("\n--- 正在与官方捕获数据进行精度对比 ---")
    official_path = os.path.join(CAPTURED_DIR, "prompt_inputs_embeds.npy")
    if not os.path.exists(official_path):
        print("❌ 找不到官方捕获数据进行对比。")
        return
        
    official = np.load(official_path)
    print(f"  官方维度: {official.shape}")
    print(f"  我的维度: {constructed.shape}")
    
    if official.shape != constructed.shape:
        print("❌ 维度不匹配，请检查文本是否一致（官方默认: '今天天气不错'）。")
        return
        
    mae = np.mean(np.abs(official - constructed))
    
    # 余弦相似度
    off_flat = official.flatten()
    my_flat = constructed.flatten()
    cos_sim = np.dot(off_flat, my_flat) / (np.linalg.norm(off_flat) * np.linalg.norm(my_flat) + 1e-9)
    
    print(f"  MAE: {mae:.8f}")
    print(f"  CosSim: {cos_sim:.8f}")
    
    if cos_sim > 0.9999:
        print("✅ 完美对齐！动态输入构造逻辑正确。")
    else:
        print("❌ 对齐精度不足，请检查 ID 映射。")

if __name__ == "__main__":
    # 官方捕获数据对应的文本是 "今天天气不错"
    my_embeds = construct_dynamic_input("今天天气不错")
    verify_against_official(my_embeds)
