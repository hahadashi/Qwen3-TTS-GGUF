"""
分析大师模型的输入输出维度
"""
import json
import torch
from safetensors import safe_open

# 加载配置
with open('Standalone-Bare-Master/config.json', 'r') as f:
    config = json.load(f)

print("="*60)
print("大师模型维度分析")
print("="*60)

print("\n【输入维度】")
print(f"1. Codec Embedding 词表大小: {config['vocab_size']} = 3072")
print(f"   - 输入的 token ID 范围: 0 ~ {config['vocab_size'] - 1}")
print(f"   - Codec token 最大 ID: 3071")

print(f"\n2. Text Embedding 词表大小: {config['text_vocab_size']} = 151936")
print(f"   - Text token 最大 ID: 151935")

print(f"\n3. Embedding 输出维度 (hidden_size): {config['hidden_size']} = 2048")
print(f"   - 无论是 codec 还是 text token，embed 后都是 2048 维")

print("\n【模型内部】")
print(f"1. Transformer 层数: {config['num_hidden_layers']} = 28 层")
print(f"2. 注意力头数: {config['num_attention_heads']} = 16")
print(f"3. KV 头数: {config['num_key_value_heads']} = 8 (GQA)")
print(f"4. Head 维度: {config['head_dim']} = 128")
print(f"5. FFN 中间层维度: {config['intermediate_size']} = 6144")

print("\n【输出维度】")
print(f"1. 隐藏层输出: [batch, seq_len, 2048]")
print(f"2. Codec Head: Linear(2048 -> 3072)")
print(f"   - 输入: 2048 维")
print(f"   - 输出: 3072 维 (logits)")
print(f"   - 生成的 token ID 范围: 0 ~ 3071")

# 验证 codec_head 权重
with safe_open('Standalone-Bare-Master/codec_head.safetensors', framework='pt', device='cpu') as f:
    weight = f.get_tensor('weight')
    print(f"\n【Codec Head 权重验证】")
    print(f"权重形状: {weight.shape}")
    print(f"  - 期望: [vocab_size, hidden_size] = [3072, 2048]")
    print(f"  - 实际: [{weight.shape[0]}, {weight.shape[1]}]")
    if weight.shape == torch.Size([3072, 2048]):
        print(f"  [OK] 形状正确!")
    else:
        print(f"  [ERROR] 形状不匹配!")

print("\n【数据流总结】")
print("输入 (token ID: 0~3071)")
print("  ↓")
print("Codec Embedding (3072 -> 2048)")
print("  ↓")
print("28x Transformer Layers (保持 2048 维)")
print("  ↓")
print("输出 Hidden State [batch, seq, 2048]")
print("  ↓")
print("取最后一个 token [batch, 2048]")
print("  ↓")
print("Codec Head Linear (2048 -> 3072)")
print("  ↓")
print("输出 Logits [batch, 3072]")
print("  ↓")
print("Argmax -> 下一个 token ID (0~3071)")
print("="*60)
