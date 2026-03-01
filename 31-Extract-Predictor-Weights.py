import os
import json
import torch
import numpy as np
from safetensors.torch import save_file, load_file

from export_config import MODEL_DIR, EXPORT_DIR

def extract_predictor_hf():
    model_path = os.path.join(MODEL_DIR, "model.safetensors")
    output_dir = os.path.join(EXPORT_DIR, "predictor_hf")
    os.makedirs(output_dir, exist_ok=True)
    
    embedding_output_dir = os.path.join(EXPORT_DIR, 'embeddings')
    os.makedirs(embedding_output_dir, exist_ok=True)

    print(f"--- 正在提取 Predictor 权重 ---")
    print(f"源文件: {model_path}")
    
    weights = load_file(model_path)
    # weights = {k: v for k, v in weights.items()}
    
    # 动态检测投影层 (1.7B 专用)
    proj_key_w = "talker.code_predictor.small_to_mtp_projection.weight"
    proj_key_b = "talker.code_predictor.small_to_mtp_projection.bias"
    has_projection = proj_key_w in weights
    
    if has_projection:
        print("💡 检测到投影层 (1.7B 逻辑)")
        proj_w = weights[proj_key_w]
        proj_b = weights[proj_key_b]
    else:
        print("💡 未检测到投影层 (0.6B 逻辑)")
        proj_w = None
        proj_b = None
    
    # 1. 处理 Embedding (自动搜索 Index)
    emb_list = []
    i = 0
    while True:
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key not in weights:
            break
        
        raw_emb = weights[key]
        if has_projection:
            # 1.7B: 执行预投影
            proj_emb = torch.nn.functional.linear(raw_emb, proj_w, proj_b)
            emb_list.append(proj_emb)
        else:
            # 0.6B: 直接加入
            emb_list.append(raw_emb)
        i += 1
        
    if not emb_list:
        print("❌ 未在权重中找到 Codec Embedding！")
        return
        
    cat_emb = torch.cat(emb_list, dim=0) 
    print(f"   拼接后的 Embedding 形状: {cat_emb.shape} (共 {i} 组)")
    
    # 2. 处理 LM Head
    head_list = []
    j = 0
    while True:
        key = f"talker.code_predictor.lm_head.{j}.weight"
        if key not in weights:
            break
        head_list.append(weights[key])
        j += 1
    
    if not head_list:
        print("❌ 未在权重中找到 LM Head！")
        return
        
    cat_head = torch.cat(head_list, dim=0) 
    print(f"   拼接后的 LM Head 形状: {cat_head.shape} (共 {j} 组)")
    
    # 3. 提取 Backbone Layers
    new_weights = {
        "embed_tokens.weight": cat_emb,
        "lm_head.weight": cat_head,
        "norm.weight": weights["talker.code_predictor.model.norm.weight"]
    }
    
    layer_count = 0
    while True:
        prefix = f"talker.code_predictor.model.layers.{layer_count}."
        # 探测该层是否存在
        found_layer = False
        for k in weights.keys():
            if k.startswith(prefix):
                found_layer = True
                new_key = k.replace(prefix, f"layers.{layer_count}.")
                new_weights[new_key] = weights[k]
        
        if not found_layer:
            break
        layer_count += 1
        
    print(f"   已导出 Transformer 层数: {layer_count}")

    # 保存 Safetensors
    save_file(new_weights, os.path.join(output_dir, "model.safetensors"))
    print(f"   权重保存完成: {os.path.join(output_dir, 'model.safetensors')}")
    
    # 4. 自动化生成 Config (根据权重探测结果)
    hidden_size = cat_emb.shape[1]
    vocab_size = cat_emb.shape[0]
    
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 3, # Predictor 通常是 3x
        "num_hidden_layers": layer_count,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "vocab_size": vocab_size, 
        "rope_theta": 1000000.0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "max_position_embeddings": 32768
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   配置生成完成: {config_path}")
        
    # 5. 推理资产导出 (如果是 1.7B 带投影，还要额外保存给 GGUF 外的资产使用)
    if has_projection:
        np.save(os.path.join(embedding_output_dir, "proj_weight.npy"), proj_w.float().numpy())
        np.save(os.path.join(embedding_output_dir, "proj_bias.npy"), proj_b.float().numpy())
        print(f"📊 额外导出 Numpy 投影层资产至: {embedding_output_dir}")
        
    print(f"✅ Predictor HF 格式剥离完成: {output_dir}")

if __name__ == "__main__":
    extract_predictor_hf()
