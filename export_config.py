"""
export_config.py - Qwen3-TTS 导出工具链全局配置
只需在此修改路径，所有 1x-3x 脚本将自动同步。
"""
from pathlib import Path
model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()


# [源模型路径] 官方下载好的 SafeTensors 模型文件夹
MODEL_BASE   =  model_home / 'Qwen3-TTS-12Hz-1.7B-Base'
MODEL_CUSTOM =  model_home / 'Qwen3-TTS-12Hz-1.7B-CustomVoice'
MODEL_DESIGN =  model_home / 'Qwen3-TTS-12Hz-1.7B-VoiceDesign'

MODEL_DIR =  MODEL_BASE

# [导出目标路径] 转换后的 ONNX, GGUF 和权重汇总目录
EXPORT_DIR = r'./model-base'
# EXPORT_DIR = r'./model-custom'
# EXPORT_DIR = r'./model-design'
