# Master Model - Ready for GGUF Conversion

This directory contains the master model prepared for conversion to GGUF format using llama.cpp's convert_hf_to_gguf.py.

## Conversion Instructions

```bash
python convert_hf_to_gguf.py \
    --model . \
    --outfile ../qwen3-tts-master-f16.gguf \
    --outtype f16
```

For quantized versions:
```bash
python convert_hf_to_gguf.py \
    --model . \
    --outfile ../qwen3-tts-master-q4_k_m.gguf \
    --outtype q4_k_m
```

## Notes

- This model is disguised as Qwen3VLForConditionalGeneration to work with llama.cpp's converter
- The vision_config is empty (no visual components)
- MRoPE sections: [24, 20, 20]
- All weights are in model.safetensors

## Original Model

- Source: Standalone-Bare-Master
- Architecture: Qwen3-VL based (28 layers, 2048 hidden)
- Vocab: 3072 (codec tokens)
