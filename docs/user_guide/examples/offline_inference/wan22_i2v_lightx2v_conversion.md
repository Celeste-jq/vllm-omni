# Wan2.2 I2V LightX2V Conversion

Source <https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_video/wan22_i2v_lightx2v_conversion.md>.


This guide shows how to use LightX2V Wan2.2 LoRAs with vLLM-Omni by **offline converting and assembling** a Diffusers-style model directory.

As of **March 24, 2026**, this is the lowest-risk integration path on `upstream/main`.

## Recommended Route

Use this route:

1. Start from official base model `Wan-AI/Wan2.2-I2V-A14B`
2. Merge LightX2V LoRA into each DiT (`high_noise_model` / `low_noise_model`) with `converter.py`
3. Assemble a final Diffusers-style directory
4. Load that directory directly in vLLM-Omni

Why this route is recommended:

- No runtime changes in vLLM-Omni inference pipeline
- Keeps model loading in standard Diffusers layout
- Avoids extending current single-LoRA request protocol to a dual-LoRA pair protocol

## Prerequisites

Prepare these assets locally:

- Official base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- LightX2V LoRA repo: `lightx2v/Wan2.2-Distill-Loras`
- LightX2V converter script: `tools/convert/converter.py` from LightX2V project

## Step 1: Convert High/Low DiT Weights

Run LightX2V converter twice (high-noise and low-noise).

```bash
python converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

```bash
python converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

Expected converted files:

- `/tmp/wan22_lightx2v/high_noise_out/diffusion_pytorch_model.safetensors`
- `/tmp/wan22_lightx2v/low_noise_out/diffusion_pytorch_model.safetensors`

## Step 2: Assemble Diffusers Layout

Use the helper script added in this repository:

```bash
python tools/wan22/assemble_lightx2v_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --high-noise-weight /tmp/wan22_lightx2v/high_noise_out \
  --low-noise-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers \
  --asset-mode symlink \
  --overwrite
```

Notes:

- `--asset-mode symlink` avoids copying large tokenizer/text_encoder/vae assets.
- Use `--asset-mode copy` if you need a fully standalone directory.
- `--high-noise-weight` / `--low-noise-weight` accepts either a single weight file or a sharded directory with `*.index.json` + shard files.

## Step 3: Verify Final Directory

At minimum, ensure these paths exist:

```text
Wan2.2-I2V-A14B-LightX2V-Diffusers/
├── model_index.json
├── tokenizer/
├── text_encoder/
├── vae/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── transformer_2/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## Step 4: Load with vLLM-Omni

```bash
vllm serve /path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers --omni --port 8091
```

Or in Python:

```python
from vllm_omni import Omni

omni = Omni(model="/path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers")
```

## Do We Need to Modify vLLM-Omni Code?

For this offline route, usually **no runtime code changes** are required.

If you want to support **online direct loading** of LightX2V dual LoRAs (without offline merge), then code changes are needed in:

- Request schema and parser (`lora` must support high/low pair)
- Diffusion sampling params (single `lora_request` -> pair/group)
- LoRA activation in worker (activate transformer and transformer_2 separately)
- LoRA manager loader (support bare safetensors and key conversion)

For current upstream behavior, Diffusion LoRA docs still describe PEFT adapter format:

- `docs/user_guide/diffusion/lora.md`

## References

- LightX2V converter docs:
  <https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme_zh.md>
- LightX2V Wan2.2 LoRA:
  <https://huggingface.co/lightx2v/Wan2.2-Distill-Loras>
- Diffusers PR for Wan LoRA loading support:
  <https://github.com/huggingface/diffusers/pull/12074>

