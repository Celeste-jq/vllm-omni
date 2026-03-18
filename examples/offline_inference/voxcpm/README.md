# VoxCPM

This directory contains an offline demo for running a native VoxCPM model with vLLM Omni.

It supports:

- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`
- both the single-stage config (`voxcpm_full.yaml`) and the split latent/VAE config (`voxcpm.yaml`)

## Prerequisites

Install the VoxCPM codebase in one of these ways:

```bash
pip install voxcpm
```

or point vLLM Omni to the local VoxCPM source tree:

```bash
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

The example writes WAV files with `soundfile`:

```bash
pip install soundfile
```

## Model Path

Pass the native VoxCPM model directory directly. The `config.json` can stay in VoxCPM native format; `vllm-omni` will render an HF-compatible config automatically at runtime.

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
```

## Quick Start

Text-only synthesis with the default full-pipeline stage config:

```bash
python end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a VoxCPM synthesis example running on vLLM Omni."
```

Voice cloning:

```bash
python end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio."
```

Use the split latent/VAE pipeline:

```bash
python end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-mode split \
  --text "Run VoxCPM with the split stage config."
```

Generated audio is saved to `output_audio/` by default.

## Useful Arguments

- `--stage-mode full|split`: choose `voxcpm_full.yaml` or `voxcpm.yaml`
- `--stage-configs-path`: override the stage config path explicitly
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length
- `--normalize`: enable normalization in full-pipeline mode
- `--denoise`: enable denoising in full-pipeline mode

## Notes

- `full` mode is the simplest option and is the recommended starting point.
- `split` mode is useful when you want to run the latent generator and audio VAE as separate stages.
- Voice cloning requires both `--ref-audio` and `--ref-text`.
