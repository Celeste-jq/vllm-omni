# VoxCPM

This directory contains the offline VoxCPM example for running native VoxCPM in vLLM Omni.

It covers:

- split-stage inference with `vllm_omni/model_executor/stage_configs/voxcpm_no_async_chunk.yaml`
- AR streaming with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`

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

Pass the native VoxCPM model directory directly. The original VoxCPM `config.json` can stay in native format. `vllm-omni` will render the HF-compatible config it needs at runtime.

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
```

## Quick Start

Text-only synthesis:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Voice cloning:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio."
```

AR streaming:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
  --text "This is a split-stage VoxCPM streaming example running on vLLM Omni."
```

Generated audio is saved to `output_audio/` by default for sync mode, and `output_audio_streaming/` for streaming mode.

## Useful Arguments

- `--stage-configs-path`: choose sync (`voxcpm_no_async_chunk.yaml`) or streaming (`voxcpm.yaml`)
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length
- `--streaming-prefix-len`: latent overlap window used by streaming decode
- `--num-runs`: repeat the same request multiple times for quick stability checks

## Two Modes

- `voxcpm_no_async_chunk.yaml`: non-streaming one-shot latent generation plus VAE decode. This path stays compatible with the original offline split-stage inference flow.
- `voxcpm.yaml`: AR streaming. Stage0 emits latent chunks incrementally and Stage1 decodes them through the VAE as they arrive.

## AR Streaming Design

- Stage0 `latent_generator` now runs as `worker_type: ar` with `OmniARScheduler`.
- Stage1 keeps `worker_type: generation` because it only decodes each latent chunk through the VAE.
- Stage0 emits `latent_audio_feat` plus `omni_stream_continue` / `omni_stream_gen_exhausted` (legacy aliases `latent_stream_*` are still accepted).
- [`latent2vae_async_chunk`](../../../vllm_omni/model_executor/stage_input_processors/voxcpm.py) forwards `latent_audio_feat`, optional `sr`, `code_predictor_codes: [0]`, and `finished` to Stage1.
- Stage1 trims the overlap introduced by the streaming latent window via `trim_streaming_patch`.

**Stage0→Stage1 payload contract:** `finished` becomes `true` when the request ends, `omni_stream_continue` becomes false, or the latent iterator is exhausted. In non-streaming mode, Stage0 generates the full latent once and Stage1 decodes it directly.

## Notes

- This branch only keeps the split-stage `latent_generator -> vae` pipeline.
- It does not include the single-stage `voxcpm_full.yaml` path.
- It does not include the OpenAI-compatible online speech serving adaptation.
- Voice cloning requires both `--ref-audio` and `--ref-text`.
