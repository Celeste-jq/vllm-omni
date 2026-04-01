# VoxCPM

This directory contains the offline AR-streaming example for running native VoxCPM in vLLM Omni.

It covers:

- split-stage AR streaming with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
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
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Voice cloning:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio."
```

Generated audio is saved to `output_audio_streaming/` by default.

## Useful Arguments

- `--stage-configs-path`: override the AR streaming stage config path explicitly
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length
- `--streaming-prefix-len`: latent overlap window used by streaming decode
- `--num-runs`: repeat the same request multiple times for quick stability checks

## AR Streaming Design

VoxCPM streaming uses `async_chunk: true`, [`OmniChunkTransferAdapter`](../../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py), `SharedMemoryConnector`, and a `custom_process_next_stage_input_func` to move latent chunks from Stage0 to Stage1.

- Stage0 `latent_generator` now runs as `worker_type: ar` with `OmniARScheduler`.
- Stage1 keeps `worker_type: generation` because it only decodes each latent chunk through the VAE.
- Stage0 emits `latent_audio_feat` plus `omni_stream_continue` / `omni_stream_gen_exhausted` (legacy aliases `latent_stream_*` are still accepted).
- [`latent2vae_async_chunk`](../../../vllm_omni/model_executor/stage_input_processors/voxcpm.py) forwards `latent_audio_feat`, optional `sr`, `code_predictor_codes: [0]`, and `finished` to Stage1.
- Stage1 trims the overlap introduced by the streaming latent window via `trim_streaming_patch`.

**Stage0→Stage1 payload contract:** `finished` becomes `true` when the request ends, `omni_stream_continue` becomes false, or the latent iterator is exhausted.

## Notes

- This branch only keeps the split-stage `latent_generator -> vae` pipeline.
- It does not include the single-stage `voxcpm_full.yaml` path.
- It does not include the OpenAI-compatible online speech serving adaptation.
- Voice cloning requires both `--ref-audio` and `--ref-text`.
