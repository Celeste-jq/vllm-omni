# VoxCPM

This directory contains the minimal offline example for running native VoxCPM in vLLM Omni on the `pure_voxcpm` branch.

It covers:

- split-stage inference with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
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

Generated audio is saved to `output_audio/` by default.

## Useful Arguments

- `--stage-configs-path`: override the split-stage config path explicitly
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length

## Omni async_chunk vs Qwen3-TTS (same transport, different Stage0 semantics)

Both pipelines use `async_chunk: true`, [`OmniChunkTransferAdapter`](../../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py), `SharedMemoryConnector`, and a `custom_process_next_stage_input_func` to build the Stage0→Stage1 payload (`code_predictor_codes`, `finished`, plus modality-specific fields).

| Aspect | Qwen3-TTS | VoxCPM |
|--------|-----------|--------|
| Stage0 `worker_type` | `ar` (connector merges payloads across AR steps) | `generation` (each chunk replaces `prompt_token_ids` for Stage1) |
| Stage0 scheduler | `OmniARScheduler` | `OmniGenerationScheduler` |
| What each Stage0 step produces | One speech-token frame (`audio_codes` in pooler) | One latent window from an internal iterator (`latent_audio_feat`) |
| “More chunks?” signal | Implicit via AR decode until EOS | Explicit: `omni_stream_continue` / `omni_stream_gen_exhausted` (legacy: `latent_stream_*`) in pooler; see [`omni_streaming_keys.py`](../../../vllm_omni/core/omni_streaming_keys.py) |
| Connector `codec_streaming` | `true` + frame windowing in [`talker2code2wav_async_chunk`](../../../vllm_omni/model_executor/stage_input_processors/qwen3_tts.py) | `false` — each chunk is a full latent for VAE ([`latent2vae_async_chunk`](../../../vllm_omni/model_executor/stage_input_processors/voxcpm.py)) |
| Stage1 | Code2Wav | VAE decode (`trim_streaming_patch` trims overlap) |

**Stage0→Stage1 payload contract (VoxCPM streaming):** `latent2vae_async_chunk` sends `latent_audio_feat`, optional `sr`, `code_predictor_codes: [0]`, and `finished` when the request is done, the stream no longer continues, or the generator is exhausted. Pooler flags are interpreted via `pooler_stream_continues` / `pooler_stream_gen_exhausted` (supports both `omni_*` and legacy `latent_stream_*` keys).

## Notes

- This branch only keeps the split-stage `latent_generator -> vae` pipeline.
- It does not include the single-stage `voxcpm_full.yaml` path.
- It does not include the OpenAI-compatible online speech serving adaptation.
- Voice cloning requires both `--ref-audio` and `--ref-text`.
