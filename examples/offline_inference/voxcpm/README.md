# VoxCPM

This directory contains the offline example for running native VoxCPM in vLLM Omni on the `pure_voxcpm` branch.

It covers:

- split-stage inference with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`
- batch processing with voice cloning support

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

### Single Text Synthesis

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

### Voice Cloning

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --prompt-audio /path/to/reference.wav \
  --prompt-text "Transcript of the reference audio."
```

### Batch Processing from Text File

Process multiple texts (one per line):

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --input example_texts.txt \
  --output-dir ./outs
```

### Batch Processing with Voice Cloning

Process multiple texts with a shared reference audio:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --input example_texts.txt \
  --prompt-audio reference.wav \
  --prompt-text "reference transcript" \
  --output-dir ./outs
```

### Batch Processing from JSONL

Process multiple texts with individual reference audio files:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --jsonl-file example_batch.jsonl \
  --output-dir ./outs
```

JSONL file format (one JSON object per line):
```json
{"audio": "reference1.wav", "text": "This is the first example text."}
{"audio": "reference2.wav", "text": "This is the second example."}
{"audio": "reference3.wav", "text": "You can use absolute or relative paths."}
```

## Running Examples

You can run all examples at once using the provided script:

```bash
bash examples/offline_inference/voxcpm/run_examples.sh
```

Or run individual examples by copying the commands from the script.

## Useful Arguments

### Input Modes (mutually exclusive)

- `--text` / `-t`: single text to synthesize
- `--input` / `-i`: path to text file (one text per line)
- `--jsonl-file`: path to JSONL file with audio/text pairs

### Voice Cloning

- `--prompt-audio` / `-pa`: reference audio file path (clone mode)
- `--prompt-text` / `-pt`: transcript of the reference audio

### Generation Parameters

- `--cfg-value`: guidance value passed to VoxCPM (default: 2.0)
- `--inference-timesteps`: number of diffusion timesteps (default: 10)
- `--min-len`: minimum token length (default: 2)
- `--max-new-tokens`: maximum token length (default: 4096)

### Output and Runtime

- `--output` / `-o`: output audio file path (single or clone mode)
- `--output-dir` / `-od`: output directory (batch mode only)
- `--stage-configs-path`: override the split-stage config path explicitly
- `--stage-init-timeout`: stage initialization timeout in seconds (default: 600)
- `--log-stats`: enable vLLM Omni stats logging

## Architecture

The script follows a clean, modular architecture similar to `qwen3_tts/end2end.py`:

- **`cmd_synthesize(args)`**: Build inputs for single text synthesis (no voice cloning)
- **`cmd_clone(args)`**: Build inputs for voice cloning with reference audio
- **`cmd_batch(args)`**: Build inputs for batch processing from text file or JSONL file
- **`main(args)`**: Unified entry point that routes to appropriate command and calls `omni.generate()`

This design ensures:
- Clear separation of concerns
- Easy to extend with new modes
- Consistent with vLLM Omni patterns
- Minimal code duplication

## Notes

- This branch only keeps the split-stage `latent_generator -> vae` pipeline.
- It does not include the single-stage `voxcpm_full.yaml` path.
- It does not include the OpenAI-compatible online speech serving adaptation.
- Voice cloning requires both `--prompt-audio` and `--prompt-text` (or audio/text in JSONL).
- Batch processing with JSONL allows individual reference audio per text sample.
- The batch processing implementation follows VoxCPM's CLI patterns from `VoxCPM/src/voxcpm/cli.py`.
- Batch processing processes texts one by one (sequential), following VoxCPM's original behavior.
- All parameter names are consistent with VoxCPM's original CLI.
