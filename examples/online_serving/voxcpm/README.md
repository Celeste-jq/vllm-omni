# VoxCPM

This directory contains an online serving example for a native VoxCPM model exposed through the OpenAI-compatible `/v1/audio/speech` API.

It covers:

- starting a VoxCPM server with `vllm-omni serve`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`

## Prerequisites

Install the VoxCPM codebase in one of these ways:

```bash
pip install voxcpm
```

or point vLLM Omni to a local checkout:

```bash
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

Set the model directory:

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
```

## Start the Server

Recommended starting point, using the single-stage pipeline:

```bash
VOXCPM_MODEL="$VOXCPM_MODEL" ./run_server.sh
```

Use the split latent/VAE stage config instead:

```bash
VOXCPM_MODEL="$VOXCPM_MODEL" STAGE_MODE=split ./run_server.sh
```

You can also pass the model path directly:

```bash
./run_server.sh /path/to/voxcpm-model
```

The server listens on `http://localhost:8091` by default.

## Send a Request

Text-only synthesis:

```bash
python openai_speech_client.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a VoxCPM speech request through the OpenAI-compatible API."
```

Voice cloning from a local WAV file:

```bash
python openai_speech_client.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio." \
  --output cloned.wav
```

The client automatically converts local audio files to a `data:` URL before sending the request.

## curl Example

Text-only request:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$VOXCPM_MODEL\",
    \"input\": \"This is a VoxCPM speech request through curl.\",
    \"response_format\": \"wav\"
  }" \
  --output voxcpm_output.wav
```

## Notes

- Pass the native VoxCPM model directory directly. `vllm-omni` will prepare an HF-compatible config automatically.
- Voice cloning requires both `ref_audio` and `ref_text`.
- For local `ref_audio` files, the Python client is more convenient than raw `curl` because it handles base64 encoding automatically.
