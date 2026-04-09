# VoxCPM Serving Benchmark

This benchmark measures VoxCPM through the OpenAI-compatible `/v1/audio/speech` API and reports:

- TTFP: time to first PCM packet
- E2E latency
- RTF: real-time factor (`e2e / audio_duration`)

## Start the Server

Async-chunk:

```bash
vllm serve /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --port 8091
```

Non-streaming:

```bash
vllm serve /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --port 8091
```

## Run the Benchmark

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_serve.py \
    --host 127.0.0.1 \
    --port 8091 \
    --num-prompts 20 \
    --max-concurrency 1 \
    --result-dir /tmp/voxcpm_bench
```

Voice cloning benchmark:

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_serve.py \
    --host 127.0.0.1 \
    --port 8091 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --ref-audio https://example.com/reference.wav \
    --ref-text "The exact transcript spoken in the reference audio." \
    --result-dir /tmp/voxcpm_clone_bench
```

## Notes

- The benchmark uses `stream=true` and `response_format=pcm` so TTFP is measured from the first audio packet.
- `RTF < 1.0` means the server generates audio faster than real time.
- For `voxcpm_async_chunk.yaml`, keep concurrency at `1`. This matches native VoxCPM streaming more closely.
- Do not benchmark concurrent online streaming on `voxcpm_async_chunk.yaml`; use `voxcpm.yaml` for multi-request throughput runs.
