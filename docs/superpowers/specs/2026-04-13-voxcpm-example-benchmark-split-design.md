# VoxCPM Example And Benchmark Split

## Goal

Address review feedback that `examples/offline_inference/voxcpm/end2end.py` should be a small getting-started example rather than a benchmark runner.

## Scope

This change only reorganizes VoxCPM offline example and benchmark code.

It does not change:
- VoxCPM model behavior
- stage configs
- online serving benchmark naming

## File Layout

Keep the minimal example at:
- `examples/offline_inference/voxcpm/end2end.py`

Move benchmark-oriented logic to:
- `benchmarks/voxcpm/vllm_omni/bench_tts_offline.py`

Keep the existing online benchmark at:
- `benchmarks/voxcpm/vllm_omni/bench_tts_serve.py`

Update benchmark docs at:
- `benchmarks/voxcpm/README.md`

Update example docs at:
- `examples/offline_inference/voxcpm/README.md`

Remove:
- `examples/offline_inference/voxcpm/advanced_runner.py`

## Responsibilities

`end2end.py` only supports:
- single text-to-speech
- single voice cloning
- sync route
- streaming route
- saving generated wav output

`bench_tts_offline.py` owns:
- warmup
- batch txt/jsonl inputs
- repeated runs
- TTFP / RTF / stage metrics
- profiling options
- benchmark-oriented output formatting

## Naming

Follow the existing benchmark naming style already used in the repo:
- online serving: `bench_tts_serve.py`
- offline benchmark: `bench_tts_offline.py`

This keeps VoxCPM aligned with `benchmarks/qwen3-tts/`.

## Acceptance Criteria

- `examples/offline_inference/voxcpm/end2end.py` stays small and only demonstrates basic usage
- no benchmark-only logic remains in `end2end.py`
- VoxCPM offline benchmark entrypoint lives under `benchmarks/voxcpm/vllm_omni/`
- docs clearly separate example usage from benchmark usage
