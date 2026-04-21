# Wan2.2 I2V Profiling

`tools/profile_wan_i2v.py` 提供一个面向 `Wan2.2-I2V` 的独立 profiling 入口，目标是尽量对齐 MindIE 侧的模块标签与 summary 口径，同时尽量少改 `vllm-omni` 主体代码。

## 输出标签

脚本会在 `torch_npu` trace 中打出这些模块标签，并在 `summary.json / summary.txt` 中按相同口径汇总：

- `TEXT_ENCODER`
- `DIT_HIGH`
- `DIT_LOW`
- `VAE_ENCODE`
- `VAE_DECODE`

其中：

- `DIT_HIGH` 对应 `Wan22I2VPipeline.transformer`
- `DIT_LOW` 对应 `Wan22I2VPipeline.transformer_2`

## 运行方式

单卡：

```bash
python tools/profile_wan_i2v.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image /path/to/image.jpg \
  --prompt "A cat playing with yarn" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --profiling-output-dir profiling_runs/wan_i2v_profile
```

多卡：

```bash
torchrun --nproc_per_node=4 tools/profile_wan_i2v.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image /path/to/image.jpg \
  --prompt "A cat playing with yarn" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --ulysses-degree 4 \
  --cfg-parallel-size 1 \
  --profiling-output-dir profiling_runs/wan_i2v_profile
```

## 输出目录

默认输出目录是 `profiling_runs/wan_i2v_profile`，目录结构如下：

```text
profiling_runs/wan_i2v_profile/
  aggregate_summary.json
  aggregate_run_metadata.json
  aggregate_summary.txt
  rank0/
    run_metadata.json
    summary.json
    summary.txt
    trace/
  rank1/
    run_metadata.json
    summary.json
    summary.txt
    trace/
  ...
```

说明：

- 每个 `rank*/summary.json` 是该 rank 自己的模块级统计。
- 每个 `rank*/run_metadata.json` 会记录该 rank 的 profiled run wall time、峰值显存和并行配置。
- `aggregate_summary.json` 是按 phase 汇总后的总视图，便于和 MindIE 侧结果做横向对比。
- `aggregate_run_metadata.json` 会聚合 rank 级 wall time 与峰值显存，便于外层 benchmark 脚本直接读取。
- `trace/` 下是 `torch_npu.profiler.tensorboard_trace_handler(...)` 产生的 trace 数据。

## 与 HSDP 对比脚本配合

如果需要自动对比 baseline 与 HSDP 两组场景，可直接使用 [`tools/benchmark_wan_i2v_hsdp.py`](../tools/benchmark_wan_i2v_hsdp.py)。

该脚本会复用本 profiling 入口，并额外产出：

- `comparison.json`
- `scenario_summary.csv`
- `phase_summary.csv`
- `comparison.md`

## 实现路径

脚本没有改 `Omni(...)` 的原始调用链，而是直接复用了 `vllm-omni` 的这些运行时组件：

- `DiffusionModelRunner`
- `DiffusersPipelineLoader`
- `init_distributed_environment`
- `initialize_model_parallel`

然后只在真实加载出来的 `Wan22I2VPipeline` 实例上做最小运行时 patch，以得到 MindIE 风格的模块标签。

## 限制

- 当前脚本只覆盖 `Wan2.2-I2V`。
- 要求运行环境已经安装 `vllm`、`vllm-omni` 依赖和 `torch_npu`。
- 当前实现优先保证模块级 profiling 与 trace 输出，不主动改动 `vllm-omni` 内置 profiler 框架。
