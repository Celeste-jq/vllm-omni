# Wan2.2 I2V HSDP Benchmark

`tools/benchmark_wan_i2v_hsdp.py` 是一个外层实验编排脚本，用于自动对比 `baseline` 与 `HSDP` 两组 `Wan2.2-I2V` 运行结果。

它不会替换现有的 `tools/profile_wan_i2v.py`，而是复用后者作为单次真实执行入口，然后额外收集：

- 进程级命令耗时
- 单次 profiled run 的 rank 级 wall time
- 峰值显存
- 模块级 profiling 汇总
- baseline/HSDP 对比 `json/csv/md`

## 运行方式

```bash
python tools/benchmark_wan_i2v_hsdp.py \
  --model /path/to/Wan2.2-I2V-A14B \
  --image /path/to/image.jpg \
  --prompt "A cat playing with yarn" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --nproc-per-node 4 \
  --ulysses-degree 4 \
  --cfg-parallel-size 1 \
  --hsdp-shard-size 4 \
  --hsdp-replicate-size 1 \
  --output-dir benchmark_runs/wan_i2v_hsdp
```

说明：

- `baseline` 场景自动使用不带 `--use-hsdp` 的命令。
- `hsdp` 场景自动使用 `--use-hsdp`，并继承 `--hsdp-shard-size` / `--hsdp-replicate-size`。
- 两个场景都会调用 `tools/profile_wan_i2v.py`，因此会保留模块级 trace 和 summary。

## 输出目录

默认输出目录为 `benchmark_runs/wan_i2v_hsdp`，结构如下：

```text
benchmark_runs/wan_i2v_hsdp/
  baseline/
    command.txt
    stdout.log
    stderr.log
    baseline.mp4
    profiling/
      aggregate_summary.json
      aggregate_run_metadata.json
      rank0/
      rank1/
      ...
  hsdp/
    command.txt
    stdout.log
    stderr.log
    hsdp.mp4
    profiling/
      aggregate_summary.json
      aggregate_run_metadata.json
      rank0/
      rank1/
      ...
  comparison.json
  scenario_summary.csv
  phase_summary.csv
  comparison.md
```

## 关键输出

### `comparison.json`

包含：

- `baseline`
- `hsdp`
- `deltas`

其中 `deltas` 当前默认包括：

- `wall_time_pct`
- `command_wall_time_pct`
- `peak_allocated_pct`
- `peak_reserved_pct`
- `total_profiled_sec_pct`

约定：

- `wall_time_pct` 为 `(hsdp - baseline) / baseline`
- `peak_allocated_pct` 为 `(baseline - hsdp) / baseline`
- 因此：
  - 时延项负值表示更快
  - 显存项正值表示 HSDP 节省了显存

### `scenario_summary.csv`

每个场景一行，便于直接拿去做表格或画图。当前包含：

- `status`
- `oom`
- `returncode`
- `wall_time_sec`
- `command_wall_time_sec`
- `total_profiled_sec`
- `max_peak_allocated_bytes`
- `max_peak_reserved_bytes`
- 各模块 phase 百分比，例如 `DIT_HIGH_pct`、`DIT_LOW_pct`、`VAE_DECODE_pct`

### `phase_summary.csv`

每个场景按 phase 展开，便于单独对比：

- `TEXT_ENCODER`
- `DIT_HIGH`
- `DIT_LOW`
- `VAE_ENCODE`
- `VAE_DECODE`

## 量化 HSDP 收益时怎么看

建议优先看这几项：

1. `max_peak_allocated_bytes`
   - 直接看单卡峰值显存是否下降
2. `wall_time_sec`
   - 看 HSDP 是否拉高或拉低单次 profiled run 时长
3. `command_wall_time_sec`
   - 看完整命令总耗时是否变化
4. `total_profiled_sec`
   - 看模块级主路径是否整体变慢
5. `DIT_HIGH_pct / DIT_LOW_pct / VAE_*_pct`
   - 看 HSDP 是否改变模块占比结构

如果出现失败，先看：

- `baseline/stderr.log`
- `hsdp/stderr.log`

脚本会把典型 `out of memory / oom` 失败标记成 `status=oom`。
