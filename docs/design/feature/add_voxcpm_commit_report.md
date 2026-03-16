# add_voxcpm 提交记录

## 目标说明

本分支的真实目标不是单纯“接入 VoxCPM”，而是：

- 让 `vllm-omni` 适配 `VoxCPM`
- 支持 `AR（主）+ 多 DiT` 的多阶段推理
- 最终运行在 `NPU` 上

因此，下面的每次提交记录，都会以这三个目标作为判断标准：

1. 是否推进了 VoxCPM 适配
2. 是否推进了 AR 主干加多 DiT 的编排能力
3. 是否推进了 NPU 运行能力

## 记录范围

- 基线提交: `dc72aed` (`[CI] add multimodal processing correctness tests for Omni models (#1445)`)
- 记录起点: `2ae2303` (`test`)
- 当前提交: `f248651` (`0314初步添加voxcpm`)
- 记录时间范围: 2026-03-14 15:12:13 +0800 到 2026-03-14 15:56:26 +0800

本文档按提交顺序记录 `add_voxcpm` 分支相对基线提交新增的每一次提交及其作用，并明确它对上述目标的覆盖情况。

## 提交 1: `2ae2303` `test`

- 提交时间: 2026-03-14 15:12:13 +0800
- 变更范围: 新增 `test.md`
- 变更统计: 1 个文件，1 行新增

### 本次提交做了什么

这是一个占位性质的测试提交，仅新增了 [test.md](/Users/cat/ott3/add_voxcpm/vllm-omni/test.md)，文件内容为 `test`。

### 影响评估

- 不涉及功能逻辑、模型接入、调度流程或测试体系改动。
- 更像是分支上的临时校验提交，可视为后续正式开发前的占位节点。

### 对总目标的覆盖情况

- VoxCPM 适配: 无
- AR（主）+ 多 DiT 推理: 无
- NPU 支持: 无

## 提交 2: `f248651` `0314初步添加voxcpm`

- 提交时间: 2026-03-14 15:56:26 +0800
- 变更统计: 14 个文件，900 行新增，106 行删除
- 目标方向: 初步支持 VoxCPM 接入，并为 AR 主干加多 DiT/多下游分支推理提供入口层能力

### 本次提交做了什么

#### 1. 初步接入 VoxCPM 模型

- 新增 VoxCPM 配置类 [configuration_voxcpm.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/model_executor/models/voxcpm/configuration_voxcpm.py)。
- 新增 VoxCPM 模型包装实现 [voxcpm.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/model_executor/models/voxcpm/voxcpm.py)，通过 `voxcpm.core.VoxCPM` 驱动生成。
- 支持文本生成音频，并兼容 `prompt_wav_path`、`ref_audio`、`ref_text` 等提示输入形式。
- 在模型注册表中加入 `VoxCPMForConditionalGeneration`，使 vLLM-Omni 能按架构名识别该模型。

#### 2. 兼容 VoxCPM 原生配置格式

- 新增 [native_config.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/model_executor/models/voxcpm/native_config.py)。
- 检测本地模型目录中的原生 VoxCPM `config.json`。
- 将原生配置转换为 HF-compatible 临时配置文件，解决 `model_type` 和 `architectures` 不完整导致的加载问题。
- 在 [arg_utils.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/engine/arg_utils.py) 中补充 `AutoConfig.register("voxcpm", VoxCPMConfig)` 和自动 `hf_config_path` 准备逻辑。
- 在 [utils.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/entrypoints/utils.py) 中补充对原生 VoxCPM 配置的识别与 stage 配置路径解析。

#### 3. 将 orchestrator 从线性 stage 转为支持分叉下游

- 在 [omni.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/entrypoints/omni.py) 中新增：
  - `engine_input_source` / `input_sources` 读取逻辑
  - `_build_downstream_stage_ids()`
  - `_build_reachable_output_modalities()`
  - `_get_requested_output_modalities()`
  - `_get_active_downstream_stage_ids()`
- 原先默认按 `stage_id + 1` 转发的流程，被扩展为按显式依赖关系决定一个 stage 的多个下游 stage。
- 同一个请求可以从上游 stage 同时 fan-out 到多个下游分支，这为“AR 主干 + 多个 DiT 分支”提供了入口层调度基础。

#### 4. 调整多分支请求的完成态和转发逻辑

- 在 [omni.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/entrypoints/omni.py) 中引入 `in_flight_requests`，追踪单个请求仍在多少个分支中执行。
- 请求完成不再简单依赖单一路径的末端 stage，而是依赖所有活跃分支完成。
- 输出 modality 可按 prompt 中的 `modalities` 做过滤，只激活能够到达目标输出的下游分支。

#### 5. 让 CFG companion 逻辑适配多下游分支

- 在 [cfg_companion_tracker.py](/Users/cat/ott3/add_voxcpm/vllm-omni/vllm_omni/entrypoints/cfg_companion_tracker.py) 中，将 parent request 的 CFG 转发从“发往下一个 stage”改为“发往 downstream stages”。
- 每个下游 stage 都会独立构造输入，并在 diffusion sampling params 上附带 `cfg_kv_request_ids`。
- 这样 CFG 相关逻辑不会阻塞多分支推理结构。

#### 6. 增加对应测试

- 在 [test_omni_llm.py](/Users/cat/ott3/add_voxcpm/vllm-omni/tests/entrypoints/test_omni_llm.py) 中新增 `test_generate_branching_pipeline_fanout`。
- 该测试验证 0 号 stage 的输出可以同时转发到 1 号和 2 号 stage，并收集多个最终输出。
- 在 [test_utils.py](/Users/cat/ott3/add_voxcpm/vllm-omni/tests/entrypoints/test_utils.py) 中新增 VoxCPM 原生配置转换及配置路径解析测试。

### 对总目标的意义

这次提交不是完整的“VoxCPM + AR 主模型 + 多 DiT + NPU”最终交付，而是先把其中最关键的两层基础设施打通：

- 模型侧: VoxCPM 已能以 vLLM-Omni 模型形式被识别和加载。
- 编排侧: orchestrator 已从单链路推进为可支持一个上游 stage 扇出多个下游 stage。

因此，这个提交可以视为该能力的第一版基础实现。

### 对总目标的覆盖情况

- VoxCPM 适配: 已开始，完成了模型注册、配置兼容、基础包装和单 stage 配置接入。
- AR（主）+ 多 DiT 推理: 已开始，完成了多下游分支编排和 CFG 分支转发能力的基础改造。
- NPU 支持: 未在本次提交中完成。当前 diff 中没有新增 `VoxCPM` 相关的 NPU 专项适配代码、NPU 设备放置逻辑、NPU 算子兼容修复或 NPU 实机测试。

### 当前仍缺的关键能力

- 还没有把 VoxCPM 的各子模块明确拆分进 `AR stage` 和多个 `DiT stage` 的正式 stage 配置。
- 还没有看到针对多 DiT 分支的 NPU 专项执行路径验证。
- 还没有看到 VoxCPM 在 NPU 上的依赖、算子、精度、设备初始化和性能参数适配。
- 还没有对应的 NPU 集成测试或运行记录。

## 当前阶段总结

截至 `f248651`，`add_voxcpm` 分支相对基线提交共新增 2 次提交：

1. `2ae2303`：占位测试提交，无实质功能改动。
2. `f248651`：完成 VoxCPM 初步接入，并把入口编排能力扩展到支持多下游分支，为 AR 主干加多 DiT 推理奠定基础。

从“`vllm-omni` 适配 `VoxCPM`，支持 `AR（主）+ 多 DiT`，且跑在 `NPU` 上”这个目标看，当前分支状态可概括为：

- `VoxCPM` 接入: 已开始
- `AR（主）+ 多 DiT` 编排: 已开始
- `NPU` 适配: 目标已明确，但从当前提交记录看，还没有形成对应的实质提交
