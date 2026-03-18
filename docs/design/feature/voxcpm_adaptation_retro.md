# VoxCPM 适配复盘

## 背景

这次工作的目标，不是单纯把一个新模型名字注册进 `vllm-omni`，而是把 **原生 VoxCPM** 接入到现有框架里，并最终在 **NPU 环境上跑通真实推理链路**。

VoxCPM 和仓库里原本已经支持的 Hugging Face 模型有几个明显差异：

- 它的本地模型目录是 **原生配置格式**，不是标准 HF `config.json`。
- 它的实际推理入口是 `VoxCPMModel.from_local(...)` / `AudioVAE`，不是 vLLM 默认的 HF 权重加载路径。
- 它既可以作为 **单 stage 端到端语音生成**，也可以拆成 **latent generator + VAE** 两个 stage。
- 在 NPU 上运行时，还会遇到 `vllm-ascend` 与 `vllm-omni` 之间的若干 **接口兼容问题**。

因此，这次适配实际上分成了四层：

1. 模型识别与注册
2. 原生配置兼容
3. 推理执行与 stage 编排
4. NPU 运行时兼容修复

## 最终结果

当前已经完成的结果是：

- `vllm-omni` 可以识别原生 VoxCPM 模型目录。
- 可以使用 `voxcpm_full.yaml` 走单 stage 端到端语音生成。
- 可以使用 `voxcpm.yaml` 走 latent + VAE 两 stage 推理。
- OpenAI 兼容的 `/v1/audio/speech` 路径已经支持 VoxCPM 的文本合成和 voice cloning。
- 离线和在线 examples 已补齐。
- NPU 路径已经完成多轮兼容修复，最终跑通 VoxCPM。

## 我们做了什么

### 1. 把 VoxCPM 注册成框架中的一个可识别模型

我们新增了 VoxCPM 的配置类和模型包装类，让框架能把它当作一个合法的 `model_arch` 来创建：

- `vllm_omni/model_executor/models/voxcpm/configuration_voxcpm.py`
- `vllm_omni/model_executor/models/voxcpm/voxcpm.py`
- `vllm_omni/model_executor/models/voxcpm/__init__.py`
- `vllm_omni/model_executor/models/registry.py`
- `vllm_omni/transformers_utils/configs/voxcpm.py`
- `vllm_omni/transformers_utils/configs/__init__.py`

这里的核心关系是：

- `configuration_voxcpm.py` 负责把 VoxCPM 暴露成一个 `AutoConfig` 可识别的 `model_type="voxcpm"`。
- `voxcpm.py` 负责把原生 VoxCPM 推理接口包装成 vLLM/vLLM-Omni 所要求的模型接口。
- `registry.py` 负责把 `VoxCPMForConditionalGeneration` 注册到框架模型表。
- `transformers_utils/configs/*` 负责把 `AutoConfig.register("voxcpm", VoxCPMConfig)` 真正接进 transformers/vLLM 识别链。

### 2. 兼容 VoxCPM 的原生模型目录

VoxCPM 的本地权重目录不是标准 HF 模型目录，所以默认的 `ModelConfig` 校验会失败。为了解决这个问题，我们加了原生配置识别和临时 HF 配置渲染逻辑：

- `vllm_omni/model_executor/models/voxcpm/native_config.py`
- `vllm_omni/engine/arg_utils.py`
- `vllm_omni/entrypoints/utils.py`
- `tests/entrypoints/test_utils.py`

这部分的关键点有两个：

- 先判断一个模型目录是不是原生 VoxCPM。
- 如果是，就生成一个 **HF-compatible 临时目录**，里面写入 `config.json`，供 vLLM 的 `hf_config_path` 使用。

这里后续还做了一次关键修复：

- 最初返回的是一个临时 `json` 文件路径。
- 实际运行时发现 vLLM 需要的是“目录”而不是“单文件”。
- 所以后来改成生成 `/tmp/.../<digest>/config.json`，并返回目录路径。

这一步是整个适配真正跑起来的第一道门槛。

### 3. 把 VoxCPM 接进 stage pipeline

为了让 VoxCPM 在 `vllm-omni` 里按 stage 执行，我们补了：

- `vllm_omni/model_executor/stage_configs/voxcpm_full.yaml`
- `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- `vllm_omni/model_executor/stage_input_processors/voxcpm.py`
- `vllm_omni/platforms/npu/stage_configs/voxcpm_full.yaml`
- `vllm_omni/platforms/npu/stage_configs/voxcpm.yaml`

它们和框架的关系是：

- `voxcpm_full.yaml`：把 VoxCPM 当成单 stage 模型，直接输出音频。
- `voxcpm.yaml`：把 VoxCPM 拆成 `latent_generator -> vae` 两个 stage。
- `stage_input_processors/voxcpm.py`：负责把上游 latent 输出转换成下游 VAE 能消费的输入格式。
- `platforms/npu/stage_configs/*`：给 NPU 平台提供对应 stage 配置，避免 GPU/NPU 复用同一份参数时不合适。

### 4. 让 orchestrator 能支持更灵活的 stage 依赖关系

为了支撑“AR 主干 + 多下游分支”的总体目标，除了 VoxCPM 本身，还修改了 orchestrator 和 CFG companion 的逻辑：

- `vllm_omni/entrypoints/omni.py`
- `vllm_omni/entrypoints/cfg_companion_tracker.py`
- `tests/entrypoints/test_omni_llm.py`

这部分和 VoxCPM 的直接关系是：

- 它不是 VoxCPM 模型内部逻辑的一部分。
- 它解决的是 `vllm-omni` 原本偏线性的 stage 流程，扩展成可 fan-out 到多个下游 stage。
- 这为后续“主干 + 多分支”的推理结构打基础。

换句话说，这部分更偏 **框架编排层增强**，不是单模型 patch。

### 5. 把 VoxCPM 接进 OpenAI 兼容语音服务

我们还把 VoxCPM 接进了 `/v1/audio/speech` 这条服务链路：

- `vllm_omni/entrypoints/openai/serving_speech.py`
- `tests/entrypoints/openai_api/test_serving_speech.py`

这里的作用是：

- 请求校验：支持文本合成和 `ref_audio + ref_text` voice cloning。
- TTS 参数构造：为 VoxCPM 生成合适的 `additional_information`。
- prompt 长度估计：VoxCPM 只需要一个占位 token。
- 输出兼容：让 OpenAI Speech API 直接返回音频。

这一步让 VoxCPM 不只是“能在内部 stage 里执行”，而是能通过框架统一服务接口被调用。

### 6. 补齐 examples，方便使用和验收

为方便离线验证和在线服务，我们新增了 examples：

- `examples/offline_inference/voxcpm/end2end.py`
- `examples/offline_inference/voxcpm/README.md`
- `examples/online_serving/voxcpm/openai_speech_client.py`
- `examples/online_serving/voxcpm/run_server.sh`
- `examples/online_serving/voxcpm/README.md`

这几份文件的定位是：

- 离线 example：验证本地模型目录、stage config、推理链是否正确。
- 在线 example：验证 `serve + OpenAI API + client` 整条链是否正确。
- README：提供最短上手路径，降低迁移和验收成本。

## 后续为“跑通”补的关键修复

在真正上服务器、尤其是 NPU 环境实跑时，又补了几轮关键修复。这些改动虽然不多，但都是“没有它就跑不起来”的阻塞点。

### 1. `hf_config_path` 从单文件改为目录

涉及文件：

- `vllm_omni/model_executor/models/voxcpm/native_config.py`
- `tests/entrypoints/test_utils.py`

原因：

- vLLM 的 `ModelConfig` 校验要求 `hf_config_path` 指向目录或可识别模型目录。
- 之前返回的是 `/tmp/xxx.json`，会被判定为非法模型路径。

修复：

- 改成生成目录并在其中写入 `config.json`。

### 2. 为 VoxCPM 模型包装类补 `load_weights()`

涉及文件：

- `vllm_omni/model_executor/models/voxcpm/voxcpm.py`
- `tests/model_executor/models/test_voxcpm.py`

原因：

- vLLM 默认 loader 会调用 `model.load_weights(...)`。
- 我们的包装类最初没有这个接口。

修复：

- 增加 `load_weights()`，但不走 vLLM 的 HF 权重迭代链，而是直接触发原生 VoxCPM runtime 加载。

### 3. 兼容 `vllm-ascend` 的 `_prepare_inputs()` 返回签名变化

涉及文件：

- `vllm_omni/platforms/npu/worker/npu_model_runner.py`
- `vllm_omni/platforms/npu/worker/npu_generation_model_runner.py`
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`

原因：

- 不同版本的 `vllm-ascend` 中，`_prepare_inputs()` 返回值数量不完全一致。
- 当前服务器环境里返回值超过 2 个，旧写法直接 `unpack` 失败。

修复：

- 增加兼容函数，只取前两个核心返回值：`logits_indices` 和 `spec_decode_metadata`。

### 4. 兼容旧版 `ForwardContext` 没有 `sp_enabled`

涉及文件：

- `vllm_omni/platforms/npu/worker/npu_model_runner.py`

原因：

- 当前服务器环境里的 `ForwardContext` 没有 `sp_enabled` 属性。
- 直接访问会在执行阶段报错。

修复：

- 改成 `getattr(forward_context, "sp_enabled", False)`。

## 文件清单与作用

下面按“新增模型能力”“框架适配”“NPU 修复”“示例与文档”“测试”五类整理。

### A. 模型与配置

| 文件 | 作用 |
| --- | --- |
| `vllm_omni/model_executor/models/voxcpm/configuration_voxcpm.py` | 定义 `VoxCPMConfig`，让 transformers/vLLM 能识别 `model_type=voxcpm` |
| `vllm_omni/model_executor/models/voxcpm/voxcpm.py` | VoxCPM 主包装类，负责 native model 加载、full/split stage 推理、`ref_audio/ref_text` 处理 |
| `vllm_omni/model_executor/models/voxcpm/native_config.py` | 原生配置识别与 HF-compatible 临时配置目录生成 |
| `vllm_omni/model_executor/models/voxcpm/__init__.py` | 模块导出 |
| `vllm_omni/model_executor/models/registry.py` | 把 `VoxCPMForConditionalGeneration` 注册进框架模型表 |
| `vllm_omni/transformers_utils/configs/voxcpm.py` | 注册 `AutoConfig` |
| `vllm_omni/transformers_utils/configs/__init__.py` | 暴露 VoxCPM config 到 transformers config 包装层 |
| `vllm_omni/engine/arg_utils.py` | 在 engine 参数创建阶段自动准备 `hf_config_path` |
| `vllm_omni/entrypoints/utils.py` | 根据原生 VoxCPM 模型目录解析默认 stage config |

### B. Stage 与编排

| 文件 | 作用 |
| --- | --- |
| `vllm_omni/model_executor/stage_configs/voxcpm_full.yaml` | 单 stage 端到端 VoxCPM 配置 |
| `vllm_omni/model_executor/stage_configs/voxcpm.yaml` | 两 stage 的 latent + VAE 配置 |
| `vllm_omni/model_executor/stage_input_processors/voxcpm.py` | 把 latent stage 输出改造成 VAE stage 输入 |
| `vllm_omni/entrypoints/omni.py` | 从线性 stage 编排扩展到支持下游分支 fan-out |
| `vllm_omni/entrypoints/cfg_companion_tracker.py` | CFG companion 转发适配多下游 stage |

### C. OpenAI 服务

| 文件 | 作用 |
| --- | --- |
| `vllm_omni/entrypoints/openai/serving_speech.py` | 让 VoxCPM 通过 `/v1/audio/speech` 提供文本转语音和 voice cloning |

### D. NPU 专项

| 文件 | 作用 |
| --- | --- |
| `vllm_omni/platforms/npu/stage_configs/voxcpm_full.yaml` | NPU 下的单 stage 配置 |
| `vllm_omni/platforms/npu/stage_configs/voxcpm.yaml` | NPU 下的两 stage 配置 |
| `vllm_omni/platforms/npu/worker/npu_model_runner.py` | NPU model runner 的 Omni 兼容与版本兼容逻辑 |
| `vllm_omni/platforms/npu/worker/npu_generation_model_runner.py` | NPU 非 AR generation 执行路径兼容 |
| `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py` | NPU AR 执行路径兼容 |

### E. 测试

| 文件 | 作用 |
| --- | --- |
| `tests/model_executor/models/test_voxcpm.py` | 测试原生运行时目录、pipeline wrapper、latent2vae、`load_weights()` |
| `tests/entrypoints/test_utils.py` | 测试原生配置识别、HF 兼容配置目录生成、stage 配置路径解析 |
| `tests/entrypoints/openai_api/test_serving_speech.py` | 测试 VoxCPM 在 OpenAI speech API 下的请求识别与参数处理 |
| `tests/entrypoints/test_omni_llm.py` | 测试 fan-out 编排能力 |

### F. 示例与文档

| 文件 | 作用 |
| --- | --- |
| `examples/offline_inference/voxcpm/end2end.py` | 离线推理示例 |
| `examples/offline_inference/voxcpm/README.md` | 离线使用说明 |
| `examples/online_serving/voxcpm/openai_speech_client.py` | OpenAI 兼容 speech client 示例 |
| `examples/online_serving/voxcpm/run_server.sh` | 在线服务启动脚本 |
| `examples/online_serving/voxcpm/README.md` | 在线服务使用说明 |
| `docs/design/feature/add_voxcpm_commit_report.md` | 早期提交复盘 |
| `docs/design/feature/voxcpm_adaptation_retro.md` | 当前这份完整复盘 |

## 和框架的关系

如果从框架分层去看，这次适配大致对应下面几层：

### 1. 模型层

对应：

- `model_executor/models/voxcpm/*`
- `transformers_utils/configs/voxcpm.py`
- `model_executor/models/registry.py`

作用：

- 解决“框架如何认识 VoxCPM”。

### 2. 配置层

对应：

- `native_config.py`
- `engine/arg_utils.py`
- `entrypoints/utils.py`
- `stage_configs/*.yaml`

作用：

- 解决“框架如何用原生 VoxCPM 模型目录启动 stage”。

### 3. 编排层

对应：

- `entrypoints/omni.py`
- `entrypoints/cfg_companion_tracker.py`
- `stage_input_processors/voxcpm.py`

作用：

- 解决“上游输出如何流向下游 stage，如何支持多分支依赖关系”。

### 4. 服务层

对应：

- `entrypoints/openai/serving_speech.py`
- `examples/online_serving/voxcpm/*`

作用：

- 解决“外部客户端如何通过统一 API 使用 VoxCPM”。

### 5. 平台层

对应：

- `platforms/npu/stage_configs/*`
- `platforms/npu/worker/*`

作用：

- 解决“同一套模型与编排逻辑如何真正落在 NPU backend 上执行”。

## 跑通路径复盘

从排障路径看，这次实际经历了下面几步：

1. 先完成基础接入：模型识别、配置兼容、stage 配置、OpenAI API、examples。
2. 首次实跑时，发现 `hf_config_path` 返回了单个 JSON 文件，vLLM 不接受。
3. 改成返回临时目录 + `config.json`。
4. 继续实跑时，发现 `VoxCPMForConditionalGeneration` 没有 `load_weights()`。
5. 增加 `load_weights()`，切回 native runtime 进行真实加载。
6. 再往下跑，遇到 `vllm-ascend` 的 `_prepare_inputs()` 返回签名差异。
7. 增加兼容解包逻辑。
8. 再往下跑，遇到 `ForwardContext.sp_enabled` 属性差异。
9. 改成向后兼容访问。
10. 最终在 NPU 上跑通 VoxCPM。

这个过程说明：

- 模型接入本身只是一部分工作。
- 真正“跑通”往往还需要处理底层 backend 版本差异。

## 当前价值

这次工作的直接价值有三点：

- 把 **原生 VoxCPM** 正式纳入 `vllm-omni` 的模型体系。
- 让 VoxCPM 同时具备 **离线推理、在线服务、NPU 运行** 三条使用路径。
- 顺手补强了 `vllm-omni` 在 **原生模型目录兼容** 和 **NPU 版本兼容** 上的工程韧性。

## 还可以继续做什么

虽然已经跑通，但后面还有一些可以继续补强的方向：

- 增加更系统的 NPU 集成测试，而不是只靠实机手工验证。
- 补充 `docs/models/supported_models.md` 中的 VoxCPM 说明。
- 把这套适配的使用方式整合进正式用户文档，而不只是 example。
- 如果后续确实要做“AR 主干 + 多 DiT”更复杂结构，可以继续扩展 stage yaml 和 orchestrator 策略。

## 总结

这次 VoxCPM 适配不是一次单点 patch，而是一次跨层改造：

- 在模型层，解决了“如何识别和加载原生 VoxCPM”。
- 在配置层，解决了“如何把原生目录接入 vLLM 的配置校验链”。
- 在编排层，解决了“如何把 VoxCPM 放进 stage pipeline，且支持更灵活的下游关系”。
- 在服务层，解决了“如何通过 OpenAI 兼容接口对外提供能力”。
- 在平台层，解决了“如何在 NPU 环境里把这条链真正跑通”。

从结果上看，这次工作已经把 VoxCPM 从“概念上接入”推进到了“实际可运行、可演示、可对外说明”的状态。
