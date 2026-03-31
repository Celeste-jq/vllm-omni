#!/bin/bash
# Unified convenience wrapper for Qwen3-TTS profiling on Ascend NPU.
#
# Examples:
#   # Run both online and offline profiling with stack capture enabled.
#   MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   DEVICE_ID=0 \
#   PROFILE_MODE=both \
#   bash run_npu_profile.sh
#
#   # Only online profiling.
#   MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   PROFILE_MODE=online \
#   bash run_npu_profile.sh
#
#   # Only offline profiling with prompt file.
#   MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   PROFILE_MODE=offline \
#   OFFLINE_TXT_PROMPTS=/path/to/prompts.txt \
#   OFFLINE_NUM_PROMPTS=2 \
#   bash run_npu_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROFILE_MODE="${PROFILE_MODE:-both}"
DEVICE_ID="${DEVICE_ID:-${ASCEND_DEVICE_ID:-0}}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
GPU_MEM_TALKER="${GPU_MEM_TALKER:-0.3}"
GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV:-0.2}"
PROFILER_STAGES="${PROFILER_STAGES:-0 1}"
PROFILER_WITH_STACK="${PROFILER_WITH_STACK:-1}"
PROFILE_WAIT_SECS="${PROFILE_WAIT_SECS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results/npu_profile}"

NUM_PROMPTS="${NUM_PROMPTS:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
SKIP_PLOT="${SKIP_PLOT:-1}"

ONLINE_STAGE_CONFIG="${ONLINE_STAGE_CONFIG:-vllm_omni/platforms/npu/stage_configs/qwen3_tts.yaml}"
ONLINE_RESULT_DIR="${ONLINE_RESULT_DIR:-${RESULT_ROOT}/online}"
ONLINE_PROFILER_DIR="${ONLINE_PROFILER_DIR:-${ONLINE_RESULT_DIR}/traces}"

OFFLINE_STAGE_CONFIG="${OFFLINE_STAGE_CONFIG:-vllm_omni/platforms/npu/stage_configs/qwen3_tts.yaml}"
OFFLINE_RESULT_DIR="${OFFLINE_RESULT_DIR:-${RESULT_ROOT}/offline}"
OFFLINE_PROFILER_DIR="${OFFLINE_PROFILER_DIR:-${OFFLINE_RESULT_DIR}/traces}"
OFFLINE_OUTPUT_DIR="${OFFLINE_OUTPUT_DIR:-${OFFLINE_RESULT_DIR}/audio}"
OFFLINE_QUERY_TYPE="${OFFLINE_QUERY_TYPE:-CustomVoice}"
OFFLINE_MODE_TAG="${OFFLINE_MODE_TAG:-icl}"
OFFLINE_BATCH_SIZE="${OFFLINE_BATCH_SIZE:-1}"
OFFLINE_NUM_PROMPTS="${OFFLINE_NUM_PROMPTS:-${NUM_PROMPTS}}"
OFFLINE_TXT_PROMPTS="${OFFLINE_TXT_PROMPTS:-}"
OFFLINE_STREAMING="${OFFLINE_STREAMING:-0}"
OFFLINE_USE_BATCH_SAMPLE="${OFFLINE_USE_BATCH_SAMPLE:-0}"

mkdir -p "${RESULT_ROOT}" "${ONLINE_RESULT_DIR}" "${OFFLINE_RESULT_DIR}" "${OFFLINE_OUTPUT_DIR}"

resolve_stage_config_path() {
    local config_path="$1"
    if [ -f "${config_path}" ]; then
        printf '%s\n' "${config_path}"
        return 0
    fi
    if [ -f "${SCRIPT_DIR}/${config_path}" ]; then
        printf '%s\n' "${SCRIPT_DIR}/${config_path}"
        return 0
    fi
    if [ -f "${PROJECT_ROOT}/${config_path}" ]; then
        printf '%s\n' "${PROJECT_ROOT}/${config_path}"
        return 0
    fi
    echo "Stage config not found: ${config_path}" >&2
    return 1
}

prepare_offline_config() {
    local config_template="$1"
    local output_path="${OFFLINE_RESULT_DIR}/offline_stage_config.yaml"

    "${PYTHON_BIN}" - <<'PY' "${config_template}" "${output_path}" "${DEVICE_ID}" "${GPU_MEM_TALKER}" "${GPU_MEM_CODE2WAV}" "${OFFLINE_PROFILER_DIR}" "${PROFILER_WITH_STACK}"
import sys
from pathlib import Path

config_template, output_path, device_id, talker_mem, code2wav_mem, profiler_dir, profiler_with_stack = sys.argv[1:]

lines = Path(config_template).read_text().splitlines()
patched = []
for line in lines:
    stripped = line.strip()
    if stripped == 'devices: "0"':
        indent = line[: len(line) - len(line.lstrip())]
        patched.append(f'{indent}devices: "{device_id}"')
        continue
    if stripped == "gpu_memory_utilization: 0.3":
        indent = line[: len(line) - len(line.lstrip())]
        patched.append(f"{indent}gpu_memory_utilization: {talker_mem}")
        continue
    if stripped == "gpu_memory_utilization: 0.2":
        indent = line[: len(line) - len(line.lstrip())]
        patched.append(f"{indent}gpu_memory_utilization: {code2wav_mem}")
        continue

    patched.append(line)

    if stripped == "engine_args:":
        indent = line[: len(line) - len(line.lstrip())] + "  "
        patched.extend(
            [
                f"{indent}profiler_config:",
                f"{indent}  profiler: torch",
                f"{indent}  torch_profiler_dir: {profiler_dir}",
                f"{indent}  torch_profiler_with_stack: {'true' if profiler_with_stack == '1' else 'false'}",
            ]
        )

Path(output_path).write_text("\n".join(patched) + "\n")
PY

    printf '%s\n' "${output_path}"
}

run_online() {
    echo "============================================================"
    echo " Running NPU online profiling"
    echo "============================================================"

    (
        export DEVICE_BACKEND=npu
        export DEVICE_ID="${DEVICE_ID}"
        export MODEL="${MODEL}"
        export STAGE_CONFIG="${ONLINE_STAGE_CONFIG}"
        export RESULT_DIR="${ONLINE_RESULT_DIR}"
        export ENABLE_PROFILING=1
        export PROFILER_DIR="${ONLINE_PROFILER_DIR}"
        export PROFILER_STAGES="${PROFILER_STAGES}"
        export PROFILER_WITH_STACK="${PROFILER_WITH_STACK}"
        export PROFILE_WAIT_SECS="${PROFILE_WAIT_SECS}"
        export NUM_PROMPTS="${NUM_PROMPTS}"
        export NUM_WARMUPS="${NUM_WARMUPS}"
        export CONCURRENCY="${CONCURRENCY}"
        export SKIP_PLOT="${SKIP_PLOT}"
        export PYTHON_BIN="${PYTHON_BIN}"
        export GPU_MEM_TALKER="${GPU_MEM_TALKER}"
        export GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV}"
        bash "${SCRIPT_DIR}/run_benchmark.sh" --async-only
    )
}

run_offline() {
    echo "============================================================"
    echo " Running NPU offline profiling"
    echo "============================================================"

    local resolved_stage_config
    local patched_stage_config
    resolved_stage_config="$(resolve_stage_config_path "${OFFLINE_STAGE_CONFIG}")"
    patched_stage_config="$(prepare_offline_config "${resolved_stage_config}")"

    local -a cmd
    cmd=(
        "${PYTHON_BIN}"
        "${PROJECT_ROOT}/examples/offline_inference/qwen3_tts/end2end.py"
        "--model" "${MODEL}"
        "--query-type" "${OFFLINE_QUERY_TYPE}"
        "--mode-tag" "${OFFLINE_MODE_TAG}"
        "--stage-configs-path" "${patched_stage_config}"
        "--output-dir" "${OFFLINE_OUTPUT_DIR}"
        "--num-prompts" "${OFFLINE_NUM_PROMPTS}"
        "--batch-size" "${OFFLINE_BATCH_SIZE}"
        "--enable-profiler"
        "--profiler-wait-secs" "${PROFILE_WAIT_SECS}"
    )

    if [ -n "${PROFILER_STAGES}" ]; then
        local normalized="${PROFILER_STAGES//,/ }"
        cmd+=("--profiler-stages")
        for stage in ${normalized}; do
            cmd+=("${stage}")
        done
    fi

    if [ -n "${OFFLINE_TXT_PROMPTS}" ]; then
        cmd+=("--txt-prompts" "${OFFLINE_TXT_PROMPTS}")
    fi

    if [ "${OFFLINE_STREAMING}" = "1" ]; then
        cmd+=("--streaming")
    fi

    if [ "${OFFLINE_USE_BATCH_SAMPLE}" = "1" ]; then
        cmd+=("--use-batch-sample")
    fi

    env \
        ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}" \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        "${cmd[@]}"

    echo "Offline profiler traces: ${OFFLINE_PROFILER_DIR}"
}

case "${PROFILE_MODE}" in
    online)
        run_online
        ;;
    offline)
        run_offline
        ;;
    both)
        run_online
        run_offline
        ;;
    *)
        echo "Unsupported PROFILE_MODE: ${PROFILE_MODE}. Expected online, offline, or both."
        exit 1
        ;;
esac

echo "============================================================"
echo " NPU profiling finished"
echo "============================================================"
echo " Result root: ${RESULT_ROOT}"
echo " Stack capture: ${PROFILER_WITH_STACK}"
echo " Online traces: ${ONLINE_PROFILER_DIR}"
echo " Offline traces: ${OFFLINE_PROFILER_DIR}"
echo " Analyse online hint: from torch_npu.profiler.profiler import analyse; analyse('${ONLINE_PROFILER_DIR}')"
echo " Analyse offline hint: from torch_npu.profiler.profiler import analyse; analyse('${OFFLINE_PROFILER_DIR}')"
