#!/bin/bash
# Qwen3-TTS Benchmark Runner
#
# Compares vllm-omni streaming serving vs HuggingFace transformers offline
# inference. Produces JSON results and comparison plots, and can optionally
# collect Omni profiler traces for CUDA or NPU runs.
#
# Usage:
#   # Full comparison (vllm-omni + HF):
#   bash run_benchmark.sh
#
#   # Only vllm-omni async_chunk config:
#   bash run_benchmark.sh --async-only
#
#   # Only HuggingFace baseline:
#   bash run_benchmark.sh --hf-only
#
#   # vllm-omni only (skip HF):
#   bash run_benchmark.sh --skip-hf
#
#   # Custom settings:
#   DEVICE_ID=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
#
#   # Use 1.7B model:
#   MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash run_benchmark.sh --async-only
#
#   # Use batch_size=4 config:
#   STAGE_CONFIG=vllm_omni/configs/qwen3_tts_bs4.yaml bash run_benchmark.sh --async-only
#
#   # Use NPU backend + local model path + stack profiling:
#   DEVICE_BACKEND=npu \
#   DEVICE_ID=0 \
#   MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   ENABLE_PROFILING=1 \
#   PROFILER_WITH_STACK=1 \
#   PROFILER_DIR=./results/npu_profile \
#   PROFILER_STAGES="0 1" \
#   NUM_PROMPTS=1 \
#   CONCURRENCY="1" \
#   NUM_WARMUPS=0 \
#   SKIP_PLOT=1 \
#   bash run_benchmark.sh --async-only
#
# Environment variables:
#   DEVICE_BACKEND   - "cuda" or "npu" (default: cuda)
#   DEVICE_ID        - Device index to expose to the server (default: 0)
#   GPU_DEVICE       - Backward-compatible alias of DEVICE_ID
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   MODEL            - Model name or local path (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
#   PORT             - Server port (default: 8000)
#   GPU_MEM_TALKER   - gpu_memory_utilization for talker stage (default: 0.3)
#   GPU_MEM_CODE2WAV - gpu_memory_utilization for code2wav stage (default: 0.2)
#   STAGE_CONFIG     - Path to stage config YAML
#   ENABLE_PROFILING - 1 to inject profiler_config and call /start_profile (default: 0)
#   PROFILER_DIR     - Directory for profiler traces (default: ./results/profiles/<backend>)
#   PROFILER_STAGES  - Optional space/comma-separated stages to start/stop (example: "0 1")
#   PROFILER_WITH_STACK - 1 to capture stack/modules info (default: 1)
#   PROFILE_WAIT_SECS - Extra wait after stop_profile for traces to flush (default: 30)
#   PYTHON_BIN       - Python executable to use (default: python3)
#   SKIP_PLOT        - 1 to skip result plotting (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
DEVICE_BACKEND="${DEVICE_BACKEND:-cuda}"
DEVICE_ID="${DEVICE_ID:-${GPU_DEVICE:-0}}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
PORT="${PORT:-8000}"
GPU_MEM_TALKER="${GPU_MEM_TALKER:-0.3}"
GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV:-0.2}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
PROFILER_DIR="${PROFILER_DIR:-${RESULT_DIR}/profiles/${DEVICE_BACKEND}}"
PROFILER_STAGES="${PROFILER_STAGES:-}"
PROFILER_WITH_STACK="${PROFILER_WITH_STACK:-1}"
PROFILE_WAIT_SECS="${PROFILE_WAIT_SECS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_PLOT="${SKIP_PLOT:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${DEVICE_BACKEND}" in
    cuda)
        DEVICE_ENV_VAR="CUDA_VISIBLE_DEVICES"
        DEFAULT_STAGE_CONFIG="vllm_omni/configs/qwen3_tts_bs1.yaml"
        ;;
    npu)
        DEVICE_ENV_VAR="ASCEND_RT_VISIBLE_DEVICES"
        DEFAULT_STAGE_CONFIG="vllm_omni/platforms/npu/stage_configs/qwen3_tts.yaml"
        ;;
    *)
        echo "Unsupported DEVICE_BACKEND: ${DEVICE_BACKEND}. Expected 'cuda' or 'npu'."
        exit 1
        ;;
esac

STAGE_CONFIG="${STAGE_CONFIG:-${DEFAULT_STAGE_CONFIG}}"

# Parse args
RUN_ASYNC=true
RUN_HF=true
for arg in "$@"; do
    case "$arg" in
        --async-only) RUN_HF=false ;;
        --hf-only) RUN_ASYNC=false ;;
        --skip-hf) RUN_HF=false ;;
    esac
done

mkdir -p "${RESULT_DIR}"

if [ "${DEVICE_BACKEND}" != "cuda" ] && [ "${RUN_HF}" = true ]; then
    echo "HF baseline is only wired for CUDA in this benchmark. Skipping HuggingFace run on ${DEVICE_BACKEND}."
    RUN_HF=false
fi

echo "============================================================"
echo " Qwen3-TTS Benchmark"
echo "============================================================"
echo " Backend:      ${DEVICE_BACKEND}"
echo " Device:       ${DEVICE_ID}"
echo " Model:        ${MODEL}"
echo " Prompts:      ${NUM_PROMPTS}"
echo " Concurrency:  ${CONCURRENCY}"
echo " Port:         ${PORT}"
echo " Stage config: ${STAGE_CONFIG}"
echo " Results:      ${RESULT_DIR}"
echo " Profiling:    ${ENABLE_PROFILING}"
if [ "${ENABLE_PROFILING}" = "1" ]; then
    echo " Profiler dir: ${PROFILER_DIR}"
    echo " Profiler stages: ${PROFILER_STAGES:-all enabled stages}"
    echo " Stack capture: ${PROFILER_WITH_STACK}"
fi
echo "============================================================"

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

build_profiler_stage_json() {
    if [ -z "${PROFILER_STAGES}" ]; then
        printf '%s\n' ""
        return 0
    fi
    local normalized="${PROFILER_STAGES//,/ }"
    local json='{"stages": ['
    local first=true
    local stage
    for stage in ${normalized}; do
        if [ "${first}" = true ]; then
            json="${json}${stage}"
            first=false
        else
            json="${json}, ${stage}"
        fi
    done
    json="${json}]}"
    printf '%s\n' "${json}"
}

# Prepare stage config with correct device, memory settings, and optional profiler.
prepare_config() {
    local config_template="$1"
    local config_name="$2"
    local output_path="${RESULT_DIR}/${config_name}_stage_config.yaml"

    "${PYTHON_BIN}" - <<'PY' "${config_template}" "${output_path}" "${DEVICE_ID}" "${GPU_MEM_TALKER}" "${GPU_MEM_CODE2WAV}" "${ENABLE_PROFILING}" "${PROFILER_DIR}" "${PROFILER_WITH_STACK}"
import sys
from pathlib import Path

config_template, output_path, device_id, talker_mem, code2wav_mem, enable_profiling, profiler_dir, profiler_with_stack = sys.argv[1:]

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

    if enable_profiling == "1" and stripped == "engine_args:":
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

    echo "${output_path}"
}

# Start server and wait for it to be ready
start_server() {
    local stage_config="$1"
    local config_name="$2"
    local log_file="${RESULT_DIR}/server_${config_name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting server with config: ${config_name}"
    echo "  Stage config: ${stage_config}"
    echo "  Log file: ${log_file}"

    env \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        "${DEVICE_ENV_VAR}=${DEVICE_ID}" \
        "${PYTHON_BIN}" -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
            --omni \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --stage-configs-path "${stage_config}" \
            --stage-init-timeout 120 \
            --trust-remote-code \
            --disable-log-stats \
            > "${log_file}" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    # Wait for server to be ready
    echo "  Waiting for server to be ready..."
    local max_wait=300
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        # Check if process is still alive
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "  ERROR: Server process died. Check log: ${log_file}"
            tail -20 "${log_file}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "  ERROR: Server did not start within ${max_wait}s. Check log: ${log_file}"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

# Stop the server
stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "  Stopping server (PID: ${SERVER_PID})..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        # Kill any remaining child processes on the port
        local pids
        pids=$(lsof -ti:${PORT} 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo "  Cleaning up remaining processes on port ${PORT}..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
        fi
        echo "  Server stopped."
        SERVER_PID=""
    fi
}

# Cleanup on exit
trap 'stop_server' EXIT

start_profiler() {
    if [ "${ENABLE_PROFILING}" != "1" ]; then
        return 0
    fi

    local payload
    payload="$(build_profiler_stage_json)"
    echo "  Starting profiler..."
    if [ -n "${payload}" ]; then
        curl -sf -X POST "http://127.0.0.1:${PORT}/start_profile" \
            -H "Content-Type: application/json" \
            -d "${payload}" > /dev/null
    else
        curl -sf -X POST "http://127.0.0.1:${PORT}/start_profile" > /dev/null
    fi
    echo "  Profiler started."
}

stop_profiler() {
    if [ "${ENABLE_PROFILING}" != "1" ]; then
        return 0
    fi

    local payload
    payload="$(build_profiler_stage_json)"
    echo "  Stopping profiler..."
    if [ -n "${payload}" ]; then
        curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" \
            -H "Content-Type: application/json" \
            -d "${payload}" > /dev/null
    else
        curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" > /dev/null
    fi
    echo "  Waiting ${PROFILE_WAIT_SECS}s for trace flush..."
    sleep "${PROFILE_WAIT_SECS}"
}

# Run benchmark for a given config
run_bench() {
    local config_name="$1"
    local config_template="$2"

    echo ""
    echo "============================================================"
    echo " Benchmarking: ${config_name}"
    echo "============================================================"

    local stage_config
    stage_config=$(prepare_config "${config_template}" "${config_name}")

    start_server "${stage_config}" "${config_name}"
    start_profiler

    # Convert concurrency string to args
    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/vllm_omni/bench_tts_serve.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --config-name "${config_name}" \
        --result-dir "${RESULT_DIR}"

    stop_profiler
    stop_server

    # Allow device memory to settle.
    sleep 5
}

# Run vllm-omni benchmark
if [ "${RUN_ASYNC}" = true ]; then
    RESOLVED_STAGE_CONFIG="$(resolve_stage_config_path "${STAGE_CONFIG}")"
    run_bench "async_chunk" "${RESOLVED_STAGE_CONFIG}"
fi

# Run HuggingFace baseline benchmark
if [ "${RUN_HF}" = true ]; then
    echo ""
    echo "============================================================"
    echo " Benchmarking: HuggingFace transformers (offline)"
    echo "============================================================"

    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/transformers/bench_tts_hf.py" \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --num-warmups "${NUM_WARMUPS}" \
        --gpu-device "${DEVICE_ID}" \
        --config-name "hf_transformers" \
        --result-dir "${RESULT_DIR}"

    # Allow device memory to settle.
    sleep 5
fi

# Plot results
if [ "${SKIP_PLOT}" != "1" ]; then
    echo ""
    echo "============================================================"
    echo " Generating plots..."
    echo "============================================================"

    RESULT_FILES=""
    LABELS=""

    if [ "${RUN_ASYNC}" = true ]; then
        ASYNC_FILE=$(ls -t "${RESULT_DIR}"/bench_async_chunk_*.json 2>/dev/null | head -1)
        if [ -n "${ASYNC_FILE}" ]; then
            RESULT_FILES="${ASYNC_FILE}"
            LABELS="async_chunk"
        fi
    fi

    if [ "${RUN_HF}" = true ]; then
        HF_FILE=$(ls -t "${RESULT_DIR}"/bench_hf_transformers_*.json 2>/dev/null | head -1)
        if [ -n "${HF_FILE}" ]; then
            if [ -n "${RESULT_FILES}" ]; then
                RESULT_FILES="${RESULT_FILES} ${HF_FILE}"
                LABELS="${LABELS} hf_transformers"
            else
                RESULT_FILES="${HF_FILE}"
                LABELS="hf_transformers"
            fi
        fi
    fi

    if [ -n "${RESULT_FILES}" ]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_results.py" \
            --results ${RESULT_FILES} \
            --labels ${LABELS} \
            --output "${RESULT_DIR}/qwen3_tts_benchmark_${TIMESTAMP}.png"
    fi
else
    echo "Skipping plot generation because SKIP_PLOT=1."
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
if [ "${ENABLE_PROFILING}" = "1" ]; then
    echo " Profiler traces: ${PROFILER_DIR}"
    if [ "${DEVICE_BACKEND}" = "npu" ]; then
        echo " NPU analyse hint: from torch_npu.profiler.profiler import analyse; analyse('${PROFILER_DIR}')"
    fi
fi
echo "============================================================"
