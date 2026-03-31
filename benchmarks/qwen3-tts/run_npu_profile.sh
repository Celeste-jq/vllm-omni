#!/bin/bash
# Convenience wrapper for Qwen3-TTS profiling on Ascend NPU.
#
# Example:
#   MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   DEVICE_ID=0 \
#   PROFILER_STAGES="0 1" \
#   bash run_npu_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${MODEL:-}" ]; then
    echo "Please set MODEL to a local model path or HuggingFace model ID."
    echo "Example:"
    echo "  MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice bash run_npu_profile.sh"
    exit 1
fi

export DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
export DEVICE_ID="${DEVICE_ID:-${ASCEND_DEVICE_ID:-0}}"
export ENABLE_PROFILING="${ENABLE_PROFILING:-1}"
export PROFILER_WITH_STACK="${PROFILER_WITH_STACK:-1}"
export PROFILER_STAGES="${PROFILER_STAGES:-0 1}"
export PROFILER_DIR="${PROFILER_DIR:-${SCRIPT_DIR}/results/npu_profile}"
export NUM_PROMPTS="${NUM_PROMPTS:-1}"
export NUM_WARMUPS="${NUM_WARMUPS:-0}"
export CONCURRENCY="${CONCURRENCY:-1}"
export SKIP_PLOT="${SKIP_PLOT:-1}"

exec bash "${SCRIPT_DIR}/run_benchmark.sh" --async-only "$@"
