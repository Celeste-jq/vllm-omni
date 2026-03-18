#!/bin/bash
# Launch vLLM-Omni server for a native VoxCPM model.
#
# Usage:
#   VOXCPM_MODEL=/path/to/model ./run_server.sh
#   ./run_server.sh /path/to/model
#   STAGE_MODE=split ./run_server.sh /path/to/model

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-${VOXCPM_MODEL:-}}"
PORT="${PORT:-8091}"
STAGE_MODE="${STAGE_MODE:-full}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [[ -z "$MODEL" ]]; then
    echo "Missing VoxCPM model path. Pass it as the first argument or set VOXCPM_MODEL." >&2
    exit 1
fi

case "$STAGE_MODE" in
    full)
        STAGE_CONFIG="$REPO_ROOT/vllm_omni/model_executor/stage_configs/voxcpm_full.yaml"
        ;;
    split)
        STAGE_CONFIG="$REPO_ROOT/vllm_omni/model_executor/stage_configs/voxcpm.yaml"
        ;;
    *)
        echo "Unsupported STAGE_MODE: $STAGE_MODE (expected: full or split)" >&2
        exit 1
        ;;
esac

echo "Starting VoxCPM server"
echo "  model: $MODEL"
echo "  stage config: $STAGE_CONFIG"
echo "  port: $PORT"

vllm-omni serve "$MODEL" \
    --stage-configs-path "$STAGE_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enforce-eager \
    --omni
