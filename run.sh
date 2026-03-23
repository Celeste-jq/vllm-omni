# Point Python at VoxCPM's ``src`` (parent of ``voxcpm/model`` and ``voxcpm/modules``) if not next to this repo.
export VLLM_OMNI_VOXCPM_CODE_PATH=/home/l00613087/voxcpm/VoxCPM/src

export VOXCPM_MODEL=/home/l00613087/voxcpm/weights/VoxCPM1.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"
DEFAULT_NO_ASYNC="${SCRIPT_DIR}/vllm_omni/model_executor/stage_configs/voxcpm_no_async_chunk.yaml"

# One-shot split-stage (non-streaming yaml; export VOXCPM_STAGE_CONFIG to override).
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path "${VOXCPM_STAGE_CONFIG:-$DEFAULT_NO_ASYNC}" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."

# Async_chunk streaming smoke test (AsyncOmni + ``voxcpm.yaml`` with connectors).
python examples/offline_inference/voxcpm/end2end_streaming.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."