#!/bin/bash
# Example script demonstrating all VoxCPM inference modes

# Set your VoxCPM model path
export VOXCPM_MODEL=${VOXCPM_MODEL:-"/path/to/voxcpm-model"}

# Check if model path is set
if [ ! -d "$VOXCPM_MODEL" ]; then
    echo "Error: VOXCPM_MODEL not set or directory does not exist"
    echo "Please set the VOXCPM_MODEL environment variable to your VoxCPM model directory"
    exit 1
fi

echo "Using VoxCPM model: $VOXCPM_MODEL"
echo ""

# Example 1: Single text synthesis
echo "=== Example 1: Single text synthesis ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "Hello, this is a test of single text synthesis." \
  --output output_audio/single_test.wav

echo ""

# Example 2: Voice cloning with single reference
echo "=== Example 2: Voice cloning with single reference ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is synthesized with a cloned voice." \
  --prompt-audio examples/offline_inference/voxcpm/example.wav \
  --prompt-text "This is an example audio transcript for training." \
  --output output_audio/clone_test.wav

echo ""

# Example 3: Batch processing from text file
echo "=== Example 3: Batch processing from text file ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --input examples/offline_inference/voxcpm/example_texts.txt \
  --output-dir output_audio/batch_text

echo ""

# Example 4: Batch processing from text file with voice cloning
echo "=== Example 4: Batch processing from text file with voice cloning ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --input examples/offline_inference/voxcpm/example_texts.txt \
  --prompt-audio examples/offline_inference/voxcpm/example.wav \
  --prompt-text "This is an example audio transcript for training." \
  --output-dir output_audio/batch_clone

echo ""

# Example 5: Batch processing from JSONL file
echo "=== Example 5: Batch processing from JSONL file ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --jsonl-file examples/offline_inference/voxcpm/example_batch.jsonl \
  --output-dir output_audio/batch_jsonl

echo ""

# Example 6: Custom generation parameters
echo "=== Example 6: Custom generation parameters ==="
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This uses custom generation parameters." \
  --cfg-value 2.5 \
  --inference-timesteps 15 \
  --output output_audio/custom_params.wav

echo ""
echo "=== All examples completed ==="
echo "Output files are in the output_audio/ directory"
echo ""
echo "Summary of examples:"
echo "  1. Single text synthesis"
echo "  2. Voice cloning"
echo "  3. Batch processing (text file)"
echo "  4. Batch processing (text file + voice cloning)"
echo "  5. Batch processing (JSONL file)"
echo "  6. Custom generation parameters"
