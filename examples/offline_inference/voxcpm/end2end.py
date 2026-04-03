"""Offline split-stage VoxCPM inference with voice cloning and batch processing for vLLM Omni."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_async_chunk.yaml"


def _save_wav(output_dir: Path, request_id: str, mm: dict) -> None:
    import soundfile as sf
    import torch

    audio_data = mm.get("audio", mm.get("model_outputs"))
    if audio_data is None:
        raise ValueError("No audio output found in multimodal output.")
    
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    sr = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    
    if isinstance(audio_data, list):
        audio_tensor = torch.cat(audio_data, dim=-1)
    else:
        audio_tensor = torch.as_tensor(audio_data)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = output_dir / f"output_{request_id}.wav"
    sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
    print(f"Saved: {out_wav}")


def _build_synthesize_input(args) -> list[dict]:
    additional_information = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    
    return [{
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }]


def _build_clone_input(args) -> list[dict]:
    additional_information = {
        "text": [args.text],
        "ref_audio": [args.prompt_audio],
        "ref_text": [args.prompt_text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    
    return [{
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }]


def _build_batch_input(args) -> list[dict]:
    inputs = []
    
    if args.jsonl_file:
        with open(args.jsonl_file, "r", encoding="utf-8") as f:
            items = [json.loads(line.strip()) for line in f if line.strip()]
        
        for item in items:
            additional_information = {
                "text": [item["text"]],
                "cfg_value": [args.cfg_value],
                "inference_timesteps": [args.inference_timesteps],
                "min_len": [args.min_len],
                "max_new_tokens": [args.max_new_tokens],
            }
            
            if "audio" in item:
                additional_information["ref_audio"] = [item["audio"]]
            if "text" in item:
                additional_information["ref_text"] = [item["text"]]
            
            inputs.append({
                "prompt_token_ids": [1],
                "additional_information": additional_information,
            })
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        for text in texts:
            additional_information = {
                "text": [text],
                "cfg_value": [args.cfg_value],
                "inference_timesteps": [args.inference_timesteps],
                "min_len": [args.min_len],
                "max_new_tokens": [args.max_new_tokens],
            }
            
            if args.prompt_audio:
                additional_information["ref_audio"] = [args.prompt_audio]
            if args.prompt_text:
                additional_information["ref_text"] = [args.prompt_text]
            
            inputs.append({
                "prompt_token_ids": [1],
                "additional_information": additional_information,
            })
    
    return inputs


def parse_args():
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        FlexibleArgumentParser = argparse.ArgumentParser

    parser = FlexibleArgumentParser(
        description="Offline split-stage VoxCPM inference with voice cloning and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text synthesis
  python end2end.py --model "$VOXCPM_MODEL" --text "Hello world"

  # Voice cloning
  python end2end.py --model "$VOXCPM_MODEL" --text "Hello" --prompt-audio ref.wav --prompt-text "hi"

  # Batch processing from text file
  python end2end.py --model "$VOXCPM_MODEL" --input texts.txt --output-dir ./outs

  # Batch processing from text file with voice cloning
  python end2end.py --model "$VOXCPM_MODEL" --input texts.txt --prompt-audio ref.wav --prompt-text "reference" --output-dir ./outs

  # Batch processing from JSONL file
  python end2end.py --model "$VOXCPM_MODEL" --jsonl-file data.jsonl --output-dir ./outs
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXCPM_MODEL"),
        help="Local VoxCPM model directory. Defaults to $VOXCPM_MODEL.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_CONFIG),
        help="Stage config YAML path. Defaults to split-stage VoxCPM config.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (single or clone mode).",
    )
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input text file (batch mode only).",
    )
    input_group.add_argument(
        "--jsonl-file",
        type=str,
        default=None,
        help="Path to a JSONL file with audio/text pairs for batch processing.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (single or clone mode).",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=str,
        default="output_audio",
        help="Output directory (batch mode only).",
    )

    parser.add_argument(
        "--prompt-audio",
        "-pa",
        type=str,
        default=None,
        help="Reference audio file path (clone mode).",
    )
    parser.add_argument(
        "--prompt-text",
        "-pt",
        type=str,
        default=None,
        help="Reference text corresponding to audio (clone mode).",
    )

    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="CFG guidance scale (float, recommended 0.5-5.0, default: 2.0).",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Inference steps (int, 1-100, default: 10).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum generated token length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated token length.",
    )

    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout in seconds.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM Omni stats logging.",
    )

    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless $VOXCPM_MODEL is set")

    if args.jsonl_file:
        if args.prompt_audio or args.prompt_text:
            parser.error("--prompt-audio and --prompt-text are not compatible with --jsonl-file")
    else:
        if (args.prompt_audio is None) != (args.prompt_text is None):
            parser.error("--prompt-audio and --prompt-text must be provided together")

    if args.input:
        if not Path(args.input).exists():
            parser.error(f"Input file not found: {args.input}")

    if args.jsonl_file:
        if not Path(args.jsonl_file).exists():
            parser.error(f"JSONL file not found: {args.jsonl_file}")

    if args.prompt_audio:
        if not Path(args.prompt_audio).exists():
            parser.error(f"Reference audio not found: {args.prompt_audio}")

    return args


def main(args) -> None:
    from vllm_omni import Omni

    if args.text:
        output_dir = Path(args.output).parent if args.output else Path("output_audio")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Output directory: {output_dir}")

    if args.text:
        if args.prompt_audio:
            print(f"Mode: Voice cloning")
            inputs = _build_clone_input(args)
        else:
            print(f"Mode: Single text synthesis")
            inputs = _build_synthesize_input(args)
    elif args.jsonl_file:
        print(f"Mode: Batch processing from JSONL file")
        inputs = _build_batch_input(args)
    else:
        print(f"Mode: Batch processing from text file")
        if args.prompt_audio:
            print(f"Voice cloning: enabled")
        inputs = _build_batch_input(args)

    print(f"Total requests: {len(inputs)}")

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    saved_count = 0

    for i, prompt in enumerate(inputs, 1):
        try:
            print(f"\nProcessing {i}/{len(inputs)}...")
            for stage_outputs in omni.generate([prompt]):
                output = stage_outputs.request_output
                _save_wav(output_dir, output.request_id, output.outputs[0].multimodal_output)
                saved_count += 1
        except Exception as e:
            print(f"Failed on {i}: {e}")

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Generation finished in {elapsed:.2f}s")
    print(f"Success: {saved_count}/{len(inputs)}")
    print(f"Average time per sample: {elapsed/max(len(inputs), 1):.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(parse_args())
