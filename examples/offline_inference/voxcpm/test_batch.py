"""Test script for verifying VoxCPM batch processing functionality."""

import json
import os
import sys
from pathlib import Path

def test_file_structure():
    """Verify that all required files exist."""
    print("Testing file structure...")
    
    voxcpm_dir = Path(__file__).parent
    required_files = [
        "end2end.py",
        "example_texts.txt",
        "example_batch.jsonl",
        "README.md",
        "run_examples.sh",
    ]
    
    missing_files = []
    for file in required_files:
        file_path = voxcpm_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"  ✓ {file}")
    
    if missing_files:
        print(f"\n✗ Missing files: {missing_files}")
        return False
    
    print("\n✓ All required files exist")
    return True


def test_example_files():
    """Verify that example files have valid content."""
    print("\nTesting example files...")
    
    voxcpm_dir = Path(__file__).parent
    
    # Test text file
    text_file = voxcpm_dir / "example_texts.txt"
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"  ✓ example_texts.txt contains {len(texts)} texts")
    
    # Test JSONL file
    jsonl_file = voxcpm_dir / "example_batch.jsonl"
    with open(jsonl_file, "r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"  ✓ example_batch.jsonl contains {len(items)} items")
    
    # Validate JSONL structure
    for i, item in enumerate(items):
        if "text" not in item:
            print(f"  ✗ Item {i+1} missing 'text' field")
            return False
        if "audio" not in item:
            print(f"  ✗ Item {i+1} missing 'audio' field")
            return False
    
    print("  ✓ All JSONL items have valid structure")
    return True


def test_imports():
    """Test that the script can be imported."""
    print("\nTesting imports...")
    
    try:
        import soundfile as sf
        print("  ✓ soundfile imported successfully")
    except ImportError:
        print("  ✗ soundfile not installed. Run: pip install soundfile")
        return False
    
    try:
        import torch
        print("  ✓ torch imported successfully")
    except ImportError:
        print("  ✗ torch not installed")
        return False
    
    try:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        print("  ✓ vllm imported successfully")
    except ImportError:
        print("  ✗ vllm not installed")
        return False
    
    try:
        from vllm_omni import Omni
        print("  ✓ vllm_omni imported successfully")
    except ImportError:
        print("  ✗ vllm_omni not installed")
        return False
    
    return True


def test_script_syntax():
    """Test that the script has valid Python syntax."""
    print("\nTesting script syntax...")
    
    voxcpm_dir = Path(__file__).parent
    script_path = voxcpm_dir / "end2end.py"
    
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, script_path, "exec")
        print("  ✓ Script has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def test_argument_parsing():
    """Test that argument parsing works correctly."""
    print("\nTesting argument parsing...")
    
    voxcpm_dir = Path(__file__).parent
    script_path = voxcpm_dir / "end2end.py"
    
    # Test with --help
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print("  ✓ --help works correctly")
        print("\n  Help output preview:")
        lines = result.stdout.split("\n")[:10]
        for line in lines:
            print(f"    {line}")
        return True
    else:
        print(f"  ✗ --help failed: {result.stderr}")
        return False


def main():
    print("="*60)
    print("VoxCPM Batch Processing - Test Suite")
    print("="*60)
    
    tests = [
        test_file_structure,
        test_example_files,
        test_imports,
        test_script_syntax,
        test_argument_parsing,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        print("\nThe VoxCPM batch processing implementation is ready to use.")
        print("\nTo test with actual model inference:")
        print("  1. Set VOXCPM_MODEL environment variable")
        print("  2. Install dependencies: pip install soundfile torch vllm")
        print("  3. Run: python examples/offline_inference/voxcpm/end2end.py --help")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
