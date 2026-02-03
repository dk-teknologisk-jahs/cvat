#!/usr/bin/env python3
"""
SAM3 ONNX Test Suite - Unified Test Runner

This is the single entry point for all SAM3 ONNX model tests.

Test Suites Available:
1. encoder-decoder - Test vision encoder and decoder with synthetic/real images
2. memory - Test memory components for video propagation (16 tests)
3. decoder-compat - Test decoder browser compatibility (SAM2/SAM3 paths)
4. text-encoder - Test text encoder for PCS text-to-segment mode

Usage:
    # Run all tests
    python test_onnx.py --all

    # Run specific test suite
    python test_onnx.py --suite encoder-decoder
    python test_onnx.py --suite memory
    python test_onnx.py --suite decoder-compat
    python test_onnx.py --suite text-encoder

    # Run with specific options (passed to underlying test)
    python test_onnx.py --suite encoder-decoder --image /path/to/test.jpg
    python test_onnx.py --suite memory --verbose

    # List available tests
    python test_onnx.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Test suite configuration
TEST_SUITES = {
    "encoder-decoder": {
        "script": "test_sam3_onnx.py",
        "description": "Vision encoder + decoder tests with synthetic shapes",
        "default_args": ["--synthetic"],
    },
    "memory": {
        "script": "test_memory_onnx.py",
        "description": "Memory components for video propagation (16 tests)",
        "default_args": [],
    },
    "decoder-compat": {
        "script": "test_decoder.py",
        "description": "Decoder browser compatibility (SAM2/SAM3 paths)",
        "default_args": ["--model", "all"],
    },
    "text-encoder": {
        "script": "test_text_encoder.py",
        "description": "Text encoder for PCS text-to-segment mode",
        "default_args": [],
    },
}


def run_test_suite(suite_name: str, extra_args: list) -> int:
    """Run a test suite and return exit code."""
    if suite_name not in TEST_SUITES:
        print(f"Error: Unknown test suite '{suite_name}'")
        print(f"Available: {', '.join(TEST_SUITES.keys())}")
        return 1

    suite = TEST_SUITES[suite_name]
    script_path = SCRIPT_DIR / suite["script"]

    if not script_path.exists():
        print(f"Error: Test script not found: {script_path}")
        return 1

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add default args unless user provided alternatives
    if extra_args:
        cmd.extend(extra_args)
    else:
        cmd.extend(suite["default_args"])

    print(f"\n{'='*60}")
    print(f"Running: {suite_name}")
    print(f"Description: {suite['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 ONNX Test Suite - Unified Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_onnx.py --all                          # Run all test suites
    python test_onnx.py --suite encoder-decoder        # Run encoder-decoder tests
    python test_onnx.py --suite memory --verbose       # Run memory tests with verbose output
    python test_onnx.py --list                         # List available test suites
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test suites",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(TEST_SUITES.keys()),
        help="Run specific test suite",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test suites",
    )

    # Parse known args, pass rest to test script
    args, extra_args = parser.parse_known_args()

    if args.list:
        print("Available test suites:")
        print("-" * 60)
        for name, config in TEST_SUITES.items():
            print(f"  {name}")
            print(f"    Script: {config['script']}")
            print(f"    Description: {config['description']}")
            print()
        return 0

    if args.all:
        total_failed = 0
        results = {}

        for suite_name in TEST_SUITES:
            ret = run_test_suite(suite_name, [])
            results[suite_name] = ret
            if ret != 0:
                total_failed += 1

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for suite_name, ret in results.items():
            status = "✅ PASS" if ret == 0 else "❌ FAIL"
            print(f"  {status}: {suite_name}")

        print(f"\n{len(results) - total_failed}/{len(results)} suites passed")
        return 1 if total_failed > 0 else 0

    if args.suite:
        return run_test_suite(args.suite, extra_args)

    # No suite specified - show help
    parser.print_help()
    print("\nTip: Use --list to see available test suites")
    return 1


if __name__ == "__main__":
    sys.exit(main())
