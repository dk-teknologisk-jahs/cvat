#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Unified ONNX Smoke Tests

Quick sanity checks that don't require HuggingFace authentication.
Just verifies the ONNX models load and produce outputs of expected shapes.

Usage:
    python test_smoke.py --model-dir ./onnx-exports
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def print_ok(msg):
    print(f"  \033[92m✓\033[0m {msg}")


def print_fail(msg):
    print(f"  \033[91m✗\033[0m {msg}")


def print_info(msg):
    print(f"  \033[94mℹ\033[0m {msg}")


def test_vision_encoder(model_path: Path, device: str = "cpu") -> bool:
    """Test vision encoder loads and produces correct output shapes."""
    import onnxruntime as ort

    print("\n[Vision Encoder]")

    if not model_path.exists():
        print_fail(f"Not found: {model_path}")
        return False

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        print_ok(f"Loaded: {model_path.name}")
    except Exception as e:
        print_fail(f"Load failed: {e}")
        return False

    # Test input
    test_input = np.random.randn(1, 3, 1008, 1008).astype(np.float32)
    input_name = session.get_inputs()[0].name

    try:
        start = time.time()
        outputs = session.run(None, {input_name: test_input})
        elapsed = time.time() - start
        print_ok(f"Inference: {elapsed*1000:.1f}ms")
    except Exception as e:
        print_fail(f"Inference failed: {e}")
        return False

    # Check output shapes
    expected = [
        ("fpn_feat_0", (1, 256, 288, 288)),
        ("fpn_feat_1", (1, 256, 144, 144)),
        ("fpn_feat_2", (1, 256, 72, 72)),
        ("fpn_pos_2", (1, 256, 72, 72)),
    ]

    for i, (name, shape) in enumerate(expected):
        if i < len(outputs):
            if outputs[i].shape == shape:
                print_ok(f"Output {name}: {outputs[i].shape}")
            else:
                print_fail(f"Output {name}: expected {shape}, got {outputs[i].shape}")
                return False
        else:
            print_fail(f"Missing output {name}")
            return False

    return True


def test_tracker_decoder(model_path: Path, device: str = "cpu") -> bool:
    """Test tracker decoder loads and produces correct output shapes."""
    import onnxruntime as ort

    print("\n[Tracker Decoder]")

    if not model_path.exists():
        print_fail(f"Not found: {model_path}")
        return False

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        print_ok(f"Loaded: {model_path.name}")
    except Exception as e:
        print_fail(f"Load failed: {e}")
        return False

    # Test inputs
    # Note: HuggingFace SAM3 expects 4D point_coords [B, num_objects, num_points, 2]
    test_inputs = {
        "fpn_feat_0": np.random.randn(1, 256, 288, 288).astype(np.float32),
        "fpn_feat_1": np.random.randn(1, 256, 144, 144).astype(np.float32),
        "fpn_feat_2": np.random.randn(1, 256, 72, 72).astype(np.float32),
        "point_coords": np.array([[[[504.0, 504.0]]]], dtype=np.float32),  # [B, num_objects, num_points, 2]
        "point_labels": np.array([[[1.0]]], dtype=np.float32),  # [B, num_objects, num_points]
        "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
        "has_mask_input": np.array([0.0], dtype=np.float32),
    }

    try:
        start = time.time()
        outputs = session.run(None, test_inputs)
        elapsed = time.time() - start
        print_ok(f"Inference: {elapsed*1000:.1f}ms")
    except Exception as e:
        print_fail(f"Inference failed: {e}")
        return False

    # Check output shapes
    expected = [
        ("masks", (1, 3, 1008, 1008)),
        ("iou_predictions", (1, 3)),
        ("low_res_masks", (1, 3, 288, 288)),
        ("object_score_logits", (1, 1)),
    ]

    for i, (name, shape) in enumerate(expected):
        if i < len(outputs):
            if outputs[i].shape == shape:
                print_ok(f"Output {name}: {outputs[i].shape}")
            else:
                print_fail(f"Output {name}: expected {shape}, got {outputs[i].shape}")
                return False
        else:
            print_fail(f"Missing output {name}")
            return False

    return True


def test_text_encoder(model_path: Path, device: str = "cpu") -> bool:
    """Test text encoder loads and produces correct output shapes."""
    import onnxruntime as ort

    print("\n[Text Encoder]")

    if not model_path.exists():
        print_info(f"Not found: {model_path} (skipped - optional model)")
        return None  # Return None to indicate skipped (not True=pass, not False=fail)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        print_ok(f"Loaded: {model_path.name}")
    except Exception as e:
        print_fail(f"Load failed: {e}")
        return False

    # Test inputs (fake tokens) - SAM3 uses 32 token context length
    batch_size = 1
    seq_len = 32  # SAM3 context length, not CLIP's 77
    test_inputs = {
        "input_ids": np.ones((batch_size, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
    }

    try:
        start = time.time()
        outputs = session.run(None, test_inputs)
        elapsed = time.time() - start
        print_ok(f"Inference: {elapsed*1000:.1f}ms")
    except Exception as e:
        print_fail(f"Inference failed: {e}")
        return False

    # Check we got outputs
    if len(outputs) >= 2:
        print_ok(f"Output text_features: {outputs[0].shape}")
        print_ok(f"Output text_mask: {outputs[1].shape}")
    else:
        print_fail(f"Expected 2 outputs, got {len(outputs)}")
        return False

    return True


def test_unified_handler(model_dir: Path, device: str = "cpu") -> bool:
    """Test the unified handler module."""
    print("\n[Unified Handler]")

    # Set environment variables
    os.environ["SAM3_MODEL_DIR"] = str(model_dir)
    os.environ["SAM3_DEVICE"] = device
    os.environ["SAM3_VISION_ENCODER"] = str(model_dir / "vision_encoder.onnx")
    os.environ["SAM3_TEXT_ENCODER"] = str(model_dir / "text_encoder.onnx")
    os.environ["SAM3_PCS_DECODER"] = str(model_dir / "pcs_decoder.onnx")
    os.environ["SAM3_TRACKER_DECODER"] = str(model_dir / "tracker_decoder.onnx")

    # Import handler
    handler_dir = Path(__file__).parent
    sys.path.insert(0, str(handler_dir))

    try:
        from model_handler import get_handler, reset_handler
        print_ok("Imported model_handler")
    except ImportError as e:
        print_fail(f"Import failed: {e}")
        return False

    # Get handler instance (reset first to pick up SAM3_DEVICE env var)
    try:
        reset_handler()
        handler = get_handler()
        print_ok("Created handler instance")
    except Exception as e:
        print_fail(f"Handler creation failed: {e}")
        return False

    # Test get_model_info
    try:
        info = handler.get_model_info()
        print_ok(f"get_model_info: vision={info.get('vision_encoder')}, tracker={info.get('tracker_decoder')}")
    except Exception as e:
        print_fail(f"get_model_info failed: {e}")
        return False

    # Test encode with synthetic image
    try:
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        start = time.time()
        embeddings = handler.encode(img)
        elapsed = time.time() - start
        print_ok(f"encode: {len(embeddings)} outputs in {elapsed*1000:.1f}ms")

        for name, arr in embeddings.items():
            print_info(f"  {name}: {arr.shape}")
    except Exception as e:
        print_fail(f"encode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="SAM3 ONNX Smoke Tests")
    parser.add_argument("--model-dir", type=str, required=True, help="ONNX model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = args.device

    print("=" * 60)
    print("SAM3 Unified ONNX Smoke Tests")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")

    results = []

    # Run tests
    results.append(("Vision Encoder", test_vision_encoder(model_dir / "vision_encoder.onnx", device)))
    results.append(("Tracker Decoder", test_tracker_decoder(model_dir / "tracker_decoder.onnx", device)))
    results.append(("Text Encoder", test_text_encoder(model_dir / "text_encoder.onnx", device)))
    results.append(("Unified Handler", test_unified_handler(model_dir, device)))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    skipped_count = 0
    for name, passed in results:
        if passed is None:
            status = "\033[93mSKIP\033[0m"
            skipped_count += 1
        elif passed:
            status = "\033[92mPASS\033[0m"
        else:
            status = "\033[91mFAIL\033[0m"
            all_passed = False
        print(f"  {name}: {status}")

    print()
    if all_passed:
        if skipped_count > 0:
            print(f"\033[92m\033[1mAll smoke tests passed! ({skipped_count} skipped)\033[0m")
        else:
            print("\033[92m\033[1mAll smoke tests passed!\033[0m")
    else:
        print("\033[91m\033[1mSome tests failed!\033[0m")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
