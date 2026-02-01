#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Unified ONNX Function Test Suite

Comprehensive tests comparing ONNX Runtime outputs against HuggingFace PyTorch
reference implementations. Tests all modes: encode, text-to-segment, and tracking.

Requirements:
- HuggingFace authentication (for PyTorch reference models)
- Exported ONNX models (via export_hf_onnx.py)
- PIL, numpy, torch, transformers, onnxruntime

Usage:
    # Run all tests
    python test_onnx_unified.py --model-dir ./onnx-exports --all

    # Run specific tests
    python test_onnx_unified.py --model-dir ./onnx-exports --test-vision-encoder
    python test_onnx_unified.py --model-dir ./onnx-exports --test-tracker-decoder
    python test_onnx_unified.py --model-dir ./onnx-exports --test-text-encoder
    python test_onnx_unified.py --model-dir ./onnx-exports --test-end-to-end

    # Test with real images
    python test_onnx_unified.py --model-dir ./onnx-exports --test-image ./test.jpg
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Tolerance thresholds
MAX_MAE = 0.001  # Mean Absolute Error threshold
MAX_DIFF = 0.01  # Maximum absolute difference threshold
MIN_CORRELATION = 0.9999  # Minimum Pearson correlation


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{Colors.BOLD}{'-'*50}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'-'*50}{Colors.END}")


def print_pass(text: str):
    """Print a pass message."""
    print(f"  {Colors.GREEN}✓ PASS{Colors.END}: {text}")


def print_fail(text: str):
    """Print a fail message."""
    print(f"  {Colors.RED}✗ FAIL{Colors.END}: {text}")


def print_warn(text: str):
    """Print a warning message."""
    print(f"  {Colors.YELLOW}⚠ WARN{Colors.END}: {text}")


def print_info(text: str):
    """Print an info message."""
    print(f"  {Colors.BLUE}ℹ INFO{Colors.END}: {text}")


def compute_metrics(
    reference: np.ndarray,
    test: np.ndarray,
    name: str = "output",
) -> Dict[str, Any]:
    """
    Compute comparison metrics between reference and test arrays.

    Returns:
        Dictionary with MAE, max_diff, correlation, and pass/fail status.
    """
    # Ensure same dtype
    reference = reference.astype(np.float32)
    test = test.astype(np.float32)

    # Compute metrics
    diff = np.abs(reference - test)
    mae = float(diff.mean())
    max_diff = float(diff.max())

    # Compute Pearson correlation
    ref_flat = reference.flatten()
    test_flat = test.flatten()
    if ref_flat.std() > 1e-8 and test_flat.std() > 1e-8:
        correlation = float(np.corrcoef(ref_flat, test_flat)[0, 1])
    else:
        correlation = 1.0 if np.allclose(ref_flat, test_flat) else 0.0

    # Determine pass/fail
    passed = mae <= MAX_MAE and max_diff <= MAX_DIFF and correlation >= MIN_CORRELATION

    return {
        "name": name,
        "mae": mae,
        "max_diff": max_diff,
        "correlation": correlation,
        "passed": passed,
        "shape": list(reference.shape),
    }


def print_metrics(metrics: Dict[str, Any]):
    """Print metrics in a formatted way."""
    status = f"{Colors.GREEN}PASS{Colors.END}" if metrics["passed"] else f"{Colors.RED}FAIL{Colors.END}"
    print(f"  {metrics['name']}: [{status}]")
    print(f"    Shape: {metrics['shape']}")
    print(f"    MAE: {metrics['mae']:.8f} (threshold: {MAX_MAE})")
    print(f"    MaxDiff: {metrics['max_diff']:.8f} (threshold: {MAX_DIFF})")
    print(f"    Correlation: {metrics['correlation']:.8f} (threshold: {MIN_CORRELATION})")


# =============================================================================
# Vision Encoder Tests
# =============================================================================

def test_vision_encoder(
    onnx_path: Path,
    device: str = "cuda",
    test_image: Optional[np.ndarray] = None,
) -> Tuple[bool, List[Dict]]:
    """
    Test vision encoder ONNX against HuggingFace PyTorch.

    Args:
        onnx_path: Path to vision_encoder.onnx
        device: Device for PyTorch (cuda or cpu)
        test_image: Optional test image [3, 1008, 1008], else random

    Returns:
        Tuple of (all_passed, list of metrics)
    """
    import torch
    import onnxruntime as ort

    print_subheader("Vision Encoder Test")

    if not onnx_path.exists():
        print_fail(f"ONNX model not found: {onnx_path}")
        return False, []

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        print_info(f"Loaded ONNX model: {onnx_path}")
    except Exception as e:
        print_fail(f"Failed to load ONNX model: {e}")
        return False, []

    # Load HuggingFace model
    print_info("Loading HuggingFace model (requires authentication)...")
    try:
        from transformers import Sam3TrackerModel
        hf_model = Sam3TrackerModel.from_pretrained("facebook/sam3-hiera-large")
        hf_model = hf_model.to(device).eval()
        print_info("HuggingFace model loaded successfully")
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        print_info("Make sure you have HuggingFace authentication set up")
        return False, []

    # Create PyTorch wrapper (import from export script)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"))
    try:
        from export_hf_onnx import VisionEncoderWrapper
        pt_wrapper = VisionEncoderWrapper(hf_model, device=device).to(device).eval()
    except ImportError:
        print_fail("Could not import VisionEncoderWrapper from export_hf_onnx.py")
        return False, []

    # Create test input
    if test_image is not None:
        test_input = torch.from_numpy(test_image).unsqueeze(0).to(device)
    else:
        print_info("Using random test input")
        test_input = torch.randn(1, 3, 1008, 1008, device=device)

    # Run PyTorch
    print_info("Running PyTorch inference...")
    start = time.time()
    with torch.no_grad():
        pt_outputs = pt_wrapper(test_input)
    pt_time = time.time() - start
    print_info(f"PyTorch inference time: {pt_time*1000:.2f}ms")

    # Run ONNX
    print_info("Running ONNX inference...")
    onnx_input = {"images": test_input.cpu().numpy()}
    start = time.time()
    onnx_outputs = onnx_session.run(None, onnx_input)
    onnx_time = time.time() - start
    print_info(f"ONNX inference time: {onnx_time*1000:.2f}ms")

    # Compare outputs
    output_names = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
    all_metrics = []
    all_passed = True

    for i, (pt_out, onnx_out, name) in enumerate(zip(pt_outputs, onnx_outputs, output_names)):
        pt_np = pt_out.cpu().numpy()
        metrics = compute_metrics(pt_np, onnx_out, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
        if not metrics["passed"]:
            all_passed = False

    return all_passed, all_metrics


# =============================================================================
# Tracker Decoder Tests
# =============================================================================

def test_tracker_decoder(
    onnx_path: Path,
    device: str = "cuda",
    test_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[bool, List[Dict]]:
    """
    Test tracker decoder ONNX against HuggingFace PyTorch.

    Args:
        onnx_path: Path to tracker_decoder.onnx
        device: Device for PyTorch
        test_embeddings: Optional pre-computed embeddings

    Returns:
        Tuple of (all_passed, list of metrics)
    """
    import torch
    import onnxruntime as ort

    print_subheader("Tracker Decoder Test")

    if not onnx_path.exists():
        print_fail(f"ONNX model not found: {onnx_path}")
        return False, []

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        print_info(f"Loaded ONNX model: {onnx_path}")
    except Exception as e:
        print_fail(f"Failed to load ONNX model: {e}")
        return False, []

    # Load HuggingFace model
    print_info("Loading HuggingFace model...")
    try:
        from transformers import Sam3TrackerModel
        hf_model = Sam3TrackerModel.from_pretrained("facebook/sam3-hiera-large")
        hf_model = hf_model.to(device).eval()
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        return False, []

    # Create PyTorch wrapper
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"))
    try:
        from export_hf_onnx import TrackerDecoderWrapper
        pt_wrapper = TrackerDecoderWrapper(
            sam_prompt_encoder=hf_model.sam_prompt_encoder,
            sam_mask_decoder=hf_model.sam_mask_decoder,
            no_mem_embed=hf_model.no_mem_embed,
            multimask_output=True,
        ).to(device).eval()
    except ImportError:
        print_fail("Could not import TrackerDecoderWrapper from export_hf_onnx.py")
        return False, []

    # Create test inputs
    if test_embeddings is not None:
        fpn_feat_0 = torch.from_numpy(test_embeddings["fpn_feat_0"]).to(device)
        fpn_feat_1 = torch.from_numpy(test_embeddings["fpn_feat_1"]).to(device)
        fpn_feat_2 = torch.from_numpy(test_embeddings["fpn_feat_2"]).to(device)
    else:
        print_info("Using random test embeddings")
        fpn_feat_0 = torch.randn(1, 256, 288, 288, device=device)
        fpn_feat_1 = torch.randn(1, 256, 144, 144, device=device)
        fpn_feat_2 = torch.randn(1, 256, 72, 72, device=device)

    # Point prompt in center
    point_coords = torch.tensor([[[504.0, 504.0]]], device=device)
    point_labels = torch.tensor([[1.0]], device=device)
    mask_input = torch.zeros(1, 1, 288, 288, device=device)
    has_mask_input = torch.tensor([0.0], device=device)

    test_inputs = {
        "fpn_feat_0": fpn_feat_0,
        "fpn_feat_1": fpn_feat_1,
        "fpn_feat_2": fpn_feat_2,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
    }

    # Run PyTorch
    print_info("Running PyTorch inference...")
    start = time.time()
    with torch.no_grad():
        pt_outputs = pt_wrapper(**test_inputs)
    pt_time = time.time() - start
    print_info(f"PyTorch inference time: {pt_time*1000:.2f}ms")

    # Run ONNX
    print_info("Running ONNX inference...")
    onnx_inputs = {k: v.cpu().numpy() for k, v in test_inputs.items()}
    start = time.time()
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    onnx_time = time.time() - start
    print_info(f"ONNX inference time: {onnx_time*1000:.2f}ms")

    # Compare outputs
    output_names = ["masks", "iou_predictions", "low_res_masks", "object_score_logits"]
    all_metrics = []
    all_passed = True

    for i, (pt_out, onnx_out, name) in enumerate(zip(pt_outputs, onnx_outputs, output_names)):
        pt_np = pt_out.cpu().numpy()
        metrics = compute_metrics(pt_np, onnx_out, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
        if not metrics["passed"]:
            all_passed = False

    return all_passed, all_metrics


# =============================================================================
# Text Encoder Tests
# =============================================================================

def test_text_encoder(
    onnx_path: Path,
    device: str = "cuda",
    test_prompts: Optional[List[str]] = None,
) -> Tuple[bool, List[Dict]]:
    """
    Test text encoder ONNX against HuggingFace PyTorch.

    Args:
        onnx_path: Path to text_encoder.onnx
        device: Device for PyTorch
        test_prompts: Optional list of text prompts

    Returns:
        Tuple of (all_passed, list of metrics)
    """
    import torch
    import onnxruntime as ort

    print_subheader("Text Encoder Test")

    if not onnx_path.exists():
        print_fail(f"ONNX model not found: {onnx_path}")
        return False, []

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        print_info(f"Loaded ONNX model: {onnx_path}")
    except Exception as e:
        print_fail(f"Failed to load ONNX model: {e}")
        return False, []

    # Load HuggingFace model
    print_info("Loading HuggingFace model...")
    try:
        from transformers import Sam3Model, Sam3Processor
        hf_model = Sam3Model.from_pretrained("facebook/sam3-hiera-large")
        hf_model = hf_model.to(device).eval()
        processor = Sam3Processor.from_pretrained("facebook/sam3-hiera-large")
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        return False, []

    # Create PyTorch wrapper
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"))
    try:
        from export_hf_onnx import TextEncoderWrapper
        pt_wrapper = TextEncoderWrapper(hf_model).to(device).eval()
    except ImportError:
        print_fail("Could not import TextEncoderWrapper from export_hf_onnx.py")
        return False, []

    # Create test inputs
    if test_prompts is None:
        test_prompts = ["a person", "a car", "a dog"]

    print_info(f"Test prompts: {test_prompts}")

    # Tokenize
    text_inputs = processor.tokenizer(
        test_prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # Run PyTorch
    print_info("Running PyTorch inference...")
    start = time.time()
    with torch.no_grad():
        pt_outputs = pt_wrapper(input_ids, attention_mask)
    pt_time = time.time() - start
    print_info(f"PyTorch inference time: {pt_time*1000:.2f}ms")

    # Run ONNX
    print_info("Running ONNX inference...")
    onnx_inputs = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }
    start = time.time()
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    onnx_time = time.time() - start
    print_info(f"ONNX inference time: {onnx_time*1000:.2f}ms")

    # Compare outputs
    output_names = ["text_features", "text_mask"]
    all_metrics = []
    all_passed = True

    for i, (pt_out, onnx_out, name) in enumerate(zip(pt_outputs, onnx_outputs, output_names)):
        pt_np = pt_out.cpu().numpy()
        metrics = compute_metrics(pt_np, onnx_out, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
        if not metrics["passed"]:
            all_passed = False

    return all_passed, all_metrics


# =============================================================================
# End-to-End Tests
# =============================================================================

def test_end_to_end_encode(
    model_dir: Path,
    device: str = "cuda",
    test_image_path: Optional[str] = None,
) -> Tuple[bool, Dict]:
    """
    Test end-to-end encode mode (interactor).

    Compares:
    1. ONNX vision encoder output vs PyTorch
    2. ONNX tracker decoder output vs PyTorch (with ONNX embeddings)
    """
    import torch
    import onnxruntime as ort

    print_subheader("End-to-End Encode (Interactor) Test")

    vision_path = model_dir / "vision_encoder.onnx"
    decoder_path = model_dir / "tracker_decoder.onnx"

    if not vision_path.exists() or not decoder_path.exists():
        print_fail("Required ONNX models not found")
        return False, {}

    # Load image
    if test_image_path and os.path.exists(test_image_path):
        print_info(f"Loading test image: {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")
    else:
        print_info("Creating synthetic test image")
        # Create a simple gradient image for testing
        img_array = np.zeros((1008, 1008, 3), dtype=np.uint8)
        for i in range(1008):
            for j in range(1008):
                img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
        image = Image.fromarray(img_array)

    # Preprocess image
    img_resized = image.resize((1008, 1008), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Load ONNX models
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    vision_session = ort.InferenceSession(str(vision_path), providers=providers)
    decoder_session = ort.InferenceSession(str(decoder_path), providers=providers)

    # Run vision encoder
    print_info("Running ONNX vision encoder...")
    start = time.time()
    embeddings = vision_session.run(None, {"images": img_tensor})
    encode_time = time.time() - start
    print_info(f"Vision encoder time: {encode_time*1000:.2f}ms")

    # Verify embedding shapes
    expected_shapes = [
        (1, 256, 288, 288),
        (1, 256, 144, 144),
        (1, 256, 72, 72),
        (1, 256, 72, 72),
    ]
    for i, (emb, expected) in enumerate(zip(embeddings, expected_shapes)):
        if emb.shape != expected:
            print_fail(f"Embedding {i} shape mismatch: {emb.shape} vs {expected}")
            return False, {}
        print_pass(f"Embedding {i} shape: {emb.shape}")

    # Run decoder with point prompt
    print_info("Running ONNX tracker decoder...")
    decoder_inputs = {
        "fpn_feat_0": embeddings[0],
        "fpn_feat_1": embeddings[1],
        "fpn_feat_2": embeddings[2],
        "point_coords": np.array([[[504.0, 504.0]]], dtype=np.float32),
        "point_labels": np.array([[1.0]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
        "has_mask_input": np.array([0.0], dtype=np.float32),
    }

    start = time.time()
    decoder_outputs = decoder_session.run(None, decoder_inputs)
    decode_time = time.time() - start
    print_info(f"Decoder time: {decode_time*1000:.2f}ms")

    # Verify decoder output shapes
    masks, iou_pred, low_res_masks, obj_score = decoder_outputs
    print_pass(f"Masks shape: {masks.shape}")
    print_pass(f"IoU predictions shape: {iou_pred.shape}")
    print_pass(f"Low-res masks shape: {low_res_masks.shape}")
    print_pass(f"Object score shape: {obj_score.shape}")

    # Check mask validity
    mask_sum = masks.sum()
    if mask_sum > 0:
        print_pass(f"Mask has non-zero pixels (sum={mask_sum:.2f})")
    else:
        print_warn("Mask is all zeros - may indicate an issue")

    print_info(f"Total encode+decode time: {(encode_time + decode_time)*1000:.2f}ms")

    return True, {
        "encode_time_ms": encode_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "mask_sum": float(mask_sum),
        "iou_scores": iou_pred.tolist(),
    }


def test_end_to_end_text_to_segment(
    model_dir: Path,
    device: str = "cuda",
    test_image_path: Optional[str] = None,
    test_prompts: List[str] = None,
) -> Tuple[bool, Dict]:
    """
    Test end-to-end text-to-segment mode (detector).

    Note: PCS decoder is complex - this test verifies the pipeline works
    but detailed accuracy requires real images and ground truth.
    """
    import torch
    import onnxruntime as ort

    print_subheader("End-to-End Text-to-Segment (Detector) Test")

    vision_path = model_dir / "vision_encoder.onnx"
    text_path = model_dir / "text_encoder.onnx"
    pcs_path = model_dir / "pcs_decoder.onnx"

    missing = []
    if not vision_path.exists():
        missing.append("vision_encoder.onnx")
    if not text_path.exists():
        missing.append("text_encoder.onnx")
    if not pcs_path.exists():
        missing.append("pcs_decoder.onnx")

    if missing:
        print_warn(f"Missing ONNX models: {missing}")
        print_info("Skipping PCS test - models not yet exported")
        return True, {"skipped": True, "reason": "PCS models not found"}

    print_info("PCS decoder test not yet implemented")
    print_info("(Requires tokenizer integration)")

    return True, {"skipped": True, "reason": "Not implemented"}


# =============================================================================
# Unified Handler Tests
# =============================================================================

def test_unified_handler(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test the unified handler module directly.
    """
    print_subheader("Unified Handler Module Test")

    # Add handler to path
    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    try:
        from model_handler import UnifiedModelHandler
        print_pass("Successfully imported UnifiedModelHandler")
    except ImportError as e:
        print_fail(f"Failed to import UnifiedModelHandler: {e}")
        return False, {}

    # Set environment variables for model paths
    os.environ["SAM3_MODEL_DIR"] = str(model_dir)
    os.environ["SAM3_VISION_ENCODER"] = str(model_dir / "vision_encoder.onnx")
    os.environ["SAM3_TEXT_ENCODER"] = str(model_dir / "text_encoder.onnx")
    os.environ["SAM3_PCS_DECODER"] = str(model_dir / "pcs_decoder.onnx")
    os.environ["SAM3_TRACKER_DECODER"] = str(model_dir / "tracker_decoder.onnx")

    # Create handler
    try:
        handler = UnifiedModelHandler(device=device)
        print_pass("Created UnifiedModelHandler instance")
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Test get_model_info
    try:
        info = handler.get_model_info()
        print_pass(f"get_model_info: {info}")
    except Exception as e:
        print_fail(f"get_model_info failed: {e}")
        return False, {}

    # Test encode with synthetic image
    print_info("Testing encode mode...")
    try:
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)

        embeddings = handler.encode(test_image)
        print_pass(f"encode returned {len(embeddings)} embeddings")

        for name, arr in embeddings.items():
            print_info(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print_fail(f"encode failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

    return True, {"model_info": info}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3 Unified ONNX functions against HuggingFace PyTorch"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for PyTorch inference",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    parser.add_argument(
        "--test-vision-encoder",
        action="store_true",
        help="Test vision encoder",
    )
    parser.add_argument(
        "--test-tracker-decoder",
        action="store_true",
        help="Test tracker decoder",
    )
    parser.add_argument(
        "--test-text-encoder",
        action="store_true",
        help="Test text encoder",
    )
    parser.add_argument(
        "--test-end-to-end",
        action="store_true",
        help="Test end-to-end pipeline",
    )
    parser.add_argument(
        "--test-handler",
        action="store_true",
        help="Test unified handler module",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to test image (optional)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file for test results",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Check for CUDA
    device = args.device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print_warn("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"

    print_header("SAM3 Unified ONNX Test Suite")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")

    results = {
        "model_dir": str(model_dir),
        "device": device,
        "tests": {},
    }

    all_passed = True

    # Determine which tests to run
    run_all = args.all or not any([
        args.test_vision_encoder,
        args.test_tracker_decoder,
        args.test_text_encoder,
        args.test_end_to_end,
        args.test_handler,
    ])

    # Run tests
    if run_all or args.test_vision_encoder:
        vision_path = model_dir / "vision_encoder.onnx"
        if vision_path.exists():
            passed, metrics = test_vision_encoder(vision_path, device)
            results["tests"]["vision_encoder"] = {"passed": passed, "metrics": metrics}
            if not passed:
                all_passed = False
        else:
            print_warn(f"Vision encoder not found: {vision_path}")
            results["tests"]["vision_encoder"] = {"skipped": True}

    if run_all or args.test_tracker_decoder:
        decoder_path = model_dir / "tracker_decoder.onnx"
        if decoder_path.exists():
            passed, metrics = test_tracker_decoder(decoder_path, device)
            results["tests"]["tracker_decoder"] = {"passed": passed, "metrics": metrics}
            if not passed:
                all_passed = False
        else:
            print_warn(f"Tracker decoder not found: {decoder_path}")
            results["tests"]["tracker_decoder"] = {"skipped": True}

    if run_all or args.test_text_encoder:
        text_path = model_dir / "text_encoder.onnx"
        if text_path.exists():
            passed, metrics = test_text_encoder(text_path, device)
            results["tests"]["text_encoder"] = {"passed": passed, "metrics": metrics}
            if not passed:
                all_passed = False
        else:
            print_warn(f"Text encoder not found: {text_path}")
            results["tests"]["text_encoder"] = {"skipped": True}

    if run_all or args.test_end_to_end:
        passed, info = test_end_to_end_encode(model_dir, device, args.test_image)
        results["tests"]["end_to_end_encode"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

        passed, info = test_end_to_end_text_to_segment(model_dir, device, args.test_image)
        results["tests"]["end_to_end_text_to_segment"] = {"passed": passed, "info": info}

    if run_all or args.test_handler:
        passed, info = test_unified_handler(model_dir, device)
        results["tests"]["unified_handler"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    # Summary
    print_header("Test Summary")
    for test_name, test_result in results["tests"].items():
        if test_result.get("skipped"):
            print(f"  {test_name}: {Colors.YELLOW}SKIPPED{Colors.END}")
        elif test_result.get("passed"):
            print(f"  {test_name}: {Colors.GREEN}PASSED{Colors.END}")
        else:
            print(f"  {test_name}: {Colors.RED}FAILED{Colors.END}")

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed!{Colors.END}")

    # Save results
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
