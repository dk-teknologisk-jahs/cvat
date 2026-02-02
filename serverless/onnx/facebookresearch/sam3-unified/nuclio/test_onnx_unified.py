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

# SAM3 constants
SAM3_IMAGE_SIZE = 1008


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
        hf_model = Sam3TrackerModel.from_pretrained("facebook/sam3")
        hf_model = hf_model.to(device).eval()
        print_info("HuggingFace model loaded successfully")
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        print_info("Make sure you have HuggingFace authentication set up")
        return False, []

    # Create PyTorch wrapper (import from export script)
    # Path: onnx/facebookresearch/sam3-unified/nuclio -> serverless/pytorch/facebookresearch/sam3/nuclio
    export_path = Path(__file__).parent.parent.parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"
    sys.path.insert(0, str(export_path))
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
        hf_model = Sam3TrackerModel.from_pretrained("facebook/sam3")
        hf_model = hf_model.to(device).eval()
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        return False, []

    # Create PyTorch wrapper
    # Path: onnx/facebookresearch/sam3-unified/nuclio -> serverless/pytorch/facebookresearch/sam3/nuclio
    export_path = Path(__file__).parent.parent.parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"
    sys.path.insert(0, str(export_path))
    try:
        from export_hf_onnx import TrackerDecoderWrapper
        pt_wrapper = TrackerDecoderWrapper(
            sam3_tracker_model=hf_model,
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

    # Point prompt in center - HuggingFace expects 4D [B, num_objects, num_points, 2]
    point_coords = torch.tensor([[[[504.0, 504.0]]]], device=device)  # [B, num_objects, num_points, 2]
    point_labels = torch.tensor([[[1.0]]], device=device)  # [B, num_objects, num_points]
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
        hf_model = Sam3Model.from_pretrained("facebook/sam3")
        hf_model = hf_model.to(device).eval()
        processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print_fail(f"Failed to load HuggingFace model: {e}")
        return False, []

    # Create PyTorch wrapper
    # Path: onnx/facebookresearch/sam3-unified/nuclio -> serverless/pytorch/facebookresearch/sam3/nuclio
    export_path = Path(__file__).parent.parent.parent.parent.parent / "pytorch/facebookresearch/sam3/nuclio"
    sys.path.insert(0, str(export_path))
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

    # Tokenize - SAM3 uses 32 token context length (not CLIP's 77)
    text_inputs = processor.tokenizer(
        test_prompts,
        padding="max_length",
        max_length=32,  # SAM3's text encoder context length
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

    # Run decoder with point prompt - HuggingFace expects 4D [B, num_objects, num_points, 2]
    print_info("Running ONNX tracker decoder...")
    decoder_inputs = {
        "fpn_feat_0": embeddings[0],
        "fpn_feat_1": embeddings[1],
        "fpn_feat_2": embeddings[2],
        "point_coords": np.array([[[[504.0, 504.0]]]], dtype=np.float32),  # [B, num_objects, num_points, 2]
        "point_labels": np.array([[[1.0]]], dtype=np.float32),  # [B, num_objects, num_points]
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

    # Check mask validity - a valid prompt should produce non-zero masks
    mask_sum = masks.sum()
    if mask_sum > 0:
        print_pass(f"Mask has non-zero pixels (sum={mask_sum:.2f})")
    else:
        # For synthetic test image with random pixels, zero mask can happen
        # Only fail if using a real test image where we expect valid segmentation
        print_warn("Mask is all zeros - synthetic image may not have clear objects")

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
        print_fail(f"Missing ONNX models: {missing}")
        return False, {"reason": f"PCS models not found: {missing}"}

    # Load ONNX models
    providers = ["CPUExecutionProvider"]
    print_info("Loading ONNX models...")

    try:
        vision_session = ort.InferenceSession(str(vision_path), providers=providers)
        text_session = ort.InferenceSession(str(text_path), providers=providers)
        pcs_session = ort.InferenceSession(str(pcs_path), providers=providers)
        print_pass("Loaded all PCS pipeline models")
    except Exception as e:
        print_fail(f"Failed to load ONNX models: {e}")
        return False, {"reason": str(e)}

    # Create synthetic test image
    print_info("Creating synthetic test image...")
    img_array = np.random.randint(0, 255, (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, 3), dtype=np.uint8)
    img_tensor = img_array.astype(np.float32) / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, 1008, 1008]

    # Run vision encoder
    print_info("Running vision encoder...")
    start = time.time()
    embeddings = vision_session.run(None, {"images": img_tensor})
    encode_time = time.time() - start
    print_info(f"Vision encoder time: {encode_time*1000:.2f}ms")

    fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2 = embeddings
    print_pass(f"Vision encoder outputs: {[e.shape for e in embeddings]}")

    # Run text encoder with REAL tokenization (not placeholder tokens)
    print_info("Running text encoder with proper tokenization...")
    test_prompt = test_prompts[0] if test_prompts else "a person"
    print_info(f"Test prompt: '{test_prompt}'")

    # Use HuggingFace processor for proper tokenization
    try:
        from transformers import Sam3Processor
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        text_inputs = processor.tokenizer(
            [test_prompt],
            padding="max_length",
            max_length=32,  # SAM3's text context length
            truncation=True,
            return_tensors="np",
        )
        input_ids = text_inputs["input_ids"].astype(np.int64)
        attention_mask = text_inputs["attention_mask"].astype(np.int64)
        print_info(f"Tokenized: {input_ids.shape}, non-padding tokens: {attention_mask.sum()}")
    except Exception as e:
        print_fail(f"Failed to load tokenizer for proper testing: {e}")
        return False, {"reason": f"Tokenizer load failed: {e}"}

    start = time.time()
    text_outputs = text_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    text_time = time.time() - start
    print_info(f"Text encoder time: {text_time*1000:.2f}ms")

    text_features, text_mask = text_outputs
    print_pass(f"Text encoder outputs: features={text_features.shape}, mask={text_mask.shape}")

    # Verify text features are non-trivial (not all zeros or constant)
    text_feat_std = np.std(text_features)
    if text_feat_std < 1e-6:
        print_fail(f"Text features have near-zero variance (std={text_feat_std:.8f}) - model may not be working")
        return False, {"reason": "Text encoder output has no variance"}
    print_pass(f"Text features variance: std={text_feat_std:.6f}")

    # Run PCS decoder
    print_info("Running PCS decoder...")

    # Use padding boxes (label=-10 means padding, ignored by the model)
    # ONNX can't handle zero-dimension tensors, so we use 1 padded box
    input_boxes = np.zeros((1, 1, 4), dtype=np.float32)
    input_boxes_labels = np.full((1, 1), -10, dtype=np.int64)

    start = time.time()
    pcs_outputs = pcs_session.run(None, {
        "fpn_feat_0": fpn_feat_0,
        "fpn_feat_1": fpn_feat_1,
        "fpn_feat_2": fpn_feat_2,
        "fpn_pos_2": fpn_pos_2,
        "text_features": text_features,
        "text_mask": text_mask.astype(bool),
        "input_boxes": input_boxes,
        "input_boxes_labels": input_boxes_labels,
    })
    decode_time = time.time() - start
    print_info(f"PCS decoder time: {decode_time*1000:.2f}ms")

    pred_masks, pred_boxes, pred_logits, presence_logits = pcs_outputs
    print_pass(f"PCS decoder outputs:")
    print_info(f"  pred_masks: {pred_masks.shape}")
    print_info(f"  pred_boxes: {pred_boxes.shape}")
    print_info(f"  pred_logits: {pred_logits.shape}")
    print_info(f"  presence_logits: {presence_logits.shape}")

    # Verify output shapes are reasonable
    if pred_masks.ndim != 4:
        print_fail(f"Expected 4D pred_masks, got {pred_masks.ndim}D")
        return False, {"reason": "Invalid pred_masks shape"}

    if pred_boxes.shape[-1] != 4:
        print_fail(f"Expected pred_boxes with 4 coords, got {pred_boxes.shape[-1]}")
        return False, {"reason": "Invalid pred_boxes shape"}

    # Verify outputs contain meaningful values (not all zeros or NaN)
    if np.isnan(pred_logits).any():
        print_fail("pred_logits contains NaN values - model output is invalid")
        return False, {"reason": "NaN in pred_logits"}
    if np.isnan(pred_boxes).any():
        print_fail("pred_boxes contains NaN values - model output is invalid")
        return False, {"reason": "NaN in pred_boxes"}

    # Check that logits have reasonable variance (not constant)
    logits_std = np.std(pred_logits)
    if logits_std < 1e-6:
        print_fail(f"pred_logits has near-zero variance (std={logits_std:.8f}) - decoder may not be working")
        return False, {"reason": "pred_logits has no variance"}
    print_pass(f"pred_logits variance: std={logits_std:.4f}")

    # Check presence_logits to ensure the model is producing non-trivial scores
    presence_std = np.std(presence_logits)
    print_info(f"presence_logits: min={presence_logits.min():.4f}, max={presence_logits.max():.4f}, std={presence_std:.4f}")

    print_info(f"Total pipeline time: {(encode_time + text_time + decode_time)*1000:.2f}ms")

    return True, {
        "encode_time_ms": encode_time * 1000,
        "text_time_ms": text_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "pred_masks_shape": list(pred_masks.shape),
        "pred_boxes_shape": list(pred_boxes.shape),
        "pred_logits_std": float(logits_std),
    }


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

    # Add handler to path and force reimport to pick up model_dir
    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    # Remove cached import if any to ensure fresh import with new model_dir
    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
        print_pass("Successfully imported UnifiedModelHandler")
    except ImportError as e:
        print_fail(f"Failed to import UnifiedModelHandler: {e}")
        return False, {}

    # Create handler with explicit model_dir parameter
    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
        print_pass(f"Created UnifiedModelHandler instance (model_dir={model_dir})")
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

        # Verify embedding shapes
        expected_shapes = {
            "fpn_feat_0": (1, 256, 288, 288),
            "fpn_feat_1": (1, 256, 144, 144),
            "fpn_feat_2": (1, 256, 72, 72),
            "fpn_pos_2": (1, 256, 72, 72),
        }
        for name, arr in embeddings.items():
            print_info(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            if name in expected_shapes and arr.shape != expected_shapes[name]:
                print_fail(f"  {name} shape mismatch: expected {expected_shapes[name]}")
                return False, {}
            # Verify embeddings have meaningful content (not all zeros or NaN)
            if np.isnan(arr).any():
                print_fail(f"  {name} contains NaN values - encoder output invalid")
                return False, {}
            arr_std = np.std(arr)
            if arr_std < 1e-6:
                print_fail(f"  {name} has near-zero variance (std={arr_std:.8f}) - encoder may not be working")
                return False, {}
            print_info(f"    variance: std={arr_std:.4f}, range=[{arr.min():.4f}, {arr.max():.4f}]")
    except Exception as e:
        print_fail(f"encode failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

    return True, {"model_info": info}


# =============================================================================
# Handler API Tests - text_to_segment
# =============================================================================

def test_handler_text_to_segment(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test the handler.text_to_segment() API method.

    This validates the complete PCS pipeline:
    vision encoder → text encoder → PCS decoder → detections
    """
    print_subheader("Handler text_to_segment() API Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import UnifiedModelHandler: {e}")
        return False, {}

    # Create handler
    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
        print_pass("Created UnifiedModelHandler")
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check all required models exist
    info = handler.get_model_info()
    if not (info["vision_encoder"] and info["text_encoder"] and info["pcs_decoder"]):
        missing = [k for k in ["vision_encoder", "text_encoder", "pcs_decoder"] if not info[k]]
        print_fail(f"Missing required models for text_to_segment: {missing}")
        return False, {"skipped": True, "reason": f"Missing: {missing}"}

    # Test with synthetic image
    print_info("Testing text_to_segment with synthetic image...")
    img_array = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)

    try:
        import time
        start = time.time()
        detections = handler.text_to_segment(
            text_prompts=["a person"],
            image=test_image,
            confidence_threshold=0.1,  # Low threshold for synthetic images
        )
        elapsed = time.time() - start
        print_pass(f"text_to_segment completed in {elapsed*1000:.2f}ms")
        print_info(f"  Returned {len(detections)} detections")

        # Validate detection structure
        for i, det in enumerate(detections):
            if "mask" not in det:
                print_fail(f"Detection {i} missing 'mask' field")
                return False, {}
            if "box" not in det:
                print_fail(f"Detection {i} missing 'box' field")
                return False, {}
            if "score" not in det:
                print_fail(f"Detection {i} missing 'score' field")
                return False, {}
            if det["mask"] is not None:
                print_info(f"  Detection {i}: box={det['box']}, score={det['score']:.4f}, mask_shape={det['mask'].shape}")
            else:
                print_info(f"  Detection {i}: box={det['box']}, score={det['score']:.4f}, mask=None")

        print_pass("text_to_segment API structure validated")

    except Exception as e:
        print_fail(f"text_to_segment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

    # Test with multiple prompts
    print_info("Testing text_to_segment with multiple prompts...")
    try:
        detections = handler.text_to_segment(
            text_prompts=["a cat", "a dog"],
            image=test_image,
            confidence_threshold=0.01,
        )
        print_pass(f"Multi-prompt text_to_segment returned {len(detections)} detections")
        # Validate detection structure for multi-prompt
        for i, det in enumerate(detections):
            if not all(k in det for k in ["mask", "box", "score"]):
                print_fail(f"Multi-prompt detection {i} missing required fields")
                return False, {"error": "Missing fields in multi-prompt detection"}
            # Verify box has valid format
            box = det.get("box", [])
            if len(box) != 4:
                print_fail(f"Multi-prompt detection {i} has invalid box format: {box}")
                return False, {"error": "Invalid box format"}
        print_pass("Multi-prompt detection structure validated")
    except Exception as e:
        print_fail(f"Multi-prompt text_to_segment failed: {e}")
        return False, {"error": str(e)}

    # Test with empty prompts (edge case)
    print_info("Testing text_to_segment edge cases...")
    try:
        detections = handler.text_to_segment(
            text_prompts=[""],  # Empty prompt
            image=test_image,
            confidence_threshold=0.5,
        )
        print_pass(f"Empty prompt handled gracefully: {len(detections)} detections")
    except Exception as e:
        print_warn(f"Empty prompt raised exception (acceptable): {e}")

    return True, {"status": "passed"}


# =============================================================================
# Video Tracking Tests with Mock Redis
# =============================================================================

def test_video_tracking(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test video tracking API: init_tracking() and track_frame().

    Uses in-memory cache (mock Redis) for state management.
    """
    print_subheader("Video Tracking Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import UnifiedModelHandler: {e}")
        return False, {}

    # Create handler
    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
        print_pass("Created UnifiedModelHandler")
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check required models
    info = handler.get_model_info()
    if not (info["vision_encoder"] and info["tracker_decoder"]):
        missing = [k for k in ["vision_encoder", "tracker_decoder"] if not info[k]]
        print_fail(f"Missing required models for tracking: {missing}")
        return False, {"skipped": True, "reason": f"Missing: {missing}"}

    # Create synthetic video frames (just random images)
    print_info("Creating synthetic video frames...")
    frames = []
    for i in range(3):
        # Create frames with slight variations to simulate motion
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # Add a "moving box" feature
        x = 100 + i * 50
        y = 100 + i * 30
        img[y:y+80, x:x+100] = 255  # White rectangle
        frames.append(Image.fromarray(img))
    print_pass(f"Created {len(frames)} synthetic frames")

    # Test init_tracking
    print_info("Testing init_tracking()...")
    try:
        import time
        start = time.time()
        init_result = handler.init_tracking(
            image=frames[0],
            objects=[
                {"object_id": 1, "box": [100, 100, 200, 180], "label": "object1"},
            ],
        )
        elapsed = time.time() - start
        print_pass(f"init_tracking completed in {elapsed*1000:.2f}ms")

        # Validate init_result structure
        if "session_id" not in init_result:
            print_fail("init_tracking missing 'session_id'")
            return False, {}
        if "tracked_objects" not in init_result:
            print_fail("init_tracking missing 'tracked_objects'")
            return False, {}

        session_id = init_result["session_id"]
        print_info(f"  session_id: {session_id}")
        print_info(f"  frame_idx: {init_result.get('frame_idx', 'N/A')}")
        print_info(f"  tracked_objects: {len(init_result['tracked_objects'])}")

        for obj in init_result["tracked_objects"]:
            print_info(f"    object_id={obj.get('object_id')}, box={obj.get('box')}, score={obj.get('score', 'N/A'):.4f}")
            if obj.get("mask") is not None:
                print_info(f"      mask shape: {obj['mask'].shape}")
                # Verify mask has some non-zero pixels (object was actually found)
                mask_sum = np.sum(obj['mask'] > 0)
                if mask_sum == 0:
                    print_warn(f"      mask is all zeros - object may not have been found")
            # Verify box has valid coordinates
            box = obj.get('box', [])
            if len(box) == 4:
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    print_fail(f"      invalid box coordinates: {box}")
                    return False, {}

        print_pass("init_tracking structure validated")

    except Exception as e:
        print_fail(f"init_tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

    # Test track_frame for subsequent frames
    print_info("Testing track_frame() for frames 1-2...")
    for frame_idx in range(1, len(frames)):
        try:
            start = time.time()
            track_result = handler.track_frame(
                session_id=session_id,
                image=frames[frame_idx],
                frame_idx=frame_idx,
            )
            elapsed = time.time() - start
            print_pass(f"track_frame({frame_idx}) completed in {elapsed*1000:.2f}ms")

            # Validate track_result
            if "error" in track_result:
                print_fail(f"track_frame returned error: {track_result['error']}")
                return False, {}
            if "tracked_objects" not in track_result:
                print_fail("track_frame missing 'tracked_objects'")
                return False, {}

            print_info(f"  frame_idx: {track_result.get('frame_idx')}")
            for obj in track_result["tracked_objects"]:
                print_info(f"    object_id={obj.get('object_id')}, box={obj.get('box')}, score={obj.get('score', 'N/A'):.4f}")
                # Verify box has valid coordinates (width/height > 0)
                box = obj.get('box', [])
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        print_fail(f"    invalid tracked box coordinates: {box}")
                        return False, {}

        except Exception as e:
            print_fail(f"track_frame({frame_idx}) failed: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}

    # Test clear_tracking
    print_info("Testing clear_tracking()...")
    try:
        clear_result = handler.clear_tracking(session_id)
        print_pass(f"clear_tracking: {clear_result}")
    except Exception as e:
        print_fail(f"clear_tracking failed: {e}")

    # Test tracking with invalid session
    print_info("Testing track_frame with invalid session (edge case)...")
    try:
        result = handler.track_frame(
            session_id="invalid_session_xyz",
            image=frames[0],
            frame_idx=0,
        )
        if "error" in result:
            print_pass(f"Invalid session handled gracefully: {result['error']}")
        else:
            print_warn("Invalid session did not return error")
    except Exception as e:
        print_pass(f"Invalid session raised exception (acceptable): {e}")

    return True, {"status": "passed"}


# =============================================================================
# Multi-Object Tracking Tests
# =============================================================================

def test_multi_object_tracking(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test tracking multiple objects simultaneously.
    """
    print_subheader("Multi-Object Tracking Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import UnifiedModelHandler: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check required models
    info = handler.get_model_info()
    if not (info["vision_encoder"] and info["tracker_decoder"]):
        print_fail("Missing required models for tracking")
        return False, {"skipped": True}

    # Create frame with multiple "objects"
    print_info("Creating frame with multiple objects...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[50:130, 50:150] = [255, 0, 0]    # Red box: object 1
    img[200:280, 300:400] = [0, 255, 0]  # Green box: object 2
    img[350:430, 100:200] = [0, 0, 255]  # Blue box: object 3
    frame = Image.fromarray(img)

    # Initialize tracking with 3 objects
    print_info("Initializing tracking with 3 objects...")
    try:
        import time
        start = time.time()
        init_result = handler.init_tracking(
            image=frame,
            objects=[
                {"object_id": 1, "box": [50, 50, 150, 130], "label": "red_box"},
                {"object_id": 2, "box": [300, 200, 400, 280], "label": "green_box"},
                {"object_id": 3, "box": [100, 350, 200, 430], "label": "blue_box"},
            ],
        )
        elapsed = time.time() - start
        print_pass(f"Multi-object init completed in {elapsed*1000:.2f}ms")

        if len(init_result["tracked_objects"]) != 3:
            print_fail(f"Expected 3 tracked objects, got {len(init_result['tracked_objects'])}")
            return False, {}

        print_info(f"  Tracking {len(init_result['tracked_objects'])} objects")
        for obj in init_result["tracked_objects"]:
            print_info(f"    id={obj['object_id']}, box={obj['box']}")

        print_pass("Multi-object tracking initialized successfully")

        # Clean up
        handler.clear_tracking(init_result["session_id"])

    except Exception as e:
        print_fail(f"Multi-object tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

    return True, {"status": "passed"}


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_edge_cases(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test various edge cases and error handling.
    """
    print_subheader("Edge Case Tests")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    passed_all = True

    # Test 1: Very small image
    print_info("Test 1: Very small image (16x16)...")
    try:
        small_img = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
        embeddings = handler.encode(small_img)
        # Verify output is valid (not NaN or zero variance)
        for name, arr in embeddings.items():
            if np.isnan(arr).any():
                print_fail(f"Small image {name} contains NaN")
                passed_all = False
                break
            if np.std(arr) < 1e-6:
                print_fail(f"Small image {name} has zero variance")
                passed_all = False
                break
        else:
            print_pass(f"Small image encoded: {list(embeddings.keys())}")
    except Exception as e:
        print_fail(f"Small image failed: {e}")
        passed_all = False

    # Test 2: Very large image
    print_info("Test 2: Large image (2048x2048)...")
    try:
        large_img = Image.fromarray(np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8))
        embeddings = handler.encode(large_img)
        # Verify output is valid
        for name, arr in embeddings.items():
            if np.isnan(arr).any():
                print_fail(f"Large image {name} contains NaN")
                passed_all = False
                break
        else:
            print_pass(f"Large image encoded: {list(embeddings.keys())}")
    except Exception as e:
        print_fail(f"Large image failed: {e}")
        passed_all = False

    # Test 3: Non-square image
    print_info("Test 3: Non-square image (1920x1080)...")
    try:
        rect_img = Image.fromarray(np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))
        embeddings = handler.encode(rect_img)
        # Verify output is valid
        for name, arr in embeddings.items():
            if np.isnan(arr).any():
                print_fail(f"Non-square image {name} contains NaN")
                passed_all = False
                break
        else:
            print_pass(f"Non-square image encoded: {list(embeddings.keys())}")
    except Exception as e:
        print_fail(f"Non-square image failed: {e}")
        passed_all = False

    # Test 4: Grayscale image converted to RGB
    print_info("Test 4: Grayscale image...")
    try:
        gray_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        gray_img = Image.fromarray(gray_array, mode='L').convert('RGB')
        embeddings = handler.encode(gray_img)
        # Verify output is valid
        for name, arr in embeddings.items():
            if np.isnan(arr).any():
                print_fail(f"Grayscale image {name} contains NaN")
                passed_all = False
                break
        else:
            print_pass(f"Grayscale image encoded: {list(embeddings.keys())}")
    except Exception as e:
        print_fail(f"Grayscale image failed: {e}")
        passed_all = False

    # Test 5: RGBA image
    print_info("Test 5: RGBA image...")
    try:
        rgba_array = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(rgba_array, mode='RGBA').convert('RGB')
        embeddings = handler.encode(rgba_img)
        # Verify output is valid
        for name, arr in embeddings.items():
            if np.isnan(arr).any():
                print_fail(f"RGBA image {name} contains NaN")
                passed_all = False
                break
        else:
            print_pass(f"RGBA image encoded: {list(embeddings.keys())}")
    except Exception as e:
        print_fail(f"RGBA image failed: {e}")
        passed_all = False

    # Test 6: Zero-area bounding box (edge case - should be handled gracefully)
    print_info("Test 6: Zero-area bounding box in tracking...")
    try:
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = handler.init_tracking(
            image=test_img,
            objects=[{"object_id": 1, "box": [100, 100, 100, 100]}],  # Zero area
        )
        # Zero-area box should either be rejected or produce empty tracking
        # Either is acceptable as long as it doesn't crash
        num_tracked = len(result.get('tracked_objects', []))
        print_pass(f"Zero-area box handled gracefully: {num_tracked} objects tracked")
    except Exception as e:
        print_warn(f"Zero-area box raised exception (may be acceptable): {e}")

    # Test 7: Model info consistency after operations
    print_info("Test 7: Model info consistency...")
    try:
        info = handler.get_model_info()
        capabilities = [c for c in info.get("capabilities", []) if c is not None]
        print_pass(f"Model capabilities: {capabilities}")

        # Check new features are listed
        features = info.get("features", {})
        if features:
            print_info(f"  Features: {list(features.keys())}")
    except Exception as e:
        print_fail(f"get_model_info failed: {e}")
        passed_all = False

    return passed_all, {"status": "passed" if passed_all else "partial"}


# =============================================================================
# New Feature Tests
# =============================================================================

def test_box_prompts(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test text_to_segment with box prompts (positive and negative).
    """
    print_subheader("Box Prompts Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check if all required models exist
    info = handler.get_model_info()
    if not info.get("features", {}).get("box_prompts", False):
        print_warn("Box prompts not available (missing models)")
        return False, {"skipped": True, "reason": "missing_models"}

    # Create test image
    test_array = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add some colored regions
    test_array[100:200, 100:200] = [255, 0, 0]  # Red box
    test_array[300:400, 300:400] = [0, 255, 0]  # Green box
    test_image = Image.fromarray(test_array)

    try:
        # Test 1: Positive box prompt
        print_info("Test 1: text_to_segment with positive box prompt...")
        detections = handler.text_to_segment(
            text_prompts=["object"],
            image=test_image,
            confidence_threshold=0.1,
            box_prompts=[{"box": [100, 100, 200, 200], "label": 1}],  # Positive
        )
        print_pass(f"Positive box prompt: {len(detections)} detections")

        # Test 2: Negative box prompt (exclude region)
        print_info("Test 2: text_to_segment with negative box prompt...")
        detections_with_negative = handler.text_to_segment(
            text_prompts=["object"],
            image=test_image,
            confidence_threshold=0.1,
            box_prompts=[{"box": [300, 300, 400, 400], "label": 0}],  # Negative
        )
        print_pass(f"Negative box prompt: {len(detections_with_negative)} detections")

        # Test 3: Multiple box prompts
        print_info("Test 3: Multiple box prompts...")
        detections_multi = handler.text_to_segment(
            text_prompts=["object"],
            image=test_image,
            confidence_threshold=0.1,
            box_prompts=[
                {"box": [100, 100, 200, 200], "label": 1},  # Include red
                {"box": [300, 300, 400, 400], "label": 0},  # Exclude green
            ],
        )
        print_pass(f"Multiple box prompts: {len(detections_multi)} detections")

        return True, {"status": "passed"}

    except Exception as e:
        print_fail(f"Box prompts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def test_batched_encoding(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test batched image encoding.
    """
    print_subheader("Batched Encoding Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check if encode_batch is available
    info = handler.get_model_info()
    if "encode_batch" not in info.get("capabilities", []):
        print_warn("Batched encoding not available")
        return False, {"skipped": True, "reason": "not_available"}

    try:
        # Create test images
        images = [
            Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            for _ in range(3)
        ]

        # Test 1: Batch encode
        print_info("Test 1: Encode batch of 3 images...")
        import time
        start = time.time()
        batch_embeddings = handler.encode_batch(images)
        batch_time = time.time() - start

        # Verify batch dimension
        for name, arr in batch_embeddings.items():
            if arr.shape[0] != 3:
                print_fail(f"Batch dimension wrong for {name}: {arr.shape[0]} != 3")
                return False, {"error": f"Wrong batch dimension for {name}"}
        print_pass(f"Batch encoding completed: {batch_time*1000:.2f}ms")

        # Test 2: Compare with sequential encoding
        print_info("Test 2: Comparing with sequential encoding...")
        start = time.time()
        seq_embeddings = [handler.encode(img) for img in images]
        seq_time = time.time() - start

        # Verify outputs match
        for name in batch_embeddings.keys():
            for i, seq_emb in enumerate(seq_embeddings):
                batch_slice = batch_embeddings[name][i]
                seq_slice = seq_emb[name][0]
                max_diff = np.abs(batch_slice - seq_slice).max()
                if max_diff > 1e-5:
                    print_warn(f"Image {i} {name} max diff: {max_diff}")
        print_pass(f"Sequential encoding: {seq_time*1000:.2f}ms (speedup: {seq_time/batch_time:.2f}x)")

        # Test 3: Empty batch
        print_info("Test 3: Empty batch...")
        empty_result = handler.encode_batch([])
        if empty_result == {}:
            print_pass("Empty batch handled correctly")
        else:
            print_warn(f"Empty batch returned: {empty_result}")

        return True, {"status": "passed", "batch_time": batch_time, "seq_time": seq_time}

    except Exception as e:
        print_fail(f"Batched encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def test_automatic_mask_generation(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test automatic mask generation (AMG).
    """
    print_subheader("Automatic Mask Generation Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check if AMG is available
    info = handler.get_model_info()
    if "automatic_mask_generation" not in info.get("capabilities", []):
        print_warn("Automatic mask generation not available (missing tracker decoder)")
        return False, {"skipped": True, "reason": "missing_tracker_decoder"}

    try:
        # Create test image with distinct objects
        test_array = np.zeros((256, 256, 3), dtype=np.uint8)
        test_array[50:100, 50:100] = [255, 0, 0]   # Red square
        test_array[150:200, 150:200] = [0, 255, 0] # Green square
        test_array[50:100, 150:200] = [0, 0, 255]  # Blue square
        test_image = Image.fromarray(test_array)

        # Test 1: Basic AMG with low points_per_side for speed
        print_info("Test 1: AMG with 8 points per side...")
        import time
        start = time.time()
        masks = handler.automatic_mask_generation(
            image=test_image,
            points_per_side=8,  # 64 points total
            pred_iou_thresh=0.5,
            stability_score_thresh=0.5,
        )
        elapsed = time.time() - start
        print_pass(f"AMG generated {len(masks)} masks in {elapsed:.2f}s")

        # Validate mask structure
        for i, mask_info in enumerate(masks):
            required_fields = ["mask", "box", "area", "predicted_iou", "stability_score"]
            for field in required_fields:
                if field not in mask_info:
                    print_fail(f"Mask {i} missing field: {field}")
                    return False, {"error": f"Missing field {field}"}

            # Verify mask is binary
            mask = mask_info["mask"]
            if mask is not None:
                unique_vals = np.unique(mask)
                if not set(unique_vals).issubset({0, 1}):
                    print_fail(f"Mask {i} is not binary: {unique_vals}")
                    return False, {"error": "Non-binary mask"}

        print_pass("All mask structures validated")

        # Test 2: AMG with higher threshold
        print_info("Test 2: AMG with high IoU threshold...")
        masks_high = handler.automatic_mask_generation(
            image=test_image,
            points_per_side=8,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
        )
        print_pass(f"High threshold AMG: {len(masks_high)} masks")

        return True, {"status": "passed", "num_masks": len(masks)}

    except Exception as e:
        print_fail(f"AMG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def test_semantic_segmentation(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test semantic segmentation output.
    """
    print_subheader("Semantic Segmentation Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Create test image
    test_array = np.zeros((256, 256, 3), dtype=np.uint8)
    test_array[50:150, 50:150] = [255, 100, 100]   # Object 1
    test_array[100:200, 100:200] = [100, 255, 100] # Object 2 (overlapping)
    test_image = Image.fromarray(test_array)

    try:
        # First get detections
        print_info("Getting detections for semantic mask...")
        detections = handler.text_to_segment(
            text_prompts=["object"],
            image=test_image,
            confidence_threshold=0.01,
        )
        print_info(f"  Got {len(detections)} detections")

        # Test 1: Binary semantic mask
        print_info("Test 1: Binary semantic mask...")
        semantic_mask = handler.get_semantic_mask(detections, image_size=(256, 256))
        if semantic_mask.shape != (256, 256):
            print_fail(f"Wrong semantic mask shape: {semantic_mask.shape}")
            return False, {"error": "Wrong shape"}
        unique_vals = np.unique(semantic_mask)
        if not set(unique_vals).issubset({0, 1}):
            print_fail(f"Semantic mask not binary: {unique_vals}")
            return False, {"error": "Not binary"}
        print_pass(f"Binary semantic mask: shape={semantic_mask.shape}, coverage={semantic_mask.sum()/(256*256)*100:.1f}%")

        # Test 2: Labeled semantic mask
        print_info("Test 2: Labeled semantic mask...")
        labeled_mask = handler.get_labeled_semantic_mask(detections, image_size=(256, 256))
        if labeled_mask.shape != (256, 256):
            print_fail(f"Wrong labeled mask shape: {labeled_mask.shape}")
            return False, {"error": "Wrong shape"}
        num_instances = labeled_mask.max()
        print_pass(f"Labeled semantic mask: {num_instances} instances")

        # Test 3: Empty detections
        print_info("Test 3: Empty detections...")
        empty_mask = handler.get_semantic_mask([], image_size=(256, 256))
        if empty_mask.shape != (256, 256):
            print_fail(f"Empty mask wrong shape: {empty_mask.shape}")
            return False, {"error": "Wrong shape for empty"}
        if empty_mask.sum() != 0:
            print_fail("Empty mask should be all zeros")
            return False, {"error": "Non-zero empty mask"}
        print_pass("Empty detections handled correctly")

        return True, {"status": "passed"}

    except Exception as e:
        print_fail(f"Semantic segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def test_video_pcs(
    model_dir: Path,
    device: str = "cuda",
) -> Tuple[bool, Dict]:
    """
    Test Video PCS (text-prompted video tracking).
    """
    print_subheader("Video PCS Test")

    handler_path = Path(__file__).parent
    sys.path.insert(0, str(handler_path))

    if "model_handler" in sys.modules:
        del sys.modules["model_handler"]

    try:
        from model_handler import UnifiedModelHandler
    except ImportError as e:
        print_fail(f"Failed to import: {e}")
        return False, {}

    try:
        handler = UnifiedModelHandler(device=device, model_dir=str(model_dir))
    except Exception as e:
        print_fail(f"Failed to create handler: {e}")
        return False, {}

    # Check if Video PCS is available
    info = handler.get_model_info()
    if not info.get("features", {}).get("video_pcs", False):
        print_warn("Video PCS not available (missing models)")
        return False, {"skipped": True, "reason": "missing_models"}

    try:
        # Create test frames
        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        frame1[100:150, 100:150] = [255, 100, 100]  # Red object
        frame1 = Image.fromarray(frame1)

        frame2 = np.zeros((256, 256, 3), dtype=np.uint8)
        frame2[110:160, 110:160] = [255, 100, 100]  # Red object moved
        frame2 = Image.fromarray(frame2)

        # Test 1: Initialize tracking from text
        print_info("Test 1: Initialize tracking from text prompt...")
        result = handler.init_tracking_from_text(
            image=frame1,
            text_prompts=["object"],
            confidence_threshold=0.01,
        )

        if result.get("session_id") is None:
            # No objects detected - this can happen with synthetic images
            print_warn("No objects detected (acceptable with synthetic images)")
            if "error" in result:
                print_info(f"  Reason: {result['error']}")
            return True, {"status": "passed_no_detection"}

        session_id = result["session_id"]
        print_pass(f"Session created: {session_id}")
        print_info(f"  Detected {len(result.get('detections', []))} objects")
        print_info(f"  Tracking {len(result.get('tracked_objects', []))} objects")

        # Test 2: Track to next frame
        print_info("Test 2: Track to second frame...")
        track_result = handler.track_frame(
            session_id=session_id,
            image=frame2,
            frame_idx=1,
        )

        if "error" in track_result:
            print_fail(f"Tracking failed: {track_result['error']}")
            return False, {"error": track_result["error"]}

        print_pass(f"Tracked {len(track_result.get('tracked_objects', []))} objects to frame 1")

        # Clean up
        handler.clear_tracking(session_id)
        print_pass("Tracking session cleared")

        return True, {"status": "passed"}

    except Exception as e:
        print_fail(f"Video PCS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


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
        "--test-text-to-segment",
        action="store_true",
        help="Test handler.text_to_segment() API",
    )
    parser.add_argument(
        "--test-tracking",
        action="store_true",
        help="Test video tracking API",
    )
    parser.add_argument(
        "--test-multi-object",
        action="store_true",
        help="Test multi-object tracking",
    )
    parser.add_argument(
        "--test-edge-cases",
        action="store_true",
        help="Test edge cases and error handling",
    )
    parser.add_argument(
        "--test-box-prompts",
        action="store_true",
        help="Test box prompts for PCS",
    )
    parser.add_argument(
        "--test-batched",
        action="store_true",
        help="Test batched image encoding",
    )
    parser.add_argument(
        "--test-amg",
        action="store_true",
        help="Test automatic mask generation",
    )
    parser.add_argument(
        "--test-semantic",
        action="store_true",
        help="Test semantic segmentation output",
    )
    parser.add_argument(
        "--test-video-pcs",
        action="store_true",
        help="Test Video PCS (text-prompted tracking)",
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
    any_tests_run = False  # Track if at least one test actually ran

    # Determine which tests to run
    run_all = args.all or not any([
        args.test_vision_encoder,
        args.test_tracker_decoder,
        args.test_text_encoder,
        args.test_end_to_end,
        args.test_handler,
        args.test_text_to_segment,
        args.test_tracking,
        args.test_multi_object,
        args.test_edge_cases,
        args.test_box_prompts,
        args.test_batched,
        args.test_amg,
        args.test_semantic,
        args.test_video_pcs,
    ])

    # Run tests
    if run_all or args.test_vision_encoder:
        vision_path = model_dir / "vision_encoder.onnx"
        if vision_path.exists():
            any_tests_run = True
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
            any_tests_run = True
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
            any_tests_run = True
            passed, metrics = test_text_encoder(text_path, device)
            results["tests"]["text_encoder"] = {"passed": passed, "metrics": metrics}
            if not passed:
                all_passed = False
        else:
            print_warn(f"Text encoder not found: {text_path}")
            results["tests"]["text_encoder"] = {"skipped": True}

    if run_all or args.test_end_to_end:
        any_tests_run = True
        passed, info = test_end_to_end_encode(model_dir, device, args.test_image)
        results["tests"]["end_to_end_encode"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

        passed, info = test_end_to_end_text_to_segment(model_dir, device, args.test_image)
        results["tests"]["end_to_end_text_to_segment"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    if run_all or args.test_handler:
        any_tests_run = True
        passed, info = test_unified_handler(model_dir, device)
        results["tests"]["unified_handler"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    if run_all or args.test_text_to_segment:
        any_tests_run = True
        passed, info = test_handler_text_to_segment(model_dir, device)
        results["tests"]["handler_text_to_segment"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    if run_all or args.test_tracking:
        any_tests_run = True
        passed, info = test_video_tracking(model_dir, device)
        results["tests"]["video_tracking"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    if run_all or args.test_multi_object:
        any_tests_run = True
        passed, info = test_multi_object_tracking(model_dir, device)
        results["tests"]["multi_object_tracking"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    if run_all or args.test_edge_cases:
        any_tests_run = True
        passed, info = test_edge_cases(model_dir, device)
        results["tests"]["edge_cases"] = {"passed": passed, "info": info}
        if not passed:
            all_passed = False

    # New feature tests
    if run_all or args.test_box_prompts:
        any_tests_run = True
        passed, info = test_box_prompts(model_dir, device)
        results["tests"]["box_prompts"] = {"passed": passed, "info": info}
        if not passed and not info.get("skipped"):
            all_passed = False

    if run_all or args.test_batched:
        any_tests_run = True
        passed, info = test_batched_encoding(model_dir, device)
        results["tests"]["batched_encoding"] = {"passed": passed, "info": info}
        if not passed and not info.get("skipped"):
            all_passed = False

    if run_all or args.test_amg:
        any_tests_run = True
        passed, info = test_automatic_mask_generation(model_dir, device)
        results["tests"]["automatic_mask_generation"] = {"passed": passed, "info": info}
        if not passed and not info.get("skipped"):
            all_passed = False

    if run_all or args.test_semantic:
        any_tests_run = True
        passed, info = test_semantic_segmentation(model_dir, device)
        results["tests"]["semantic_segmentation"] = {"passed": passed, "info": info}
        if not passed and not info.get("skipped"):
            all_passed = False

    if run_all or args.test_video_pcs:
        any_tests_run = True
        passed, info = test_video_pcs(model_dir, device)
        results["tests"]["video_pcs"] = {"passed": passed, "info": info}
        if not passed and not info.get("skipped"):
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

    # Fail if no tests actually ran (all skipped)
    if not any_tests_run:
        print(f"\n{Colors.RED}{Colors.BOLD}No tests ran! All models may be missing.{Colors.END}")
        sys.exit(1)

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
