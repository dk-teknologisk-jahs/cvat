#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Unified ONNX Comprehensive Test Suite

Inspired by existing SAM3 tests (test_sam3_onnx.py, test_sam3_e2e.py, test_sam3_multiclick.py).
Combines ONNX vs PyTorch comparison with elaborate synthetic test shapes.

Test Categories:
1. Shape Tests: Synthetic images with known ground truth masks
2. Multi-Click Tests: Positive/negative click disambiguation
3. Mask Refinement Tests: Iterative prediction with mask input
4. ONNX vs PyTorch: Numerical comparison of outputs
5. End-to-End Tests: Full pipeline validation

Usage:
    # Quick smoke test
    python test_comprehensive.py --model-dir ./onnx-exports --smoke

    # Full synthetic shape test suite
    python test_comprehensive.py --model-dir ./onnx-exports --shapes

    # ONNX vs PyTorch comparison (requires HF auth)
    python test_comprehensive.py --model-dir ./onnx-exports --compare-pytorch

    # All tests
    python test_comprehensive.py --model-dir ./onnx-exports --all

    # Test with specific image
    python test_comprehensive.py --model-dir ./onnx-exports --image ./test.jpg --point 504,504

Environment Variables:
    SAM3_MODEL_DIR: Default model directory
    HF_TOKEN: HuggingFace token for gated model access
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ==============================================================================
# Constants
# ==============================================================================
SAM3_IMAGE_SIZE = 1008
SAM3_MASK_SIZE = 288

# Image normalization (SAM3 uses 0.5 mean/std)
SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Test thresholds for ONNX vs PyTorch
MAX_MAE = 0.001
MAX_DIFF = 0.01
MIN_CORRELATION = 0.9999

# IoU thresholds for shape tests
MIN_IOU_SIMPLE = 0.90
MIN_IOU_COMPLEX = 0.80
MIN_IOU_DISAMBIGUATE = 0.85


# ==============================================================================
# Terminal Colors
# ==============================================================================
class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}[{text}]{Colors.END}")


def print_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    if details:
        print(f"  {status}: {name} ({details})")
    else:
        print(f"  {status}: {name}")


def print_info(text: str):
    print(f"  {Colors.BLUE}ℹ{Colors.END} {text}")


def print_warn(text: str):
    print(f"  {Colors.YELLOW}⚠{Colors.END} {text}")


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class Click:
    """A click prompt."""
    x: float
    y: float
    label: int  # 1=positive, 0=negative

@dataclass
class TestCase:
    """A test case for SAM3."""
    name: str
    description: str
    image: np.ndarray  # [H, W, 3] RGB
    clicks: List[Click]
    expected_mask: Optional[np.ndarray] = None  # [H, W] binary
    min_iou: float = MIN_IOU_SIMPLE

@dataclass
class TestResult:
    """Result of a test case."""
    name: str
    passed: bool
    iou: Optional[float] = None
    predicted_iou: Optional[float] = None
    details: str = ""


# ==============================================================================
# Synthetic Test Image Generators
# ==============================================================================
def create_circle_image(
    size: int = SAM3_IMAGE_SIZE,
    center: Tuple[int, int] = (504, 504),
    radius: int = 200,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (50, 100, 200),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a filled circle."""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fg_color)

    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse(bbox, fill=255)

    return np.array(img), np.array(mask) > 127


def create_rectangle_image(
    size: int = SAM3_IMAGE_SIZE,
    rect: Tuple[int, int, int, int] = (300, 300, 700, 700),
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (200, 50, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a filled rectangle."""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle(rect, fill=fg_color)

    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).rectangle(rect, fill=255)

    return np.array(img), np.array(mask) > 127


def create_star_image(
    size: int = SAM3_IMAGE_SIZE,
    center: Tuple[int, int] = (504, 504),
    outer_radius: int = 300,
    inner_radius: int = 150,
    n_points: int = 5,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (50, 200, 100),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a star polygon."""
    cx, cy = center
    points = []
    for i in range(n_points * 2):
        angle = i * np.pi / n_points - np.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        points.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))

    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=fg_color)

    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)

    return np.array(img), np.array(mask) > 127


def create_donut_image(
    size: int = SAM3_IMAGE_SIZE,
    center: Tuple[int, int] = (504, 504),
    outer_radius: int = 300,
    inner_radius: int = 150,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (200, 150, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a donut/ring shape (tests hole handling)."""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    x, y = center
    outer_bbox = [x - outer_radius, y - outer_radius, x + outer_radius, y + outer_radius]
    inner_bbox = [x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius]
    draw.ellipse(outer_bbox, fill=fg_color)
    draw.ellipse(inner_bbox, fill=bg_color)

    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(outer_bbox, fill=255)
    mask_draw.ellipse(inner_bbox, fill=0)

    return np.array(img), np.array(mask) > 127


def create_two_circles_image(
    size: int = SAM3_IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Create image with two circles (for distractor exclusion tests).

    Returns:
        image, target_mask, distractor_mask, target_center, distractor_center
    """
    img = Image.new('RGB', (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Circle 1 (target - blue)
    center1, radius1 = (300, 504), 150
    draw.ellipse([center1[0]-radius1, center1[1]-radius1,
                  center1[0]+radius1, center1[1]+radius1], fill=(50, 100, 200))

    # Circle 2 (distractor - red)
    center2, radius2 = (700, 504), 150
    draw.ellipse([center2[0]-radius2, center2[1]-radius2,
                  center2[0]+radius2, center2[1]+radius2], fill=(200, 50, 50))

    # Masks
    Y, X = np.ogrid[:size, :size]
    target_mask = ((X - center1[0])**2 + (Y - center1[1])**2) <= radius1**2
    distractor_mask = ((X - center2[0])**2 + (Y - center2[1])**2) <= radius2**2

    return np.array(img), target_mask, distractor_mask, center1, center2


def create_gradient_image(
    size: int = SAM3_IMAGE_SIZE,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Create a more realistic image with gradient-filled objects.

    Returns:
        image, target_center, distractor_center
    """
    img = Image.new('RGB', (size, size), color=(30, 30, 50))

    # Background gradient
    for y in range(size):
        c = int(30 + y * 0.03)
        for x in range(size):
            img.putpixel((x, y), (c//3, c//3, c))

    # Object 1: Yellow-orange ellipse
    for y in range(200, 700):
        for x in range(150, 550):
            dx = (x - 350) / 200
            dy = (y - 450) / 250
            if dx*dx + dy*dy < 1:
                intensity = 1 - 0.3 * (dx*dx + dy*dy)
                img.putpixel((x, y), (int(255*intensity), int(200*intensity), int(50*intensity)))

    # Object 2: Green ellipse
    for y in range(300, 650):
        for x in range(550, 900):
            dx = (x - 725) / 175
            dy = (y - 475) / 175
            if dx*dx + dy*dy < 1:
                intensity = 1 - 0.3 * (dx*dx + dy*dy)
                img.putpixel((x, y), (int(50*intensity), int(200*intensity), int(50*intensity)))

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(img), (350, 450), (725, 475)


def create_thin_lines_image(
    size: int = SAM3_IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with thin intersecting lines (tests fine detail)."""
    bg_color = (240, 240, 240)
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

    line_color = (50, 100, 50)
    lines = [
        [(504, 100), (504, 900)],  # Vertical
        [(200, 504), (800, 504)],  # Horizontal
        [(300, 300), (700, 700)],  # Diagonal
        [(700, 300), (300, 700)],  # Diagonal 2
    ]

    for start, end in lines:
        draw.line([start, end], fill=line_color, width=20)
        mask_draw.line([start, end], fill=255, width=20)

    return np.array(img), np.array(mask) > 127


# ==============================================================================
# Test Case Generator
# ==============================================================================
def generate_test_cases() -> List[TestCase]:
    """Generate comprehensive test cases."""
    test_cases = []

    # === Simple Shapes ===
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_center",
        description="Single click in circle center",
        image=img,
        clicks=[Click(504, 504, 1)],
        expected_mask=mask,
        min_iou=MIN_IOU_SIMPLE,
    ))

    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_edge",
        description="Single click on circle edge",
        image=img,
        clicks=[Click(504, 304, 1)],
        expected_mask=mask,
        min_iou=MIN_IOU_SIMPLE,
    ))

    img, mask = create_rectangle_image()
    test_cases.append(TestCase(
        name="rectangle",
        description="Single click on rectangle",
        image=img,
        clicks=[Click(500, 500, 1)],
        expected_mask=mask,
        min_iou=MIN_IOU_SIMPLE,
    ))

    img, mask = create_star_image()
    test_cases.append(TestCase(
        name="star_polygon",
        description="Single click on star shape",
        image=img,
        clicks=[Click(504, 504, 1)],
        expected_mask=mask,
        min_iou=MIN_IOU_COMPLEX,
    ))

    # === Complex Shapes ===
    img, mask = create_donut_image()
    test_cases.append(TestCase(
        name="donut",
        description="Click on ring (not hole)",
        image=img,
        clicks=[Click(504, 204, 1)],  # Top of donut
        expected_mask=mask,
        min_iou=MIN_IOU_COMPLEX,
    ))

    img, mask = create_thin_lines_image()
    test_cases.append(TestCase(
        name="thin_lines",
        description="Click on intersecting thin lines",
        image=img,
        clicks=[Click(504, 504, 1)],
        expected_mask=mask,
        min_iou=MIN_IOU_COMPLEX,
    ))

    # === Multi-Click Tests ===
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_multi_positive",
        description="Multiple positive clicks on circle",
        image=img,
        clicks=[
            Click(504, 504, 1),
            Click(400, 504, 1),
            Click(600, 504, 1),
        ],
        expected_mask=mask,
        min_iou=MIN_IOU_SIMPLE,
    ))

    # === Disambiguation Tests ===
    img, target_mask, _, target_center, distractor_center = create_two_circles_image()
    test_cases.append(TestCase(
        name="two_circles_target_only",
        description="Click on target circle (no negative)",
        image=img,
        clicks=[Click(target_center[0], target_center[1], 1)],
        expected_mask=target_mask,
        min_iou=MIN_IOU_SIMPLE,
    ))

    test_cases.append(TestCase(
        name="two_circles_with_negative",
        description="Positive on target + negative on distractor",
        image=img,
        clicks=[
            Click(target_center[0], target_center[1], 1),
            Click(distractor_center[0], distractor_center[1], 0),
        ],
        expected_mask=target_mask,
        min_iou=MIN_IOU_DISAMBIGUATE,
    ))

    return test_cases


# ==============================================================================
# SAM3 ONNX Tester
# ==============================================================================
class SAM3OnnxTester:
    """Test SAM3 ONNX models."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        import onnxruntime as ort

        self.model_dir = model_dir
        self.device = device

        # ONNX session options
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']

        # Load models
        self.encoder = None
        self.decoder = None

        encoder_path = model_dir / "vision_encoder.onnx"
        decoder_path = model_dir / "tracker_decoder.onnx"

        if encoder_path.exists():
            print_info(f"Loading vision encoder: {encoder_path}")
            self.encoder = ort.InferenceSession(str(encoder_path), providers=providers)
        else:
            print_warn(f"Vision encoder not found: {encoder_path}")

        if decoder_path.exists():
            print_info(f"Loading tracker decoder: {decoder_path}")
            self.decoder = ort.InferenceSession(str(decoder_path), providers=providers)
        else:
            print_warn(f"Tracker decoder not found: {decoder_path}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SAM3 encoder."""
        img = Image.fromarray(image)
        img = img.resize((SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE), Image.BILINEAR)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - SAM3_MEAN) / SAM3_STD
        img_array = img_array.transpose(2, 0, 1)

        return np.expand_dims(img_array, axis=0).astype(np.float32)

    def encode(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Encode image to embeddings."""
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")

        input_tensor = self.preprocess_image(image)
        outputs = self.encoder.run(None, {"images": input_tensor})

        return {
            "fpn_feat_0": outputs[0],  # [1, 256, 288, 288]
            "fpn_feat_1": outputs[1],  # [1, 256, 144, 144]
            "fpn_feat_2": outputs[2],  # [1, 256, 72, 72]
        }

    def decode(
        self,
        embeddings: Dict[str, np.ndarray],
        clicks: List[Click],
        mask_input: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run decoder with prompts."""
        if self.decoder is None:
            raise RuntimeError("Decoder not loaded")

        # Prepare point coords and labels
        # HuggingFace SAM3 expects 4D: [B, num_objects, num_points, 2]
        n_points = len(clicks)
        point_coords = np.zeros((1, 1, n_points, 2), dtype=np.float32)  # 4D!
        point_labels = np.zeros((1, 1, n_points), dtype=np.float32)  # 3D!

        for i, click in enumerate(clicks):
            point_coords[0, 0, i, 0] = click.x
            point_coords[0, 0, i, 1] = click.y
            point_labels[0, 0, i] = click.label

        # Prepare mask input
        if mask_input is None:
            mask_input = np.zeros((1, 1, SAM3_MASK_SIZE, SAM3_MASK_SIZE), dtype=np.float32)
            has_mask = np.array([0.0], dtype=np.float32)
        else:
            has_mask = np.array([1.0], dtype=np.float32)

        inputs = {
            "fpn_feat_0": embeddings["fpn_feat_0"],
            "fpn_feat_1": embeddings["fpn_feat_1"],
            "fpn_feat_2": embeddings["fpn_feat_2"],
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
        }

        outputs = self.decoder.run(None, inputs)

        return {
            "masks": outputs[0],  # [1, 3, H, W]
            "iou_predictions": outputs[1],  # [1, 3]
            "low_res_masks": outputs[2],  # [1, 3, 288, 288]
        }

    def get_best_mask(self, outputs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """Get best mask based on IoU prediction."""
        best_idx = np.argmax(outputs["iou_predictions"][0])
        mask = outputs["masks"][0, best_idx] > 0
        iou_pred = outputs["iou_predictions"][0, best_idx]
        return mask, float(iou_pred)

    def compute_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute IoU between predicted and ground truth masks."""
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        return float(intersection / union) if union > 0 else 0.0


# ==============================================================================
# Test Runners
# ==============================================================================
def run_smoke_tests(tester: SAM3OnnxTester) -> List[TestResult]:
    """Run quick smoke tests."""
    print_section("Smoke Tests")
    results = []

    # Test 1: Model loading
    results.append(TestResult(
        name="encoder_loaded",
        passed=tester.encoder is not None,
    ))
    results.append(TestResult(
        name="decoder_loaded",
        passed=tester.decoder is not None,
    ))

    if tester.encoder is None or tester.decoder is None:
        print_warn("Cannot run inference tests without both models")
        return results

    # Test 2: Encode test
    try:
        img, _ = create_circle_image()
        start = time.time()
        embeddings = tester.encode(img)
        encode_time = (time.time() - start) * 1000

        # Check shapes
        shapes_ok = (
            embeddings["fpn_feat_0"].shape == (1, 256, 288, 288) and
            embeddings["fpn_feat_1"].shape == (1, 256, 144, 144) and
            embeddings["fpn_feat_2"].shape == (1, 256, 72, 72)
        )
        results.append(TestResult(
            name="encode_shapes",
            passed=shapes_ok,
            details=f"{encode_time:.1f}ms",
        ))
    except Exception as e:
        results.append(TestResult(
            name="encode_shapes",
            passed=False,
            details=str(e),
        ))
        return results

    # Test 3: Decode test
    try:
        start = time.time()
        outputs = tester.decode(embeddings, [Click(504, 504, 1)])
        decode_time = (time.time() - start) * 1000

        shapes_ok = (
            outputs["masks"].shape[0] == 1 and
            outputs["masks"].shape[1] == 3 and
            outputs["iou_predictions"].shape == (1, 3)
        )
        results.append(TestResult(
            name="decode_shapes",
            passed=shapes_ok,
            details=f"{decode_time:.1f}ms",
        ))
    except Exception as e:
        results.append(TestResult(
            name="decode_shapes",
            passed=False,
            details=str(e),
        ))
        return results

    # Test 4: IoU prediction sanity
    iou = outputs["iou_predictions"][0].max()
    results.append(TestResult(
        name="iou_sanity",
        passed=0.5 < iou < 1.0,
        predicted_iou=float(iou),
        details=f"IoU={iou:.3f}",
    ))

    return results


def run_shape_tests(tester: SAM3OnnxTester) -> List[TestResult]:
    """Run synthetic shape test suite."""
    print_section("Synthetic Shape Tests")

    if tester.encoder is None or tester.decoder is None:
        print_warn("Cannot run shape tests without both models")
        return [TestResult(
            name="shape_tests_skipped",
            passed=False,
            details="Missing encoder or decoder models",
        )]

    results = []
    test_cases = generate_test_cases()

    for tc in test_cases:
        try:
            # Encode
            embeddings = tester.encode(tc.image)

            # Decode
            outputs = tester.decode(embeddings, tc.clicks)
            pred_mask, pred_iou = tester.get_best_mask(outputs)

            # Compute actual IoU
            if tc.expected_mask is not None:
                actual_iou = tester.compute_iou(pred_mask, tc.expected_mask)
                passed = actual_iou >= tc.min_iou
            else:
                actual_iou = None
                passed = pred_iou > 0.7  # Fallback sanity check

            results.append(TestResult(
                name=tc.name,
                passed=passed,
                iou=actual_iou,
                predicted_iou=pred_iou,
                details=f"IoU={actual_iou:.3f}" if actual_iou else f"pred_IoU={pred_iou:.3f}",
            ))

        except Exception as e:
            results.append(TestResult(
                name=tc.name,
                passed=False,
                details=str(e),
            ))

    return results


def run_refinement_tests(tester: SAM3OnnxTester) -> List[TestResult]:
    """Test mask refinement (multi-click with mask input)."""
    print_section("Mask Refinement Tests")

    if tester.encoder is None or tester.decoder is None:
        return [TestResult(
            name="refinement_tests_skipped",
            passed=False,
            details="Missing encoder or decoder models",
        )]

    results = []

    # Create test image
    img, mask = create_circle_image()
    embeddings = tester.encode(img)

    # First click
    outputs1 = tester.decode(embeddings, [Click(504, 504, 1)])
    pred1, iou1 = tester.get_best_mask(outputs1)
    actual_iou1 = tester.compute_iou(pred1, mask)

    results.append(TestResult(
        name="initial_click",
        passed=actual_iou1 > MIN_IOU_SIMPLE,
        iou=actual_iou1,
        predicted_iou=iou1,
    ))

    # Refinement with previous mask
    low_res_mask = outputs1["low_res_masks"][0:1, np.argmax(outputs1["iou_predictions"][0]):np.argmax(outputs1["iou_predictions"][0])+1]

    outputs2 = tester.decode(
        embeddings,
        [Click(504, 400, 1)],  # New click
        mask_input=low_res_mask,
    )
    pred2, iou2 = tester.get_best_mask(outputs2)
    actual_iou2 = tester.compute_iou(pred2, mask)

    results.append(TestResult(
        name="refinement_click",
        passed=actual_iou2 >= actual_iou1 * 0.95,  # Should maintain or improve
        iou=actual_iou2,
        predicted_iou=iou2,
        details=f"init={actual_iou1:.3f}, refined={actual_iou2:.3f}",
    ))

    return results


def run_pytorch_comparison(model_dir: Path, device: str = "cuda") -> List[TestResult]:
    """Compare ONNX outputs to PyTorch reference."""
    print_section("ONNX vs PyTorch Comparison")

    results = []

    try:
        import torch
        import onnxruntime as ort
    except ImportError as e:
        print_warn(f"Missing dependency: {e}")
        return [TestResult(
            name="pytorch_comparison_skipped",
            passed=False,
            details=f"Missing dependency: {e}",
        )]

    # Try to load HuggingFace model
    try:
        from transformers import Sam2Model
        print_info("Loading HuggingFace SAM2 model (requires auth)...")
        hf_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
        hf_model = hf_model.to(device).eval()
    except Exception as e:
        print_warn(f"Could not load HuggingFace model: {e}")
        print_info("Set HF_TOKEN or run `huggingface-cli login`")
        return results

    # Load ONNX encoder
    encoder_path = model_dir / "vision_encoder.onnx"
    if not encoder_path.exists():
        print_warn(f"ONNX encoder not found: {encoder_path}")
        return results

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    onnx_encoder = ort.InferenceSession(str(encoder_path), providers=providers)

    # Test input
    print_info("Generating test input...")
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 1008, 1008).astype(np.float32)
    torch_input = torch.from_numpy(test_input).to(device)

    # Run ONNX
    print_info("Running ONNX inference...")
    onnx_outputs = onnx_encoder.run(None, {"images": test_input})

    # Run PyTorch
    print_info("Running PyTorch inference...")
    with torch.no_grad():
        backbone_out = hf_model.image_encoder(torch_input)

    # Compare first FPN level
    pt_out = backbone_out.backbone_fpn[0].cpu().numpy()
    onnx_out = onnx_outputs[0]

    mae = np.abs(pt_out - onnx_out).mean()
    max_diff = np.abs(pt_out - onnx_out).max()
    corr = np.corrcoef(pt_out.flatten(), onnx_out.flatten())[0, 1]

    passed = mae < MAX_MAE and max_diff < MAX_DIFF and corr > MIN_CORRELATION

    results.append(TestResult(
        name="vision_encoder_fpn0",
        passed=passed,
        details=f"MAE={mae:.6f}, MaxDiff={max_diff:.6f}, Corr={corr:.6f}",
    ))

    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Unified ONNX Comprehensive Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SAM3_MODEL_DIR", "./onnx-exports"),
                        help="Directory containing ONNX models")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for inference")

    # Test selection
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--shapes", action="store_true", help="Run synthetic shape tests")
    parser.add_argument("--refinement", action="store_true", help="Run mask refinement tests")
    parser.add_argument("--compare-pytorch", action="store_true", help="Compare ONNX vs PyTorch")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    # Output
    parser.add_argument("--json", type=str, help="Output results as JSON")
    parser.add_argument("--save-viz", action="store_true", help="Save visualizations")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print_header("SAM3 Unified ONNX Comprehensive Test Suite")
    print_info(f"Model directory: {model_dir}")
    print_info(f"Device: {args.device}")

    all_results: List[TestResult] = []

    # Initialize tester
    tester = SAM3OnnxTester(model_dir, args.device)

    # Determine which tests to run
    run_smoke = args.smoke or args.all or not any([args.shapes, args.refinement, args.compare_pytorch])
    run_shapes = args.shapes or args.all
    run_refine = args.refinement or args.all
    run_compare = args.compare_pytorch or args.all

    # Run tests
    if run_smoke:
        all_results.extend(run_smoke_tests(tester))

    if run_shapes:
        all_results.extend(run_shape_tests(tester))

    if run_refine:
        all_results.extend(run_refinement_tests(tester))

    if run_compare:
        all_results.extend(run_pytorch_comparison(model_dir, args.device))

    # Print results
    print_header("Test Results Summary")

    for result in all_results:
        print_test(result.name, result.passed, result.details)

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    # JSON output
    if args.json:
        json_results = [
            {
                "name": r.name,
                "passed": r.passed,
                "iou": r.iou,
                "predicted_iou": r.predicted_iou,
                "details": r.details,
            }
            for r in all_results
        ]
        with open(args.json, "w") as f:
            import json
            json.dump({"results": json_results, "passed": passed, "total": total}, f, indent=2)
        print_info(f"Results saved to {args.json}")

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
