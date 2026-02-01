#!/usr/bin/env python3
"""
SAM3 ONNX Model Test Suite

Comprehensive tests for SAM3 ONNX models with elaborate test shapes:
1. Single click on various objects (circles, rectangles, complex shapes)
2. Multi-click refinement (add positive/negative clicks)
3. Box prompts
4. Mask refinement (iterative prediction)
5. Comparison with PyTorch model outputs

Test Images:
- Synthetic: geometric shapes (circles, rectangles, polygons)
- Complex: overlapping shapes, thin structures, holes

Usage:
    # Test with synthetic shapes
    python test_sam3_onnx.py --synthetic

    # Test with real image
    python test_sam3_onnx.py --image /path/to/image.jpg

    # Compare ONNX vs PyTorch
    python test_sam3_onnx.py --compare-pytorch --image /path/to/image.jpg

    # Full test suite
    python test_sam3_onnx.py --full-suite
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Test Configuration
# ============================================================================
SAM3_IMAGE_SIZE = 1008
SAM3_EMBED_SIZE = 72
SAM3_MASK_SIZE = 288

# Default paths
DEFAULT_ENCODER_PATH = "/home/jahs/GitHub/cvat/serverless/pytorch/facebookresearch/sam3/nuclio/sam3_vision_encoder.onnx"
DEFAULT_DECODER_PATH = "/home/jahs/GitHub/cvat/cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx"

# Image normalization (SAM3 uses 0.5 mean/std)
SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


@dataclass
class Click:
    """A click prompt."""
    x: float  # pixel x coordinate
    y: float  # pixel y coordinate
    label: int  # 1=positive, 0=negative

@dataclass
class Box:
    """A box prompt."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TestCase:
    """A test case for SAM3."""
    name: str
    description: str
    image: np.ndarray  # [H, W, 3] RGB
    clicks: List[Click]
    boxes: List[Box]
    expected_mask: Optional[np.ndarray] = None  # [H, W] binary
    mask_input: Optional[np.ndarray] = None  # [1, 288, 288] for refinement


# ============================================================================
# Synthetic Test Image Generation
# ============================================================================
def create_circle_image(
    size: Tuple[int, int] = (1008, 1008),
    center: Tuple[int, int] = (504, 504),
    radius: int = 200,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (50, 100, 200),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create image with a filled circle.

    Returns:
        image: [H, W, 3] RGB
        mask: [H, W] binary mask
    """
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fg_color)

    # Create mask
    mask = Image.new('L', size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(bbox, fill=255)

    return np.array(img), np.array(mask) > 127


def create_rectangle_image(
    size: Tuple[int, int] = (1008, 1008),
    rect: Tuple[int, int, int, int] = (300, 300, 700, 700),
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (200, 50, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a filled rectangle."""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle(rect, fill=fg_color)

    mask = Image.new('L', size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(rect, fill=255)

    return np.array(img), np.array(mask) > 127


def create_polygon_image(
    size: Tuple[int, int] = (1008, 1008),
    points: Optional[List[Tuple[int, int]]] = None,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (50, 200, 100),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a filled polygon (star shape by default)."""
    if points is None:
        # Create a star shape
        cx, cy = size[0] // 2, size[1] // 2
        outer_r, inner_r = 300, 150
        n_points = 5
        points = []
        for i in range(n_points * 2):
            angle = i * np.pi / n_points - np.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            points.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))

    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=fg_color)

    mask = Image.new('L', size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(points, fill=255)

    return np.array(img), np.array(mask) > 127


def create_donut_image(
    size: Tuple[int, int] = (1008, 1008),
    center: Tuple[int, int] = (504, 504),
    outer_radius: int = 300,
    inner_radius: int = 150,
    bg_color: Tuple[int, int, int] = (240, 240, 240),
    fg_color: Tuple[int, int, int] = (200, 150, 50),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with a donut/ring shape (tests hole handling)."""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    x, y = center
    # Draw outer circle
    outer_bbox = [x - outer_radius, y - outer_radius, x + outer_radius, y + outer_radius]
    draw.ellipse(outer_bbox, fill=fg_color)
    # Draw inner circle (hole) with background color
    inner_bbox = [x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius]
    draw.ellipse(inner_bbox, fill=bg_color)

    # Create mask
    mask = Image.new('L', size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(outer_bbox, fill=255)
    mask_draw.ellipse(inner_bbox, fill=0)

    return np.array(img), np.array(mask) > 127


def create_overlapping_shapes_image(
    size: Tuple[int, int] = (1008, 1008),
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Create image with overlapping shapes for testing ambiguous clicks.

    Returns:
        image: [H, W, 3] RGB
        masks: List of [H, W] binary masks for each shape
    """
    bg_color = (240, 240, 240)
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    masks = []

    # Shape 1: Large circle
    circle1_center = (400, 504)
    circle1_radius = 250
    draw.ellipse([
        circle1_center[0] - circle1_radius,
        circle1_center[1] - circle1_radius,
        circle1_center[0] + circle1_radius,
        circle1_center[1] + circle1_radius
    ], fill=(100, 150, 200))

    mask1 = Image.new('L', size, 0)
    ImageDraw.Draw(mask1).ellipse([
        circle1_center[0] - circle1_radius,
        circle1_center[1] - circle1_radius,
        circle1_center[0] + circle1_radius,
        circle1_center[1] + circle1_radius
    ], fill=255)
    masks.append(np.array(mask1) > 127)

    # Shape 2: Overlapping circle
    circle2_center = (600, 504)
    circle2_radius = 250
    draw.ellipse([
        circle2_center[0] - circle2_radius,
        circle2_center[1] - circle2_radius,
        circle2_center[0] + circle2_radius,
        circle2_center[1] + circle2_radius
    ], fill=(200, 100, 100))

    mask2 = Image.new('L', size, 0)
    ImageDraw.Draw(mask2).ellipse([
        circle2_center[0] - circle2_radius,
        circle2_center[1] - circle2_radius,
        circle2_center[0] + circle2_radius,
        circle2_center[1] + circle2_radius
    ], fill=255)
    masks.append(np.array(mask2) > 127)

    return np.array(img), masks


def create_thin_structure_image(
    size: Tuple[int, int] = (1008, 1008),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create image with thin structures (tests fine details)."""
    bg_color = (240, 240, 240)
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    mask = Image.new('L', size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # Draw thin lines/branches
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


def create_multiple_instances_image(
    size: Tuple[int, int] = (1008, 1008),
    n_instances: int = 5,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Create image with multiple separate instances."""
    bg_color = (240, 240, 240)
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    masks = []
    colors = [
        (200, 100, 100),
        (100, 200, 100),
        (100, 100, 200),
        (200, 200, 100),
        (200, 100, 200),
    ]

    # Create grid of circles
    grid_size = int(np.ceil(np.sqrt(n_instances)))
    cell_w = size[0] // grid_size
    cell_h = size[1] // grid_size
    radius = min(cell_w, cell_h) // 3

    for i in range(n_instances):
        row = i // grid_size
        col = i % grid_size
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2

        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=colors[i % len(colors)])

        mask = Image.new('L', size, 0)
        ImageDraw.Draw(mask).ellipse(bbox, fill=255)
        masks.append(np.array(mask) > 127)

    return np.array(img), masks


# ============================================================================
# Test Case Generators
# ============================================================================
def generate_test_cases() -> List[TestCase]:
    """Generate comprehensive test cases."""
    test_cases = []

    # === Single Click Tests ===

    # 1. Simple circle - click in center
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_center_click",
        description="Single positive click on circle center",
        image=img,
        clicks=[Click(504, 504, 1)],
        boxes=[],
        expected_mask=mask,
    ))

    # 2. Circle - click on edge
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_edge_click",
        description="Single positive click on circle edge",
        image=img,
        clicks=[Click(504, 304, 1)],  # Top edge of circle
        boxes=[],
        expected_mask=mask,
    ))

    # 3. Rectangle
    img, mask = create_rectangle_image()
    test_cases.append(TestCase(
        name="rectangle_click",
        description="Single positive click on rectangle",
        image=img,
        clicks=[Click(500, 500, 1)],
        boxes=[],
        expected_mask=mask,
    ))

    # 4. Star polygon
    img, mask = create_polygon_image()
    test_cases.append(TestCase(
        name="star_click",
        description="Single positive click on star shape",
        image=img,
        clicks=[Click(504, 504, 1)],
        boxes=[],
        expected_mask=mask,
    ))

    # 5. Donut (tests hole handling)
    img, mask = create_donut_image()
    test_cases.append(TestCase(
        name="donut_click",
        description="Single positive click on donut (ring with hole)",
        image=img,
        clicks=[Click(504, 204, 1)],  # Click on the ring, not the hole
        boxes=[],
        expected_mask=mask,
    ))

    # === Multi-Click Tests ===

    # 6. Circle with multiple positive clicks
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
        boxes=[],
        expected_mask=mask,
    ))

    # 7. Circle with positive and negative clicks
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_pos_neg_clicks",
        description="Positive click on circle, negative click on background",
        image=img,
        clicks=[
            Click(504, 504, 1),  # On circle
            Click(100, 100, 0),  # On background
        ],
        boxes=[],
        expected_mask=mask,
    ))

    # 8. Overlapping shapes - disambiguation
    img, masks = create_overlapping_shapes_image()
    # Click in overlap region - should select based on stability
    test_cases.append(TestCase(
        name="overlapping_click",
        description="Click in overlapping region (ambiguous)",
        image=img,
        clicks=[Click(504, 504, 1)],
        boxes=[],
        expected_mask=None,  # Ambiguous - either shape is valid
    ))

    # 9. Overlapping shapes - negative click to disambiguate
    img, masks = create_overlapping_shapes_image()
    test_cases.append(TestCase(
        name="overlapping_disambiguate",
        description="Click in overlap + negative click to select left circle",
        image=img,
        clicks=[
            Click(400, 504, 1),  # On left circle
            Click(700, 504, 0),  # Negative on right circle
        ],
        boxes=[],
        expected_mask=masks[0],  # Left circle
    ))

    # 10. Thin structure
    img, mask = create_thin_structure_image()
    test_cases.append(TestCase(
        name="thin_structure",
        description="Multiple clicks on thin intersecting lines",
        image=img,
        clicks=[
            Click(504, 504, 1),  # Center intersection
            Click(504, 200, 1),  # Vertical line
            Click(700, 504, 1),  # Horizontal line
        ],
        boxes=[],
        expected_mask=mask,
    ))

    # === Box Prompt Tests ===

    # 11. Box around circle
    img, mask = create_circle_image()
    test_cases.append(TestCase(
        name="circle_box",
        description="Box prompt around circle",
        image=img,
        clicks=[],
        boxes=[Box(304, 304, 704, 704)],
        expected_mask=mask,
    ))

    # 12. Box + click
    img, mask = create_rectangle_image()
    test_cases.append(TestCase(
        name="rect_box_click",
        description="Box prompt + positive click on rectangle",
        image=img,
        clicks=[Click(500, 500, 1)],
        boxes=[Box(290, 290, 710, 710)],
        expected_mask=mask,
    ))

    # === Multiple Instances Tests ===

    # 13. Multiple instances - select one
    img, masks = create_multiple_instances_image(n_instances=4)
    test_cases.append(TestCase(
        name="multi_instance_select_one",
        description="Click on one of multiple instances",
        image=img,
        clicks=[Click(252, 252, 1)],  # First instance (top-left)
        boxes=[],
        expected_mask=masks[0],
    ))

    return test_cases


# ============================================================================
# ONNX Model Testing
# ============================================================================
class SAM3OnnxTester:
    """Test SAM3 ONNX models."""

    def __init__(
        self,
        encoder_path: str = DEFAULT_ENCODER_PATH,
        decoder_path: str = DEFAULT_DECODER_PATH,
    ):
        import onnxruntime as ort

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        # Check paths exist
        if encoder_path and os.path.exists(encoder_path):
            print(f"Loading encoder: {encoder_path}")
            self.encoder = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
            self._print_model_info("Encoder", self.encoder)
        else:
            print(f"Encoder not found: {encoder_path}")
            self.encoder = None

        if os.path.exists(decoder_path):
            print(f"Loading decoder: {decoder_path}")
            self.decoder = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
            self._print_model_info("Decoder", self.decoder)
        else:
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        # Check decoder input names to determine format
        self.decoder_input_names = [inp.name for inp in self.decoder.get_inputs()]
        self.supports_mask_input = "mask_input" in self.decoder_input_names or "has_mask_input" in self.decoder_input_names
        self.is_usls_format = "input_points" in self.decoder_input_names

        print(f"\nDecoder format: {'usls' if self.is_usls_format else 'custom'}")
        print(f"Supports mask refinement: {self.supports_mask_input}")

    def _print_model_info(self, name: str, session) -> None:
        """Print model input/output info."""
        print(f"\n{name} inputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        print(f"{name} outputs:")
        for out in session.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for SAM3 encoder.

        Args:
            image: [H, W, 3] RGB uint8

        Returns:
            tensor: [1, 3, 1008, 1008] float32 normalized
        """
        # Resize to 1008x1008
        img = Image.fromarray(image)
        img = img.resize((SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE), Image.BILINEAR)

        # Convert to float and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - SAM3_MEAN) / SAM3_STD

        # Transpose to CHW and add batch
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def encode_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode image to SAM3 embeddings.

        Args:
            image: [H, W, 3] RGB uint8

        Returns:
            dict with embeddings
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")

        input_tensor = self.preprocess_image(image)
        outputs = self.encoder.run(None, {"images": input_tensor})

        return {
            "high_res_feats_0": outputs[0],  # [1, 32, 288, 288]
            "high_res_feats_1": outputs[1],  # [1, 64, 144, 144]
            "image_embed": outputs[2],       # [1, 256, 72, 72]
        }

    def prepare_prompts(
        self,
        clicks: List[Click],
        boxes: List[Box],
        image_size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare prompt inputs for decoder.

        Args:
            clicks: List of Click objects
            boxes: List of Box objects
            image_size: (W, H) of original image

        Returns:
            dict with point_coords, point_labels, etc.
        """
        w, h = image_size
        scale_x = SAM3_IMAGE_SIZE / w
        scale_y = SAM3_IMAGE_SIZE / h

        if self.is_usls_format:
            return self._prepare_prompts_usls(clicks, boxes, scale_x, scale_y)
        else:
            return self._prepare_prompts_custom(clicks, boxes, scale_x, scale_y)

    def _prepare_prompts_usls(
        self,
        clicks: List[Click],
        boxes: List[Box],
        scale_x: float,
        scale_y: float,
    ) -> Dict[str, np.ndarray]:
        """Prepare prompts in usls format."""
        # Points: [batch, 1, num_points, 2]
        n_clicks = len(clicks) if clicks else 1
        point_coords = np.zeros((1, 1, n_clicks, 2), dtype=np.float32)
        point_labels = np.zeros((1, 1, n_clicks), dtype=np.int64)

        if clicks:
            for i, click in enumerate(clicks):
                point_coords[0, 0, i, 0] = click.x * scale_x
                point_coords[0, 0, i, 1] = click.y * scale_y
                point_labels[0, 0, i] = click.label
        else:
            # Dummy point with label -1
            point_labels[0, 0, 0] = -1

        # Boxes: [batch, num_boxes, 4]
        n_boxes = len(boxes)
        box_coords = np.zeros((1, n_boxes, 4), dtype=np.float32)
        for i, box in enumerate(boxes):
            box_coords[0, i] = [
                box.x1 * scale_x,
                box.y1 * scale_y,
                box.x2 * scale_x,
                box.y2 * scale_y,
            ]

        return {
            "input_points": point_coords,
            "input_labels": point_labels,
            "input_boxes": box_coords,
        }

    def _prepare_prompts_custom(
        self,
        clicks: List[Click],
        boxes: List[Box],
        scale_x: float,
        scale_y: float,
    ) -> Dict[str, np.ndarray]:
        """Prepare prompts in custom format (with mask support)."""
        # Combine clicks and box corners
        n_clicks = len(clicks)
        n_boxes = len(boxes)
        total_points = n_clicks + n_boxes * 2

        if total_points == 0:
            total_points = 1  # Need at least one dummy point

        point_coords = np.zeros((1, total_points, 2), dtype=np.float32)
        point_labels = np.zeros((1, total_points), dtype=np.float32)

        # Add clicks
        for i, click in enumerate(clicks):
            point_coords[0, i, 0] = click.x * scale_x
            point_coords[0, i, 1] = click.y * scale_y
            point_labels[0, i] = click.label

        # Add box corners as points (label 2=TL, 3=BR)
        for i, box in enumerate(boxes):
            idx = n_clicks + i * 2
            point_coords[0, idx, 0] = box.x1 * scale_x
            point_coords[0, idx, 1] = box.y1 * scale_y
            point_labels[0, idx] = 2  # Top-left
            point_coords[0, idx + 1, 0] = box.x2 * scale_x
            point_coords[0, idx + 1, 1] = box.y2 * scale_y
            point_labels[0, idx + 1] = 3  # Bottom-right

        # Fill dummy if needed
        if n_clicks == 0 and n_boxes == 0:
            point_labels[0, 0] = -1

        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    def decode(
        self,
        embeddings: Dict[str, np.ndarray],
        prompts: Dict[str, np.ndarray],
        mask_input: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run decoder to get masks.

        Args:
            embeddings: Dict from encode_image()
            prompts: Dict from prepare_prompts()
            mask_input: Optional [1, 1, 288, 288] previous mask for refinement

        Returns:
            dict with masks, iou_scores, etc.
        """
        if self.is_usls_format:
            inputs = {
                "input_points": prompts["input_points"],
                "input_labels": prompts["input_labels"],
                "input_boxes": prompts["input_boxes"],
                "image_embeddings.0": embeddings["high_res_feats_0"],
                "image_embeddings.1": embeddings["high_res_feats_1"],
                "image_embeddings.2": embeddings["image_embed"],
            }
        else:
            inputs = {
                "image_embed": embeddings["image_embed"],
                "high_res_feats_0": embeddings["high_res_feats_0"],
                "high_res_feats_1": embeddings["high_res_feats_1"],
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
            }

            # Add mask input if supported
            if self.supports_mask_input:
                if mask_input is not None:
                    inputs["mask_input"] = mask_input
                    inputs["has_mask_input"] = np.array([1.0], dtype=np.float32)
                else:
                    inputs["mask_input"] = np.zeros((1, 1, 288, 288), dtype=np.float32)
                    inputs["has_mask_input"] = np.array([0.0], dtype=np.float32)

            # Add orig_im_size if needed
            if "orig_im_size" in self.decoder_input_names:
                inputs["orig_im_size"] = np.array([1008, 1008], dtype=np.int64)

        outputs = self.decoder.run(None, inputs)
        output_names = [out.name for out in self.decoder.get_outputs()]

        return dict(zip(output_names, outputs))

    def run_test(self, test_case: TestCase, verbose: bool = True) -> Dict:
        """
        Run a single test case.

        Returns:
            dict with test results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Test: {test_case.name}")
            print(f"Description: {test_case.description}")
            print(f"{'='*60}")

        # Get image size
        h, w = test_case.image.shape[:2]

        # Encode if we have encoder
        if self.encoder:
            embeddings = self.encode_image(test_case.image)
            if verbose:
                print(f"Encoded image: {w}x{h}")
        else:
            # Use dummy embeddings
            embeddings = {
                "high_res_feats_0": np.random.randn(1, 32, 288, 288).astype(np.float32),
                "high_res_feats_1": np.random.randn(1, 64, 144, 144).astype(np.float32),
                "image_embed": np.random.randn(1, 256, 72, 72).astype(np.float32),
            }
            if verbose:
                print("Using dummy embeddings (no encoder)")

        # Prepare prompts
        prompts = self.prepare_prompts(test_case.clicks, test_case.boxes, (w, h))

        if verbose:
            print(f"Clicks: {len(test_case.clicks)}")
            for i, click in enumerate(test_case.clicks):
                print(f"  [{i}] ({click.x:.0f}, {click.y:.0f}) label={click.label}")
            print(f"Boxes: {len(test_case.boxes)}")
            for i, box in enumerate(test_case.boxes):
                print(f"  [{i}] ({box.x1:.0f}, {box.y1:.0f}, {box.x2:.0f}, {box.y2:.0f})")

        # Decode
        outputs = self.decode(embeddings, prompts, test_case.mask_input)

        # Process outputs
        masks = outputs.get("pred_masks") or outputs.get("masks")
        iou_scores = outputs.get("iou_scores") or outputs.get("iou_predictions")
        low_res_masks = outputs.get("low_res_masks")

        if verbose:
            print(f"\nOutputs:")
            print(f"  Masks shape: {masks.shape}")
            print(f"  IoU scores: {iou_scores.flatten()}")
            if low_res_masks is not None:
                print(f"  Low-res masks shape: {low_res_masks.shape}")

        # Select best mask (simple: highest IoU)
        if masks.ndim == 5:
            # [1, 1, 3, H, W] -> [3, H, W]
            masks = masks[0, 0]
            iou_scores = iou_scores.flatten()
        elif masks.ndim == 4:
            # [1, 3, H, W] -> [3, H, W]
            masks = masks[0]
            iou_scores = iou_scores.flatten()

        best_idx = np.argmax(iou_scores)
        best_mask = masks[best_idx]
        best_iou = iou_scores[best_idx]

        if verbose:
            print(f"\nSelected mask {best_idx} (IoU={best_iou:.3f})")
            print(f"  Mask min/max: {best_mask.min():.3f}/{best_mask.max():.3f}")

        # Convert logits to binary mask
        binary_mask = best_mask > 0
        positive_pixels = binary_mask.sum()
        total_pixels = binary_mask.size

        if verbose:
            print(f"  Positive pixels: {positive_pixels}/{total_pixels} ({100*positive_pixels/total_pixels:.1f}%)")

        # Compare with expected mask if provided
        iou_with_expected = None
        if test_case.expected_mask is not None:
            # Resize expected mask to match output
            from PIL import Image
            expected = Image.fromarray(test_case.expected_mask.astype(np.uint8) * 255)
            expected = expected.resize((best_mask.shape[1], best_mask.shape[0]), Image.NEAREST)
            expected = np.array(expected) > 127

            # Compute IoU
            intersection = (binary_mask & expected).sum()
            union = (binary_mask | expected).sum()
            iou_with_expected = intersection / (union + 1e-8)

            if verbose:
                print(f"  IoU with expected: {iou_with_expected:.3f}")

        return {
            "test_name": test_case.name,
            "masks": masks,
            "best_mask": best_mask,
            "best_iou": best_iou,
            "iou_with_expected": iou_with_expected,
            "binary_mask": binary_mask,
            "low_res_masks": low_res_masks,
        }

    def run_all_tests(self, verbose: bool = True) -> List[Dict]:
        """Run all test cases."""
        test_cases = generate_test_cases()
        results = []

        print(f"\n{'#'*60}")
        print(f"Running {len(test_cases)} test cases")
        print(f"{'#'*60}")

        for test_case in test_cases:
            try:
                result = self.run_test(test_case, verbose)
                result["success"] = True
            except Exception as e:
                result = {
                    "test_name": test_case.name,
                    "success": False,
                    "error": str(e),
                }
                if verbose:
                    print(f"❌ Test failed: {e}")
            results.append(result)

        # Summary
        print(f"\n{'#'*60}")
        print("Summary")
        print(f"{'#'*60}")

        passed = sum(1 for r in results if r["success"])
        print(f"Passed: {passed}/{len(results)}")

        for result in results:
            status = "✓" if result["success"] else "❌"
            iou_str = ""
            if "iou_with_expected" in result and result.get("iou_with_expected") is not None:
                iou_str = f" (IoU={result['iou_with_expected']:.3f})"
            elif "best_iou" in result:
                iou_str = f" (pred_IoU={result['best_iou']:.3f})"
            print(f"  {status} {result['test_name']}{iou_str}")

        return results


# ============================================================================
# Mask Refinement Test
# ============================================================================
def test_mask_refinement(tester: SAM3OnnxTester, verbose: bool = True) -> None:
    """
    Test iterative mask refinement.

    Simulates adding clicks one by one and using previous mask as input.
    """
    print(f"\n{'='*60}")
    print("Testing Mask Refinement")
    print(f"{'='*60}")

    if not tester.supports_mask_input:
        print("Decoder does not support mask input - skipping")
        return

    # Create test image
    img, expected_mask = create_circle_image()
    h, w = img.shape[:2]

    # Encode once
    if tester.encoder:
        embeddings = tester.encode_image(img)
    else:
        embeddings = {
            "high_res_feats_0": np.random.randn(1, 32, 288, 288).astype(np.float32),
            "high_res_feats_1": np.random.randn(1, 64, 144, 144).astype(np.float32),
            "image_embed": np.random.randn(1, 256, 72, 72).astype(np.float32),
        }

    # Click sequence: center -> edge -> another edge
    click_sequence = [
        Click(504, 504, 1),  # Center
        Click(504, 304, 1),  # Top edge
        Click(704, 504, 1),  # Right edge
    ]

    prev_low_res_mask = None

    for i, click in enumerate(click_sequence):
        print(f"\nIteration {i+1}: Click at ({click.x}, {click.y})")

        # Use all clicks up to current
        current_clicks = click_sequence[:i+1]
        prompts = tester.prepare_prompts(current_clicks, [], (w, h))

        # Decode with previous mask
        outputs = tester.decode(embeddings, prompts, prev_low_res_mask)

        # Get outputs
        masks = outputs.get("pred_masks") or outputs.get("masks")
        iou_scores = outputs.get("iou_scores") or outputs.get("iou_predictions")
        low_res_masks = outputs.get("low_res_masks")

        # Select best mask
        if masks.ndim == 5:
            masks = masks[0, 0]
            iou_scores = iou_scores.flatten()
        elif masks.ndim == 4:
            masks = masks[0]
            iou_scores = iou_scores.flatten()

        best_idx = np.argmax(iou_scores)
        best_mask = masks[best_idx]

        binary = best_mask > 0
        print(f"  Best mask: idx={best_idx}, IoU={iou_scores[best_idx]:.3f}")
        print(f"  Positive pixels: {binary.sum()} ({100*binary.mean():.1f}%)")

        # Store low-res mask for next iteration
        if low_res_masks is not None:
            if low_res_masks.ndim == 5:
                prev_low_res_mask = low_res_masks[0, 0, best_idx:best_idx+1].reshape(1, 1, 288, 288)
            else:
                prev_low_res_mask = low_res_masks[0, best_idx:best_idx+1].reshape(1, 1, 288, 288)
            # Clamp to [-32, 32]
            prev_low_res_mask = np.clip(prev_low_res_mask, -32, 32)
            print(f"  Stored low-res mask for refinement")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3 ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--encoder-path",
        type=str,
        default=DEFAULT_ENCODER_PATH,
        help="Path to encoder ONNX model",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        default=DEFAULT_DECODER_PATH,
        help="Path to decoder ONNX model",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run tests with synthetic shapes",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Test with a real image",
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Run full test suite",
    )
    parser.add_argument(
        "--test-refinement",
        action="store_true",
        help="Test mask refinement",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )

    args = parser.parse_args()

    # Create tester
    tester = SAM3OnnxTester(
        encoder_path=args.encoder_path if os.path.exists(args.encoder_path) else None,
        decoder_path=args.decoder_path,
    )

    verbose = not args.quiet

    if args.synthetic or args.full_suite:
        tester.run_all_tests(verbose=verbose)

    if args.test_refinement or args.full_suite:
        test_mask_refinement(tester, verbose=verbose)

    if args.image:
        # Load and test with real image
        print(f"\n{'='*60}")
        print(f"Testing with image: {args.image}")
        print(f"{'='*60}")

        img = np.array(Image.open(args.image).convert('RGB'))
        h, w = img.shape[:2]

        # Simple test: click in center
        test_case = TestCase(
            name="real_image_center",
            description=f"Click in center of {args.image}",
            image=img,
            clicks=[Click(w // 2, h // 2, 1)],
            boxes=[],
        )

        tester.run_test(test_case, verbose=verbose)

    if not (args.synthetic or args.full_suite or args.test_refinement or args.image):
        # Default: run synthetic tests
        tester.run_all_tests(verbose=verbose)


if __name__ == "__main__":
    main()
