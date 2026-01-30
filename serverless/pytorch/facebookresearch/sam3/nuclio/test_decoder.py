#!/usr/bin/env python3
"""
Test script to verify SAM2 and SAM3 decoder inference matches browser behavior.

This script tests the same logic used in the browser-side inference workers
to ensure the ONNX models work correctly with Python/onnxruntime before
testing in the browser.

Usage:
    python test_decoder.py --model sam2 --image /path/to/test.jpg
    python test_decoder.py --model sam3 --image /path/to/test.jpg
    python test_decoder.py --model sam3 --real-embeddings  # Uses real server embeddings
"""

import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import json
import sys
import os


# ===================== SAM2 Configuration =====================
SAM2_IMAGE_SIZE = 1024
SAM2_DECODER_PATH = "/home/jahs/GitHub/cvat/cvat-ui/plugins/sam2/assets/sam2.1_hiera_large.decoder.onnx"
SAM2_ENCODER_URL = "pth-facebookresearch-sam2.1-hiera-large"  # Model ID for server

# SAM2 embedding shapes (from server)
# image_embed: [1, 256, 64, 64]
# high_res_feats_0: [1, 32, 256, 256]
# high_res_feats_1: [1, 64, 128, 128]


# ===================== SAM3 Configuration =====================
SAM3_IMAGE_SIZE = 1008
SAM3_DECODER_USLS_PATH = "/home/jahs/GitHub/cvat/cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder.onnx"
SAM3_DECODER_CUSTOM_PATH = "/home/jahs/GitHub/cvat/cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx"
SAM3_ENCODER_PATH = "/tmp/vision_encoder.onnx"  # Downloaded HuggingFace encoder

# SAM3 embedding shapes (from server)
# image_embeddings.0: [1, 32, 288, 288]
# image_embeddings.1: [1, 64, 144, 144]
# image_embeddings.2: [1, 256, 72, 72]


def create_dummy_sam2_embeddings():
    """Create dummy embeddings matching SAM2 server output shapes."""
    return {
        'image_embed': np.random.randn(1, 256, 64, 64).astype(np.float32),
        'high_res_feats_0': np.random.randn(1, 32, 256, 256).astype(np.float32),
        'high_res_feats_1': np.random.randn(1, 64, 128, 128).astype(np.float32),
    }


def create_dummy_sam3_embeddings():
    """Create dummy embeddings matching SAM3 server output shapes."""
    return {
        'image_embeddings.0': np.random.randn(1, 32, 288, 288).astype(np.float32),
        'image_embeddings.1': np.random.randn(1, 64, 144, 144).astype(np.float32),
        'image_embeddings.2': np.random.randn(1, 256, 72, 72).astype(np.float32),
    }


def test_sam2_decoder(embeddings: dict, clicks: list, image_size: tuple, mask_input=None):
    """
    Test SAM2 decoder with the same logic as inference.worker.ts

    Args:
        embeddings: Dict with image_embed, high_res_feats_0, high_res_feats_1
        clicks: List of dicts with 'x', 'y', 'label' (0=neg, 1=pos)
        image_size: (width, height) of original image
        mask_input: Optional previous mask for refinement

    Returns:
        Dict with masks, iou_predictions, etc.
    """
    print("\n" + "=" * 60)
    print("Testing SAM2 Decoder")
    print("=" * 60)

    # Load decoder
    print(f"Loading decoder: {SAM2_DECODER_PATH}")
    session = ort.InferenceSession(SAM2_DECODER_PATH, providers=['CPUExecutionProvider'])

    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Calculate model scale (same as browser)
    w, h = image_size
    scale_x = SAM2_IMAGE_SIZE / w
    scale_y = SAM2_IMAGE_SIZE / h
    print(f"\nImage size: {w}x{h}")
    print(f"Model scale: scaleX={scale_x:.4f}, scaleY={scale_y:.4f}")

    # Prepare point coordinates and labels (same as browser modelData function)
    n = len(clicks)
    point_coords = np.zeros((1, n, 2), dtype=np.float32)
    point_labels = np.zeros((1, n), dtype=np.float32)

    for i, click in enumerate(clicks):
        point_coords[0, i, 0] = click['x'] * scale_x
        point_coords[0, i, 1] = click['y'] * scale_y
        point_labels[0, i] = click['label']

    print(f"\nPoints ({n} total):")
    for i, click in enumerate(clicks):
        scaled_x = click['x'] * scale_x
        scaled_y = click['y'] * scale_y
        print(f"  [{i}] original=({click['x']}, {click['y']}) -> scaled=({scaled_x:.1f}, {scaled_y:.1f}), label={click['label']}")

    # Prepare orig_im_size
    orig_im_size = np.array([h, w], dtype=np.int32)

    # Prepare mask input
    if mask_input is None:
        mask_input_tensor = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
    else:
        mask_input_tensor = mask_input
        has_mask_input = np.array([1.0], dtype=np.float32)

    # Run inference
    print("\nRunning inference...")
    inputs = {
        'image_embed': embeddings['image_embed'],
        'high_res_feats_0': embeddings['high_res_feats_0'],
        'high_res_feats_1': embeddings['high_res_feats_1'],
        'point_coords': point_coords,
        'point_labels': point_labels,
        'orig_im_size': orig_im_size,
        'mask_input': mask_input_tensor,
        'has_mask_input': has_mask_input,
    }

    print("\nInput shapes:")
    for name, arr in inputs.items():
        print(f"  {name}: {arr.shape} ({arr.dtype})")

    try:
        outputs = session.run(None, inputs)
        output_names = [o.name for o in session.get_outputs()]
        results = dict(zip(output_names, outputs))

        print("\nOutput shapes:")
        for name, arr in results.items():
            if isinstance(arr, np.ndarray):
                print(f"  {name}: {arr.shape} ({arr.dtype})")
            else:
                print(f"  {name}: {arr}")

        # Process masks
        masks = results.get('masks')
        if masks is not None:
            print(f"\nMask stats: min={masks.min():.3f}, max={masks.max():.3f}, mean={masks.mean():.3f}")
            # Apply sigmoid if needed (check if values are logits)
            if masks.min() < 0 or masks.max() > 1:
                print("  -> Values appear to be logits, applying sigmoid")
                masks_prob = 1.0 / (1.0 + np.exp(-np.clip(masks, -50, 50)))
            else:
                masks_prob = masks
            print(f"  -> After sigmoid: min={masks_prob.min():.3f}, max={masks_prob.max():.3f}")

            # Count positive pixels
            binary_mask = masks_prob > 0.5
            positive_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            print(f"  -> Binary mask: {positive_pixels}/{total_pixels} positive pixels ({100*positive_pixels/total_pixels:.1f}%)")

        return results

    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        raise


def test_sam3_decoder_usls(embeddings: dict, clicks: list, boxes: list, image_size: tuple):
    """
    Test SAM3 usls decoder with the same logic as inference.worker.ts

    Args:
        embeddings: Dict with image_embeddings.0, .1, .2
        clicks: List of dicts with 'x', 'y', 'label' (0=neg, 1=pos, -1=dummy)
        boxes: List of [x1, y1, x2, y2] boxes
        image_size: (width, height) of original image

    Returns:
        Dict with masks, iou_scores, etc.
    """
    print("\n" + "=" * 60)
    print("Testing SAM3 Decoder (usls)")
    print("=" * 60)

    # Load decoder
    print(f"Loading decoder: {SAM3_DECODER_USLS_PATH}")
    session = ort.InferenceSession(SAM3_DECODER_USLS_PATH, providers=['CPUExecutionProvider'])

    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Calculate model scale
    w, h = image_size
    scale_x = SAM3_IMAGE_SIZE / w
    scale_y = SAM3_IMAGE_SIZE / h
    print(f"\nImage size: {w}x{h}")
    print(f"Model scale: scaleX={scale_x:.4f}, scaleY={scale_y:.4f}")

    # Prepare points (usls format: [batch, 1, num_points, 2])
    n = len(clicks) if clicks else 1
    point_coords = np.zeros((1, 1, n, 2), dtype=np.float32)
    point_labels = np.zeros((1, 1, n), dtype=np.int64)  # INT64 for usls!

    if clicks:
        for i, click in enumerate(clicks):
            point_coords[0, 0, i, 0] = click['x'] * scale_x
            point_coords[0, 0, i, 1] = click['y'] * scale_y
            point_labels[0, 0, i] = int(click['label'])
    else:
        # Dummy point with label -1 when no points
        point_coords[0, 0, 0, 0] = 0
        point_coords[0, 0, 0, 1] = 0
        point_labels[0, 0, 0] = -1

    print(f"\nPoints ({n} total):")
    for i in range(n):
        x = point_coords[0, 0, i, 0]
        y = point_coords[0, 0, i, 1]
        label = point_labels[0, 0, i]
        print(f"  [{i}] ({x:.1f}, {y:.1f}), label={label}")

    # Prepare boxes (usls format: [batch, num_boxes, 4])
    num_boxes = len(boxes)
    box_coords = np.zeros((1, num_boxes, 4), dtype=np.float32) if num_boxes > 0 else np.zeros((1, 0, 4), dtype=np.float32)
    for i, box in enumerate(boxes):
        box_coords[0, i, 0] = box[0] * scale_x
        box_coords[0, i, 1] = box[1] * scale_y
        box_coords[0, i, 2] = box[2] * scale_x
        box_coords[0, i, 3] = box[3] * scale_y

    print(f"\nBoxes ({num_boxes} total):")
    for i in range(num_boxes):
        print(f"  [{i}] [{box_coords[0, i, 0]:.1f}, {box_coords[0, i, 1]:.1f}, {box_coords[0, i, 2]:.1f}, {box_coords[0, i, 3]:.1f}]")

    # Run inference
    print("\nRunning inference...")
    inputs = {
        'input_points': point_coords,
        'input_labels': point_labels,
        'input_boxes': box_coords,
        'image_embeddings.0': embeddings['image_embeddings.0'],
        'image_embeddings.1': embeddings['image_embeddings.1'],
        'image_embeddings.2': embeddings['image_embeddings.2'],
    }

    print("\nInput shapes:")
    for name, arr in inputs.items():
        print(f"  {name}: {arr.shape} ({arr.dtype})")

    try:
        outputs = session.run(None, inputs)
        output_names = [o.name for o in session.get_outputs()]
        results = dict(zip(output_names, outputs))

        print("\nOutput shapes:")
        for name, arr in results.items():
            if isinstance(arr, np.ndarray):
                print(f"  {name}: {arr.shape} ({arr.dtype})")
            else:
                print(f"  {name}: {arr}")

        # Process masks
        masks = results.get('pred_masks')
        if masks is not None:
            print(f"\nMask stats: min={masks.min():.3f}, max={masks.max():.3f}, mean={masks.mean():.3f}")
            # Apply sigmoid if needed
            if masks.min() < 0 or masks.max() > 1:
                print("  -> Values appear to be logits, applying sigmoid")
                masks_prob = 1.0 / (1.0 + np.exp(-np.clip(masks, -50, 50)))
            else:
                masks_prob = masks
            print(f"  -> After sigmoid: min={masks_prob.min():.3f}, max={masks_prob.max():.3f}")

            # Count positive pixels
            binary_mask = masks_prob > 0.5
            positive_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            print(f"  -> Binary mask: {positive_pixels}/{total_pixels} positive pixels ({100*positive_pixels/total_pixels:.1f}%)")

        # IoU scores
        iou = results.get('iou_scores')
        if iou is not None:
            print(f"\nIoU scores: {iou.flatten()}")

        return results

    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        raise


def test_sam3_decoder_custom(embeddings: dict, clicks: list, boxes: list, image_size: tuple, mask_input=None):
    """
    Test SAM3 custom decoder (with mask input) - NOTE: Has random weights!
    """
    print("\n" + "=" * 60)
    print("Testing SAM3 Decoder (custom with mask input)")
    print("WARNING: This decoder has RANDOM WEIGHTS - expect garbage output!")
    print("=" * 60)

    # Load decoder
    print(f"Loading decoder: {SAM3_DECODER_CUSTOM_PATH}")
    session = ort.InferenceSession(SAM3_DECODER_CUSTOM_PATH, providers=['CPUExecutionProvider'])

    # Print model info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Calculate model scale
    w, h = image_size
    scale_x = SAM3_IMAGE_SIZE / w
    scale_y = SAM3_IMAGE_SIZE / h

    # Prepare points (custom format: [batch, num_points, 2])
    # Convert boxes to points with labels 2 and 3
    all_points = list(clicks) if clicks else []

    # Add dummy point if no points
    if not all_points and not boxes:
        all_points.append({'x': 0, 'y': 0, 'label': -1})

    # Add box corners as points
    for box in boxes:
        all_points.append({'x': box[0], 'y': box[1], 'label': 2})  # top-left
        all_points.append({'x': box[2], 'y': box[3], 'label': 3})  # bottom-right

    n = len(all_points)
    point_coords = np.zeros((1, n, 2), dtype=np.float32)
    point_labels = np.zeros((1, n), dtype=np.float32)  # FLOAT32 for custom decoder

    for i, pt in enumerate(all_points):
        point_coords[0, i, 0] = pt['x'] * scale_x
        point_coords[0, i, 1] = pt['y'] * scale_y
        point_labels[0, i] = pt['label']

    # Prepare mask input
    if mask_input is None:
        mask_input_tensor = np.zeros((1, 1, 288, 288), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
    else:
        mask_input_tensor = mask_input
        has_mask_input = np.array([1.0], dtype=np.float32)

    # Run inference
    print("\nRunning inference...")
    inputs = {
        'image_embed': embeddings['image_embeddings.2'],  # 256-channel main embed
        'high_res_feats_0': embeddings['image_embeddings.0'],  # 32-channel
        'high_res_feats_1': embeddings['image_embeddings.1'],  # 64-channel
        'point_coords': point_coords,
        'point_labels': point_labels,
        'mask_input': mask_input_tensor,
        'has_mask_input': has_mask_input,
    }

    print("\nInput shapes:")
    for name, arr in inputs.items():
        print(f"  {name}: {arr.shape} ({arr.dtype})")

    try:
        outputs = session.run(None, inputs)
        output_names = [o.name for o in session.get_outputs()]
        results = dict(zip(output_names, outputs))

        print("\nOutput shapes:")
        for name, arr in results.items():
            if isinstance(arr, np.ndarray):
                print(f"  {name}: {arr.shape} ({arr.dtype})")
            else:
                print(f"  {name}: {arr}")

        return results

    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        raise


def preprocess_image_sam3(image_path: str):
    """
    Preprocess image for SAM3 encoder (same as model_handler.py does).

    Returns:
        pixel_values: numpy array [1, 3, 1008, 1008]
        original_size: (width, height)
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)

    # Resize to 1008x1008
    img_resized = img.resize((SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE), Image.BILINEAR)

    # Convert to float32 and normalize
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # Normalize with mean=0.5, std=0.5
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img_array = (img_array - mean) / std

    # Convert to NCHW format
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # CHW -> NCHW

    return img_array.astype(np.float32), original_size


def test_sam3_with_real_encoder(image_path: str, clicks: list, boxes: list):
    """
    Test SAM3 end-to-end with real encoder embeddings.

    Args:
        image_path: Path to test image
        clicks: List of dicts with 'x', 'y', 'label'
        boxes: List of [x1, y1, x2, y2] boxes (in original image coordinates)
    """
    print("\n" + "=" * 60)
    print("Testing SAM3 with REAL encoder embeddings")
    print("=" * 60)

    # Check encoder exists
    if not os.path.exists(SAM3_ENCODER_PATH):
        print(f"\n⚠️ SAM3 encoder not found at {SAM3_ENCODER_PATH}")
        print("Download with:")
        print('  curl -L -o /tmp/vision_encoder.onnx "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx"')
        print('  curl -L -o /tmp/vision_encoder.onnx_data "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx_data"')
        return None

    # Load encoder
    print(f"Loading encoder: {SAM3_ENCODER_PATH}")
    # Force CPU to avoid GPU OOM issues during testing
    encoder = ort.InferenceSession(SAM3_ENCODER_PATH, providers=['CPUExecutionProvider'])
    print(f"  Using provider: {encoder.get_providers()[0]}")

    # Load decoder
    print(f"Loading decoder: {SAM3_DECODER_USLS_PATH}")
    decoder = ort.InferenceSession(SAM3_DECODER_USLS_PATH, providers=['CPUExecutionProvider'])

    # Preprocess image
    print(f"\nLoading image: {image_path}")
    pixel_values, original_size = preprocess_image_sam3(image_path)
    w, h = original_size
    print(f"  Original size: {w}x{h}")
    print(f"  Preprocessed shape: {pixel_values.shape}")
    print(f"  Value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

    # Run encoder
    print("\nRunning encoder...")
    encoder_outputs = encoder.run(None, {'pixel_values': pixel_values})
    encoder_output_names = [o.name for o in encoder.get_outputs()]
    embeddings = dict(zip(encoder_output_names, encoder_outputs))

    print("Encoder outputs:")
    for name, arr in embeddings.items():
        print(f"  {name}: {arr.shape}")

    # Prepare decoder inputs
    scale_x = SAM3_IMAGE_SIZE / w
    scale_y = SAM3_IMAGE_SIZE / h
    print(f"\nModel scale: scaleX={scale_x:.4f}, scaleY={scale_y:.4f}")

    # Prepare points
    n = len(clicks) if clicks else 1
    point_coords = np.zeros((1, 1, n, 2), dtype=np.float32)
    point_labels = np.zeros((1, 1, n), dtype=np.int64)

    if clicks:
        for i, click in enumerate(clicks):
            point_coords[0, 0, i, 0] = click['x'] * scale_x
            point_coords[0, 0, i, 1] = click['y'] * scale_y
            point_labels[0, 0, i] = int(click['label'])
    else:
        point_coords[0, 0, 0, 0] = 0
        point_coords[0, 0, 0, 1] = 0
        point_labels[0, 0, 0] = -1

    print(f"\nPoints ({n} total):")
    for i in range(n):
        x = point_coords[0, 0, i, 0]
        y = point_coords[0, 0, i, 1]
        label = point_labels[0, 0, i]
        print(f"  [{i}] ({x:.1f}, {y:.1f}), label={label}")

    # Prepare boxes
    num_boxes = len(boxes)
    box_coords = np.zeros((1, num_boxes, 4), dtype=np.float32) if num_boxes > 0 else np.zeros((1, 0, 4), dtype=np.float32)
    for i, box in enumerate(boxes):
        box_coords[0, i, 0] = box[0] * scale_x
        box_coords[0, i, 1] = box[1] * scale_y
        box_coords[0, i, 2] = box[2] * scale_x
        box_coords[0, i, 3] = box[3] * scale_y

    print(f"\nBoxes ({num_boxes} total):")
    for i in range(num_boxes):
        print(f"  [{i}] [{box_coords[0, i, 0]:.1f}, {box_coords[0, i, 1]:.1f}, {box_coords[0, i, 2]:.1f}, {box_coords[0, i, 3]:.1f}]")

    # Run decoder
    print("\nRunning decoder...")
    decoder_inputs = {
        'input_points': point_coords,
        'input_labels': point_labels,
        'input_boxes': box_coords,
        'image_embeddings.0': embeddings['image_embeddings.0'],
        'image_embeddings.1': embeddings['image_embeddings.1'],
        'image_embeddings.2': embeddings['image_embeddings.2'],
    }

    decoder_outputs = decoder.run(None, decoder_inputs)
    decoder_output_names = [o.name for o in decoder.get_outputs()]
    results = dict(zip(decoder_output_names, decoder_outputs))

    print("\nDecoder outputs:")
    for name, arr in results.items():
        print(f"  {name}: {arr.shape}")

    # Process masks
    pred_masks = results['pred_masks']
    iou_scores = results['iou_scores']

    print(f"\nMask stats: min={pred_masks.min():.3f}, max={pred_masks.max():.3f}")
    print(f"IoU scores: {iou_scores.flatten()}")

    # Find best mask
    iou_flat = iou_scores.flatten()
    best_idx = np.argmax(iou_flat)
    print(f"Best mask index: {best_idx} (IoU={iou_flat[best_idx]:.4f})")

    # Extract best mask and apply sigmoid
    # Shape: [1, 1, 3, 288, 288]
    best_mask_logits = pred_masks[0, 0, best_idx]  # [288, 288]
    best_mask_prob = 1.0 / (1.0 + np.exp(-np.clip(best_mask_logits, -50, 50)))
    binary_mask = (best_mask_prob > 0.5).astype(np.uint8)

    positive_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    print(f"Binary mask: {positive_pixels}/{total_pixels} positive pixels ({100*positive_pixels/total_pixels:.1f}%)")

    # Find bounding box
    ys, xs = np.where(binary_mask > 0)
    if len(xs) > 0:
        xtl, ytl, xbr, ybr = xs.min(), ys.min(), xs.max(), ys.max()
        print(f"Mask bounding box (in 288x288): [{xtl}, {ytl}, {xbr}, {ybr}]")

        # Scale to original image size
        mask_h, mask_w = binary_mask.shape
        xtl_orig = int(xtl / mask_w * w)
        ytl_orig = int(ytl / mask_h * h)
        xbr_orig = int(xbr / mask_w * w)
        ybr_orig = int(ybr / mask_h * h)
        print(f"Mask bounding box (original): [{xtl_orig}, {ytl_orig}, {xbr_orig}, {ybr_orig}]")
    else:
        print("No positive pixels in mask!")

    # Save visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        img = Image.open(image_path)
        axes[0].imshow(img)
        axes[0].set_title(f'Original ({w}x{h})')

        # Add click markers
        for click in clicks:
            color = 'g' if click['label'] == 1 else 'r'
            axes[0].plot(click['x'], click['y'], f'{color}o', markersize=10)

        # Add box
        for box in boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                  fill=False, color='blue', linewidth=2)
            axes[0].add_patch(rect)

        # Mask at 288x288
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title(f'Mask (288x288)')

        # Mask probability
        axes[2].imshow(best_mask_prob, cmap='jet')
        axes[2].set_title(f'Probability (IoU={iou_flat[best_idx]:.3f})')

        output_path = '/tmp/sam3_test_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
        plt.close()

    except ImportError:
        print("\n⚠️ matplotlib not available, skipping visualization")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test SAM2/SAM3 decoder inference')
    parser.add_argument('--model', choices=['sam2', 'sam3-usls', 'sam3-custom', 'sam3-real', 'all'],
                        default='all', help='Which model to test')
    parser.add_argument('--image-size', type=int, nargs=2, default=[800, 600],
                        help='Simulated image size (width height)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image (required for sam3-real)')
    parser.add_argument('--point', type=int, nargs=2, default=None,
                        help='Test point (x y) in original image coordinates')
    parser.add_argument('--box', type=int, nargs=4, default=None,
                        help='Test box (x1 y1 x2 y2) in original image coordinates')
    args = parser.parse_args()

    image_size = tuple(args.image_size)

    # Test clicks: one positive point in the center
    if args.point:
        clicks = [{'x': args.point[0], 'y': args.point[1], 'label': 1}]
    else:
        clicks = [{'x': image_size[0] // 2, 'y': image_size[1] // 2, 'label': 1}]

    # Test box
    if args.box:
        boxes = [[args.box[0], args.box[1], args.box[2], args.box[3]]]
    else:
        boxes = [[100, 100, 300, 300]]  # [x1, y1, x2, y2]

    print("=" * 60)
    print("SAM Decoder Test Suite")
    print("=" * 60)
    print(f"Simulated image size: {image_size[0]}x{image_size[1]}")
    print(f"Test clicks: {clicks}")
    print(f"Test boxes: {boxes}")

    if args.model in ['sam2', 'all']:
        try:
            embeddings = create_dummy_sam2_embeddings()
            test_sam2_decoder(embeddings, clicks, image_size)
            print("\n✅ SAM2 decoder test PASSED (inference works)")
        except FileNotFoundError:
            print(f"\n⚠️ SAM2 decoder not found at {SAM2_DECODER_PATH}")
        except Exception as e:
            print(f"\n❌ SAM2 decoder test FAILED: {e}")

    if args.model in ['sam3-usls', 'all']:
        try:
            embeddings = create_dummy_sam3_embeddings()
            test_sam3_decoder_usls(embeddings, clicks, boxes, image_size)
            print("\n✅ SAM3 usls decoder test PASSED (inference works)")
        except FileNotFoundError:
            print(f"\n⚠️ SAM3 usls decoder not found at {SAM3_DECODER_USLS_PATH}")
        except Exception as e:
            print(f"\n❌ SAM3 usls decoder test FAILED: {e}")

    if args.model in ['sam3-custom', 'all']:
        try:
            embeddings = create_dummy_sam3_embeddings()
            test_sam3_decoder_custom(embeddings, clicks, boxes, image_size)
            print("\n✅ SAM3 custom decoder test PASSED (inference works, but output is garbage due to random weights)")
        except FileNotFoundError:
            print(f"\n⚠️ SAM3 custom decoder not found at {SAM3_DECODER_CUSTOM_PATH}")
        except Exception as e:
            print(f"\n❌ SAM3 custom decoder test FAILED: {e}")

    if args.model == 'sam3-real':
        if not args.image:
            print("\n❌ --image is required for sam3-real test")
            sys.exit(1)

        # Get image size for clicks/boxes if not specified
        img = Image.open(args.image)
        w, h = img.size

        if not args.point:
            clicks = [{'x': w // 2, 'y': h // 2, 'label': 1}]

        if not args.box:
            # Use a box around center
            cx, cy = w // 2, h // 2
            bw, bh = w // 4, h // 4
            boxes = [[cx - bw, cy - bh, cx + bw, cy + bh]]

        try:
            test_sam3_with_real_encoder(args.image, clicks, boxes)
            print("\n✅ SAM3 real encoder test completed")
        except Exception as e:
            import traceback
            print(f"\n❌ SAM3 real encoder test FAILED: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
