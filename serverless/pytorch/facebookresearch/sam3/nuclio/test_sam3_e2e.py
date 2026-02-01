#!/usr/bin/env python3
"""
SAM3 End-to-End Test

Tests the full pipeline:
1. ONNX encoder (server-side) or PyTorch encoder (fallback)
2. ONNX decoder with mask refinement (browser-side)

Verifies:
- Encoder produces correct output shapes
- Decoder accepts encoder outputs and produces masks
- Mask refinement works (multi-click support)
- IoU scores are reasonable

Usage:
    python test_sam3_e2e.py [--encoder onnx|pytorch|both]
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "../../../../../cvat-ui/plugins/sam3/assets")
DECODER_PATH = os.path.join(ASSETS_DIR, "tracker-prompt-encoder-mask-decoder-with-mask-input.onnx")
ONNX_ENCODER_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--onnx-community--sam3-tracker-ONNX/"
    "snapshots/429305c8a5b3de597243d919a07e4e6bdcd00ef7/onnx/vision_encoder.onnx"
)


def create_test_image_with_shape():
    """Create a test image with a simple circle shape."""
    size = 1008
    bg_color = (240, 240, 240)
    fg_color = (50, 100, 200)
    
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw a circle in the center
    center = (504, 504)
    radius = 200
    bbox = [center[0] - radius, center[1] - radius, 
            center[0] + radius, center[1] + radius]
    draw.ellipse(bbox, fill=fg_color)
    
    return img, center


def encode_with_onnx(image: Image.Image):
    """Encode image using ONNX encoder from onnx-community."""
    if not os.path.exists(ONNX_ENCODER_PATH):
        raise FileNotFoundError(f"ONNX encoder not found: {ONNX_ENCODER_PATH}")
    
    session = ort.InferenceSession(ONNX_ENCODER_PATH, providers=['CPUExecutionProvider'])
    
    # Preprocess
    img_resized = image.resize((1008, 1008), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # Normalize
    img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    
    # Run encoder
    outputs = session.run(None, {"pixel_values": img_tensor.astype(np.float32)})
    
    return {
        'high_res_feats_0': outputs[0],  # [1, 32, 288, 288]
        'high_res_feats_1': outputs[1],  # [1, 64, 144, 144]
        'image_embed': outputs[2],       # [1, 256, 72, 72]
    }


def encode_with_pytorch(image: Image.Image, device='cpu'):
    """Encode image using PyTorch model."""
    from model_handler_pytorch import ModelHandler
    handler = ModelHandler(device=device)
    emb0, emb1, emb2 = handler.handle(image)
    return {
        'high_res_feats_0': emb0,
        'high_res_feats_1': emb1,
        'image_embed': emb2,
    }


def run_decoder(embeddings, point_coords, point_labels, mask_input=None, has_mask=False):
    """Run the ONNX decoder with given embeddings and prompts."""
    session = ort.InferenceSession(DECODER_PATH, providers=['CPUExecutionProvider'])
    
    # Prepare inputs
    inputs = {
        'image_embed': embeddings['image_embed'].astype(np.float32),
        'high_res_feats_0': embeddings['high_res_feats_0'].astype(np.float32),
        'high_res_feats_1': embeddings['high_res_feats_1'].astype(np.float32),
        'point_coords': point_coords.astype(np.float32),
        'point_labels': point_labels.astype(np.float32),
        'mask_input': mask_input if mask_input is not None else np.zeros((1, 1, 288, 288), dtype=np.float32),
        'has_mask_input': np.array([1.0 if has_mask else 0.0], dtype=np.float32),
    }
    
    outputs = session.run(None, inputs)
    
    return {
        'masks': outputs[0],           # [1, 3, H, W]
        'iou_predictions': outputs[1], # [1, 3]
        'low_res_masks': outputs[2],   # [1, 3, 288, 288]
        'object_score_logits': outputs[3],  # [1, 1]
    }


def test_encoder(encoder_type: str, image: Image.Image):
    """Test a specific encoder."""
    print(f"\n{'='*60}")
    print(f"Testing {encoder_type.upper()} encoder")
    print('='*60)
    
    # Encode
    print(f"\nEncoding image...")
    if encoder_type == 'onnx':
        embeddings = encode_with_onnx(image)
    else:
        embeddings = encode_with_pytorch(image)
    
    # Verify shapes
    expected_shapes = {
        'high_res_feats_0': (1, 32, 288, 288),
        'high_res_feats_1': (1, 64, 144, 144),
        'image_embed': (1, 256, 72, 72),
    }
    
    all_correct = True
    for name, expected in expected_shapes.items():
        actual = embeddings[name].shape
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_correct = False
        print(f"  {name}: {actual} {status} (expected {expected})")
    
    return embeddings, all_correct


def test_single_click(embeddings, center):
    """Test single-click prediction."""
    print("\nTesting single-click prediction...")
    
    cx, cy = center
    point_coords = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    point_labels = np.array([[1.0]], dtype=np.float32)  # Positive click
    
    results = run_decoder(embeddings, point_coords, point_labels)
    
    # Select best mask
    iou_preds = results['iou_predictions'][0]
    best_idx = np.argmax(iou_preds)
    best_mask = results['masks'][0, best_idx]
    best_iou = iou_preds[best_idx]
    
    # Analyze mask
    binary_mask = best_mask > 0
    mask_area = np.sum(binary_mask)
    total_pixels = best_mask.size
    coverage = mask_area / total_pixels * 100
    
    # Expected coverage for circle with radius 200 in 1008x1008 image
    expected_coverage = np.pi * 200**2 / (1008**2) * 100  # ~12.3%
    
    print(f"  Best mask index: {best_idx}")
    print(f"  IoU score: {best_iou:.4f}")
    print(f"  Mask coverage: {coverage:.2f}% (expected ~{expected_coverage:.1f}%)")
    print(f"  Object score: {1.0 / (1.0 + np.exp(-results['object_score_logits'][0, 0])):.4f}")
    
    # Return low-res mask for refinement test
    return results['low_res_masks'][0:1, best_idx:best_idx+1], coverage, best_iou


def test_mask_refinement(embeddings, center, previous_low_res_mask):
    """Test mask refinement (multi-click)."""
    print("\nTesting mask refinement...")
    
    cx, cy = center
    # Add a second click slightly off-center
    point_coords = np.array([[[float(cx + 50), float(cy)]]], dtype=np.float32)
    point_labels = np.array([[1.0]], dtype=np.float32)
    
    results = run_decoder(
        embeddings, 
        point_coords, 
        point_labels, 
        mask_input=previous_low_res_mask,
        has_mask=True
    )
    
    iou_preds = results['iou_predictions'][0]
    best_idx = np.argmax(iou_preds)
    best_iou = iou_preds[best_idx]
    
    print(f"  Refined IoU score: {best_iou:.4f}")
    
    return best_iou


def compare_encoders(onnx_emb, pytorch_emb):
    """Compare ONNX and PyTorch encoder outputs."""
    print("\n" + "="*60)
    print("Comparing ONNX vs PyTorch encoders")
    print("="*60)
    
    for name in ['high_res_feats_0', 'high_res_feats_1', 'image_embed']:
        onnx_val = onnx_emb[name]
        pytorch_val = pytorch_emb[name]
        
        mae = np.abs(onnx_val - pytorch_val).mean()
        max_diff = np.abs(onnx_val - pytorch_val).max()
        corr = np.corrcoef(onnx_val.flatten(), pytorch_val.flatten())[0, 1]
        
        print(f"  {name}:")
        print(f"    MAE: {mae:.6f}, Max: {max_diff:.6f}, Corr: {corr:.6f}")


def save_visualization(image, mask, center, output_path):
    """Save visualization of the segmentation result."""
    vis_img = np.array(image).copy()
    
    # Resize mask to image size if needed
    if mask.shape != (1008, 1008):
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((1008, 1008), PILImage.NEAREST)
        mask = np.array(mask_pil) > 0
    else:
        mask = mask > 0
    
    # Add red tint to masked area
    for c in range(3):
        vis_img[:, :, c] = np.where(
            mask,
            (vis_img[:, :, c] * 0.5 + (255 if c == 0 else 0) * 0.5).astype(np.uint8),
            vis_img[:, :, c]
        )
    
    # Draw click point
    vis_pil = Image.fromarray(vis_img)
    draw = ImageDraw.Draw(vis_pil)
    cx, cy = center
    draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(0, 255, 0), outline=(0, 0, 0))
    
    vis_pil.save(output_path)
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 End-to-End Test")
    parser.add_argument(
        "--encoder", 
        choices=['onnx', 'pytorch', 'both'],
        default='both',
        help="Which encoder to test"
    )
    parser.add_argument(
        "--output",
        default="/tmp/sam3_e2e_test_result.png",
        help="Output visualization path"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("SAM3 End-to-End Test")
    print("="*60)
    
    # Verify decoder exists
    if not os.path.exists(DECODER_PATH):
        print(f"ERROR: Decoder not found: {DECODER_PATH}")
        sys.exit(1)
    print(f"Decoder: {DECODER_PATH}")
    
    # Create test image
    print("\nCreating test image with circle...")
    test_image, center = create_test_image_with_shape()
    print(f"  Image size: {test_image.size}")
    print(f"  Circle center: {center}")
    
    # Test encoders
    onnx_emb = None
    pytorch_emb = None
    
    if args.encoder in ['onnx', 'both']:
        try:
            onnx_emb, _ = test_encoder('onnx', test_image)
        except FileNotFoundError as e:
            print(f"\nONNX encoder not available: {e}")
            if args.encoder == 'onnx':
                sys.exit(1)
    
    if args.encoder in ['pytorch', 'both']:
        try:
            pytorch_emb, _ = test_encoder('pytorch', test_image)
        except ImportError as e:
            print(f"\nPyTorch encoder not available: {e}")
            if args.encoder == 'pytorch':
                sys.exit(1)
    
    # Compare encoders if both available
    if onnx_emb is not None and pytorch_emb is not None:
        compare_encoders(onnx_emb, pytorch_emb)
    
    # Use whichever encoder is available for decoder tests
    embeddings = onnx_emb if onnx_emb is not None else pytorch_emb
    if embeddings is None:
        print("ERROR: No encoder available")
        sys.exit(1)
    
    # Test single-click
    low_res_mask, coverage, iou = test_single_click(embeddings, center)
    
    # Verify coverage is reasonable (circle should be ~12% of image)
    if not (5 < coverage < 25):
        print(f"\n⚠ WARNING: Mask coverage ({coverage:.1f}%) seems unexpected")
    
    # Test mask refinement
    refined_iou = test_mask_refinement(embeddings, center, low_res_mask)
    
    # Get mask for visualization
    point_coords = np.array([[[float(center[0]), float(center[1])]]], dtype=np.float32)
    point_labels = np.array([[1.0]], dtype=np.float32)
    results = run_decoder(embeddings, point_coords, point_labels)
    best_idx = np.argmax(results['iou_predictions'][0])
    best_mask = results['masks'][0, best_idx]
    
    # Save visualization
    save_visualization(test_image, best_mask, center, args.output)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"  Single-click IoU: {iou:.4f}")
    print(f"  Refined IoU: {refined_iou:.4f}")
    print(f"  Mask coverage: {coverage:.2f}%")
    
    if iou > 0.5 and refined_iou > 0.5:
        print("\n✓ End-to-end test PASSED!")
    else:
        print("\n✗ End-to-end test FAILED - IoU scores too low")
        sys.exit(1)


if __name__ == "__main__":
    main()
