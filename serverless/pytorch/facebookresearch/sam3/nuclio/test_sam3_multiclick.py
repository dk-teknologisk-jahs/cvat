#!/usr/bin/env python3
"""
Comprehensive SAM3 Multi-Click Test Suite

Tests the PyTorch encoder + ONNX decoder pipeline with various scenarios:
1. Single positive click
2. Multiple positive clicks
3. Positive + negative clicks (distractor exclusion)
4. Mask refinement (iterative feedback)
5. Different mask selection indices

Requirements:
- SAM3 package installed
- ONNX decoder at: cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx
- model_handler_pytorch.py in same directory

Usage:
    python test_sam3_multiclick.py [--device cpu|cuda] [--save-viz]
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import onnxruntime as ort

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_handler_pytorch import ModelHandler


# ============================================================================
# Test Image Generators
# ============================================================================

def create_two_circles_image():
    """Create image with two circles (test distractor exclusion)."""
    img = Image.new('RGB', (1008, 1008), color='black')
    draw = ImageDraw.Draw(img)
    
    # Circle 1 (target)
    center1, radius1 = (300, 504), 150
    draw.ellipse([center1[0]-radius1, center1[1]-radius1, 
                  center1[0]+radius1, center1[1]+radius1], fill='white')
    
    # Circle 2 (distractor)
    center2, radius2 = (700, 504), 150
    draw.ellipse([center2[0]-radius2, center2[1]-radius2,
                  center2[0]+radius2, center2[1]+radius2], fill='white')
    
    # Ground truth mask (only circle 1)
    Y, X = np.ogrid[:1008, :1008]
    gt_mask = ((X - center1[0])**2 + (Y - center1[1])**2) <= radius1**2
    distractor_mask = ((X - center2[0])**2 + (Y - center2[1])**2) <= radius2**2
    
    return img, {
        'target_center': center1,
        'target_radius': radius1,
        'distractor_center': center2,
        'distractor_radius': radius2,
        'gt_mask': gt_mask,
        'distractor_mask': distractor_mask,
    }


def create_gradient_objects_image():
    """Create image with gradient-filled objects (more realistic)."""
    img = Image.new('RGB', (1008, 1008), color='black')
    
    # Background gradient
    for y in range(1008):
        c = int(30 + y * 0.03)
        for x in range(1008):
            img.putpixel((x, y), (c//3, c//3, c))
    
    # Object 1: Yellow-orange ellipse (target)
    for y in range(200, 700):
        for x in range(150, 550):
            dx = (x - 350) / 200
            dy = (y - 450) / 250
            if dx*dx + dy*dy < 1:
                intensity = 1 - 0.3 * (dx*dx + dy*dy)
                img.putpixel((x, y), (int(255*intensity), int(200*intensity), int(50*intensity)))
    
    # Object 2: Green ellipse (distractor)
    for y in range(300, 650):
        for x in range(550, 900):
            dx = (x - 725) / 175
            dy = (y - 475) / 175
            if dx*dx + dy*dy < 1:
                intensity = 1 - 0.3 * (dx*dx + dy*dy)
                img.putpixel((x, y), (int(50*intensity), int(200*intensity), int(50*intensity)))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    return img, {
        'target_center': (350, 450),
        'distractor_center': (725, 475),
        'distractor_region': (300, 650, 600, 850),  # y1, y2, x1, x2
    }


def create_single_circle_image():
    """Create simple image with single circle (basic test)."""
    img = Image.new('RGB', (1008, 1008), color='black')
    draw = ImageDraw.Draw(img)
    
    center, radius = (504, 504), 200
    draw.ellipse([center[0]-radius, center[1]-radius,
                  center[0]+radius, center[1]+radius], fill='white')
    
    Y, X = np.ogrid[:1008, :1008]
    gt_mask = ((X - center[0])**2 + (Y - center[1])**2) <= radius**2
    
    return img, {
        'center': center,
        'radius': radius,
        'gt_mask': gt_mask,
    }


# ============================================================================
# Test Functions
# ============================================================================

class SAM3Tester:
    """SAM3 test runner."""
    
    def __init__(self, device='cpu', onnx_path=None):
        print(f"Initializing SAM3 tester on {device}...")
        self.encoder = ModelHandler(device=device)
        
        if onnx_path is None:
            # Default path relative to this file
            script_dir = Path(__file__).parent
            onnx_path = script_dir.parent.parent.parent.parent.parent / \
                       'cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx'
        
        print(f"Loading ONNX decoder from {onnx_path}...")
        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        print("Initialization complete.\n")
    
    def decode(self, embeddings, point_coords, point_labels, mask_input=None, has_mask_input=0):
        """Run ONNX decoder."""
        if mask_input is None:
            mask_input = np.zeros((1, 1, 288, 288), dtype=np.float32)
        
        inputs = {
            'image_embed': embeddings['image_embed'],
            'high_res_feats_0': embeddings['high_res_feats_0'],
            'high_res_feats_1': embeddings['high_res_feats_1'],
            'point_coords': np.array([[point_coords]], dtype=np.float32).reshape(1, -1, 2),
            'point_labels': np.array([point_labels], dtype=np.float32).reshape(1, -1),
            'mask_input': mask_input,
            'has_mask_input': np.array([has_mask_input], dtype=np.float32),
        }
        
        outputs = self.session.run(None, inputs)
        
        return {
            'masks': outputs[0],          # [B, 3, H, W]
            'iou_predictions': outputs[1], # [B, 3]
            'low_res_masks': outputs[2],   # [B, 3, 288, 288]
            'object_score': outputs[3],    # [B, 1]
        }
    
    def get_best_mask(self, outputs):
        """Get best mask based on IoU predictions."""
        best_idx = np.argmax(outputs['iou_predictions'][0])
        return {
            'mask': outputs['masks'][0, best_idx] > 0,
            'low_res_mask': outputs['low_res_masks'][0, best_idx],
            'iou': outputs['iou_predictions'][0, best_idx],
            'idx': best_idx,
        }
    
    def compute_iou(self, pred_mask, gt_mask):
        """Compute IoU between predicted and ground truth masks."""
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        return intersection / union if union > 0 else 0


def run_tests(tester, save_viz=False):
    """Run all test cases."""
    results = []
    
    # ========================================================================
    # Test Suite 1: Single Circle (Basic Tests)
    # ========================================================================
    print("=" * 60)
    print("TEST SUITE 1: Single Circle")
    print("=" * 60)
    
    img1, meta1 = create_single_circle_image()
    emb1 = tester.encoder.encode(img1)
    
    # Test 1.1: Single positive click
    print("\n--- Test 1.1: Single positive click ---")
    out1_1 = tester.decode(emb1, [meta1['center']], [1])
    best1_1 = tester.get_best_mask(out1_1)
    iou1_1 = tester.compute_iou(best1_1['mask'], meta1['gt_mask'])
    print(f"  Predicted IoU: {best1_1['iou']:.4f}")
    print(f"  Ground truth IoU: {iou1_1:.4f}")
    results.append(('1.1 Single positive', iou1_1, iou1_1 > 0.9))
    
    # Test 1.2: Mask refinement
    print("\n--- Test 1.2: Mask refinement ---")
    mask_input = best1_1['low_res_mask'][np.newaxis, np.newaxis, :, :]
    out1_2 = tester.decode(emb1, [meta1['center']], [1], 
                           mask_input=mask_input.astype(np.float32), has_mask_input=1)
    best1_2 = tester.get_best_mask(out1_2)
    iou1_2 = tester.compute_iou(best1_2['mask'], meta1['gt_mask'])
    print(f"  Predicted IoU: {best1_2['iou']:.4f}")
    print(f"  Ground truth IoU: {iou1_2:.4f}")
    results.append(('1.2 Mask refinement', iou1_2, iou1_2 > 0.9))
    
    # ========================================================================
    # Test Suite 2: Two Circles (Distractor Exclusion)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST SUITE 2: Two Circles (Distractor Exclusion)")
    print("=" * 60)
    
    img2, meta2 = create_two_circles_image()
    emb2 = tester.encoder.encode(img2)
    
    # Test 2.1: Single positive on target
    print("\n--- Test 2.1: Single positive on target ---")
    out2_1 = tester.decode(emb2, [meta2['target_center']], [1])
    best2_1 = tester.get_best_mask(out2_1)
    iou2_1 = tester.compute_iou(best2_1['mask'], meta2['gt_mask'])
    dist_overlap = (best2_1['mask'] & meta2['distractor_mask']).sum() / meta2['distractor_mask'].sum() * 100
    print(f"  Predicted IoU: {best2_1['iou']:.4f}")
    print(f"  Ground truth IoU: {iou2_1:.4f}")
    print(f"  Distractor overlap: {dist_overlap:.1f}%")
    results.append(('2.1 Single positive (two circles)', iou2_1, iou2_1 > 0.9))
    
    # Test 2.2: Positive on target + Negative on distractor
    print("\n--- Test 2.2: Positive + Negative ---")
    points = [meta2['target_center'], meta2['distractor_center']]
    labels = [1, 0]  # positive, negative
    out2_2 = tester.decode(emb2, points, labels)
    best2_2 = tester.get_best_mask(out2_2)
    iou2_2 = tester.compute_iou(best2_2['mask'], meta2['gt_mask'])
    dist_overlap2 = (best2_2['mask'] & meta2['distractor_mask']).sum() / meta2['distractor_mask'].sum() * 100
    print(f"  Predicted IoU: {best2_2['iou']:.4f}")
    print(f"  Ground truth IoU: {iou2_2:.4f}")
    print(f"  Distractor overlap: {dist_overlap2:.1f}% (should be ~0%)")
    results.append(('2.2 Positive + Negative', iou2_2, iou2_2 > 0.9 and dist_overlap2 < 5))
    
    # Test 2.3: Multiple positive clicks
    print("\n--- Test 2.3: Multiple positive clicks ---")
    cx, cy = meta2['target_center']
    points = [(cx, cy), (cx-80, cy), (cx+80, cy), (cx, cy-80)]
    labels = [1, 1, 1, 1]
    out2_3 = tester.decode(emb2, points, labels)
    best2_3 = tester.get_best_mask(out2_3)
    iou2_3 = tester.compute_iou(best2_3['mask'], meta2['gt_mask'])
    print(f"  Predicted IoU: {best2_3['iou']:.4f}")
    print(f"  Ground truth IoU: {iou2_3:.4f}")
    results.append(('2.3 Multiple positives', iou2_3, iou2_3 > 0.9))
    
    # Test 2.4: Multiple positive + negative with mask refinement
    print("\n--- Test 2.4: Multi-click + mask refinement ---")
    mask_input = best2_2['low_res_mask'][np.newaxis, np.newaxis, :, :]
    out2_4 = tester.decode(emb2, [meta2['target_center'], meta2['distractor_center']], [1, 0],
                           mask_input=mask_input.astype(np.float32), has_mask_input=1)
    best2_4 = tester.get_best_mask(out2_4)
    iou2_4 = tester.compute_iou(best2_4['mask'], meta2['gt_mask'])
    print(f"  Predicted IoU: {best2_4['iou']:.4f}")
    print(f"  Ground truth IoU: {iou2_4:.4f}")
    results.append(('2.4 Multi-click + refinement', iou2_4, iou2_4 > 0.85))
    
    # ========================================================================
    # Test Suite 3: Gradient Objects (More Realistic)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST SUITE 3: Gradient Objects (Realistic)")
    print("=" * 60)
    
    img3, meta3 = create_gradient_objects_image()
    emb3 = tester.encoder.encode(img3)
    
    # Test 3.1: Single positive
    print("\n--- Test 3.1: Single positive ---")
    out3_1 = tester.decode(emb3, [meta3['target_center']], [1])
    best3_1 = tester.get_best_mask(out3_1)
    y1, y2, x1, x2 = meta3['distractor_region']
    dist_pixels3_1 = best3_1['mask'][y1:y2, x1:x2].sum()
    print(f"  Predicted IoU: {best3_1['iou']:.4f}")
    print(f"  Distractor region pixels: {dist_pixels3_1}")
    results.append(('3.1 Gradient single positive', best3_1['iou'] > 0.9, best3_1['iou'] > 0.9))
    
    # Test 3.2: Positive + Negative
    print("\n--- Test 3.2: Positive + Negative ---")
    out3_2 = tester.decode(emb3, [meta3['target_center'], meta3['distractor_center']], [1, 0])
    best3_2 = tester.get_best_mask(out3_2)
    dist_pixels3_2 = best3_2['mask'][y1:y2, x1:x2].sum()
    print(f"  Predicted IoU: {best3_2['iou']:.4f}")
    print(f"  Distractor region pixels: {dist_pixels3_2}")
    results.append(('3.2 Gradient pos + neg', best3_2['iou'] > 0.9, best3_2['iou'] > 0.9))
    
    # Test 3.3: Multiple clicks
    print("\n--- Test 3.3: Multiple clicks (3 positive + 1 negative) ---")
    points = [(250, 400), (350, 350), (450, 500), meta3['distractor_center']]
    labels = [1, 1, 1, 0]
    out3_3 = tester.decode(emb3, points, labels)
    best3_3 = tester.get_best_mask(out3_3)
    dist_pixels3_3 = best3_3['mask'][y1:y2, x1:x2].sum()
    print(f"  Predicted IoU: {best3_3['iou']:.4f}")
    print(f"  Distractor region pixels: {dist_pixels3_3}")
    results.append(('3.3 Gradient multi-click', best3_3['iou'] > 0.9, best3_3['iou'] > 0.9))
    
    # ========================================================================
    # Test Suite 4: Embedding Shape Verification
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST SUITE 4: Embedding Shape Verification")
    print("=" * 60)
    
    print("\n--- Test 4.1: Embedding shapes ---")
    shape_ok = True
    expected = {
        'high_res_feats_0': (1, 32, 288, 288),
        'high_res_feats_1': (1, 64, 144, 144),
        'image_embed': (1, 256, 72, 72),
    }
    for name, expected_shape in expected.items():
        actual = emb1[name].shape
        ok = actual == expected_shape
        shape_ok = shape_ok and ok
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {actual} (expected {expected_shape})")
    results.append(('4.1 Embedding shapes', shape_ok, shape_ok))
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, _, ok in results if ok)
    total = len(results)
    
    for name, value, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        if isinstance(value, float):
            print(f"  {status}: {name} (IoU={value:.4f})")
        else:
            print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if save_viz:
        # Save visualization of last test
        viz_path = Path(__file__).parent / 'test_output'
        viz_path.mkdir(exist_ok=True)
        
        # Save gradient test visualization
        viz = img3.copy().convert('RGBA')
        overlay = Image.new('RGBA', viz.size, (0, 0, 0, 0))
        for y in range(1008):
            for x in range(1008):
                if best3_3['mask'][y, x]:
                    overlay.putpixel((x, y), (0, 255, 0, 100))
        viz = Image.alpha_composite(viz, overlay).convert('RGB')
        draw = ImageDraw.Draw(viz)
        for (px, py), lbl in zip(points, labels):
            color = 'lime' if lbl == 1 else 'red'
            draw.ellipse([px-10, py-10, px+10, py+10], fill=color, outline='white', width=2)
        viz.save(viz_path / 'gradient_multiclick.png')
        print(f"\nSaved visualization to {viz_path / 'gradient_multiclick.png'}")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description='SAM3 Multi-Click Test Suite')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run on')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--onnx-path', type=str, default=None,
                        help='Path to ONNX decoder model')
    args = parser.parse_args()
    
    tester = SAM3Tester(device=args.device, onnx_path=args.onnx_path)
    success = run_tests(tester, save_viz=args.save_viz)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
