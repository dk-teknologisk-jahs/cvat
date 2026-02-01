#!/usr/bin/env python3
"""
Compare ONNX vs PyTorch decoder outputs with identical inputs.
"""

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image, ImageDraw

# Constants
SAM3_IMAGE_SIZE = 1008


def create_test_image():
    """Create a test image with a simple circle."""
    size = SAM3_IMAGE_SIZE
    bg_color = (240, 240, 240)
    fg_color = (50, 100, 200)
    
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    center = (504, 504)
    radius = 200
    bbox = [center[0] - radius, center[1] - radius, 
            center[0] + radius, center[1] + radius]
    draw.ellipse(bbox, fill=fg_color)
    
    return img, center


def preprocess_image(image: Image.Image, device='cpu'):
    """Preprocess image for SAM3."""
    img_resized = image.resize((SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor.to(device)


def main():
    print("=" * 60)
    print("Compare ONNX vs PyTorch Decoder")
    print("=" * 60)
    
    # Create test image
    test_image, (cx, cy) = create_test_image()
    
    # Load full SAM3 model
    print("\nLoading SAM3-Tracker...")
    from sam3.model_builder import build_tracker
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    tracker = tracker.to('cpu').eval()
    
    # Get embeddings using PyTorch encoder
    print("\nEncoding image with PyTorch...")
    img_tensor = preprocess_image(test_image, 'cpu')
    
    with torch.no_grad():
        backbone_out = tracker.forward_image(img_tensor)
        _, vision_feats, _, _ = tracker._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + tracker.no_mem_embed
        
        feat_sizes = [(288, 288), (144, 144), (72, 72)]
        reshaped_feats = []
        for feat, size in zip(vision_feats[::-1], feat_sizes[::-1]):
            B = feat.shape[1]
            feat_reshaped = feat.permute(1, 2, 0).view(B, -1, *size)
            reshaped_feats.append(feat_reshaped)
        reshaped_feats = reshaped_feats[::-1]
        
        high_res_feats_0 = reshaped_feats[0]
        high_res_feats_1 = reshaped_feats[1]
        image_embed = reshaped_feats[2]
    
    print(f"  high_res_feats_0: {high_res_feats_0.shape}")
    print(f"  high_res_feats_1: {high_res_feats_1.shape}")
    print(f"  image_embed: {image_embed.shape}")
    
    # Create point prompt
    point_coords = torch.tensor([[[float(cx), float(cy)]]], dtype=torch.float32)
    point_labels = torch.tensor([[1.0]], dtype=torch.float32)
    mask_input = torch.zeros(1, 1, 288, 288, dtype=torch.float32)
    has_mask_input = torch.tensor([0.0], dtype=torch.float32)
    
    # Run PyTorch decoder
    print("\n" + "=" * 60)
    print("PYTORCH DECODER")
    print("=" * 60)
    
    with torch.no_grad():
        sparse_embeddings, _ = tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        no_mask_embed = tracker.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        dense_embeddings = no_mask_embed.expand(1, -1, 72, 72)
        
        image_pe = tracker.sam_prompt_encoder.get_dense_pe()
        
        high_res_features = [high_res_feats_0, high_res_feats_1]
        
        pt_low_res_masks, pt_iou_pred, _, pt_obj_score = tracker.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        pt_masks = F.interpolate(
            pt_low_res_masks.float(),
            size=(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    
    print(f"  iou_pred: {pt_iou_pred}")
    print(f"  obj_score: {pt_obj_score}")
    print(f"  low_res_masks stats: min={pt_low_res_masks.min():.4f}, max={pt_low_res_masks.max():.4f}")
    print(f"  masks stats: min={pt_masks.min():.4f}, max={pt_masks.max():.4f}")
    
    best_idx_pt = pt_iou_pred[0].argmax().item()
    pt_coverage = (pt_masks[0, best_idx_pt] > 0).float().mean().item() * 100
    print(f"  Best mask index: {best_idx_pt}, IoU: {pt_iou_pred[0, best_idx_pt]:.4f}")
    print(f"  Mask coverage: {pt_coverage:.2f}%")
    
    # Run ONNX decoder
    print("\n" + "=" * 60)
    print("ONNX DECODER")
    print("=" * 60)
    
    decoder_path = "/home/jahs/GitHub/cvat/cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx"
    decoder_session = ort.InferenceSession(decoder_path)
    
    onnx_inputs = {
        'image_embed': image_embed.numpy(),
        'high_res_feats_0': high_res_feats_0.numpy(),
        'high_res_feats_1': high_res_feats_1.numpy(),
        'point_coords': point_coords.numpy(),
        'point_labels': point_labels.numpy(),
        'mask_input': mask_input.numpy(),
        'has_mask_input': has_mask_input.numpy(),
    }
    
    onnx_outputs = decoder_session.run(None, onnx_inputs)
    onnx_masks = onnx_outputs[0]
    onnx_iou = onnx_outputs[1]
    onnx_low_res = onnx_outputs[2]
    onnx_obj_score = onnx_outputs[3]
    
    print(f"  iou_pred: {onnx_iou}")
    print(f"  obj_score: {onnx_obj_score}")
    print(f"  low_res_masks stats: min={onnx_low_res.min():.4f}, max={onnx_low_res.max():.4f}")
    print(f"  masks stats: min={onnx_masks.min():.4f}, max={onnx_masks.max():.4f}")
    
    best_idx_onnx = np.argmax(onnx_iou[0])
    onnx_coverage = (onnx_masks[0, best_idx_onnx] > 0).mean() * 100
    print(f"  Best mask index: {best_idx_onnx}, IoU: {onnx_iou[0, best_idx_onnx]:.4f}")
    print(f"  Mask coverage: {onnx_coverage:.2f}%")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    # Compare IoU predictions
    iou_diff = np.abs(pt_iou_pred.numpy() - onnx_iou)
    print(f"  IoU difference: {iou_diff}")
    
    # Compare low-res masks
    low_res_diff = np.abs(pt_low_res_masks.numpy() - onnx_low_res)
    print(f"  Low-res mask diff: mean={low_res_diff.mean():.6f}, max={low_res_diff.max():.6f}")
    
    # Compare final masks
    mask_diff = np.abs(pt_masks.numpy() - onnx_masks)
    print(f"  Mask diff: mean={mask_diff.mean():.6f}, max={mask_diff.max():.6f}")
    
    if mask_diff.max() < 0.001:
        print("\n✓ ONNX decoder matches PyTorch decoder!")
    else:
        print("\n⚠ Significant difference between ONNX and PyTorch outputs")


if __name__ == "__main__":
    main()
