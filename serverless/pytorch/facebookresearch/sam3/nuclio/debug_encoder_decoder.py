#!/usr/bin/env python3
"""
Debug SAM3 Encoder/Decoder Mismatch

Compares PyTorch encoder outputs against expected ONNX decoder behavior.
"""

import numpy as np
import torch
import torch.nn.functional as F
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
    print("SAM3 Encoder/Decoder Debug")
    print("=" * 60)
    
    # Create test image
    test_image, (cx, cy) = create_test_image()
    print(f"\nTest image: circle at ({cx}, {cy})")
    
    # Load full SAM3 model
    print("\nLoading SAM3-Tracker with backbone...")
    from sam3.model_builder import build_tracker
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    tracker = tracker.to('cpu').eval()
    
    # Preprocess image
    img_tensor = preprocess_image(test_image, 'cpu')
    print(f"Input tensor shape: {img_tensor.shape}")
    
    with torch.no_grad():
        # Get backbone features using forward_image
        print("\n1. Running forward_image...")
        backbone_out = tracker.forward_image(img_tensor)
        print(f"   backbone_out keys: {list(backbone_out.keys())}")
        
        # Check vision_features shape
        if 'vision_features' in backbone_out:
            vf = backbone_out['vision_features']
            print(f"   vision_features shape: {vf.shape}")
        
        if 'backbone_fpn' in backbone_out:
            fpn = backbone_out['backbone_fpn']
            print(f"   backbone_fpn: {[f.shape for f in fpn]}")
            
        # Prepare features
        print("\n2. Running _prepare_backbone_features...")
        _, vision_feats, _, _ = tracker._prepare_backbone_features(backbone_out)
        print(f"   vision_feats shapes (low to high res):")
        for i, feat in enumerate(vision_feats):
            print(f"     [{i}]: {feat.shape}")
        
        # Add no_mem_embed
        print("\n3. Adding no_mem_embed...")
        vision_feats[-1] = vision_feats[-1] + tracker.no_mem_embed
        print(f"   no_mem_embed shape: {tracker.no_mem_embed.shape}")
        
        # Reshape features
        print("\n4. Reshaping features...")
        feat_sizes = [(288, 288), (144, 144), (72, 72)]
        reshaped_feats = []
        for feat, size in zip(vision_feats[::-1], feat_sizes[::-1]):
            # feat: [HW, B, C] -> [B, C, H, W]
            B = feat.shape[1]
            feat_reshaped = feat.permute(1, 2, 0).view(B, -1, *size)
            reshaped_feats.append(feat_reshaped)
        reshaped_feats = reshaped_feats[::-1]
        
        high_res_feats_0 = reshaped_feats[0]  # [1, 32, 288, 288]
        high_res_feats_1 = reshaped_feats[1]  # [1, 64, 144, 144]
        image_embed = reshaped_feats[2]        # [1, 256, 72, 72]
        
        print(f"   high_res_feats_0: {high_res_feats_0.shape}")
        print(f"   high_res_feats_1: {high_res_feats_1.shape}")
        print(f"   image_embed: {image_embed.shape}")
        
        # Now run decoder through PyTorch (not ONNX)
        print("\n5. Running SAM3 prompt encoder + mask decoder (PyTorch)...")
        
        # Create point prompt
        point_coords = torch.tensor([[[float(cx), float(cy)]]], dtype=torch.float32)
        point_labels = torch.tensor([[1.0]], dtype=torch.float32)
        
        # Get prompt embeddings
        sparse_embeddings, _ = tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        print(f"   sparse_embeddings: {sparse_embeddings.shape}")
        
        # Get dense embeddings (no mask input)
        no_mask_embed = tracker.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        dense_embeddings = no_mask_embed.expand(1, -1, 72, 72)
        print(f"   dense_embeddings: {dense_embeddings.shape}")
        
        # Get positional encoding
        image_pe = tracker.sam_prompt_encoder.get_dense_pe()
        print(f"   image_pe: {image_pe.shape}")
        
        # Prepare high-res features
        high_res_features = [high_res_feats_0, high_res_feats_1]
        
        # Run mask decoder
        print("\n6. Running mask decoder...")
        low_res_masks, iou_pred, _, obj_score = tracker.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        print(f"   low_res_masks: {low_res_masks.shape}")
        print(f"   iou_pred: {iou_pred}")
        print(f"   obj_score: {obj_score}")
        
        # Upsample masks
        masks = F.interpolate(
            low_res_masks.float(),
            size=(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        print(f"   masks (upsampled): {masks.shape}")
        
        # Analyze best mask
        best_idx = iou_pred[0].argmax().item()
        best_mask = masks[0, best_idx]
        best_iou = iou_pred[0, best_idx].item()
        
        print(f"\n7. Analysis:")
        print(f"   Best mask index: {best_idx}")
        print(f"   IoU score: {best_iou:.4f}")
        
        binary_mask = (best_mask > 0).float()
        mask_area = binary_mask.sum().item()
        total_pixels = binary_mask.numel()
        coverage = mask_area / total_pixels * 100
        
        print(f"   Mask coverage: {coverage:.1f}%")
        print(f"   Expected: ~12.4%")
        
        # Save visualization
        print("\n8. Saving visualization...")
        vis_img = np.array(test_image).copy()
        mask_np = (best_mask.numpy() > 0).astype(np.uint8)
        
        for c in range(3):
            vis_img[:, :, c] = np.where(
                mask_np,
                (vis_img[:, :, c] * 0.5 + (255 if c == 0 else 0) * 0.5).astype(np.uint8),
                vis_img[:, :, c]
            )
        
        vis_pil = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(vis_pil)
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(0, 255, 0))
        
        vis_pil.save("/tmp/sam3_pytorch_debug.png")
        print(f"   Saved to: /tmp/sam3_pytorch_debug.png")


if __name__ == "__main__":
    main()
