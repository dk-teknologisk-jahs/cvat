#!/usr/bin/env python3
"""
SAM3 ONNX Export Script

Exports SAM3-Tracker models to ONNX format:
1. Vision Encoder: For server-side image embedding computation
2. Decoder (SAM2-compatible): Browser-side prompt encoder + mask decoder with:
   - Dynamic output resolution via orig_im_size input
   - Mask refinement support (mask_input + has_mask_input)
   - Bounding box computation in-model
   - Proper upsampling matching PyTorch F.interpolate(align_corners=False)

Architecture (SAM3-Tracker):
- Image size: 1008×1008 (vs SAM2's 1024×1024)
- Backbone stride: 14 (vs SAM2's 16)
- Embedding size: 72×72 (1008/14 = 72)
- Low-res mask size: 288×288 (72×4)
- High-res feature levels: 3 (288×288, 144×144, 72×72)

Usage:
    # Export vision encoder
    python export_sam3_onnx.py --export encoder --output vision_encoder.onnx

    # Export decoder (SAM2-compatible with mask refinement)
    python export_sam3_onnx.py --export decoder --output decoder.onnx

    # Export both
    python export_sam3_onnx.py --export both --output-dir ./onnx_models/

    # Verify exported models
    python export_sam3_onnx.py --verify --encoder-path vision_encoder.onnx --decoder-path decoder.onnx
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SAM3 Constants
# ============================================================================
SAM3_IMAGE_SIZE = 1008
SAM3_BACKBONE_STRIDE = 14
SAM3_EMBED_SIZE = SAM3_IMAGE_SIZE // SAM3_BACKBONE_STRIDE  # 72
SAM3_MASK_INPUT_SIZE = 4 * SAM3_EMBED_SIZE  # 288
SAM3_EMBED_DIM = 256

# Feature sizes (in order: high-res to low-res)
SAM3_FEAT_SIZES = [
    (288, 288),  # High-res feat 0: 32 channels
    (144, 144),  # High-res feat 1: 64 channels
    (72, 72),    # Main embedding: 256 channels
]


# ============================================================================
# Vision Encoder Wrapper
# ============================================================================
class SAM3VisionEncoderWrapper(nn.Module):
    """
    Wrapper for SAM3-Tracker vision encoder for ONNX export.
    
    Takes an image and produces three feature maps:
    - high_res_feats_0: [B, 32, 288, 288]
    - high_res_feats_1: [B, 64, 144, 144]  
    - image_embed: [B, 256, 72, 72]
    
    This wrapper follows the same encoding path as SAM3InteractiveImagePredictor.set_image(),
    using forward_image() and _prepare_backbone_features() from the tracker model.
    """
    
    def __init__(self, tracker_model: nn.Module):
        super().__init__()
        # We need the full tracker model because:
        # - backbone.forward_image() returns the feature pyramid
        # - sam_mask_decoder.conv_s0/conv_s1 project to correct channel sizes
        # - no_mem_embed is added to the lowest-res features
        self.tracker = tracker_model
        self.image_size = SAM3_IMAGE_SIZE
        
        # Spatial sizes for feature reshaping (from SAM3InteractiveImagePredictor)
        self._bb_feat_sizes = SAM3_FEAT_SIZES
        
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to SAM3 feature embeddings.
        
        This matches the encoding path in SAM3InteractiveImagePredictor.set_image():
        1. Run backbone via forward_image() 
        2. Prepare features via _prepare_backbone_features()
        3. Add no_mem_embed to lowest-res features
        4. Reshape to [B, C, H, W] format
        
        Args:
            images: [B, 3, 1008, 1008] normalized images (mean=0.5, std=0.5)
            
        Returns:
            high_res_feats_0: [B, 32, 288, 288]
            high_res_feats_1: [B, 64, 144, 144]
            image_embed: [B, 256, 72, 72]
        """
        B = images.shape[0]
        
        # Use tracker's forward_image which applies backbone + conv_s0/conv_s1 projections
        backbone_out = self.tracker.forward_image(images)
        
        # Prepare features (this flattens to [HW, B, C] format)
        _, vision_feats, _, _ = self.tracker._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed to lowest resolution features (tells model no video memory)
        # vision_feats is list of 3 levels, lowest-res is last (-1)
        vision_feats[-1] = vision_feats[-1] + self.tracker.no_mem_embed
        
        # Reshape features: [HW, B, C] -> [B, C, H, W]
        # vision_feats is in low-res to high-res order, we need high-res to low-res
        feats = []
        for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1]):
            # feat shape: [HW, B, C] -> [B, C, H, W]
            feat = feat.permute(1, 2, 0).view(B, -1, *feat_size)
            feats.append(feat)
        feats = feats[::-1]  # Back to high-res to low-res order
        
        # feats[0]: [B, 32, 288, 288] - high_res_feats_0 (already projected by conv_s0)
        # feats[1]: [B, 64, 144, 144] - high_res_feats_1 (already projected by conv_s1)
        # feats[2]: [B, 256, 72, 72] - image_embed (main backbone features)
        
        high_res_feats_0 = feats[0]
        high_res_feats_1 = feats[1]
        image_embed = feats[2]
        
        return high_res_feats_0, high_res_feats_1, image_embed


# ============================================================================
# SAM2-Compatible Decoder Wrapper
# ============================================================================
class SAM3DecoderSAM2Compatible(nn.Module):
    """
    SAM3-Tracker decoder wrapper with SAM2-compatible interface.
    
    Combines prompt encoder + mask decoder with:
    - Dynamic output resolution via orig_im_size input
    - Mask refinement support (mask_input + has_mask_input)
    - Bounding box computation
    - In-model bilinear upsampling (matching PyTorch align_corners=False)
    
    Inputs:
        image_embed: [B, 256, 72, 72] - main backbone embedding
        high_res_feats_0: [B, 32, 288, 288] - level 0 high-res features
        high_res_feats_1: [B, 64, 144, 144] - level 1 high-res features
        point_coords: [B, N, 2] - click/box coordinates in 1008x1008 space
        point_labels: [B, N] - point type labels (0=neg, 1=pos, -1=pad, 2=box TL, 3=box BR)
        mask_input: [B, 1, 288, 288] - previous low-res mask logits
        has_mask_input: [B] - whether mask_input is valid (1.0 or 0.0)
        orig_im_size: [2] - original image [H, W] for output upsampling
        
    Outputs:
        masks: [B, 1, orig_H, orig_W] - upsampled mask at original resolution
        iou_predictions: [B, 1] - IoU quality score
        low_res_masks: [B, 1, 288, 288] - for next iteration refinement
        xtl, ytl, xbr, ybr: scalar - bounding box coordinates (normalized 0-1)
    """
    
    def __init__(
        self,
        sam_prompt_encoder: nn.Module,
        sam_mask_decoder: nn.Module,
        multimask_output: bool = True,
        use_stability_selection: bool = True,
    ):
        super().__init__()
        self.sam_prompt_encoder = sam_prompt_encoder
        self.sam_mask_decoder = sam_mask_decoder
        self.multimask_output = multimask_output
        self.use_stability_selection = use_stability_selection
        
        # SAM3 constants
        self.image_size = SAM3_IMAGE_SIZE
        self.image_embedding_size = SAM3_EMBED_SIZE
        self.mask_input_size = (SAM3_MASK_INPUT_SIZE, SAM3_MASK_INPUT_SIZE)
        
        # Stability selection thresholds (from SAM3 official config)
        self.stability_delta = 0.05
        self.stability_thresh = 0.98
        
    def _compute_stability(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute mask stability scores.
        
        Stability = area(logits > delta) / area(logits > -delta)
        Higher stability means more confident mask predictions.
        
        Args:
            masks: [B, num_masks, H, W] mask logits
            
        Returns:
            stability: [B, num_masks] stability scores
        """
        area_inner = (masks > self.stability_delta).float().sum(dim=(-2, -1))
        area_outer = (masks > -self.stability_delta).float().sum(dim=(-2, -1))
        stability = area_inner / (area_outer + 1e-8)
        return stability
        
    def _select_best_mask(
        self,
        masks: torch.Tensor,
        iou_pred: torch.Tensor,
        num_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamic mask selection matching SAM3's _dynamic_multimask_via_stability.
        
        Logic:
        - For SINGLE prompt (ambiguous): select best IoU from multi-mask outputs (masks 1+)
        - For MULTIPLE prompts (non-ambiguous): use mask 0 if stable, else best multi-mask
        
        Args:
            masks: [B, 4, H, W] all mask outputs (mask 0 + 3 multi-masks)
            iou_pred: [B, 4] IoU predictions
            num_points: [B] number of non-padding points per sample
            
        Returns:
            selected_mask: [B, 1, H, W]
            selected_iou: [B, 1]
        """
        B = masks.shape[0]
        
        # Compute stability for mask 0 (single-object token)
        stability = self._compute_stability(masks[:, 0:1, :, :])  # [B, 1]
        is_stable = stability[:, 0] >= self.stability_thresh  # [B]
        
        # Find best multi-mask (masks 1, 2, 3) by IoU
        multi_iou = iou_pred[:, 1:]  # [B, 3]
        best_multi_idx = multi_iou.argmax(dim=1)  # [B]
        
        # For single prompt: always use best multi-mask
        # For multiple prompts: use mask 0 if stable, else best multi-mask
        is_single_prompt = (num_points == 1)
        use_mask_0 = (~is_single_prompt) & is_stable
        
        # Select indices: 0 for mask_0, or 1+best_multi_idx for multi-mask
        selected_idx = torch.where(use_mask_0, torch.zeros_like(best_multi_idx), 1 + best_multi_idx)
        
        # Gather selected masks and IoUs
        batch_idx = torch.arange(B, device=masks.device)
        selected_mask = masks[batch_idx, selected_idx].unsqueeze(1)  # [B, 1, H, W]
        selected_iou = iou_pred[batch_idx, selected_idx].unsqueeze(1)  # [B, 1]
        
        return selected_mask, selected_iou
        
    def _compute_bbox(self, mask: torch.Tensor, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bounding box from mask logits.
        
        Args:
            mask: [B, 1, H, W] mask logits
            threshold: logit threshold for positive pixels
            
        Returns:
            xtl, ytl, xbr, ybr: [B] normalized coordinates (0-1)
        """
        B, _, H, W = mask.shape
        binary = (mask > threshold).float()  # [B, 1, H, W]
        
        # Find positive pixels
        binary_flat = binary.view(B, -1)  # [B, H*W]
        has_positive = binary_flat.sum(dim=1) > 0  # [B]
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=mask.device).float() / H
        x_coords = torch.arange(W, device=mask.device).float() / W
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand for batch
        xx = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        
        # Masked coordinates
        xx_masked = torch.where(binary > 0, xx, torch.ones_like(xx))
        yy_masked = torch.where(binary > 0, yy, torch.ones_like(yy))
        xx_masked_max = torch.where(binary > 0, xx, torch.zeros_like(xx))
        yy_masked_max = torch.where(binary > 0, yy, torch.zeros_like(yy))
        
        # Compute min/max
        xtl = xx_masked.view(B, -1).min(dim=1)[0]
        ytl = yy_masked.view(B, -1).min(dim=1)[0]
        xbr = xx_masked_max.view(B, -1).max(dim=1)[0]
        ybr = yy_masked_max.view(B, -1).max(dim=1)[0]
        
        # Handle empty masks
        xtl = torch.where(has_positive, xtl, torch.zeros_like(xtl))
        ytl = torch.where(has_positive, ytl, torch.zeros_like(ytl))
        xbr = torch.where(has_positive, xbr, torch.ones_like(xbr))
        ybr = torch.where(has_positive, ybr, torch.ones_like(ybr))
        
        return xtl, ytl, xbr, ybr
        
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass combining prompt encoder and mask decoder.
        
        Returns:
            masks: [B, 1, orig_H, orig_W] - upsampled to original resolution
            iou_predictions: [B, 1] - IoU quality score
            low_res_masks: [B, 1, 288, 288] - for iterative refinement
            xtl, ytl, xbr, ybr: scalar - bounding box (normalized 0-1)
        """
        B = image_embed.shape[0]
        
        # Ensure mask_input is correct size
        if mask_input.shape[-2:] != self.mask_input_size:
            mask_input = F.interpolate(
                mask_input.float(),
                size=self.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
            
        # Clamp mask_input to [-32, 32] (official SAM3 behavior)
        mask_input = torch.clamp(mask_input, -32.0, 32.0)
        
        # Get sparse embeddings from points
        sparse_embeddings, _ = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # Get dense embeddings based on has_mask_input
        dense_embeddings_from_mask = self.sam_prompt_encoder._embed_masks(mask_input)
        no_mask_embed = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        no_mask_embed = no_mask_embed.expand(B, -1, self.image_embedding_size, self.image_embedding_size)
        
        # Conditionally select dense embeddings
        has_mask = has_mask_input.view(-1, 1, 1, 1).float()
        dense_embeddings = has_mask * dense_embeddings_from_mask + (1 - has_mask) * no_mask_embed
        
        # Get positional encoding
        image_pe = self.sam_prompt_encoder.get_dense_pe()
        
        # Prepare high resolution features
        high_res_features = [high_res_feats_0, high_res_feats_1]
        
        # Run mask decoder
        # Returns: (masks, iou_pred, sam_tokens, object_score_logits)
        low_res_multimasks, iou_pred, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,  # Always get all masks for selection
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # low_res_multimasks shape: [B, 4, 288, 288] (1 single-object + 3 multi-object)
        # iou_pred shape: [B, 4]
        
        # Count non-padding points for stability selection
        num_points = (point_labels >= 0).sum(dim=1).float()  # [B]
        
        if self.use_stability_selection:
            # Use dynamic mask selection
            selected_mask, selected_iou = self._select_best_mask(
                low_res_multimasks, iou_pred, num_points
            )
        else:
            # Simple selection: mask 0 for multimask=False
            selected_mask = low_res_multimasks[:, 0:1, :, :]
            selected_iou = iou_pred[:, 0:1]
            
        # Compute bounding box from low-res mask (faster)
        xtl, ytl, xbr, ybr = self._compute_bbox(selected_mask)
        
        # Upsample to original image size
        # orig_im_size is [H, W] tensor
        orig_h = orig_im_size[0].int().item()
        orig_w = orig_im_size[1].int().item()
        
        masks = F.interpolate(
            selected_mask.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        
        return masks, selected_iou, selected_mask, xtl, ytl, xbr, ybr


# ============================================================================
# Simplified Decoder Wrapper (for ONNX export without dynamic shapes)
# ============================================================================
class SAM3DecoderSimplified(nn.Module):
    """
    Simplified SAM3 decoder for ONNX export.
    
    Outputs masks at fixed 1008x1008 resolution (caller handles final resize).
    This avoids ONNX dynamic shape issues while keeping mask refinement.
    
    Inputs:
        image_embed: [B, 256, 72, 72]
        high_res_feats_0: [B, 32, 288, 288]
        high_res_feats_1: [B, 64, 144, 144]
        point_coords: [B, N, 2] in 1008x1008 space
        point_labels: [B, N] (0=neg, 1=pos, -1=pad, 2=box TL, 3=box BR)
        mask_input: [B, 1, 288, 288]
        has_mask_input: [B]
        
    Outputs:
        masks: [B, 3, 1008, 1008] - all multi-mask outputs
        iou_predictions: [B, 3] - IoU for each mask
        low_res_masks: [B, 3, 288, 288] - for refinement
        object_score_logits: [B, 1]
    """
    
    def __init__(
        self,
        sam_prompt_encoder: nn.Module,
        sam_mask_decoder: nn.Module,
    ):
        super().__init__()
        self.sam_prompt_encoder = sam_prompt_encoder
        self.sam_mask_decoder = sam_mask_decoder
        
        self.image_size = SAM3_IMAGE_SIZE
        self.image_embedding_size = SAM3_EMBED_SIZE
        self.mask_input_size = (SAM3_MASK_INPUT_SIZE, SAM3_MASK_INPUT_SIZE)
        
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        B = image_embed.shape[0]
        
        # Ensure mask_input is correct size
        if mask_input.shape[-2:] != self.mask_input_size:
            mask_input = F.interpolate(
                mask_input.float(),
                size=self.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
            
        # Clamp mask_input
        mask_input = torch.clamp(mask_input, -32.0, 32.0)
        
        # Get sparse embeddings from points
        sparse_embeddings, _ = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # Get dense embeddings based on has_mask_input
        dense_embeddings_from_mask = self.sam_prompt_encoder._embed_masks(mask_input)
        no_mask_embed = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        no_mask_embed = no_mask_embed.expand(B, -1, self.image_embedding_size, self.image_embedding_size)
        
        # Conditionally select dense embeddings
        has_mask = has_mask_input.view(-1, 1, 1, 1).float()
        dense_embeddings = has_mask * dense_embeddings_from_mask + (1 - has_mask) * no_mask_embed
        
        # Get positional encoding
        image_pe = self.sam_prompt_encoder.get_dense_pe()
        
        # Prepare high resolution features
        high_res_features = [high_res_feats_0, high_res_feats_1]
        
        # Run mask decoder with multimask=True
        # Note: When multimask_output=True, the decoder already returns only the 
        # multi-mask outputs (indices 1, 2, 3 from the 4 mask tokens), giving [B, 3, H, W]
        low_res_multimasks, iou_pred, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # low_res_multimasks already has shape [B, 3, 288, 288] when multimask_output=True
        # (the decoder internally selects masks[:, 1:, :, :] from the 4 mask tokens)
        low_res_masks = low_res_multimasks
        iou_predictions = iou_pred
        
        # Upsample to 1008x1008
        masks = F.interpolate(
            low_res_masks.float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        return masks, iou_predictions, low_res_masks, object_score_logits


# ============================================================================
# Export Functions
# ============================================================================
def export_vision_encoder(
    tracker_model: nn.Module,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """
    Export SAM3 vision encoder to ONNX.
    
    Args:
        tracker_model: SAM3 tracker model (Sam3TrackerPredictor)
        output_path: Output ONNX file path
        opset_version: ONNX opset version
        device: Device for export
    """
    print(f"\n{'='*60}")
    print("Exporting SAM3 Vision Encoder")
    print(f"{'='*60}")
    
    wrapper = SAM3VisionEncoderWrapper(tracker_model).to(device).eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, device=device)
    
    # Input/output names
    input_names = ["images"]
    output_names = ["high_res_feats_0", "high_res_feats_1", "image_embed"]
    
    # Dynamic axes for batch size
    dynamic_axes = {
        "images": {0: "batch_size"},
        "high_res_feats_0": {0: "batch_size"},
        "high_res_feats_1": {0: "batch_size"},
        "image_embed": {0: "batch_size"},
    }
    
    print(f"Input shape: {list(dummy_input.shape)}")
    print(f"Output names: {output_names}")
    print(f"Opset version: {opset_version}")
    
    with torch.no_grad():
        # Test forward pass
        out0, out1, out2 = wrapper(dummy_input)
        print(f"\nOutput shapes:")
        print(f"  high_res_feats_0: {list(out0.shape)}")
        print(f"  high_res_feats_1: {list(out1.shape)}")
        print(f"  image_embed: {list(out2.shape)}")
        
        # Export
        print(f"\nExporting to {output_path}...")
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        
    print(f"✓ Vision encoder exported to {output_path}")
    _verify_onnx(output_path)


def export_decoder(
    tracker_model: nn.Module,
    output_path: str,
    simplified: bool = True,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """
    Export SAM3 decoder to ONNX.
    
    Args:
        tracker_model: SAM3 tracker model (Sam3TrackerPredictor)
        output_path: Output ONNX file path
        simplified: Use simplified decoder (fixed output size) for better ONNX compatibility
        opset_version: ONNX opset version
        device: Device for export
    """
    print(f"\n{'='*60}")
    print(f"Exporting SAM3 Decoder ({'simplified' if simplified else 'SAM2-compatible'})")
    print(f"{'='*60}")
    
    if simplified:
        wrapper = SAM3DecoderSimplified(
            sam_prompt_encoder=tracker_model.sam_prompt_encoder,
            sam_mask_decoder=tracker_model.sam_mask_decoder,
        ).to(device).eval()
    else:
        wrapper = SAM3DecoderSAM2Compatible(
            sam_prompt_encoder=tracker_model.sam_prompt_encoder,
            sam_mask_decoder=tracker_model.sam_mask_decoder,
            multimask_output=True,
            use_stability_selection=True,
        ).to(device).eval()
    
    # Dummy inputs
    batch_size = 1
    num_points = 2
    
    dummy_inputs = {
        "image_embed": torch.randn(batch_size, 256, 72, 72, device=device),
        "high_res_feats_0": torch.randn(batch_size, 32, 288, 288, device=device),
        "high_res_feats_1": torch.randn(batch_size, 64, 144, 144, device=device),
        "point_coords": torch.randint(0, 1008, (batch_size, num_points, 2), dtype=torch.float32, device=device),
        "point_labels": torch.ones(batch_size, num_points, dtype=torch.float32, device=device),
        "mask_input": torch.zeros(batch_size, 1, 288, 288, device=device),
        "has_mask_input": torch.zeros(batch_size, dtype=torch.float32, device=device),
    }
    
    if not simplified:
        dummy_inputs["orig_im_size"] = torch.tensor([1008, 1008], dtype=torch.int64, device=device)
    
    input_names = list(dummy_inputs.keys())
    
    if simplified:
        output_names = ["masks", "iou_predictions", "low_res_masks", "object_score_logits"]
    else:
        output_names = ["masks", "iou_predictions", "low_res_masks", "xtl", "ytl", "xbr", "ybr"]
    
    # Dynamic axes
    dynamic_axes = {
        "image_embed": {0: "batch_size"},
        "high_res_feats_0": {0: "batch_size"},
        "high_res_feats_1": {0: "batch_size"},
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        "mask_input": {0: "batch_size"},
        "has_mask_input": {0: "batch_size"},
        "masks": {0: "batch_size"},
        "iou_predictions": {0: "batch_size"},
        "low_res_masks": {0: "batch_size"},
    }
    
    print("Input shapes:")
    for name, tensor in dummy_inputs.items():
        print(f"  {name}: {list(tensor.shape)}")
    print(f"Output names: {output_names}")
    print(f"Opset version: {opset_version}")
    
    with torch.no_grad():
        # Test forward pass
        outputs = wrapper(*dummy_inputs.values())
        print(f"\nOutput shapes:")
        for i, name in enumerate(output_names):
            if isinstance(outputs[i], torch.Tensor):
                print(f"  {name}: {list(outputs[i].shape)}")
            else:
                print(f"  {name}: {outputs[i]}")
        
        # Export
        print(f"\nExporting to {output_path}...")
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        
    print(f"✓ Decoder exported to {output_path}")
    _verify_onnx(output_path)


def _verify_onnx(path: str) -> None:
    """Verify ONNX model and print info."""
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print("✓ ONNX model validation passed")
        
        print("\nModel inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")
            
        print("\nModel outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")
            
        # File size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\nFile size: {size_mb:.1f} MB")
        
    except ImportError:
        print("Note: Install onnx package to verify the exported model")


def verify_models(encoder_path: str, decoder_path: str) -> None:
    """Verify exported models with onnxruntime."""
    import onnxruntime as ort
    
    print(f"\n{'='*60}")
    print("Verifying Exported Models")
    print(f"{'='*60}")
    
    # Test encoder
    print(f"\nTesting encoder: {encoder_path}")
    enc_session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    
    dummy_image = np.random.randn(1, 3, 1008, 1008).astype(np.float32)
    enc_outputs = enc_session.run(None, {"images": dummy_image})
    
    print("Encoder outputs:")
    for i, out in enumerate(enc_session.get_outputs()):
        print(f"  {out.name}: {enc_outputs[i].shape}")
    
    # Test decoder
    print(f"\nTesting decoder: {decoder_path}")
    dec_session = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
    
    dec_inputs = {
        "image_embed": np.random.randn(1, 256, 72, 72).astype(np.float32),
        "high_res_feats_0": np.random.randn(1, 32, 288, 288).astype(np.float32),
        "high_res_feats_1": np.random.randn(1, 64, 144, 144).astype(np.float32),
        "point_coords": np.array([[[504, 504], [600, 600]]], dtype=np.float32),
        "point_labels": np.array([[1, 1]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
        "has_mask_input": np.array([0.0], dtype=np.float32),
    }
    
    # Check if decoder needs orig_im_size
    input_names = [inp.name for inp in dec_session.get_inputs()]
    if "orig_im_size" in input_names:
        dec_inputs["orig_im_size"] = np.array([1008, 1008], dtype=np.int64)
    
    dec_outputs = dec_session.run(None, dec_inputs)
    
    print("Decoder outputs:")
    for i, out in enumerate(dec_session.get_outputs()):
        if isinstance(dec_outputs[i], np.ndarray):
            print(f"  {out.name}: {dec_outputs[i].shape}")
        else:
            print(f"  {out.name}: {dec_outputs[i]}")
    
    print("\n✓ Both models verified successfully!")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3-Tracker models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export vision encoder
    python export_sam3_onnx.py --export encoder --output vision_encoder.onnx
    
    # Export decoder
    python export_sam3_onnx.py --export decoder --output decoder.onnx
    
    # Export both
    python export_sam3_onnx.py --export both --output-dir ./models/
    
    # Verify exported models
    python export_sam3_onnx.py --verify --encoder-path enc.onnx --decoder-path dec.onnx
        """
    )
    
    parser.add_argument(
        "--export",
        type=str,
        choices=["encoder", "decoder", "both"],
        help="What to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output ONNX file path (for single export)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (for 'both' export)",
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified decoder (default, better ONNX compatibility)",
    )
    parser.add_argument(
        "--sam2-compatible",
        action="store_true",
        help="Use SAM2-compatible decoder (dynamic output size)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported models",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        help="Encoder path for verification",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        help="Decoder path for verification",
    )
    
    args = parser.parse_args()
    
    # Verification mode
    if args.verify:
        if not args.encoder_path or not args.decoder_path:
            parser.error("--verify requires --encoder-path and --decoder-path")
        verify_models(args.encoder_path, args.decoder_path)
        return
    
    # Export mode
    if not args.export:
        parser.error("Must specify --export or --verify")
    
    # Load SAM3 model with proper checkpoint loading
    # NOTE: build_tracker alone does NOT load weights!
    # We need to use build_sam3_image_model with enable_inst_interactivity=True
    # which internally creates the tracker and loads the checkpoint.
    print("Loading SAM3 model from HuggingFace (with checkpoint)...")
    from sam3.model_builder import build_sam3_image_model
    
    sam3_model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=False,  # We only need the tracker/SAM parts
        enable_inst_interactivity=True,  # This creates the tracker with loaded weights
    )
    
    # Extract the tracker from the image model
    tracker = sam3_model.inst_interactive_predictor.model
    print(f"Tracker loaded: {type(tracker).__name__}")
    
    # Verify weights are loaded by checking a known parameter
    pe_weight = tracker.sam_mask_decoder.iou_prediction_head.layers[0].weight
    print(f"Sample weight stats: mean={pe_weight.mean().item():.6f}, std={pe_weight.std().item():.6f}")
    
    simplified = not args.sam2_compatible
    
    if args.export == "encoder":
        output = args.output or "sam3_vision_encoder.onnx"
        export_vision_encoder(tracker, output, args.opset, args.device)
        
    elif args.export == "decoder":
        output = args.output or "sam3_decoder.onnx"
        export_decoder(tracker, output, simplified, args.opset, args.device)
        
    elif args.export == "both":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_vision_encoder(
            tracker,
            str(output_dir / "sam3_vision_encoder.onnx"),
            args.opset,
            args.device,
        )
        export_decoder(
            tracker,
            str(output_dir / "sam3_decoder.onnx"),
            simplified,
            args.opset,
            args.device,
        )
        
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)


if __name__ == "__main__":
    main()
