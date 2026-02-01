#!/usr/bin/env python3
"""
Export Unified SAM3 PCS Model to ONNX

This creates a single ONNX model that handles text-to-segment with optional box guidance.
The model includes the full PCS pipeline and can work with:
- Text prompts only
- Text prompts + box prompts (for guided segmentation)

Key changes from previous export:
1. Trace with valid (non-empty) geometry prompt to avoid 0-dimension issues
2. Use box_mask to indicate which boxes are valid (allows text-only at inference)
3. All FPN features and positions are inputs (no optimization away)

Usage:
    python export_unified_pcs.py

Output:
    unified_pcs.onnx - Complete PCS pipeline supporting text and box prompts
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add SAM3 to path
SAM3_PATH = "/home/jahs/GitHub/cvat/sam3"
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import inverse_sigmoid
from sam3.model.box_ops import box_cxcywh_to_xyxy


class UnifiedPCSWrapper(nn.Module):
    """
    Unified SAM3 PCS wrapper for ONNX export.

    Supports both text-only and text+box prompt modes through masking.

    The key insight is that we need to trace with valid (non-empty) geometry
    to avoid ONNX baking in 0-dimension shapes. At inference time, we use
    box_mask to indicate which boxes are valid (all False = text-only mode).
    """

    def __init__(self, sam3_model):
        super().__init__()
        self.num_feature_levels = sam3_model.num_feature_levels  # 1

        # Core components
        self.geometry_encoder = sam3_model.geometry_encoder
        self.transformer = sam3_model.transformer
        self.segmentation_head = sam3_model.segmentation_head
        self.dot_prod_scoring = sam3_model.dot_prod_scoring
        self.hidden_dim = sam3_model.hidden_dim

        # Clear any cached tensors
        self._clear_caches()

    def _clear_caches(self):
        """Clear internal caches that might have wrong device tensors."""
        if hasattr(self.transformer.decoder, 'compilable_cord_cache'):
            self.transformer.decoder.compilable_cord_cache = None
        if hasattr(self.transformer.decoder, 'coord_cache'):
            self.transformer.decoder.coord_cache = {}
        if hasattr(self.transformer.decoder, 'compilable_stored_size'):
            self.transformer.decoder.compilable_stored_size = None

    def forward(
        self,
        # FPN features (all 3 levels, all 256 channels)
        fpn_feat_0: torch.Tensor,  # [B, 256, 288, 288]
        fpn_feat_1: torch.Tensor,  # [B, 256, 144, 144]
        fpn_feat_2: torch.Tensor,  # [B, 256, 72, 72]
        # FPN position encodings (all 3 levels)
        fpn_pos_0: torch.Tensor,   # [B, 256, 288, 288]
        fpn_pos_1: torch.Tensor,   # [B, 256, 144, 144]
        fpn_pos_2: torch.Tensor,   # [B, 256, 72, 72]
        # Text features
        text_features: torch.Tensor,  # [32, B, 256]
        text_mask: torch.Tensor,      # [B, 32] - True means padding (ignore)
        # Box prompts (for guided segmentation)
        box_coords: torch.Tensor,     # [B, num_boxes, 4] - normalized cxcywh format
        box_mask: torch.Tensor,       # [B, num_boxes] - True means VALID box
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run unified PCS pipeline.

        Args:
            fpn_feat_*: FPN features from vision encoder
            fpn_pos_*: Position encodings for FPN features
            text_features: Text embeddings [32, B, 256]
            text_mask: Text padding mask [B, 32], True = padding
            box_coords: Box coordinates [B, num_boxes, 4] in cxcywh normalized format
            box_mask: Box validity mask [B, num_boxes], True = VALID (opposite of text_mask!)

        Returns:
            masks: [B, 200, H, W] predicted masks at 288x288
            boxes: [B, 200, 4] predicted boxes in xyxy normalized format
            scores: [B, 200] confidence scores
        """
        batch_size = fpn_feat_0.shape[0]
        device = fpn_feat_0.device

        # All FPN levels (for segmentation head upsampling)
        backbone_fpn_all = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        vis_pos_enc_all = [fpn_pos_0, fpn_pos_1, fpn_pos_2]

        # Only LAST level for transformer encoder (num_feature_levels=1)
        vis_feats = [fpn_feat_2]
        vis_pos_enc = [fpn_pos_2]
        vis_feat_sizes = [(72, 72)]

        # Flatten to sequence format: [B, C, H, W] -> [H*W, B, C]
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]

        # Encode geometry prompts
        # box_coords: [B, num_boxes, 4] -> [num_boxes, B, 4]
        box_embeddings = box_coords.permute(1, 0, 2)  # [num_boxes, B, 4]

        # Invert box_mask for geometry encoder (it expects True=padding, but we use True=valid)
        geo_mask = ~box_mask  # [B, num_boxes] - now True=padding (invalid)

        # Run geometry encoder
        geo_feats, geo_masks = self._encode_geometry(
            box_embeddings=box_embeddings,
            box_mask=geo_mask,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        # Combine text and geometry prompts
        # text_features: [32, B, 256]
        # geo_feats: [num_geo, B, 256]
        prompt = torch.cat([text_features, geo_feats], dim=0)
        prompt_mask = torch.cat([text_mask, geo_masks], dim=1)

        # Run transformer encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        encoder_hidden_states = memory["memory"]
        pos_embed = memory["pos_embed"]
        padding_mask = memory["padding_mask"]

        # Run transformer decoder
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=tgt,
            memory=encoder_hidden_states,
            memory_key_padding_mask=padding_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        # Convert to batch-first
        hs = hs.transpose(1, 2)  # [layers, B, queries, C]
        reference_boxes = reference_boxes.transpose(1, 2)

        # Compute scores
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        presence_score = dec_presence_out.transpose(1, 2).sigmoid()
        final_scores = (outputs_class[-1].sigmoid() * presence_score[-1].unsqueeze(-1)).squeeze(-1)

        # Compute boxes
        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord[-1])

        # Run segmentation head
        seg_outputs = self.segmentation_head(
            backbone_feats=backbone_fpn_all,
            obj_queries=hs,
            image_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        masks = seg_outputs["pred_masks"]

        return masks, boxes_xyxy, final_scores

    def _encode_geometry(
        self,
        box_embeddings: torch.Tensor,  # [num_boxes, B, 4]
        box_mask: torch.Tensor,        # [B, num_boxes] True=padding
        img_feats: List[torch.Tensor],
        img_sizes: List[Tuple[int, int]],
        img_pos_embeds: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode geometry prompts using the geometry encoder.

        This is a simplified version that only handles box prompts.
        """
        from sam3.model.geometry_encoders import Prompt

        batch_size = box_mask.shape[0]
        num_boxes = box_embeddings.shape[0]
        device = box_embeddings.device

        # Create box labels (all positive)
        box_labels = torch.ones(num_boxes, batch_size, dtype=torch.bool, device=device)

        # Create Prompt object
        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
        )

        # Run geometry encoder
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=img_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        return geo_feats, geo_masks


def get_sample_inputs(batch_size: int = 1, num_boxes: int = 1, device: str = "cpu"):
    """Create sample inputs for tracing with VALID geometry."""
    return {
        # FPN features
        "fpn_feat_0": torch.randn(batch_size, 256, 288, 288, device=device),
        "fpn_feat_1": torch.randn(batch_size, 256, 144, 144, device=device),
        "fpn_feat_2": torch.randn(batch_size, 256, 72, 72, device=device),
        # FPN positions
        "fpn_pos_0": torch.randn(batch_size, 256, 288, 288, device=device),
        "fpn_pos_1": torch.randn(batch_size, 256, 144, 144, device=device),
        "fpn_pos_2": torch.randn(batch_size, 256, 72, 72, device=device),
        # Text features
        "text_features": torch.randn(32, batch_size, 256, device=device),
        "text_mask": torch.zeros(batch_size, 32, dtype=torch.bool, device=device),
        # Box prompts (VALID boxes for tracing)
        "box_coords": torch.tensor([[[0.5, 0.5, 0.3, 0.3]]], device=device).expand(batch_size, num_boxes, 4),
        "box_mask": torch.ones(batch_size, num_boxes, dtype=torch.bool, device=device),  # True = valid
    }


def test_wrapper(wrapper, device="cpu"):
    """Test the wrapper with sample inputs."""
    print("\nTesting unified PCS wrapper...")

    # Test with valid box
    print("\n1. Testing with valid box prompt...")
    inputs = get_sample_inputs(batch_size=1, num_boxes=1, device=device)

    with torch.no_grad():
        masks, boxes, scores = wrapper(**inputs)

    print(f"  Masks shape: {masks.shape}")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Test text-only mode (all boxes masked as invalid)
    print("\n2. Testing text-only mode (no valid boxes)...")
    inputs["box_mask"] = torch.zeros(1, 1, dtype=torch.bool, device=device)  # False = invalid

    with torch.no_grad():
        masks2, boxes2, scores2 = wrapper(**inputs)

    print(f"  Masks shape: {masks2.shape}")
    print(f"  Score range: [{scores2.min():.4f}, {scores2.max():.4f}]")

    return masks, boxes, scores


def export_to_onnx(wrapper, output_path: str, device: str = "cpu"):
    """Export the unified PCS model to ONNX."""
    print(f"\nExporting unified PCS to {output_path}...")

    # Use 1 box for tracing (can be masked at inference)
    inputs = get_sample_inputs(batch_size=1, num_boxes=1, device=device)

    input_names = [
        "fpn_feat_0", "fpn_feat_1", "fpn_feat_2",
        "fpn_pos_0", "fpn_pos_1", "fpn_pos_2",
        "text_features", "text_mask",
        "box_coords", "box_mask",
    ]

    output_names = ["masks", "boxes", "scores"]

    # Dynamic axes
    dynamic_axes = {
        "fpn_feat_0": {0: "batch"},
        "fpn_feat_1": {0: "batch"},
        "fpn_feat_2": {0: "batch"},
        "fpn_pos_0": {0: "batch"},
        "fpn_pos_1": {0: "batch"},
        "fpn_pos_2": {0: "batch"},
        "text_features": {1: "batch"},
        "text_mask": {0: "batch"},
        "box_coords": {0: "batch", 1: "num_boxes"},
        "box_mask": {0: "batch", 1: "num_boxes"},
        "masks": {0: "batch"},
        "boxes": {0: "batch"},
        "scores": {0: "batch"},
    }

    # Export
    torch.onnx.export(
        wrapper,
        tuple(inputs[name] for name in input_names),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Exported to {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model verified!")

    # Print size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # Print inputs/outputs
    print("\nONNX inputs:")
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")

    print("\nONNX outputs:")
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


def verify_onnx(wrapper, onnx_path: str, device: str = "cpu"):
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort

    print(f"\nVerifying ONNX against PyTorch...")

    # Load ONNX
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Get actual input names from ONNX model
    onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]
    print(f"ONNX expects inputs: {onnx_input_names}")

    # Create test inputs
    inputs = get_sample_inputs(batch_size=1, num_boxes=1, device=device)

    # PyTorch forward
    with torch.no_grad():
        pt_masks, pt_boxes, pt_scores = wrapper(**inputs)

    # ONNX forward - only provide inputs that ONNX expects
    onnx_inputs = {}
    for name in onnx_input_names:
        if name in inputs:
            val = inputs[name]
            if isinstance(val, torch.Tensor):
                onnx_inputs[name] = val.numpy()
            else:
                onnx_inputs[name] = val

    onnx_outputs = onnx_session.run(None, onnx_inputs)
    onnx_masks, onnx_boxes, onnx_scores = onnx_outputs

    # Compare
    pt_masks_np = pt_masks.numpy()
    pt_boxes_np = pt_boxes.numpy()
    pt_scores_np = pt_scores.numpy()

    masks_mae = np.abs(pt_masks_np - onnx_masks).mean()
    boxes_mae = np.abs(pt_boxes_np - onnx_boxes).mean()
    scores_mae = np.abs(pt_scores_np - onnx_scores).mean()

    print(f"\nComparison (Mean Absolute Error):")
    print(f"  Masks MAE: {masks_mae:.6f}")
    print(f"  Boxes MAE: {boxes_mae:.6f}")
    print(f"  Scores MAE: {scores_mae:.6f}")

    if masks_mae < 0.001 and boxes_mae < 0.001 and scores_mae < 0.001:
        print("\n✅ ONNX matches PyTorch!")
        return True
    else:
        print("\n⚠️ ONNX differs from PyTorch")
        return False


def main():
    print("=" * 60)
    print("Unified SAM3 PCS Export")
    print("=" * 60)

    device = "cpu"
    print(f"\nUsing device: {device}")

    # Build SAM3 model
    print("\nLoading SAM3 model...")
    sam3_model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    # Create wrapper
    print("\nCreating unified PCS wrapper...")
    wrapper = UnifiedPCSWrapper(sam3_model)
    wrapper.eval()

    # Test wrapper
    test_wrapper(wrapper, device)

    # Export to ONNX
    output_path = os.path.join(os.path.dirname(__file__), "unified_pcs.onnx")

    try:
        export_to_onnx(wrapper, output_path, device)

        # Verify ONNX
        verify_onnx(wrapper, output_path, device)

    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
