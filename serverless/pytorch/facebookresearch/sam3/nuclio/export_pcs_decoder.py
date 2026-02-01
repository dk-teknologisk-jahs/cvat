#!/usr/bin/env python3
"""
Export SAM3 PCS Decoder to ONNX

This exports the PCS (Perception Continuum Segmentation) decoder pipeline that takes:
- FPN features from vision encoder (only 72x72 level for transformer encoder)
- Text features from text encoder
- Optional box prompts

And outputs:
- Predicted masks
- Predicted boxes
- Confidence scores

The exported model allows running text-to-segment without requiring the gated
HuggingFace SAM3 weights at inference time.

Architecture:
    Input Features
         │
         ▼
    ┌─────────────┐
    │  Geometry   │ (encodes optional box prompts)
    │   Encoder   │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Transformer │ (fuses image[72x72] + text/geometry features)
    │   Encoder   │  NOTE: Only uses LAST FPN level (72x72) 
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Transformer │ (generates 200 object queries)
    │   Decoder   │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Segmentation│ (uses ALL FPN levels for mask upsampling)
    │    Head     │
    └─────────────┘
         │
         ▼
    Masks, Boxes, Scores

Usage:
    python export_pcs_decoder.py

Output:
    pcs_decoder.onnx - Full PCS decoder pipeline
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
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import inverse_sigmoid
from sam3.model.box_ops import box_cxcywh_to_xyxy


class PCSDecoderWrapper(nn.Module):
    """
    Wrapper for SAM3 PCS decoder that can be exported to ONNX.
    
    Takes pre-computed vision and text features and runs the PCS pipeline.
    
    Key insight: SAM3 uses num_feature_levels=1, meaning the transformer encoder
    only operates on the LAST (smallest) FPN level (72x72), but the segmentation
    head uses ALL FPN levels for upsampling masks.
    """
    
    def __init__(self, sam3_model):
        super().__init__()
        self.sam3 = sam3_model
        self.num_feature_levels = sam3_model.num_feature_levels  # Should be 1
        
        # Extract components we need
        self.geometry_encoder = sam3_model.geometry_encoder
        self.transformer = sam3_model.transformer
        self.segmentation_head = sam3_model.segmentation_head
        self.dot_prod_scoring = sam3_model.dot_prod_scoring
        self.hidden_dim = sam3_model.hidden_dim
        
        # Clear any cached tensors that might be on wrong device
        self._clear_caches()
    
    def _clear_caches(self):
        """Clear internal caches that might have wrong device tensors."""
        # Clear decoder coord cache
        if hasattr(self.transformer.decoder, 'compilable_cord_cache'):
            self.transformer.decoder.compilable_cord_cache = None
        if hasattr(self.transformer.decoder, 'coord_cache'):
            self.transformer.decoder.coord_cache = {}
        if hasattr(self.transformer.decoder, 'compilable_stored_size'):
            self.transformer.decoder.compilable_stored_size = None
        
    def forward(
        self,
        # Vision encoder outputs (FPN features) - all 3 levels for segmentation head
        fpn_feat_0: torch.Tensor,  # [B, 256, 288, 288] high_res_feats_0
        fpn_feat_1: torch.Tensor,  # [B, 256, 144, 144] high_res_feats_1
        fpn_feat_2: torch.Tensor,  # [B, 256, 72, 72] image_embed
        # Vision position encodings - all 3 levels
        fpn_pos_0: torch.Tensor,   # [B, 256, 288, 288]
        fpn_pos_1: torch.Tensor,   # [B, 256, 144, 144]
        fpn_pos_2: torch.Tensor,   # [B, 256, 72, 72]
        # Text encoder outputs
        text_features: torch.Tensor,  # [32, B, 256]
        text_mask: torch.Tensor,      # [B, 32] - True for padding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run PCS decoder pipeline.
        
        Returns:
            masks: [B, num_queries, H, W] predicted masks at 72x72 resolution
            boxes: [B, num_queries, 4] predicted boxes (xyxy format, normalized 0-1)
            scores: [B, num_queries] confidence scores
        """
        batch_size = fpn_feat_0.shape[0]
        device = fpn_feat_0.device
        
        # All FPN levels (for segmentation head)
        backbone_fpn_all = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        vis_pos_enc_all = [fpn_pos_0, fpn_pos_1, fpn_pos_2]
        
        # Only LAST level for transformer encoder (num_feature_levels=1)
        vis_feats = backbone_fpn_all[-self.num_feature_levels:]  # [fpn_feat_2]
        vis_pos_enc = vis_pos_enc_all[-self.num_feature_levels:]  # [fpn_pos_2]
        vis_feat_sizes = [(72, 72)]  # Only 72x72
        
        # Flatten to sequence format: [B, C, H, W] -> [H*W, B, C]
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        
        # Create empty geometry prompt (text-only mode)
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, batch_size, 4, device=device),
            box_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
        )
        
        # Encode geometry (empty for text-only)
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        
        # Combine text and geometry prompts
        # text_features: [32, B, 256]
        # geo_feats: [0, B, 256] (empty for text-only)
        prompt = torch.cat([text_features, geo_feats], dim=0)
        prompt_mask = torch.cat([text_mask, geo_masks], dim=1)
        
        # Run transformer encoder (only on 72x72 features)
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
        hs = hs.transpose(1, 2)  # [num_layers, B, num_queries, C]
        reference_boxes = reference_boxes.transpose(1, 2)
        
        # Compute scores using dot product scoring
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        
        # Apply presence score to final class scores
        presence_score = dec_presence_out.transpose(1, 2).sigmoid()  # [layers, B, queries]
        final_scores = (outputs_class[-1].sigmoid() * presence_score[-1].unsqueeze(-1)).squeeze(-1)
        
        # Compute boxes
        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord[-1])  # [B, num_queries, 4]
        
        # Run segmentation head with ALL FPN levels
        seg_outputs = self.segmentation_head(
            backbone_feats=backbone_fpn_all,
            obj_queries=hs,
            image_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        
        masks = seg_outputs["pred_masks"]  # [B, num_queries, H, W]
        
        return masks, boxes_xyxy, final_scores


def get_sample_inputs(batch_size: int = 1, device: str = "cuda"):
    """Create sample inputs for tracing."""
    return {
        # FPN features (vision encoder outputs) - ALL 256 channels
        "fpn_feat_0": torch.randn(batch_size, 256, 288, 288, device=device),
        "fpn_feat_1": torch.randn(batch_size, 256, 144, 144, device=device),
        "fpn_feat_2": torch.randn(batch_size, 256, 72, 72, device=device),
        # Position encodings
        "fpn_pos_0": torch.randn(batch_size, 256, 288, 288, device=device),
        "fpn_pos_1": torch.randn(batch_size, 256, 144, 144, device=device),
        "fpn_pos_2": torch.randn(batch_size, 256, 72, 72, device=device),
        # Text features [seq_len=32, batch, 256]
        "text_features": torch.randn(32, batch_size, 256, device=device),
        "text_mask": torch.zeros(batch_size, 32, dtype=torch.bool, device=device),
    }


def test_wrapper(wrapper, device="cuda"):
    """Test the wrapper with sample inputs."""
    print("\nTesting PCS decoder wrapper...")
    
    inputs = get_sample_inputs(batch_size=1, device=device)
    
    with torch.no_grad():
        masks, boxes, scores = wrapper(
            inputs["fpn_feat_0"],
            inputs["fpn_feat_1"],
            inputs["fpn_feat_2"],
            inputs["fpn_pos_0"],
            inputs["fpn_pos_1"],
            inputs["fpn_pos_2"],
            inputs["text_features"],
            inputs["text_mask"],
        )
    
    print(f"  Masks shape: {masks.shape}")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return masks, boxes, scores


def export_to_onnx(wrapper, output_path: str, device: str = "cuda"):
    """Export the PCS decoder to ONNX."""
    print(f"\nExporting PCS decoder to {output_path}...")
    
    inputs = get_sample_inputs(batch_size=1, device=device)
    
    # Input names
    input_names = [
        "fpn_feat_0", "fpn_feat_1", "fpn_feat_2",
        "fpn_pos_0", "fpn_pos_1", "fpn_pos_2",
        "text_features", "text_mask",
    ]
    
    # Output names
    output_names = ["masks", "boxes", "scores"]
    
    # Dynamic axes for batch size
    dynamic_axes = {
        "fpn_feat_0": {0: "batch"},
        "fpn_feat_1": {0: "batch"},
        "fpn_feat_2": {0: "batch"},
        "fpn_pos_0": {0: "batch"},
        "fpn_pos_1": {0: "batch"},
        "fpn_pos_2": {0: "batch"},
        "text_features": {1: "batch"},
        "text_mask": {0: "batch"},
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


def main():
    print("=" * 60)
    print("SAM3 PCS Decoder Export")
    print("=" * 60)
    
    # Use CPU for export to avoid GPU memory issues
    # The model will still work on GPU at inference time
    device = "cpu"
    print(f"\nUsing device: {device} (for export, avoids GPU memory issues)")
    
    # Build SAM3 model with segmentation enabled
    print("\nLoading SAM3 model...")
    sam3_model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    
    # Create wrapper
    print("\nCreating PCS decoder wrapper...")
    wrapper = PCSDecoderWrapper(sam3_model)
    wrapper.eval()
    
    # Test wrapper
    test_wrapper(wrapper, device)
    
    # Export to ONNX
    output_path = os.path.join(os.path.dirname(__file__), "pcs_decoder.onnx")
    
    try:
        export_to_onnx(wrapper, output_path, device)
    except Exception as e:
        print(f"\nONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Export failed - analyzing the issue...")
        print("=" * 60)
        print("""
Common issues with ONNX export:
1. Dynamic control flow (if/else based on tensor values)
2. In-place operations
3. Unsupported operations
4. Data-dependent shapes

Alternatives:
1. Use torch.jit.trace with strict=False
2. Export components separately
3. Use torch.onnx.export with custom symbolic functions
""")


if __name__ == "__main__":
    main()
