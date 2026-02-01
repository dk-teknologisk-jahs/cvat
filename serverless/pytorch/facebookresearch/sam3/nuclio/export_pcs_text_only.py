#!/usr/bin/env python3
"""
Export SAM3 PCS decoder for text-only mode to ONNX.

This creates a clean ONNX model that:
1. Takes backbone FPN features and text features (from text encoder)
2. Uses only CLS token for geometry (no points/boxes)
3. Produces segmentation masks based on text prompts

The key insight is that for text-only mode, the geometry encoder:
- Only has the CLS token (no points/boxes)
- The CLS token goes through 3 transformer layers cross-attending to image features
- We can export this path cleanly without dynamic scatter/expand operations

Inputs:
- backbone_fpn_0: [1, 256, 288, 288] - highest resolution FPN
- backbone_fpn_1: [1, 256, 144, 144] - mid resolution FPN
- backbone_fpn_2: [1, 256, 72, 72] - lowest resolution FPN (used for transformer)
- vision_pos_2: [1, 256, 72, 72] - positional encoding for FPN level 2
- text_features: [32, 1, 256] - encoded text features (seq-first)
- text_mask: [1, 32] - text attention mask

Outputs:
- pred_masks: [1, 300, H, W] - predicted segmentation masks
- pred_boxes: [1, 300, 4] - predicted boxes (cxcywh format)
- pred_logits: [1, 300, 1] - class logits
"""

import sys
import os
sys.path.insert(0, "/home/jahs/GitHub/cvat/sam3")

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np


class TextOnlyPCSDecoder(nn.Module):
    """
    Wrapper for SAM3 PCS that handles text-only decoding.

    For text-only mode, the geometry encoder just processes:
    1. CLS token (learnable embedding)
    2. Through 3 transformer layers cross-attending to image features

    This avoids the dynamic scatter/expand operations in the full geometry encoder.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hidden_dim = model.hidden_dim  # 256

        # Get the geometry encoder components we need
        geo_enc = model.geometry_encoder
        self.cls_embed = geo_enc.cls_embed  # Embedding(1, 256)
        self.geo_final_proj = geo_enc.final_proj  # Linear(256, 256)
        self.geo_norm = geo_enc.norm  # LayerNorm(256)
        self.geo_encode_layers = geo_enc.encode  # 3 transformer layers
        self.geo_encode_norm = geo_enc.encode_norm  # LayerNorm(256)

        # Main transformer
        self.transformer = model.transformer

        # Segmentation head
        self.segmentation_head = model.segmentation_head

        # Output heads
        self.dot_prod_scoring = model.dot_prod_scoring

    def _encode_geometry_text_only(self, img_feats, img_pos_embeds, batch_size):
        """
        Encode geometry for text-only mode.
        Only the CLS token, cross-attending to image features.

        Args:
            img_feats: [HW, B, C] image features (seq-first)
            img_pos_embeds: [HW, B, C] positional encodings (seq-first)
            batch_size: int

        Returns:
            geo_feats: [1, B, C] geometry features (just CLS)
            geo_mask: [B, 1] attention mask (all False = all valid)
        """
        # Get CLS token and expand to batch
        cls = self.cls_embed.weight.view(1, 1, self.hidden_dim).expand(1, batch_size, -1)
        # Mask is all False (all tokens valid)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=cls.device)

        # Apply projection and norm
        final_embeds = self.geo_norm(self.geo_final_proj(cls))

        # Cross-attend to image features through transformer layers
        for layer in self.geo_encode_layers:
            final_embeds = layer(
                tgt=final_embeds,
                memory=img_feats,
                tgt_key_padding_mask=cls_mask,
                pos=img_pos_embeds,
            )
        final_embeds = self.geo_encode_norm(final_embeds)

        return final_embeds, cls_mask

    def forward(self, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2, vision_pos_2, text_features, text_mask):
        """
        Forward pass for text-only PCS decoding.

        Args:
            backbone_fpn_0: [B, C, H0, W0] - FPN level 0 (288x288)
            backbone_fpn_1: [B, C, H1, W1] - FPN level 1 (144x144)
            backbone_fpn_2: [B, C, H2, W2] - FPN level 2 (72x72)
            vision_pos_2: [B, C, H2, W2] - positional encoding for level 2
            text_features: [L, B, C] - text features (32, 1, 256)
            text_mask: [B, L] - text attention mask (1, 32)

        Returns:
            pred_masks: [B, num_queries, H, W]
            pred_boxes: [B, num_queries, 4]
            pred_logits: [B, num_queries, 1]
        """
        batch_size = backbone_fpn_2.shape[0]
        device = backbone_fpn_2.device

        # Get spatial dimensions
        h2, w2 = backbone_fpn_2.shape[-2:]  # 72, 72

        # Convert FPN features to seq-first format [HW, B, C]
        fpn_feat = backbone_fpn_2.flatten(2).permute(2, 0, 1)  # [5184, B, 256]
        fpn_pos = vision_pos_2.flatten(2).permute(2, 0, 1)  # [5184, B, 256]

        img_feats = [fpn_feat]
        img_pos_embeds = [fpn_pos]
        vis_feat_sizes = [(h2, w2)]

        # Encode geometry (text-only mode = just CLS token)
        geo_feats, geo_mask = self._encode_geometry_text_only(
            fpn_feat, fpn_pos, batch_size
        )

        # Combine text and geometry features
        # txt_feats: [L, B, C], geo_feats: [1, B, C]
        prompt = torch.cat([text_features, geo_feats], dim=0)
        prompt_mask = torch.cat([text_mask, geo_mask], dim=1)

        # Run the encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        memory_out = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        # Extract encoder outputs
        memory = memory_out["memory"]
        pos_embed = memory_out["pos_embed"]
        padding_mask = memory_out["padding_mask"]
        level_start_index = memory_out["level_start_index"]
        spatial_shapes = memory_out["spatial_shapes"]
        valid_ratios = memory_out["valid_ratios"]
        prompt_after_enc = memory_out.get("memory_text", prompt)

        # Run the decoder
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        hs, reference_boxes, _, _ = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=padding_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            memory_text=prompt_after_enc,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        # Convert from seq-first to batch-first
        hs = hs.transpose(1, 2)  # [num_layers, seq, batch, dim] -> [num_layers, batch, seq, dim]
        reference_boxes = reference_boxes.transpose(1, 2)

        # Get class scores via dot product scoring
        outputs_class = self.dot_prod_scoring(hs, prompt_after_enc, prompt_mask)

        # Get box predictions
        anchor_box_offsets = self.transformer.decoder.bbox_embed(hs)
        reference_boxes_inv_sig = self._inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()

        # Get final layer outputs
        final_hs = hs[-1]  # [B, num_queries, C]
        final_boxes = outputs_coord[-1]  # [B, num_queries, 4]
        final_logits = outputs_class[-1]  # [B, num_queries, 1]

        # Simplified mask generation without full pixel decoder
        # Use the mask_predictor's mask_embed directly on the smallest FPN level
        # This produces lower resolution masks but is more ONNX-compatible
        mask_embed = self.segmentation_head.mask_predictor.mask_embed(final_hs)  # [B, num_queries, C]

        # Create pixel features by using encoder visual features
        # Extract visual embeddings from encoder memory
        spatial_dim = h2 * w2  # 5184
        encoder_visual_embed = memory[:spatial_dim]  # [5184, B, C]
        pixel_embed = encoder_visual_embed.permute(1, 2, 0).view(batch_size, -1, h2, w2)  # [B, C, H, W]

        # Compute masks via einsum
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_embed)  # [B, num_queries, 72, 72]

        return pred_masks, final_boxes, final_logits

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))


def export_text_only_decoder():
    """Export the text-only PCS decoder to ONNX."""
    import gc

    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading SAM3 model...")
    from model_handler_pcs import ModelHandlerPCS
    handler = ModelHandlerPCS()
    model = handler.model
    model.eval()

    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    print("Creating text-only decoder wrapper...")
    decoder = TextOnlyPCSDecoder(model)
    decoder.eval()
    decoder.cuda()

    # Create dummy inputs - only need FPN level 2 for the simplified version
    batch_size = 1
    hidden_dim = 256
    text_len = 32

    # FPN level 2 (lowest resolution, used for transformer)
    backbone_fpn_2 = torch.randn(batch_size, hidden_dim, 72, 72).cuda()
    vision_pos_2 = torch.randn(batch_size, hidden_dim, 72, 72).cuda()

    # Dummy FPN levels (not used in simplified version but keep for interface)
    backbone_fpn_0 = torch.randn(batch_size, hidden_dim, 288, 288).cuda()
    backbone_fpn_1 = torch.randn(batch_size, hidden_dim, 144, 144).cuda()

    # Text features (seq-first)
    text_features = torch.randn(text_len, batch_size, hidden_dim).cuda()
    text_mask = torch.zeros(batch_size, text_len, dtype=torch.bool).cuda()

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        pred_masks, pred_boxes, pred_logits = decoder(
            backbone_fpn_0, backbone_fpn_1, backbone_fpn_2, vision_pos_2,
            text_features, text_mask
        )

    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  pred_boxes: {pred_boxes.shape}")
    print(f"  pred_logits: {pred_logits.shape}")

    # Clear memory before tracing
    print(f"\n  GPU memory before export: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    torch.cuda.empty_cache()
    gc.collect()

    # Clear any cached tensors
    if hasattr(model.transformer.decoder, '_reference_boxes'):
        model.transformer.decoder._reference_boxes = None

    # Export to ONNX - fully move model and all buffers to CPU
    print("\nMoving entire model to CPU for export...")

    # Move the entire model (including all buffers and cached tensors) to CPU
    model = model.cpu()
    decoder = TextOnlyPCSDecoder(model)  # Recreate wrapper with CPU model
    decoder.eval()

    # Clear all CUDA caches
    if hasattr(model.transformer.decoder, '_reference_boxes'):
        model.transformer.decoder._reference_boxes = None
    if hasattr(model.transformer.decoder, 'compilable_stored_size'):
        model.transformer.decoder.compilable_stored_size = None

    backbone_fpn_0 = backbone_fpn_0.cpu()
    backbone_fpn_1 = backbone_fpn_1.cpu()
    backbone_fpn_2 = backbone_fpn_2.cpu()
    vision_pos_2 = vision_pos_2.cpu()
    text_features = text_features.cpu()
    text_mask = text_mask.cpu()

    torch.cuda.empty_cache()
    gc.collect()

    output_path = "/home/jahs/GitHub/cvat/serverless/pytorch/facebookresearch/sam3/nuclio/pcs_text_only.onnx"

    print(f"\nExporting to {output_path}...")

    torch.onnx.export(
        decoder,
        (backbone_fpn_0, backbone_fpn_1, backbone_fpn_2, vision_pos_2, text_features, text_mask),
        output_path,
        input_names=[
            "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
            "vision_pos_2", "text_features", "text_mask"
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits"],
        dynamic_axes={
            "backbone_fpn_0": {0: "batch"},
            "backbone_fpn_1": {0: "batch"},
            "backbone_fpn_2": {0: "batch"},
            "vision_pos_2": {0: "batch"},
            "text_features": {1: "batch"},
            "text_mask": {0: "batch"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    # Check file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    # Verify with ONNX checker
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid")

    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    sess = ort.InferenceSession(
        output_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # Get input/output names
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print("  Inputs:", input_names)
    print("  Outputs:", output_names)

    # Run inference - only use the inputs that are actually in the ONNX model
    inputs = {}
    if "backbone_fpn_0" in input_names:
        inputs["backbone_fpn_0"] = backbone_fpn_0.numpy()
    if "backbone_fpn_1" in input_names:
        inputs["backbone_fpn_1"] = backbone_fpn_1.numpy()
    if "backbone_fpn_2" in input_names:
        inputs["backbone_fpn_2"] = backbone_fpn_2.numpy()
    if "vision_pos_2" in input_names:
        inputs["vision_pos_2"] = vision_pos_2.numpy()
    if "text_features" in input_names:
        inputs["text_features"] = text_features.numpy()
    if "text_mask" in input_names:
        inputs["text_mask"] = text_mask.numpy()

    ort_outputs = sess.run(None, inputs)

    print("\n  ONNX Runtime outputs:")
    print(f"    pred_masks: {ort_outputs[0].shape}")
    print(f"    pred_boxes: {ort_outputs[1].shape}")
    print(f"    pred_logits: {ort_outputs[2].shape}")

    # Compare with PyTorch
    print("\nComparing PyTorch vs ONNX Runtime...")
    with torch.no_grad():
        pt_masks, pt_boxes, pt_logits = decoder(
            backbone_fpn_0, backbone_fpn_1, backbone_fpn_2, vision_pos_2,
            text_features, text_mask
        )

    pt_masks_np = pt_masks.numpy()
    pt_boxes_np = pt_boxes.numpy()
    pt_logits_np = pt_logits.numpy()

    masks_diff = np.abs(ort_outputs[0] - pt_masks_np).max()
    boxes_diff = np.abs(ort_outputs[1] - pt_boxes_np).max()
    logits_diff = np.abs(ort_outputs[2] - pt_logits_np).max()

    print(f"  Max diff pred_masks: {masks_diff:.6f}")
    print(f"  Max diff pred_boxes: {boxes_diff:.6f}")
    print(f"  Max diff pred_logits: {logits_diff:.6f}")

    if masks_diff < 0.01 and boxes_diff < 1e-3 and logits_diff < 1e-3:
        print("\n✓ ONNX export successful! Outputs match PyTorch within tolerance.")
    else:
        print("\n⚠ Warning: Outputs differ more than expected. Check numerical precision.")

    return output_path


if __name__ == "__main__":
    export_text_only_decoder()
