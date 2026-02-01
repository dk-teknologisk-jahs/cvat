#!/usr/bin/env python3
"""
SAM3 HuggingFace ONNX Export Script

Exports all SAM3 components from HuggingFace Transformers to ONNX format.
This gives us full control over the ONNX models instead of relying on external sources.

Components exported:
1. Vision Encoder - outputs 256/256/256 channels (no projections baked in)
2. Tracker Decoder - includes conv_s0/conv_s1 projections for point/box prompts
3. Text Encoder - for PCS text-to-segment mode
4. PCS Decoder - for text-to-segment detection

Key design decisions:
- Vision encoder outputs raw FPN features (256 channels at all levels)
- Tracker decoder includes conv_s0/conv_s1 projections internally
- This allows sharing one vision encoder between tracker and PCS modes

Usage:
    # Export all components
    python export_hf_onnx.py --all --output-dir ./onnx-exports

    # Export specific components
    python export_hf_onnx.py --vision-encoder --output-dir ./onnx-exports
    python export_hf_onnx.py --tracker-decoder --output-dir ./onnx-exports
    python export_hf_onnx.py --text-encoder --output-dir ./onnx-exports
    python export_hf_onnx.py --pcs-decoder --output-dir ./onnx-exports

    # Verify equivalence with PyTorch
    python export_hf_onnx.py --verify --output-dir ./onnx-exports
"""

import argparse
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Vision Encoder Wrapper
# =============================================================================

def compute_sine_position_encoding(
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """Compute sine position encoding for transformer attention."""
    batch_size, channels, height, width = shape

    y_embed = (
        torch.arange(1, height + 1, dtype=dtype, device=device)
        .view(1, height, 1)
        .expand(batch_size, height, width)
    )
    x_embed = (
        torch.arange(1, width + 1, dtype=dtype, device=device)
        .view(1, 1, width)
        .expand(batch_size, height, width)
    )

    eps = 1e-6
    y_embed = y_embed / (height + eps) * scale
    x_embed = x_embed / (width + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)

    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for HuggingFace SAM3 vision encoder.

    Outputs raw FPN features with 256 channels at all levels.
    Does NOT include conv_s0/conv_s1 projections - those are in the tracker decoder.

    Outputs:
        fpn_feat_0: [B, 256, 288, 288] - highest resolution
        fpn_feat_1: [B, 256, 144, 144] - mid resolution
        fpn_feat_2: [B, 256, 72, 72]   - lowest resolution (main embedding)
        fpn_pos_2:  [B, 256, 72, 72]   - position encoding for level 2
    """

    def __init__(
        self,
        sam3_model,
        device: str = "cpu",
        image_height: int = 1008,
        image_width: int = 1008,
    ):
        super().__init__()
        from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding

        backbone = sam3_model.vision_encoder.backbone
        self.patch_embeddings = backbone.embeddings.patch_embeddings
        self.dropout = backbone.embeddings.dropout
        self.layer_norm = backbone.layer_norm
        self.layers = backbone.layers
        self.neck = sam3_model.vision_encoder.neck

        patch_size = backbone.config.patch_size  # 14
        self.height_patches = image_height // patch_size  # 72
        self.width_patches = image_width // patch_size    # 72
        hidden_size = backbone.config.hidden_size  # 1024

        # Update rotary embeddings for the target resolution
        for layer in self.layers:
            if getattr(layer, "window_size", 0) != 0:
                continue
            config = getattr(layer, "config", backbone.config)
            rotary_scale = config.window_size / self.height_patches
            layer.rotary_emb = Sam3ViTRotaryEmbedding(
                config,
                end_x=self.height_patches,
                end_y=self.width_patches,
                scale=rotary_scale,
            ).to(device)

        # Interpolate position embeddings to target resolution
        orig_pos_embed = backbone.embeddings.position_embeddings.data
        pretrain_size = int(orig_pos_embed.shape[1] ** 0.5)

        pos_embed = orig_pos_embed.reshape(
            1, pretrain_size, pretrain_size, hidden_size
        ).permute(0, 3, 1, 2)
        repeat_h = self.height_patches // pretrain_size + 1
        repeat_w = self.width_patches // pretrain_size + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[
            :, :, : self.height_patches, : self.width_patches
        ]
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(
            1, self.height_patches * self.width_patches, hidden_size
        )
        self.register_buffer("vit_pos_embed", pos_embed.to(device))

        # Pre-compute position encoding for level 2
        num_pos_feats = sam3_model.vision_encoder.neck.config.fpn_hidden_size // 2
        pos_enc_2 = compute_sine_position_encoding(
            shape=(1, 256, self.height_patches, self.width_patches),
            device=torch.device(device),
            dtype=torch.float32,
            num_pos_feats=num_pos_feats,
        )
        self.register_buffer("pos_enc_2", pos_enc_2)

    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]

        embeddings = self.patch_embeddings(images)
        embeddings = embeddings + self.vit_pos_embed
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings.view(
            batch_size, self.height_patches, self.width_patches, -1
        )
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.view(
            batch_size, self.height_patches * self.width_patches, -1
        )
        hidden_states_spatial = hidden_states.view(
            batch_size, self.height_patches, self.width_patches, -1
        ).permute(0, 3, 1, 2)

        fpn_hidden_states, _ = self.neck(hidden_states_spatial)

        return (
            fpn_hidden_states[0],  # [B, 256, 288, 288]
            fpn_hidden_states[1],  # [B, 256, 144, 144]
            fpn_hidden_states[2],  # [B, 256, 72, 72]
            self.pos_enc_2.expand(batch_size, -1, -1, -1),  # [B, 256, 72, 72]
        )


# =============================================================================
# Tracker Decoder Wrapper (for point/box prompts - Interactor mode)
# =============================================================================

class TrackerDecoderWrapper(nn.Module):
    """
    SAM3 Tracker Decoder wrapper for HuggingFace Sam3TrackerModel.

    This decoder:
    1. Projects high-res features: 256ch → 32ch (conv_s0) and 256ch → 64ch (conv_s1)
    2. Adds no_memory_embedding to the main embedding
    3. Encodes point/box prompts
    4. Runs the SAM mask decoder
    5. Returns multiple mask candidates with IoU scores

    Inputs:
        fpn_feat_0: [B, 256, 288, 288] - from vision encoder
        fpn_feat_1: [B, 256, 144, 144] - from vision encoder
        fpn_feat_2: [B, 256, 72, 72]   - from vision encoder
        point_coords: [B, N, 2] - point coordinates in 1008x1008 space
        point_labels: [B, N] - point labels (1=positive, 0=negative)
        mask_input: [B, 1, 288, 288] - previous mask for refinement
        has_mask_input: [B] - whether mask_input is valid

    Outputs:
        masks: [B, 3, 1008, 1008] - upsampled mask candidates
        iou_predictions: [B, 3] - IoU scores for each mask
        low_res_masks: [B, 3, 288, 288] - for iterative refinement
        object_score_logits: [B, 1] - object presence score
    """

    IMAGE_SIZE = 1008
    EMBED_SIZE = 72  # 1008 / 14
    MASK_SIZE = 288  # 72 * 4

    def __init__(
        self,
        sam3_tracker_model,
        multimask_output: bool = True,
    ):
        super().__init__()
        self.prompt_encoder = sam3_tracker_model.prompt_encoder
        self.mask_decoder = sam3_tracker_model.mask_decoder
        self.multimask_output = multimask_output

        # Register no_memory_embedding as buffer - reshape for broadcasting [1, 256, 1, 1]
        # Original shape is [1, 1, 256], we need [1, 256, 1, 1] for [B, C, H, W] addition
        no_mem_data = sam3_tracker_model.no_memory_embedding.data.clone().view(1, -1, 1, 1)
        self.register_buffer("no_mem_embed", no_mem_data)

        # Get image-wide positional embeddings (stored for ONNX export)
        image_pe = sam3_tracker_model.get_image_wide_positional_embeddings()  # [1, 256, 72, 72]
        self.register_buffer("image_pe", image_pe)

        # Get conv_s0 and conv_s1 from mask decoder
        # These project 256ch → 32ch and 256ch → 64ch
        self.conv_s0 = self.mask_decoder.conv_s0
        self.conv_s1 = self.mask_decoder.conv_s1

    def forward(
        self,
        fpn_feat_0: torch.Tensor,      # [B, 256, 288, 288]
        fpn_feat_1: torch.Tensor,      # [B, 256, 144, 144]
        fpn_feat_2: torch.Tensor,      # [B, 256, 72, 72]
        point_coords: torch.Tensor,    # [B, num_objects, num_points, 2]
        point_labels: torch.Tensor,    # [B, num_objects, num_points]
        mask_input: torch.Tensor,      # [B, 1, 288, 288]
        has_mask_input: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = fpn_feat_2.shape[0]

        # Step 1: Apply channel projections
        high_res_feats_0 = self.conv_s0(fpn_feat_0)  # [B, 32, 288, 288]
        high_res_feats_1 = self.conv_s1(fpn_feat_1)  # [B, 64, 144, 144]

        # Step 2: Add no_mem_embed to main embedding (broadcasts over spatial dims)
        image_embed = fpn_feat_2 + self.no_mem_embed  # [B, 256, 72, 72]

        # Step 3: Encode point prompts using HuggingFace interface
        # HuggingFace expects: input_points [B, num_objects, num_points, 2]
        #                      input_labels [B, num_objects, num_points]
        sparse_embeddings, dense_embeddings_no_mask = self.prompt_encoder(
            input_points=point_coords,
            input_labels=point_labels.long(),  # Labels must be long/int
            input_boxes=None,
            input_masks=None,
        )

        # Step 4: Handle mask input for refinement
        # When has_mask_input is True, encode the mask; otherwise use no_mask_embed
        mask_embed = self.prompt_encoder.mask_embed(mask_input)  # [B, 256, 72, 72]
        no_mask_embed = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        no_mask_embed = no_mask_embed.expand(B, -1, self.EMBED_SIZE, self.EMBED_SIZE)

        # Select based on has_mask_input
        has_mask = has_mask_input.view(-1, 1, 1, 1).float()
        dense_embeddings = has_mask * mask_embed + (1 - has_mask) * no_mask_embed

        # Step 5: Get position encoding (repeat for batch size)
        image_pe = self.image_pe.expand(B, -1, -1, -1)

        # Step 6: Run mask decoder
        high_res_features = [high_res_feats_0, high_res_feats_1]

        # HuggingFace mask decoder outputs: [B, num_objects, num_masks, H, W]
        low_res_multimasks, iou_pred, _, object_score_logits = self.mask_decoder(
            image_embeddings=image_embed,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            high_resolution_features=high_res_features,
        )

        # Squeeze out the num_objects dimension (we only support 1 object for simplicity)
        # Input: [B, 1, 3, 288, 288] -> Output: [B, 3, 288, 288]
        low_res_masks = low_res_multimasks.squeeze(1)  # [B, num_masks, H, W]
        iou_predictions = iou_pred.squeeze(1)  # [B, num_masks]
        object_score_logits = object_score_logits.squeeze(1)  # [B, 1]

        # Step 8: Upsample to image size
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks, iou_predictions, low_res_masks, object_score_logits


# =============================================================================
# Text Encoder Wrapper (for PCS mode)
# =============================================================================

class TextEncoderWrapper(nn.Module):
    """
    Wrapper for SAM3 text encoder.

    Inputs:
        input_ids: [B, seq_len] - tokenized text
        attention_mask: [B, seq_len] - attention mask

    Outputs:
        text_features: [B, seq_len, 256] - projected text features
        text_mask: [B, seq_len] - boolean mask
    """

    def __init__(self, sam3_model):
        super().__init__()
        self.text_encoder = sam3_model.text_encoder
        self.text_projection = sam3_model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_features = self.text_projection(text_features)
        text_mask = attention_mask > 0
        return text_features, text_mask


# =============================================================================
# PCS Decoder Wrapper (for text-to-segment detection)
# =============================================================================

class PCSDecoderWrapper(nn.Module):
    """
    SAM3 PCS Decoder for text-to-segment detection.

    Includes geometry encoder, DETR encoder/decoder, and mask head.

    Inputs:
        fpn_feat_0: [B, 256, 288, 288]
        fpn_feat_1: [B, 256, 144, 144]
        fpn_feat_2: [B, 256, 72, 72]
        fpn_pos_2: [B, 256, 72, 72]
        text_features: [B, seq_len, 256]
        text_mask: [B, seq_len]
        input_boxes: [B, num_boxes, 4] - optional geometry prompts (cxcywh normalized)
        input_boxes_labels: [B, num_boxes] - box labels (1=positive, 0=negative, -10=padding)

    Outputs:
        pred_masks: [B, 200, H, W] - predicted masks
        pred_boxes: [B, 200, 4] - predicted boxes (xyxy normalized)
        pred_logits: [B, 200] - classification logits
        presence_logits: [B, 200] - object presence logits
    """

    def __init__(self, sam3_model):
        super().__init__()
        self.geometry_encoder = sam3_model.geometry_encoder
        self.detr_encoder = sam3_model.detr_encoder
        self.detr_decoder = sam3_model.detr_decoder
        self.mask_decoder = sam3_model.mask_decoder
        self.dot_product_scoring = sam3_model.dot_product_scoring
        self.box_head = sam3_model.detr_decoder.box_head

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
    ):
        # Encode geometry prompts
        geometry_features, geometry_mask = self._encode_geometry(
            input_boxes, input_boxes_labels, fpn_feat_2, fpn_pos_2
        )

        # Combine text and geometry prompts
        prompt_features = torch.cat([text_features, geometry_features], dim=1)
        prompt_mask = torch.cat([text_mask, geometry_mask], dim=1)

        # Run DETR encoder
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=prompt_mask,
        )

        # Run DETR decoder
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoder_outputs.spatial_shapes,
        )

        # Compute boxes
        all_box_offsets = self.box_head(decoder_outputs.intermediate_hidden_states)
        reference_boxes_inv_sig = self._inverse_sigmoid(decoder_outputs.reference_boxes)
        all_pred_boxes = self._box_cxcywh_to_xyxy(
            (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        )

        # Compute logits via dot product scoring
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
            text_features=encoder_outputs.text_features,
            text_mask=prompt_mask,
        ).squeeze(-1)

        # Get final layer outputs
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]

        # Compute masks
        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=[fpn_feat_0, fpn_feat_1, fpn_feat_2],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )

        return mask_outputs.pred_masks, pred_boxes, pred_logits, presence_logits

    def _encode_geometry(self, input_boxes, input_boxes_labels, fpn_feat, fpn_pos):
        """Encode geometry (box) prompts."""
        import torchvision
        from transformers.masking_utils import create_bidirectional_mask

        batch_size, num_boxes = input_boxes.shape[:2]
        device = input_boxes.device

        ge = self.geometry_encoder

        box_mask = input_boxes_labels != -10
        box_labels = torch.where(
            input_boxes_labels == -10,
            torch.zeros_like(input_boxes_labels),
            input_boxes_labels,
        )

        vision_feats_flat = fpn_feat.flatten(2).transpose(1, 2)
        vision_pos_embeds_flat = fpn_pos.flatten(2).transpose(1, 2)

        img_feats_last = fpn_feat.permute(0, 2, 3, 1)
        normalized_img_feats = ge.vision_layer_norm(img_feats_last).permute(0, 3, 1, 2)

        height, width = normalized_img_feats.shape[-2:]
        boxes_embed = ge.boxes_direct_project(input_boxes)

        boxes_xyxy = self._box_cxcywh_to_xyxy(input_boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=device)
        boxes_xyxy = boxes_xyxy * scale.view(1, 1, 4)

        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1)
            .expand(-1, num_boxes)
            .reshape(-1, 1)
            .float()
        )
        boxes_flat = boxes_xyxy.view(-1, 4)
        boxes_with_batch = torch.cat([batch_indices, boxes_flat], dim=1)

        dtype = torch.float16 if normalized_img_feats.dtype == torch.bfloat16 else normalized_img_feats.dtype
        sampled_features = torchvision.ops.roi_align(
            normalized_img_feats.to(dtype),
            boxes_with_batch.to(dtype),
            ge.roi_size,
            sampling_ratio=0,
        ).to(normalized_img_feats.dtype)

        pooled_projection = ge.boxes_pool_project(sampled_features).view(
            batch_size, num_boxes, ge.hidden_size
        )
        boxes_embed = boxes_embed + pooled_projection

        center_x, center_y = input_boxes[:, :, 0], input_boxes[:, :, 1]
        box_width, box_height = input_boxes[:, :, 2], input_boxes[:, :, 3]
        pos_enc = ge._encode_box_coordinates(
            center_x.flatten(), center_y.flatten(),
            box_width.flatten(), box_height.flatten(),
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        boxes_embed = boxes_embed + ge.boxes_pos_enc_project(pos_enc)

        label_embed = ge.label_embed(box_labels.long())
        prompt_embeds = label_embed + boxes_embed
        prompt_mask = box_mask

        cls_embed = ge.cls_embed.weight.view(1, 1, ge.hidden_size).expand(batch_size, -1, -1)
        cls_mask_internal = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        cls_mask_out = box_mask.any(dim=1, keepdim=True)

        prompt_embeds = torch.cat([prompt_embeds, cls_embed], dim=1)
        prompt_mask_internal = torch.cat([prompt_mask, cls_mask_internal], dim=1)
        prompt_mask = torch.cat([prompt_mask, cls_mask_out], dim=1)

        prompt_embeds = ge.prompt_layer_norm(ge.final_proj(prompt_embeds))

        prompt_attention_mask = create_bidirectional_mask(
            config=ge.config,
            input_embeds=prompt_embeds,
            attention_mask=prompt_mask_internal,
        )

        for layer in ge.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        return ge.output_layer_norm(prompt_embeds), prompt_mask

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
        )


# =============================================================================
# Export Functions
# =============================================================================

def export_vision_encoder(
    model,
    output_dir: Path,
    device: str = "cuda",
    image_height: int = 1008,
    image_width: int = 1008,
    opset_version: int = 17,
):
    """Export vision encoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Vision Encoder")
    print(f"{'='*60}")

    wrapper = VisionEncoderWrapper(
        model, device=device, image_height=image_height, image_width=image_width
    ).to(device).eval()

    output_path = output_dir / "vision-encoder.onnx"

    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)

    print(f"  Input shape: {dummy_input.shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["images"],
            output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "images": {0: "batch"},
                "fpn_feat_0": {0: "batch"},
                "fpn_feat_1": {0: "batch"},
                "fpn_feat_2": {0: "batch"},
                "fpn_pos_2": {0: "batch"},
            },
        )

    print(f"  Exported: {output_path}")
    print(f"  Outputs: fpn_feat_0 [B,256,288,288], fpn_feat_1 [B,256,144,144], fpn_feat_2 [B,256,72,72], fpn_pos_2 [B,256,72,72]")

    return output_path


def export_tracker_decoder(
    tracker_model,
    output_dir: Path,
    device: str = "cuda",
    opset_version: int = 17,
):
    """Export tracker decoder (for point/box prompts) to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Tracker Decoder")
    print(f"{'='*60}")

    # For HuggingFace Sam3TrackerModel, pass the whole model
    wrapper = TrackerDecoderWrapper(
        sam3_tracker_model=tracker_model,
        multimask_output=True,
    ).to(device).eval()

    output_path = output_dir / "tracker-decoder.onnx"

    # Dummy inputs - HuggingFace uses [B, num_objects, num_points, 2] for points
    batch_size = 1
    num_objects = 1
    num_points = 2

    dummy_inputs = {
        "fpn_feat_0": torch.randn(batch_size, 256, 288, 288, device=device),
        "fpn_feat_1": torch.randn(batch_size, 256, 144, 144, device=device),
        "fpn_feat_2": torch.randn(batch_size, 256, 72, 72, device=device),
        "point_coords": torch.randint(0, 1008, (batch_size, num_objects, num_points, 2), dtype=torch.float32, device=device),
        "point_labels": torch.ones(batch_size, num_objects, num_points, dtype=torch.float32, device=device),
        "mask_input": torch.zeros(batch_size, 1, 288, 288, dtype=torch.float32, device=device),
        "has_mask_input": torch.zeros(batch_size, dtype=torch.float32, device=device),
    }

    print(f"  Input shapes:")
    for name, tensor in dummy_inputs.items():
        print(f"    {name}: {list(tensor.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs.values()),
            str(output_path),
            input_names=list(dummy_inputs.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks", "object_score_logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "fpn_feat_0": {0: "batch"},
                "fpn_feat_1": {0: "batch"},
                "fpn_feat_2": {0: "batch"},
                "point_coords": {0: "batch", 1: "num_objects", 2: "num_points"},
                "point_labels": {0: "batch", 1: "num_objects", 2: "num_points"},
                "mask_input": {0: "batch"},
                "has_mask_input": {0: "batch"},
                "masks": {0: "batch"},
                "iou_predictions": {0: "batch"},
                "low_res_masks": {0: "batch"},
                "object_score_logits": {0: "batch"},
            },
        )

    print(f"  Exported: {output_path}")
    print(f"  Outputs: masks [B,3,1008,1008], iou_predictions [B,3], low_res_masks [B,3,288,288], object_score_logits [B,1]")

    return output_path


def export_text_encoder(
    model,
    output_dir: Path,
    device: str = "cuda",
    opset_version: int = 17,
):
    """Export text encoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Text Encoder")
    print(f"{'='*60}")

    wrapper = TextEncoderWrapper(model).to(device).eval()

    output_path = output_dir / "text-encoder.onnx"

    seq_len = 32
    dummy_inputs = (
        torch.randint(0, 49408, (1, seq_len), device=device, dtype=torch.long),
        torch.ones(1, seq_len, dtype=torch.long, device=device),
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["text_features", "text_mask"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "text_features": {0: "batch"},
                "text_mask": {0: "batch"},
            },
        )

    print(f"  Exported: {output_path}")
    print(f"  Outputs: text_features [B,32,256], text_mask [B,32]")

    return output_path


def export_pcs_decoder(
    model,
    output_dir: Path,
    device: str = "cuda",
    opset_version: int = 17,
):
    """Export PCS decoder (for text-to-segment) to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting PCS Decoder")
    print(f"{'='*60}")

    wrapper = PCSDecoderWrapper(model).to(device).eval()

    output_path = output_dir / "pcs-decoder.onnx"

    # Dummy inputs
    dummy_inputs = (
        torch.randn(1, 256, 288, 288, device=device),  # fpn_feat_0
        torch.randn(1, 256, 144, 144, device=device),  # fpn_feat_1
        torch.randn(1, 256, 72, 72, device=device),    # fpn_feat_2
        torch.randn(1, 256, 72, 72, device=device),    # fpn_pos_2
        torch.randn(1, 32, 256, device=device),        # text_features
        torch.ones(1, 32, dtype=torch.bool, device=device),  # text_mask
        torch.rand(1, 1, 4, device=device),            # input_boxes (padding)
        torch.full((1, 1), -10, dtype=torch.long, device=device),  # input_boxes_labels (padding)
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=[
                "fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2",
                "text_features", "text_mask",
                "input_boxes", "input_boxes_labels",
            ],
            output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "fpn_feat_0": {0: "batch"},
                "fpn_feat_1": {0: "batch"},
                "fpn_feat_2": {0: "batch"},
                "fpn_pos_2": {0: "batch"},
                "text_features": {0: "batch"},
                "text_mask": {0: "batch"},
                "input_boxes": {0: "batch", 1: "num_boxes"},
                "input_boxes_labels": {0: "batch", 1: "num_boxes"},
                "pred_masks": {0: "batch"},
                "pred_boxes": {0: "batch"},
                "pred_logits": {0: "batch"},
                "presence_logits": {0: "batch"},
            },
        )

    print(f"  Exported: {output_path}")
    print(f"  Outputs: pred_masks [B,200,H,W], pred_boxes [B,200,4], pred_logits [B,200], presence_logits [B,200]")

    return output_path


# =============================================================================
# Verification Functions
# =============================================================================

def verify_vision_encoder(
    onnx_path: Path,
    hf_model,
    device: str = "cuda",
):
    """Verify ONNX vision encoder matches HuggingFace PyTorch."""
    import onnxruntime as ort

    print(f"\n{'='*60}")
    print("Verifying Vision Encoder")
    print(f"{'='*60}")

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    # Create wrapper for PyTorch
    wrapper = VisionEncoderWrapper(hf_model, device=device).to(device).eval()

    # Test with random input
    test_input = torch.randn(1, 3, 1008, 1008, device=device)

    # Run PyTorch
    with torch.no_grad():
        pt_outputs = wrapper(test_input)

    # Run ONNX
    onnx_input = test_input.cpu().numpy()
    onnx_outputs = session.run(None, {"images": onnx_input})

    # Compare
    for i, (pt_out, onnx_out) in enumerate(zip(pt_outputs, onnx_outputs)):
        pt_np = pt_out.cpu().numpy()
        mae = np.abs(pt_np - onnx_out).mean()
        max_diff = np.abs(pt_np - onnx_out).max()
        print(f"  Output {i}: MAE={mae:.6f}, MaxDiff={max_diff:.6f}")

        if mae > 0.001:
            print(f"    ⚠️ WARNING: MAE > 0.001")
        else:
            print(f"    ✓ PASS")

    return True


def verify_tracker_decoder(
    onnx_path: Path,
    tracker_model,
    device: str = "cuda",
):
    """Verify ONNX tracker decoder matches PyTorch."""
    import onnxruntime as ort

    print(f"\n{'='*60}")
    print("Verifying Tracker Decoder")
    print(f"{'='*60}")

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    # Create wrapper using HuggingFace Sam3TrackerModel
    wrapper = TrackerDecoderWrapper(
        sam3_tracker_model=tracker_model,
        multimask_output=True,
    ).to(device).eval()

    # Test inputs - HuggingFace uses [B, num_objects, num_points, 2] for points
    test_inputs = {
        "fpn_feat_0": torch.randn(1, 256, 288, 288, device=device),
        "fpn_feat_1": torch.randn(1, 256, 144, 144, device=device),
        "fpn_feat_2": torch.randn(1, 256, 72, 72, device=device),
        "point_coords": torch.tensor([[[[500.0, 500.0]]]], device=device),  # [B, num_objects, num_points, 2]
        "point_labels": torch.tensor([[[1.0]]], device=device),  # [B, num_objects, num_points]
        "mask_input": torch.zeros(1, 1, 288, 288, device=device),
        "has_mask_input": torch.tensor([0.0], device=device),
    }

    # Run PyTorch
    with torch.no_grad():
        pt_outputs = wrapper(**test_inputs)

    # Run ONNX
    onnx_inputs = {k: v.cpu().numpy() for k, v in test_inputs.items()}
    onnx_outputs = session.run(None, onnx_inputs)

    # Compare
    output_names = ["masks", "iou_predictions", "low_res_masks", "object_score_logits"]
    for i, (pt_out, onnx_out, name) in enumerate(zip(pt_outputs, onnx_outputs, output_names)):
        pt_np = pt_out.cpu().numpy()
        mae = np.abs(pt_np - onnx_out).mean()
        max_diff = np.abs(pt_np - onnx_out).max()
        print(f"  {name}: MAE={mae:.6f}, MaxDiff={max_diff:.6f}")

        if mae > 0.001:
            print(f"    ⚠️ WARNING: MAE > 0.001")
        else:
            print(f"    ✓ PASS")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export SAM3 ONNX models from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Export all components")
    parser.add_argument("--vision-encoder", action="store_true", help="Export vision encoder")
    parser.add_argument("--tracker-decoder", action="store_true", help="Export tracker decoder")
    parser.add_argument("--text-encoder", action="store_true", help="Export text encoder")
    parser.add_argument("--pcs-decoder", action="store_true", help="Export PCS decoder")
    parser.add_argument("--verify", action="store_true", help="Verify exports match PyTorch")
    parser.add_argument("--output-dir", type=str, default="./onnx-exports", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--model-path", type=str, default="facebook/sam3", help="HuggingFace model path")
    args = parser.parse_args()

    if not any([args.all, args.vision_encoder, args.tracker_decoder, args.text_encoder, args.pcs_decoder, args.verify]):
        parser.error("Please specify at least one export option or --verify")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Model Loading Strategy:
    # - Sam3TrackerModel: For vision encoder and tracker decoder (PVS mode)
    # - Sam3Model: For text encoder and PCS decoder (PCS mode)
    #
    # These are DIFFERENT models with DIFFERENT weights!
    # Sam3TrackerModel has 685 params, Sam3Model has 1468 params
    # =========================================================================

    tracker_model = None
    pcs_model = None

    # Load Sam3TrackerModel for vision encoder and tracker decoder
    if args.all or args.vision_encoder or args.tracker_decoder or args.verify:
        print(f"\nLoading HuggingFace Sam3TrackerModel from {args.model_path}...")
        from transformers import Sam3TrackerModel
        tracker_model = Sam3TrackerModel.from_pretrained(args.model_path).to(device).eval()
        print(f"  Sam3TrackerModel loaded: {sum(p.numel() for p in tracker_model.parameters())} parameters")

    # Load Sam3Model for text encoder and PCS decoder
    if args.all or args.text_encoder or args.pcs_decoder:
        print(f"\nLoading HuggingFace Sam3Model from {args.model_path}...")
        from transformers import Sam3Model
        pcs_model = Sam3Model.from_pretrained(args.model_path).to(device).eval()
        print(f"  Sam3Model loaded: {sum(p.numel() for p in pcs_model.parameters())} parameters")

    # Export components
    if args.all or args.vision_encoder:
        if tracker_model is None:
            print("ERROR: Sam3TrackerModel not loaded for vision encoder export")
        else:
            export_vision_encoder(tracker_model, output_dir, device)

    if args.all or args.tracker_decoder:
        if tracker_model is None:
            print("ERROR: Sam3TrackerModel not loaded for tracker decoder export")
        else:
            export_tracker_decoder(tracker_model, output_dir, device)

    if args.all or args.text_encoder:
        if pcs_model is None:
            print("ERROR: Sam3Model not loaded for text encoder export")
        else:
            export_text_encoder(pcs_model, output_dir, device)

    if args.all or args.pcs_decoder:
        if pcs_model is None:
            print("ERROR: Sam3Model not loaded for PCS decoder export")
        else:
            export_pcs_decoder(pcs_model, output_dir, device)

    # Verify
    if args.verify:
        if (output_dir / "vision-encoder.onnx").exists() and tracker_model:
            verify_vision_encoder(output_dir / "vision-encoder.onnx", tracker_model, device)

        if (output_dir / "tracker-decoder.onnx").exists() and tracker_model:
            verify_tracker_decoder(output_dir / "tracker-decoder.onnx", tracker_model, device)

    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    for f in output_dir.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
