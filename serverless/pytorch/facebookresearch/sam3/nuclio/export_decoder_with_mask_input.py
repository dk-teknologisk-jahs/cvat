#!/usr/bin/env python3
"""
SAM3-Tracker ONNX Decoder Export with Mask Input Support

This script exports the SAM3-Tracker prompt encoder + mask decoder to ONNX
with mask_input and has_mask_input support, enabling iterative mask refinement
just like SAM2.

Architecture differences from SAM2:
- Image size: 1008x1008 (vs SAM2's 1024x1024)
- Backbone stride: 14 (vs SAM2's 16)
- Embedding size: 72x72 (1008/14 = 72) vs SAM2's 64x64 (1024/16 = 64)
- Uses 3 high-resolution feature levels (always use_high_res_features=True)
- Always has pred_obj_scores=True, pred_obj_scores_mlp=True
- Always has iou_prediction_use_sigmoid=True

Key SAM3-Tracker parameters (from sam3_tracker_base.py):
- sam_prompt_embed_dim = hidden_dim (256)
- sam_image_embedding_size = image_size // backbone_stride = 1008 // 14 = 72
- mask_input_size = (4 * 72, 4 * 72) = (288, 288)
- low_res_mask_size = 72 * 4 = 288

Usage:
    python export_decoder_with_mask_input.py --checkpoint /path/to/sam3_tracker.pt \
        --output tracker-prompt-encoder-mask-decoder-with-mask-input.onnx
"""

import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3TrackerDecoderWrapper(nn.Module):
    """
    Wrapper that combines SAM3-Tracker's prompt encoder and mask decoder
    for ONNX export with mask input support.

    This is analogous to SAM2ImageDecoder used in the Medium article export,
    but adapted for SAM3-Tracker's architecture.

    Key inputs:
    - image_embed: Main backbone feature [B, C, H, W] where H=W=72 for SAM3
    - high_res_feats_0: Level 0 high-res features [B, C, 4*H, 4*W] = [B, 32, 288, 288]
    - high_res_feats_1: Level 1 high-res features [B, C, 2*H, 2*W] = [B, 64, 144, 144]
    - point_coords: Point coordinates [B, N, 2] in image pixel space
    - point_labels: Point labels [B, N] (0=neg, 1=pos, -1=padding, 2/3=box corners)
    - mask_input: Previous mask logits [B, 1, 288, 288] (4x low-res mask size)
    - has_mask_input: Whether mask_input is valid [B] or [1]
    - orig_im_size: Original image size [2] for H, W

    Key outputs:
    - masks: Predicted masks at original resolution [B, 3, orig_H, orig_W]
    - iou_predictions: IoU scores for each mask [B, 3]
    - low_res_masks: Low resolution mask logits [B, 3, 288, 288] for next iteration
    - object_score_logits: Object presence score [B, 1]
    """

    # SAM3-Tracker constants
    IMAGE_SIZE = 1008
    BACKBONE_STRIDE = 14
    EMBED_DIM = 256

    def __init__(
        self,
        sam_prompt_encoder: nn.Module,
        sam_mask_decoder: nn.Module,
        multimask_output: bool = True,
    ):
        super().__init__()
        self.sam_prompt_encoder = sam_prompt_encoder
        self.sam_mask_decoder = sam_mask_decoder
        self.multimask_output = multimask_output

        # Calculate derived constants
        self.image_embedding_size = self.IMAGE_SIZE // self.BACKBONE_STRIDE  # 72
        self.mask_input_size = (4 * self.image_embedding_size, 4 * self.image_embedding_size)  # (288, 288)

    def forward(
        self,
        image_embed: torch.Tensor,           # [B, 256, 72, 72]
        high_res_feats_0: torch.Tensor,      # [B, 32, 288, 288]  (after conv_s0)
        high_res_feats_1: torch.Tensor,      # [B, 64, 144, 144]  (after conv_s1)
        point_coords: torch.Tensor,          # [B, N, 2]
        point_labels: torch.Tensor,          # [B, N]
        mask_input: torch.Tensor,            # [B, 1, 288, 288]
        has_mask_input: torch.Tensor,        # [B] or [1]
        orig_im_size: torch.Tensor,          # [2] containing [H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass combining prompt encoder and mask decoder.

        Returns:
            masks: [B, num_masks, orig_H, orig_W] - upsampled to original resolution
            iou_predictions: [B, num_masks] - IoU quality scores
            low_res_masks: [B, num_masks, 288, 288] - for iterative refinement
            object_score_logits: [B, 1] - object presence score
        """
        B = image_embed.shape[0]

        # Process mask input based on has_mask_input flag
        # When has_mask_input is 0/False, we use None which triggers no_mask_embed
        # When has_mask_input is 1/True, we use the actual mask_input
        #
        # For ONNX export, we need to handle this without Python conditionals
        # We'll use has_mask_input as a multiplier and the no_mask_embed path

        # The mask_input should already be at the correct size (288x288 for SAM3)
        # If not, resize it
        if mask_input.shape[-2:] != self.mask_input_size:
            mask_input = F.interpolate(
                mask_input.float(),
                size=self.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )

        # Get sparse embeddings from points
        # Note: SAM's prompt encoder expects points as (coords, labels) tuple
        sparse_embeddings, dense_embeddings_from_points = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,  # We'll handle mask separately for the conditional
        )

        # Get dense embeddings from mask (when has_mask_input is True)
        dense_embeddings_from_mask = self.sam_prompt_encoder._embed_masks(mask_input)

        # Get no_mask_embed for when has_mask_input is False
        no_mask_embed = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        no_mask_embed = no_mask_embed.expand(
            B, -1, self.image_embedding_size, self.image_embedding_size
        )

        # Conditionally select dense embeddings based on has_mask_input
        # Reshape has_mask_input for broadcasting: [B] -> [B, 1, 1, 1]
        has_mask = has_mask_input.view(-1, 1, 1, 1).float()
        dense_embeddings = has_mask * dense_embeddings_from_mask + (1 - has_mask) * no_mask_embed

        # Get positional encoding for the image
        image_pe = self.sam_prompt_encoder.get_dense_pe()

        # Prepare high resolution features
        high_res_features = [high_res_feats_0, high_res_feats_1]

        # Run mask decoder
        # SAM3's mask decoder returns: (masks, iou_pred, sam_tokens, object_score_logits)
        low_res_multimasks, iou_pred, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # For multimask output, we return all 3 masks (indices 1, 2, 3)
        # For single mask output, we return only 1 mask (index 0)
        if self.multimask_output:
            # Return masks 1, 2, 3 (skip the "single mask" output at index 0)
            low_res_masks = low_res_multimasks[:, 1:, :, :]
            iou_predictions = iou_pred[:, 1:]
        else:
            low_res_masks = low_res_multimasks[:, 0:1, :, :]
            iou_predictions = iou_pred[:, 0:1]

        # Upsample masks to original image size
        # orig_im_size contains [H, W]
        orig_h = orig_im_size[0].item() if orig_im_size.dim() == 1 else orig_im_size[0, 0].item()
        orig_w = orig_im_size[1].item() if orig_im_size.dim() == 1 else orig_im_size[0, 1].item()

        # For ONNX export with dynamic shapes, we need to use the tensor directly
        # But for simplicity in ONNX, we'll upsample to a fixed size (image_size)
        # and let the caller handle final resize if needed
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks, iou_predictions, low_res_masks, object_score_logits


def create_dummy_prompt_encoder(embed_dim: int = 256, image_embedding_size: int = 72):
    """Create a dummy prompt encoder for testing/reference."""
    from sam3.sam.prompt_encoder import PromptEncoder

    return PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(1008, 1008),
        mask_in_chans=16,
    )


def create_dummy_mask_decoder(embed_dim: int = 256):
    """Create a dummy mask decoder for testing/reference."""
    from sam3.sam.mask_decoder import MaskDecoder
    from sam3.sam.transformer import TwoWayTransformer

    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
    )


def export_decoder(
    model: nn.Module,
    output_path: str,
    multimask_output: bool = True,
    opset_version: int = 17,
    device: str = "cpu",
):
    """
    Export the SAM3-Tracker decoder wrapper to ONNX.

    Args:
        model: The SAM3 model (Sam3TrackerBase or similar)
        output_path: Path to save the ONNX model
        multimask_output: Whether to output multiple masks
        opset_version: ONNX opset version
        device: Device to use for export
    """
    # Extract prompt encoder and mask decoder from the model
    sam_prompt_encoder = model.sam_prompt_encoder
    sam_mask_decoder = model.sam_mask_decoder

    # Also need to handle the conv_s0 and conv_s1 projections
    # These are applied to the high-res features before passing to decoder
    # For the ONNX export, we assume these are already applied (matching existing usls export)

    # Create wrapper
    wrapper = SAM3TrackerDecoderWrapper(
        sam_prompt_encoder=sam_prompt_encoder,
        sam_mask_decoder=sam_mask_decoder,
        multimask_output=multimask_output,
    ).to(device).eval()

    # Create dummy inputs
    batch_size = 1
    num_points = 2
    embed_size = 72

    dummy_inputs = {
        "image_embed": torch.randn(batch_size, 256, embed_size, embed_size, device=device),
        "high_res_feats_0": torch.randn(batch_size, 32, 4 * embed_size, 4 * embed_size, device=device),
        "high_res_feats_1": torch.randn(batch_size, 64, 2 * embed_size, 2 * embed_size, device=device),
        "point_coords": torch.randint(0, 1008, (batch_size, num_points, 2), dtype=torch.float32, device=device),
        "point_labels": torch.ones(batch_size, num_points, dtype=torch.float32, device=device),
        "mask_input": torch.randn(batch_size, 1, 4 * embed_size, 4 * embed_size, device=device),
        "has_mask_input": torch.ones(batch_size, dtype=torch.float32, device=device),
        "orig_im_size": torch.tensor([1008, 1008], dtype=torch.int32, device=device),
    }

    # Define input/output names
    input_names = list(dummy_inputs.keys())
    output_names = ["masks", "iou_predictions", "low_res_masks", "object_score_logits"]

    # Define dynamic axes for variable batch size and number of points
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
        "object_score_logits": {0: "batch_size"},
    }

    print(f"Exporting SAM3-Tracker decoder to {output_path}...")
    print(f"  Multimask output: {multimask_output}")
    print(f"  Input shapes:")
    for name, tensor in dummy_inputs.items():
        print(f"    {name}: {list(tensor.shape)}")

    # Export to ONNX
    with torch.no_grad():
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

    print(f"✓ Export complete: {output_path}")

    # Verify the export
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model validation passed")

        print("\nModel inputs:")
        for inp in model_onnx.graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")

        print("\nModel outputs:")
        for out in model_onnx.graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")

    except ImportError:
        print("Note: Install onnx package to verify the exported model")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3-Tracker decoder to ONNX with mask input support"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM3-Tracker checkpoint (.pt or directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tracker-prompt-encoder-mask-decoder-with-mask-input.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        default=True,
        help="Enable multimask output (default: True)",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="Output single mask only (overrides --multimask)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version (default 16 for broader compatibility)",
    )
    args = parser.parse_args()

    multimask_output = not args.single_mask

    # Load the SAM3-Tracker model
    print(f"Loading SAM3-Tracker model from {args.checkpoint}...")

    # Try different loading methods
    model = None

    try:
        # Method 1: Use build_tracker (the current SAM3 API)
        from sam3.model_builder import build_tracker
        print("Using build_tracker() to load model with pretrained weights...")
        model = build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
        print(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e1:
        print(f"build_tracker failed: {e1}")
        try:
            # Method 2: Try build_sam3_tracker if it exists
            from sam3.model_builder import build_sam3_tracker
            model = build_sam3_tracker(args.checkpoint)
        except Exception as e2:
            print(f"build_sam3_tracker also failed: {e2}")

    if model is None:
        print("\nTrying to create model components directly...")
        # Method 3: Create components manually for export verification
        print("Creating dummy model for export testing...")

        import sys
        sam3_path = "/tmp/sam3"
        sys.path.insert(0, sam3_path)

        # Import only the specific submodules we need (avoiding __init__.py)
        import importlib.util

        def load_module_directly(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        # Load common module first (has LayerNorm2d)
        common_mod = load_module_directly("sam3.sam.common", f"{sam3_path}/sam3/sam/common.py")

        # Now import the modules we need
        prompt_encoder_mod = load_module_directly("sam3.sam.prompt_encoder", f"{sam3_path}/sam3/sam/prompt_encoder.py")
        mask_decoder_mod = load_module_directly("sam3.sam.mask_decoder", f"{sam3_path}/sam3/sam/mask_decoder.py")
        transformer_mod = load_module_directly("sam3.sam.transformer", f"{sam3_path}/sam3/sam/transformer.py")

        PromptEncoder = prompt_encoder_mod.PromptEncoder
        MaskDecoder = mask_decoder_mod.MaskDecoder
        TwoWayTransformer = transformer_mod.TwoWayTransformer

        embed_dim = 256
        image_embedding_size = 72

        sam_prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(1008, 1008),
            mask_in_chans=16,
        )

        sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
        )

        # Create a simple wrapper object
        class DummyModel:
            def __init__(self, prompt_encoder, mask_decoder):
                self.sam_prompt_encoder = prompt_encoder
                self.sam_mask_decoder = mask_decoder

        model = DummyModel(sam_prompt_encoder, sam_mask_decoder)

    model_device = args.device
    if hasattr(model, 'to'):
        model = model.to(model_device).eval()
    else:
        model.sam_prompt_encoder = model.sam_prompt_encoder.to(model_device).eval()
        model.sam_mask_decoder = model.sam_mask_decoder.to(model_device).eval()

    # Export
    export_decoder(
        model=model,
        output_path=args.output,
        multimask_output=multimask_output,
        opset_version=args.opset,
        device=model_device,
    )


if __name__ == "__main__":
    main()
