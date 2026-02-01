# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3-Tracker Model Handler using PyTorch

Handles the vision encoder for SAM3-Tracker using native PyTorch.
This is the recommended approach since the encoder cannot be exported to ONNX
due to complex number operations (view_as_complex) in the RoPE positional embeddings.

Returns embeddings that can be decoded by the browser-side ONNX decoder.

Outputs:
  - high_res_feats_0: [batch, 32, 288, 288] FLOAT32
  - high_res_feats_1: [batch, 64, 144, 144] FLOAT32
  - image_embed: [batch, 256, 72, 72] FLOAT32

Usage:
    from model_handler_pytorch import ModelHandler
    handler = ModelHandler(device='cuda')
    emb0, emb1, emb2 = handler.handle(pil_image)
"""

import os
import numpy as np
import torch
from PIL import Image

# SAM3 Constants
SAM3_IMAGE_SIZE = 1008
SAM3_BACKBONE_STRIDE = 14
SAM3_EMBED_SIZE = SAM3_IMAGE_SIZE // SAM3_BACKBONE_STRIDE  # 72

# Feature sizes (high-res to low-res)
SAM3_FEAT_SIZES = [
    (288, 288),  # High-res feat 0: 32 channels
    (144, 144),  # High-res feat 1: 64 channels
    (72, 72),    # Main embedding: 256 channels
]


class ModelHandler:
    """SAM3-Tracker Vision Encoder using PyTorch."""

    def __init__(self, device: str = None):
        """
        Initialize the SAM3-Tracker model.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Build the SAM3 model with proper checkpoint loading
        # We use build_sam3_image_model with enable_inst_interactivity=True
        # This creates a model with:
        # - model.backbone: the full visual-language backbone
        # - model.inst_interactive_predictor: the SAM3InteractiveImagePredictor
        # - model.inst_interactive_predictor.model: the tracker (without backbone)
        from sam3.model_builder import build_sam3_image_model
        print(f"Loading SAM3-Tracker model on {device}...")
        
        self.sam3_model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=False,  # We only need the tracker/SAM parts
            enable_inst_interactivity=True,  # This creates the tracker with loaded weights
        )
        
        # Get the backbone from the main model for image encoding
        self.backbone = self.sam3_model.backbone
        
        # Get the tracker for feature preparation and no_mem_embed
        self.tracker = self.sam3_model.inst_interactive_predictor.model
        self.tracker.eval()
        
        # Image preprocessing params
        self.image_size = SAM3_IMAGE_SIZE
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        
        # Feature sizes for reshaping
        self._bb_feat_sizes = SAM3_FEAT_SIZES
        
        print(f"SAM3-Tracker model loaded successfully")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for SAM3-Tracker encoder.

        Args:
            image: PIL Image (any size)

        Returns:
            Preprocessed tensor [1, 3, 1008, 1008]
        """
        # Resize to 1008x1008 (exact fit)
        image_resized = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        # Convert to numpy and normalize to 0-1
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Normalize with mean/std
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    @torch.no_grad()
    def encode(self, image: Image.Image) -> dict:
        """
        Encode image to SAM3 embeddings.
        
        This follows the same encoding path as SAM3TrackerBase.forward_image():
        1. Run backbone.forward_image() to get sam2_backbone_out
        2. Apply conv_s0/conv_s1 projections from the mask decoder
        3. Prepare features via _prepare_backbone_features()
        4. Add no_mem_embed to lowest-res features
        5. Reshape to [B, C, H, W] format

        Args:
            image: PIL Image

        Returns:
            Dictionary with embeddings (high_res_feats_0, high_res_feats_1, image_embed)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        B = input_tensor.shape[0]
        
        # Step 1: Get raw backbone features from the main model's backbone
        # This is what tracker.forward_image does: backbone.forward_image()["sam2_backbone_out"]
        full_backbone_out = self.backbone.forward_image(input_tensor)
        backbone_out = full_backbone_out["sam2_backbone_out"]
        
        # Step 2: Apply conv_s0/conv_s1 projections to the high-res features
        # This is done by tracker.forward_image() using sam_mask_decoder.conv_s0/conv_s1
        backbone_out["backbone_fpn"][0] = self.tracker.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.tracker.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        
        # Step 3: Prepare features (this flattens to [HW, B, C] format)
        _, vision_feats, _, _ = self.tracker._prepare_backbone_features(backbone_out)
        
        # Step 4: Add no_mem_embed to lowest resolution features
        vision_feats[-1] = vision_feats[-1] + self.tracker.no_mem_embed
        
        # Step 5: Reshape features: [HW, B, C] -> [B, C, H, W]
        # vision_feats is in low-res to high-res order, we need high-res to low-res
        feats = []
        for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1]):
            # feat shape: [HW, B, C] -> [B, C, H, W]
            feat = feat.permute(1, 2, 0).view(B, -1, *feat_size)
            feats.append(feat)
        feats = feats[::-1]  # Back to high-res to low-res order
        
        # feats[0]: [B, 32, 288, 288] - high_res_feats_0
        # feats[1]: [B, 64, 144, 144] - high_res_feats_1
        # feats[2]: [B, 256, 72, 72] - image_embed
        
        return {
            'high_res_feats_0': feats[0].cpu().numpy(),
            'high_res_feats_1': feats[1].cpu().numpy(),
            'image_embed': feats[2].cpu().numpy(),
        }

    def handle(self, image: Image.Image) -> tuple:
        """
        Handle image encoding (compatible interface with main.py).

        Args:
            image: PIL Image

        Returns:
            Tuple of (high_res_feats_0, high_res_feats_1, image_embed) as numpy arrays
        """
        embeddings = self.encode(image)
        return (
            embeddings['high_res_feats_0'],  # [1, 32, 288, 288]
            embeddings['high_res_feats_1'],  # [1, 64, 144, 144]
            embeddings['image_embed'],        # [1, 256, 72, 72]
        )


# Quick test when run directly
if __name__ == "__main__":
    import sys
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color=(128, 128, 128))
    
    # Test the handler
    device = sys.argv[1] if len(sys.argv) > 1 else None
    handler = ModelHandler(device=device)
    
    emb0, emb1, emb2 = handler.handle(test_image)
    
    print(f"high_res_feats_0 shape: {emb0.shape}")  # [1, 32, 288, 288]
    print(f"high_res_feats_1 shape: {emb1.shape}")  # [1, 64, 144, 144]
    print(f"image_embed shape: {emb2.shape}")       # [1, 256, 72, 72]
    
    print("\nEncoding test passed!")
