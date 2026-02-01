#!/usr/bin/env python3
"""
Export SAM3 Vision Encoder to ONNX.

This creates an ONNX model that:
1. Takes an input image [B, 3, 1008, 1008]
2. Produces FPN features and positional encodings for the decoder

Outputs:
- backbone_fpn_2: [B, 256, 72, 72] - lowest resolution FPN (for transformer)
- vision_pos_2: [B, 256, 72, 72] - positional encoding for level 2

Note: The full SAM3 backbone produces 3 FPN levels but for text-only mode,
we only need level 2 (72x72) for the transformer encoder.
Input resolution is 1008x1008 (processor resolution).
"""

import sys
import os
sys.path.insert(0, "/home/jahs/GitHub/cvat/sam3")

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import gc


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for SAM3 backbone that produces FPN features.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        """
        Forward pass for vision encoder.

        Args:
            image: [B, 3, 1008, 1008] - input image (preprocessed)

        Returns:
            backbone_fpn_2: [B, 256, 72, 72] - FPN level 2 features
            vision_pos_2: [B, 256, 72, 72] - positional encoding
        """
        # Run backbone
        backbone_out = self.backbone.forward_image(image)

        # Extract FPN features - we only need the last level for text-only mode
        # backbone_fpn has 3 levels: [288x288, 144x144, 72x72]
        backbone_fpn = backbone_out["backbone_fpn"]
        vision_pos_enc = backbone_out["vision_pos_enc"]

        # Return only level 2 (72x72) for text-only mode
        backbone_fpn_2 = backbone_fpn[-1]  # [B, 256, 72, 72]
        vision_pos_2 = vision_pos_enc[-1]  # [B, 256, 72, 72]

        return backbone_fpn_2, vision_pos_2


def export_vision_encoder():
    """Export the vision encoder to ONNX."""

    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading SAM3 model...")
    from model_handler_pcs import ModelHandlerPCS
    handler = ModelHandlerPCS()
    model = handler.model
    model.eval()

    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    print("Creating vision encoder wrapper...")
    backbone = model.backbone
    encoder = VisionEncoderWrapper(backbone)
    encoder.eval()
    encoder.cuda()

    # Create dummy input - 1008x1008 as per processor resolution
    batch_size = 1
    image = torch.randn(batch_size, 3, 1008, 1008).cuda()

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        backbone_fpn_2, vision_pos_2 = encoder(image)

    print(f"  backbone_fpn_2: {backbone_fpn_2.shape}")
    print(f"  vision_pos_2: {vision_pos_2.shape}")

    # Move to CPU for export (to avoid GPU memory issues)
    print("\nMoving model to CPU for export...")
    backbone = backbone.cpu()
    encoder = VisionEncoderWrapper(backbone)
    encoder.eval()
    image = image.cpu()

    torch.cuda.empty_cache()
    gc.collect()

    output_path = "/home/jahs/GitHub/cvat/serverless/pytorch/facebookresearch/sam3/nuclio/pcs_vision_encoder.onnx"

    print(f"\nExporting to {output_path}...")
    print("  (This may take a few minutes due to large model size)")

    torch.onnx.export(
        encoder,
        (image,),
        output_path,
        input_names=["image"],
        output_names=["backbone_fpn_2", "vision_pos_2"],
        dynamic_axes={
            "image": {0: "batch"},
            "backbone_fpn_2": {0: "batch"},
            "vision_pos_2": {0: "batch"},
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
    print("  Inputs:", [i.name for i in sess.get_inputs()])
    print("  Outputs:", [o.name for o in sess.get_outputs()])

    # Run inference
    inputs = {"image": image.numpy()}
    ort_outputs = sess.run(None, inputs)

    print("\n  ONNX Runtime outputs:")
    print(f"    backbone_fpn_2: {ort_outputs[0].shape}")
    print(f"    vision_pos_2: {ort_outputs[1].shape}")

    # Compare with PyTorch
    print("\nComparing PyTorch vs ONNX Runtime...")
    with torch.no_grad():
        pt_fpn_2, pt_pos_2 = encoder(image)

    pt_fpn_2_np = pt_fpn_2.numpy()
    pt_pos_2_np = pt_pos_2.numpy()

    fpn_diff = np.abs(ort_outputs[0] - pt_fpn_2_np).max()
    pos_diff = np.abs(ort_outputs[1] - pt_pos_2_np).max()

    print(f"  Max diff backbone_fpn_2: {fpn_diff:.6f}")
    print(f"  Max diff vision_pos_2: {pos_diff:.6f}")

    if fpn_diff < 0.01 and pos_diff < 0.01:
        print("\n✓ ONNX export successful! Outputs match PyTorch within tolerance.")
    else:
        print("\n⚠ Warning: Outputs differ more than expected. Check numerical precision.")

    return output_path


if __name__ == "__main__":
    export_vision_encoder()
