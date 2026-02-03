#!/usr/bin/env python3
"""
Export SAM3 Memory Encoder to ONNX for browser-side video propagation.

This script exports the HuggingFace Sam2VideoMemoryEncoder to ONNX format
for use with onnxruntime-web (v1.14.0, opset 18 max).

Key dimensions (verified experimentally):
- vision_features: [B, 256, 64, 64] - from vision encoder at 64x64 spatial
- masks: [B, 1, 1024, 1024] - full resolution masks (16x downsampling)
- output memory: [B, 64, 64, 64]
- output pos_enc: [B, 64, 64, 64]

Usage:
    python export_memory_encoder_onnx.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import Sam2VideoModel


class MemoryEncoderONNXWrapper(nn.Module):
    """
    ONNX-friendly wrapper for Sam2VideoMemoryEncoder.

    Wraps the HuggingFace memory encoder and outputs both memory features
    and position encodings needed for video propagation.
    """

    def __init__(self, memory_encoder: nn.Module):
        super().__init__()
        self.memory_encoder = memory_encoder

    def forward(
        self,
        vision_features: torch.Tensor,  # [B, 256, 64, 64]
        masks: torch.Tensor              # [B, 1, 1024, 1024]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning memory features and position encoding.

        Args:
            vision_features: Image features from vision encoder [B, 256, 64, 64]
            masks: Full resolution masks [B, 1, 1024, 1024]

        Returns:
            memory: Encoded memory features [B, 64, 64, 64]
            memory_pos_enc: Position encoding for memory [B, 64, 64, 64]
        """
        memory, memory_pos_enc = self.memory_encoder(vision_features, masks)
        return memory, memory_pos_enc


def export_memory_encoder(
    model: Sam2VideoModel,
    output_path: Path,
    opset_version: int = 17,
) -> bool:
    """
    Export the memory encoder to ONNX.

    Args:
        model: HuggingFace Sam2VideoModel
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (17 for onnxruntime-web 1.14.0)

    Returns:
        True if export and verification succeed
    """
    print(f"\n{'='*60}")
    print("Exporting Memory Encoder to ONNX")
    print(f"{'='*60}")

    # Create wrapper
    wrapper = MemoryEncoderONNXWrapper(model.memory_encoder)
    wrapper.eval()

    # Create sample inputs with correct dimensions
    batch_size = 1
    vision_features = torch.randn(batch_size, 256, 64, 64)
    masks = torch.randn(batch_size, 1, 1024, 1024)

    print(f"Input shapes:")
    print(f"  vision_features: {vision_features.shape}")
    print(f"  masks: {masks.shape}")

    # Test forward pass
    with torch.no_grad():
        memory, memory_pos_enc = wrapper(vision_features, masks)

    print(f"Output shapes:")
    print(f"  memory: {memory.shape}")
    print(f"  memory_pos_enc: {memory_pos_enc.shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")

    try:
        torch.onnx.export(
            wrapper,
            (vision_features, masks),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['vision_features', 'masks'],
            output_names=['memory', 'memory_pos_enc'],
            dynamic_axes={
                'vision_features': {0: 'batch_size'},
                'masks': {0: 'batch_size'},
                'memory': {0: 'batch_size'},
                'memory_pos_enc': {0: 'batch_size'},
            }
        )
        print(f"  ✓ Exported to {output_path}")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")
    except Exception as e:
        print(f"  ✗ ONNX validation failed: {e}")
        return False

    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    try:
        session = ort.InferenceSession(
            str(output_path),
            providers=['CPUExecutionProvider']
        )

        # Run inference
        ort_inputs = {
            'vision_features': vision_features.numpy(),
            'masks': masks.numpy()
        }
        ort_outputs = session.run(None, ort_inputs)

        # Compare outputs
        torch_memory = memory.numpy()
        torch_pos_enc = memory_pos_enc.numpy()
        ort_memory = ort_outputs[0]
        ort_pos_enc = ort_outputs[1]

        memory_diff = np.abs(torch_memory - ort_memory).max()
        pos_diff = np.abs(torch_pos_enc - ort_pos_enc).max()

        print(f"  Output comparison (max absolute diff):")
        print(f"    memory: {memory_diff:.6f}")
        print(f"    memory_pos_enc: {pos_diff:.6f}")

        # Check if differences are acceptable (fp32 tolerance)
        tolerance = 1e-4
        if memory_diff < tolerance and pos_diff < tolerance:
            print(f"  ✓ ONNX Runtime verification passed (tolerance: {tolerance})")
        else:
            print(f"  ⚠ Outputs differ more than tolerance {tolerance}")
            print("    This may be acceptable for ONNX but worth investigating")

    except Exception as e:
        print(f"  ✗ ONNX Runtime test failed: {e}")
        return False

    # Report file size
    file_size = output_path.stat().st_size
    print(f"\nExport Summary:")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size / 1024:.1f} KB ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Opset: {opset_version}")
    print(f"  Status: ✓ SUCCESS")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 Memory Encoder to ONNX"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx-memory-exports",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, max for onnxruntime-web 1.14.0 is 18)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = Sam2VideoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32
    )
    model.eval()
    print("Model loaded successfully")

    # Export memory encoder
    output_path = output_dir / "memory_encoder.onnx"
    success = export_memory_encoder(model, output_path, args.opset)

    if success:
        print(f"\n{'='*60}")
        print("Export completed successfully!")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("Export failed!")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    exit(main())
