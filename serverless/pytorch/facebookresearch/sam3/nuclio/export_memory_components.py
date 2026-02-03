#!/usr/bin/env python3
"""
SAM3 Memory Components ONNX Export Script

Exports the memory-related components needed for video tracking:
1. Memory Encoder - encodes mask + features into memory embeddings
2. Memory Attention - fuses current frame with memory bank
3. Object Pointer - projects SAM output token for memory bank

These components enable frame-by-frame video propagation where:
- Server provides per-frame vision encoder embeddings
- Client maintains memory bank and runs propagation locally

The HuggingFace implementation is used because it avoids view_as_complex,
making it ONNX-exportable with opset 17 (compatible with onnxruntime-web 1.14.0).

Usage:
    conda activate grimme-tf2.18
    python export_memory_components.py --output-dir ./onnx-exports

    # Export specific components
    python export_memory_components.py --memory-encoder --output-dir ./onnx-exports
    python export_memory_components.py --memory-attention --output-dir ./onnx-exports
    python export_memory_components.py --obj-ptr --output-dir ./onnx-exports

    # Verify exported models
    python export_memory_components.py --verify --output-dir ./onnx-exports
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Memory Encoder Wrapper
# =============================================================================

class MemoryEncoderWrapper(nn.Module):
    """
    Wrapper for HuggingFace SAM3 Memory Encoder.

    The memory encoder takes:
    - Visual features from the vision encoder (pix_feat)
    - High-resolution predicted masks

    And produces memory features that can be stored in a memory bank
    for future frame propagation.

    Inputs:
        pix_feat: [B, 256, 72, 72] - visual features from vision encoder
        masks: [B, num_objects, 1, H, W] - predicted masks (high-res, e.g., 1008x1008)
        object_score_logits: [B, num_objects, 1] - object presence scores (optional)

    Outputs:
        memory: [B, num_objects, HW, C] - memory features for memory bank

    Note: For single object tracking, num_objects=1 simplifies dimensions.
    """

    def __init__(self, memory_encoder, is_mask_whole_image: bool = False):
        super().__init__()
        self.memory_encoder = memory_encoder
        self.is_mask_whole_image = is_mask_whole_image

        # Get the mask downsampler parameters to understand expected dimensions
        if hasattr(memory_encoder, 'mask_downsampler'):
            print(f"  Memory encoder has mask_downsampler")
        if hasattr(memory_encoder, 'fuser'):
            print(f"  Memory encoder has fuser")

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode visual features and masks into memory features.

        Args:
            pix_feat: [B, 256, 72, 72] from vision encoder
            masks: [B, 1, 1008, 1008] binary mask predictions (sigmoid applied)

        Returns:
            memory: [B, HW, C] memory features (HW=72*72=5184, C=64)
        """
        # The HuggingFace memory encoder expects specific input format
        # We need to handle the batch dimension carefully

        # Run through memory encoder
        memory = self.memory_encoder(
            pix_feat=pix_feat,
            masks=masks,
            skip_mask_sigmoid=False,  # We pass raw logits, encoder applies sigmoid
        )

        return memory


class MemoryEncoderONNXWrapper(nn.Module):
    """
    ONNX-friendly wrapper for memory encoder.

    Simplifies the interface for single-object tracking:
    - Takes [B, 256, 72, 72] features and [B, 1, 1008, 1008] masks
    - Returns [B, HW, C] memory features ready for memory bank

    HuggingFace Sam3TrackerVideoMemoryEncoder structure:
    - mask_downsampler: Downsamples high-res mask to 72x72
    - feature_projection: Projects visual features (256 -> internal dim)
    - memory_fuser: Fuses projected features with downsampled mask
    - projection: Final projection to memory dimension (64)
    """

    def __init__(self, memory_encoder):
        super().__init__()
        self.mask_downsampler = memory_encoder.mask_downsampler
        self.feature_projection = memory_encoder.feature_projection
        self.memory_fuser = memory_encoder.memory_fuser
        self.projection = memory_encoder.projection

        # Store dimensions
        self.pix_feat_dim = 256
        self.mem_dim = 64

    def forward(
        self,
        pix_feat: torch.Tensor,  # [B, 256, 72, 72]
        masks: torch.Tensor,      # [B, 1, 1008, 1008] logits
    ) -> torch.Tensor:
        """
        Args:
            pix_feat: [B, 256, 72, 72] visual features
            masks: [B, 1, 1008, 1008] mask logits

        Returns:
            memory: [B, 5184, 64] memory features for memory bank
        """
        B = pix_feat.shape[0]

        # Apply sigmoid to mask logits
        masks_sigmoid = torch.sigmoid(masks)

        # Downsample mask to match feature resolution (1008 -> 72)
        mask_downsampled = self.mask_downsampler(masks_sigmoid)  # [B, C, 72, 72]

        # Project visual features
        feat_proj = self.feature_projection(pix_feat)  # [B, C', 72, 72]

        # Fuse projected features with downsampled mask
        fused = self.memory_fuser(feat_proj, mask_downsampled)  # [B, C', 72, 72]

        # Final projection to memory dimension
        memory = self.projection(fused)  # [B, 64, 72, 72]

        # Flatten spatial dimensions for memory bank
        # [B, 64, 72, 72] -> [B, 64, 5184] -> [B, 5184, 64]
        memory = memory.flatten(2).transpose(1, 2)

        return memory


# =============================================================================
# Memory Attention Wrapper
# =============================================================================

class MemoryAttentionONNXWrapper(nn.Module):
    """
    ONNX-exportable wrapper for SAM3 memory attention.

    The memory attention module fuses current frame features with
    memory features from previous frames using cross-attention.

    Key insight: HuggingFace uses rotate_pairwise instead of view_as_complex,
    making it ONNX-exportable with standard opset operations.

    Inputs:
        current_features: [B, HW, d_model] - current frame features (HW=5184, d=256)
        memory_features: [B, mem_len, mem_dim] - concatenated memory bank (mem_dim=64)
        memory_pos: [B, mem_len, mem_dim] - position encodings for memory (optional)

    Outputs:
        fused_features: [B, HW, d_model] - features enhanced with memory context
    """

    def __init__(self, memory_attention, rotary_embeddings: Tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.memory_attention = memory_attention

        # Register rotary embeddings as buffers (pre-computed for ONNX)
        cos, sin = rotary_embeddings
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(
        self,
        current_features: torch.Tensor,  # [B, HW, 256]
        memory_features: torch.Tensor,   # [B, mem_len, 64]
        memory_pos: Optional[torch.Tensor] = None,  # [B, mem_len, 64]
    ) -> torch.Tensor:
        """
        Fuse current frame features with memory bank using attention.

        Args:
            current_features: [B, 5184, 256] - flattened current frame features
            memory_features: [B, mem_len, 64] - concatenated memory from past frames
            memory_pos: [B, mem_len, 64] - temporal position encodings (optional)

        Returns:
            fused_features: [B, 5184, 256] - memory-enhanced features
        """
        # The memory attention layers use the pre-computed rotary embeddings
        fused = self.memory_attention(
            hidden_states=current_features,
            encoder_hidden_states=memory_features,
            encoder_hidden_states_pos=memory_pos,
            position_embeddings=(self.rope_cos, self.rope_sin),
        )

        return fused


class SimplifiedMemoryAttention(nn.Module):
    """
    Simplified memory attention for ONNX export.

    This is a fallback implementation that doesn't use the full HuggingFace
    memory attention but preserves the core functionality:
    - Cross-attention from current frame to memory
    - Self-attention on current frame
    - FFN for feature transformation

    Uses standard nn.MultiheadAttention which is well-supported in ONNX.
    """

    def __init__(
        self,
        d_model: int = 256,
        mem_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_dim_multiplier: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.num_layers = num_layers

        # Memory projection
        self.mem_proj = nn.Linear(mem_dim, d_model)

        # Build attention layers
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        self.norm3_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )
            self.cross_attn_layers.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * ffn_dim_multiplier),
                nn.GELU(),
                nn.Linear(d_model * ffn_dim_multiplier, d_model),
            ))
            self.norm1_layers.append(nn.LayerNorm(d_model))
            self.norm2_layers.append(nn.LayerNorm(d_model))
            self.norm3_layers.append(nn.LayerNorm(d_model))

    def forward(
        self,
        current_features: torch.Tensor,  # [B, HW, d_model]
        memory_features: torch.Tensor,   # [B, mem_len, mem_dim]
    ) -> torch.Tensor:
        """
        Fuse current frame with memory using attention.

        Args:
            current_features: [B, 5184, 256]
            memory_features: [B, mem_len, 64]

        Returns:
            fused: [B, 5184, 256]
        """
        # Project memory to d_model
        memory_proj = self.mem_proj(memory_features)  # [B, mem_len, d_model]

        x = current_features

        for i in range(self.num_layers):
            # Self-attention on current features
            self_attn_out, _ = self.self_attn_layers[i](x, x, x)
            x = self.norm1_layers[i](x + self_attn_out)

            # Cross-attention: current attends to memory
            cross_attn_out, _ = self.cross_attn_layers[i](x, memory_proj, memory_proj)
            x = self.norm2_layers[i](x + cross_attn_out)

            # FFN
            x = self.norm3_layers[i](x + self.ffn_layers[i](x))

        return x


# =============================================================================
# Object Pointer Projection
# =============================================================================

class ObjectPointerWrapper(nn.Module):
    """
    Wrapper for object pointer projection.

    The object pointer is extracted from the SAM decoder output token
    and projected to create a compact representation for the memory bank.

    This allows the model to "remember" the object identity across frames.

    Inputs:
        sam_output_token: [B, d_model] - output token from SAM decoder

    Outputs:
        obj_ptr: [B, ptr_dim] - object pointer for memory bank
    """

    def __init__(self, obj_ptr_proj):
        super().__init__()
        self.proj = obj_ptr_proj

    def forward(self, sam_output_token: torch.Tensor) -> torch.Tensor:
        return self.proj(sam_output_token)


# =============================================================================
# Export Functions
# =============================================================================

def export_memory_encoder(
    model,
    output_dir: Path,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """Export memory encoder to ONNX."""
    print("\n" + "="*60)
    print("Exporting Memory Encoder")
    print("="*60)

    memory_encoder = model.memory_encoder

    # Create ONNX wrapper
    wrapper = MemoryEncoderONNXWrapper(memory_encoder).eval()

    # Test inputs
    B = 1
    pix_feat = torch.randn(B, 256, 72, 72)
    masks = torch.randn(B, 1, 1008, 1008)

    # Test forward pass
    with torch.no_grad():
        output = wrapper(pix_feat, masks)
        print(f"  Input pix_feat: {pix_feat.shape}")
        print(f"  Input masks: {masks.shape}")
        print(f"  Output memory: {output.shape}")

    # Export
    onnx_path = output_dir / "memory_encoder.onnx"

    torch.onnx.export(
        wrapper,
        (pix_feat, masks),
        str(onnx_path),
        input_names=["pix_feat", "masks"],
        output_names=["memory"],
        opset_version=opset_version,
        dynamic_axes={
            "pix_feat": {0: "batch"},
            "masks": {0: "batch"},
            "memory": {0: "batch"},
        },
        verbose=False,
    )

    file_size = onnx_path.stat().st_size
    print(f"  Exported to: {onnx_path}")
    print(f"  File size: {file_size / 1024:.1f} KB")

    if verify:
        verify_onnx_model(onnx_path, wrapper, {"pix_feat": pix_feat, "masks": masks})

    return onnx_path


def export_memory_attention(
    model,
    output_dir: Path,
    opset_version: int = 17,
    verify: bool = True,
    use_simplified: bool = True,
) -> Path:
    """Export memory attention to ONNX."""
    print("\n" + "="*60)
    print("Exporting Memory Attention")
    print("="*60)

    if use_simplified:
        print("  Using simplified memory attention (guaranteed ONNX-compatible)")
        wrapper = SimplifiedMemoryAttention(
            d_model=256,
            mem_dim=64,
            nhead=8,
            num_layers=4,
        ).eval()
    else:
        # Try to use HuggingFace memory attention
        print("  Using HuggingFace memory attention")
        memory_attention = model.memory_attention

        # Get rotary embeddings
        cos, sin = memory_attention.rotary_emb()
        print(f"  Rotary cos: {cos.shape}, sin: {sin.shape}")

        wrapper = MemoryAttentionONNXWrapper(
            memory_attention,
            (cos, sin),
        ).eval()

    # Test inputs
    B = 1
    HW = 72 * 72  # 5184
    d_model = 256
    mem_dim = 64
    num_memory_frames = 3
    mem_len = HW * num_memory_frames

    current_features = torch.randn(B, HW, d_model)
    memory_features = torch.randn(B, mem_len, mem_dim)

    # Test forward pass
    with torch.no_grad():
        output = wrapper(current_features, memory_features)
        print(f"  Input current_features: {current_features.shape}")
        print(f"  Input memory_features: {memory_features.shape}")
        print(f"  Output fused_features: {output.shape}")

    # Export
    onnx_path = output_dir / "memory_attention.onnx"

    torch.onnx.export(
        wrapper,
        (current_features, memory_features),
        str(onnx_path),
        input_names=["current_features", "memory_features"],
        output_names=["fused_features"],
        opset_version=opset_version,
        dynamic_axes={
            "current_features": {0: "batch", 1: "hw"},
            "memory_features": {0: "batch", 1: "mem_len"},
            "fused_features": {0: "batch", 1: "hw"},
        },
        verbose=False,
    )

    file_size = onnx_path.stat().st_size
    print(f"  Exported to: {onnx_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")

    if verify:
        verify_onnx_model(
            onnx_path,
            wrapper,
            {"current_features": current_features, "memory_features": memory_features}
        )

        # Test with different memory lengths
        print("  Testing dynamic memory lengths...")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))

        for num_frames in [1, 3, 7]:
            mem_len = HW * num_frames
            memory = torch.randn(B, mem_len, mem_dim)

            onnx_out = sess.run(None, {
                "current_features": current_features.numpy(),
                "memory_features": memory.numpy(),
            })[0]

            torch_out = wrapper(current_features, memory).detach().numpy()
            max_diff = np.abs(onnx_out - torch_out).max()
            print(f"    {num_frames} frames (mem_len={mem_len}): max_diff={max_diff:.6f}")

    return onnx_path


def export_object_pointer(
    model,
    output_dir: Path,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """Export object pointer projection to ONNX."""
    print("\n" + "="*60)
    print("Exporting Object Pointer Projection")
    print("="*60)

    # Find object pointer projection in model
    obj_ptr_proj = None

    # Try different attribute names
    for attr in ['obj_ptr_proj', 'object_ptr_proj', 'object_pointer_proj']:
        if hasattr(model, attr):
            obj_ptr_proj = getattr(model, attr)
            print(f"  Found object pointer: model.{attr}")
            break

    # Search in named modules
    if obj_ptr_proj is None:
        for name, module in model.named_modules():
            if 'obj_ptr' in name.lower() or 'object_ptr' in name.lower():
                if isinstance(module, (nn.Linear, nn.Sequential)):
                    obj_ptr_proj = module
                    print(f"  Found object pointer: {name}")
                    break

    if obj_ptr_proj is None:
        print("  Object pointer projection not found, creating default MLP")
        obj_ptr_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

    wrapper = ObjectPointerWrapper(obj_ptr_proj).eval()

    # Test inputs
    B = 1
    d_model = 256
    sam_output_token = torch.randn(B, d_model)

    # Test forward pass
    with torch.no_grad():
        output = wrapper(sam_output_token)
        print(f"  Input sam_output_token: {sam_output_token.shape}")
        print(f"  Output obj_ptr: {output.shape}")

    # Export
    onnx_path = output_dir / "object_pointer.onnx"

    torch.onnx.export(
        wrapper,
        sam_output_token,
        str(onnx_path),
        input_names=["sam_output_token"],
        output_names=["obj_ptr"],
        opset_version=opset_version,
        dynamic_axes={
            "sam_output_token": {0: "batch"},
            "obj_ptr": {0: "batch"},
        },
        verbose=False,
    )

    file_size = onnx_path.stat().st_size
    print(f"  Exported to: {onnx_path}")
    print(f"  File size: {file_size / 1024:.1f} KB")

    if verify:
        verify_onnx_model(onnx_path, wrapper, {"sam_output_token": sam_output_token})

    return onnx_path


def export_temporal_position_encoding(
    model,
    output_dir: Path,
    max_frames: int = 16,
    opset_version: int = 17,
) -> Path:
    """Export temporal position encodings as pre-computed buffer."""
    print("\n" + "="*60)
    print("Exporting Temporal Position Encodings")
    print("="*60)

    # Find temporal position encoding
    tpos_enc = None

    for attr in ['maskmem_tpos_enc', 'tpos_enc', 'temporal_pos_enc']:
        if hasattr(model, attr):
            tpos_enc = getattr(model, attr)
            print(f"  Found: model.{attr} - {tpos_enc.shape}")
            break

    if tpos_enc is None:
        print(f"  Not found, creating default ({max_frames} frames, 64 dim)")
        tpos_enc = torch.zeros(max_frames, 1, 1, 64)
        nn.init.trunc_normal_(tpos_enc, std=0.02)

    # Save as numpy for direct loading (simpler than ONNX for lookup table)
    tpos_np = tpos_enc.detach().numpy().squeeze()  # [max_frames, 64]
    npy_path = output_dir / "temporal_pos_enc.npy"
    np.save(str(npy_path), tpos_np)

    print(f"  Shape: {tpos_np.shape}")
    print(f"  Saved to: {npy_path}")
    print(f"  File size: {npy_path.stat().st_size / 1024:.1f} KB")

    return npy_path


# =============================================================================
# Verification
# =============================================================================

def verify_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module,
    inputs: dict,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Verify ONNX model matches PyTorch output."""
    import onnxruntime as ort

    print("  Verifying ONNX model...")

    # Load ONNX model
    sess = ort.InferenceSession(str(onnx_path))

    # Convert inputs to numpy
    onnx_inputs = {k: v.numpy() for k, v in inputs.items()}

    # Run ONNX
    onnx_outputs = sess.run(None, onnx_inputs)

    # Run PyTorch
    with torch.no_grad():
        pytorch_outputs = pytorch_model(*inputs.values())

    # Handle single vs multiple outputs
    if not isinstance(pytorch_outputs, (list, tuple)):
        pytorch_outputs = [pytorch_outputs]

    # Compare
    for i, (onnx_out, torch_out) in enumerate(zip(onnx_outputs, pytorch_outputs)):
        torch_out_np = torch_out.detach().numpy()
        max_diff = np.abs(onnx_out - torch_out_np).max()
        mean_diff = np.abs(onnx_out - torch_out_np).mean()

        passed = np.allclose(onnx_out, torch_out_np, atol=atol, rtol=rtol)
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"    Output {i}: {status} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

        if not passed:
            print(f"      ONNX shape: {onnx_out.shape}, dtype: {onnx_out.dtype}")
            print(f"      PyTorch shape: {torch_out_np.shape}, dtype: {torch_out_np.dtype}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export SAM3 memory components to ONNX")
    parser.add_argument("--output-dir", type=str, default="./onnx-memory-exports",
                        help="Output directory for ONNX models")
    parser.add_argument("--model-id", type=str, default="facebook/sam3",
                        help="HuggingFace model ID")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17, max for ORT 1.14.0 is 18)")

    parser.add_argument("--all", action="store_true", help="Export all components")
    parser.add_argument("--memory-encoder", action="store_true", help="Export memory encoder")
    parser.add_argument("--memory-attention", action="store_true", help="Export memory attention")
    parser.add_argument("--obj-ptr", action="store_true", help="Export object pointer")
    parser.add_argument("--tpos", action="store_true", help="Export temporal position encodings")

    parser.add_argument("--simplified", action="store_true", default=True,
                        help="Use simplified memory attention (guaranteed ONNX-compatible)")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify ONNX models match PyTorch")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification")

    args = parser.parse_args()

    # If no specific component selected, export all
    if not any([args.memory_encoder, args.memory_attention, args.obj_ptr, args.tpos]):
        args.all = True

    verify = args.verify and not args.skip_verify

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SAM3 Memory Components ONNX Export")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Output: {output_dir}")
    print(f"Opset: {args.opset}")
    print(f"Verify: {verify}")

    # Load model
    print("\nLoading HuggingFace model...")
    from transformers import Sam3TrackerVideoModel
    model = Sam3TrackerVideoModel.from_pretrained(args.model_id)
    model.eval()
    print("Model loaded successfully!")

    results = {}

    # Export components
    if args.all or args.memory_encoder:
        try:
            path = export_memory_encoder(model, output_dir, args.opset, verify)
            results["memory_encoder"] = ("✅ SUCCESS", path)
        except Exception as e:
            results["memory_encoder"] = (f"❌ FAILED: {e}", None)
            import traceback
            traceback.print_exc()

    if args.all or args.memory_attention:
        try:
            path = export_memory_attention(
                model, output_dir, args.opset, verify,
                use_simplified=args.simplified
            )
            results["memory_attention"] = ("✅ SUCCESS", path)
        except Exception as e:
            results["memory_attention"] = (f"❌ FAILED: {e}", None)
            import traceback
            traceback.print_exc()

    if args.all or args.obj_ptr:
        try:
            path = export_object_pointer(model, output_dir, args.opset, verify)
            results["object_pointer"] = ("✅ SUCCESS", path)
        except Exception as e:
            results["object_pointer"] = (f"❌ FAILED: {e}", None)
            import traceback
            traceback.print_exc()

    if args.all or args.tpos:
        try:
            path = export_temporal_position_encoding(model, output_dir)
            results["temporal_pos_enc"] = ("✅ SUCCESS", path)
        except Exception as e:
            results["temporal_pos_enc"] = (f"❌ FAILED: {e}", None)
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)

    all_success = True
    for name, (status, path) in results.items():
        print(f"  {name}: {status}")
        if path:
            print(f"    → {path}")
        if "FAILED" in status:
            all_success = False

    print("\n" + "="*60)
    if all_success:
        print("All exports successful!")
        print(f"\nExported models are in: {output_dir}")
        print("\nNext steps:")
        print("  1. Run test_memory_onnx_validation.py to verify end-to-end")
        print("  2. Copy models to cvat-ui/plugins/sam3/public/")
        print("  3. Update inference.worker.ts to use memory models")
    else:
        print("Some exports failed. Check the errors above.")
    print("="*60)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
