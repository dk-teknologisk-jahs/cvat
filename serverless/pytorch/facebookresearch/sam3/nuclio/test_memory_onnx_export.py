#!/usr/bin/env python3
"""
Test ONNX exportability of SAM3 memory attention components.

This script investigates whether the key components needed for video tracking
can be exported to ONNX format:

1. Memory Encoder (SimpleMaskEncoder) - encodes mask + features into memory
2. Transformer Encoder (memory attention) - fuses current features with memory bank
3. Object Pointer Projection - extracts object identity tokens
4. RoPE (Rotary Position Embeddings) - the key blocker!

Key challenges to watch for:
- Dynamic shapes (variable number of memory frames)
- Complex attention patterns
- view_as_complex operations (RoPE) - THIS IS THE MAIN BLOCKER
- Control flow

Usage:
    conda activate grimme-tf2.18
    python test_memory_onnx_export.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add sam3 to path
SAM3_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "sam3"
sys.path.insert(0, str(SAM3_PATH))

# Force CPU for testing
DEVICE = 'cpu'

print(f"SAM3 path: {SAM3_PATH}")
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX opset support: {torch.onnx.producer_version}")
print(f"Testing on device: {DEVICE}")


def test_basic_attention_export():
    """Test if basic multi-head attention can be exported."""
    print("\n" + "="*60)
    print("Test 1: Basic Multi-Head Attention Export")
    print("="*60)

    class SimpleAttention(nn.Module):
        def __init__(self, d_model=256, nhead=8):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=False)

        def forward(self, query, key, value):
            # query: [seq_len, batch, d_model]
            # key: [mem_len, batch, d_model]
            # value: [mem_len, batch, d_model]
            out, _ = self.attn(query, key, value)
            return out

    model = SimpleAttention().eval()

    # Test with fixed shapes
    seq_len, mem_len, batch, d_model = 5184, 1000, 1, 256  # 72*72 = 5184
    query = torch.randn(seq_len, batch, d_model)
    key = torch.randn(mem_len, batch, d_model)
    value = torch.randn(mem_len, batch, d_model)

    try:
        torch.onnx.export(
            model,
            (query, key, value),
            "/tmp/test_attention.onnx",
            input_names=["query", "key", "value"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes={
                "query": {0: "seq_len", 1: "batch"},
                "key": {0: "mem_len", 1: "batch"},
                "value": {0: "mem_len", 1: "batch"},
                "output": {0: "seq_len", 1: "batch"},
            }
        )
        print("✅ Basic attention export: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Basic attention export: FAILED - {e}")
        return False


def test_memory_encoder_export():
    """Test if SimpleMaskEncoder can be exported."""
    print("\n" + "="*60)
    print("Test 2: Memory Encoder (SimpleMaskEncoder) Export")
    print("="*60)

    try:
        from sam3.model.memory import SimpleMaskEncoder, SimpleMaskDownSampler, SimpleFuser, CXBlock
        from sam3.model.position_encoding import PositionEmbeddingSine

        # Build a SimpleMaskEncoder similar to SAM3's config
        mask_downsampler = SimpleMaskDownSampler(
            embed_dim=256,
            kernel_size=4,
            stride=4,
            padding=0,
            total_stride=16,
        )

        fuser_layer = CXBlock(dim=256, kernel_size=7, padding=3)
        fuser = SimpleFuser(layer=fuser_layer, num_layers=2)

        pos_enc = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

        memory_encoder = SimpleMaskEncoder(
            out_dim=64,  # mem_dim
            mask_downsampler=mask_downsampler,
            fuser=fuser,
            position_encoding=pos_enc,
            in_dim=256,
        ).eval()

        # Test inputs
        batch = 1
        pix_feat = torch.randn(batch, 256, 72, 72)  # Visual features
        masks = torch.randn(batch, 1, 1008, 1008)   # High-res mask

        # Test forward
        with torch.no_grad():
            out = memory_encoder(pix_feat, masks)
            print(f"   Memory encoder output shape: {out['vision_features'].shape}")

        # Try ONNX export
        class MemoryEncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, pix_feat, masks):
                out = self.encoder(pix_feat, masks)
                return out['vision_features']

        wrapper = MemoryEncoderWrapper(memory_encoder).eval()

        torch.onnx.export(
            wrapper,
            (pix_feat, masks),
            "/tmp/test_memory_encoder.onnx",
            input_names=["pix_feat", "masks"],
            output_names=["memory_features"],
            opset_version=17,
            dynamic_axes={
                "pix_feat": {0: "batch"},
                "masks": {0: "batch"},
                "memory_features": {0: "batch"},
            }
        )

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession("/tmp/test_memory_encoder.onnx")
        onnx_out = sess.run(None, {
            "pix_feat": pix_feat.numpy(),
            "masks": masks.numpy(),
        })[0]

        torch_out = wrapper(pix_feat, masks).numpy()
        max_diff = np.abs(onnx_out - torch_out).max()
        print(f"   Max difference PyTorch vs ONNX: {max_diff:.6f}")

        print("✅ Memory encoder export: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ Memory encoder export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_encoder_fusion_export():
    """Test if TransformerEncoderFusion (memory attention) can be exported."""
    print("\n" + "="*60)
    print("Test 3: Transformer Encoder Fusion (Memory Attention) Export")
    print("="*60)

    try:
        from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
        from sam3.model.model_misc import MultiHeadAttention

        # Build encoder layer similar to SAM3
        d_model = 256
        mem_dim = 64
        nhead = 8

        # The key is that the encoder does cross-attention between:
        # - current frame features (src)
        # - memory bank features (prompt)

        # For simplicity, let's test a minimal version
        class MinimalMemoryAttention(nn.Module):
            """Minimal memory attention module for ONNX export testing."""

            def __init__(self, d_model=256, mem_dim=64, nhead=8):
                super().__init__()
                self.d_model = d_model
                self.mem_dim = mem_dim

                # Project memory to d_model
                self.mem_proj = nn.Linear(mem_dim, d_model)

                # Cross attention: current features attend to memory
                self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=False)

                # FFN
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )

                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, src, src_pos, memory, memory_pos):
                """
                Args:
                    src: [HW, B, d_model] - current frame features
                    src_pos: [HW, B, d_model] - position encoding
                    memory: [mem_len, B, mem_dim] - memory bank features
                    memory_pos: [mem_len, B, mem_dim] - memory position encoding

                Returns:
                    out: [HW, B, d_model] - features fused with memory
                """
                # Project memory to d_model
                memory_proj = self.mem_proj(memory)
                memory_pos_proj = self.mem_proj(memory_pos)

                # Cross attention
                q = src + src_pos
                k = memory_proj + memory_pos_proj
                v = memory_proj

                attn_out, _ = self.cross_attn(q, k, v)
                src = self.norm1(src + attn_out)

                # FFN
                src = self.norm2(src + self.ffn(src))

                return src

        model = MinimalMemoryAttention().eval()

        # Test inputs - simulating memory bank with multiple frames
        HW = 72 * 72  # 5184
        batch = 1
        mem_frames = 7  # Number of memory frames
        mem_len = HW * mem_frames  # Total memory tokens

        src = torch.randn(HW, batch, 256)
        src_pos = torch.randn(HW, batch, 256)
        memory = torch.randn(mem_len, batch, 64)
        memory_pos = torch.randn(mem_len, batch, 64)

        # Test forward
        with torch.no_grad():
            out = model(src, src_pos, memory, memory_pos)
            print(f"   Output shape: {out.shape}")

        # ONNX export with dynamic memory length
        torch.onnx.export(
            model,
            (src, src_pos, memory, memory_pos),
            "/tmp/test_memory_attention.onnx",
            input_names=["src", "src_pos", "memory", "memory_pos"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes={
                "src": {0: "hw", 1: "batch"},
                "src_pos": {0: "hw", 1: "batch"},
                "memory": {0: "mem_len", 1: "batch"},
                "memory_pos": {0: "mem_len", 1: "batch"},
                "output": {0: "hw", 1: "batch"},
            }
        )

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession("/tmp/test_memory_attention.onnx")

        # Test with different memory lengths
        for num_frames in [1, 3, 7]:
            mem_len = HW * num_frames
            memory = torch.randn(mem_len, batch, 64)
            memory_pos = torch.randn(mem_len, batch, 64)

            onnx_out = sess.run(None, {
                "src": src.numpy(),
                "src_pos": src_pos.numpy(),
                "memory": memory.numpy(),
                "memory_pos": memory_pos.numpy(),
            })[0]

            torch_out = model(src, src_pos, memory, memory_pos).numpy()
            max_diff = np.abs(onnx_out - torch_out).max()
            print(f"   {num_frames} memory frames - Max diff: {max_diff:.6f}")

        print("✅ Memory attention export: SUCCESS (with dynamic memory length)")
        return True

    except Exception as e:
        print(f"❌ Memory attention export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_tracker_encoder_export():
    """Test if the full SAM3 tracker's transformer encoder can be exported."""
    print("\n" + "="*60)
    print("Test 4: Full SAM3 Tracker Transformer Encoder Export")
    print("="*60)

    try:
        from sam3.model_builder import build_tracker

        print("   Building SAM3 tracker...")
        tracker = build_tracker(
            apply_temporal_disambiguation=False,
            with_backbone=False,  # Don't need backbone for this test
        )
        tracker = tracker.to('cpu').eval()

        # The transformer encoder is: tracker.transformer.encoder
        encoder = tracker.transformer.encoder
        print(f"   Encoder type: {type(encoder)}")

        # Get the encoder's forward signature
        import inspect
        sig = inspect.signature(encoder.forward)
        print(f"   Encoder forward params: {list(sig.parameters.keys())}")

        # Create test inputs matching _prepare_memory_conditioned_features
        HW = 72 * 72
        batch = 1
        d_model = 256
        mem_dim = 64
        num_mem_frames = 3

        # Current frame features (list of 3 levels, but encoder uses only the last)
        feat_sizes = [(72, 72)]
        src = [torch.randn(HW, batch, d_model)]
        src_pos = [torch.randn(HW, batch, d_model)]

        # Memory prompt
        mem_len = HW * num_mem_frames
        prompt = torch.randn(mem_len, batch, mem_dim)
        prompt_pos = torch.randn(mem_len, batch, mem_dim)

        print(f"   src shape: {src[0].shape}")
        print(f"   prompt shape: {prompt.shape}")

        # Test forward
        with torch.no_grad():
            out = encoder(
                src=src,
                src_key_padding_mask=[None],
                src_pos=src_pos,
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=None,
                feat_sizes=feat_sizes,
                num_obj_ptr_tokens=0,
            )
            print(f"   Output memory shape: {out['memory'].shape}")

        # Create wrapper for ONNX export
        class TrackerEncoderWrapper(nn.Module):
            def __init__(self, encoder, feat_sizes):
                super().__init__()
                self.encoder = encoder
                self.feat_sizes = feat_sizes

            def forward(self, src, src_pos, prompt, prompt_pos):
                out = self.encoder(
                    src=[src],
                    src_key_padding_mask=[None],
                    src_pos=[src_pos],
                    prompt=prompt,
                    prompt_pos=prompt_pos,
                    prompt_key_padding_mask=None,
                    feat_sizes=self.feat_sizes,
                    num_obj_ptr_tokens=0,
                )
                return out['memory']

        wrapper = TrackerEncoderWrapper(encoder, feat_sizes).eval()

        # Test wrapper
        with torch.no_grad():
            test_out = wrapper(src[0], src_pos[0], prompt, prompt_pos)
            print(f"   Wrapper output shape: {test_out.shape}")

        # Try ONNX export
        torch.onnx.export(
            wrapper,
            (src[0], src_pos[0], prompt, prompt_pos),
            "/tmp/test_tracker_encoder.onnx",
            input_names=["src", "src_pos", "prompt", "prompt_pos"],
            output_names=["memory"],
            opset_version=17,
            dynamic_axes={
                "src": {0: "hw", 1: "batch"},
                "src_pos": {0: "hw", 1: "batch"},
                "prompt": {0: "mem_len", 1: "batch"},
                "prompt_pos": {0: "mem_len", 1: "batch"},
                "memory": {0: "hw", 1: "batch"},
            }
        )

        print("✅ Full tracker encoder export: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ Full tracker encoder export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_object_pointer_export():
    """Test if object pointer projection can be exported."""
    print("\n" + "="*60)
    print("Test 5: Object Pointer Projection Export")
    print("="*60)

    try:
        from sam3.sam.mask_decoder import MLP

        d_model = 256
        obj_ptr_proj = MLP(d_model, d_model, d_model, 3)
        obj_ptr_proj.eval()

        # Input is the SAM output token
        batch = 1
        sam_output_token = torch.randn(batch, d_model)

        with torch.no_grad():
            obj_ptr = obj_ptr_proj(sam_output_token)
            print(f"   Object pointer shape: {obj_ptr.shape}")

        torch.onnx.export(
            obj_ptr_proj,
            sam_output_token,
            "/tmp/test_obj_ptr_proj.onnx",
            input_names=["sam_output_token"],
            output_names=["obj_ptr"],
            opset_version=17,
            dynamic_axes={
                "sam_output_token": {0: "batch"},
                "obj_ptr": {0: "batch"},
            }
        )

        # Verify
        import onnxruntime as ort
        sess = ort.InferenceSession("/tmp/test_obj_ptr_proj.onnx")
        onnx_out = sess.run(None, {"sam_output_token": sam_output_token.numpy()})[0]
        torch_out = obj_ptr_proj(sam_output_token).detach().numpy()
        max_diff = np.abs(onnx_out - torch_out).max()
        print(f"   Max diff: {max_diff:.6f}")

        print("✅ Object pointer projection export: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ Object pointer projection export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("SAM3 Memory Attention ONNX Export Investigation")
    print("="*60)

    results = {}

    # Run all tests
    results["basic_attention"] = test_basic_attention_export()
    results["memory_encoder"] = test_memory_encoder_export()
    results["memory_attention"] = test_transformer_encoder_fusion_export()
    results["full_tracker_encoder"] = test_full_tracker_encoder_export()
    results["object_pointer"] = test_object_pointer_export()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("CONCLUSION: Memory attention CAN be exported to ONNX!")
        print("Hybrid architecture is FEASIBLE.")
    else:
        print("CONCLUSION: Some components cannot be exported to ONNX.")
        print("Server-side propagation is RECOMMENDED.")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
