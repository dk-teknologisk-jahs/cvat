#!/usr/bin/env python3
"""
Export SAM3 Memory Attention module to ONNX format.

Memory attention takes:
- Current frame vision features (to be updated with memory context)
- Memory from previous frames (encoded frame features)
- Position embeddings for both

The memory attention allows current frame features to attend to past frame
memories, enabling temporal tracking in video segmentation.

SAM3 uses:
- memory_attention_rope_feat_sizes: [72, 72] -> 5184 sequence length
- memory_attention_hidden_size: 256
- memory_attention_num_layers: 4

For onnxruntime-web 1.14.0 compatibility, uses opset 17 (max supported is 18).
"""

import os
import torch
import torch.nn as nn
import numpy as np


class MemoryAttentionONNXWrapper(nn.Module):
    """
    Wrapper for Sam3TrackerVideoMemoryAttention that handles ONNX export constraints.

    The original module has:
    - RoPE (rotary position embeddings) which need special handling
    - Dynamic reshapes based on sequence lengths
    - Integer parameter (num_object_pointer_tokens) that must be fixed at export time

    This wrapper fixes num_object_pointer_tokens and provides clean tensor I/O.
    """

    def __init__(self, memory_attention_module, num_object_pointer_tokens=1):
        super().__init__()
        self.memory_attention = memory_attention_module
        self.num_object_pointer_tokens = num_object_pointer_tokens

    def forward(
        self,
        current_vision_features: torch.Tensor,  # [batch, curr_seq_len, 256]
        memory: torch.Tensor,                    # [batch, mem_seq_len, 64]
        current_vision_pos_enc: torch.Tensor,   # [batch, curr_seq_len, 256]
        memory_pos_enc: torch.Tensor,           # [batch, mem_seq_len, 64]
    ) -> torch.Tensor:
        """
        Forward pass for ONNX export.

        Args:
            current_vision_features: Current frame features [B, curr_seq, 256]
            memory: Memory features from past frames [B, mem_seq, 64]
            current_vision_pos_enc: Position encoding for current features [B, curr_seq, 256]
            memory_pos_enc: Position encoding for memory [B, mem_seq, 64]

        Returns:
            Updated current frame features with memory context [B, curr_seq, 256]
        """
        # Transpose to seq-first format expected by HuggingFace module
        # [B, seq, D] -> [seq, B, D]
        current_features_seq_first = current_vision_features.transpose(0, 1)
        memory_seq_first = memory.transpose(0, 1)
        current_pos_seq_first = current_vision_pos_enc.transpose(0, 1)
        memory_pos_seq_first = memory_pos_enc.transpose(0, 1)

        # Call the memory attention module
        output = self.memory_attention(
            current_vision_features=current_features_seq_first,
            memory=memory_seq_first,
            current_vision_position_embeddings=current_pos_seq_first,
            memory_posision_embeddings=memory_pos_seq_first,  # Note: typo in HF code
            num_object_pointer_tokens=self.num_object_pointer_tokens,
        )

        # HF module outputs (1, B, seq, D) due to internal transpose after layer processing
        # We need (B, seq, D) so: transpose(0,1) -> (B, 1, seq, D), then squeeze(1) -> (B, seq, D)
        output = output.transpose(0, 1).squeeze(1)
        return output


def export_memory_attention(
    output_dir: str = "onnx-memory-exports",
    curr_seq_len: int = 5184,  # 72x72 spatial grid for SAM3
    mem_seq_len: int = 5184,   # Same for memory (can be N * 5184 for N memory frames)
    batch_size: int = 1,
    opset_version: int = 17,   # Safe for onnxruntime-web 1.14.0
    num_object_pointer_tokens: int = 0,  # Set to 0 for ONNX export to avoid RoPE issues
):
    """
    Export SAM3 memory attention module to ONNX.

    Uses the SAM3 tracker video model from HuggingFace.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "memory_attention.onnx")

    print("=" * 60)
    print("SAM3 Memory Attention ONNX Export")
    print("=" * 60)
    print(f"Target: {output_path}")
    print(f"Opset: {opset_version}")
    print(f"Current seq len: {curr_seq_len}")
    print(f"Memory seq len: {mem_seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"num_object_pointer_tokens (fixed): {num_object_pointer_tokens}")
    print()

    print("Loading SAM3 model configuration...")

    try:
        from transformers import Sam3VideoConfig
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoMemoryAttention,
        )
        from transformers.models.sam3_tracker_video.configuration_sam3_tracker_video import (
            Sam3TrackerVideoConfig,
        )

        # Load config from facebook/sam3
        full_config = Sam3VideoConfig.from_pretrained('facebook/sam3')

        # Get tracker config (which contains memory attention settings)
        # It may already be a config object or a dict
        tracker_config = full_config.tracker_config
        if isinstance(tracker_config, dict):
            tracker_config = Sam3TrackerVideoConfig(**tracker_config)

        # Ensure attn_implementation is set for the attention layers
        if not hasattr(tracker_config, '_attn_implementation') or tracker_config._attn_implementation is None:
            tracker_config._attn_implementation = 'eager'

        print(f"Memory attention hidden size: {tracker_config.memory_attention_hidden_size}")
        print(f"Memory attention layers: {tracker_config.memory_attention_num_layers}")
        print(f"Memory attention RoPE feat sizes: {tracker_config.memory_attention_rope_feat_sizes}")

        expected_seq = tracker_config.memory_attention_rope_feat_sizes[0] * tracker_config.memory_attention_rope_feat_sizes[1]
        print(f"Expected sequence length: {expected_seq}")

        # Create just the memory attention module
        print("\nConstructing SAM3 memory attention module...")
        mem_attn = Sam3TrackerVideoMemoryAttention(tracker_config)

        # Try to load weights from the full model checkpoint
        try:
            from huggingface_hub import hf_hub_download
            import safetensors.torch

            print("\nAttempting to load pretrained weights...")
            weights_file = hf_hub_download(
                "facebook/sam3",
                "model.safetensors",
                local_files_only=True
            )

            all_weights = safetensors.torch.load_file(weights_file)

            # Filter to tracker_model.memory_attention weights
            mem_attn_weights = {}
            prefix = "tracker_model.memory_attention."
            for key, value in all_weights.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    mem_attn_weights[new_key] = value

            if mem_attn_weights:
                mem_attn.load_state_dict(mem_attn_weights)
                print(f"Loaded {len(mem_attn_weights)} weight tensors")
            else:
                print("No memory attention weights found, using random initialization")

        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using random initialization (graph structure will be correct)")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Create wrapper for ONNX export
    print("\nCreating ONNX export wrapper...")
    wrapper = MemoryAttentionONNXWrapper(
        mem_attn,
        num_object_pointer_tokens=num_object_pointer_tokens
    )
    wrapper.eval()

    # Create sample inputs - MUST use the exact RoPE sequence length
    # The RoPE embeddings are pre-computed for 72x72 = 5184 positions
    test_curr_seq = expected_seq  # Must match RoPE size
    test_mem_seq = expected_seq

    print(f"\nCreating sample inputs:")
    print(f"  Current seq: {test_curr_seq} (must match RoPE feat size)")
    print(f"  Memory seq: {test_mem_seq}")

    current_features = torch.randn(batch_size, test_curr_seq, 256)
    memory = torch.randn(batch_size, test_mem_seq, 64)
    current_pos_enc = torch.randn(batch_size, test_curr_seq, 256)
    memory_pos_enc = torch.randn(batch_size, test_mem_seq, 64)

    print(f"\nInput shapes:")
    print(f"  current_vision_features: {current_features.shape}")
    print(f"  memory: {memory.shape}")
    print(f"  current_vision_pos_enc: {current_pos_enc.shape}")
    print(f"  memory_pos_enc: {memory_pos_enc.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            output = wrapper(current_features, memory, current_pos_enc, memory_pos_enc)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass error: {e}")
            raise

    # Export to ONNX
    print("\nExporting to ONNX...")

    # Define dynamic axes for variable sequence lengths
    dynamic_axes = {
        'current_vision_features': {0: 'batch', 1: 'curr_seq'},
        'memory': {0: 'batch', 1: 'mem_seq'},
        'current_vision_pos_enc': {0: 'batch', 1: 'curr_seq'},
        'memory_pos_enc': {0: 'batch', 1: 'mem_seq'},
        'output': {0: 'batch', 1: 'curr_seq'},
    }

    try:
        torch.onnx.export(
            wrapper,
            (current_features, memory, current_pos_enc, memory_pos_enc),
            output_path,
            opset_version=opset_version,
            input_names=[
                'current_vision_features',
                'memory',
                'current_vision_pos_enc',
                'memory_pos_enc'
            ],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ Export successful!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size:.2f} MB")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Validate with ONNX Runtime
    print("\nValidating with ONNX Runtime...")
    try:
        import onnxruntime as ort
        import onnx

        # Check ONNX model
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("  ONNX model check passed")

        # Test inference
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

        # Get input/output info
        print("\n  Model inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape} ({inp.type})")
        print("\n  Model outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape} ({out.type})")

        # Run inference
        ort_inputs = {
            'current_vision_features': current_features.numpy(),
            'memory': memory.numpy(),
            'current_vision_pos_enc': current_pos_enc.numpy(),
            'memory_pos_enc': memory_pos_enc.numpy(),
        }

        ort_output = session.run(None, ort_inputs)[0]

        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = wrapper(current_features, memory, current_pos_enc, memory_pos_enc).numpy()

        max_diff = np.abs(ort_output - torch_output).max()
        mean_diff = np.abs(ort_output - torch_output).mean()

        print(f"\n  Validation results:")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        print(f"    Output shape: {ort_output.shape}")

        if max_diff < 1e-4:
            print("\n✓ Validation passed!")
        else:
            print(f"\n⚠ Warning: Max diff {max_diff} exceeds 1e-4 threshold")

    except ImportError as e:
        print(f"  Skipping validation (missing dependency): {e}")
    except Exception as e:
        print(f"  Validation error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export SAM3 Memory Attention to ONNX")
    parser.add_argument("--output-dir", default="onnx-memory-exports", help="Output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--curr-seq-len", type=int, default=5184, help="Current frame sequence length (72x72 for SAM3)")
    parser.add_argument("--mem-seq-len", type=int, default=5184, help="Memory sequence length (72x72 for SAM3)")
    parser.add_argument("--num-obj-ptrs", type=int, default=0, help="Number of object pointer tokens (fixed, 0 for ONNX export)")

    args = parser.parse_args()

    export_memory_attention(
        output_dir=args.output_dir,
        curr_seq_len=args.curr_seq_len,
        mem_seq_len=args.mem_seq_len,
        opset_version=args.opset,
        num_object_pointer_tokens=args.num_obj_ptrs,
    )
