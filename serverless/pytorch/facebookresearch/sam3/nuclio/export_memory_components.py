#!/usr/bin/env python3
"""
Export SAM3 Memory Components to ONNX format.

Exports:
1. Memory Attention - fuses current frame features with past memories
2. Memory Encoder - encodes current frame + mask into memory representation
3. Object Pointer Projection - projects mask decoder output to object pointer

For onnxruntime-web 1.14.0 compatibility, uses opset 17 (max supported is 18).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


# ============================================================================
# Memory Attention Wrapper
# ============================================================================

class MemoryAttentionONNXWrapper(nn.Module):
    """
    Wrapper for Sam3TrackerVideoMemoryAttention for ONNX export.
    Fixes num_object_pointer_tokens=0 to avoid RoPE dimension issues.
    """

    def __init__(self, memory_attention_module, num_object_pointer_tokens: int = 0):
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

        Returns:
            Updated current frame features with memory context [B, curr_seq, 256]
        """
        # Transpose to seq-first format expected by HuggingFace module
        current_features_seq_first = current_vision_features.transpose(0, 1)
        memory_seq_first = memory.transpose(0, 1)
        current_pos_seq_first = current_vision_pos_enc.transpose(0, 1)
        memory_pos_seq_first = memory_pos_enc.transpose(0, 1)

        output = self.memory_attention(
            current_vision_features=current_features_seq_first,
            memory=memory_seq_first,
            current_vision_position_embeddings=current_pos_seq_first,
            memory_posision_embeddings=memory_pos_seq_first,
            num_object_pointer_tokens=self.num_object_pointer_tokens,
        )

        # Output: (1, B, seq, D) -> (B, seq, D)
        output = output.transpose(0, 1).squeeze(1)
        return output


# ============================================================================
# Memory Encoder Wrapper
# ============================================================================

class MemoryEncoderONNXWrapper(nn.Module):
    """
    Wrapper for Sam3TrackerVideoMemoryEncoder for ONNX export.
    """

    def __init__(self, memory_encoder_module):
        super().__init__()
        self.memory_encoder = memory_encoder_module

    def forward(
        self,
        vision_features: torch.Tensor,  # [batch, 256, H, W]
        masks: torch.Tensor,             # [batch, 1, H*4, W*4] (at 4x resolution)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.

        Returns:
            Tuple of:
            - memory_features: [B, 64, H, W]
            - memory_pos_enc: [B, 64, H, W]
        """
        memory_features, memory_pos_enc = self.memory_encoder(vision_features, masks)
        return memory_features, memory_pos_enc


# ============================================================================
# Object Pointer Wrapper
# ============================================================================

class ObjectPointerONNXWrapper(nn.Module):
    """
    Wrapper for object pointer projection.

    The object pointer is computed from the mask decoder output tokens.
    In SAM3, this is done via an MLP that projects the output tokens.
    """

    def __init__(self, obj_ptr_proj_module, no_obj_ptr_embedding):
        super().__init__()
        self.obj_ptr_proj = obj_ptr_proj_module
        self.register_buffer("no_obj_ptr", no_obj_ptr_embedding)

    def forward(
        self,
        output_token: torch.Tensor,     # [batch, 1, 256] - from mask decoder
        object_score_logits: torch.Tensor,  # [batch, 1] - objectness score
    ) -> torch.Tensor:
        """
        Forward pass for ONNX export.

        Returns:
            Object pointer: [B, 256]
        """
        # Project the output token
        obj_ptr = self.obj_ptr_proj(output_token)  # [B, 1, 256]
        obj_ptr = obj_ptr.squeeze(1)  # [B, 256]

        # Mix with no-object pointer based on object score
        # If object_score_logits < 0, object is not present, use no_obj_ptr
        is_obj_present = (object_score_logits > 0).float()  # [B, 1]
        obj_ptr = obj_ptr * is_obj_present + self.no_obj_ptr * (1 - is_obj_present)

        return obj_ptr


# ============================================================================
# Export Functions
# ============================================================================

def load_sam3_config():
    """Load SAM3 configuration from HuggingFace."""
    from transformers import Sam3VideoConfig
    from transformers.models.sam3_tracker_video.configuration_sam3_tracker_video import (
        Sam3TrackerVideoConfig,
    )

    full_config = Sam3VideoConfig.from_pretrained('facebook/sam3')
    tracker_config = full_config.tracker_config

    if isinstance(tracker_config, dict):
        tracker_config = Sam3TrackerVideoConfig(**tracker_config)

    if not hasattr(tracker_config, '_attn_implementation') or tracker_config._attn_implementation is None:
        tracker_config._attn_implementation = 'eager'

    return tracker_config


def load_sam3_weights():
    """Load SAM3 weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    weights_file = hf_hub_download("facebook/sam3", "model.safetensors", local_files_only=True)
    return safetensors.torch.load_file(weights_file)


def export_memory_attention(output_dir: str, opset_version: int = 17, num_memory_frames: int = 7):
    """
    Export memory attention module to ONNX.

    Args:
        output_dir: Directory for output ONNX file
        opset_version: ONNX opset version (default 17 for browser compatibility)
        num_memory_frames: Number of memory frames to trace with (default 7, max supported)
                          This affects how RoPE repeat_freqs_k is traced.
    """
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoMemoryAttention,
    )

    print("\n" + "=" * 60)
    print("Exporting Memory Attention")
    print("=" * 60)

    config = load_sam3_config()
    all_weights = load_sam3_weights()

    # Create and load weights
    mem_attn = Sam3TrackerVideoMemoryAttention(config)

    prefix = "tracker_model.memory_attention."
    weights = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    mem_attn.load_state_dict(weights)
    print(f"Loaded {len(weights)} weight tensors")

    # Create wrapper
    wrapper = MemoryAttentionONNXWrapper(mem_attn, num_object_pointer_tokens=0)
    wrapper.eval()

    # Get expected sequence length from config
    seq_len = config.memory_attention_rope_feat_sizes[0] * config.memory_attention_rope_feat_sizes[1]
    print(f"Query sequence length: {seq_len} ({config.memory_attention_rope_feat_sizes})")

    # Memory sequence length = num_memory_frames * seq_len
    # This is critical for tracing RoPE repeat_freqs_k correctly
    mem_seq_len = seq_len * num_memory_frames
    print(f"Memory sequence length: {mem_seq_len} ({num_memory_frames} frames × {seq_len})")
    print(f"Max supported frames (num_maskmem): {config.num_maskmem}")

    # Create sample inputs with multi-frame memory
    batch_size = 1
    current_features = torch.randn(batch_size, seq_len, 256)
    memory = torch.randn(batch_size, mem_seq_len, 64)  # Multi-frame memory
    current_pos_enc = torch.randn(batch_size, seq_len, 256)
    memory_pos_enc = torch.randn(batch_size, mem_seq_len, 64)  # Multi-frame pos enc

    # Test forward
    with torch.no_grad():
        output = wrapper(current_features, memory, current_pos_enc, memory_pos_enc)
        print(f"Output shape: {output.shape}")

    # Export
    output_path = os.path.join(output_dir, "memory_attention.onnx")

    dynamic_axes = {
        'current_vision_features': {0: 'batch', 1: 'curr_seq'},
        'memory': {0: 'batch', 1: 'mem_seq'},
        'current_vision_pos_enc': {0: 'batch', 1: 'curr_seq'},
        'memory_pos_enc': {0: 'batch', 1: 'mem_seq'},
        'output': {0: 'batch', 1: 'curr_seq'},
    }

    torch.onnx.export(
        wrapper,
        (current_features, memory, current_pos_enc, memory_pos_enc),
        output_path,
        opset_version=opset_version,
        input_names=['current_vision_features', 'memory', 'current_vision_pos_enc', 'memory_pos_enc'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Exported: {output_path} ({file_size:.2f} MB)")

    return output_path


def export_memory_encoder(output_dir: str, opset_version: int = 17):
    """Export memory encoder module to ONNX."""
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoMemoryEncoder,
    )

    print("\n" + "=" * 60)
    print("Exporting Memory Encoder")
    print("=" * 60)

    config = load_sam3_config()
    all_weights = load_sam3_weights()

    # Create and load weights
    mem_enc = Sam3TrackerVideoMemoryEncoder(config)

    prefix = "tracker_model.memory_encoder."
    weights = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    mem_enc.load_state_dict(weights)
    print(f"Loaded {len(weights)} weight tensors")

    # Create wrapper
    wrapper = MemoryEncoderONNXWrapper(mem_enc)
    wrapper.eval()

    # Create sample inputs
    # Vision features: [B, 256, H, W] where H, W = 72 (from 1024/14 ≈ 73, but SAM3 uses 72)
    # Masks: [B, 1, H*stride, W*stride] - downsampled by total_stride=16
    batch_size = 1
    H, W = 72, 72
    total_stride = config.mask_downsampler_total_stride  # 16
    vision_features = torch.randn(batch_size, 256, H, W)
    masks = torch.randn(batch_size, 1, H * total_stride, W * total_stride)  # 1152x1152 mask

    print(f"Vision features shape: {vision_features.shape}")
    print(f"Masks shape: {masks.shape} (total_stride={total_stride})")

    # Test forward
    with torch.no_grad():
        memory_features, memory_pos_enc = wrapper(vision_features, masks)
        print(f"Memory features shape: {memory_features.shape}")
        print(f"Memory pos enc shape: {memory_pos_enc.shape}")

    # Export
    output_path = os.path.join(output_dir, "memory_encoder.onnx")

    dynamic_axes = {
        'vision_features': {0: 'batch', 2: 'height', 3: 'width'},
        'masks': {0: 'batch', 2: 'mask_height', 3: 'mask_width'},
        'memory_features': {0: 'batch', 2: 'height', 3: 'width'},
        'memory_pos_enc': {0: 'batch', 2: 'height', 3: 'width'},
    }

    torch.onnx.export(
        wrapper,
        (vision_features, masks),
        output_path,
        opset_version=opset_version,
        input_names=['vision_features', 'masks'],
        output_names=['memory_features', 'memory_pos_enc'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Exported: {output_path} ({file_size:.2f} MB)")

    return output_path


def export_object_pointer(output_dir: str, opset_version: int = 17):
    """Export object pointer projection to ONNX."""
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoModel,
    )

    print("\n" + "=" * 60)
    print("Exporting Object Pointer Projection")
    print("=" * 60)

    all_weights = load_sam3_weights()

    # The object pointer projection is an MLP
    # From weights: object_pointer_proj.proj_in, object_pointer_proj.layers.0, object_pointer_proj.proj_out
    # This matches the MLP pattern with 256 -> 256 -> 256

    # Create simple MLP matching the weight structure
    class ObjectPointerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(256, 256)
            self.layers = nn.ModuleList([nn.Linear(256, 256)])
            self.proj_out = nn.Linear(256, 256)
            self.activation = nn.GELU()

        def forward(self, x):
            x = self.activation(self.proj_in(x))
            for layer in self.layers:
                x = self.activation(layer(x))
            x = self.proj_out(x)
            return x

    obj_ptr_proj = ObjectPointerMLP()

    # Load weights
    prefix = "tracker_model.object_pointer_proj."
    weights = {}
    for k, v in all_weights.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            weights[new_key] = v

    obj_ptr_proj.load_state_dict(weights)
    print(f"Loaded {len(weights)} weight tensors for obj_ptr_proj")

    # Get no_object_pointer embedding
    no_obj_ptr = all_weights["tracker_model.no_object_pointer"]
    print(f"no_object_pointer shape: {no_obj_ptr.shape}")

    # Create wrapper
    wrapper = ObjectPointerONNXWrapper(obj_ptr_proj, no_obj_ptr)
    wrapper.eval()

    # Create sample inputs
    batch_size = 1
    output_token = torch.randn(batch_size, 1, 256)
    object_score_logits = torch.randn(batch_size, 1)

    print(f"Output token shape: {output_token.shape}")
    print(f"Object score logits shape: {object_score_logits.shape}")

    # Test forward
    with torch.no_grad():
        obj_ptr = wrapper(output_token, object_score_logits)
        print(f"Object pointer shape: {obj_ptr.shape}")

    # Export
    output_path = os.path.join(output_dir, "object_pointer.onnx")

    dynamic_axes = {
        'output_token': {0: 'batch'},
        'object_score_logits': {0: 'batch'},
        'object_pointer': {0: 'batch'},
    }

    torch.onnx.export(
        wrapper,
        (output_token, object_score_logits),
        output_path,
        opset_version=opset_version,
        input_names=['output_token', 'object_score_logits'],
        output_names=['object_pointer'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Exported: {output_path} ({file_size:.2f} MB)")

    return output_path


def export_temporal_position_encoding(output_dir: str):
    """Export temporal position encoding as numpy array."""
    print("\n" + "=" * 60)
    print("Exporting Temporal Position Encoding")
    print("=" * 60)

    all_weights = load_sam3_weights()

    # Look for temporal position encoding
    tpos_keys = [k for k in all_weights.keys() if 'tpos' in k.lower() or 'temporal' in k.lower()]
    print(f"Found temporal position keys: {tpos_keys}")

    # In SAM3 HuggingFace, the key is memory_temporal_positional_encoding
    mem_tpos_key = "tracker_model.memory_temporal_positional_encoding"
    if mem_tpos_key in all_weights:
        tensor = all_weights[mem_tpos_key]
        output_path = os.path.join(output_dir, "temporal_pos_enc.npy")
        np.save(output_path, tensor.numpy())
        print(f"✓ Exported: {output_path} (shape: {tensor.shape})")
        return output_path

    # Also save the projection layer if it exists
    proj_weight_key = "tracker_model.temporal_positional_encoding_projection_layer.weight"
    proj_bias_key = "tracker_model.temporal_positional_encoding_projection_layer.bias"

    if proj_weight_key in all_weights:
        weight = all_weights[proj_weight_key]
        bias = all_weights.get(proj_bias_key, torch.zeros(weight.shape[0]))

        output_path = os.path.join(output_dir, "temporal_pos_enc_proj_weight.npy")
        np.save(output_path, weight.numpy())
        print(f"✓ Exported projection weight: {output_path} (shape: {weight.shape})")

        output_path = os.path.join(output_dir, "temporal_pos_enc_proj_bias.npy")
        np.save(output_path, bias.numpy())
        print(f"✓ Exported projection bias: {output_path} (shape: {bias.shape})")

    # Create a default temporal encoding based on SAM3 config if not found
    config = load_sam3_config()
    num_maskmem = getattr(config, 'num_maskmem', 7)
    hidden_dim = config.memory_attention_hidden_size

    # Simple learned temporal encoding placeholder
    temporal_enc = np.zeros((num_maskmem, 1, 1, hidden_dim), dtype=np.float32)
    output_path = os.path.join(output_dir, "temporal_pos_enc.npy")
    np.save(output_path, temporal_enc)
    print(f"✓ Created placeholder: {output_path} (shape: {temporal_enc.shape})")
    return output_path


def validate_onnx(output_path: str, sample_inputs: tuple, pytorch_output):
    """Validate ONNX model against PyTorch output."""
    import onnxruntime as ort
    import onnx

    # Check model
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    # Run inference
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

    input_names = [inp.name for inp in session.get_inputs()]
    ort_inputs = {name: arr.numpy() for name, arr in zip(input_names, sample_inputs)}

    ort_outputs = session.run(None, ort_inputs)

    # Compare
    if isinstance(pytorch_output, tuple):
        for i, (ort_out, pt_out) in enumerate(zip(ort_outputs, pytorch_output)):
            max_diff = np.abs(ort_out - pt_out.numpy()).max()
            print(f"  Output {i}: max diff = {max_diff:.6f}")
    else:
        max_diff = np.abs(ort_outputs[0] - pytorch_output.numpy()).max()
        print(f"  Max diff = {max_diff:.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export SAM3 Memory Components to ONNX")
    parser.add_argument("--output-dir", default="onnx-memory-exports", help="Output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--component", choices=["all", "attention", "encoder", "pointer", "temporal"],
                        default="all", help="Which component to export")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM3 Memory Components ONNX Export")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset: {args.opset}")
    print(f"Component: {args.component}")

    if args.component in ["all", "attention"]:
        export_memory_attention(args.output_dir, args.opset)

    if args.component in ["all", "encoder"]:
        export_memory_encoder(args.output_dir, args.opset)

    if args.component in ["all", "pointer"]:
        export_object_pointer(args.output_dir, args.opset)

    if args.component in ["all", "temporal"]:
        export_temporal_position_encoding(args.output_dir)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)

    # List exported files
    print("\nExported files:")
    for f in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, f)
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            print(f"  {f}: {size / (1024*1024):.2f} MB")
        else:
            print(f"  {f}: {size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
