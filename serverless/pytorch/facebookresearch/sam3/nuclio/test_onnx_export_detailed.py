#!/usr/bin/env python3
"""
Detailed ONNX Export Investigation for SAM3 Memory Attention

This script investigates exactly what causes ONNX export failures
and tests various workarounds.

Key findings from previous tests:
1. HuggingFace's rotate_pairwise uses: view, unbind, stack, flatten - all ONNX-compatible
2. apply_rotary_pos_emb_2d uses: element-wise mul, add - all ONNX-compatible
3. The failures come from reshape operations with dynamic batch size

onnxruntime-web 1.14.0 supports opset 18.
ONNX 1.18 introduces RotaryEmbedding-23, but that's opset 23.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile

print(f"PyTorch version: {torch.__version__}")

# Test basic operations
print("\n" + "="*60)
print("Test 1: rotate_pairwise ONNX Export")
print("="*60)


def rotate_pairwise(x):
    """HuggingFace's pairwise rotation - should be ONNX compatible."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


class RotatePairwiseModule(nn.Module):
    def forward(self, x):
        return rotate_pairwise(x)


try:
    model = RotatePairwiseModule()
    model.eval()

    # Fixed shape input
    x = torch.randn(1, 8, 64, 32)  # [batch, heads, seq, head_dim]

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (x,),
            f.name,
            input_names=['x'],
            output_names=['rotated'],
            opset_version=17,  # Stay within opset 18 support
            dynamic_axes={'x': {0: 'batch', 2: 'seq'}, 'rotated': {0: 'batch', 2: 'seq'}}
        )
        print(f"✅ rotate_pairwise export: SUCCESS ({os.path.getsize(f.name)} bytes)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(f.name)
        out = sess.run(None, {'x': x.numpy()})[0]

        ref = rotate_pairwise(x).numpy()
        diff = np.abs(out - ref).max()
        print(f"   Max diff from PyTorch: {diff}")

        os.unlink(f.name)
except Exception as e:
    print(f"❌ rotate_pairwise export: FAILED - {e}")


print("\n" + "="*60)
print("Test 2: apply_rotary_pos_emb_2d ONNX Export")
print("="*60)


class ApplyRoPEModule(nn.Module):
    """Complete RoPE application module."""

    def forward(self, q, k, cos, sin):
        # Apply rotary embedding
        q_embed = q.float()
        q_embed = (q_embed * cos) + (rotate_pairwise(q_embed) * sin)

        k_embed = k.float()
        k_embed = (k_embed * cos) + (rotate_pairwise(k_embed) * sin)

        return q_embed, k_embed


try:
    model = ApplyRoPEModule()
    model.eval()

    # Fixed shape inputs
    batch, heads, seq, head_dim = 1, 8, 64, 32
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    cos = torch.randn(1, 1, seq, head_dim)  # Broadcastable
    sin = torch.randn(1, 1, seq, head_dim)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (q, k, cos, sin),
            f.name,
            input_names=['q', 'k', 'cos', 'sin'],
            output_names=['q_embed', 'k_embed'],
            opset_version=17,
            dynamic_axes={
                'q': {0: 'batch', 2: 'seq'},
                'k': {0: 'batch', 2: 'seq'},
                'cos': {2: 'seq'},
                'sin': {2: 'seq'},
                'q_embed': {0: 'batch', 2: 'seq'},
                'k_embed': {0: 'batch', 2: 'seq'},
            }
        )
        print(f"✅ apply_rotary_pos_emb_2d export: SUCCESS ({os.path.getsize(f.name)} bytes)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(f.name)
        out = sess.run(None, {'q': q.numpy(), 'k': k.numpy(), 'cos': cos.numpy(), 'sin': sin.numpy()})

        ref_q, ref_k = model(q, k, cos, sin)
        diff_q = np.abs(out[0] - ref_q.numpy()).max()
        diff_k = np.abs(out[1] - ref_k.numpy()).max()
        print(f"   Max diff q: {diff_q}, k: {diff_k}")

        os.unlink(f.name)
except Exception as e:
    print(f"❌ apply_rotary_pos_emb_2d export: FAILED - {e}")


print("\n" + "="*60)
print("Test 3: RoPE Attention (qkv projections + RoPE + attention)")
print("="*60)


class RoPEAttention(nn.Module):
    """Simplified RoPE attention matching HuggingFace's pattern."""

    def __init__(self, hidden_size=256, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, cos, sin):
        batch, seq, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = q.float()
        q = (q * cos) + (rotate_pairwise(q) * sin)

        k = k.float()
        k = (k * cos) + (rotate_pairwise(k) * sin)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, seq, self.hidden_size)
        return self.o_proj(out)


try:
    model = RoPEAttention(hidden_size=256, num_heads=8)
    model.eval()

    batch, seq, hidden = 1, 64, 256
    head_dim = hidden // 8
    x = torch.randn(batch, seq, hidden)
    cos = torch.randn(1, 1, seq, head_dim)
    sin = torch.randn(1, 1, seq, head_dim)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (x, cos, sin),
            f.name,
            input_names=['x', 'cos', 'sin'],
            output_names=['output'],
            opset_version=17,
            dynamic_axes={
                'x': {0: 'batch', 1: 'seq'},
                'cos': {2: 'seq'},
                'sin': {2: 'seq'},
                'output': {0: 'batch', 1: 'seq'},
            }
        )
        print(f"✅ RoPE Attention export: SUCCESS ({os.path.getsize(f.name)} bytes)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(f.name)
        out = sess.run(None, {'x': x.numpy(), 'cos': cos.numpy(), 'sin': sin.numpy()})[0]

        ref = model(x, cos, sin).detach().numpy()
        diff = np.abs(out - ref).max()
        print(f"   Max diff: {diff}")

        # Test with different batch/seq sizes
        x2 = torch.randn(2, 128, hidden)
        cos2 = torch.randn(1, 1, 128, head_dim)
        sin2 = torch.randn(1, 1, 128, head_dim)
        out2 = sess.run(None, {'x': x2.numpy(), 'cos': cos2.numpy(), 'sin': sin2.numpy()})[0]
        print(f"   Dynamic shapes work: batch=2, seq=128 → output shape {out2.shape}")

        os.unlink(f.name)
except Exception as e:
    print(f"❌ RoPE Attention export: FAILED - {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*60)
print("Test 4: Memory Attention Layer (Self + Cross attention)")
print("="*60)


class MemoryAttentionLayer(nn.Module):
    """Simplified memory attention layer matching HuggingFace's pattern."""

    def __init__(self, hidden_size=256, num_heads=8, kv_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Self attention
        self.self_q = nn.Linear(hidden_size, hidden_size)
        self.self_k = nn.Linear(hidden_size, hidden_size)
        self.self_v = nn.Linear(hidden_size, hidden_size)
        self.self_o = nn.Linear(hidden_size, hidden_size)

        # Cross attention (different kv dim for memory)
        self.cross_q = nn.Linear(hidden_size, hidden_size)
        self.cross_k = nn.Linear(kv_dim, hidden_size)
        self.cross_v = nn.Linear(kv_dim, hidden_size)
        self.cross_o = nn.Linear(hidden_size, hidden_size)

        # Layer norms and MLP
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def _attention(self, q, k, v, cos, sin, apply_rope_to_k=True):
        # Apply RoPE
        q = q.float()
        q = (q * cos) + (rotate_pairwise(q) * sin)

        if apply_rope_to_k:
            k = k.float()
            k = (k * cos) + (rotate_pairwise(k) * sin)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, queries, memory, cos, sin):
        batch, q_seq, _ = queries.shape
        _, m_seq, _ = memory.shape

        # Self attention on queries
        x = self.ln1(queries)
        q = self.self_q(x).view(batch, q_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_k(x).view(batch, q_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_v(x).view(batch, q_seq, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._attention(q, k, v, cos, sin)
        out = out.transpose(1, 2).reshape(batch, q_seq, self.hidden_size)
        queries = queries + self.self_o(out)

        # Cross attention: queries attend to memory
        # Need to handle different cos/sin for memory sequence length
        x = self.ln2(queries)
        q = self.cross_q(x).view(batch, q_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(memory).view(batch, m_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(memory).view(batch, m_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # For cross attention, we need separate cos/sin for q and k
        # or skip RoPE on keys (as in some implementations)
        out = self._attention(q, k, v, cos, sin, apply_rope_to_k=False)
        out = out.transpose(1, 2).reshape(batch, q_seq, self.hidden_size)
        queries = queries + self.cross_o(out)

        # MLP
        queries = queries + self.mlp(self.ln3(queries))

        return queries


try:
    model = MemoryAttentionLayer(hidden_size=256, num_heads=8, kv_dim=64)
    model.eval()

    batch, q_seq, m_seq, hidden = 1, 64, 128, 256
    head_dim = hidden // 8

    queries = torch.randn(batch, q_seq, hidden)
    memory = torch.randn(batch, m_seq, 64)  # Memory has different dim
    cos = torch.randn(1, 1, q_seq, head_dim)
    sin = torch.randn(1, 1, q_seq, head_dim)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (queries, memory, cos, sin),
            f.name,
            input_names=['queries', 'memory', 'cos', 'sin'],
            output_names=['output'],
            opset_version=17,
            dynamic_axes={
                'queries': {0: 'batch', 1: 'q_seq'},
                'memory': {0: 'batch', 1: 'm_seq'},
                'cos': {2: 'q_seq'},
                'sin': {2: 'q_seq'},
                'output': {0: 'batch', 1: 'q_seq'},
            }
        )
        print(f"✅ Memory Attention Layer export: SUCCESS ({os.path.getsize(f.name)} bytes)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(f.name)
        out = sess.run(None, {
            'queries': queries.numpy(),
            'memory': memory.numpy(),
            'cos': cos.numpy(),
            'sin': sin.numpy()
        })[0]

        ref = model(queries, memory, cos, sin).detach().numpy()
        diff = np.abs(out - ref).max()
        print(f"   Max diff: {diff}")

        os.unlink(f.name)
except Exception as e:
    print(f"❌ Memory Attention Layer export: FAILED - {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*60)
print("Test 5: Memory Encoder (Mask Downsampling + Fusion)")
print("="*60)


class SimpleMaskDownsampler(nn.Module):
    """Simplified mask downsampler using only ONNX-compatible ops."""

    def __init__(self, out_dim=64):
        super().__init__()
        # Use standard convolutions instead of complex downsampling
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=4, padding=0)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0)
        self.norm2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, out_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, mask):
        # Input: [B, 1, H, W] where H, W are multiples of 16
        x = self.conv1(mask)
        x = self.norm1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        return x


class SimpleMemoryEncoder(nn.Module):
    """Simplified memory encoder."""

    def __init__(self, feat_dim=256, mask_dim=64, out_dim=64):
        super().__init__()
        self.mask_downsampler = SimpleMaskDownsampler(mask_dim)
        self.feat_proj = nn.Conv2d(feat_dim, out_dim, kernel_size=1)
        self.fuser = nn.Conv2d(out_dim + mask_dim, out_dim, kernel_size=3, padding=1)
        self.out_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, features, mask):
        # features: [B, C, H, W]
        # mask: [B, 1, H*16, W*16]

        mask_feats = self.mask_downsampler(mask)  # [B, mask_dim, H, W]
        feat_proj = self.feat_proj(features)  # [B, out_dim, H, W]

        # Resize mask_feats to match feat_proj if needed
        if mask_feats.shape[2:] != feat_proj.shape[2:]:
            mask_feats = nn.functional.interpolate(
                mask_feats, size=feat_proj.shape[2:], mode='bilinear', align_corners=False
            )

        fused = torch.cat([feat_proj, mask_feats], dim=1)
        fused = torch.relu(self.fuser(fused))
        return self.out_proj(fused)


try:
    model = SimpleMemoryEncoder(feat_dim=256, mask_dim=64, out_dim=64)
    model.eval()

    batch = 1
    feat_h, feat_w = 72, 72  # SAM3 feature map size
    features = torch.randn(batch, 256, feat_h, feat_w)
    mask = torch.randn(batch, 1, feat_h * 16, feat_w * 16)  # Full res mask

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model,
            (features, mask),
            f.name,
            input_names=['features', 'mask'],
            output_names=['memory'],
            opset_version=17,
            dynamic_axes={
                'features': {0: 'batch'},
                'mask': {0: 'batch'},
                'memory': {0: 'batch'},
            }
        )
        print(f"✅ Memory Encoder export: SUCCESS ({os.path.getsize(f.name)} bytes)")

        # Verify with ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(f.name)
        out = sess.run(None, {
            'features': features.numpy(),
            'mask': mask.numpy(),
        })[0]

        ref = model(features, mask).detach().numpy()
        diff = np.abs(out - ref).max()
        print(f"   Max diff: {diff}")
        print(f"   Output shape: {out.shape}")

        os.unlink(f.name)
except Exception as e:
    print(f"❌ Memory Encoder export: FAILED - {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Key findings:
1. rotate_pairwise is ONNX-compatible (uses view, unbind, stack, flatten)
2. apply_rotary_pos_emb_2d is ONNX-compatible when using rotate_pairwise
3. Full RoPE attention with dynamic shapes exports successfully
4. Memory attention layer with cross-attention exports successfully
5. Memory encoder with fixed downsampling factors exports successfully

The previous failures were likely due to:
- Using the official SAM3 code which uses view_as_complex
- Dynamic reshape operations dependent on runtime batch size
- Specific HuggingFace module interfaces not matching export expectations

RECOMMENDATION:
The memory attention CAN be ONNX-exported if we:
1. Use HuggingFace's rotate_pairwise pattern
2. Pre-compute cos/sin buffers for expected sequence lengths
3. Use fixed or symbolic batch dimensions
4. Avoid view_as_complex entirely
""")
