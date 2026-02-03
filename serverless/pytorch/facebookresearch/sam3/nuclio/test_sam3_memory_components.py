#!/usr/bin/env python3
"""
SAM3 Memory Components ONNX Test Suite

Tests the exported SAM3 memory ONNX models by:
1. Loading each ONNX model and verifying it runs
2. Comparing outputs against PyTorch reference (numerical accuracy)
3. Testing with various input sizes (dynamic shapes)
4. Simulating video propagation pipeline
5. Verifying browser compatibility (opset 17)

Usage:
    conda activate grimme-tf2.18

    # Run all tests
    python test_sam3_memory_components.py

    # Run specific test suite
    python test_sam3_memory_components.py --test encoder
    python test_sam3_memory_components.py --test attention
    python test_sam3_memory_components.py --test pointer
    python test_sam3_memory_components.py --test pipeline

    # Verbose output
    python test_sam3_memory_components.py -v
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Default paths
SCRIPT_DIR = Path(__file__).parent
ONNX_DIR = SCRIPT_DIR / "onnx-memory-exports"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    duration_ms: float
    max_diff: Optional[float] = None


class Colors:
    """ANSI colors for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_result(result: TestResult, verbose: bool = False):
    """Print test result with colors."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result.passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} {result.name} ({result.duration_ms:.1f}ms)")
    if verbose or not result.passed:
        print(f"       {result.message}")
        if result.max_diff is not None:
            print(f"       Max diff: {result.max_diff:.6e}")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


# =============================================================================
# ONNX Model Loading and Info
# =============================================================================

def load_onnx_session(model_path: Path):
    """Load ONNX model and return session."""
    import onnxruntime as ort

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])


def get_model_info(session) -> Dict:
    """Get model input/output information."""
    inputs = {inp.name: (inp.shape, inp.type) for inp in session.get_inputs()}
    outputs = {out.name: (out.shape, out.type) for out in session.get_outputs()}
    return {"inputs": inputs, "outputs": outputs}


def check_onnx_opset(model_path: Path) -> int:
    """Check ONNX model opset version."""
    import onnx
    model = onnx.load(str(model_path))
    return model.opset_import[0].version


# =============================================================================
# Memory Encoder Tests
# =============================================================================

def test_memory_encoder_loads(onnx_dir: Path) -> TestResult:
    """Test that memory encoder ONNX loads successfully."""
    name = "memory_encoder_loads"
    start = time.time()

    try:
        model_path = onnx_dir / "memory_encoder.onnx"
        session = load_onnx_session(model_path)
        info = get_model_info(session)
        opset = check_onnx_opset(model_path)

        duration = (time.time() - start) * 1000
        return TestResult(
            name=name,
            passed=True,
            message=f"Loaded successfully (opset {opset}). Inputs: {list(info['inputs'].keys())}, Outputs: {list(info['outputs'].keys())}",
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_encoder_inference(onnx_dir: Path) -> TestResult:
    """Test memory encoder inference with sample inputs."""
    name = "memory_encoder_inference"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "memory_encoder.onnx")

        # SAM3 uses 72x72 feature maps with 16x downsampling for masks
        B, H, W = 1, 72, 72
        total_stride = 16

        vision_features = np.random.randn(B, 256, H, W).astype(np.float32)
        masks = np.random.randn(B, 1, H * total_stride, W * total_stride).astype(np.float32)

        outputs = session.run(None, {
            "vision_features": vision_features,
            "masks": masks,
        })

        memory_features = outputs[0]
        memory_pos_enc = outputs[1]

        # Expected output shape: [B, 64, H, W]
        expected_shape = (B, 64, H, W)

        duration = (time.time() - start) * 1000

        if memory_features.shape == expected_shape and memory_pos_enc.shape == expected_shape:
            return TestResult(
                name=name,
                passed=True,
                message=f"Output shapes correct: features={memory_features.shape}, pos_enc={memory_pos_enc.shape}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Wrong shapes: features={memory_features.shape}, pos_enc={memory_pos_enc.shape}, expected={expected_shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_encoder_vs_pytorch(onnx_dir: Path) -> TestResult:
    """Compare ONNX memory encoder output against PyTorch reference."""
    name = "memory_encoder_vs_pytorch"
    start = time.time()

    try:
        import torch
        from transformers import Sam3VideoConfig
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoMemoryEncoder,
        )
        from transformers.models.sam3_tracker_video.configuration_sam3_tracker_video import (
            Sam3TrackerVideoConfig,
        )
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Load PyTorch model
        full_config = Sam3VideoConfig.from_pretrained('facebook/sam3')
        tracker_config = full_config.tracker_config
        if isinstance(tracker_config, dict):
            tracker_config = Sam3TrackerVideoConfig(**tracker_config)

        mem_enc = Sam3TrackerVideoMemoryEncoder(tracker_config)

        # Load weights
        weights_file = hf_hub_download("facebook/sam3", "model.safetensors", local_files_only=True)
        all_weights = safetensors.torch.load_file(weights_file)
        prefix = "tracker_model.memory_encoder."
        weights = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
        mem_enc.load_state_dict(weights)
        mem_enc.eval()

        # Load ONNX model
        session = load_onnx_session(onnx_dir / "memory_encoder.onnx")

        # Create test inputs
        B, H, W = 1, 72, 72
        total_stride = tracker_config.mask_downsampler_total_stride

        vision_features = torch.randn(B, 256, H, W)
        masks = torch.randn(B, 1, H * total_stride, W * total_stride)

        # PyTorch forward
        with torch.no_grad():
            pt_features, pt_pos_enc = mem_enc(vision_features, masks)

        # ONNX forward
        onnx_outputs = session.run(None, {
            "vision_features": vision_features.numpy(),
            "masks": masks.numpy(),
        })
        onnx_features = onnx_outputs[0]
        onnx_pos_enc = onnx_outputs[1]

        # Compare
        max_diff_features = np.abs(onnx_features - pt_features.numpy()).max()
        max_diff_pos_enc = np.abs(onnx_pos_enc - pt_pos_enc.numpy()).max()
        max_diff = max(max_diff_features, max_diff_pos_enc)

        duration = (time.time() - start) * 1000

        # Allow small numerical differences
        if max_diff < 1e-4:
            return TestResult(
                name=name,
                passed=True,
                message=f"ONNX matches PyTorch (features diff: {max_diff_features:.2e}, pos_enc diff: {max_diff_pos_enc:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Numerical mismatch (features diff: {max_diff_features:.2e}, pos_enc diff: {max_diff_pos_enc:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


# =============================================================================
# Memory Attention Tests
# =============================================================================

def test_memory_attention_loads(onnx_dir: Path) -> TestResult:
    """Test that memory attention ONNX loads successfully."""
    name = "memory_attention_loads"
    start = time.time()

    try:
        model_path = onnx_dir / "memory_attention.onnx"
        session = load_onnx_session(model_path)
        info = get_model_info(session)
        opset = check_onnx_opset(model_path)

        duration = (time.time() - start) * 1000
        return TestResult(
            name=name,
            passed=True,
            message=f"Loaded successfully (opset {opset}). Inputs: {list(info['inputs'].keys())}",
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_attention_inference(onnx_dir: Path) -> TestResult:
    """Test memory attention inference with sample inputs."""
    name = "memory_attention_inference"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "memory_attention.onnx")

        # SAM3 memory attention: 72*72 = 5184 sequence length
        B = 1
        seq_len = 5184  # 72 * 72
        d_model = 256
        mem_dim = 64

        current_features = np.random.randn(B, seq_len, d_model).astype(np.float32)
        memory = np.random.randn(B, seq_len, mem_dim).astype(np.float32)
        current_pos_enc = np.random.randn(B, seq_len, d_model).astype(np.float32)
        memory_pos_enc = np.random.randn(B, seq_len, mem_dim).astype(np.float32)

        outputs = session.run(None, {
            "current_vision_features": current_features,
            "memory": memory,
            "current_vision_pos_enc": current_pos_enc,
            "memory_pos_enc": memory_pos_enc,
        })

        output = outputs[0]
        expected_shape = (B, seq_len, d_model)

        duration = (time.time() - start) * 1000

        if output.shape == expected_shape:
            return TestResult(
                name=name,
                passed=True,
                message=f"Output shape correct: {output.shape}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Wrong shape: {output.shape}, expected {expected_shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_attention_multi_frame_memory(onnx_dir: Path) -> TestResult:
    """
    Test memory attention with varying memory bank sizes (multiple frames).

    The ONNX export was traced with 7-frame memory (36288 tokens) to capture
    the RoPE repeat_freqs_k logic. This test verifies it works with 1-7 frames.
    """
    name = "memory_attention_multi_frame"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "memory_attention.onnx")

        B = 1
        seq_len = 5184
        d_model = 256
        mem_dim = 64

        current_features = np.random.randn(B, seq_len, d_model).astype(np.float32)
        current_pos_enc = np.random.randn(B, seq_len, d_model).astype(np.float32)

        # Test with 1, 3, 5, 7 frames of memory
        results = []
        for num_frames in [1, 3, 5, 7]:
            mem_seq_len = seq_len * num_frames
            memory = np.random.randn(B, mem_seq_len, mem_dim).astype(np.float32)
            memory_pos_enc = np.random.randn(B, mem_seq_len, mem_dim).astype(np.float32)

            try:
                output = session.run(None, {
                    "current_vision_features": current_features,
                    "memory": memory,
                    "current_vision_pos_enc": current_pos_enc,
                    "memory_pos_enc": memory_pos_enc,
                })[0]

                passed = output.shape == (B, seq_len, d_model)
                results.append((num_frames, passed, output.shape if passed else "shape mismatch"))
            except Exception as e:
                results.append((num_frames, False, str(e)[:60]))

        duration = (time.time() - start) * 1000

        all_passed = all(r[1] for r in results)
        details = ", ".join([f"{r[0]}fr:{'OK' if r[1] else 'FAIL'}" for r in results])

        return TestResult(
            name=name,
            passed=all_passed,
            message=f"Multi-frame memory: {details}",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_attention_vs_pytorch(onnx_dir: Path) -> TestResult:
    """Compare ONNX memory attention output against PyTorch reference."""
    name = "memory_attention_vs_pytorch"
    start = time.time()

    try:
        import torch
        from transformers import Sam3VideoConfig
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoMemoryAttention,
        )
        from transformers.models.sam3_tracker_video.configuration_sam3_tracker_video import (
            Sam3TrackerVideoConfig,
        )
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Load PyTorch model
        full_config = Sam3VideoConfig.from_pretrained('facebook/sam3')
        tracker_config = full_config.tracker_config
        if isinstance(tracker_config, dict):
            tracker_config = Sam3TrackerVideoConfig(**tracker_config)
        tracker_config._attn_implementation = 'eager'

        mem_attn = Sam3TrackerVideoMemoryAttention(tracker_config)

        # Load weights
        weights_file = hf_hub_download("facebook/sam3", "model.safetensors", local_files_only=True)
        all_weights = safetensors.torch.load_file(weights_file)
        prefix = "tracker_model.memory_attention."
        weights = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
        mem_attn.load_state_dict(weights)
        mem_attn.eval()

        # Load ONNX model
        session = load_onnx_session(onnx_dir / "memory_attention.onnx")

        # Create test inputs
        B = 1
        seq_len = 5184
        d_model = 256
        mem_dim = 64

        current_features = torch.randn(B, seq_len, d_model)
        memory = torch.randn(B, seq_len, mem_dim)
        current_pos_enc = torch.randn(B, seq_len, d_model)
        memory_pos_enc = torch.randn(B, seq_len, mem_dim)

        # PyTorch forward (seq-first format)
        with torch.no_grad():
            # The HuggingFace module expects seq-first inputs
            pt_output = mem_attn(
                current_vision_features=current_features.transpose(0, 1),
                memory=memory.transpose(0, 1),
                current_vision_position_embeddings=current_pos_enc.transpose(0, 1),
                memory_posision_embeddings=memory_pos_enc.transpose(0, 1),
                num_object_pointer_tokens=0,
            )
            # Output is (1, B, seq, D), convert to (B, seq, D)
            pt_output = pt_output.transpose(0, 1).squeeze(1)

        # ONNX forward
        onnx_output = session.run(None, {
            "current_vision_features": current_features.numpy(),
            "memory": memory.numpy(),
            "current_vision_pos_enc": current_pos_enc.numpy(),
            "memory_pos_enc": memory_pos_enc.numpy(),
        })[0]

        # Compare
        max_diff = np.abs(onnx_output - pt_output.numpy()).max()
        mean_diff = np.abs(onnx_output - pt_output.numpy()).mean()

        duration = (time.time() - start) * 1000

        if max_diff < 1e-4:
            return TestResult(
                name=name,
                passed=True,
                message=f"ONNX matches PyTorch (max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Numerical mismatch (max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        import traceback
        return TestResult(name=name, passed=False, message=f"{e}\n{traceback.format_exc()}", duration_ms=duration)


# =============================================================================
# Object Pointer Tests
# =============================================================================

def test_object_pointer_loads(onnx_dir: Path) -> TestResult:
    """Test that object pointer ONNX loads successfully."""
    name = "object_pointer_loads"
    start = time.time()

    try:
        model_path = onnx_dir / "object_pointer.onnx"
        session = load_onnx_session(model_path)
        info = get_model_info(session)
        opset = check_onnx_opset(model_path)

        duration = (time.time() - start) * 1000
        return TestResult(
            name=name,
            passed=True,
            message=f"Loaded successfully (opset {opset}). Inputs: {list(info['inputs'].keys())}",
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_object_pointer_inference(onnx_dir: Path) -> TestResult:
    """Test object pointer inference."""
    name = "object_pointer_inference"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "object_pointer.onnx")

        B = 1
        d_model = 256

        # Input: mask decoder output token [B, 1, 256] and objectness score [B, 1]
        output_token = np.random.randn(B, 1, d_model).astype(np.float32)
        object_score_logits = np.array([[1.5]], dtype=np.float32)  # Positive = object present

        outputs = session.run(None, {
            "output_token": output_token,
            "object_score_logits": object_score_logits,
        })

        obj_ptr = outputs[0]
        expected_shape = (B, d_model)

        duration = (time.time() - start) * 1000

        if obj_ptr.shape == expected_shape:
            return TestResult(
                name=name,
                passed=True,
                message=f"Output shape correct: {obj_ptr.shape}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Wrong shape: {obj_ptr.shape}, expected {expected_shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_object_pointer_no_object(onnx_dir: Path) -> TestResult:
    """Test object pointer with no object present (should use no_obj_ptr embedding)."""
    name = "object_pointer_no_object"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "object_pointer.onnx")

        B = 1
        d_model = 256

        output_token = np.random.randn(B, 1, d_model).astype(np.float32)

        # Test with object present vs not present
        obj_present = np.array([[2.0]], dtype=np.float32)   # score > 0
        obj_absent = np.array([[-2.0]], dtype=np.float32)   # score < 0

        ptr_present = session.run(None, {
            "output_token": output_token,
            "object_score_logits": obj_present,
        })[0]

        ptr_absent = session.run(None, {
            "output_token": output_token,
            "object_score_logits": obj_absent,
        })[0]

        # Outputs should be different when object is absent
        diff = np.abs(ptr_present - ptr_absent).max()

        duration = (time.time() - start) * 1000

        if diff > 1e-3:  # Should be meaningfully different
            return TestResult(
                name=name,
                passed=True,
                message=f"Object present/absent outputs differ by {diff:.4f}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Outputs too similar (diff={diff:.6f}), no_obj_ptr not applied correctly",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_object_pointer_vs_pytorch(onnx_dir: Path) -> TestResult:
    """Compare ONNX object pointer output against PyTorch reference."""
    name = "object_pointer_vs_pytorch"
    start = time.time()

    try:
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Recreate the PyTorch object pointer MLP matching the export
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

        # Load weights
        weights_file = hf_hub_download("facebook/sam3", "model.safetensors", local_files_only=True)
        all_weights = safetensors.torch.load_file(weights_file)

        obj_ptr_proj = ObjectPointerMLP()
        prefix = "tracker_model.object_pointer_proj."
        weights = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
        obj_ptr_proj.load_state_dict(weights)
        obj_ptr_proj.eval()

        no_obj_ptr = all_weights["tracker_model.no_object_pointer"]

        # Load ONNX model
        session = load_onnx_session(onnx_dir / "object_pointer.onnx")

        # Test with object present
        B = 1
        d_model = 256
        output_token = torch.randn(B, 1, d_model)
        obj_score_present = torch.tensor([[2.0]])
        obj_score_absent = torch.tensor([[-2.0]])

        # PyTorch forward - object present
        with torch.no_grad():
            pt_ptr = obj_ptr_proj(output_token).squeeze(1)  # [B, 256]
            is_obj_present = (obj_score_present > 0).float()
            pt_ptr_present = pt_ptr * is_obj_present + no_obj_ptr * (1 - is_obj_present)

            is_obj_present = (obj_score_absent > 0).float()
            pt_ptr_absent = pt_ptr * is_obj_present + no_obj_ptr * (1 - is_obj_present)

        # ONNX forward
        onnx_ptr_present = session.run(None, {
            "output_token": output_token.numpy(),
            "object_score_logits": obj_score_present.numpy(),
        })[0]

        onnx_ptr_absent = session.run(None, {
            "output_token": output_token.numpy(),
            "object_score_logits": obj_score_absent.numpy(),
        })[0]

        # Compare
        max_diff_present = np.abs(onnx_ptr_present - pt_ptr_present.numpy()).max()
        max_diff_absent = np.abs(onnx_ptr_absent - pt_ptr_absent.numpy()).max()
        max_diff = max(max_diff_present, max_diff_absent)

        duration = (time.time() - start) * 1000

        if max_diff < 1e-4:
            return TestResult(
                name=name,
                passed=True,
                message=f"ONNX matches PyTorch (present diff: {max_diff_present:.2e}, absent diff: {max_diff_absent:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Numerical mismatch (present diff: {max_diff_present:.2e}, absent diff: {max_diff_absent:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        import traceback
        return TestResult(name=name, passed=False, message=f"{e}\n{traceback.format_exc()}", duration_ms=duration)


def test_object_pointer_batch(onnx_dir: Path) -> TestResult:
    """Test object pointer with different batch sizes."""
    name = "object_pointer_batch"
    start = time.time()

    try:
        session = load_onnx_session(onnx_dir / "object_pointer.onnx")

        d_model = 256
        results = []

        for B in [1, 2, 4]:
            output_token = np.random.randn(B, 1, d_model).astype(np.float32)
            object_score_logits = np.random.randn(B, 1).astype(np.float32)

            try:
                obj_ptr = session.run(None, {
                    "output_token": output_token,
                    "object_score_logits": object_score_logits,
                })[0]

                passed = obj_ptr.shape == (B, d_model)
                results.append((B, passed))
            except Exception as e:
                results.append((B, False))

        duration = (time.time() - start) * 1000

        all_passed = all(r[1] for r in results)
        details = ", ".join([f"B={r[0]}:{'OK' if r[1] else 'FAIL'}" for r in results])

        return TestResult(
            name=name,
            passed=all_passed,
            message=f"Batch tests: {details}",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


# =============================================================================
# Temporal Position Encoding Tests
# =============================================================================

def test_temporal_pos_enc_loads(onnx_dir: Path) -> TestResult:
    """Test that temporal position encoding loads correctly."""
    name = "temporal_pos_enc_loads"
    start = time.time()

    try:
        npy_path = onnx_dir / "temporal_pos_enc.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"File not found: {npy_path}")

        temporal_enc = np.load(npy_path)

        # Expected shape: [num_maskmem, 1, 1, 64] or similar
        duration = (time.time() - start) * 1000

        if len(temporal_enc.shape) == 4:
            return TestResult(
                name=name,
                passed=True,
                message=f"Loaded successfully. Shape: {temporal_enc.shape}, dtype: {temporal_enc.dtype}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Unexpected shape: {temporal_enc.shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_temporal_pos_enc_vs_pytorch(onnx_dir: Path) -> TestResult:
    """Compare exported temporal position encoding against PyTorch weights."""
    name = "temporal_pos_enc_vs_pytorch"
    start = time.time()

    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Load exported numpy array
        npy_path = onnx_dir / "temporal_pos_enc.npy"
        exported_enc = np.load(npy_path)

        # Load original PyTorch weights
        weights_file = hf_hub_download("facebook/sam3", "model.safetensors", local_files_only=True)
        all_weights = safetensors.torch.load_file(weights_file)

        pt_temporal_enc = all_weights["tracker_model.memory_temporal_positional_encoding"]
        pt_enc = pt_temporal_enc.numpy()

        # Compare
        max_diff = np.abs(exported_enc - pt_enc).max()

        duration = (time.time() - start) * 1000

        if max_diff < 1e-6:
            return TestResult(
                name=name,
                passed=True,
                message=f"Matches PyTorch exactly (max_diff: {max_diff:.2e}). Shape: {exported_enc.shape}",
                duration_ms=duration,
                max_diff=max_diff,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Mismatch (max_diff: {max_diff:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        import traceback
        return TestResult(name=name, passed=False, message=f"{e}\n{traceback.format_exc()}", duration_ms=duration)


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

def test_video_propagation_pipeline(onnx_dir: Path) -> TestResult:
    """
    Test simulated video propagation pipeline with growing memory bank.

    Simulates full 5-frame video propagation where each new frame can
    access memory from all previous frames (up to 7 max).
    """
    name = "video_propagation_pipeline"
    start = time.time()

    try:
        # Load all components
        encoder_session = load_onnx_session(onnx_dir / "memory_encoder.onnx")
        attention_session = load_onnx_session(onnx_dir / "memory_attention.onnx")
        pointer_session = load_onnx_session(onnx_dir / "object_pointer.onnx")
        temporal_enc = np.load(onnx_dir / "temporal_pos_enc.npy")

        # Simulate 5-frame video propagation with growing memory bank
        B, H, W = 1, 72, 72
        seq_len = H * W
        d_model = 256
        mem_dim = 64
        total_stride = 16

        memory_bank = []  # List of [B, seq_len, 64] arrays
        memory_pos_bank = []

        for frame_idx in range(5):
            # Simulate vision features for this frame
            vision_features = np.random.randn(B, d_model, H, W).astype(np.float32)
            masks = np.random.randn(B, 1, H * total_stride, W * total_stride).astype(np.float32)

            # Step 1: If we have previous memory, condition features
            if len(memory_bank) > 0:
                current_features = vision_features.reshape(B, d_model, -1).transpose(0, 2, 1)  # [B, seq, 256]
                current_pos_enc = np.random.randn(B, seq_len, d_model).astype(np.float32)

                # Concatenate all memory frames
                all_memory = np.concatenate(memory_bank, axis=1)  # [B, N*seq_len, 64]
                all_pos = np.concatenate(memory_pos_bank, axis=1)

                conditioned_features = attention_session.run(None, {
                    "current_vision_features": current_features,
                    "memory": all_memory,
                    "current_vision_pos_enc": current_pos_enc,
                    "memory_pos_enc": all_pos,
                })[0]

                # Verify output shape
                if conditioned_features.shape != (B, seq_len, d_model):
                    raise ValueError(f"Wrong conditioned features shape at frame {frame_idx}: {conditioned_features.shape}")

            # Step 2: Encode current frame into memory for next frame
            memory_out = encoder_session.run(None, {
                "vision_features": vision_features,
                "masks": masks,
            })
            frame_memory = memory_out[0]  # [B, 64, H, W]
            frame_pos_enc = memory_out[1]  # [B, 64, H, W]

            # Flatten spatial dims for attention
            frame_memory_flat = frame_memory.reshape(B, mem_dim, -1).transpose(0, 2, 1)  # [B, seq, 64]
            frame_pos_flat = frame_pos_enc.reshape(B, mem_dim, -1).transpose(0, 2, 1)

            # Add temporal encoding
            t_idx = min(frame_idx, temporal_enc.shape[0] - 1)
            frame_pos_flat = frame_pos_flat + temporal_enc[t_idx, :, :, :mem_dim]

            memory_bank.append(frame_memory_flat)
            memory_pos_bank.append(frame_pos_flat)

            # Step 3: Simulate object pointer extraction
            output_token = np.random.randn(B, 1, d_model).astype(np.float32)
            obj_score = np.array([[1.0]], dtype=np.float32)

            obj_ptr = pointer_session.run(None, {
                "output_token": output_token,
                "object_score_logits": obj_score,
            })[0]

            if obj_ptr.shape != (B, d_model):
                raise ValueError(f"Wrong object pointer shape at frame {frame_idx}: {obj_ptr.shape}")

        duration = (time.time() - start) * 1000

        return TestResult(
            name=name,
            passed=True,
            message=f"Successfully propagated through 5 frames with growing memory bank (1→4 frames)",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        import traceback
        return TestResult(name=name, passed=False, message=f"{e}\n{traceback.format_exc()}", duration_ms=duration)


def test_opset_compatibility(onnx_dir: Path) -> TestResult:
    """Test that all models use opset <= 18 for browser compatibility."""
    name = "opset_compatibility"
    start = time.time()

    try:
        import onnx

        max_allowed_opset = 18  # onnxruntime-web 1.14.0 max
        results = []

        for model_name in ["memory_encoder.onnx", "memory_attention.onnx", "object_pointer.onnx"]:
            model_path = onnx_dir / model_name
            if model_path.exists():
                opset = check_onnx_opset(model_path)
                ok = opset <= max_allowed_opset
                results.append((model_name, opset, ok))

        duration = (time.time() - start) * 1000

        all_ok = all(r[2] for r in results)
        details = ", ".join([f"{r[0]}:opset{r[1]}" for r in results])

        return TestResult(
            name=name,
            passed=all_ok,
            message=f"Opset versions: {details} (max allowed: {max_allowed_opset})",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests(onnx_dir: Path, test_suite: str = "all", verbose: bool = False) -> Tuple[int, int]:
    """Run tests and return (passed, total) counts."""
    results = []

    # Define test suites
    encoder_tests = [
        test_memory_encoder_loads,
        test_memory_encoder_inference,
        test_memory_encoder_vs_pytorch,
    ]

    attention_tests = [
        test_memory_attention_loads,
        test_memory_attention_inference,
        test_memory_attention_multi_frame_memory,
        test_memory_attention_vs_pytorch,
    ]

    pointer_tests = [
        test_object_pointer_loads,
        test_object_pointer_inference,
        test_object_pointer_no_object,
        test_object_pointer_vs_pytorch,
        test_object_pointer_batch,
    ]

    temporal_tests = [
        test_temporal_pos_enc_loads,
        test_temporal_pos_enc_vs_pytorch,
    ]

    pipeline_tests = [
        test_video_propagation_pipeline,
        test_opset_compatibility,
    ]

    # Select tests based on suite
    if test_suite == "encoder":
        all_tests = [("Memory Encoder", encoder_tests)]
    elif test_suite == "attention":
        all_tests = [("Memory Attention", attention_tests)]
    elif test_suite == "pointer":
        all_tests = [("Object Pointer", pointer_tests)]
    elif test_suite == "temporal":
        all_tests = [("Temporal Position Encoding", temporal_tests)]
    elif test_suite == "pipeline":
        all_tests = [("Pipeline Integration", pipeline_tests)]
    else:  # all
        all_tests = [
            ("Memory Encoder", encoder_tests),
            ("Memory Attention", attention_tests),
            ("Object Pointer", pointer_tests),
            ("Temporal Position Encoding", temporal_tests),
            ("Pipeline Integration", pipeline_tests),
        ]

    total_passed = 0
    total_tests = 0

    for suite_name, tests in all_tests:
        print_section(suite_name)

        for test_func in tests:
            result = test_func(onnx_dir)
            results.append(result)
            print_result(result, verbose)

            if result.passed:
                total_passed += 1
            total_tests += 1

    return total_passed, total_tests


def main():
    parser = argparse.ArgumentParser(description="SAM3 Memory Components ONNX Test Suite")
    parser.add_argument("--model-dir", type=Path, default=ONNX_DIR,
                        help="Directory containing ONNX models")
    parser.add_argument("--test", choices=["all", "encoder", "attention", "pointer", "temporal", "pipeline"],
                        default="all", help="Test suite to run")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    print(f"{Colors.BOLD}SAM3 Memory Components ONNX Test Suite{Colors.END}")
    print(f"Model directory: {args.model_dir}")
    print(f"Test suite: {args.test}")

    if not args.model_dir.exists():
        print(f"{Colors.RED}Error: Model directory not found: {args.model_dir}{Colors.END}")
        sys.exit(1)

    # List available models
    print(f"\nAvailable models:")
    for f in sorted(args.model_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name}: {size / (1024*1024):.2f} MB")
        else:
            print(f"  {f.name}: {size / 1024:.2f} KB")

    passed, total = run_tests(args.model_dir, args.test, args.verbose)

    # Summary
    print_section("Summary")
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}All {total} tests passed!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}{passed}/{total} tests passed ({total - passed} failed){Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
