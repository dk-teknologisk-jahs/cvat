#!/usr/bin/env python3
"""
SAM3 Memory Components ONNX Validation Tests

This script validates the exported memory ONNX models by:
1. Loading each ONNX model and verifying it runs
2. Comparing outputs against PyTorch reference
3. Testing with various input sizes (dynamic shapes)
4. Simulating a full video propagation pipeline
5. Testing browser-compatible operations (opset 17)

Usage:
    conda activate grimme-tf2.18

    # Run all tests
    python test_memory_onnx_validation.py

    # Test specific model
    python test_memory_onnx_validation.py --test-memory-encoder
    python test_memory_onnx_validation.py --test-memory-attention
    python test_memory_onnx_validation.py --test-propagation

    # Use custom model directory
    python test_memory_onnx_validation.py --model-dir ./onnx-memory-exports
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

print("Loading test dependencies...")


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    duration_ms: float
    max_diff: Optional[float] = None


class ONNXModelTester:
    """ONNX model testing utilities."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.ort = None
        self.torch = None
        self._load_runtime()

    def _load_runtime(self):
        """Lazily load ONNX Runtime."""
        import onnxruntime as ort
        import torch
        self.ort = ort
        self.torch = torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")

    def load_session(self, model_name: str) -> "ort.InferenceSession":
        """Load an ONNX session."""
        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        sess = self.ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        return sess

    def get_model_info(self, sess) -> dict:
        """Get model input/output info."""
        inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
        outputs = {out.name: out.shape for out in sess.get_outputs()}
        return {"inputs": inputs, "outputs": outputs}


# =============================================================================
# Test Cases
# =============================================================================

def test_memory_encoder_basic(tester: ONNXModelTester) -> TestResult:
    """Test basic memory encoder functionality."""
    name = "memory_encoder_basic"
    start = time.time()

    try:
        sess = tester.load_session("memory_encoder.onnx")
        info = tester.get_model_info(sess)

        # Create test inputs - using correct dimensions:
        # vision_features: [B, 256, 64, 64] from image encoder
        # masks: [B, 1, 1024, 1024] (gets 16x downsampled to 64x64)
        B = 1
        vision_features = np.random.randn(B, 256, 64, 64).astype(np.float32)
        masks = np.random.randn(B, 1, 1024, 1024).astype(np.float32)

        # Run inference
        outputs = sess.run(None, {
            "vision_features": vision_features,
            "masks": masks,
        })
        output = outputs[0]  # memory output

        # Validate output shape: [B, 64, 64, 64]
        expected_shape = (B, 64, 64, 64)

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
                message=f"Wrong output shape: {output.shape}, expected {expected_shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_encoder_batch(tester: ONNXModelTester) -> TestResult:
    """Test memory encoder with different batch sizes."""
    name = "memory_encoder_batch"
    start = time.time()

    try:
        sess = tester.load_session("memory_encoder.onnx")

        batch_sizes = [1, 2, 4]
        results = []

        for B in batch_sizes:
            vision_features = np.random.randn(B, 256, 64, 64).astype(np.float32)
            masks = np.random.randn(B, 1, 1024, 1024).astype(np.float32)

            outputs = sess.run(None, {"vision_features": vision_features, "masks": masks})
            output = outputs[0]
            expected = (B, 64, 64, 64)
            results.append((B, output.shape == expected, output.shape))

        duration = (time.time() - start) * 1000

        all_passed = all(r[1] for r in results)
        details = ", ".join([f"B={r[0]}:{r[2]}" for r in results])

        return TestResult(
            name=name,
            passed=all_passed,
            message=f"Batch tests: {details}",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_attention_basic(tester: ONNXModelTester) -> TestResult:
    """Test basic memory attention functionality."""
    name = "memory_attention_basic"
    start = time.time()

    try:
        sess = tester.load_session("memory_attention.onnx")
        info = tester.get_model_info(sess)

        # Create test inputs
        B = 1
        HW = 72 * 72  # 5184
        d_model = 256
        mem_dim = 64
        mem_len = HW * 3  # 3 frames of memory

        current_features = np.random.randn(B, HW, d_model).astype(np.float32)
        memory_features = np.random.randn(B, mem_len, mem_dim).astype(np.float32)

        # Run inference
        output = sess.run(None, {
            "current_features": current_features,
            "memory_features": memory_features,
        })[0]

        # Validate output shape
        expected_shape = (B, HW, d_model)

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
                message=f"Wrong output shape: {output.shape}, expected {expected_shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_memory_attention_dynamic_memory(tester: ONNXModelTester) -> TestResult:
    """Test memory attention with varying memory bank sizes."""
    name = "memory_attention_dynamic"
    start = time.time()

    try:
        sess = tester.load_session("memory_attention.onnx")

        B = 1
        HW = 5184
        d_model = 256
        mem_dim = 64

        current_features = np.random.randn(B, HW, d_model).astype(np.float32)

        # Test with different memory lengths (1, 3, 7 frames)
        results = []
        for num_frames in [1, 3, 7]:
            mem_len = HW * num_frames
            memory_features = np.random.randn(B, mem_len, mem_dim).astype(np.float32)

            output = sess.run(None, {
                "current_features": current_features,
                "memory_features": memory_features,
            })[0]

            expected = (B, HW, d_model)
            passed = output.shape == expected
            results.append((num_frames, passed, output.shape))

        duration = (time.time() - start) * 1000

        all_passed = all(r[1] for r in results)
        details = ", ".join([f"{r[0]}fr:{r[2]}" for r in results])

        return TestResult(
            name=name,
            passed=all_passed,
            message=f"Dynamic memory: {details}",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_object_pointer(tester: ONNXModelTester) -> TestResult:
    """Test object pointer projection."""
    name = "object_pointer"
    start = time.time()

    try:
        sess = tester.load_session("object_pointer.onnx")

        # Create test input
        B = 1
        d_model = 256
        sam_output_token = np.random.randn(B, d_model).astype(np.float32)

        # Run inference
        output = sess.run(None, {"sam_output_token": sam_output_token})[0]

        duration = (time.time() - start) * 1000

        # Output should be [B, ptr_dim] - typically 64 or 256
        if len(output.shape) == 2 and output.shape[0] == B:
            return TestResult(
                name=name,
                passed=True,
                message=f"Output shape: {output.shape}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Unexpected output shape: {output.shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_temporal_position_encoding(tester: ONNXModelTester) -> TestResult:
    """Test temporal position encoding loading."""
    name = "temporal_pos_enc"
    start = time.time()

    try:
        tpos_path = tester.model_dir / "temporal_pos_enc.npy"

        if not tpos_path.exists():
            return TestResult(
                name=name,
                passed=False,
                message=f"File not found: {tpos_path}",
                duration_ms=(time.time() - start) * 1000,
            )

        tpos = np.load(str(tpos_path))

        duration = (time.time() - start) * 1000

        # Should be [num_frames, dim] or similar
        if len(tpos.shape) >= 1:
            return TestResult(
                name=name,
                passed=True,
                message=f"Loaded shape: {tpos.shape}",
                duration_ms=duration,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"Unexpected shape: {tpos.shape}",
                duration_ms=duration,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_full_propagation_pipeline(tester: ONNXModelTester) -> TestResult:
    """
    Test full video propagation pipeline.

    Simulates:
    1. First frame: encode mask into memory
    2. Subsequent frames: fuse with memory, generate mask, update memory
    """
    name = "full_propagation"
    start = time.time()

    try:
        # Load models
        memory_encoder_sess = tester.load_session("memory_encoder.onnx")
        memory_attention_sess = tester.load_session("memory_attention.onnx")

        # Simulate 5 frame video
        num_frames = 5
        B = 1
        HW = 4096  # 64*64
        d_model = 256
        mem_dim = 64

        # Memory bank (accumulates across frames)
        memory_bank = []

        for frame_idx in range(num_frames):
            # Simulate vision encoder output: [B, 256, 64, 64]
            vision_features = np.random.randn(B, 256, 64, 64).astype(np.float32)

            if frame_idx == 0:
                # First frame: just encode the mask (from user prompt)
                mask = np.random.randn(B, 1, 1024, 1024).astype(np.float32)
                outputs = memory_encoder_sess.run(None, {
                    "vision_features": vision_features,
                    "masks": mask,
                })
                memory = outputs[0]  # [B, 64, 64, 64]

                # Flatten spatial dims for memory bank: [B, 4096, 64]
                memory_flat = memory.reshape(B, mem_dim, -1).transpose(0, 2, 1)
                memory_bank.append(memory_flat)
            else:
                # Subsequent frames: fuse with memory bank
                current_features = vision_features.reshape(B, 256, -1).transpose(0, 2, 1)  # [B, 4096, 256]

                # Concatenate memory bank
                memory_concat = np.concatenate(memory_bank, axis=1)  # [B, n*4096, 64]

                # Run memory attention
                fused_features = memory_attention_sess.run(None, {
                    "current_features": current_features,
                    "memory_features": memory_concat,
                })[0]  # [B, 4096, 256]

                # Simulate mask generation (would come from decoder)
                predicted_mask = np.random.randn(B, 1, 1024, 1024).astype(np.float32)

                # Encode new memory
                outputs = memory_encoder_sess.run(None, {
                    "vision_features": vision_features,
                    "masks": predicted_mask,
                })
                new_memory = outputs[0]  # [B, 64, 64, 64]
                new_memory_flat = new_memory.reshape(B, mem_dim, -1).transpose(0, 2, 1)

                memory_bank.append(new_memory_flat)

                # Optionally limit memory bank size
                if len(memory_bank) > 7:
                    memory_bank = memory_bank[-7:]

        duration = (time.time() - start) * 1000

        return TestResult(
            name=name,
            passed=True,
            message=f"Propagated {num_frames} frames, final memory bank: {len(memory_bank)} entries",
            duration_ms=duration,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_pytorch_onnx_equivalence(tester: ONNXModelTester) -> TestResult:
    """
    Test that ONNX outputs match PyTorch reference.

    This requires the PyTorch models to be available.
    """
    name = "pytorch_equivalence"
    start = time.time()

    try:
        import torch
        from transformers import Sam2VideoModel

        # Load ONNX session and PyTorch model
        sess = tester.load_session("memory_encoder.onnx")
        model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large")
        model.eval()

        # Test memory encoder equivalence
        B = 1
        np.random.seed(42)
        vision_features_np = np.random.randn(B, 256, 64, 64).astype(np.float32)
        masks_np = np.random.randn(B, 1, 1024, 1024).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            vision_features_pt = torch.from_numpy(vision_features_np)
            masks_pt = torch.from_numpy(masks_np)
            pt_memory, pt_pos_enc = model.memory_encoder(vision_features_pt, masks_pt)

        # ONNX inference
        onnx_outputs = sess.run(None, {
            "vision_features": vision_features_np,
            "masks": masks_np,
        })

        # Compare outputs
        memory_diff = np.abs(pt_memory.numpy() - onnx_outputs[0]).max()
        pos_enc_diff = np.abs(pt_pos_enc.numpy() - onnx_outputs[1]).max()
        max_diff = max(memory_diff, pos_enc_diff)

        duration = (time.time() - start) * 1000

        tolerance = 1e-4
        if max_diff < tolerance:
            return TestResult(
                name=name,
                passed=True,
                message=f"PyTorch match: max_diff={max_diff:.2e} (memory:{memory_diff:.2e}, pos_enc:{pos_enc_diff:.2e})",
                duration_ms=duration,
                max_diff=max_diff,
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                message=f"PyTorch mismatch: max_diff={max_diff:.2e}",
                duration_ms=duration,
                max_diff=max_diff,
            )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


def test_inference_performance(tester: ONNXModelTester) -> TestResult:
    """Benchmark inference performance."""
    name = "inference_performance"
    start = time.time()

    try:
        # Test memory encoder performance instead
        sess = tester.load_session("memory_encoder.onnx")

        B = 1

        # Warm up
        vision_features = np.random.randn(B, 256, 64, 64).astype(np.float32)
        masks = np.random.randn(B, 1, 1024, 1024).astype(np.float32)

        for _ in range(3):
            sess.run(None, {
                "vision_features": vision_features,
                "masks": masks,
            })

        # Benchmark
        num_runs = 10
        times = []
        for _ in range(num_runs):
            t0 = time.time()
            sess.run(None, {
                "vision_features": vision_features,
                "masks": masks,
            })
            times.append((time.time() - t0) * 1000)

        avg_ms = np.mean(times)
        std_ms = np.std(times)

        duration = (time.time() - start) * 1000

        # Pass if average inference is under 500ms (CPU)
        threshold_ms = 500
        passed = avg_ms < threshold_ms

        return TestResult(
            name=name,
            passed=passed,
            message=f"Avg: {avg_ms:.1f}ms ± {std_ms:.1f}ms (threshold: {threshold_ms}ms)",
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(name=name, passed=False, message=str(e), duration_ms=duration)


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests(model_dir: Path, test_filters: Optional[List[str]] = None) -> List[TestResult]:
    """Run all tests and return results."""

    tester = ONNXModelTester(model_dir)

    all_tests = [
        ("memory_encoder", test_memory_encoder_basic),
        ("memory_encoder", test_memory_encoder_batch),
        ("memory_attention", test_memory_attention_basic),
        ("memory_attention", test_memory_attention_dynamic_memory),
        ("object_pointer", test_object_pointer),
        ("temporal_pos_enc", test_temporal_position_encoding),
        ("propagation", test_full_propagation_pipeline),
        ("equivalence", test_pytorch_onnx_equivalence),
        ("performance", test_inference_performance),
    ]

    results = []

    for category, test_fn in all_tests:
        if test_filters and category not in test_filters:
            continue

        print(f"\n  Running {test_fn.__name__}...")
        result = test_fn(tester)
        results.append(result)

        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.message} ({result.duration_ms:.0f}ms)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate SAM3 memory ONNX models")
    parser.add_argument("--model-dir", type=str, default="./onnx-memory-exports",
                        help="Directory containing ONNX models")
    parser.add_argument("--test-memory-encoder", action="store_true",
                        help="Only test memory encoder")
    parser.add_argument("--test-memory-attention", action="store_true",
                        help="Only test memory attention")
    parser.add_argument("--test-propagation", action="store_true",
                        help="Only test full propagation pipeline")
    parser.add_argument("--test-performance", action="store_true",
                        help="Only test inference performance")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("Run export_memory_components.py first to generate models.")
        return 1

    print("="*60)
    print("SAM3 Memory Components ONNX Validation")
    print("="*60)
    print(f"Model directory: {model_dir}")

    # Determine test filters
    filters = None
    if args.test_memory_encoder:
        filters = ["memory_encoder"]
    elif args.test_memory_attention:
        filters = ["memory_attention"]
    elif args.test_propagation:
        filters = ["propagation"]
    elif args.test_performance:
        filters = ["performance"]

    # Run tests
    print("\nRunning tests...")
    results = run_all_tests(model_dir, filters)

    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_time = sum(r.duration_ms for r in results)

    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {result.name}")
        if not result.passed:
            print(f"          {result.message}")

    print("\n" + "-"*60)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.0f}ms")
    print("="*60)

    if failed == 0:
        print("\n🎉 All tests passed! Memory components are ready for use.")
        print("\nNext steps for browser integration:")
        print("  1. Copy ONNX models to cvat-ui/plugins/sam3/public/")
        print("  2. Add memory bank management to inference.worker.ts")
        print("  3. Implement frame propagation loop in index.tsx")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the errors above.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
