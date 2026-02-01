#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 ONNX Tests (pytest compatible)

Run with:
    pytest test_onnx_pytest.py -v --model-dir=/path/to/onnx-exports

Or set SAM3_MODEL_DIR environment variable.
"""

import os
from pathlib import Path

import numpy as np
import pytest


def get_model_dir():
    """Get model directory from env or pytest args."""
    return Path(os.environ.get("SAM3_MODEL_DIR", "./onnx-exports"))


@pytest.fixture(scope="module")
def model_dir():
    """Fixture providing model directory path."""
    return get_model_dir()


@pytest.fixture(scope="module")
def vision_encoder_session(model_dir):
    """Fixture providing vision encoder ONNX session."""
    import onnxruntime as ort

    path = model_dir / "vision_encoder.onnx"
    if not path.exists():
        pytest.skip(f"Vision encoder not found: {path}")

    return ort.InferenceSession(
        str(path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )


@pytest.fixture(scope="module")
def tracker_decoder_session(model_dir):
    """Fixture providing tracker decoder ONNX session."""
    import onnxruntime as ort

    path = model_dir / "tracker_decoder.onnx"
    if not path.exists():
        pytest.skip(f"Tracker decoder not found: {path}")

    return ort.InferenceSession(
        str(path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )


@pytest.fixture(scope="module")
def text_encoder_session(model_dir):
    """Fixture providing text encoder ONNX session."""
    import onnxruntime as ort

    path = model_dir / "text_encoder.onnx"
    if not path.exists():
        pytest.skip(f"Text encoder not found: {path}")

    return ort.InferenceSession(
        str(path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )


class TestVisionEncoder:
    """Vision encoder ONNX tests."""

    def test_input_shape(self, vision_encoder_session):
        """Verify expected input shape."""
        input_info = vision_encoder_session.get_inputs()[0]
        assert input_info.name in ["pixel_values", "input"]
        # Shape should be [1, 3, H, W] where H, W are dynamic or 1008
        assert input_info.shape[0] == 1
        assert input_info.shape[1] == 3

    def test_output_count(self, vision_encoder_session):
        """Verify we get 4 outputs."""
        outputs = vision_encoder_session.get_outputs()
        assert len(outputs) >= 4

    def test_inference_shapes(self, vision_encoder_session):
        """Test inference produces correct output shapes."""
        test_input = np.random.randn(1, 3, 1008, 1008).astype(np.float32)
        input_name = vision_encoder_session.get_inputs()[0].name

        outputs = vision_encoder_session.run(None, {input_name: test_input})

        # Check shapes (256ch for all FPN levels)
        assert outputs[0].shape == (1, 256, 288, 288), "fpn_feat_0 shape"
        assert outputs[1].shape == (1, 256, 144, 144), "fpn_feat_1 shape"
        assert outputs[2].shape == (1, 256, 72, 72), "fpn_feat_2 shape"
        assert outputs[3].shape == (1, 256, 72, 72), "fpn_pos_2 shape"

    def test_output_finite(self, vision_encoder_session):
        """Verify outputs contain finite values."""
        test_input = np.random.randn(1, 3, 1008, 1008).astype(np.float32)
        input_name = vision_encoder_session.get_inputs()[0].name

        outputs = vision_encoder_session.run(None, {input_name: test_input})

        for i, out in enumerate(outputs):
            assert np.isfinite(out).all(), f"Output {i} contains non-finite values"


class TestTrackerDecoder:
    """Tracker decoder ONNX tests."""

    def test_inference_with_point(self, tracker_decoder_session):
        """Test inference with point prompt."""
        # HuggingFace SAM3 expects 4D point_coords [B, num_objects, num_points, 2]
        test_inputs = {
            "fpn_feat_0": np.random.randn(1, 256, 288, 288).astype(np.float32),
            "fpn_feat_1": np.random.randn(1, 256, 144, 144).astype(np.float32),
            "fpn_feat_2": np.random.randn(1, 256, 72, 72).astype(np.float32),
            "point_coords": np.array([[[[504.0, 504.0]]]], dtype=np.float32),  # [B, num_objects, num_points, 2]
            "point_labels": np.array([[[1.0]]], dtype=np.float32),  # [B, num_objects, num_points]
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0.0], dtype=np.float32),
        }

        outputs = tracker_decoder_session.run(None, test_inputs)

        # Masks should be [1, 3, 1008, 1008]
        assert outputs[0].shape == (1, 3, 1008, 1008), "masks shape"
        # IoU predictions should be [1, 3]
        assert outputs[1].shape == (1, 3), "iou_predictions shape"
        # Low-res masks should be [1, 3, 288, 288]
        assert outputs[2].shape == (1, 3, 288, 288), "low_res_masks shape"

    def test_output_finite(self, tracker_decoder_session):
        """Verify outputs contain finite values."""
        # HuggingFace SAM3 expects 4D point_coords [B, num_objects, num_points, 2]
        test_inputs = {
            "fpn_feat_0": np.random.randn(1, 256, 288, 288).astype(np.float32),
            "fpn_feat_1": np.random.randn(1, 256, 144, 144).astype(np.float32),
            "fpn_feat_2": np.random.randn(1, 256, 72, 72).astype(np.float32),
            "point_coords": np.array([[[[504.0, 504.0]]]], dtype=np.float32),  # [B, num_objects, num_points, 2]
            "point_labels": np.array([[[1.0]]], dtype=np.float32),  # [B, num_objects, num_points]
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0.0], dtype=np.float32),
        }

        outputs = tracker_decoder_session.run(None, test_inputs)

        for i, out in enumerate(outputs):
            assert np.isfinite(out).all(), f"Output {i} contains non-finite values"


class TestTextEncoder:
    """Text encoder ONNX tests."""

    def test_inference(self, text_encoder_session):
        \"\"\"Test basic text encoding.\"\"\"
        batch_size = 1
        seq_len = 32  # SAM3 uses 32 token context length
        test_inputs = {
            "input_ids": np.ones((batch_size, seq_len), dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        }

        outputs = text_encoder_session.run(None, test_inputs)

        assert len(outputs) >= 2
        assert outputs[0].shape[0] == batch_size, "batch dimension"

    def test_output_finite(self, text_encoder_session):
        \"\"\"Verify outputs contain finite values.\"\"\"
        test_inputs = {
            "input_ids": np.ones((1, 32), dtype=np.int64),  # SAM3 context length
            "attention_mask": np.ones((1, 32), dtype=np.int64),
        }

        outputs = text_encoder_session.run(None, test_inputs)

        for i, out in enumerate(outputs):
            assert np.isfinite(out).all(), f"Output {i} contains non-finite values"


class TestUnifiedHandler:
    """Tests for the unified model handler."""

    @pytest.fixture
    def handler(self, model_dir):
        """Get handler instance."""
        import sys

        os.environ["SAM3_MODEL_DIR"] = str(model_dir)
        os.environ["SAM3_VISION_ENCODER"] = str(model_dir / "vision_encoder.onnx")
        os.environ["SAM3_TEXT_ENCODER"] = str(model_dir / "text_encoder.onnx")
        os.environ["SAM3_PCS_DECODER"] = str(model_dir / "pcs_decoder.onnx")
        os.environ["SAM3_TRACKER_DECODER"] = str(model_dir / "tracker_decoder.onnx")

        handler_dir = Path(__file__).parent
        sys.path.insert(0, str(handler_dir))

        from model_handler import get_handler
        return get_handler()

    def test_get_model_info(self, handler):
        """Test model info retrieval."""
        info = handler.get_model_info()
        assert "vision_encoder" in info
        assert "tracker_decoder" in info

    def test_encode_rgb_image(self, handler):
        """Test encoding RGB image."""
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        embeddings = handler.encode(img)

        # Should have at least fpn features
        assert len(embeddings) >= 3

    def test_encode_returns_numpy(self, handler):
        """Verify encode returns numpy arrays."""
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        embeddings = handler.encode(img)

        for name, arr in embeddings.items():
            assert isinstance(arr, np.ndarray), f"{name} is not numpy array"
            assert arr.dtype == np.float32 or arr.dtype == np.float16


# Comparison tests (require HuggingFace models)
class TestONNXvsPyTorch:
    """Tests comparing ONNX to PyTorch reference (requires HF auth)."""

    @pytest.fixture
    def pytorch_model(self):
        """Load PyTorch model for comparison."""
        try:
            import torch
            from transformers import SAM2Model
        except ImportError:
            pytest.skip("transformers or torch not available")

        try:
            model = SAM2Model.from_pretrained(
                "facebook/sam2.1-hiera-large",
                trust_remote_code=True
            )
            model.eval()
            return model
        except Exception as e:
            pytest.skip(f"Could not load PyTorch model: {e}")

    @pytest.mark.slow
    def test_vision_encoder_equivalence(self, vision_encoder_session, pytorch_model):
        """Compare vision encoder outputs."""
        import torch

        # Random test image
        np.random.seed(42)
        test_input = np.random.randn(1, 3, 1008, 1008).astype(np.float32)

        # ONNX inference
        input_name = vision_encoder_session.get_inputs()[0].name
        onnx_outputs = vision_encoder_session.run(None, {input_name: test_input})

        # PyTorch inference
        with torch.no_grad():
            torch_input = torch.from_numpy(test_input)
            torch_outputs = pytorch_model.vision_encoder(torch_input)

        # Compare (just first level for now)
        onnx_out = onnx_outputs[0]
        torch_out = torch_outputs.backbone_fpn[0].cpu().numpy()

        mae = np.mean(np.abs(onnx_out - torch_out))
        max_diff = np.max(np.abs(onnx_out - torch_out))

        assert mae < 0.001, f"MAE too high: {mae}"
        assert max_diff < 0.01, f"Max diff too high: {max_diff}"
