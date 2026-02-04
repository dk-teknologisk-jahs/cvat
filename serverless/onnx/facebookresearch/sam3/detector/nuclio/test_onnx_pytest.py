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
        providers=['CPUExecutionProvider']
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
        providers=['CPUExecutionProvider']
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
        providers=['CPUExecutionProvider']
    )


class TestVisionEncoder:
    """Vision encoder ONNX tests."""

    def test_input_shape(self, vision_encoder_session):
        """Verify expected input shape."""
        input_info = vision_encoder_session.get_inputs()[0]
        assert input_info.name in ["pixel_values", "input", "images"]
        # Shape should be [B, 3, H, W] where B can be 1 or dynamic ('batch')
        batch_dim = input_info.shape[0]
        assert batch_dim == 1 or batch_dim == "batch", f"Unexpected batch dim: {batch_dim}"
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
        """Test basic text encoding."""
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
        """Verify outputs contain finite values."""
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

        os.environ["SAM3_DEVICE"] = "cpu"
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

class TestTextTrackInit:
    """Tests for the text-track-init (Video PCS) functionality."""

    @pytest.fixture
    def handler(self, model_dir):
        """Get handler instance with all required models."""
        import sys

        os.environ["SAM3_DEVICE"] = "cpu"
        os.environ["SAM3_MODEL_DIR"] = str(model_dir)
        os.environ["SAM3_VISION_ENCODER"] = str(model_dir / "vision_encoder.onnx")
        os.environ["SAM3_TEXT_ENCODER"] = str(model_dir / "text_encoder.onnx")
        os.environ["SAM3_PCS_DECODER"] = str(model_dir / "pcs_decoder.onnx")
        os.environ["SAM3_TRACKER_DECODER"] = str(model_dir / "tracker_decoder.onnx")

        handler_dir = Path(__file__).parent
        sys.path.insert(0, str(handler_dir))

        # Reset any cached handler
        from model_handler import get_handler, reset_handler
        reset_handler()
        return get_handler()

    def test_video_pcs_available(self, handler):
        """Check if Video PCS feature is available."""
        info = handler.get_model_info()
        # This test will skip if models are missing
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available (missing models)")

    def test_init_tracking_from_text_basic(self, handler):
        """Test basic init_tracking_from_text functionality."""
        from PIL import Image

        info = handler.get_model_info()
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available")

        # Create test image with colored object
        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        img_arr[80:180, 80:180] = [255, 128, 64]  # Orange square
        img = Image.fromarray(img_arr)

        result = handler.init_tracking_from_text(
            image=img,
            text_prompts=["object"],
            confidence_threshold=0.01,
        )

        # Should not have an error (may have no detections with synthetic images)
        if "error" in result:
            # No detections is acceptable
            assert "No objects detected" in result["error"]
        else:
            assert "session_id" in result
            assert "tracked_objects" in result
            # Clean up
            if result.get("session_id"):
                handler.clear_tracking(result["session_id"])

    def test_init_tracking_from_text_returns_detections(self, handler):
        """Test that init_tracking_from_text returns detection info."""
        from PIL import Image

        info = handler.get_model_info()
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available")

        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        img_arr[50:200, 50:200] = [100, 200, 100]
        img = Image.fromarray(img_arr)

        result = handler.init_tracking_from_text(
            image=img,
            text_prompts=["object"],
            confidence_threshold=0.01,
        )

        if result.get("session_id"):
            # Should have detection info included
            assert "detections" in result or "tracked_objects" in result
            handler.clear_tracking(result["session_id"])

    def test_handle_text_track_init_response_format(self, handler, model_dir):
        """Test that handle_text_track_init returns CVAT-compatible format."""
        import sys
        from PIL import Image

        info = handler.get_model_info()
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available")

        handler_dir = Path(__file__).parent
        sys.path.insert(0, str(handler_dir))

        # Import handler function
        from main import handle_text_track_init

        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        img_arr[60:180, 60:180] = [200, 100, 100]
        img = Image.fromarray(img_arr)

        data = {
            "text_prompts": ["object"],
            "threshold": 0.01,
        }

        result = handle_text_track_init(handler, data, img)

        # Check CVAT-compatible response format
        if "error" not in result:
            assert "shapes" in result, "Missing 'shapes' in response"
            assert "states" in result, "Missing 'states' in response"
            assert "session_id" in result, "Missing 'session_id' in response"

            # Validate shapes structure
            for shape in result.get("shapes", []):
                assert "type" in shape
                assert "points" in shape
                assert "clientID" in shape

            # Validate states structure
            for state in result.get("states", []):
                assert "session_id" in state
                assert "object_id" in state

            # Clean up
            if result.get("session_id"):
                handler.clear_tracking(result["session_id"])

    def test_text_track_init_empty_prompts(self, handler, model_dir):
        """Test handle_text_track_init with empty prompts."""
        import sys
        from PIL import Image

        info = handler.get_model_info()
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available")

        handler_dir = Path(__file__).parent
        sys.path.insert(0, str(handler_dir))

        from main import handle_text_track_init

        img = Image.new("RGB", (256, 256), color=(128, 128, 128))

        data = {
            "text_prompts": [],
            "threshold": 0.5,
        }

        result = handle_text_track_init(handler, data, img)

        # Should return an error for empty prompts
        assert "error" in result, "Empty prompts should return error"

    def test_text_track_then_track_frame(self, handler):
        """Test full video PCS flow: text-track-init → track/frame."""
        from PIL import Image

        info = handler.get_model_info()
        if not info.get("features", {}).get("video_pcs", False):
            pytest.skip("Video PCS not available")

        # Frame 1
        frame1_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        frame1_arr[80:180, 80:180] = [255, 100, 100]
        frame1 = Image.fromarray(frame1_arr)

        # Frame 2 (object moved)
        frame2_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        frame2_arr[90:190, 90:190] = [255, 100, 100]
        frame2 = Image.fromarray(frame2_arr)

        # Initialize tracking from text
        init_result = handler.init_tracking_from_text(
            image=frame1,
            text_prompts=["object"],
            confidence_threshold=0.01,
        )

        if not init_result.get("session_id"):
            pytest.skip("No objects detected (synthetic image limitation)")

        session_id = init_result["session_id"]

        try:
            # Track to frame 2
            track_result = handler.track_frame(
                session_id=session_id,
                image=frame2,
                frame_idx=1,
            )

            # Should succeed without error
            assert "error" not in track_result, f"Track failed: {track_result.get('error')}"
            assert "tracked_objects" in track_result

        finally:
            # Clean up
            handler.clear_tracking(session_id)
