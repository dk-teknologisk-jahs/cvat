#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Unified SAM3 ONNX Model Handler

Uses ONNX Runtime for inference - NO HuggingFace auth needed at runtime!
The ONNX models are exported once (requires HF auth) then baked into Docker images.

Supports all three CVAT AI tool modes:
1. encode - Interactor mode (returns embeddings for browser-side decoding)
2. text-to-segment - Detector/PCS mode (returns masks + boxes)
3. track/* - Video tracking with Redis state management

ONNX Models required (exported via export_hf_onnx.py):
- vision_encoder.onnx (1.8 GB) - outputs 256ch at all FPN levels
- text_encoder.onnx (1.3 GB) - CLIP text encoding for PCS
- pcs_decoder.onnx (123 MB) - DETR decoder for text-to-segment
- tracker_decoder.onnx (16 MB) - mask decoder for interactor/tracking

Key design:
- Vision encoder outputs 256/256/256 channels (no projections baked in)
- Tracker decoder includes conv_s0/conv_s1 projections internally
- This allows one vision encoder to serve both tracker and PCS modes
"""

import logging
import math
import os
import pickle
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "cvat_redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_TTL = int(os.environ.get("REDIS_TTL", "3600"))  # 1 hour

# Default model directory (can be overridden by environment or constructor)
DEFAULT_MODEL_DIR = "/opt/nuclio/sam3/models"

# GitHub release URL for pre-exported ONNX models
DEFAULT_GITHUB_RELEASE_URL = "https://github.com/dk-teknologisk-jahs/cvat/releases/download/sam3"

# Constants
SAM3_IMAGE_SIZE = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


def download_model_if_missing(model_path: str, model_name: str) -> bool:
    """
    Download a model from GitHub release if it doesn't exist.

    Args:
        model_path: Full path where model should be saved
        model_name: Filename of the model (e.g., 'vision_encoder.onnx')

    Returns:
        True if model exists or was downloaded successfully
    """
    import urllib.request
    from pathlib import Path

    path = Path(model_path)
    if path.exists() and path.stat().st_size > 1000:  # Basic sanity check
        return True

    github_url = os.environ.get("SAM3_GITHUB_RELEASE_URL", DEFAULT_GITHUB_RELEASE_URL)
    url = f"{github_url}/{model_name}"

    logger.info(f"Model not found: {model_path}")
    logger.info(f"Downloading from: {url}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 / total_size)
                mb = block_num * block_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
        print()  # Newline after progress

        if path.exists() and path.stat().st_size > 1000:
            logger.info(f"Downloaded successfully: {model_path}")
            return True
        else:
            logger.error(f"Download failed or file too small: {model_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def get_model_paths(model_dir: Optional[str] = None) -> Dict[str, str]:
    """Get model paths from environment or provided directory."""
    base_dir = model_dir or os.environ.get("SAM3_MODEL_DIR", DEFAULT_MODEL_DIR)
    return {
        "model_dir": base_dir,
        "vision_encoder": os.environ.get("SAM3_VISION_ENCODER", f"{base_dir}/vision_encoder.onnx"),
        "text_encoder": os.environ.get("SAM3_TEXT_ENCODER", f"{base_dir}/text_encoder.onnx"),
        "pcs_decoder": os.environ.get("SAM3_PCS_DECODER", f"{base_dir}/pcs_decoder.onnx"),
        "tracker_decoder": os.environ.get("SAM3_TRACKER_DECODER", f"{base_dir}/tracker_decoder.onnx"),
        # Memory components for video propagation (server-side only)
        "memory_attention": os.environ.get("SAM3_MEMORY_ATTENTION", f"{base_dir}/memory_attention.onnx"),
        "memory_encoder": os.environ.get("SAM3_MEMORY_ENCODER", f"{base_dir}/memory_encoder.onnx"),
        "object_pointer": os.environ.get("SAM3_OBJECT_POINTER", f"{base_dir}/object_pointer.onnx"),
        "temporal_pos_enc": os.environ.get("SAM3_TEMPORAL_POS_ENC", f"{base_dir}/temporal_pos_enc.npy"),
    }


class RedisCache:
    """Redis cache manager for video tracking state."""

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        password: str = REDIS_PASSWORD,
        ttl: int = REDIS_TTL,
    ):
        self.ttl = ttl
        self.client = None
        self._memory_cache: Dict[str, Any] = {}

        try:
            import redis
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password if password else None,
                decode_responses=False,
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.client = None

    def _make_key(self, prefix: str, identifier: str) -> str:
        return f"sam3:{prefix}:{identifier}"

    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        key = self._make_key(prefix, identifier)
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        return self._memory_cache.get(key)

    def set(self, prefix: str, identifier: str, value: Any) -> bool:
        key = self._make_key(prefix, identifier)
        try:
            data = pickle.dumps(value)
            if self.client:
                self.client.setex(key, self.ttl, data)
            else:
                self._memory_cache[key] = value
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    def delete(self, prefix: str, identifier: str) -> bool:
        key = self._make_key(prefix, identifier)
        try:
            if self.client:
                self.client.delete(key)
            elif key in self._memory_cache:
                del self._memory_cache[key]
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False


class UnifiedModelHandler:
    """
    Unified SAM3 handler using ONNX Runtime.

    Supports:
    - encode(): Returns embeddings for browser-side mask decoding (interactor)
    - text_to_segment(): Text-to-segment detection (PCS/detector)
    - init_tracking() / track_frame(): Video object tracking

    All inference uses ONNX Runtime - no HuggingFace auth needed!
    """

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None):
        import onnxruntime as ort

        self.device = device
        self.cache = RedisCache()

        # Get model paths dynamically (supports environment override or explicit path)
        self.paths = get_model_paths(model_dir)

        # Configure ONNX Runtime providers
        if device == "cuda":
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Lazy-loaded ONNX sessions
        self._vision_encoder: Optional[ort.InferenceSession] = None
        self._text_encoder: Optional[ort.InferenceSession] = None
        self._pcs_decoder: Optional[ort.InferenceSession] = None
        self._tracker_decoder: Optional[ort.InferenceSession] = None
        # Memory components for video propagation
        self._memory_attention: Optional[ort.InferenceSession] = None
        self._memory_encoder: Optional[ort.InferenceSession] = None
        self._object_pointer: Optional[ort.InferenceSession] = None
        self._temporal_pos_enc: Optional[np.ndarray] = None  # Pre-computed numpy array

        # Image preprocessing params
        self.image_size = SAM3_IMAGE_SIZE
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        logger.info(f"UnifiedModelHandler initialized (device={device}, model_dir={self.paths['model_dir']})")

    # =========================================================================
    # Lazy Model Loading
    # =========================================================================

    def _get_vision_encoder(self):
        """Lazy load vision encoder ONNX model (auto-downloads if missing)."""
        if self._vision_encoder is None:
            import onnxruntime as ort
            path = self.paths["vision_encoder"]
            if not os.path.exists(path):
                if not download_model_if_missing(path, "vision_encoder.onnx"):
                    raise FileNotFoundError(
                        f"Vision encoder not found: {path}\n"
                        "Download failed. Check network or set SAM3_GITHUB_RELEASE_URL."
                    )
            logger.info(f"Loading vision encoder: {path}")
            self._vision_encoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._vision_encoder

    def _get_text_encoder(self):
        """Lazy load text encoder ONNX model (auto-downloads if missing)."""
        if self._text_encoder is None:
            import onnxruntime as ort
            path = self.paths["text_encoder"]
            if not os.path.exists(path):
                if not download_model_if_missing(path, "text_encoder.onnx"):
                    raise FileNotFoundError(
                        f"Text encoder not found: {path}\n"
                        "Download failed. Check network or set SAM3_GITHUB_RELEASE_URL."
                    )
            logger.info(f"Loading text encoder: {path}")
            self._text_encoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._text_encoder

    def _get_pcs_decoder(self):
        """Lazy load PCS decoder ONNX model (auto-downloads if missing)."""
        if self._pcs_decoder is None:
            import onnxruntime as ort
            path = self.paths["pcs_decoder"]
            if not os.path.exists(path):
                if not download_model_if_missing(path, "pcs_decoder.onnx"):
                    raise FileNotFoundError(
                        f"PCS decoder not found: {path}\n"
                        "Download failed. Check network or set SAM3_GITHUB_RELEASE_URL."
                    )
            logger.info(f"Loading PCS decoder: {path}")
            self._pcs_decoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._pcs_decoder

    def _get_tracker_decoder(self):
        """Lazy load tracker decoder ONNX model (auto-downloads if missing)."""
        if self._tracker_decoder is None:
            import onnxruntime as ort
            path = self.paths["tracker_decoder"]
            if not os.path.exists(path):
                if not download_model_if_missing(path, "tracker_decoder.onnx"):
                    raise FileNotFoundError(
                        f"Tracker decoder not found: {path}\n"
                        "Download failed. Check network or set SAM3_GITHUB_RELEASE_URL."
                    )
            logger.info(f"Loading tracker decoder: {path}")
            self._tracker_decoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._tracker_decoder

    def _get_memory_attention(self):
        """Lazy load memory attention ONNX model for video propagation."""
        if self._memory_attention is None:
            import onnxruntime as ort
            path = self.paths["memory_attention"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Memory attention model not found: {path}\n"
                    "Run export_sam3_memory_components.py to generate it."
                )
            logger.info(f"Loading memory attention: {path}")
            self._memory_attention = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._memory_attention

    def _get_memory_encoder(self):
        """Lazy load memory encoder ONNX model for video propagation."""
        if self._memory_encoder is None:
            import onnxruntime as ort
            path = self.paths["memory_encoder"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Memory encoder model not found: {path}\n"
                    "Run export_sam3_memory_components.py to generate it."
                )
            logger.info(f"Loading memory encoder: {path}")
            self._memory_encoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._memory_encoder

    def _get_object_pointer(self):
        """Lazy load object pointer ONNX model for video propagation."""
        if self._object_pointer is None:
            import onnxruntime as ort
            path = self.paths["object_pointer"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Object pointer model not found: {path}\n"
                    "Run export_sam3_memory_components.py to generate it."
                )
            logger.info(f"Loading object pointer: {path}")
            self._object_pointer = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._object_pointer

    def _get_temporal_pos_enc(self) -> np.ndarray:
        """Load pre-computed temporal position encoding table."""
        if self._temporal_pos_enc is None:
            path = self.paths["temporal_pos_enc"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Temporal position encoding not found: {path}\n"
                    "Run export_sam3_memory_components.py to generate it."
                )
            logger.info(f"Loading temporal position encoding: {path}")
            self._temporal_pos_enc = np.load(path)
        return self._temporal_pos_enc

    # =========================================================================
    # Image Preprocessing
    # =========================================================================

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for SAM3 vision encoder.

        Args:
            image: PIL Image (any size)

        Returns:
            Preprocessed tensor [1, 3, 1008, 1008]
        """
        # Resize to 1008x1008
        image_resized = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR,
        )

        # Convert to numpy and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std

        # Transpose to CHW and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    # =========================================================================
    # Encode (Interactor Mode)
    # =========================================================================

    def encode(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Encode image to SAM3 embeddings for browser-side decoding.

        Uses vision encoder ONNX model. Outputs 256ch at all FPN levels.

        Args:
            image: PIL Image

        Returns:
            Dictionary with embeddings:
            - fpn_feat_0: [1, 256, 288, 288]
            - fpn_feat_1: [1, 256, 144, 144]
            - fpn_feat_2: [1, 256, 72, 72]
            - fpn_pos_2: [1, 256, 72, 72]
        """
        encoder = self._get_vision_encoder()

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Get input/output names
        input_name = encoder.get_inputs()[0].name
        output_names = [o.name for o in encoder.get_outputs()]

        # Run encoder
        outputs = encoder.run(output_names, {input_name: input_tensor})

        # Map outputs to standard names
        # Our exported encoder outputs: fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2
        result = {}
        for name, arr in zip(output_names, outputs):
            result[name] = arr

        logger.info(f"Encoded image: {image.size} -> {list(result.keys())}")
        return result

    def encode_batch(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """
        Encode multiple images in a single forward pass.

        More efficient than calling encode() multiple times for video or multi-image scenarios.

        Args:
            images: List of PIL Images (all will be resized to 1008x1008)

        Returns:
            Dictionary with batched embeddings:
            - fpn_feat_0: [B, 256, 288, 288]
            - fpn_feat_1: [B, 256, 144, 144]
            - fpn_feat_2: [B, 256, 72, 72]
            - fpn_pos_2: [B, 256, 72, 72]
        """
        if not images:
            return {}

        encoder = self._get_vision_encoder()

        # Preprocess all images
        input_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = np.concatenate(input_tensors, axis=0)  # [B, 3, 1008, 1008]

        # Get input/output names
        input_name = encoder.get_inputs()[0].name
        output_names = [o.name for o in encoder.get_outputs()]

        # Run encoder on batch
        outputs = encoder.run(output_names, {input_name: batch_tensor})

        # Map outputs to standard names
        result = {}
        for name, arr in zip(output_names, outputs):
            result[name] = arr

        logger.info(f"Encoded batch of {len(images)} images -> {list(result.keys())}")
        return result

    # =========================================================================
    # Text-to-Segment (Detector/PCS Mode)
    # =========================================================================

    def text_to_segment(
        self,
        text_prompts: List[str],
        image: Image.Image,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        box_prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run text-to-segment (PCS mode) using ONNX models.

        Args:
            text_prompts: List of text prompts (e.g., ["a person", "a car"])
            image: PIL Image
            confidence_threshold: Minimum confidence for detections
            box_prompts: Optional list of box prompts for geometric guidance.
                         Each box is a dict with:
                         - "box": [x1, y1, x2, y2] in image coordinates
                         - "label": 1 for positive (include), 0 for negative (exclude)
                         Example: [{"box": [100, 100, 200, 200], "label": 1}]

        Returns:
            List of detections: [{"mask": np.ndarray, "box": [x1,y1,x2,y2], "score": float, "label": str}, ...]
        """
        # Get encoders and decoder
        vision_encoder = self._get_vision_encoder()
        text_encoder = self._get_text_encoder()
        pcs_decoder = self._get_pcs_decoder()

        original_size = image.size  # (W, H)
        orig_w, orig_h = original_size

        # 1. Encode image (once for all prompts)
        input_tensor = self.preprocess_image(image)
        vision_input = vision_encoder.get_inputs()[0].name
        vision_outputs = [o.name for o in vision_encoder.get_outputs()]
        vision_features = vision_encoder.run(vision_outputs, {vision_input: input_tensor})

        # Map vision outputs by name
        vision_dict = dict(zip(vision_outputs, vision_features))
        fpn_feat_0 = vision_dict.get("fpn_feat_0", vision_features[0])
        fpn_feat_1 = vision_dict.get("fpn_feat_1", vision_features[1])
        fpn_feat_2 = vision_dict.get("fpn_feat_2", vision_features[2])
        fpn_pos_2 = vision_dict.get("fpn_pos_2", vision_features[3])

        # Prepare box prompts (convert from xyxy image coords to cxcywh normalized)
        if box_prompts and len(box_prompts) > 0:
            num_boxes = len(box_prompts)
            input_boxes = np.zeros((1, num_boxes, 4), dtype=np.float32)
            input_boxes_labels = np.zeros((1, num_boxes), dtype=np.int64)

            for i, bp in enumerate(box_prompts):
                box = bp.get("box", [0, 0, 1, 1])
                label = bp.get("label", 1)  # Default to positive

                # Convert xyxy to cxcywh normalized
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0 / orig_w
                cy = (y1 + y2) / 2.0 / orig_h
                w = (x2 - x1) / orig_w
                h = (y2 - y1) / orig_h

                input_boxes[0, i] = [cx, cy, w, h]
                input_boxes_labels[0, i] = int(label)

            logger.info(f"Using {num_boxes} box prompts for PCS")
        else:
            # No box prompts - use padding (label=-10)
            input_boxes = np.zeros((1, 1, 4), dtype=np.float32)
            input_boxes_labels = np.full((1, 1), -10, dtype=np.int64)

        # Process prompts one at a time since PCS decoder is exported with batch_size=1
        all_detections = []
        for prompt_idx, prompt in enumerate(text_prompts):
            # 2. Encode single text prompt
            text_features, text_mask = self._encode_text(text_encoder, [prompt])

            # Ensure text_mask is bool type for ONNX
            text_mask = text_mask.astype(bool)

            # 3. Run PCS decoder with all required inputs
            pcs_inputs = {
                "fpn_feat_0": fpn_feat_0,
                "fpn_feat_1": fpn_feat_1,
                "fpn_feat_2": fpn_feat_2,
                "fpn_pos_2": fpn_pos_2,
                "text_features": text_features,
                "text_mask": text_mask,
                "input_boxes": input_boxes,
                "input_boxes_labels": input_boxes_labels,
            }
            pcs_output_names = [o.name for o in pcs_decoder.get_outputs()]
            pcs_outputs = pcs_decoder.run(pcs_output_names, pcs_inputs)

            # Parse outputs (boxes, scores, masks)
            detections = self._parse_pcs_outputs(
                pcs_outputs,
                pcs_output_names,
                original_size,
                [prompt],
                confidence_threshold,
            )

            # Tag detections with prompt label
            for det in detections:
                det["label"] = prompt

            all_detections.extend(detections)

        logger.info(f"Text-to-segment found {len(all_detections)} objects for '{text_prompts}'")
        return all_detections

    def _get_tokenizer(self):
        """Get or create a CLIP tokenizer for text encoding."""
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = None
            try:
                # Try to use HuggingFace tokenizer (best option)
                from transformers import CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                self._tokenizer.model_max_length = 32  # SAM3 uses 32 context length
                logger.info("Using HuggingFace CLIP tokenizer")
            except ImportError:
                logger.warning("transformers not available - using fallback tokenization")
            except Exception as e:
                logger.warning(f"Failed to load CLIP tokenizer: {e}")
        return self._tokenizer

    def _encode_text(self, text_encoder, text_prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode text prompts using text encoder ONNX model.

        Returns:
            Tuple of (text_features, text_mask)
        """
        text_input_name = text_encoder.get_inputs()[0].name
        text_input = text_encoder.get_inputs()[0]

        # SAM3 uses 32 token context length
        max_len = 32

        if text_input.type == "tensor(int64)":
            # Try to use proper CLIP tokenizer
            tokenizer = self._get_tokenizer()
            if tokenizer is not None:
                # Use HuggingFace tokenizer
                encoding = tokenizer(
                    text_prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="np",
                )
                tokens = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]

                # Check if attention_mask is needed
                input_names = [inp.name for inp in text_encoder.get_inputs()]
                if len(input_names) > 1 and "attention_mask" in input_names[1]:
                    inputs = {
                        text_input_name: tokens,
                        input_names[1]: attention_mask,
                    }
                else:
                    inputs = {text_input_name: tokens}
            else:
                # Fallback: simple character-level encoding
                logger.warning("Using fallback character-level tokenization")
                tokens = np.zeros((len(text_prompts), max_len), dtype=np.int64)
                attention_mask = np.zeros((len(text_prompts), max_len), dtype=np.int64)
                # SOT token at start
                tokens[:, 0] = 49406  # <start_of_text>
                attention_mask[:, 0] = 1
                for i, prompt in enumerate(text_prompts):
                    for j, char in enumerate(prompt[:max_len-2]):
                        tokens[i, j+1] = ord(char) % 49407
                        attention_mask[i, j+1] = 1
                    # EOT token after text
                    eot_pos = min(len(prompt)+1, max_len-1)
                    tokens[i, eot_pos] = 49407  # <end_of_text>
                    attention_mask[i, eot_pos] = 1
                inputs = {text_input_name: tokens}
        else:
            # Assumes string input (unlikely for ONNX)
            inputs = {text_input_name: np.array(text_prompts)}

        output_names = [o.name for o in text_encoder.get_outputs()]
        outputs = text_encoder.run(output_names, inputs)

        # Text encoder returns (text_features, text_mask)
        text_features = outputs[0]
        text_mask = outputs[1] if len(outputs) > 1 else attention_mask

        return text_features, text_mask

    def _parse_pcs_outputs(
        self,
        outputs: List[np.ndarray],
        output_names: List[str],
        original_size: Tuple[int, int],
        text_prompts: List[str],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Parse PCS decoder outputs into detections."""
        detections = []

        # Map outputs by name
        output_dict = dict(zip(output_names, outputs))

        # PCS decoder outputs: pred_masks, pred_boxes, pred_logits, presence_logits
        boxes = output_dict.get("pred_boxes", outputs[1])
        logits = output_dict.get("pred_logits", outputs[2])  # These are the class logits
        masks = output_dict.get("pred_masks", outputs[0])
        presence_logits = output_dict.get("presence_logits", outputs[3] if len(outputs) > 3 else None)

        # Convert logits to scores via sigmoid
        # Official SAM3: out_probs = (out_logits.sigmoid() * presence_score.sigmoid())
        # The presence_logits is a global "is anything matching present" score
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        if presence_logits is not None:
            presence_score = 1.0 / (1.0 + np.exp(-presence_logits))  # sigmoid
            scores = scores * presence_score  # broadcasts [1,1] to [1,200]

        if scores is None:
            logger.warning("No scores in PCS output")
            return detections

        # Flatten batch dimension
        if boxes.ndim > 2:
            boxes = boxes.squeeze(0)
        if scores.ndim > 1:
            scores = scores.squeeze(0)
        if masks is not None and masks.ndim > 3:
            masks = masks.squeeze(0)

        orig_w, orig_h = original_size

        for i in range(len(scores)):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            # Get box (normalized or absolute)
            box = boxes[i]
            if box.max() <= 1.0:
                # Normalized coordinates
                box = [
                    float(box[0] * orig_w),
                    float(box[1] * orig_h),
                    float(box[2] * orig_w),
                    float(box[3] * orig_h),
                ]
            else:
                box = [float(x) for x in box]

            # Get mask - use official SAM3 post-processing pipeline
            mask = None
            if masks is not None:
                mask_raw = masks[i]
                # Official SAM3 image mode: max_hole_area=0, max_sprinkle_area=0
                # Post-process includes: resize + threshold at 0
                mask = self.postprocess_mask_logits(
                    mask_raw,
                    original_size=(orig_w, orig_h),
                    max_hole_area=0,  # Disabled for image mode (matching SAM3)
                    max_sprinkle_area=0,
                )

            detections.append({
                "mask": mask,
                "box": box,
                "score": float(score),
                "label": text_prompts[0] if text_prompts else "object",
            })

        return detections

    def get_semantic_mask(
        self,
        detections: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Convert instance detections to a semantic segmentation mask.

        Takes the union of all instance masks to produce a single binary mask
        showing all detected objects.

        Args:
            detections: List of detection dicts from text_to_segment() or
                       automatic_mask_generation(), each containing a "mask" key
            image_size: (width, height) tuple. If not provided, inferred from masks.

        Returns:
            Binary semantic mask [H, W] as uint8 (0 or 1)
        """
        if not detections:
            if image_size:
                return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)

        # Get image size from first valid mask if not provided
        if image_size is None:
            for det in detections:
                mask = det.get("mask")
                if mask is not None:
                    h, w = mask.shape[:2]
                    image_size = (w, h)
                    break

        if image_size is None:
            return np.zeros((1, 1), dtype=np.uint8)

        # Union all masks
        semantic_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        for det in detections:
            mask = det.get("mask")
            if mask is not None:
                # Ensure mask matches expected size
                if mask.shape[:2] != (image_size[1], image_size[0]):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                semantic_mask = np.logical_or(semantic_mask, mask > 0).astype(np.uint8)

        return semantic_mask

    def get_labeled_semantic_mask(
        self,
        detections: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Convert instance detections to a labeled semantic segmentation mask.

        Each instance gets a unique label ID. Overlapping regions get the
        label of the detection with highest score.

        Args:
            detections: List of detection dicts from text_to_segment() or
                       automatic_mask_generation()
            image_size: (width, height) tuple. If not provided, inferred from masks.

        Returns:
            Labeled mask [H, W] as int32 (0=background, 1,2,3...=instances)
        """
        if not detections:
            if image_size:
                return np.zeros((image_size[1], image_size[0]), dtype=np.int32)
            return np.zeros((1, 1), dtype=np.int32)

        # Get image size from first valid mask if not provided
        if image_size is None:
            for det in detections:
                mask = det.get("mask")
                if mask is not None:
                    h, w = mask.shape[:2]
                    image_size = (w, h)
                    break

        if image_size is None:
            return np.zeros((1, 1), dtype=np.int32)

        # Sort by score (lowest first so higher scores overwrite)
        sorted_dets = sorted(detections, key=lambda x: x.get("score", 0))

        # Paint masks with instance IDs
        labeled_mask = np.zeros((image_size[1], image_size[0]), dtype=np.int32)
        for i, det in enumerate(sorted_dets, start=1):
            mask = det.get("mask")
            if mask is not None:
                # Ensure mask matches expected size
                if mask.shape[:2] != (image_size[1], image_size[0]):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                labeled_mask[mask > 0] = i

        return labeled_mask

    # =========================================================================
    # Automatic Mask Generation (AMG)
    # =========================================================================

    def automatic_mask_generation(
        self,
        image: Image.Image,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_delta: float = 0.05,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for all objects in an image using a grid of point prompts.

        This implements SAM's Automatic Mask Generation (AMG) mode, which samples
        single-point prompts in a grid over the image and filters the resulting masks.

        Args:
            image: PIL Image
            points_per_side: Number of points to sample per side of the grid (32 = 1024 points)
            pred_iou_thresh: Filter masks with predicted IoU below this threshold
            stability_score_thresh: Filter masks with stability score below this threshold
            stability_delta: Delta around logit 0 for stability calculation (default 0.05,
                           matching official SAM3 dynamic_multimask_stability_delta)
            box_nms_thresh: IoU threshold for NMS to remove duplicate masks
            min_mask_region_area: Remove masks with area smaller than this (requires cv2)

        Returns:
            List of mask records: [{"mask": np.ndarray, "box": [x,y,w,h], "area": int,
                                    "predicted_iou": float, "stability_score": float}, ...]
        """
        decoder = self._get_tracker_decoder()
        orig_w, orig_h = image.size

        # Encode image once
        embeddings = self.encode(image)

        # Generate point grid normalized to [0, 1]
        points_grid = self._build_point_grid(points_per_side)

        # Process points in batches
        all_masks = []
        all_iou_preds = []
        all_points = []

        for point_xy in points_grid:
            # Convert from [0,1] to image coordinates
            point_x = point_xy[0] * SAM3_IMAGE_SIZE
            point_y = point_xy[1] * SAM3_IMAGE_SIZE

            # Run decoder with single positive point prompt
            point_coords = np.array([[[[point_x, point_y]]]], dtype=np.float32)
            point_labels = np.array([[[1.0]]], dtype=np.float32)

            try:
                result = self._run_tracker_decoder(
                    embeddings, point_coords, point_labels
                )

                if result is not None:
                    masks = result["masks"]  # [1, 3, 1008, 1008]
                    iou_preds = result["iou_predictions"]  # [1, 3]

                    # Add all 3 mask candidates
                    for mask_idx in range(3):
                        all_masks.append(masks[0, mask_idx])  # [1008, 1008]
                        all_iou_preds.append(iou_preds[0, mask_idx])
                        all_points.append([point_x, point_y])
            except Exception as e:
                logger.debug(f"Point prompt at ({point_x:.1f}, {point_y:.1f}) failed: {e}")
                continue

        if not all_masks:
            logger.warning("No masks generated from point grid")
            return []

        # Convert to numpy arrays
        all_masks = np.stack(all_masks, axis=0)  # [N, 1008, 1008]
        all_iou_preds = np.array(all_iou_preds)
        all_points = np.array(all_points)

        # Filter by predicted IoU
        keep = all_iou_preds > pred_iou_thresh
        all_masks = all_masks[keep]
        all_iou_preds = all_iou_preds[keep]
        all_points = all_points[keep]

        if len(all_masks) == 0:
            return []

        # Calculate stability scores (using logit thresholds, matching official SAM3)
        stability_scores = self._calculate_stability_scores(
            all_masks, stability_delta
        )

        # Filter by stability score
        keep = stability_scores >= stability_score_thresh
        all_masks = all_masks[keep]
        all_iou_preds = all_iou_preds[keep]
        all_points = all_points[keep]
        stability_scores = stability_scores[keep]

        if len(all_masks) == 0:
            return []

        # Keep original logits for post-processing, use binarized for NMS
        all_masks_logits = all_masks  # Original logits
        all_masks_binary = (all_masks > 0).astype(np.float32)  # For NMS

        # Calculate boxes from binarized masks
        boxes = self._batched_mask_to_box(all_masks_binary)

        # Apply NMS
        keep_idxs = self._nms_boxes(boxes, all_iou_preds, box_nms_thresh)
        all_masks_logits = all_masks_logits[keep_idxs]
        all_iou_preds = all_iou_preds[keep_idxs]
        all_points = all_points[keep_idxs]
        stability_scores = stability_scores[keep_idxs]
        boxes = boxes[keep_idxs]

        # Post-process and resize masks to original size
        # Note: Official SAM3 uses max_hole_area=0 for image mode (no post-processing)
        results = []
        for i in range(len(all_masks_logits)):
            # Use official SAM3 post-processing pipeline on logits
            mask_binary = self.postprocess_mask_logits(
                all_masks_logits[i],
                original_size=(orig_w, orig_h),
                max_hole_area=0,  # Disabled for AMG (matching SAM3 image mode)
                max_sprinkle_area=0,
            )

            # Calculate area
            area = int(mask_binary.sum())

            # Filter by min area
            if min_mask_region_area > 0 and area < min_mask_region_area:
                continue

            # Scale box to original size
            box = boxes[i]
            scale_x = orig_w / SAM3_IMAGE_SIZE
            scale_y = orig_h / SAM3_IMAGE_SIZE
            box_scaled = [
                float(box[0] * scale_x),
                float(box[1] * scale_y),
                float((box[2] - box[0]) * scale_x),  # width
                float((box[3] - box[1]) * scale_y),  # height
            ]

            results.append({
                "mask": mask_binary,
                "box": box_scaled,  # XYWH format
                "area": area,
                "predicted_iou": float(all_iou_preds[i]),
                "stability_score": float(stability_scores[i]),
                "point_coords": [[float(all_points[i][0]), float(all_points[i][1])]],
            })

        logger.info(f"Automatic mask generation: {len(results)} masks from {len(points_grid)} points")
        return results

    def _build_point_grid(self, n_per_side: int) -> np.ndarray:
        """Build a grid of points normalized to [0, 1]."""
        offset = 1 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1 - offset, n_per_side)
        points_x = np.tile(points_one_side, n_per_side)
        points_y = np.repeat(points_one_side, n_per_side)
        return np.stack([points_x, points_y], axis=-1)

    def _run_tracker_decoder(
        self,
        embeddings: Dict[str, np.ndarray],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Run tracker decoder with point prompts."""
        decoder = self._get_tracker_decoder()

        fpn_feat_0 = embeddings.get("fpn_feat_0")
        fpn_feat_1 = embeddings.get("fpn_feat_1")
        fpn_feat_2 = embeddings.get("fpn_feat_2")

        if fpn_feat_0 is None or fpn_feat_1 is None or fpn_feat_2 is None:
            return None

        inputs = {
            "fpn_feat_0": fpn_feat_0,
            "fpn_feat_1": fpn_feat_1,
            "fpn_feat_2": fpn_feat_2,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0.0], dtype=np.float32),
        }

        outputs = decoder.run(None, inputs)

        return {
            "masks": outputs[0],  # [B, 3, 1008, 1008]
            "iou_predictions": outputs[1],  # [B, 3]
            "low_res_masks": outputs[2],  # [B, 3, 288, 288]
            "object_score_logits": outputs[3],  # [B, 1]
        }

    # =========================================================================
    # Mask Post-Processing (matching official SAM3)
    # =========================================================================

    def postprocess_mask_logits(
        self,
        mask_logits: np.ndarray,
        original_size: Tuple[int, int],
        max_hole_area: int = 0,
        max_sprinkle_area: int = 0,
    ) -> np.ndarray:
        """
        Post-process mask logits following official SAM3 implementation.

        This EXACTLY matches sam3/model/utils/sam1_utils.py SAM2Transforms.postprocess_masks():
        1. Fill small holes in background (if max_hole_area > 0)
        2. Remove small sprinkles in foreground (if max_sprinkle_area > 0)
        3. Upsample to original size using bilinear interpolation
        4. Return binary mask (logits > 0)

        Args:
            mask_logits: Mask logits [H, W] at decoder resolution (e.g., 288x288 or 1008x1008)
            original_size: (width, height) of original image
            max_hole_area: Maximum area of holes to fill (0 = disabled, matching SAM3 image mode)
            max_sprinkle_area: Maximum area of sprinkles to remove (0 = disabled)

        Returns:
            Binary mask [orig_H, orig_W] as uint8
        """
        orig_w, orig_h = original_size

        # Apply hole filling and sprinkle removal on logits (before upsampling)
        if max_hole_area > 0 or max_sprinkle_area > 0:
            mask_logits = self._fill_holes_and_remove_sprinkles(
                mask_logits, max_hole_area, max_sprinkle_area
            )

        # Upsample to original size (bilinear, matching F.interpolate align_corners=False)
        if mask_logits.shape != (orig_h, orig_w):
            mask_logits = cv2.resize(
                mask_logits.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Threshold at 0 (logit > 0 = foreground)
        return (mask_logits > 0).astype(np.uint8)

    def _fill_holes_and_remove_sprinkles(
        self,
        mask_logits: np.ndarray,
        max_hole_area: int = 0,
        max_sprinkle_area: int = 0,
    ) -> np.ndarray:
        """
        Fill small holes and remove small sprinkles in mask logits.

        EXACTLY matches official SAM3 sam1_utils.py:

        For holes (max_hole_area > 0):
            labels, areas = connected_components((mask <= 0).astype(uint8))
            is_hole = (labels > 0) & (areas <= max_hole_area)
            mask = where(is_hole, 10.0, mask)

        For sprinkles (max_sprinkle_area > 0):
            labels, areas = connected_components((mask > 0).astype(uint8))
            is_sprinkle = (labels > 0) & (areas <= max_sprinkle_area)
            mask = where(is_sprinkle, -10.0, mask)

        Note: labels > 0 means "any component except the largest one"
        """
        if max_hole_area <= 0 and max_sprinkle_area <= 0:
            return mask_logits

        mask_logits = mask_logits.copy()
        h, w = mask_logits.shape

        # Fill holes: small background components
        if max_hole_area > 0:
            bg_mask = (mask_logits <= 0).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)

            if num_labels > 1:
                # Find largest background component (label 0 is always background in OpenCV)
                # Stats: [left, top, width, height, area]
                areas = stats[:, cv2.CC_STAT_AREA]
                largest_label = np.argmax(areas)

                for label_id in range(num_labels):
                    if label_id == largest_label:
                        continue  # Skip largest component
                    if areas[label_id] <= max_hole_area:
                        # Fill this hole with positive logit
                        mask_logits[labels == label_id] = 10.0

        # Remove sprinkles: small foreground components
        if max_sprinkle_area > 0:
            fg_mask = (mask_logits > 0).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

            if num_labels > 1:
                areas = stats[:, cv2.CC_STAT_AREA]
                largest_label = np.argmax(areas)

                for label_id in range(num_labels):
                    if label_id == largest_label:
                        continue  # Skip largest component
                    if areas[label_id] <= max_sprinkle_area:
                        # Remove this sprinkle with negative logit
                        mask_logits[labels == label_id] = -10.0

        return mask_logits

    def _calculate_stability_scores(
        self,
        mask_logits: np.ndarray,
        stability_delta: float = 0.05,
    ) -> np.ndarray:
        """
        Calculate mask stability scores by comparing different thresholds.

        Matches the official SAM3 implementation from sam3/sam/mask_decoder.py:
            area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
            area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
            stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)

        Args:
            mask_logits: Mask logits (NOT probabilities) - threshold at 0 = prob 0.5
            stability_delta: Delta around logit 0 (default 0.05, matching official SAM3)

        Returns:
            Stability scores for each mask
        """
        # Higher threshold: more confident foreground
        area_i = (mask_logits > stability_delta).sum(axis=(1, 2))
        # Lower threshold: all potential foreground (including uncertain)
        area_u = (mask_logits > -stability_delta).sum(axis=(1, 2))

        # Stability = ratio of confident to total foreground (like IoU)
        stability = np.where(area_u > 0, area_i / area_u, 1.0)
        return stability

    def _batched_mask_to_box(self, masks: np.ndarray) -> np.ndarray:
        """Convert masks to bounding boxes [x1, y1, x2, y2]."""
        n_masks = masks.shape[0]
        boxes = np.zeros((n_masks, 4), dtype=np.float32)

        for i in range(n_masks):
            mask = masks[i] > 0
            ys, xs = np.where(mask)
            if len(xs) > 0:
                boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

        return boxes

    def _nms_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float,
    ) -> np.ndarray:
        """Apply non-maximum suppression to boxes."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)

    # =========================================================================
    # Video Tracking
    # =========================================================================

    def init_tracking_from_text(
        self,
        image: Image.Image,
        text_prompts: List[str],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Initialize video tracking from text prompts (Video PCS mode).

        Detects objects matching text prompts in the first frame, then sets up
        tracking for subsequent frames.

        Args:
            image: First frame as PIL Image
            text_prompts: List of text prompts to detect (e.g., ["person", "car"])
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dict with session_id, initial detections, and tracked objects
        """
        # First, detect objects using text-to-segment
        detections = self.text_to_segment(
            text_prompts=text_prompts,
            image=image,
            confidence_threshold=confidence_threshold,
        )

        if not detections:
            return {
                "session_id": None,
                "error": "No objects detected matching text prompts",
                "detections": [],
            }

        # Convert detections to tracking objects
        objects = []
        for i, det in enumerate(detections):
            box = det.get("box", [0, 0, 1, 1])
            # Convert from xyxy to list if needed
            if isinstance(box, np.ndarray):
                box = box.tolist()
            objects.append({
                "object_id": i,
                "box": box,
                "label": det.get("label", "object"),
            })

        # Initialize tracking
        tracking_result = self.init_tracking(image, objects)

        # Add detection info to result
        tracking_result["detections"] = detections
        tracking_result["text_prompts"] = text_prompts

        return tracking_result

    def init_tracking(
        self,
        image: Image.Image,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Initialize video tracking session.

        Args:
            image: First frame as PIL Image
            objects: List of objects to track, each with:
                - object_id: unique ID
                - box: [x1, y1, x2, y2] bounding box

        Returns:
            Dict with session_id and initial tracking results
        """
        session_id = f"sam3_track_{uuid.uuid4().hex[:12]}"

        # Encode first frame
        embeddings = self.encode(image)

        # Initialize tracking state for each object
        tracking_state = {
            "session_id": session_id,
            "frame_idx": 0,
            "objects": {},
            "image_size": image.size,
            "last_embeddings": embeddings,
        }

        results = []
        for obj in objects:
            obj_id = obj.get("object_id", len(tracking_state["objects"]))
            box = obj.get("box", [0, 0, 100, 100])

            # Store initial state
            tracking_state["objects"][obj_id] = {
                "box": box,
                "label": obj.get("label", "object"),
                "memory": None,  # Will store object memory features
            }

            # Generate initial mask using tracker decoder
            mask_result = self._decode_box_prompt(embeddings, box, image.size)

            results.append({
                "object_id": obj_id,
                "box": box,
                "mask": mask_result.get("mask"),
                "score": mask_result.get("score", 1.0),
            })

            # Update memory with mask
            tracking_state["objects"][obj_id]["memory"] = mask_result.get("memory")

        # Store state in Redis
        self.cache.set("tracking", session_id, tracking_state)

        return {
            "session_id": session_id,
            "frame_idx": 0,
            "tracked_objects": results,
        }

    def track_frame(
        self,
        session_id: str,
        image: Image.Image,
        frame_idx: int,
    ) -> Dict[str, Any]:
        """
        Track objects to a new frame.

        Args:
            session_id: Tracking session ID
            image: New frame as PIL Image
            frame_idx: Frame index

        Returns:
            Dict with updated tracking results
        """
        # Get tracking state
        tracking_state = self.cache.get("tracking", session_id)
        if tracking_state is None:
            return {"error": f"Session not found: {session_id}"}

        # Encode new frame
        embeddings = self.encode(image)

        results = []
        for obj_id, obj_state in tracking_state["objects"].items():
            # Propagate using memory
            prev_memory = obj_state.get("memory")
            prev_box = obj_state.get("box")

            # Run tracker decoder with memory conditioning
            track_result = self._track_with_memory(
                embeddings,
                prev_memory,
                prev_box,
                image.size,
            )

            # Update state
            obj_state["box"] = track_result.get("box", prev_box)
            obj_state["memory"] = track_result.get("memory", prev_memory)

            results.append({
                "object_id": obj_id,
                "box": obj_state["box"],
                "mask": track_result.get("mask"),
                "score": track_result.get("score", 1.0),
            })

        # Update tracking state
        tracking_state["frame_idx"] = frame_idx
        tracking_state["last_embeddings"] = embeddings
        self.cache.set("tracking", session_id, tracking_state)

        return {
            "session_id": session_id,
            "frame_idx": frame_idx,
            "tracked_objects": results,
        }

    def _decode_box_prompt(
        self,
        embeddings: Dict[str, np.ndarray],
        box: List[float],
        original_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Decode a box prompt to mask using tracker decoder.

        Converts the box to two points (top-left and bottom-right corners)
        and runs the decoder to get a mask.
        """
        decoder = self._get_tracker_decoder()

        # Get FPN features from embeddings
        fpn_feat_0 = embeddings.get("fpn_feat_0")
        fpn_feat_1 = embeddings.get("fpn_feat_1")
        fpn_feat_2 = embeddings.get("fpn_feat_2")

        if fpn_feat_0 is None or fpn_feat_1 is None or fpn_feat_2 is None:
            logger.error("Missing FPN features in embeddings")
            return {"mask": None, "box": box, "score": 0.0, "memory": None}

        # Convert box [x1, y1, x2, y2] to point prompts
        # For a box, we use two corner points with label=2 (box mode in SAM3)
        # Alternatively, use center point with label=1 (positive click)
        orig_w, orig_h = original_size

        # Scale box from original image coords to SAM3 coords (1008x1008)
        scale_x = SAM3_IMAGE_SIZE / orig_w
        scale_y = SAM3_IMAGE_SIZE / orig_h
        box_scaled = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y,
        ]

        # Use center point as a positive click
        center_x = (box_scaled[0] + box_scaled[2]) / 2
        center_y = (box_scaled[1] + box_scaled[3]) / 2

        # Prepare inputs - HuggingFace expects 4D point_coords [B, num_objects, num_points, 2]
        point_coords = np.array([[[[center_x, center_y]]]], dtype=np.float32)
        point_labels = np.array([[[1.0]]], dtype=np.float32)  # 1 = positive

        inputs = {
            "fpn_feat_0": fpn_feat_0,
            "fpn_feat_1": fpn_feat_1,
            "fpn_feat_2": fpn_feat_2,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0.0], dtype=np.float32),
        }

        try:
            outputs = decoder.run(None, inputs)

            # Parse outputs: masks [B,3,1008,1008], iou_predictions [B,3], low_res_masks [B,3,288,288]
            masks = outputs[0]
            iou_predictions = outputs[1]

            # Select best mask based on IoU prediction
            best_idx = int(np.argmax(iou_predictions[0]))
            mask = masks[0, best_idx]  # [1008, 1008]
            score = float(iou_predictions[0, best_idx])

            # Resize mask to original size
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h))

            mask_binary = (mask > 0).astype(np.uint8)

            # Compute bounding box from mask
            ys, xs = np.where(mask_binary > 0)
            if len(xs) > 0:
                new_box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            else:
                new_box = box

            return {
                "mask": mask_binary,
                "box": new_box,
                "score": score,
                "memory": fpn_feat_2,  # Store embeddings for potential memory-based tracking
            }
        except Exception as e:
            logger.error(f"Decoder failed: {e}", exc_info=True)
            return {"mask": None, "box": box, "score": 0.0, "memory": None}

    def _track_with_memory(
        self,
        embeddings: Dict[str, np.ndarray],
        prev_memory: Optional[Dict[str, np.ndarray]],
        prev_box: List[float],
        original_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Track object using memory from previous frames.

        This implements SAM3's video propagation pipeline:
        1. Condition current frame features with memory (memory_attention)
        2. Decode mask using conditioned features (tracker_decoder)
        3. Encode new mask into memory (memory_encoder)
        4. Extract object pointer (object_pointer)

        Args:
            embeddings: Current frame embeddings from vision encoder
            prev_memory: Memory state from previous frames containing:
                - memory_bank: [B, num_frames*seq, 64] accumulated memory features
                - memory_pos_bank: [B, num_frames*seq, 64] accumulated position encodings
                - object_pointer: [B, 256] object representation
                - frame_count: number of frames in memory
            prev_box: Previous frame bounding box [x1, y1, x2, y2]
            original_size: Original image size (width, height)

        Returns:
            Dict with mask, box, score, and updated memory state
        """
        # Check if memory components are available
        try:
            mem_attn = self._get_memory_attention()
            mem_enc = self._get_memory_encoder()
            obj_ptr = self._get_object_pointer()
            temporal_pos = self._get_temporal_pos_enc()
        except FileNotFoundError as e:
            logger.warning(f"Memory components not available: {e}")
            # Fall back to box-only tracking
            return self._decode_box_prompt(embeddings, prev_box, original_size)

        # Get FPN features
        fpn_feat_0 = embeddings.get("fpn_feat_0")  # [1, 256, 288, 288]
        fpn_feat_1 = embeddings.get("fpn_feat_1")  # [1, 256, 144, 144]
        fpn_feat_2 = embeddings.get("fpn_feat_2")  # [1, 256, 72, 72]

        if fpn_feat_0 is None or fpn_feat_1 is None or fpn_feat_2 is None:
            logger.error("Missing FPN features in embeddings")
            return self._decode_box_prompt(embeddings, prev_box, original_size)

        # Get spatial dimensions
        H, W = fpn_feat_2.shape[2], fpn_feat_2.shape[3]  # 72, 72
        seq_len = H * W  # 5184

        # Prepare current frame features for memory attention
        # Flatten spatial dims: [B, 256, H, W] -> [B, H*W, 256]
        current_features = fpn_feat_2.transpose(0, 2, 3, 1).reshape(1, seq_len, 256)

        # Generate position encoding for current frame (2D sine/cosine)
        current_pos_enc = self._generate_2d_pos_enc(H, W, dim=256)  # [1, seq_len, 256]

        # If no previous memory, use box prompt only for first frame
        if prev_memory is None or prev_memory.get("frame_count", 0) == 0:
            result = self._decode_box_prompt(embeddings, prev_box, original_size)

            # Initialize memory from first frame mask
            if result.get("mask") is not None:
                memory = self._encode_mask_to_memory(
                    fpn_feat_2, result["mask"], original_size, H, W, mem_enc
                )
                # Get object pointer from decoder output (if available)
                obj_pointer = np.zeros((1, 256), dtype=np.float32)

                result["memory"] = {
                    "memory_bank": memory["features"].reshape(1, seq_len, 64),
                    "memory_pos_bank": memory["pos_enc"].reshape(1, seq_len, 64),
                    "object_pointer": obj_pointer,
                    "frame_count": 1,
                }
            return result

        # Run memory attention to condition current features
        memory_bank = prev_memory["memory_bank"]  # [1, num_frames*seq, 64]
        memory_pos_bank = prev_memory["memory_pos_bank"]  # [1, num_frames*seq, 64]

        # Add temporal position encoding to memory
        frame_count = prev_memory.get("frame_count", 1)
        memory_with_temporal = self._add_temporal_pos_enc(
            memory_bank, temporal_pos, frame_count, seq_len
        )

        # Run memory attention
        try:
            conditioned_features = mem_attn.run(
                None,
                {
                    "current_vision_features": current_features.astype(np.float32),
                    "memory": memory_with_temporal.astype(np.float32),
                    "current_vision_pos_enc": current_pos_enc.astype(np.float32),
                    "memory_pos_enc": memory_pos_bank.astype(np.float32),
                }
            )[0]  # [1, seq_len, 256]
        except Exception as e:
            logger.error(f"Memory attention failed: {e}")
            return self._decode_box_prompt(embeddings, prev_box, original_size)

        # Reshape back to spatial format for decoder
        # [1, seq_len, 256] -> [1, 256, H, W]
        conditioned_fpn_2 = conditioned_features.transpose(0, 2, 1).reshape(1, 256, H, W)

        # Run decoder with conditioned features
        decoder = self._get_tracker_decoder()

        # Use center of previous box as prompt
        orig_w, orig_h = original_size
        scale_x = SAM3_IMAGE_SIZE / orig_w
        scale_y = SAM3_IMAGE_SIZE / orig_h

        center_x = (prev_box[0] + prev_box[2]) / 2 * scale_x
        center_y = (prev_box[1] + prev_box[3]) / 2 * scale_y

        point_coords = np.array([[[[center_x, center_y]]]], dtype=np.float32)
        point_labels = np.array([[[1.0]]], dtype=np.float32)

        try:
            outputs = decoder.run(
                None,
                {
                    "fpn_feat_0": fpn_feat_0.astype(np.float32),
                    "fpn_feat_1": fpn_feat_1.astype(np.float32),
                    "fpn_feat_2": conditioned_fpn_2.astype(np.float32),  # Use conditioned features
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
                    "has_mask_input": np.array([0.0], dtype=np.float32),
                }
            )

            # Parse outputs
            pred_masks = outputs[0]  # [1, 1, H, W]
            iou_scores = outputs[1] if len(outputs) > 1 else np.array([[1.0]])

            # Get best mask
            mask_idx = np.argmax(iou_scores)
            mask = pred_masks[0, mask_idx]
            score = float(iou_scores[0, mask_idx])

            # Resize mask to original size
            mask_resized = self._resize_mask(mask, original_size)

            # Convert to binary and get bounding box
            binary_mask = (mask_resized > 0).astype(np.uint8)
            new_box = self._mask_to_box(binary_mask) or prev_box

        except Exception as e:
            logger.error(f"Decoder with memory failed: {e}")
            return self._decode_box_prompt(embeddings, prev_box, original_size)

        # Encode new mask into memory
        new_memory_feat = self._encode_mask_to_memory(
            conditioned_fpn_2, mask_resized, original_size, H, W, mem_enc
        )

        # Update memory bank (keep last N frames, FIFO)
        max_memory_frames = 7  # SAM3 default
        if frame_count >= max_memory_frames:
            # Remove oldest frame
            memory_bank = memory_bank[:, seq_len:, :]
            memory_pos_bank = memory_pos_bank[:, seq_len:, :]

        # Append new memory
        new_memory_flat = new_memory_feat["features"].reshape(1, seq_len, 64)
        new_pos_flat = new_memory_feat["pos_enc"].reshape(1, seq_len, 64)

        updated_memory_bank = np.concatenate([memory_bank, new_memory_flat], axis=1)
        updated_pos_bank = np.concatenate([memory_pos_bank, new_pos_flat], axis=1)

        return {
            "mask": mask_resized,
            "box": new_box,
            "score": score,
            "memory": {
                "memory_bank": updated_memory_bank,
                "memory_pos_bank": updated_pos_bank,
                "object_pointer": prev_memory.get("object_pointer", np.zeros((1, 256))),
                "frame_count": min(frame_count + 1, max_memory_frames),
            },
        }

    def _generate_2d_pos_enc(self, h: int, w: int, dim: int = 256) -> np.ndarray:
        """Generate 2D sinusoidal position encoding."""
        y_pos = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
        x_pos = np.arange(w).reshape(1, -1).repeat(h, axis=0)

        pos_enc = np.zeros((h, w, dim), dtype=np.float32)

        div_term = np.exp(np.arange(0, dim // 2, 2) * -(np.log(10000.0) / (dim // 2)))

        pos_enc[:, :, 0::4] = np.sin(x_pos[:, :, np.newaxis] * div_term)
        pos_enc[:, :, 1::4] = np.cos(x_pos[:, :, np.newaxis] * div_term)
        pos_enc[:, :, 2::4] = np.sin(y_pos[:, :, np.newaxis] * div_term)
        pos_enc[:, :, 3::4] = np.cos(y_pos[:, :, np.newaxis] * div_term)

        return pos_enc.reshape(1, h * w, dim)

    def _add_temporal_pos_enc(
        self,
        memory_bank: np.ndarray,
        temporal_pos: np.ndarray,
        frame_count: int,
        seq_len: int,
    ) -> np.ndarray:
        """Add temporal position encoding to memory bank."""
        # temporal_pos shape: [max_frames, 64]
        # memory_bank shape: [1, frame_count * seq_len, 64]

        result = memory_bank.copy()

        for i in range(frame_count):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            if i < len(temporal_pos):
                result[0, start_idx:end_idx, :] += temporal_pos[i:i+1, :]

        return result

    def _encode_mask_to_memory(
        self,
        fpn_feat: np.ndarray,
        mask: np.ndarray,
        original_size: Tuple[int, int],
        h: int,
        w: int,
        mem_enc,
    ) -> Dict[str, np.ndarray]:
        """Encode mask into memory representation."""
        # Resize mask to memory encoder expected size
        mask_h = h * 16  # 1152
        mask_w = w * 16  # 1152

        # Resize mask from original size to expected size
        from PIL import Image
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_resized = np.array(mask_img.resize((mask_w, mask_h), Image.NEAREST)) / 255.0
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)

        try:
            outputs = mem_enc.run(
                None,
                {
                    "vision_features": fpn_feat.astype(np.float32),
                    "masks": mask_input,
                }
            )
            return {
                "features": outputs[0],  # [1, 64, H, W]
                "pos_enc": outputs[1],   # [1, 64, H, W]
            }
        except Exception as e:
            logger.error(f"Memory encoder failed: {e}")
            return {
                "features": np.zeros((1, 64, h, w), dtype=np.float32),
                "pos_enc": np.zeros((1, 64, h, w), dtype=np.float32),
            }

    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize mask to target size using PIL."""
        from PIL import Image
        mask_img = Image.fromarray(mask.astype(np.float32))
        resized = mask_img.resize(target_size, Image.BILINEAR)
        return np.array(resized)

    def _mask_to_box(self, binary_mask: np.ndarray) -> Optional[List[float]]:
        """Extract bounding box from binary mask."""
        coords = np.argwhere(binary_mask > 0)
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return [float(x_min), float(y_min), float(x_max), float(y_max)]

    def clear_tracking(self, session_id: str) -> Dict[str, Any]:
        """Clear a tracking session."""
        success = self.cache.delete("tracking", session_id)
        return {"cleared": success, "session_id": session_id}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and capabilities."""
        has_vision = os.path.exists(self.paths["vision_encoder"])
        has_text = os.path.exists(self.paths["text_encoder"])
        has_pcs = os.path.exists(self.paths["pcs_decoder"])
        has_tracker = os.path.exists(self.paths["tracker_decoder"])

        capabilities = []
        if has_vision:
            capabilities.append("encode")
            capabilities.append("encode_batch")
        if has_vision and has_text and has_pcs:
            capabilities.append("text_to_segment")
            capabilities.append("text_to_segment_with_boxes")
            capabilities.append("get_semantic_mask")
            capabilities.append("get_labeled_semantic_mask")
        if has_vision and has_tracker:
            capabilities.append("track")
            capabilities.append("init_tracking")
            capabilities.append("track_frame")
            capabilities.append("automatic_mask_generation")
        if has_vision and has_text and has_pcs and has_tracker:
            capabilities.append("init_tracking_from_text")  # Video PCS

        # Check memory components for enhanced video propagation
        has_memory_attention = os.path.exists(self.paths["memory_attention"])
        has_memory_encoder = os.path.exists(self.paths["memory_encoder"])
        has_object_pointer = os.path.exists(self.paths["object_pointer"])
        has_memory_components = has_memory_attention and has_memory_encoder and has_object_pointer

        if has_memory_components:
            capabilities.append("video_propagation")  # Full memory-based tracking

        return {
            "vision_encoder": has_vision,
            "text_encoder": has_text,
            "pcs_decoder": has_pcs,
            "tracker_decoder": has_tracker,
            "memory_attention": has_memory_attention,
            "memory_encoder": has_memory_encoder,
            "object_pointer": has_object_pointer,
            "model_dir": self.paths["model_dir"],
            "device": self.device,
            "capabilities": capabilities,
            "features": {
                "box_prompts": has_pcs,  # Box prompts for PCS
                "negative_prompts": has_pcs,  # Negative boxes (exclude regions)
                "batched_inference": has_vision,  # encode_batch
                "automatic_mask_generation": has_tracker,  # AMG
                "semantic_segmentation": has_pcs,  # get_semantic_mask
                "video_pcs": has_vision and has_text and has_pcs and has_tracker,
                "memory_propagation": has_memory_components,  # Memory-based video propagation
            },
        }


# =============================================================================
# Singleton Pattern
# =============================================================================

_handler: Optional[UnifiedModelHandler] = None
_handler_lock = threading.Lock()


def get_handler() -> UnifiedModelHandler:
    """Get or create the singleton handler instance (thread-safe)."""
    global _handler
    if _handler is None:
        with _handler_lock:
            if _handler is None:
                # Check environment variable first, then auto-detect
                device = os.environ.get("SAM3_DEVICE")
                if device is None:
                    try:
                        import onnxruntime as ort
                        providers = ort.get_available_providers()
                        device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
                    except:
                        device = "cpu"
                logger.info(f"Initializing UnifiedModelHandler on {device}")
                _handler = UnifiedModelHandler(device=device)
    return _handler


def reset_handler() -> None:
    """Reset the singleton handler (for testing)."""
    global _handler
    with _handler_lock:
        _handler = None
