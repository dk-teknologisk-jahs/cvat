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

# Constants
SAM3_IMAGE_SIZE = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


def get_model_paths(model_dir: Optional[str] = None) -> Dict[str, str]:
    """Get model paths from environment or provided directory."""
    base_dir = model_dir or os.environ.get("SAM3_MODEL_DIR", DEFAULT_MODEL_DIR)
    return {
        "model_dir": base_dir,
        "vision_encoder": os.environ.get("SAM3_VISION_ENCODER", f"{base_dir}/vision_encoder.onnx"),
        "text_encoder": os.environ.get("SAM3_TEXT_ENCODER", f"{base_dir}/text_encoder.onnx"),
        "pcs_decoder": os.environ.get("SAM3_PCS_DECODER", f"{base_dir}/pcs_decoder.onnx"),
        "tracker_decoder": os.environ.get("SAM3_TRACKER_DECODER", f"{base_dir}/tracker_decoder.onnx"),
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

        # Image preprocessing params
        self.image_size = SAM3_IMAGE_SIZE
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        logger.info(f"UnifiedModelHandler initialized (device={device}, model_dir={self.paths['model_dir']})")

    # =========================================================================
    # Lazy Model Loading
    # =========================================================================

    def _get_vision_encoder(self):
        """Lazy load vision encoder ONNX model."""
        if self._vision_encoder is None:
            import onnxruntime as ort
            path = self.paths["vision_encoder"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Vision encoder not found: {path}\n"
                    "Run export_hf_onnx.py to export ONNX models."
                )
            logger.info(f"Loading vision encoder: {path}")
            self._vision_encoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._vision_encoder

    def _get_text_encoder(self):
        """Lazy load text encoder ONNX model."""
        if self._text_encoder is None:
            import onnxruntime as ort
            path = self.paths["text_encoder"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Text encoder not found: {path}\n"
                    "Run export_hf_onnx.py to export ONNX models."
                )
            logger.info(f"Loading text encoder: {path}")
            self._text_encoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._text_encoder

    def _get_pcs_decoder(self):
        """Lazy load PCS decoder ONNX model."""
        if self._pcs_decoder is None:
            import onnxruntime as ort
            path = self.paths["pcs_decoder"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"PCS decoder not found: {path}\n"
                    "Run export_hf_onnx.py to export ONNX models."
                )
            logger.info(f"Loading PCS decoder: {path}")
            self._pcs_decoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._pcs_decoder

    def _get_tracker_decoder(self):
        """Lazy load tracker decoder ONNX model."""
        if self._tracker_decoder is None:
            import onnxruntime as ort
            path = self.paths["tracker_decoder"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Tracker decoder not found: {path}\n"
                    "Run export_hf_onnx.py to export ONNX models."
                )
            logger.info(f"Loading tracker decoder: {path}")
            self._tracker_decoder = ort.InferenceSession(
                path,
                sess_options=self.sess_options,
                providers=self.providers,
            )
        return self._tracker_decoder

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

    # =========================================================================
    # Text-to-Segment (Detector/PCS Mode)
    # =========================================================================

    def text_to_segment(
        self,
        text_prompts: List[str],
        image: Image.Image,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Run text-to-segment (PCS mode) using ONNX models.

        Args:
            text_prompts: List of text prompts (e.g., ["a person", "a car"])
            image: PIL Image
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detections: [{"mask": np.ndarray, "box": [x1,y1,x2,y2], "score": float, "label": str}, ...]
        """
        # Get encoders and decoder
        vision_encoder = self._get_vision_encoder()
        text_encoder = self._get_text_encoder()
        pcs_decoder = self._get_pcs_decoder()

        original_size = image.size  # (W, H)

        # 1. Encode image
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

        # 2. Encode text prompts
        text_features, text_mask = self._encode_text(text_encoder, text_prompts)

        # Ensure text_mask is bool type for ONNX
        text_mask = text_mask.astype(bool)

        # 3. Run PCS decoder with all required inputs
        # PCS decoder expects: fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
        #                      text_features, text_mask, input_boxes, input_boxes_labels

        # Use padding boxes (label=-10 means ignored/padding)
        batch_size = text_features.shape[0]
        input_boxes = np.zeros((batch_size, 1, 4), dtype=np.float32)
        input_boxes_labels = np.full((batch_size, 1), -10, dtype=np.int64)

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
            text_prompts,
            confidence_threshold,
        )

        logger.info(f"Text-to-segment found {len(detections)} objects for '{text_prompts}'")
        return detections

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

        # Convert logits to scores via sigmoid
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

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

            # Get mask
            mask = None
            if masks is not None:
                mask_raw = masks[i]
                # Resize mask to original size
                if mask_raw.shape[-2:] != (orig_h, orig_w):
                    mask_raw = cv2.resize(
                        mask_raw.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = (mask_raw > 0).astype(np.uint8)

            detections.append({
                "mask": mask,
                "box": box,
                "score": float(score),
                "label": text_prompts[0] if text_prompts else "object",
            })

        return detections

    # =========================================================================
    # Video Tracking
    # =========================================================================

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
        prev_memory: Optional[np.ndarray],
        prev_box: List[float],
        original_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Track object using memory from previous frame."""
        # For now, use box prompt decoding
        # Full video tracking would use memory bank cross-attention
        return self._decode_box_prompt(embeddings, prev_box, original_size)

    def clear_tracking(self, session_id: str) -> Dict[str, Any]:
        """Clear a tracking session."""
        success = self.cache.delete("tracking", session_id)
        return {"cleared": success, "session_id": session_id}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "vision_encoder": os.path.exists(self.paths["vision_encoder"]),
            "text_encoder": os.path.exists(self.paths["text_encoder"]),
            "pcs_decoder": os.path.exists(self.paths["pcs_decoder"]),
            "tracker_decoder": os.path.exists(self.paths["tracker_decoder"]),
            "model_dir": self.paths["model_dir"],
            "device": self.device,
            "capabilities": [
                "encode" if os.path.exists(self.paths["vision_encoder"]) else None,
                "text_to_segment" if (
                    os.path.exists(self.paths["vision_encoder"]) and
                    os.path.exists(self.paths["text_encoder"]) and
                    os.path.exists(self.paths["pcs_decoder"])
                ) else None,
                "track" if (
                    os.path.exists(self.paths["vision_encoder"]) and
                    os.path.exists(self.paths["tracker_decoder"])
                ) else None,
            ],
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
                # Check for GPU
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
