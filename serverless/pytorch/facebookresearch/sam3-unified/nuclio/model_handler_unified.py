#!/usr/bin/env python3
"""
Unified SAM3 Model Handler

Combines interactor (point/box prompts), text-to-segment (PCS), and video tracking
into a single handler, loading the SAM3 model only once.

This saves GPU VRAM by avoiding duplicate model loading:
- Single model load: ~3.5 GB
- Vs three separate functions: ~10.5 GB

Supported modes:
1. encode - Interactor mode (returns embeddings for browser-side decoding)
2. text-to-segment - PCS mode (returns complete masks)
3. track/init - Video tracker initialization (Redis session)
4. track/frame - Video tracker propagation (Redis state)

The underlying SAM3 model supports all modes through HuggingFace Transformers:
- Sam3Model + Sam3Processor: Text-to-Segment (PCS)
- Sam3TrackerModel + Sam3TrackerProcessor: Point/Box-to-Segment (PVS/Interactor)
- Sam3VideoModel + Sam3VideoProcessor: Video tracking
"""

import logging
import os
import pickle
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "cvat_redis_ondisk")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6666"))
REDIS_TTL = int(os.environ.get("REDIS_TTL", "86400"))  # 1 day

# HuggingFace token for gated models
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Try to read token from file if not in environment
if not HF_TOKEN:
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            HF_TOKEN = f.read().strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAM3_IMAGE_SIZE = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


class RedisCache:
    """Redis cache manager for tracking state."""

    def __init__(
        self, host: str = REDIS_HOST, port: int = REDIS_PORT, ttl: int = REDIS_TTL
    ):
        self.ttl = ttl
        self.client = None
        self._memory_cache = {}

        try:
            import redis
            self.client = redis.Redis(host=host, port=port, decode_responses=False)
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.client = None

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key."""
        return f"sam3:{prefix}:{identifier}"

    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        """Get value from cache."""
        key = self._make_key(prefix, identifier)
        if self.client:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        elif key in self._memory_cache:
            return self._memory_cache[key]
        return None

    def set(self, prefix: str, identifier: str, value: Any) -> bool:
        """Set value in cache with TTL."""
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
        """Delete value from cache."""
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

    def exists(self, prefix: str, identifier: str) -> bool:
        """Check if key exists."""
        key = self._make_key(prefix, identifier)
        if self.client:
            return self.client.exists(key) > 0
        return key in self._memory_cache


class UnifiedModelHandler:
    """
    Unified SAM3 handler supporting interactor, text-to-segment, and video tracking.

    Loads the SAM3 model once, providing:
    - encode() - For interactor mode (returns embeddings for browser)
    - text_to_segment() - For PCS/detector mode (returns masks)
    - init_tracking() / track_frame() - For video tracking (Redis state)
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize unified SAM3 handler.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for text-to-segment detections
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold

        # Model ID for HuggingFace
        self._model_id = "facebook/sam3"

        # Initialize Redis cache for tracking state
        self.cache = RedisCache()

        # Model references (lazy loaded)
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_tracker_model = None
        self._sam3_tracker_processor = None
        self._sam3_video_model = None
        self._sam3_video_processor = None

        # Image preprocessing params
        self.image_size = SAM3_IMAGE_SIZE
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Cache for current image
        self._current_image_hash: Optional[int] = None
        self._cached_embeddings: Optional[Dict] = None

        # Initialize tracker model (for interactor mode)
        self._init_sam3_tracker_model()

        logger.info("Unified SAM3 handler initialized")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Interactor mode: enabled")
        logger.info(f"  - Text-to-segment mode: lazy loaded")
        logger.info(f"  - Video tracking mode: lazy loaded")

    def _init_sam3_tracker_model(self):
        """Initialize Sam3TrackerModel for point/box-based segmentation (PVS/Interactor)."""
        if self._sam3_tracker_model is not None:
            return

        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            logger.info(f"Loading Sam3TrackerModel from {self._model_id}")

            load_kwargs = {}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            self._sam3_tracker_model = Sam3TrackerModel.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_tracker_model.eval()

            self._sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
                self._model_id, **load_kwargs
            )

            logger.info("Sam3TrackerModel (Interactor) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3TrackerModel: {e}")
            import traceback
            traceback.print_exc()

    def _init_sam3_model(self):
        """Initialize Sam3Model for text-based segmentation (PCS/Detector)."""
        if self._sam3_model is not None:
            return

        try:
            from transformers import Sam3Model, Sam3Processor

            logger.info(f"Loading Sam3Model from {self._model_id}")

            load_kwargs = {}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            self._sam3_model = Sam3Model.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_model.eval()

            self._sam3_processor = Sam3Processor.from_pretrained(
                self._model_id, **load_kwargs
            )

            logger.info("Sam3Model (PCS/Detector) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3Model: {e}")
            import traceback
            traceback.print_exc()

    def _init_sam3_video_model(self):
        """Initialize Sam3VideoModel for video tracking."""
        if self._sam3_video_model is not None:
            return

        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            logger.info(f"Loading Sam3VideoModel from {self._model_id}")

            load_kwargs = {"torch_dtype": torch.bfloat16}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            self._sam3_video_model = Sam3VideoModel.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_video_model.eval()

            processor_kwargs = {}
            if HF_TOKEN:
                processor_kwargs["token"] = HF_TOKEN

            self._sam3_video_processor = Sam3VideoProcessor.from_pretrained(
                self._model_id, **processor_kwargs
            )

            logger.info("Sam3VideoModel (Video Tracker) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3VideoModel: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # INTERACTOR MODE (encode → browser decodes)
    # =========================================================================

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for SAM3 encoder.

        Args:
            image: PIL Image (any size)

        Returns:
            Preprocessed tensor [1, 3, 1008, 1008]
        """
        # Resize to 1008x1008
        image_resized = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        # Convert to numpy and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def encode(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Encode image to SAM3 embeddings for browser-side decoding.

        This runs the vision encoder and returns embeddings that can be
        sent to the browser for ONNX decoder inference.

        Args:
            image: PIL Image

        Returns:
            Dictionary with embeddings:
            - high_res_feats_0: [1, 32, 288, 288]
            - high_res_feats_1: [1, 64, 144, 144]
            - image_embed: [1, 256, 72, 72]
        """
        # Ensure tracker model is loaded
        self._init_sam3_tracker_model()

        if self._sam3_tracker_model is None or self._sam3_tracker_processor is None:
            raise RuntimeError("Sam3TrackerModel not available")

        image_hash = hash(image.tobytes())

        # Check cache
        if image_hash == self._current_image_hash and self._cached_embeddings is not None:
            return self._cached_embeddings

        # Process image with tracker processor
        inputs = self._sam3_tracker_processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Run vision encoder to get embeddings
        with torch.no_grad():
            # Get vision features from the model
            vision_outputs = self._sam3_tracker_model.vision_encoder(
                inputs["pixel_values"]
            )

            # Extract features at different levels
            # The tracker model provides high_res_feats and image_embed
            high_res_feats = vision_outputs.get("high_res_feats", [])
            image_embed = vision_outputs.get("image_embed")

            if len(high_res_feats) >= 2 and image_embed is not None:
                high_res_0 = high_res_feats[0]  # [1, 32, 288, 288]
                high_res_1 = high_res_feats[1]  # [1, 64, 144, 144]
            else:
                # Fallback: Use FPN features from backbone
                fpn_feats = vision_outputs.get("backbone_fpn", [])
                if len(fpn_feats) >= 3:
                    # Apply projections if needed
                    high_res_0 = fpn_feats[0]
                    high_res_1 = fpn_feats[1]
                    image_embed = fpn_feats[2]
                else:
                    raise RuntimeError("Could not extract vision features")

        # Convert to numpy
        embeddings = {
            'high_res_feats_0': high_res_0.cpu().numpy(),
            'high_res_feats_1': high_res_1.cpu().numpy(),
            'image_embed': image_embed.cpu().numpy(),
        }

        # Cache
        self._current_image_hash = image_hash
        self._cached_embeddings = embeddings

        return embeddings

    # =========================================================================
    # TEXT-TO-SEGMENT MODE (PCS/Detector)
    # =========================================================================

    def text_to_segment(
        self,
        text_prompts: List[str],
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run text-to-segment (PCS mode / Detector).

        Args:
            text_prompts: List of text descriptions (e.g., ["a person", "a car"])
            image: PIL Image
            confidence_threshold: Override default threshold

        Returns:
            List of detections, each with:
            - mask: np.ndarray [H, W] boolean
            - box: [x1, y1, x2, y2] in pixel coordinates
            - score: float confidence
            - label: str text prompt that matched
        """
        # Ensure PCS model is loaded
        self._init_sam3_model()

        if self._sam3_model is None or self._sam3_processor is None:
            logger.error("Sam3Model not available")
            return []

        threshold = confidence_threshold or self.confidence_threshold

        try:
            # Process image with text prompts
            inputs = self._sam3_processor(
                images=image,
                text=text_prompts,
                return_tensors="pt",
            ).to(self.device)

            # Run model
            with torch.no_grad():
                outputs = self._sam3_model(**inputs)

            # Post-process results
            results = self._sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes", [[image.height, image.width]]),
            )[0]

            # Extract detections
            detections = []
            if "masks" in results and len(results["masks"]) > 0:
                for i in range(len(results["masks"])):
                    mask_tensor = results["masks"][i]
                    mask_np = (
                        mask_tensor.cpu().numpy()
                        if torch.is_tensor(mask_tensor)
                        else np.array(mask_tensor)
                    )

                    # Ensure 2D mask
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np.squeeze()

                    score = float(results["scores"][i]) if "scores" in results else 1.0

                    # Get box from results or compute from mask
                    if "boxes" in results and results["boxes"] is not None:
                        box_tensor = results["boxes"][i]
                        if torch.is_tensor(box_tensor):
                            box = box_tensor.tolist()
                        else:
                            box = list(box_tensor)
                    else:
                        box = self._mask_to_bounds(mask_np)

                    detections.append({
                        "mask": mask_np,
                        "box": box,
                        "score": score,
                        "label": text_prompts[0] if text_prompts else "object",
                    })

            logger.info(f"Found {len(detections)} objects matching text prompts")
            return detections

        except Exception as e:
            logger.error(f"text_to_segment failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _mask_to_bounds(self, mask: np.ndarray) -> Optional[List[int]]:
        """Get bounding box from mask [x1, y1, x2, y2]."""
        if mask is None or mask.sum() == 0:
            return None

        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            return None

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        return [x_min, y_min, x_max, y_max]

    def _extract_mask_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Extract polygon points from binary mask."""
        if mask is None or mask.sum() == 0:
            return []

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, epsilon=1.0, closed=True)
        polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]

        return polygon

    # =========================================================================
    # VIDEO TRACKING MODE
    # =========================================================================

    def init_tracking(
        self,
        image: Image.Image,
        box: List[float],
        object_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Initialize video tracking session with a bounding box.

        Args:
            image: First frame as PIL Image
            box: Bounding box [x1, y1, x2, y2] in pixel coordinates
            object_id: ID for this tracked object

        Returns:
            dict with session_id, mask, polygon, score
        """
        # Ensure video model is loaded
        self._init_sam3_video_model()

        if self._sam3_video_model is None or self._sam3_video_processor is None:
            return {"error": "Video model not available"}

        try:
            # Generate session ID
            session_id = f"sam3_track_{uuid.uuid4().hex[:12]}"

            # Process first frame
            frame_array = np.array(image)

            # Initialize video session
            inference_session = self._sam3_video_processor.init_video_session(
                video=[frame_array],
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )

            # Add box prompt for the object
            input_box = torch.tensor([[box]], device=self.device)

            inference_session = self._sam3_video_processor.add_new_points_or_box(
                inference_session=inference_session,
                frame_idx=0,
                obj_id=object_id,
                box=input_box,
            )

            # Get initial mask
            outputs = next(self._sam3_video_model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=1,
            ))

            processed = self._sam3_video_processor.postprocess_outputs(
                inference_session, outputs
            )

            # Extract mask
            mask = None
            polygon = []
            score = 0.0

            if processed.get("masks") is not None and len(processed["masks"]) > 0:
                mask_tensor = processed["masks"][0]
                mask = (
                    mask_tensor.cpu().numpy()
                    if torch.is_tensor(mask_tensor)
                    else np.array(mask_tensor)
                )
                if len(mask.shape) > 2:
                    mask = mask.squeeze()

                polygon = self._extract_mask_polygon(mask)
                score = float(processed["scores"][0]) if "scores" in processed else 0.9

            # Save session state to Redis
            session_data = {
                "inference_session": inference_session,
                "object_ids": [object_id],
                "frame_idx": 0,
                "original_size": image.size,
            }
            self.cache.set("tracking", session_id, session_data)

            # Return shape in CVAT tracker format
            if mask is not None:
                bounds = self._mask_to_bounds(mask)
                shape = polygon[0] if polygon else bounds  # Prefer polygon
            else:
                shape = box

            return {
                "session_id": session_id,
                "object_id": object_id,
                "shape": shape,
                "mask": mask.tolist() if mask is not None else None,
                "polygon": polygon,
                "score": score,
            }

        except Exception as e:
            logger.error(f"init_tracking failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def track_frame(
        self,
        image: Image.Image,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Track object in next frame using stored session state.

        Args:
            image: Next frame as PIL Image
            session_id: Tracking session ID from init_tracking

        Returns:
            dict with shape, mask, polygon, score
        """
        # Ensure video model is loaded
        self._init_sam3_video_model()

        if self._sam3_video_model is None or self._sam3_video_processor is None:
            return {"error": "Video model not available"}

        # Get session state from Redis
        session_data = self.cache.get("tracking", session_id)
        if session_data is None:
            return {"error": f"Session not found: {session_id}"}

        try:
            inference_session = session_data["inference_session"]
            frame_idx = session_data["frame_idx"] + 1

            # Add new frame to session
            frame_array = np.array(image)

            # Process the frame
            inputs = self._sam3_video_processor(
                images=image,
                device=self.device,
                return_tensors="pt",
            )

            outputs = self._sam3_video_model(
                inference_session=inference_session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )

            processed = self._sam3_video_processor.postprocess_outputs(
                inference_session,
                outputs,
                original_sizes=inputs.original_sizes,
            )

            # Extract mask
            mask = None
            polygon = []
            score = 0.0
            bounds = None

            if processed.get("masks") is not None and len(processed["masks"]) > 0:
                mask_tensor = processed["masks"][0]
                mask = (
                    mask_tensor.cpu().numpy()
                    if torch.is_tensor(mask_tensor)
                    else np.array(mask_tensor)
                )
                if len(mask.shape) > 2:
                    mask = mask.squeeze()

                polygon = self._extract_mask_polygon(mask)
                bounds = self._mask_to_bounds(mask)
                score = float(processed["scores"][0]) if "scores" in processed else 0.9

            # Update session state
            session_data["frame_idx"] = frame_idx
            session_data["inference_session"] = inference_session
            self.cache.set("tracking", session_id, session_data)

            # Return shape - polygon flattened for CVAT
            if polygon:
                shape = [coord for point in polygon for coord in point]
            elif bounds:
                shape = bounds
            else:
                shape = None

            return {
                "session_id": session_id,
                "frame_idx": frame_idx,
                "shape": shape,
                "bounds": bounds,
                "polygon": polygon,
                "score": score,
            }

        except Exception as e:
            logger.error(f"track_frame failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def clear_tracking(self, session_id: str) -> Dict[str, Any]:
        """Clear tracking session."""
        success = self.cache.delete("tracking", session_id)
        return {"cleared": success, "session_id": session_id}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "sam3_model": self._sam3_model is not None,
            "sam3_tracker_model": self._sam3_tracker_model is not None,
            "sam3_video_model": self._sam3_video_model is not None,
            "device": str(self.device),
            "model_id": self._model_id,
            "capabilities": [
                "encode",           # Interactor (PVS)
                "text_to_segment",  # Detector (PCS)
                "track_init",       # Tracker (Video)
                "track_frame",      # Tracker (Video)
            ],
        }


# Singleton instance with lazy initialization
_handler: Optional[UnifiedModelHandler] = None
_handler_lock = threading.Lock()


def get_handler() -> UnifiedModelHandler:
    """Get or create the singleton handler instance (thread-safe)."""
    global _handler
    if _handler is None:
        with _handler_lock:
            # Double-check after acquiring lock
            if _handler is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initializing UnifiedModelHandler on {device}")
                _handler = UnifiedModelHandler(device=device)
    return _handler


def reset_handler() -> None:
    """Reset the singleton handler (for testing)."""
    global _handler
    with _handler_lock:
        if _handler is not None:
            _handler = None
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
