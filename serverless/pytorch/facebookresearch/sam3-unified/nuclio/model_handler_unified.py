#!/usr/bin/env python3
"""
Unified SAM3 Model Handler

Combines both interactor (point/box prompts) and text-to-segment (PCS) modes
into a single handler, loading the SAM3 model only once.

This saves GPU VRAM by avoiding duplicate model loading:
- Single model load: ~3.5 GB
- Vs two separate functions: ~5.5 GB

Supported endpoints:
1. /api/encode - Interactor mode (returns embeddings for browser-side decoding)
2. /api/text-to-segment - PCS mode (returns complete masks)

The underlying SAM3 model supports both modes through:
- `set_image()` - Run vision encoder, cache features
- `predict()` - Point/box prompts → masks (interactor)
- `forward_grounding()` - Text prompts → masks (PCS)
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Constants
SAM3_IMAGE_SIZE = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


class UnifiedModelHandler:
    """
    Unified SAM3 handler supporting both interactor and text-to-segment modes.

    Loads the full SAM3 model once, providing:
    - encode() - For interactor mode (returns embeddings)
    - text_to_segment() - For PCS mode (returns masks)
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

        # Build the full SAM3 model with all capabilities
        print(f"Loading unified SAM3 model on {self.device}...")
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,        # Enable PCS/text-to-segment
            enable_inst_interactivity=True,  # Enable tracker/interactor mode
        )

        # Create processor for text-to-segment API
        self.processor = Sam3Processor(
            model=self.model,
            resolution=SAM3_IMAGE_SIZE,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

        # Get tracker for interactor mode (created by enable_inst_interactivity=True)
        self.tracker = self.model.tracker

        # Image preprocessing params
        self.image_size = SAM3_IMAGE_SIZE
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Cache for current image
        self._current_image_hash: Optional[int] = None
        self._cached_embeddings: Optional[Dict] = None
        self._cached_backbone_out: Optional[Dict] = None

        print("Unified SAM3 handler initialized")
        print(f"  - Interactor mode: {'enabled' if self.tracker else 'disabled'}")
        print(f"  - Text-to-segment mode: enabled")

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
        image_hash = hash(image.tobytes())

        # Check cache
        if image_hash == self._current_image_hash and self._cached_embeddings is not None:
            return self._cached_embeddings

        # Preprocess
        input_tensor = self.preprocess_image(image)
        input_tensor = torch.from_numpy(input_tensor).to(self.device)

        # Run vision encoder
        with torch.no_grad():
            backbone_out = self.model.backbone.forward_image(input_tensor)

        # Extract FPN features
        fpn = backbone_out["backbone_fpn"]

        # Apply channel projections for high-res features (same as tracker does)
        high_res_0 = self.tracker.sam_mask_decoder.conv_s0(fpn[0])  # 256 → 32 channels
        high_res_1 = self.tracker.sam_mask_decoder.conv_s1(fpn[1])  # 256 → 64 channels
        image_embed = fpn[2]  # 256 channels (no projection)

        # Convert to numpy
        embeddings = {
            'high_res_feats_0': high_res_0.cpu().numpy(),  # [1, 32, 288, 288]
            'high_res_feats_1': high_res_1.cpu().numpy(),  # [1, 64, 144, 144]
            'image_embed': image_embed.cpu().numpy(),      # [1, 256, 72, 72]
        }

        # Cache
        self._current_image_hash = image_hash
        self._cached_embeddings = embeddings
        self._cached_backbone_out = backbone_out

        return embeddings

    # =========================================================================
    # TEXT-TO-SEGMENT MODE (PCS)
    # =========================================================================

    def set_image_for_text(self, image: Image.Image) -> Dict:
        """
        Set image for text-to-segment mode.

        Uses the processor API which handles its own caching.

        Args:
            image: PIL Image

        Returns:
            Dictionary with image metadata
        """
        # Use processor's set_image
        self.processor.set_image(image)

        return {
            "original_size": image.size,  # (W, H)
            "model_input_size": (self.image_size, self.image_size),
        }

    def text_to_segment(
        self,
        text_prompts: List[str],
        image: Optional[Image.Image] = None,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run text-to-segment (PCS mode).

        Args:
            text_prompts: List of text descriptions (e.g., ["a person", "a car"])
            image: PIL Image (optional if set_image_for_text was called)
            confidence_threshold: Override default threshold

        Returns:
            List of detections, each with:
            - mask: np.ndarray [H, W] boolean
            - box: [x1, y1, x2, y2] in pixel coordinates
            - score: float confidence
            - label: str text prompt that matched
        """
        if image is not None:
            self.set_image_for_text(image)

        threshold = confidence_threshold or self.confidence_threshold

        # Run forward_grounding through processor
        results = self.processor.forward_grounding(
            texts=text_prompts,
            confidence_threshold=threshold,
        )

        # Extract detections
        detections = []
        for i, (mask, box, score, label_idx) in enumerate(zip(
            results.get("masks", []),
            results.get("boxes", []),
            results.get("scores", []),
            results.get("labels", []),
        )):
            if score >= threshold:
                detections.append({
                    "mask": mask.cpu().numpy() if hasattr(mask, 'cpu') else mask,
                    "box": box.tolist() if hasattr(box, 'tolist') else list(box),
                    "score": float(score),
                    "label": text_prompts[label_idx] if label_idx < len(text_prompts) else f"object_{i}",
                })

        return detections

    def text_and_box_to_segment(
        self,
        text_prompt: str,
        box: List[float],
        image: Optional[Image.Image] = None,
    ) -> List[Dict]:
        """
        Run text-to-segment with box guidance.

        Args:
            text_prompt: Text description
            box: [x1, y1, x2, y2] bounding box in pixel coordinates
            image: PIL Image (optional if set_image_for_text was called)

        Returns:
            List of detections within the box
        """
        if image is not None:
            self.set_image_for_text(image)

        # Run forward_grounding with box prompt
        results = self.processor.forward_grounding(
            texts=[text_prompt],
            boxes=torch.tensor([box], device=self.device),
        )

        # Extract detections
        detections = []
        for mask, det_box, score in zip(
            results.get("masks", []),
            results.get("boxes", []),
            results.get("scores", []),
        ):
            detections.append({
                "mask": mask.cpu().numpy() if hasattr(mask, 'cpu') else mask,
                "box": det_box.tolist() if hasattr(det_box, 'tolist') else list(det_box),
                "score": float(score),
                "label": text_prompt,
            })

        return detections


# Singleton instance
_handler: Optional[UnifiedModelHandler] = None


def get_handler() -> UnifiedModelHandler:
    """Get or create the singleton handler instance."""
    global _handler
    if _handler is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _handler = UnifiedModelHandler(device=device)
    return _handler
