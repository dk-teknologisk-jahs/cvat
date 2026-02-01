#!/usr/bin/env python3
"""
SAM3 PCS (Perception Continuum Segmentation) Handler

Server-side handler for text-to-segment functionality using PyTorch.
Handles the full pipeline: vision encoder + text encoder + PCS decoder.

This runs entirely on the server because the PCS decoder is too complex
for efficient ONNX export (tight coupling between transformer components,
dynamic control flow, complex attention patterns).

Usage:
    handler = ModelHandlerPCS()

    # Set image (caches vision features)
    handler.set_image(pil_image)

    # Text-to-segment
    results = handler.text_to_segment(["a person", "a car"])
    # Returns: list of {mask: np.ndarray, box: [x1,y1,x2,y2], score: float}

    # Combined text + box prompt
    results = handler.text_and_box_to_segment("a person", box=[100, 100, 200, 200])
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# SAM3 constants
SAM3_IMAGE_SIZE = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


class ModelHandlerPCS:
    """
    Server-side handler for SAM3 text-to-segment (PCS mode).

    Uses the official SAM3 processor API which handles:
    - Vision encoding (caches features for reuse)
    - Text encoding
    - PCS decoder (geometry encoder + transformer + segmentation head)
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the SAM3 PCS handler.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold

        # Build the full SAM3 image model
        print(f"Loading SAM3 model on {self.device}...")
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,  # Don't need SAM2 interactive mode for PCS
        )

        # Create processor for high-level API
        self.processor = Sam3Processor(
            model=self.model,
            resolution=SAM3_IMAGE_SIZE,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

        # Cache for current image state
        self._current_state: Optional[Dict] = None
        self._current_image_hash: Optional[int] = None

        print("SAM3 PCS handler initialized")

    def set_image(self, image: Image.Image) -> Dict:
        """
        Set the current image and compute vision features.

        Features are cached so subsequent text prompts on the same image
        don't need to re-run the vision encoder.

        Args:
            image: PIL Image (any size, will be resized)

        Returns:
            State dict with cached vision features
        """
        # Check if we already have features for this image
        image_hash = hash(image.tobytes())
        if self._current_image_hash == image_hash and self._current_state is not None:
            return self._current_state

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run vision encoder
        self._current_state = self.processor.set_image(image)
        self._current_image_hash = image_hash

        return self._current_state

    def text_to_segment(
        self,
        text_prompts: Union[str, List[str]],
        image: Optional[Image.Image] = None,
    ) -> List[Dict]:
        """
        Segment objects matching text descriptions.

        Args:
            text_prompts: Single text string or list of text strings.
                         Each string describes an object type to find.
            image: Optional PIL Image. If provided, sets as current image.
                   If None, uses previously set image.

        Returns:
            List of detections, each containing:
            - 'mask': np.ndarray [H, W] boolean mask at original resolution
            - 'box': [x1, y1, x2, y2] bounding box in pixel coordinates
            - 'score': float confidence score
            - 'label': str the text prompt that matched
        """
        # Set image if provided
        if image is not None:
            self.set_image(image)

        if self._current_state is None:
            raise ValueError("No image set. Call set_image() first or provide image argument.")

        # Normalize to list
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # Process each text prompt
        all_results = []
        for text_prompt in text_prompts:
            # Reset prompts before each new text query
            self.processor.reset_all_prompts(self._current_state)

            # Run text-to-segment
            state = self.processor.set_text_prompt(text_prompt, self._current_state)

            # Extract results
            if 'masks' in state and state['masks'].numel() > 0:
                masks = state['masks'].cpu().numpy()  # [N, 1, H, W]
                boxes = state['boxes'].cpu().numpy()  # [N, 4] in xyxy format (pixel coords)
                scores = state['scores'].cpu().numpy()  # [N]

                for i in range(len(masks)):
                    mask = masks[i, 0]  # [H, W]
                    box = boxes[i]  # [4] - already in pixel coordinates
                    score = float(scores[i])

                    # Boxes are already in pixel coordinates (xyxy format)
                    box_pixels = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ]

                    all_results.append({
                        'mask': mask.astype(bool),
                        'box': box_pixels,
                        'score': score,
                        'label': text_prompt,
                    })

        # Sort by score descending
        all_results.sort(key=lambda x: x['score'], reverse=True)

        return all_results

    def text_and_box_to_segment(
        self,
        text_prompt: str,
        box: List[float],
        is_positive: bool = True,
        image: Optional[Image.Image] = None,
    ) -> List[Dict]:
        """
        Segment objects matching text description with box guidance.

        The box provides geometric guidance to focus the segmentation.

        Args:
            text_prompt: Text description of object to segment
            box: [x1, y1, x2, y2] bounding box in pixel coordinates
                 OR [cx, cy, w, h] normalized center format (if all values < 2)
            is_positive: True for positive box (include), False for negative (exclude)
            image: Optional PIL Image. If provided, sets as current image.

        Returns:
            List of detections (same format as text_to_segment)
        """
        # Set image if provided
        if image is not None:
            self.set_image(image)

        if self._current_state is None:
            raise ValueError("No image set. Call set_image() first or provide image argument.")

        # Reset prompts
        self.processor.reset_all_prompts(self._current_state)

        # First set text prompt
        state = self.processor.set_text_prompt(text_prompt, self._current_state)

        # Convert box to normalized center format [cx, cy, w, h] if needed
        orig_h = state['original_height']
        orig_w = state['original_width']

        if max(box) > 2:  # Assume pixel coordinates
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            w = (x2 - x1) / orig_w
            h = (y2 - y1) / orig_h
            box_normalized = [cx, cy, w, h]
        else:
            box_normalized = box

        # Add geometric prompt
        state = self.processor.add_geometric_prompt(
            box=box_normalized,
            label=is_positive,
            state=state,
        )

        # Extract results
        results = []
        if 'masks' in state and state['masks'].numel() > 0:
            masks = state['masks'].cpu().numpy()
            boxes = state['boxes'].cpu().numpy()  # Already in pixel coordinates
            scores = state['scores'].cpu().numpy()

            for i in range(len(masks)):
                mask = masks[i, 0]
                box_out = boxes[i]  # Already in pixel coordinates
                score = float(scores[i])

                box_pixels = [
                    float(box_out[0]),
                    float(box_out[1]),
                    float(box_out[2]),
                    float(box_out[3]),
                ]

                results.append({
                    'mask': mask.astype(bool),
                    'box': box_pixels,
                    'score': score,
                    'label': text_prompt,
                })

        return results

    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for detections."""
        self.confidence_threshold = threshold
        self.processor.set_confidence_threshold(threshold)

    def reset(self):
        """Clear the current image state."""
        self._current_state = None
        self._current_image_hash = None


def test_pcs_handler():
    """Test the PCS handler with a sample image."""
    print("Testing SAM3 PCS Handler...")

    # Create handler
    handler = ModelHandlerPCS(confidence_threshold=0.2)

    # Create a test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )

    # Set image
    print("\nSetting image...")
    handler.set_image(test_image)

    # Text-to-segment
    print("\nRunning text_to_segment...")
    results = handler.text_to_segment(["a person", "a car"])
    print(f"Found {len(results)} detections")
    for r in results:
        print(f"  - {r['label']}: score={r['score']:.3f}, box={r['box']}")

    # Combined text + box
    print("\nRunning text_and_box_to_segment...")
    results = handler.text_and_box_to_segment(
        text_prompt="an object",
        box=[100, 100, 300, 300],  # Pixel coordinates
    )
    print(f"Found {len(results)} detections with box guidance")

    print("\nTest complete!")


if __name__ == "__main__":
    test_pcs_handler()
