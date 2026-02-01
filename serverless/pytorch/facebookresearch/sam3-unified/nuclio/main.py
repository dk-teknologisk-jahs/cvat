#!/usr/bin/env python3
"""
Unified SAM3 Nuclio Handler

Single function that handles both:
1. Interactor mode (point/box prompts → embeddings for browser decoding)
2. Text-to-segment mode (text prompts → complete masks)

Routes requests based on the 'mode' parameter:
- mode='encode' or no mode → Interactor (returns embeddings)
- mode='text-to-segment' → PCS (returns masks)

This avoids loading SAM3 twice, saving ~2GB GPU VRAM.
"""

import base64
import io
import json
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from model_handler_unified import get_handler


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """
    Convert binary mask to CVAT RLE format.

    CVAT RLE format: [count1, count2, ...] where counts alternate
    between background and foreground, starting with background.

    Args:
        mask: Boolean or uint8 mask [H, W]

    Returns:
        RLE counts list
    """
    # Flatten mask in row-major order (C-style)
    flat = mask.flatten().astype(np.uint8)

    # Find where values change
    changes = np.diff(flat, prepend=flat[0])
    change_indices = np.where(changes != 0)[0]

    # Compute run lengths
    if len(change_indices) == 0:
        # All same value
        if flat[0] == 0:
            return [len(flat)]
        else:
            return [0, len(flat)]

    # Build RLE
    rle = []
    prev_idx = 0

    # If first pixel is foreground, start with 0 background count
    if flat[0] == 1:
        rle.append(0)

    for idx in change_indices:
        rle.append(idx - prev_idx)
        prev_idx = idx

    # Add final run
    rle.append(len(flat) - prev_idx)

    return rle


def handler(context, event):
    """
    Unified SAM3 handler for Nuclio.

    Request format:
    {
        "image": "<base64 encoded image>",
        "mode": "encode" | "text-to-segment",  # optional, default="encode"

        # For mode="encode" (interactor):
        # No additional params - just returns embeddings

        # For mode="text-to-segment":
        "text_prompts": ["a person", "a car"],  # required
        "threshold": 0.3,  # optional
        "box": [x1, y1, x2, y2],  # optional, for guided segmentation
    }

    Response for mode="encode":
    {
        "embeddings": {
            "high_res_feats_0": [...],  # base64 encoded
            "high_res_feats_1": [...],
            "image_embed": [...]
        },
        "shapes": {
            "high_res_feats_0": [1, 32, 288, 288],
            "high_res_feats_1": [1, 64, 144, 144],
            "image_embed": [1, 256, 72, 72]
        },
        "original_size": [height, width]
    }

    Response for mode="text-to-segment":
    [
        {
            "type": "mask",
            "label": "a person",
            "points": [rle_counts..., xtl, ytl, xbr, ybr],
            "score": 0.95
        },
        ...
    ]
    """
    # Parse request
    data = event.body
    if isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))

    # Decode image
    image_b64 = data.get("image", "")
    if not image_b64:
        return context.Response(
            body=json.dumps({"error": "No image provided"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size  # (W, H)

    # Get handler
    model = get_handler()

    # Route based on mode
    mode = data.get("mode", "encode")

    if mode == "encode":
        # =====================================================================
        # INTERACTOR MODE - Return embeddings for browser-side decoding
        # =====================================================================
        embeddings = model.encode(image)

        # Encode embeddings as base64
        encoded = {}
        shapes = {}
        for name, arr in embeddings.items():
            encoded[name] = base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')
            shapes[name] = list(arr.shape)

        return context.Response(
            body=json.dumps({
                "embeddings": encoded,
                "shapes": shapes,
                "original_size": [original_size[1], original_size[0]],  # [H, W]
            }),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    elif mode == "text-to-segment":
        # =====================================================================
        # TEXT-TO-SEGMENT MODE - Return complete masks
        # =====================================================================
        text_prompts = data.get("text_prompts", [])
        if not text_prompts:
            return context.Response(
                body=json.dumps({"error": "No text_prompts provided"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )

        threshold = data.get("threshold", 0.3)
        box = data.get("box")  # Optional

        # Run text-to-segment
        if box:
            detections = model.text_and_box_to_segment(
                text_prompt=text_prompts[0],
                box=box,
                image=image,
            )
        else:
            detections = model.text_to_segment(
                text_prompts=text_prompts,
                image=image,
                confidence_threshold=threshold,
            )

        # Format response for CVAT detector format
        results = []
        for det in detections:
            mask = det["mask"]
            box = det["box"]

            # Convert mask to RLE
            rle = mask_to_rle(mask > 0)

            # CVAT format: [rle_counts..., xtl, ytl, xbr, ybr]
            points = rle + [box[0], box[1], box[2], box[3]]

            results.append({
                "type": "mask",
                "label": det["label"],
                "points": points,
                "score": det["score"],
            })

        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    else:
        return context.Response(
            body=json.dumps({"error": f"Unknown mode: {mode}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
