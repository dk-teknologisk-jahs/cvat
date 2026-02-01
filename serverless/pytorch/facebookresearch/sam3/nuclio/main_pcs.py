# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Text-to-Segment Nuclio Function Handler

Server-side text-to-segment for SAM3 PCS (Perception Continuum Segmentation).
Runs the full pipeline on server and returns segmentation masks.

This handler supports:
- Text prompts: Segment objects matching text descriptions
- Combined text + box: Use box guidance with text description
- Multi-instance: Find all instances of a concept in the image

Expected input:
{
    "image": "<base64 encoded image>",
    "text_prompts": ["a person", "a car"],  # List of text descriptions
    "confidence_threshold": 0.3,  # Optional, default 0.3
    "box": [x1, y1, x2, y2],  # Optional box guidance (pixel coords)
    "is_positive_box": true,  # Optional, default true
}

Returns:
{
    "detections": [
        {
            "mask": "<base64 RLE encoded mask>",
            "box": [x1, y1, x2, y2],
            "score": 0.95,
            "label": "a person"
        },
        ...
    ]
}
"""

import base64
import io
import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils  # For RLE encoding

from model_handler_pcs import ModelHandlerPCS


# Global PCS handler (lazy init)
_pcs_handler = None


def init_context(context):
    """Initialize the SAM3 PCS model."""
    global _pcs_handler
    context.logger.info("Init SAM3-PCS context... 0%")
    _pcs_handler = ModelHandlerPCS(confidence_threshold=0.3)
    context.user_data.pcs_handler = _pcs_handler
    context.logger.info("Init SAM3-PCS context... 100%")


def encode_mask_rle(mask: np.ndarray) -> Dict:
    """
    Encode a binary mask using RLE (Run-Length Encoding).

    Args:
        mask: np.ndarray [H, W] boolean or uint8 mask

    Returns:
        RLE dict with 'counts' (base64 string) and 'size' [H, W]
    """
    # Ensure mask is uint8 and Fortran contiguous (required by pycocotools)
    mask_uint8 = np.asfortranarray(mask.astype(np.uint8))

    # Encode using COCO RLE
    rle = mask_utils.encode(mask_uint8)

    # Convert counts bytes to base64 for JSON serialization
    rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
    rle['size'] = list(rle['size'])

    return rle


def handler(context, event):
    """
    Handle text-to-segment requests.

    Expected input:
    {
        "image": "<base64 encoded image>",
        "text_prompts": ["a person", "a car"],
        "confidence_threshold": 0.3,  # Optional
        "box": [x1, y1, x2, y2],  # Optional
        "is_positive_box": true,  # Optional
    }

    Returns:
    {
        "detections": [
            {
                "mask": {"counts": "<base64 RLE>", "size": [H, W]},
                "box": [x1, y1, x2, y2],
                "score": 0.95,
                "label": "a person"
            },
            ...
        ]
    }
    """
    try:
        context.logger.info("SAM3-PCS handler called")

        data = event.body
        pcs_handler = context.user_data.pcs_handler

        # Decode image
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")

        # Get parameters
        text_prompts = data.get("text_prompts", [])
        confidence_threshold = data.get("confidence_threshold", 0.3)
        box = data.get("box")
        is_positive_box = data.get("is_positive_box", True)

        if not text_prompts:
            return context.Response(
                body=json.dumps({'error': 'No text_prompts provided'}),
                headers={},
                content_type='application/json',
                status_code=400
            )

        # Update confidence threshold if specified
        pcs_handler.set_confidence_threshold(confidence_threshold)

        # Set image
        pcs_handler.set_image(image)

        # Run text-to-segment
        if box:
            # Combined text + box
            results = pcs_handler.text_and_box_to_segment(
                text_prompt=text_prompts[0],  # Use first prompt with box
                box=box,
                is_positive=is_positive_box,
            )
        else:
            # Text-only
            results = pcs_handler.text_to_segment(text_prompts)

        # Format response
        detections = []
        for r in results:
            detection = {
                'mask': encode_mask_rle(r['mask']),
                'box': r['box'],
                'score': r['score'],
                'label': r['label'],
            }
            detections.append(detection)

        response = {
            'detections': detections,
            'image_size': [image.height, image.width],
        }

        return context.Response(
            body=json.dumps(response),
            headers={},
            content_type='application/json',
            status_code=200
        )

    except Exception as e:
        context.logger.error(f"Error in SAM3-PCS handler: {str(e)}", exc_info=True)
        return context.Response(
            body=json.dumps({'error': str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )


def test_handler():
    """Test the handler locally."""
    import time

    print("Testing SAM3-PCS handler...")

    # Mock context
    class MockContext:
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg, **kwargs): print(f"ERROR: {msg}")
        logger = MockLogger()
        class MockUserData:
            pcs_handler = None
        user_data = MockUserData()

        class Response:
            def __init__(self, body, headers, content_type, status_code):
                self.body = body
                self.status_code = status_code

    context = MockContext()

    # Initialize
    init_context(context)

    # Load test image
    test_image = Image.open("/home/jahs/GitHub/cvat/usls/assets/bus.jpg")

    # Create mock event
    class MockEvent:
        buf = io.BytesIO()
        test_image.save(buf, format='PNG')
        body = {
            "image": base64.b64encode(buf.getvalue()).decode('utf-8'),
            "text_prompts": ["a person", "a bus"],
            "confidence_threshold": 0.3,
        }

    event = MockEvent()

    # Call handler
    print("\nCalling handler...")
    start_time = time.time()
    response = handler(context, event)
    elapsed = time.time() - start_time

    print(f"\nResponse status: {response.status_code}")
    result = json.loads(response.body)

    if 'detections' in result:
        print(f"Found {len(result['detections'])} detections in {elapsed:.2f}s")
        for d in result['detections'][:5]:
            print(f"  - {d['label']}: score={d['score']:.3f}, box={[round(x) for x in d['box']]}")
            print(f"    mask size: {d['mask']['size']}")
    else:
        print(f"Error: {result.get('error')}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_handler()
