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

Expected input (from CVAT):
{
    "image": "<base64 encoded image>",
    "text_prompts": ["a person", "a car"],  # List of text descriptions
    "threshold": 0.3,  # Optional confidence threshold
}

Returns (CVAT detector format):
[
    {
        "type": "mask",
        "label": "a person",
        "mask": [rle_encoded_mask_data],  # CVAT RLE format
        "attributes": []
    },
    ...
]
"""

import base64
import io
import json
from typing import List

import numpy as np
from PIL import Image

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


def mask_to_cvat_rle(mask: np.ndarray) -> List[int]:
    """
    Convert binary mask to CVAT RLE format.
    
    CVAT RLE format: [rle_counts..., xtl, ytl, xbr, ybr]
    
    Args:
        mask: np.ndarray [H, W] boolean or uint8 binary mask
    
    Returns:
        List of integers: RLE counts followed by bounding box
    """
    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        # Empty mask
        return [0, 0, 0, 0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Convert to CVAT format (xtl, ytl, xbr, ybr)
    xtl, ytl = int(x_min), int(y_min)
    xbr, ybr = int(x_max) + 1, int(y_max) + 1
    
    # Crop mask to bounding box
    cropped = mask[ytl:ybr, xtl:xbr]
    
    # Flatten in row-major (C) order
    flat = cropped.flatten().astype(np.uint8)
    
    # Encode as RLE (run-length encoding)
    # CVAT format: starts with count of 0s, then alternates
    rle = []
    if len(flat) == 0:
        return [0, 0, 0, 0]
    
    current_val = 0  # Start counting 0s first
    count = 0
    
    for val in flat:
        if val == current_val:
            count += 1
        else:
            rle.append(count)
            count = 1
            current_val = val
    rle.append(count)
    
    # Append bounding box
    rle.extend([xtl, ytl, xbr, ybr])
    
    return rle


def handler(context, event):
    """
    Handle text-to-segment requests.
    
    Expected input (from CVAT detector runner):
    {
        "image": "<base64 encoded image>",
        "text_prompts": ["a person", "a car"],
        "threshold": 0.3,  # Optional confidence threshold
    }
    
    Returns (CVAT detector format - flat array):
    [
        {
            "type": "mask",
            "label": "a person",
            "mask": [rle_counts..., xtl, ytl, xbr, ybr],
            "attributes": []
        },
        ...
    ]
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
        confidence_threshold = data.get("threshold", data.get("confidence_threshold", 0.3))

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
        results = pcs_handler.text_to_segment(text_prompts)

        # Format response for CVAT (flat array of DetectedShape)
        response = []
        for r in results:
            detection = {
                'type': 'mask',
                'label': r['label'],
                'mask': mask_to_cvat_rle(r['mask']),
                'attributes': [],
            }
            response.append(detection)

        context.logger.info(f"SAM3-PCS returning {len(response)} detections")
        
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

    if isinstance(result, list):
        # New CVAT detector format (flat array)
        print(f"Found {len(result)} detections in {elapsed:.2f}s")
        for d in result[:5]:
            mask_data = d['mask']
            rle_counts = mask_data[:-4]
            bbox = mask_data[-4:]  # [xtl, ytl, xbr, ybr]
            print(f"  - {d['label']}: box={bbox}, rle_count_length={len(rle_counts)}")
    elif 'error' in result:
        print(f"Error: {result.get('error')}")
    else:
        print(f"Unexpected response format: {result}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_handler()
