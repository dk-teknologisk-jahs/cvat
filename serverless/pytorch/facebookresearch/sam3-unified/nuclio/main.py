#!/usr/bin/env python3
"""
Unified SAM3 Nuclio Handler

Single function that handles all three AI tool types:
1. Interactor mode (point/box prompts → embeddings for browser decoding)
2. Detector mode (text prompts → complete masks via PCS)
3. Tracker mode (video object tracking via SAM3 Video)

Routes requests based on the 'mode' parameter:
- mode='encode' or no mode → Interactor (returns embeddings)
- mode='text-to-segment' → Detector (returns masks)
- mode='track/init' → Tracker initialization
- mode='track/frame' → Tracker frame processing
- mode='track/clear' → Tracker session cleanup
- mode='info' → Model information

This avoids loading SAM3 models multiple times, saving significant GPU VRAM.
"""

import base64
import io
import json
import logging
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

from model_handler_unified import get_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handle_encode(model, image: Image.Image) -> Dict[str, Any]:
    """
    Handle interactor mode - return embeddings for browser-side decoding.
    """
    embeddings = model.encode(image)

    # Encode embeddings as base64
    encoded = {}
    shapes = {}
    for name, arr in embeddings.items():
        encoded[name] = base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')
        shapes[name] = list(arr.shape)

    return {
        "embeddings": encoded,
        "shapes": shapes,
        "original_size": [image.height, image.width],  # [H, W]
    }


def handle_text_to_segment(model, data: Dict, image: Image.Image) -> List[Dict]:
    """
    Handle detector mode - return complete masks for text prompts.
    """
    text_prompts = data.get("text_prompts", [])
    threshold = data.get("threshold", 0.3)

    # Run text-to-segment
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

    return results


def handle_track_init(model, data: Dict, image: Image.Image) -> Dict[str, Any]:
    """
    Handle tracker initialization - start tracking objects in video.

    CVAT tracker interface sends:
    - shapes: list of bbox annotations to track
    - states: list of previous states (empty on init)
    """
    shapes = data.get("shapes", [])

    if not shapes:
        return {"error": "No shapes provided for tracking initialization"}

    # Convert CVAT shapes to SAM3 format
    objects = []
    for shape in shapes:
        # CVAT sends points as [x1, y1, x2, y2, ...]
        points = shape.get("points", [])
        if len(points) >= 4:
            # Extract bounding box
            x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
            objects.append({
                "object_id": shape.get("clientID", len(objects)),
                "box": [x1, y1, x2, y2],
                "label": shape.get("label", "object"),
            })

    # Initialize tracking
    result = model.init_tracking(
        image=image,
        objects=objects,
    )

    if "error" in result:
        return result

    # Return CVAT tracker response format
    # CVAT expects: shapes (updated) + states (serialized state per object)
    session_id = result["session_id"]

    # Build response
    response_shapes = []
    response_states = []

    for obj_result in result.get("tracked_objects", []):
        # Each object gets its updated shape
        response_shapes.append({
            "type": "rectangle",
            "points": obj_result["bounds"],  # [x1, y1, x2, y2]
            "clientID": obj_result["object_id"],
        })

        # State just contains the session_id (actual state is in Redis)
        response_states.append({
            "session_id": session_id,
            "object_id": obj_result["object_id"],
        })

    return {
        "shapes": response_shapes,
        "states": response_states,
    }


def handle_track_frame(model, data: Dict, image: Image.Image) -> Dict[str, Any]:
    """
    Handle tracking a new frame.

    CVAT tracker sends:
    - shapes: current shapes (may be updated by user)
    - states: previous states (contains session_id)
    """
    states = data.get("states", [])

    if not states:
        return {"error": "No tracking states provided"}

    # Get session ID from first state
    session_id = None
    object_ids = []
    for state in states:
        if isinstance(state, dict):
            session_id = state.get("session_id")
            object_ids.append(state.get("object_id"))
        elif isinstance(state, str):
            # Legacy format - try to parse as JSON
            try:
                parsed = json.loads(state)
                session_id = parsed.get("session_id")
                object_ids.append(parsed.get("object_id"))
            except:
                pass

    if not session_id:
        return {"error": "Could not find session_id in states"}

    frame_idx = data.get("frame_idx", 0)

    # Track the frame
    result = model.track_frame(
        session_id=session_id,
        image=image,
        frame_idx=frame_idx,
    )

    if "error" in result:
        return result

    # Build CVAT response
    response_shapes = []
    response_states = []

    # Single object result for now
    if "bounds" in result:
        response_shapes.append({
            "type": "rectangle",
            "points": result["bounds"],
            "clientID": object_ids[0] if object_ids else 0,
        })
        response_states.append({
            "session_id": session_id,
            "object_id": object_ids[0] if object_ids else 0,
            "frame_idx": result["frame_idx"],
        })

        # Also include polygon if available
        if "polygon" in result and result["polygon"]:
            response_shapes.append({
                "type": "polygon",
                "points": result["polygon"],
                "clientID": object_ids[0] if object_ids else 0,
            })

    return {
        "shapes": response_shapes,
        "states": response_states,
    }


def handle_track_clear(model, data: Dict) -> Dict[str, Any]:
    """
    Handle clearing a tracking session.
    """
    session_id = data.get("session_id")

    if not session_id:
        # Try to get from states
        states = data.get("states", [])
        for state in states:
            if isinstance(state, dict):
                session_id = state.get("session_id")
                break

    if not session_id:
        return {"error": "No session_id provided"}

    return model.clear_tracking(session_id)


def handler(context, event):
    """
    Unified SAM3 handler for Nuclio.

    Request format depends on mode - see individual handlers for details.
    """
    # Parse request
    data = event.body
    if isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))

    # Get handler
    model = get_handler()

    # Route based on mode
    mode = data.get("mode", "encode")
    logger.info(f"Processing request with mode: {mode}")

    try:
        # Handle info request (no image needed)
        if mode == "info":
            return context.Response(
                body=json.dumps(model.get_model_info()),
                headers={},
                content_type="application/json",
                status_code=200,
            )

        # Handle track/clear (no image needed)
        if mode == "track/clear":
            result = handle_track_clear(model, data)
            return context.Response(
                body=json.dumps(result),
                headers={},
                content_type="application/json",
                status_code=200 if "error" not in result else 400,
            )

        # All other modes need an image
        image_b64 = data.get("image", "")
        if not image_b64:
            return context.Response(
                body=json.dumps({"error": "No image provided"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )

        image = decode_image(image_b64)

        # Route to appropriate handler
        if mode == "encode":
            result = handle_encode(model, image)

        elif mode == "text-to-segment":
            text_prompts = data.get("text_prompts", [])
            if not text_prompts:
                return context.Response(
                    body=json.dumps({"error": "No text_prompts provided"}),
                    headers={},
                    content_type="application/json",
                    status_code=400,
                )
            result = handle_text_to_segment(model, data, image)

        elif mode == "track/init":
            result = handle_track_init(model, data, image)

        elif mode == "track/frame":
            result = handle_track_frame(model, data, image)

        else:
            return context.Response(
                body=json.dumps({"error": f"Unknown mode: {mode}"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )

        # Return response
        status_code = 200 if "error" not in result else 400
        return context.Response(
            body=json.dumps(result),
            headers={},
            content_type="application/json",
            status_code=status_code,
        )

    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
