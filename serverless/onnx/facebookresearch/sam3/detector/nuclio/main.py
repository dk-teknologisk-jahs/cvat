#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Unified SAM3 ONNX Nuclio Handler

Single function that handles all three AI tool types using ONNX Runtime:
1. Interactor mode (point/box prompts → embeddings for browser decoding)
2. Detector mode (text prompts → complete masks via PCS)
3. Tracker mode (video object tracking)
4. Text-Track mode (Video PCS: text prompts → detect + track all instances)

NO HUGGINGFACE AUTH NEEDED AT RUNTIME!
All models are ONNX files baked into the Docker image.

Routes requests based on the 'mode' parameter:
- mode='encode' or no mode → Interactor (returns embeddings)
- mode='text-to-segment' → Detector (returns masks)
- mode='track/init' → Tracker initialization (from box prompts)
- mode='text-track-init' → Text-based tracker init (Video PCS mode)
- mode='track/frame' → Tracker frame processing
- mode='track/clear' → Tracker session cleanup
- mode='info' → Model information
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from model_handler import get_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """
    Convert binary mask to CVAT RLE format.

    CVAT RLE format: [count1, count2, ...] where counts alternate
    between background and foreground, starting with background.
    """
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(flat, prepend=flat[0])
    change_indices = np.where(changes != 0)[0]

    if len(change_indices) == 0:
        if flat[0] == 0:
            return [len(flat)]
        else:
            return [0, len(flat)]

    rle = []
    prev_idx = 0

    if flat[0] == 1:
        rle.append(0)

    for idx in change_indices:
        rle.append(idx - prev_idx)
        prev_idx = idx

    rle.append(len(flat) - prev_idx)
    return rle


def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handle_encode(model, image: Image.Image) -> Dict[str, Any]:
    """
    Handle interactor mode - return embeddings for browser-side decoding.

    Returns embeddings with names expected by the SAM3 UI plugin:
    - high_res_feats_0: [1, 256, 288, 288] (fpn_feat_0)
    - high_res_feats_1: [1, 256, 144, 144] (fpn_feat_1)
    - image_embed: [1, 256, 72, 72] (fpn_feat_2)
    """
    embeddings = model.encode(image)

    # Helper to encode numpy array to base64
    def encode_array(arr):
        arr = np.ascontiguousarray(arr).astype(np.float32)
        return base64.b64encode(arr.tobytes()).decode("ascii")

    # Map internal names to names expected by UI plugin
    # The UI plugin expects: high_res_feats_0, high_res_feats_1, image_embed
    emb0 = embeddings.get("fpn_feat_0")
    emb1 = embeddings.get("fpn_feat_1")
    emb2 = embeddings.get("fpn_feat_2")

    return {
        "high_res_feats_0": encode_array(emb0),
        "high_res_feats_1": encode_array(emb1),
        "image_embed": encode_array(emb2),
        "high_res_feats_0_shape": list(emb0.shape),
        "high_res_feats_1_shape": list(emb1.shape),
        "image_embed_shape": list(emb2.shape),
    }


def handle_text_to_segment(model, data: Dict, image: Image.Image) -> List[Dict]:
    """Handle detector mode - return complete masks for text prompts."""
    text_prompts = data.get("text_prompts", [])
    threshold = data.get("threshold", 0.3)

    detections = model.text_to_segment(
        text_prompts=text_prompts,
        image=image,
        confidence_threshold=threshold,
    )

    # Format response for CVAT detector format
    results = []
    for det in detections:
        mask = det.get("mask")
        box = det["box"]

        if mask is not None:
            rle = mask_to_rle(mask > 0)
            points = rle + [box[0], box[1], box[2], box[3]]
            results.append({
                "type": "mask",
                "label": det["label"],
                "points": points,
                "score": det["score"],
            })
        else:
            # No mask, just return box
            results.append({
                "type": "rectangle",
                "label": det["label"],
                "points": box,
                "score": det["score"],
            })

    return results


def handle_track_init(model, data: Dict, image: Image.Image) -> Dict[str, Any]:
    """Handle tracker initialization.
    
    CVAT sends shapes as flat coordinate arrays: [[x1, y1, x2, y2], ...]
    Each shape corresponds to a bounding box for tracking.
    """
    shapes = data.get("shapes", [])
    states = data.get("states", [])

    if not shapes:
        return {"error": "No shapes provided for tracking initialization"}

    # Convert CVAT shapes to SAM3 format
    # CVAT format: shapes is a list of coordinate arrays [[x1, y1, x2, y2], ...]
    objects = []
    for idx, shape in enumerate(shapes):
        # shape is a flat list: [x1, y1, x2, y2]
        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            x1, y1, x2, y2 = shape[0], shape[1], shape[2], shape[3]
            objects.append({
                "object_id": idx,
                "box": [x1, y1, x2, y2],
                "label": "object",
            })
        elif isinstance(shape, dict):
            # Legacy format with points key
            points = shape.get("points", [])
            if len(points) >= 4:
                x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
                objects.append({
                    "object_id": shape.get("clientID", idx),
                    "box": [x1, y1, x2, y2],
                    "label": shape.get("label", "object"),
                })

    result = model.init_tracking(image=image, objects=objects)

    if "error" in result:
        return result

    # Format for CVAT tracker response
    # CVAT expects shapes as flat coordinate arrays: [[x1, y1, x2, y2], ...]
    session_id = result["session_id"]
    response_shapes = []
    response_states = []

    for obj_result in result.get("tracked_objects", []):
        box = obj_result.get("box", [0, 0, 100, 100])
        # Return flat coordinate array, not object with points
        response_shapes.append(box)
        response_states.append({
            "session_id": session_id,
            "object_id": obj_result["object_id"],
        })

    return {"shapes": response_shapes, "states": response_states}


def handle_track_frame(model, data: Dict, image: Image.Image) -> Dict[str, Any]:
    """Handle tracking a new frame."""
    states = data.get("states", [])

    if not states:
        return {"error": "No tracking states provided"}

    # Get session ID from states
    session_id = None
    object_ids = []
    for state in states:
        if isinstance(state, dict):
            session_id = state.get("session_id")
            object_ids.append(state.get("object_id"))
        elif isinstance(state, str):
            try:
                parsed = json.loads(state)
                session_id = parsed.get("session_id")
                object_ids.append(parsed.get("object_id"))
            except:
                pass

    if not session_id:
        return {"error": "Could not find session_id in states"}

    frame_idx = data.get("frame_idx", 0)

    result = model.track_frame(
        session_id=session_id,
        image=image,
        frame_idx=frame_idx,
    )

    if "error" in result:
        return result

    # Format response - CVAT expects shapes as flat coordinate arrays
    response_shapes = []
    response_states = []

    for i, obj_result in enumerate(result.get("tracked_objects", [])):
        box = obj_result.get("box", [0, 0, 100, 100])
        # Return flat coordinate array, not object with points
        response_shapes.append(box)
        response_states.append({
            "session_id": session_id,
            "object_id": object_ids[i] if i < len(object_ids) else i,
            "frame_idx": result["frame_idx"],
        })

    return {"shapes": response_shapes, "states": response_states}


def handle_track_clear(model, data: Dict) -> Dict[str, Any]:
    """Handle clearing a tracking session."""
    session_id = data.get("session_id")

    if not session_id:
        states = data.get("states", [])
        for state in states:
            if isinstance(state, dict):
                session_id = state.get("session_id")
                break

    if not session_id:
        return {"error": "No session_id provided"}

    return model.clear_tracking(session_id)


def handle_text_track_init(model, data: Dict, image: Image.Image) -> Dict[str, Any]:
    """
    Handle Video PCS mode - text prompts → detect + track all instances.

    This implements the Phase 1 feature from ONNX_ARCHITECTURE.md:
    1. Run PCS detection on frame 0 to get all instances matching text prompts
    2. Initialize tracking state for each detected instance
    3. Use memory system to propagate all instances in subsequent frames

    Request format:
    {
        "mode": "text-track-init",
        "image": "<base64 encoded first frame>",
        "text_prompts": ["person", "car"],  # Objects to detect and track
        "threshold": 0.3  # Optional confidence threshold
    }

    Response format:
    {
        "session_id": "sam3_track_xxxxx",
        "frame_idx": 0,
        "tracked_objects": [
            {"object_id": 0, "box": [x1,y1,x2,y2], "mask": <rle>, "score": 0.95, "label": "person"},
            {"object_id": 1, "box": [x1,y1,x2,y2], "mask": <rle>, "score": 0.87, "label": "car"},
            ...
        ],
        "text_prompts": ["person", "car"]
    }
    """
    text_prompts = data.get("text_prompts", [])
    threshold = data.get("threshold", 0.3)

    if not text_prompts:
        return {"error": "No text_prompts provided for text-track-init"}

    # Use the model's init_tracking_from_text method
    result = model.init_tracking_from_text(
        image=image,
        text_prompts=text_prompts,
        confidence_threshold=threshold,
    )

    if "error" in result:
        return result

    # Format response for CVAT tracker format
    session_id = result.get("session_id")
    response_shapes = []
    response_states = []

    for obj_result in result.get("tracked_objects", []):
        obj_id = obj_result.get("object_id", len(response_shapes))
        box = obj_result.get("box", [0, 0, 100, 100])
        mask = obj_result.get("mask")
        score = obj_result.get("score", 1.0)

        # Find the label from the original detection if available
        label = "object"
        detections = result.get("detections", [])
        if obj_id < len(detections):
            label = detections[obj_id].get("label", "object")

        shape = {
            "type": "rectangle",
            "points": box,
            "clientID": obj_id,
            "label": label,
            "score": score,
        }

        # Add mask as RLE if available
        if mask is not None:
            rle = mask_to_rle(mask > 0)
            shape["mask_rle"] = rle
            shape["type"] = "mask"
            shape["points"] = rle + box  # CVAT expects RLE + bbox

        response_shapes.append(shape)
        response_states.append({
            "session_id": session_id,
            "object_id": obj_id,
            "label": label,
        })

    return {
        "shapes": response_shapes,
        "states": response_states,
        "session_id": session_id,
        "text_prompts": text_prompts,
        "num_objects_detected": len(response_shapes),
    }


def init_context(context):
    """Initialize the model handler."""
    context.logger.info("Initializing SAM3 ONNX Unified handler...")
    model = get_handler()
    context.user_data.model = model
    info = model.get_model_info()
    context.logger.info(f"SAM3 ONNX handler initialized: {info}")


def handler(context, event):
    """Unified SAM3 ONNX handler for Nuclio."""
    data = event.body
    if isinstance(data, bytes):
        data = json.loads(data.decode("utf-8"))

    model = context.user_data.model

    # Auto-detect mode from request content:
    # - If text_prompts present → text-to-segment (detector)
    # - If text_prompts + track context → text-track-init (video PCS)
    # - If pos_points/neg_points present → encode (interactor)
    # - If states present → tracker
    # - Explicit mode parameter takes precedence
    mode = data.get("mode")
    if not mode:
        if data.get("text_prompts"):
            # Check if this is a video PCS request (text + tracking context)
            if data.get("video_mode") or data.get("init_tracking"):
                mode = "text-track-init"
            else:
                mode = "text-to-segment"
        elif data.get("states") and data.get("image"):
            mode = "track/frame"
        elif data.get("shapes") and data.get("image"):
            mode = "track/init"
        else:
            mode = "encode"  # Default for interactor

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
            status = 200 if "error" not in result else 400
            return context.Response(
                body=json.dumps(result),
                headers={},
                content_type="application/json",
                status_code=status,
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

        # Route to handler
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
        elif mode == "text-track-init":
            result = handle_text_track_init(model, data, image)
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

        status = 200 if "error" not in result else 400
        return context.Response(
            body=json.dumps(result),
            headers={},
            content_type="application/json",
            status_code=status,
        )

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
