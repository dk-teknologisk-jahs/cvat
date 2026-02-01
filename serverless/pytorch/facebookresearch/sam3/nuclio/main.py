# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3-Tracker Nuclio Function Handler

Returns image embeddings that can be decoded by the browser-side ONNX decoder.
This follows the same pattern as SAM2 - server encodes, browser decodes.

The decoder runs in the browser using the ONNX decoder exported with mask refinement support.

Server returns three feature maps:
- high_res_feats_0: [1, 32, 288, 288] - for fine details
- high_res_feats_1: [1, 64, 144, 144] - for mid-level features  
- image_embed: [1, 256, 72, 72] - main embedding for the mask decoder

Encoder options:
- ONNX (default): Uses onnx-community/sam3-tracker-ONNX vision encoder (~1.7GB)
  Verified to match PyTorch output with <0.2% MAE and 0.9999+ correlation.
- PyTorch (fallback): Uses native PyTorch model if ONNX encoder not available.
"""

import base64
import io
import json
import os

import numpy as np
from PIL import Image

# Use ONNX encoder by default (much faster, verified to match PyTorch)
# Fall back to PyTorch if ONNX encoder not available
ENCODER_PATH = os.environ.get("SAM3_ENCODER_PATH", "/opt/nuclio/sam3/vision_encoder.onnx")

if os.path.exists(ENCODER_PATH):
    from model_handler import ModelHandler  # ONNX encoder
else:
    print("ONNX encoder not found, falling back to PyTorch encoder")
    from model_handler_pytorch import ModelHandler  # PyTorch fallback


def init_context(context):
    """Initialize the SAM3 encoder model."""
    context.logger.info("Init SAM3-Tracker context... 0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init SAM3-Tracker context... 100%")


def handler(context, event):
    """
    Handle encoding requests.

    Expected input:
    {
        "image": "<base64 encoded image>"
    }

    Returns:
    {
        "emb0": "<base64 encoded float32 array>",
        "emb1": "<base64 encoded float32 array>",
        "emb2": "<base64 encoded float32 array>"
    }
    """
    try:
        context.logger.info("SAM3-Tracker handler called")

        data = event.body

        # Decode image
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")

        # Encode image
        emb0, emb1, emb2 = context.user_data.model.handle(image)

        # Helper to encode numpy array to base64
        def encode_array(arr):
            # Ensure contiguous array
            arr = np.ascontiguousarray(arr)
            return base64.b64encode(arr.tobytes()).decode('utf-8')

        # Return embeddings as base64-encoded arrays
        # Names match ONNX decoder input names
        response = {
            'high_res_feats_0': encode_array(emb0),
            'high_res_feats_1': encode_array(emb1),
            'image_embed': encode_array(emb2),
            # Include shapes for reconstruction
            'high_res_feats_0_shape': list(emb0.shape),
            'high_res_feats_1_shape': list(emb1.shape),
            'image_embed_shape': list(emb2.shape),
        }

        return context.Response(
            body=json.dumps(response),
            headers={},
            content_type='application/json',
            status_code=200
        )

    except Exception as e:
        context.logger.error(f"Error in SAM3-Tracker handler: {str(e)}", exc_info=True)
        return context.Response(
            body=json.dumps({'error': str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
