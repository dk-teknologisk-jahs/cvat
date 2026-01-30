# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3-Tracker Nuclio Function Handler

Returns image embeddings that can be decoded by the browser-side ONNX decoder.
This follows the same pattern as SAM2 - server encodes, browser decodes.

The decoder runs in the browser using tracker-prompt-encoder-mask-decoder.onnx
"""

import base64
import io
import json

import numpy as np
from PIL import Image
from model_handler import ModelHandler


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
        # Names match ONNX decoder input names: image_embeddings.0, .1, .2
        response = {
            'image_embeddings_0': encode_array(emb0),
            'image_embeddings_1': encode_array(emb1),
            'image_embeddings_2': encode_array(emb2),
            # Include shapes for reconstruction
            'image_embeddings_0_shape': list(emb0.shape),
            'image_embeddings_1_shape': list(emb1.shape),
            'image_embeddings_2_shape': list(emb2.shape),
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
