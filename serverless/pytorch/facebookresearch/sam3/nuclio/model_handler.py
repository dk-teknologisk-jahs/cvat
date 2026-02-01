# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3-Tracker Model Handler

Handles the vision encoder for SAM3-Tracker using ONNX Runtime.
Returns embeddings that can be decoded in the browser.

Uses the onnx-community/sam3-tracker-ONNX encoder:
- Input: pixel_values [batch, 3, 1008, 1008] FLOAT32
- Outputs:
  - image_embeddings.0: [batch, 32, 288, 288] FLOAT32 (high_res_feats_0)
  - image_embeddings.1: [batch, 64, 144, 144] FLOAT32 (high_res_feats_1)
  - image_embeddings.2: [batch, 256, 72, 72] FLOAT32 (image_embed)

Verified to match PyTorch output:
- MAE < 0.0005 for all outputs
- Correlation > 0.9999
- End-to-end mask IoU > 0.998
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image


class ModelHandler:
    """SAM3-Tracker Vision Encoder using ONNX Runtime."""

    def __init__(self):
        model_path = os.environ.get(
            "SAM3_ENCODER_PATH",
            "/opt/nuclio/sam3/vision_encoder.onnx"
        )

        # Configure ONNX Runtime for GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Image preprocessing params (same as PyTorch handler)
        self.image_size = 1008
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        print(f"SAM3 ONNX encoder loaded: {model_path}")
        print(f"  Input: {self.input_name}")
        print(f"  Outputs: {self.output_names}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for SAM3-Tracker encoder.

        Args:
            image: PIL Image (any size)

        Returns:
            Preprocessed tensor [1, 3, 1008, 1008]
        """
        # Resize to 1008x1008 (exact fit)
        image_resized = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        # Convert to numpy and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Normalize with mean/std
        img_array = (img_array - self.mean) / self.std

        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def encode(self, image: Image.Image) -> dict:
        """
        Encode image to SAM3 embeddings.

        Args:
            image: PIL Image

        Returns:
            Dictionary with embeddings (high_res_feats_0, high_res_feats_1, image_embed)
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Run encoder
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Map ONNX output names to standard names
        # onnx-community encoder outputs: image_embeddings.0, .1, .2
        return {
            'high_res_feats_0': outputs[0],  # [1, 32, 288, 288]
            'high_res_feats_1': outputs[1],  # [1, 64, 144, 144]
            'image_embed': outputs[2],       # [1, 256, 72, 72]
        }

    def handle(self, image: Image.Image) -> tuple:
        """
        Handle image encoding (compatible interface with main.py).

        Args:
            image: PIL Image

        Returns:
            Tuple of (high_res_feats_0, high_res_feats_1, image_embed) as numpy arrays
        """
        embeddings = self.encode(image)
        return (
            embeddings['high_res_feats_0'],  # [1, 32, 288, 288]
            embeddings['high_res_feats_1'],  # [1, 64, 144, 144]
            embeddings['image_embed'],       # [1, 256, 72, 72]
        )
