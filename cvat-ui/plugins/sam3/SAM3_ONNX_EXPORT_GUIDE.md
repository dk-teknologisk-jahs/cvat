# SAM3 ONNX Export Guide: Matching SAM2 Functionality

This document describes how to export SAM3-Tracker ONNX models that match SAM2's browser-side decoder functionality, including mask refinement and in-model interpolation.

## Current State Comparison

### SAM2 Decoder ONNX (Working Reference)

The SAM2 decoder in `cvat-ui/plugins/sam2/assets/sam2.1_hiera_large.decoder.onnx` has these characteristics:

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `image_embed` | `[1, 256, 64, 64]` | Main backbone embedding |
| `high_res_feats_0` | `[1, 32, 256, 256]` | Level 0 high-res features |
| `high_res_feats_1` | `[1, 64, 128, 128]` | Level 1 high-res features |
| `point_coords` | `[num_labels, num_points, 2]` | Click/box coordinates |
| `point_labels` | `[num_labels, num_points]` | Point type labels |
| `orig_im_size` | `[2]` | **Original image [H, W]** |
| `mask_input` | `[num_labels, 1, 256, 256]` | Previous mask for refinement |
| `has_mask_input` | `[num_labels]` | Whether mask_input is valid |

**Outputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `masks` | `[1, 1, orig_H, orig_W]` | **Upsampled to original size** |
| `iou_predictions` | `[1, 1]` | IoU confidence score |
| `low_res_masks` | `[1, 1, 256, 256]` | For next iteration refinement |
| `xtl, ytl, xbr, ybr` | `[]` (scalar) | Bounding box coordinates |

**Key features:**
1. ✅ Takes `orig_im_size` as input
2. ✅ Outputs masks at original image resolution (dynamic shape)
3. ✅ Computes bounding box internally
4. ✅ Supports mask refinement via `mask_input` / `has_mask_input`
5. ✅ Uses `F.interpolate(..., mode="bilinear", align_corners=False)` internally

### SAM3 Decoder ONNX (usls version - Current)

The usls decoder in `cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder.onnx`:

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `input_points` | `[batch, 1, num_points, 2]` | Point coordinates |
| `input_labels` | `[batch, 1, num_points]` | Point labels |
| `input_boxes` | `[batch, num_boxes, 4]` | Box coordinates |
| `image_embeddings.0` | `[batch, 32, 288, 288]` | High-res features level 0 |
| `image_embeddings.1` | `[batch, 64, 144, 144]` | High-res features level 1 |
| `image_embeddings.2` | `[batch, 256, 72, 72]` | Main backbone embedding |

**Outputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `iou_scores` | `[batch, num_prompts, 3]` | IoU for 3 mask candidates |
| `pred_masks` | `[batch, num_prompts, 3, 288, 288]` | **Fixed 288×288 resolution** |
| `object_score_logits` | `[batch, num_prompts, 1]` | Object presence score |

**Missing features:**
1. ❌ No `orig_im_size` input
2. ❌ Fixed 288×288 output (requires external upsampling)
3. ❌ No bounding box computation
4. ❌ No mask refinement support (`mask_input` / `has_mask_input`)

---

## Architecture Differences: SAM2 vs SAM3

| Parameter | SAM2 | SAM3-Tracker |
|-----------|------|--------------|
| Input image size | 1024×1024 | 1008×1008 |
| Backbone stride | 16 | 14 |
| Embedding size | 64×64 (1024/16) | 72×72 (1008/14) |
| Low-res mask size | 256×256 (64×4) | 288×288 (72×4) |
| High-res feat levels | 2 | 3 |
| High-res feat 0 | 256×256 | 288×288 |
| High-res feat 1 | 128×128 | 144×144 |
| Always pred_obj_scores | No | Yes |

---

## Required Export Changes

### 1. Add `orig_im_size` Input for Dynamic Upsampling

The key missing piece is dynamic output resolution. The decoder wrapper must:

```python
class SAM3TrackerDecoderWrapper(nn.Module):
    def __init__(self, sam_model, multimask_output=True):
        super().__init__()
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder
        self.no_mem_embed = sam_model.no_mem_embed
        self.multimask_output = multimask_output
        # SAM3 constants
        self.IMAGE_SIZE = 1008
        self.EMBED_SIZE = 72  # 1008 / 14
        self.MASK_SIZE = 288  # 72 * 4

    def forward(
        self,
        image_embed: torch.Tensor,      # [B, 256, 72, 72]
        high_res_feats_0: torch.Tensor, # [B, 32, 288, 288]
        high_res_feats_1: torch.Tensor, # [B, 64, 144, 144]
        point_coords: torch.Tensor,     # [B, N, 2]
        point_labels: torch.Tensor,     # [B, N]
        mask_input: torch.Tensor,       # [B, 1, 288, 288]
        has_mask_input: torch.Tensor,   # [B] or [1]
        orig_im_size: torch.Tensor,     # [2] - NEW!
    ):
        # ... encode prompts and run decoder ...

        # Get low-res masks from decoder (288×288)
        low_res_masks, iou_pred, _, object_score_logits = self.sam_mask_decoder(...)

        # CRITICAL: Upsample to original image size
        # This must happen INSIDE the ONNX model for quality
        orig_h = orig_im_size[0].long()
        orig_w = orig_im_size[1].long()

        # Use F.interpolate with the exact same settings as official SAM3
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(orig_h, orig_w),  # Dynamic size from input
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks, iou_pred, low_res_masks, object_score_logits
```

### 2. ONNX Dynamic Shape Export

The challenge is that ONNX requires explicit handling of dynamic shapes:

```python
def export_decoder(model, output_path):
    wrapper = SAM3TrackerDecoderWrapper(model, multimask_output=True)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    num_points = 2

    dummy_inputs = {
        'image_embed': torch.randn(batch_size, 256, 72, 72),
        'high_res_feats_0': torch.randn(batch_size, 32, 288, 288),
        'high_res_feats_1': torch.randn(batch_size, 64, 144, 144),
        'point_coords': torch.randn(batch_size, num_points, 2),
        'point_labels': torch.ones(batch_size, num_points),
        'mask_input': torch.zeros(batch_size, 1, 288, 288),
        'has_mask_input': torch.tensor([0.0]),
        'orig_im_size': torch.tensor([1080.0, 1920.0]),  # Example H, W
    }

    # Define dynamic axes for ONNX
    dynamic_axes = {
        'point_coords': {1: 'num_points'},
        'point_labels': {1: 'num_points'},
        'orig_im_size': {},  # Fixed [2] shape
        # Output masks have dynamic H, W based on orig_im_size
        'masks': {2: 'orig_height', 3: 'orig_width'},
    }

    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs.values()),
        output_path,
        input_names=list(dummy_inputs.keys()),
        output_names=['masks', 'iou_predictions', 'low_res_masks', 'object_score_logits'],
        dynamic_axes=dynamic_axes,
        opset_version=17,  # Need recent opset for Resize with dynamic sizes
        do_constant_folding=True,
    )
```

### 3. Add Bounding Box Computation

SAM2's decoder computes bounding boxes internally. Add this to the wrapper:

```python
def compute_bbox(self, mask: torch.Tensor, threshold: float = 0.0):
    """
    Compute bounding box from mask logits.

    Args:
        mask: [B, C, H, W] mask logits
        threshold: Logit threshold (0.0 = sigmoid > 0.5)

    Returns:
        xtl, ytl, xbr, ybr: Bounding box coordinates
    """
    # Get best mask (highest IoU)
    binary_mask = (mask > threshold).float()

    # Find non-zero coordinates
    # Sum along batch and channel dims to get 2D mask
    mask_2d = binary_mask.sum(dim=(0, 1)) > 0  # [H, W]

    # Find bounding box
    rows = mask_2d.any(dim=1)  # [H]
    cols = mask_2d.any(dim=0)  # [W]

    if rows.any() and cols.any():
        ytl = rows.float().argmax()
        ybr = (rows.shape[0] - 1) - rows.flip(0).float().argmax()
        xtl = cols.float().argmax()
        xbr = (cols.shape[0] - 1) - cols.flip(0).float().argmax()
    else:
        # No positive pixels - return full image
        ytl = torch.tensor(0.0)
        xtl = torch.tensor(0.0)
        ybr = torch.tensor(mask.shape[2] - 1.0)
        xbr = torch.tensor(mask.shape[3] - 1.0)

    return xtl, ytl, xbr, ybr
```

### 4. Mask Refinement Support

The mask refinement inputs are already partially supported in the existing export script. The key requirements:

```python
# In the forward method:
def forward(self, ..., mask_input, has_mask_input, ...):
    # Handle mask input for refinement
    if has_mask_input.item() > 0.5:
        # mask_input is [B, 1, 288, 288] - same as SAM3's mask_in_chans expects
        dense_embeddings = self.sam_prompt_encoder.mask_downscaling(mask_input)
    else:
        # No mask input - use learned "no mask" embedding
        dense_embeddings = self.sam_prompt_encoder.no_mask_embed.weight
        dense_embeddings = dense_embeddings.reshape(1, -1, 1, 1)
        dense_embeddings = dense_embeddings.expand(
            batch_size, -1, self.EMBED_SIZE, self.EMBED_SIZE
        )
```

---

## Complete Export Script

Here's the full export script that would create a SAM2-compatible SAM3 decoder:

```python
#!/usr/bin/env python3
"""
Export SAM3-Tracker decoder with SAM2-compatible interface.

This creates an ONNX model with:
- Dynamic output resolution via orig_im_size input
- Mask refinement support
- Bounding box computation
- In-model bilinear upsampling
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3DecoderSAM2Compatible(nn.Module):
    """
    SAM3-Tracker decoder wrapper with SAM2-compatible interface.
    """

    def __init__(self, sam_model, multimask_output=False):
        super().__init__()
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder
        self.no_mem_embed = sam_model.no_mem_embed
        self.multimask_output = multimask_output

        # SAM3 constants
        self.IMAGE_SIZE = 1008
        self.EMBED_SIZE = 72
        self.MASK_SIZE = 288

    def forward(
        self,
        image_embed: torch.Tensor,      # [1, 256, 72, 72]
        high_res_feats_0: torch.Tensor, # [1, 32, 288, 288]
        high_res_feats_1: torch.Tensor, # [1, 64, 144, 144]
        point_coords: torch.Tensor,     # [num_labels, num_points, 2]
        point_labels: torch.Tensor,     # [num_labels, num_points]
        orig_im_size: torch.Tensor,     # [2] for [H, W]
        mask_input: torch.Tensor,       # [num_labels, 1, 288, 288]
        has_mask_input: torch.Tensor,   # [num_labels]
    ):
        batch_size = point_coords.shape[0]

        # Add no_mem_embed to image embedding (required by SAM3)
        image_embed = image_embed + self.no_mem_embed

        # Get positional encoding
        image_pe = self.sam_prompt_encoder.get_dense_pe()

        # Encode point prompts
        # Points are already in model coordinate space (scaled to 1008x1008)
        sparse_embeddings, _ = self.sam_prompt_encoder(
            points=(point_coords, point_labels.int()),
            boxes=None,
            masks=None,
        )

        # Handle mask input for refinement
        if has_mask_input.sum() > 0:
            # Use provided mask input
            dense_embeddings = self.sam_prompt_encoder.mask_downscaling(mask_input)
        else:
            # No mask - use learned embedding
            dense_embeddings = self.sam_prompt_encoder.no_mask_embed.weight
            dense_embeddings = dense_embeddings.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(
                batch_size, -1, self.EMBED_SIZE, self.EMBED_SIZE
            )

        # Prepare high-res features
        high_res_features = [high_res_feats_0, high_res_feats_1]

        # Run mask decoder
        low_res_masks, iou_pred, _, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Select output masks based on mode
        if self.multimask_output:
            # Return best mask based on IoU score
            best_idx = iou_pred.argmax(dim=1, keepdim=True)  # [B, 1]
            # Gather best mask: [B, 1, H, W]
            low_res_mask = torch.gather(
                low_res_masks, 1,
                best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 288, 288)
            )
            iou_predictions = torch.gather(iou_pred, 1, best_idx)
        else:
            low_res_mask = low_res_masks[:, 0:1, :, :]
            iou_predictions = iou_pred[:, 0:1]

        # CRITICAL: Upsample to original image size using bilinear interpolation
        # This matches: F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        orig_h = orig_im_size[0].long().item()
        orig_w = orig_im_size[1].long().item()

        masks = F.interpolate(
            low_res_mask.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        # Compute bounding box from upsampled mask
        binary_mask = (masks > 0).float().squeeze(0).squeeze(0)  # [H, W]

        rows = binary_mask.any(dim=1)  # [H]
        cols = binary_mask.any(dim=0)  # [W]

        if rows.any():
            ytl = rows.float().argmax()
            ybr = rows.shape[0] - 1 - rows.flip(0).float().argmax()
        else:
            ytl = torch.tensor(0.0)
            ybr = torch.tensor(float(orig_h - 1))

        if cols.any():
            xtl = cols.float().argmax()
            xbr = cols.shape[0] - 1 - cols.flip(0).float().argmax()
        else:
            xtl = torch.tensor(0.0)
            xbr = torch.tensor(float(orig_w - 1))

        return (
            masks,              # [1, 1, orig_H, orig_W] - upsampled logits
            iou_predictions,    # [1, 1] - IoU score
            low_res_mask,       # [1, 1, 288, 288] - for next iteration
            xtl, ytl, xbr, ybr, # Bounding box scalars
        )


def export_sam3_decoder_sam2_compatible(
    checkpoint_path: str,
    output_path: str,
    model_cfg: str = "sam3_hiera_l",
):
    """Export SAM3 decoder with SAM2-compatible interface."""

    # Load SAM3 model
    from sam3 import model_builder

    sam3_model = model_builder.build_sam3_tracker(
        model_cfg,
        checkpoint=checkpoint_path,
        device="cpu",
    )
    sam3_model.eval()

    # Create wrapper
    wrapper = SAM3DecoderSAM2Compatible(sam3_model, multimask_output=True)
    wrapper.eval()

    # Dummy inputs for tracing
    dummy_inputs = (
        torch.randn(1, 256, 72, 72),       # image_embed
        torch.randn(1, 32, 288, 288),      # high_res_feats_0
        torch.randn(1, 64, 144, 144),      # high_res_feats_1
        torch.randn(1, 2, 2),              # point_coords
        torch.ones(1, 2),                  # point_labels
        torch.tensor([1080.0, 1920.0]),    # orig_im_size [H, W]
        torch.zeros(1, 1, 288, 288),       # mask_input
        torch.tensor([0.0]),               # has_mask_input
    )

    input_names = [
        'image_embed',
        'high_res_feats_0',
        'high_res_feats_1',
        'point_coords',
        'point_labels',
        'orig_im_size',
        'mask_input',
        'has_mask_input',
    ]

    output_names = [
        'masks',
        'iou_predictions',
        'low_res_masks',
        'xtl', 'ytl', 'xbr', 'ybr',
    ]

    dynamic_axes = {
        'point_coords': {0: 'num_labels', 1: 'num_points'},
        'point_labels': {0: 'num_labels', 1: 'num_points'},
        'mask_input': {0: 'num_labels'},
        'has_mask_input': {0: 'num_labels'},
        # Dynamic output size based on orig_im_size
        'masks': {0: 'num_labels', 2: 'orig_height', 3: 'orig_width'},
        'low_res_masks': {0: 'num_labels'},
        'iou_predictions': {0: 'num_labels'},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

    print(f"Exported SAM3 decoder to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--model-cfg", default="sam3_hiera_l", help="Model config name")
    args = parser.parse_args()

    export_sam3_decoder_sam2_compatible(
        args.checkpoint,
        args.output,
        args.model_cfg,
    )
```

---

## ONNX Opset Requirements

For dynamic `F.interpolate` with runtime-determined output sizes, you need:

- **Opset 11+**: Basic Resize operator support
- **Opset 13+**: Better Resize with coordinate transformation modes
- **Opset 17+**: Recommended for best compatibility with `align_corners=False`

The Resize operator in ONNX uses `coordinate_transformation_mode`:
- `"half_pixel"` = PyTorch's `align_corners=False` ✅
- `"align_corners"` = PyTorch's `align_corners=True`

---

## Vision Encoder Export

The vision encoder export is simpler since it has fixed output shapes:

```python
def export_vision_encoder(sam3_model, output_path):
    """Export SAM3 vision encoder."""

    class VisionEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.image_encoder
            self.neck = model.neck  # If separate

        def forward(self, pixel_values):
            # pixel_values: [B, 3, 1008, 1008] normalized
            features = self.backbone(pixel_values)
            # Returns high_res_feats and image_embed
            return (
                features['high_res_feats'][0],  # [B, 32, 288, 288]
                features['high_res_feats'][1],  # [B, 64, 144, 144]
                features['image_embed'],        # [B, 256, 72, 72]
            )

    wrapper = VisionEncoderWrapper(sam3_model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 1008, 1008)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['pixel_values'],
        output_names=[
            'image_embeddings_0',  # high_res level 0
            'image_embeddings_1',  # high_res level 1
            'image_embeddings_2',  # main embed
        ],
        dynamic_axes={'pixel_values': {0: 'batch_size'}},
        opset_version=17,
    )
```

---

## Testing the Export

After export, verify the model matches expected behavior:

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("sam3_decoder_sam2_compatible.onnx")

# Print inputs/outputs
print("Inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")

print("Outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Test inference
outputs = session.run(
    None,
    {
        'image_embed': np.random.randn(1, 256, 72, 72).astype(np.float32),
        'high_res_feats_0': np.random.randn(1, 32, 288, 288).astype(np.float32),
        'high_res_feats_1': np.random.randn(1, 64, 144, 144).astype(np.float32),
        'point_coords': np.array([[[500, 500], [600, 600]]], dtype=np.float32),
        'point_labels': np.array([[1, 1]], dtype=np.float32),
        'orig_im_size': np.array([1080, 1920], dtype=np.float32),
        'mask_input': np.zeros((1, 1, 288, 288), dtype=np.float32),
        'has_mask_input': np.array([0], dtype=np.float32),
    }
)

masks, iou, low_res, xtl, ytl, xbr, ybr = outputs
print(f"Output mask shape: {masks.shape}")  # Should be [1, 1, 1080, 1920]
print(f"Bounding box: ({xtl}, {ytl}) - ({xbr}, {ybr})")
```

---

## Summary of Required Changes

| Feature | Current (usls) | Required (SAM2-compatible) |
|---------|----------------|---------------------------|
| `orig_im_size` input | ❌ | ✅ Add as `[2]` tensor |
| Dynamic output shape | ❌ (fixed 288×288) | ✅ `[B, 1, orig_H, orig_W]` |
| In-model upsampling | ❌ | ✅ `F.interpolate(..., align_corners=False)` |
| Bounding box outputs | ❌ | ✅ `xtl, ytl, xbr, ybr` scalars |
| Mask refinement | ❌ | ✅ `mask_input` + `has_mask_input` |
| ONNX opset | 11 | 17 (for Resize with half_pixel) |

With these changes, the SAM3 decoder would have identical interface to SAM2, enabling drop-in replacement in CVAT's browser-side inference.
