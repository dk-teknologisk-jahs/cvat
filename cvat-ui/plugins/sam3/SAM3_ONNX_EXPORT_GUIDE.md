# SAM3 ONNX Export Guide: Matching SAM2 Functionality

This document describes how to export SAM3-Tracker ONNX models that match SAM2's browser-side decoder functionality, including mask refinement and in-model interpolation.

---

## Key Findings from Official SAM3 Implementation

### 1. Dynamic Mask Selection (`_dynamic_multimask_via_stability`)

The official SAM3 decoder outputs **3 masks** in multimask mode:
- **Mask 0**: "Single object" token - designed for non-ambiguous prompts (multiple points)
- **Masks 1-2**: "Multi-object" tokens - designed for ambiguous prompts (single click)

**Selection logic** (from `sam3/sam/mask_decoder.py`):

```python
# For SINGLE prompt (ambiguous): select best IoU from multi-mask outputs (masks 1-2)
# For MULTIPLE prompts (non-ambiguous): use mask 0 if stable, else best multi-mask

stability = area(logits > delta) / area(logits > -delta)  # delta=0.05
if stability >= 0.98:  # Official threshold from model_builder.py
    use_mask_0()
else:
    use_best_multimask()  # max IoU from masks 1-2
```

**Browser implementation** must replicate this logic to avoid holes and disconnected regions.

### 2. Point Coordinate Shift (+0.5 for Pixel Center)

From `sam3/sam/prompt_encoder.py`:
```python
def _embed_points(self, points, labels, pad):
    points = points + 0.5  # Shift to center of pixel
    ...
```

**Why**: SAM3 treats pixel coordinates as referring to the **top-left corner** of each pixel. Adding 0.5 shifts them to the **center**, which gives better positional encoding.

**⚠️ NOTE**: The +0.5 shift is done **inside the prompt encoder**, not in the calling code. Since our ONNX decoder (`tracker-prompt-encoder-mask-decoder.onnx`) includes the prompt encoder, the browser code should pass **raw pixel coordinates** (scaled to 1008x1008 space). The model handles the +0.5 shift internally.

### 3. Low-Res Mask Clamping for Refinement

From `sam3/model/sam1_task_predictor.py`:
```python
low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
```

The low-resolution mask logits are **clamped to [-32, 32]** before being stored/returned for mask refinement. This prevents extreme values from dominating subsequent predictions.

✅ **FIXED**: Our browser implementation now clamps `lowResMaskData` to [-32, 32] before caching.

### 4. `no_mem_embed` Addition to Features

From `sam3/model/sam1_task_predictor.py`:
```python
# Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
```

The `no_mem_embed` tensor is added to the **lowest-resolution feature map** (72×72 for SAM3). This is a learned embedding that tells the model "there's no memory from previous frames" (image-only mode, not video tracking).

**In ONNX export**: This should be baked into the model or handled as a constant addition.

### 5. Mask Refinement is Critical for Multi-Click

From `sam3/model/sam1_task_predictor.py`:
- `mask_input`: Previous low-res mask logits `[B, 1, 288, 288]`
- `has_mask_input`: Whether mask_input is valid

**Why it matters**: Without mask refinement, each click generates **independent predictions** that may conflict. With mask refinement, new clicks **refine** the previous mask, producing coherent results.

The CVAT SAM2 plugin properly uses this:
```typescript
const isLowResMaskRelevant = clicks.slice(0, -1) === lastClicks;
// Pass previous low_res_mask if adding to existing annotation
```

### 3. Hole Filling Post-Processing

From `sam3/model/utils/sam1_utils.py` (`SAM2Transforms`):
```python
mask_threshold = 0.0
max_hole_area = 256.0    # Fill holes up to 256 pixels
max_sprinkle_area = 0.0  # Remove small disconnected regions
```

This uses **connected component analysis** to:
1. Find small holes (background regions) inside the mask
2. Fill holes ≤ `max_hole_area` pixels
3. Optionally remove small "sprinkles" (noise)

**Note**: This requires CUDA-accelerated connected components (`sam3.perflib`), which is not available in browser. Alternative: morphological operations in JavaScript.

---

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

---

## Browser-Side Workarounds (Current Implementation)

Until a SAM2-compatible ONNX decoder is available, the CVAT SAM3 plugin implements these workarounds in JavaScript:

### 1. Dynamic Mask Selection (Matching SAM3's `_dynamic_multimask_via_stability`)

```typescript
// From inference.worker.ts
const STABILITY_DELTA = 0.05;
const STABILITY_THRESH = 0.95;

// Compute stability for each mask
const stability = areaInner / areaUnion;  // (logits > delta) / (logits > -delta)

if (totalPrompts === 1) {
    // SINGLE prompt (ambiguous) → best IoU from masks 1-2
    bestIdx = argmax(iou[1:]);
} else {
    // MULTIPLE prompts (non-ambiguous) → mask 0 if stable
    if (stability[0] >= STABILITY_THRESH) {
        bestIdx = 0;
    } else {
        bestIdx = argmax(iou[1:]);  // Fall back to best multi-mask
    }
}
```

**Why this matters**: Without proper mask selection, multi-click scenarios produce holes and disconnected patches because masks 1-2 are optimized for ambiguous single-click cases.

### 2. JavaScript Bilinear Interpolation (align_corners=False)

```typescript
// Coordinate mapping matching PyTorch F.interpolate(align_corners=False)
const srcX = (dstX + 0.5) * scaleX - 0.5;
const srcY = (dstY + 0.5) * scaleY - 0.5;

// Bilinear sampling with proper edge clamping
function bilinearSample(data, w, h, x, y) {
    const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)));
    const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)));
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    // ... weighted average of 4 neighbors
}
```

**Key insight**: Interpolate **logits** (not probabilities), then threshold at 0. This produces smoother edges.

### 3. Bounding Box Optimization

Computing bbox on low-res mask first, then only interpolating within that region:

```typescript
// Step 1: Find bbox on 288×288 mask (fast - 82K pixels)
for (y, x in lowResMask) {
    if (logit > 0) updateBbox(x, y);
}

// Step 2: Map to high-res coordinates with padding
dstMinX = floor((srcMinX - padding + 0.5) / scale - 0.5);
// ...

// Step 3: Only interpolate within cropped region (~5% of pixels)
for (y in croppedH) {
    for (x in croppedW) {
        // Bilinear interpolate only this region
    }
}
```

This reduces interpolation from 12M pixels (3024×4032) to ~100K pixels.

### 4. Mask Refinement (When Decoder Supports It)

```typescript
// Check if we should use mask refinement
const isLowResMaskRelevant =
    JSON.stringify(clicks.slice(0, -1)) === JSON.stringify(lastClicks);

const useMaskRefinement =
    supportsMaskInput &&
    lowResMaskCache.has(key) &&
    isLowResMaskRelevant;

// Pass previous mask to decoder
const inputs = {
    mask_input: useMaskRefinement ? lowResMaskCache.get(key) : zeros,
    has_mask_input: useMaskRefinement ? 1.0 : 0.0,
};
```

**Note**: The current usls-exported decoder does NOT support mask_input. This requires re-exporting with the custom wrapper script.

### 5. Future: Hole Filling (Not Yet Implemented)

Browser-side alternative to SAM3's connected component hole filling:

```typescript
// Simple morphological closing (dilation then erosion)
function fillSmallHoles(mask, maxHoleSize) {
    // 1. Dilate mask (expand foreground)
    const dilated = dilate(mask, kernelSize);
    // 2. Erode back (shrink foreground, but holes are filled)
    const closed = erode(dilated, kernelSize);
    return closed;
}
```

Or use WebGL/WebGPU for accelerated connected component labeling.

---

## Comparison: Reference Implementations

| Feature | Official SAM3 (Python) | usls (Rust) | CVAT SAM3 (Browser) |
|---------|----------------------|-------------|---------------------|
| Mask selection | `_dynamic_multimask_via_stability` | Max IoU only | Dynamic stability ✅ |
| Stability threshold | 0.98 | N/A | 0.98 ✅ |
| Stability delta | 0.05 | N/A | 0.05 ✅ |
| Point coord +0.5 shift | In prompt encoder | In prompt encoder | In ONNX decoder ✅ |
| Low-res mask clamping | `clamp(-32, 32)` | None | `clamp(-32, 32)` ✅ |
| `no_mem_embed` addition | ✅ (to 72×72 feat) | In encoder | In encoder ✅ |
| Hole filling | `max_hole_area=256` | None | `max_hole_area=256` ✅ |
| Mask refinement | `mask_input` / `has_mask_input` | None | Partial ✅ |
| Interpolation | `F.interpolate(align_corners=False)` | `interpolate_1d_u8` | JS bilinear ✅ |
| Threshold | Logits > 0 | Probs > 0.5 | Logits > 0 ✅ |

---

## TODO: Known Browser Implementation Gaps

1. ~~**Point coordinate +0.5 shift**: Add `+ 0.5` to point coordinates before scaling (matches pixel center)~~ ✅ Handled by ONNX decoder (prompt encoder is included)
2. ~~**Stability threshold**: Consider changing from 0.95 to 0.98 to match official~~ ✅ Fixed
3. ~~**Low-res mask clamping**: Clamp to [-32, 32] before caching for refinement~~ ✅ Fixed
4. ~~**Hole filling**: Implement morphological closing or connected component analysis~~ ✅ Implemented with union-find connected components, matching `max_hole_area=256`

All major implementation gaps have been addressed!
