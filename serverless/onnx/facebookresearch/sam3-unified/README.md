# SAM3 Unified ONNX Handler

A unified ONNX-based inference handler for Segment Anything 3 (SAM3), supporting text-prompted segmentation (PCS), video tracking, and automatic mask generation.

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Text-to-Segment (PCS)** | Detect objects using text prompts | ✅ |
| **Box Prompts** | Positive/negative box constraints for PCS | ✅ |
| **Video Tracking** | Track objects across video frames | ✅ |
| **Video PCS** | Text-prompted video tracking | ✅ |
| **Automatic Mask Generation** | Generate masks for all objects in image | ✅ |
| **Batched Inference** | Process multiple images in one forward pass | ✅ |
| **Semantic Segmentation** | Binary and labeled semantic mask output | ✅ |
| **Multi-object Tracking** | Track multiple objects simultaneously | ✅ |

## ONNX Models Required

The handler requires 4 ONNX models exported from SAM3:

| Model | Description | Input | Output |
|-------|-------------|-------|--------|
| `vision_encoder.onnx` | Image feature extraction | `[B, 3, 1008, 1008]` | FPN features (4 tensors) |
| `text_encoder.onnx` | CLIP text encoding | `input_ids`, `attention_mask` | `text_features`, `text_mask` |
| `pcs_decoder.onnx` | Promptable Cascade Segmentation | FPN + text + boxes | masks, boxes, logits |
| `tracker_decoder.onnx` | Point-prompted segmentation | FPN + points + mask | masks, IoU scores |

### Exporting Models

Use the export script at `export_hf_onnx.py`:

```bash
python export_hf_onnx.py --output-dir /path/to/onnx-models
```

## API Reference

### UnifiedModelHandler

```python
from model_handler import UnifiedModelHandler

handler = UnifiedModelHandler(
    model_dir="/path/to/onnx-models",
    device="cuda"  # or "cpu"
)
```

### Text-to-Segment (PCS)

Detect objects using text prompts:

```python
detections = handler.text_to_segment(
    image,                    # PIL.Image or numpy array
    text_prompts=["person", "car"],
    threshold=0.3,            # Detection confidence threshold
    box_prompts=None          # Optional: [(x1, y1, x2, y2, label), ...]
)

# Returns list of dicts:
# [{"mask": np.array, "box": [x1,y1,x2,y2], "score": float, "label": str}, ...]
```

### Box Prompts for PCS

Use box prompts to guide detection:

```python
# Positive box (label=1): focus detection in this region
# Negative box (label=0): exclude this region
detections = handler.text_to_segment(
    image,
    text_prompts=["object"],
    box_prompts=[
        (100, 100, 300, 300, 1),  # Positive box
        (400, 400, 500, 500, 0),  # Negative box
    ]
)
```

### Video Tracking

Track objects across video frames:

```python
# Initialize tracking on first frame
result = handler.init_tracking(
    frame0,
    objects=[
        {"box": [x1, y1, x2, y2], "label": "object1"},
        {"box": [x1, y1, x2, y2], "label": "object2"},
    ]
)
session_id = result["session_id"]

# Track on subsequent frames
for frame in frames[1:]:
    result = handler.track_frame(frame, session_id)
    masks = result["masks"]      # List of mask arrays
    boxes = result["boxes"]      # List of [x1,y1,x2,y2]
    scores = result["scores"]    # List of confidence scores

# Clean up
handler.clear_tracking(session_id)
```

### Video PCS (Text-Prompted Tracking)

Combine text detection with video tracking:

```python
# Detect with text on frame 0, then track
result = handler.init_tracking_from_text(
    frame0,
    text_prompts=["person"],
    threshold=0.5
)
session_id = result["session_id"]

# Track detected objects on subsequent frames
for frame in frames[1:]:
    result = handler.track_frame(frame, session_id)
```

### Automatic Mask Generation (AMG)

Generate masks for all objects in an image:

```python
masks = handler.automatic_mask_generation(
    image,
    points_per_side=32,       # Grid density
    pred_iou_thresh=0.88,     # IoU threshold
    stability_score_thresh=0.95,
    box_nms_thresh=0.7
)

# Returns list of dicts:
# [{"segmentation": np.array, "bbox": [x,y,w,h], "area": int,
#   "predicted_iou": float, "stability_score": float}, ...]
```

### Semantic Segmentation

Get binary or labeled semantic masks:

```python
# Binary mask (union of all instances)
binary_mask = handler.get_semantic_mask(detections, image_size=(H, W))

# Labeled mask (each instance has unique ID)
labeled_mask = handler.get_labeled_semantic_mask(detections, image_size=(H, W))
```

### Batched Encoding

Encode multiple images efficiently:

```python
embeddings_list = handler.encode_batch([img1, img2, img3])
# Returns list of embedding dicts, one per image
```

### Model Info

Query handler capabilities:

```python
info = handler.get_model_info()
# {
#     "vision_encoder": True,
#     "text_encoder": True,
#     "pcs_decoder": True,
#     "tracker_decoder": True,
#     "model_dir": "/path/to/models",
#     "device": "cuda",
#     "capabilities": [
#         "encode", "encode_batch", "text_to_segment",
#         "text_to_segment_with_boxes", "get_semantic_mask",
#         "get_labeled_semantic_mask", "track", "init_tracking",
#         "track_frame", "automatic_mask_generation",
#         "init_tracking_from_text"
#     ],
#     "features": {
#         "box_prompts": True,
#         "negative_prompts": True,
#         "batched_inference": True,
#         "automatic_mask_generation": True,
#         "semantic_segmentation": True,
#         "video_pcs": True
#     }
# }
```

## Numerical Accuracy

ONNX outputs have been verified against PyTorch with the following MAE (Mean Absolute Error):

| Output | Shape | MAE |
|--------|-------|-----|
| **Vision Encoder** | | |
| fpn_feat_0 | [1, 256, 288, 288] | 1.4e-7 |
| fpn_feat_1 | [1, 256, 144, 144] | 8.3e-7 |
| fpn_feat_2 | [1, 256, 72, 72] | 1.0e-6 |
| fpn_pos_2 | [1, 256, 72, 72] | 0.0 |
| **Tracker Decoder** | | |
| masks | [1, 3, 1008, 1008] | 6.0e-5 |
| iou_predictions | [1, 3] | 1.3e-7 |
| low_res_masks | [1, 3, 288, 288] | 1.0e-5 |
| object_score_logits | [1, 1] | 1.9e-6 |
| **Text Encoder** | | |
| text_features | [B, 32, 256] | 3.5e-7 |
| text_mask | [B, 32] | 0.0 |

All MAEs are well below the 0.001 threshold.

## Testing

Run the comprehensive test suite:

```bash
# All tests on CPU
python test_onnx_unified.py --model-dir /path/to/models --device cpu --all

# Specific tests
python test_onnx_unified.py --model-dir /path/to/models \
    --test-vision-encoder \
    --test-text-encoder \
    --test-handler \
    --test-box-prompts \
    --test-amg
```

### Test Coverage

- `test_vision_encoder` - Vision encoder ONNX vs PyTorch
- `test_tracker_decoder` - Tracker decoder ONNX vs PyTorch
- `test_text_encoder` - Text encoder ONNX vs PyTorch
- `test_end_to_end` - Full pipeline validation
- `test_handler` - UnifiedModelHandler API
- `test_text_to_segment` - PCS text-to-segment
- `test_tracking` - Video tracking
- `test_multi_object` - Multi-object tracking
- `test_edge_cases` - Edge cases and error handling
- `test_box_prompts` - Box prompts for PCS
- `test_batched` - Batched image encoding
- `test_amg` - Automatic mask generation
- `test_semantic` - Semantic segmentation output
- `test_video_pcs` - Video PCS mode

## Dependencies

```
onnxruntime>=1.16.0  # or onnxruntime-gpu for CUDA
numpy
pillow
transformers  # for CLIP tokenizer
```

## License

This code is part of CVAT and follows its licensing terms.
SAM3 model weights are subject to Meta's license agreement.
