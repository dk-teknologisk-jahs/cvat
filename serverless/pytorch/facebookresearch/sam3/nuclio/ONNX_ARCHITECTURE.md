# SAM3 ONNX Implementation Architecture

This document describes the ONNX model architecture for SAM3 (Segment Anything 3) and compares our implementation with the official SAM3 capabilities.

## Overview

SAM3 is a unified foundation model for promptable segmentation that combines:
- **PCS (Promptable Class Segmentation)**: Text-based detection and segmentation (DETR-style)
- **PVS (Promptable Visual Segmentation)**: Point/box-based segmentation with video tracking (SAM2-style)

Both modes share a common vision encoder (Hiera), enabling efficient multi-modal prompting.

## ONNX Model Architecture

Our implementation exports SAM3 into **7 separate ONNX models** that work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SAM3 ONNX Architecture                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │    Image     │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      1. VISION ENCODER         │
                    │   (vision_encoder_sam3.onnx)   │
                    │                                │
                    │   Input: image [1,3,1024,1024] │
                    │   Output: features [1,256,64,64]│
                    │           high_res [1,32,256,256]│
                    │           feat_s0  [1,32,256,256]│
                    │           feat_s1  [1,64,128,128]│
                    └────────────────┬───────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
    ┌────────────────────┐  ┌───────────────┐  ┌────────────────────┐
    │   PVS Mode (SAM2)  │  │  PCS Mode     │  │  Video Tracking    │
    │   Point/Box Prompt │  │  Text Prompt  │  │  Memory System     │
    └─────────┬──────────┘  └───────┬───────┘  └─────────┬──────────┘
              │                     │                    │
              ▼                     ▼                    ▼
┌─────────────────────────┐ ┌─────────────────┐ ┌───────────────────────┐
│  2. TRACKER DECODER     │ │ 3. TEXT ENCODER │ │ 5. MEMORY ATTENTION   │
│ (tracker_decoder_sam3)  │ │(text_encoder_sam3)│ │(memory_attention_sam3)│
│                         │ │                 │ │                       │
│ Inputs:                 │ │ Input: text     │ │ Inputs:               │
│  - image_embed          │ │ Output:         │ │  - current_vision     │
│  - high_res_feats       │ │  - text_features│ │  - pos_encoding       │
│  - point_coords         │ │  - text_pe      │ │  - memory_0..15       │
│  - point_labels         │ │                 │ │  - memory_pos_0..15   │
│                         │ └────────┬────────┘ │ Output:               │
│ Outputs:                │          │          │  - conditioned_feats  │
│  - masks [1,3,1024,1024]│          ▼          └───────────┬───────────┘
│  - scores [1,3]         │ ┌─────────────────┐             │
│  - obj_ptr [1,256]      │ │ 4. PCS DECODER  │             ▼
└─────────────────────────┘ │(pcs_decoder_sam3)│ ┌───────────────────────┐
                            │                 │ │ 6. MEMORY ENCODER     │
                            │ Inputs:         │ │(memory_encoder_sam3)  │
                            │  - image_embed  │ │                       │
                            │  - text_features│ │ Inputs:               │
                            │  - text_pe      │ │  - pix_feat           │
                            │                 │ │  - masks              │
                            │ Outputs:        │ │  - object_score_logits│
                            │  - masks        │ │ Output:               │
                            │  - scores       │ │  - maskmem_features   │
                            │  - boxes        │ │  - maskmem_pos_enc    │
                            └─────────────────┘ │  - temporal_code      │
                                                └───────────┬───────────┘
                                                            │
                                                            ▼
                                                ┌───────────────────────┐
                                                │ 7. OBJECT POINTER     │
                                                │(object_pointer_sam3)  │
                                                │                       │
                                                │ Inputs:               │
                                                │  - mask_embed [1,256] │
                                                │                       │
                                                │ Output:               │
                                                │  - obj_ptr [1,256]    │
                                                └───────────────────────┘
```

## Model Details

### 1. Vision Encoder (`vision_encoder_sam3.onnx`)

**Purpose**: Extracts hierarchical visual features from input images.

| Property | Value |
|----------|-------|
| Architecture | Hiera (Hierarchical Vision Transformer) |
| Input Shape | `[1, 3, 1024, 1024]` (RGB image) |
| Output: features | `[1, 256, 64, 64]` (main features for decoder) |
| Output: high_res_feats | `[1, 32, 256, 256]` (high-resolution for refinement) |
| Output: feat_s0 | `[1, 32, 256, 256]` (stage 0 features) |
| Output: feat_s1 | `[1, 64, 128, 128]` (stage 1 features) |
| File Size | ~323 MB |

**Source**: Exported from `Sam3TrackerModel.shared_image_embedding` and `vision_encoder`.

### 2. Tracker Decoder (`tracker_decoder_sam3.onnx`)

**Purpose**: Generates segmentation masks from point/box prompts (PVS mode).

| Property | Value |
|----------|-------|
| Inputs | image_embed, high_res_feats, point_coords, point_labels |
| Output: masks | `[1, 3, 1024, 1024]` (3 candidate masks) |
| Output: iou_scores | `[1, 3]` (quality scores per mask) |
| Output: obj_ptr | `[1, 256]` (object pointer for tracking) |
| File Size | ~58 MB |

**Source**: Exported from `Sam3TrackerModel.prompt_encoder` and `mask_decoder`.

**Notes**:
- Returns 3 mask candidates with IoU scores for ambiguity resolution
- Object pointer (`obj_ptr`) is used for video propagation

### 3. Text Encoder (`text_encoder_sam3.onnx`)

**Purpose**: Encodes text prompts for PCS mode.

| Property | Value |
|----------|-------|
| Architecture | BERT-base (from Grounding DINO) |
| Input | Tokenized text (input_ids, attention_mask, token_type_ids) |
| Output: text_features | `[1, seq_len, 256]` |
| Output: text_pe | `[1, seq_len, 256]` (positional encoding) |
| File Size | ~109 MB |

**Source**: Exported from `Sam3Model.text_encoder`.

### 4. PCS Decoder (`pcs_decoder_sam3.onnx`)

**Purpose**: Detects and segments all instances matching text description.

| Property | Value |
|----------|-------|
| Architecture | DETR-style decoder with cross-attention |
| Inputs | image_embed, text_features, text_pe |
| Output: masks | `[1, N, H, W]` (instance masks) |
| Output: scores | `[1, N]` (detection confidence) |
| Output: boxes | `[1, N, 4]` (bounding boxes) |
| File Size | ~91 MB |

**Source**: Exported from `Sam3Model.detector`.

**Notes**:
- Detects **all instances** of the prompted class
- Returns bounding boxes alongside masks

### 5. Memory Attention (`memory_attention_sam3.onnx`)

**Purpose**: Conditions current frame features on temporal memory for video tracking.

| Property | Value |
|----------|-------|
| Inputs | current_vision, pos_encoding, memory_0..15, memory_pos_0..15 |
| Output | conditioned_feats `[1, 4096, 256]` |
| Memory Slots | Up to 16 memory frames |
| File Size | ~17 MB |

**Source**: Manually loaded from safetensors weights (not exposed by HuggingFace API).

### 6. Memory Encoder (`memory_encoder_sam3.onnx`)

**Purpose**: Encodes mask predictions into memory for future frames.

| Property | Value |
|----------|-------|
| Inputs | pix_feat, masks, object_score_logits |
| Output: maskmem_features | Memory features |
| Output: maskmem_pos_enc | Memory positional encoding |
| Output: temporal_code | Temporal embedding |
| File Size | ~5 MB |

**Source**: Manually loaded from safetensors weights.

### 7. Object Pointer (`object_pointer_sam3.onnx`)

**Purpose**: Projects mask embeddings to object pointers for identity tracking.

| Property | Value |
|----------|-------|
| Input | mask_embed `[1, 256]` |
| Output | obj_ptr `[1, 256]` |
| File Size | ~0.5 MB |

**Source**: Manually loaded from safetensors weights.

## Data Flow

### PVS Image Mode (Point/Box → Mask)

```
Image → Vision Encoder → image_embed, high_res_feats
                              ↓
Point/Box Prompts → Tracker Decoder → masks, scores, obj_ptr
```

### PCS Image Mode (Text → All Instances)

```
Image → Vision Encoder → image_embed
                              ↓
Text → Text Encoder → text_features, text_pe
                              ↓
                        PCS Decoder → masks, scores, boxes
```

### PVS Video Mode (Tracking)

```
Frame 0:
  Image → Vision Encoder → features
  Point Prompt → Tracker Decoder → mask, obj_ptr
  mask → Memory Encoder → memory_features

Frame N:
  Image → Vision Encoder → features
  features + memories → Memory Attention → conditioned_features
  conditioned_features → Tracker Decoder → mask, obj_ptr
  mask → Memory Encoder → update memories
```

### PCS Video Mode (Text-Track) ✅ NEW

```
Frame 0 (text-track-init):
  Image → Vision Encoder → image_embed
  Text → Text Encoder → text_features
  image_embed + text_features → PCS Decoder → masks[], boxes[], scores[]
  For each detected object:
    box → Tracker Decoder → initial mask, obj_ptr
    mask → Memory Encoder → memory_features

  Return: session_id, tracked_objects[], states[]

Frame N (track/frame):
  Image → Vision Encoder → features
  For each tracked object:
    features + memories → Memory Attention → conditioned_features
    conditioned_features → Tracker Decoder → mask, obj_ptr
    mask → Memory Encoder → update memories

  Return: updated tracked_objects[]
```

---

## Feature Comparison with Official SAM3

### Implemented Features ✅

| Feature | Mode | Description | Status |
|---------|------|-------------|--------|
| **Point Prompts** | PVS | Click to segment single object | ✅ Full |
| **Box Prompts** | PVS | Draw box to segment object | ✅ Full |
| **Text Prompts** | PCS | Text description → all matching instances | ✅ Full |
| **Video Object Tracking** | PVS | Track prompted object across frames | ✅ Server-side |
| **Multi-object Tracking** | PVS | Track multiple objects simultaneously | ✅ Server-side |
| **Memory-based Propagation** | PVS | Temporal consistency via memory bank | ✅ Full |

### Missing Features ❌

| Feature | Mode | Description | Priority | Implementation Notes |
|---------|------|-------------|----------|---------------------|
| **Image Exemplar Prompts** | Both | "Segment objects like this example" | 🟡 Medium | Requires exemplar encoder (exists in SAM3 but not exported) |
| **Automatic Mask Generation** | - | Generate all possible masks without prompts | 🟢 Low | Already have SAM2 AMG; similar approach possible |
| **Batched Inference** | Both | Process multiple images in single forward pass | 🟢 Low | ONNX models use batch=1; would need re-export |
| **Streaming Video Inference** | PVS | @torch.inference_mode() streaming | ✅ N/A | Implemented server-side in Python |

### Recently Implemented Features ✅

| Feature | Mode | Description | Implementation |
|---------|------|-------------|----------------|
| **PCS Video Mode** | PCS | Text prompt + track all instances in video | ✅ `text-track-init` mode in main.py |

### Feature Details

#### PCS Video Mode (✅ IMPLEMENTED)

**Official SAM3 Capability**:
```python
# Text prompt tracks ALL instances across video
with torch.inference_mode():
    state = predictor.init_state_with_pcs(video_path, text="player")
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks for ALL detected players in each frame
```

**Implementation**: The `text-track-init` mode in `main.py` provides this functionality:

```python
# API Request to initialize text-based tracking
POST /function/sam3-detector
{
    "mode": "text-track-init",
    "image": "<base64 encoded first frame>",
    "text_prompts": ["person", "car"],
    "threshold": 0.3
}

# Response
{
    "session_id": "sam3_track_xxxxx",
    "shapes": [...],  # Detected objects with masks
    "states": [...],  # Tracking states for each object
    "num_objects_detected": 5
}

# Then use track/frame for subsequent frames
POST /function/sam3-detector
{
    "mode": "track/frame",
    "image": "<base64 encoded frame N>",
    "states": [...]  # From previous response
}
```

**How it works**:
1. `init_tracking_from_text()` in `model_handler.py` runs PCS detection on frame 0
2. Converts all detections to tracking objects with unique IDs
3. Initializes memory state for each detected instance
4. `track_frame()` propagates all objects using memory-based tracking

#### Image Exemplar Prompts (Medium Priority)

**Official SAM3 Capability**:
```python
# Provide example mask, segment similar objects
predictor.set_image(image)
masks = predictor.predict(
    exemplar_image=example_img,
    exemplar_mask=example_mask
)
```

**Current Gap**: Exemplar encoder not exported to ONNX.

**Implementation Path**:
1. Identify exemplar encoder in SAM3 model
2. Export to ONNX with appropriate wrapper
3. Modify server-side inference to accept exemplar inputs

---

## Recommended Next Steps

### Phase 1: PCS Video Mode (✅ COMPLETED)

**Goal**: Enable text-based video annotation (e.g., "track all cars")

**Implementation**:
1. ✅ Added `handle_text_track_init()` function in `main.py`
2. ✅ Leverages existing `init_tracking_from_text()` in `model_handler.py`
3. ✅ Uses existing memory components for frame propagation

**Testing**:
```bash
# Activate the conda environment
conda activate grimme-tf2.18

# Run smoke tests (quick validation)
cd /home/jahs/GitHub/cvat/serverless/onnx/facebookresearch/sam3/detector/nuclio
python test_smoke.py --model-dir ./onnx-exports

# Run specific text-track-init tests
python test_onnx_unified.py --model-dir ./onnx-exports --test-text-track-init

# Run Video PCS tests
python test_onnx_unified.py --model-dir ./onnx-exports --test-video-pcs

# Run PyTorch comparison (requires HuggingFace auth)
python test_onnx_unified.py --model-dir ./onnx-exports --test-video-pcs-vs-pytorch

# Run pytest suite
pytest test_onnx_pytest.py -v -k "TextTrackInit"
```

**Usage**:
```bash
# Initialize tracking with text prompts
curl -X POST http://localhost:8080/function/sam3-detector \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "text-track-init",
    "image": "'$(base64 -w0 frame0.jpg)'",
    "text_prompts": ["person", "car"],
    "threshold": 0.3
  }'

# Track subsequent frames
curl -X POST http://localhost:8080/function/sam3-detector \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "track/frame",
    "image": "'$(base64 -w0 frame1.jpg)'",
    "states": [...]  # from init response
  }'
```

### Phase 2: Image Exemplar Support (Medium Impact)

**Goal**: "Segment objects that look like this example"

1. **Model investigation**:
   - Locate exemplar encoder in SAM3 codebase
   - Determine if it's part of existing modules or separate

2. **ONNX export**:
   - Add exemplar encoder wrapper to `export_onnx.py`
   - Export with appropriate input/output specs

3. **Server-side changes**:
   - Add exemplar image preprocessing
   - Modify inference to accept exemplar inputs

4. **Testing** (after implementation):
   ```bash
   conda activate grimme-tf2.18
   python test_onnx_unified.py --model-dir ./onnx-exports --test-exemplar
   pytest test_onnx_pytest.py -v -k "Exemplar"
   ```

5. **Estimated effort**: 3-5 days

### Phase 3: Automatic Mask Generation (Low Priority)

**Goal**: Generate all possible masks without user prompts

1. **Approach**: Adapt SAM2 AMG implementation
   - Grid-based point sampling
   - Run tracker decoder on each point
   - NMS and filtering

2. **Consider**: May not be needed if SAM2 AMG already serves this purpose

3. **Testing** (after implementation):
   ```bash
   conda activate grimme-tf2.18
   python test_onnx_unified.py --model-dir ./onnx-exports --test-amg
   pytest test_onnx_pytest.py -v -k "AMG"
   ```

4. **Estimated effort**: 1-2 days (if needed)

---

## File Organization

```
nuclio/
├── export_onnx.py              # Main export: vision, tracker, text, PCS
├── export_memory_components.py # Memory exports (requires manual weight loading)
├── test_onnx.py               # Unified test runner
├── main.py                    # Nuclio serverless function
├── onnx-exports/              # Exported ONNX models
│   ├── vision_encoder_sam3.onnx
│   ├── tracker_decoder_sam3.onnx
│   ├── text_encoder_sam3.onnx
│   └── pcs_decoder_sam3.onnx
├── onnx-memory-exports/       # Memory component ONNX models
│   ├── memory_attention_sam3.onnx
│   ├── memory_encoder_sam3.onnx
│   └── object_pointer_sam3.onnx
└── ONNX_ARCHITECTURE.md       # This document
```

---

## Development Environment Setup

### Conda Environment

All development and testing should use the `grimme-tf2.18` conda environment:

```bash
# Activate the environment
conda activate grimme-tf2.18

# Verify Python and key packages
python --version
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Running Tests

```bash
# Always activate the environment first
conda activate grimme-tf2.18

# Navigate to the test directory
cd /home/jahs/GitHub/cvat/serverless/onnx/facebookresearch/sam3/detector/nuclio

# Quick smoke tests
python test_smoke.py --model-dir /opt/nuclio/sam3/models

# Comprehensive tests
python test_onnx_unified.py --model-dir /opt/nuclio/sam3/models --all

# Specific test categories
python test_onnx_unified.py --model-dir /opt/nuclio/sam3/models --test-text-track-init
python test_onnx_unified.py --model-dir /opt/nuclio/sam3/models --test-video-pcs
python test_onnx_unified.py --model-dir /opt/nuclio/sam3/models --test-tracking

# Pytest (for CI/CD integration)
pytest test_onnx_pytest.py -v
pytest test_onnx_pytest.py -v -k "TextTrackInit"  # Specific tests
```

### Environment Variables

```bash
export SAM3_MODEL_DIR=/opt/nuclio/sam3/models
export SAM3_DEVICE=cuda  # or 'cpu'
export HF_TOKEN=your_huggingface_token  # For PyTorch comparison tests
```

---

## Technical Notes

### Why Two Export Scripts?

The memory components (`memory_attention`, `memory_encoder`, `obj_ptr_proj`) are not exposed by the HuggingFace `Sam3TrackerModel` API. They exist in the safetensors weights under the `tracker_model.*` prefix but must be loaded manually:

```python
# HuggingFace model only exposes:
model = Sam3TrackerModel.from_pretrained(...)
model.shared_image_embedding  # ✅
model.vision_encoder          # ✅
model.prompt_encoder          # ✅
model.mask_decoder            # ✅
model.memory_attention        # ❌ AttributeError
model.memory_encoder          # ❌ AttributeError
```

Therefore, `export_memory_components.py` loads weights directly from the safetensors file and reconstructs the memory modules.

### Model Compatibility

- **Framework**: ONNX Runtime 1.16+
- **Opset**: 17
- **Dynamic Axes**: Supported for variable sequence lengths (text encoder)
- **Precision**: FP32 (FP16 export possible but not tested)

---

## References

- [SAM3 Paper](https://ai.meta.com/research/publications/sam-3/)
- [Official PyTorch Implementation](https://github.com/facebookresearch/sam3)
- [HuggingFace Model](https://huggingface.co/facebook/sam3)
- [SAM2 Architecture](https://github.com/facebookresearch/sam2) (basis for PVS mode)
