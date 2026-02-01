# SAM3 ONNX Export Guide

This document describes SAM3 architecture, ONNX export strategies, and how SAM3 covers **all three CVAT AI tool categories**: Interactor, Detector, and Tracker.

> **Last Updated**: 1 February 2026 (All ONNX tests passing - 100% correlation with PyTorch)

---

## Table of Contents

1. [Current Progress](#current-progress)
2. [Test Coverage](#test-coverage)
3. [Implementation Plan](#implementation-plan)
4. [Reference Implementations](#reference-implementations)
5. [SAM3 Capabilities for CVAT](#sam3-capabilities-for-cvat)
6. [SAM3 Architecture Overview](#sam3-architecture-overview)
7. [Two Model Implementations](#two-model-implementations)
8. [ONNX Export Methods](#onnx-export-methods)
9. [Unified HuggingFace ONNX Export (Recommended)](#unified-huggingface-onnx-export-recommended)
10. [Current Working Implementation](#current-working-implementation)

---

## Current Progress

### Migration Status: HuggingFace ONNX Exports ✅ COMPLETE & VERIFIED

We have successfully migrated to **fully self-controlled ONNX exports** from HuggingFace Transformers. All ONNX models achieve **100% correlation** (1.0) with PyTorch reference outputs.

**⚠️ CRITICAL: The official SAM3 weights are GATED on HuggingFace!**

This means:
- ❌ Cannot use HuggingFace Transformers directly at runtime (requires auth)
- ✅ Must export to ONNX once (requires auth), then use ONNX Runtime at runtime (no auth)
- ✅ ONNX models can be baked into Docker images for deployment

#### ✅ Completed

| Component | Status | Notes |
|-----------|--------|-------|
| **Vision Encoder** | ✅ Exported & Verified | 1789 MB, outputs 256ch at all FPN levels, correlation=1.0 |
| **Tracker Decoder** | ✅ Exported & Verified | 21.4 MB, includes conv_s0/conv_s1 projections, correlation=1.0 |
| **Text Encoder** | ✅ Exported & Verified | 1.3 GB, CLIP + projection (32-token context), correlation=1.0 |
| **PCS Decoder** | ✅ Exported & Verified | 123 MB, DETR encoder/decoder + heads, full pipeline tested |
| **Export Script** | ✅ Fixed | Uses Sam3TrackerModel for PVS, Sam3Model for PCS |
| **Verification Tests** | ✅ All Passing | 6/6 tests pass with 100% correlation |
| **Unified ONNX Handler** | ✅ Implemented | `model_handler.py` with dynamic model paths |
| **Redis State Management** | ✅ Implemented | Video tracking session state in Redis |
| **Comprehensive Test Suite** | ✅ Created | `test_onnx_unified.py` with PyTorch comparison |
| **Browser Integration** | ✅ Implemented | `inference.worker.ts` supports unified decoder (256ch, 4D points) |

#### 🔄 Next Steps (Priority Order)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | **Add handler API tests** | 🔲 TODO | Test `text_to_segment()` and tracking through handler |
| 2 | **Add video tracking tests** | 🔲 TODO | Test `init_tracking()` / `track_frame()` with mock Redis |
| 3 | **Host ONNX models** | 🔲 TODO | Upload to GitHub releases for Docker builds |
| 4 | **Create Dockerfile** | 🔲 TODO | Download models or export if missing |
| 5 | **Integration testing** | 🔲 TODO | Full CVAT integration with browser UI |

---

## Test Coverage

### Current Test Results (1 Feb 2026)

All 6 core tests pass with **perfect correlation (1.0)** between ONNX and PyTorch:

```
======================================================================
Test Summary
======================================================================
  vision_encoder:            PASSED (correlation=1.0)
  tracker_decoder:           PASSED (correlation=1.0)
  text_encoder:              PASSED (correlation=1.0)
  end_to_end_encode:         PASSED (interactor pipeline)
  end_to_end_text_to_segment: PASSED (detector/PCS pipeline)
  unified_handler:           PASSED (handler module test)

All tests passed!
```

### Test Coverage Matrix

| Functionality | Direct ONNX Test | Handler API Test | Notes |
|---------------|------------------|------------------|-------|
| Vision Encoder | ✅ Tested | ✅ via encode() | PyTorch vs ONNX correlation=1.0 |
| Tracker Decoder | ✅ Tested | ✅ via encode() | PyTorch vs ONNX correlation=1.0 |
| Text Encoder | ✅ Tested | 🔲 Not tested | PyTorch vs ONNX correlation=1.0 |
| PCS Decoder | ✅ Tested | 🔲 Not tested | Full pipeline verified |
| `handler.encode()` | - | ✅ Tested | Returns 4 FPN embeddings |
| `handler.text_to_segment()` | - | 🔲 Not tested | Needs handler API test |
| `handler.init_tracking()` | - | 🔲 Not tested | Needs Redis mock |
| `handler.track_frame()` | - | 🔲 Not tested | Needs Redis mock |

### Missing Tests for 100% Coverage

1. **`handler.text_to_segment()` API test** - Test through UnifiedModelHandler
2. **Video tracking tests** - Test `init_tracking()` / `track_frame()` with mock Redis
3. **Multi-object tracking** - Test tracking multiple objects simultaneously
4. **Edge cases** - Empty images, invalid prompts, session timeout

#### ONNX Model Hosting Strategy

**Approach**: Create a GitHub repository (e.g., `cvat-ai/sam3-onnx-models`) and upload exported ONNX models as **release files**. This mirrors the approach used by [usls](https://github.com/jamjamjon/usls).

**Why GitHub Releases?**
- Free hosting for large files (up to 2 GB per file)
- No authentication required for downloads
- Version control via release tags
- Direct download URLs for Docker builds

**Docker Build Strategy**:
1. Try to download pre-exported models from GitHub releases
2. If download fails or models missing, export them using `export_hf_onnx.py`
3. This requires HuggingFace auth at build time (for gated `facebook/sam3` model)

**⚠️ HuggingFace Token Required for Export**:
The `facebook/sam3` model is **gated** - you must pass a HuggingFace token to export:
```bash
# Option 1: Environment variable
docker build --build-arg HF_TOKEN=$HF_TOKEN ...

# Option 2: Docker secret (more secure)
docker build --secret id=hf_token,env=HF_TOKEN ...
```

**Example Dockerfile snippet**:
```dockerfile
# Try to download pre-exported models, fallback to export
RUN mkdir -p /opt/nuclio/sam3/models && \
    (curl -L https://github.com/cvat-ai/sam3-onnx-models/releases/download/v1.0/vision-encoder.onnx \
         -o /opt/nuclio/sam3/models/vision-encoder.onnx || \
     python export_hf_onnx.py --vision-encoder --output-dir /opt/nuclio/sam3/models)
```

#### Exported ONNX Models (Ready for Hosting)

| Model | Size | Export Command |
|-------|------|----------------|
| `vision-encoder.onnx` | 1.79 GB | `python export_hf_onnx.py --vision-encoder` |
| `tracker-decoder.onnx` | 21.4 MB | `python export_hf_onnx.py --tracker-decoder` |
| `text-encoder.onnx` | 1.35 GB | `python export_hf_onnx.py --text-encoder` |
| `pcs-decoder.onnx` | 123 MB | `python export_hf_onnx.py --pcs-decoder` |
| **Total** | **~3.3 GB** | `python export_hf_onnx.py --all` |

**Export location**: `/tmp/sam3-onnx-pcs/` (or specify with `--output-dir`)

**Environment**: Use `conda run -n grimme-tf2.18` for export (requires PyTorch 2.6+, transformers with SAM3 support)

#### ⚠️ CRITICAL FINDING: Sam3Model vs Sam3TrackerModel

**Root Cause of Previous Issues**: The export script was using `Sam3Model` instead of `Sam3TrackerModel`. These are **DIFFERENT models with DIFFERENT weights**:

| Model | HuggingFace Class | Parameters | Purpose | Components |
|-------|-------------------|------------|---------|------------|
| `Sam3Model` | `from transformers import Sam3Model` | 1468 params | Full SAM3 for PCS (text prompts) | text_encoder, geometry_encoder, detr_encoder/decoder |
| `Sam3TrackerModel` | `from transformers import Sam3TrackerModel` | 685 params | Tracker for PVS (point/box prompts) | vision_encoder, prompt_encoder, mask_decoder |

**Export Strategy** (CORRECTED):
```python
# For vision encoder + tracker decoder (PVS/Interactor mode):
from transformers import Sam3TrackerModel
tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3")

# For text encoder + PCS decoder (PCS/Detector mode):
from transformers import Sam3Model
pcs_model = Sam3Model.from_pretrained("facebook/sam3")
```

**Weight Comparison Evidence**:
```
Sam3Model fpn_layers[0].proj1.weight mean:      0.000001
Sam3TrackerModel fpn_layers[0].proj1.weight mean: 0.000071
```

#### Key Findings

1. **Gated Models**: The `facebook/sam3` model is gated. HuggingFace Transformers cannot be used at runtime without auth. **Solution**: Export to ONNX, deploy ONNX models.

2. **⚠️ Sam3Model ≠ Sam3TrackerModel**: These load DIFFERENT WEIGHTS. Always use `Sam3TrackerModel` for tracker/interactor mode, `Sam3Model` for PCS/detector mode.

3. **Previous Issue**: External `onnx-community/sam3-tracker-ONNX` vision encoder had projections baked in (outputs 32/64/256ch), while PCS mode expects 256/256/256ch.

4. **Solution**: Single vision encoder outputs 256ch at all FPN levels. The tracker decoder includes conv_s0/conv_s1 projections internally.

5. **no_memory_embedding Fix**: Shape is `[1, 1, 256]`, must be reshaped to `[1, 256, 1, 1]` for spatial broadcast. Use `Sam3TrackerModel.no_memory_embedding` (not `no_mem_embed`).

6. **Input Shapes for HuggingFace**: Points must be 4D `[B, num_objects, num_points, 2]`, labels must be 3D `[B, num_objects, num_points]`.

7. **Multimask Output**: Mask decoder with `multimask_output=True` returns `[B, num_objects, 3, H, W]` - squeeze out `num_objects` dimension.

8. **Image Position Encoding**: Use `model.get_image_wide_positional_embeddings()` for the `image_pe` buffer (not computed dynamically).

9. **CVAT Tracker Interface**: Existing trackers (SiamMask, TransT) use stateless per-request pattern with jsonpickle-serialized state. SAM3 video tracking memory bank is too large for this approach - use Redis instead.

10. **Text Encoder Context Length**: SAM3 uses **32 tokens** (not 77 like standard CLIP). Ensure tokenizer `max_length=32`.

11. **PCS Decoder Padding Boxes**: When running text-only detection, pass padding boxes with shape `[B, 1, 4]` and labels `[B, 1]` with value `-10` (padding marker). ONNX cannot handle zero-dimension tensors `[B, 0, 4]`.

12. **Dynamic Model Paths**: The `UnifiedModelHandler` now accepts a `model_dir` parameter in the constructor for testability. Falls back to `SAM3_MODEL_DIR` env var or default `/opt/nuclio/sam3/models`.

#### Test Results (1 Feb 2026)

**Unified ONNX Test Suite** (`test_onnx_unified.py --all`):
```
Vision Encoder:
  fpn_feat_0: Correlation=1.00, MAE=0.00000007 ✓ PASS
  fpn_feat_1: Correlation=1.00, MAE=0.00000009 ✓ PASS
  fpn_feat_2: Correlation=1.00, MAE=0.00000009 ✓ PASS
  fpn_pos_2:  Correlation=1.00, MAE=0.00000000 ✓ PASS

Tracker Decoder:
  masks:              Correlation=1.00, MAE=0.00006221 ✓ PASS
  iou_predictions:    Correlation=1.00, MAE=0.00000018 ✓ PASS
  low_res_masks:      Correlation=1.00, MAE=0.00000744 ✓ PASS
  object_score_logits: Correlation=1.00, MAE=0.00000000 ✓ PASS

Text Encoder:
  text_features: Correlation=1.00, MAE=0.00000035 ✓ PASS
  text_mask:     Correlation=1.00, MAE=0.00000000 ✓ PASS

End-to-End Encode (Interactor):
  ✓ Vision encoder outputs 4 FPN embeddings
  ✓ Tracker decoder produces masks, IoU, low-res masks

End-to-End Text-to-Segment (Detector):
  ✓ Vision encoder → Text encoder → PCS decoder pipeline
  ✓ PCS decoder outputs: pred_masks (1,200,288,288), pred_boxes (1,200,4)

Unified Handler Module:
  ✓ Import and instantiate with custom model_dir
  ✓ get_model_info() returns all capabilities
  ✓ encode() returns 4 FPN embeddings with correct shapes
```

**All 6 tests pass with 100% correlation to PyTorch reference.**

#### Previous IoU Test Results (Historical)

**End-to-End ONNX vs PyTorch Test**:
```
PyTorch IoU with GT: 0.992
ONNX IoU with GT:    0.997
ONNX vs PyTorch:     0.995 ✓ PASS
```

**Comprehensive Shape Tests** (6 different shapes):
```
Circle centered:    IoU = 0.997 ✓ PASS
Circle off-center:  IoU = 0.995 ✓ PASS
Small circle:       IoU = 0.991 ✓ PASS
Rectangle:          IoU = 0.991 ✓ PASS
Tall rectangle:     IoU = 0.988 ✓ PASS
Triangle:           IoU = 0.992 ✓ PASS
Average IoU:        99.2%
```

---

## Implementation Plan

### Goal: Unified SAM3 Nuclio Function

Create a **single nuclio function** that handles all three CVAT AI tool types, saving GPU VRAM by loading the model once (~3.5 GB vs ~7 GB for separate functions).

### Model Deployment Strategy

| Model | Size | Location | Purpose |
|-------|------|----------|---------|
| **Vision Encoder** | 1.8 GB | `serverless/.../sam3/nuclio/` | Encode images (server) |
| **Text Encoder** | 1.3 GB | `serverless/.../sam3/nuclio/` | PCS text prompts (server) |
| **PCS Decoder** | 123 MB | `serverless/.../sam3/nuclio/` | Text→segment (server) |
| **Tracker Decoder** | 16 MB | `cvat-ui/plugins/sam3/assets/` | Click/box interactor (browser) |

### Unified Function Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAM3 UNIFIED NUCLIO FUNCTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Request routing by 'mode' parameter:                                       │
│                                                                             │
│  mode="encode"           → Interactor (returns embeddings for browser)      │
│  mode="text-to-segment"  → Detector (returns masks + boxes)                 │
│  mode="track/init"       → Tracker init (Redis session, returns mask)       │
│  mode="track/frame"      → Tracker propagate (Redis state, returns mask)    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HuggingFace Models (lazy loaded, shared backbone):                         │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Sam3Model         │  │  Sam3TrackerModel   │  │  Sam3VideoModel     │  │
│  │   (PCS/Detector)    │  │  (PVS/Interactor)   │  │  (Video Tracker)    │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  State Management:                                                          │
│  - Interactor: Stateless (embeddings sent to browser)                       │
│  - Detector: Stateless (masks returned immediately)                         │
│  - Tracker: Redis-backed session state (memory bank too large for request)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

1. ✅ **Update `sam3-unified` function** to add video tracking mode
   - ✅ Added Redis dependency for session state (`RedisCache` class)
   - ✅ Lazy load `Sam3VideoModel` only when tracking is used
   - ✅ Implemented CVAT tracker interface (`shapes` + `states` format)
   - ✅ Added `track/init`, `track/frame`, `track/clear` modes in `main.py`

2. ✅ **Browser Integration** for interactor mode
   - ✅ Updated `inference.worker.ts` to detect unified decoder (256ch inputs)
   - ✅ Added support for `fpn_feat_*` tensor names (new unified format)
   - ✅ Added 4D point_coords `[B, num_objects, num_points, 2]` for HuggingFace decoder
   - ✅ Added 3D point_labels `[B, num_objects, num_points]` for HuggingFace decoder
   - ✅ Backward compatible with legacy 32/64/256ch decoders

3. 🔄 **End-to-End Testing**
   - Test interactor mode (clicks → mask)
   - Test detector mode (text prompts → all instances)
   - Test tracker mode (propagate masks across frames)

### CVAT Tracker Interface Compatibility

Existing CVAT trackers (SiamMask, TransT) use this request/response format:

```python
# Request (per frame):
{
    "image": "<base64 frame>",
    "shapes": [[x1, y1, x2, y2], ...],  # Bounding boxes
    "states": [{"key": "value"}, ...]    # Serialized state per object
}

# Response:
{
    "shapes": [[x1, y1, x2, y2], ...],  # Updated shapes
    "states": [{"key": "value"}, ...]    # Updated state
}
```

For SAM3, the `states` will contain a Redis session ID instead of the full memory bank:
```python
states = [{"session_id": "sam3_track_12345", "object_id": 1}, ...]
```

---

## Reference Implementations

### cvat-yoloe-sam (Full SAM3 with HuggingFace)

**Location**: `/home/jahs/GitHub/cvat/cvat-yoloe-sam/`

A reference CVAT fork with comprehensive SAM3 implementation using HuggingFace Transformers.

**Key Files**:
- `serverless/pytorch/facebookresearch/sam3/nuclio/model_handler.py` - Full SAM3 handler
- `serverless/pytorch/facebookresearch/sam3/nuclio/main.py` - Multi-endpoint routing
- `serverless/pytorch/facebookresearch/sam3/nuclio/function-gpu.yaml` - Nuclio config

**Features**:
- Lazy loading of Sam3Model, Sam3TrackerModel, Sam3VideoModel
- Redis caching for embeddings and tracking state
- Multiple endpoints: `/segment-text`, `/segment-box`, `/segment-points`, `/detect`, `/track/*`
- HuggingFace token support for gated models

**HuggingFace Models Used**:
```python
from transformers import Sam3Model, Sam3Processor
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
from transformers import Sam3VideoModel, Sam3VideoProcessor
```

### Existing CVAT Trackers

**SiamMask** (`serverless/pytorch/foolwood/siammask/nuclio/`):
- Returns polygon shapes (not just boxes)
- State serialized with jsonpickle
- ~100MB model size

**TransT** (`serverless/pytorch/dschoerk/transt/nuclio/`):
- Returns bounding boxes
- State includes template features and position
- ~50MB model size

**Key Pattern**: Both use stateless request/response with all state serialized in the response. This works because their state is small (~MB). SAM3 video tracking has GB-scale memory banks, so we use Redis.

### usls ONNX Export Scripts

**Location**: `/home/jahs/GitHub/cvat/usls/scripts/sam3-image/`

Rust ONNX runtime with Python export scripts using HuggingFace Transformers.

**Key Files**:
- `export_v2.py` - Main export script (use this for ONNX export)

**Usage**:
```bash
python export_v2.py --all --model-path facebook/sam3 --output-dir /tmp/sam3-onnx
```

### Official SAM3 Repository

**Location**: `/home/jahs/GitHub/cvat/sam3/`

Official Facebook SAM3 PyTorch implementation.

**⚠️ Important**: Cannot export vision encoder to ONNX due to `torch.view_as_complex()` in RoPE. Use HuggingFace Transformers implementation instead.

### Implemented: sam3-unified ONNX Nuclio Function

**Location**: `serverless/onnx/facebookresearch/sam3-unified/nuclio/`

This is the **unified SAM3 ONNX function** that combines all three CVAT AI tool types using ONNX Runtime - **NO HuggingFace auth needed at runtime!**

**Files**:
- `model_handler_unified.py` - ONNX Runtime handler with lazy model loading
- `main_unified.py` - Request routing by `mode` parameter
- `function-gpu-unified.yaml` - Nuclio configuration

**Why ONNX?**
- The `facebook/sam3-hiera-large` model is **gated** on HuggingFace
- HuggingFace Transformers requires authentication to load the model
- ONNX models can be exported once (requires auth) then deployed without auth
- ONNX Runtime is faster and more portable than PyTorch

**Supported Modes**:
```python
# Request modes:
mode="encode"           # Interactor: Image → embeddings for browser decoding
mode="text-to-segment"  # Detector: Image + text → masks + boxes
mode="track/init"       # Tracker: Initialize tracking session
mode="track/frame"      # Tracker: Propagate to next frame
mode="track/clear"      # Tracker: Clear session
mode="info"             # Get model information
```

**ONNX Models Required** (export via `export_hf_onnx.py`):
```
/opt/nuclio/sam3/models/
├── vision_encoder.onnx      # 1.8 GB - 256ch at all FPN levels
├── text_encoder.onnx        # 1.3 GB - CLIP text encoding
├── pcs_decoder.onnx         # 123 MB - DETR decoder for PCS
└── tracker_decoder.onnx     # 16 MB - mask decoder with projections
```

**Video Tracking State Management**:
```python
# Redis-based session state (memory bank too large for request body)
class RedisCache:
    def set(self, prefix, key, data, ttl=3600): ...
    def get(self, prefix, key): ...
    def delete(self, prefix, key): ...

# Session ID returned in tracker response
states = [{"session_id": "sam3_track_abc123", "object_id": 1}]
```

**Environment Variables**:
```yaml
- SAM3_MODEL_DIR      # Directory containing ONNX models
- SAM3_VISION_ENCODER # Path to vision_encoder.onnx
- SAM3_TEXT_ENCODER   # Path to text_encoder.onnx
- SAM3_PCS_DECODER    # Path to pcs_decoder.onnx
- SAM3_TRACKER_DECODER # Path to tracker_decoder.onnx
- REDIS_HOST          # Redis server host (default: cvat_redis)
- REDIS_PORT          # Redis server port (default: 6379)
- REDIS_TTL           # Session TTL in seconds (default: 3600)
```

---

## SAM3 Capabilities for CVAT

SAM3 is a **unified foundation model** that supports all three CVAT AI tool categories with a single model:

### CVAT AI Tool Categories

| Category | Description | SAM3 Mode | Status |
|----------|-------------|-----------|--------|
| **Interactor** | User clicks points/boxes → single mask | Sam3Tracker (PVS) | ✅ Implemented |
| **Detector** | Model finds all instances of class(es) | SAM3 PCS | ✅ Implemented |
| **Tracker** | Propagate masks across video frames | Sam3TrackerVideo | 🔄 Planned |

### How SAM3 Modes Map to CVAT Tools

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SAM3: ONE MODEL, THREE TOOLS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INTERACTOR (type: interactor)        │  Input: Image + clicks/box         │
│  ════════════════════════════════     │  Output: Single mask for ONE object│
│  Sam3Tracker / Sam3 (PVS mode)        │  Use: Interactive annotation       │
│                                       │                                    │
│  User clicks → mask appears           │  ┌─────┐  click   ┌──────────┐     │
│  User refines → mask updates          │  │Image│ ───────► │ ONE mask │     │
│                                       │  └─────┘          └──────────┘     │
├───────────────────────────────────────┴────────────────────────────────────┤
│                                                                             │
│  DETECTOR (type: detector)            │  Input: Image + text prompt(s)     │
│  ═════════════════════════════        │  Output: ALL matching instances    │
│  SAM3 PCS mode                        │  Use: Auto-annotation by class     │
│                                       │                                    │
│  "person" → finds ALL people          │  ┌─────┐ "person" ┌──────────────┐ │
│  "car, dog" → finds all cars & dogs   │  │Image│ ───────► │ N masks+boxes│ │
│                                       │  └─────┘          └──────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRACKER (type: tracker)              │  Input: Video + initial mask/box   │
│  ═════════════════════════════        │  Output: Masks for ALL frames      │
│  Sam3TrackerVideo (PVS mode)          │  Use: Video object tracking        │
│                                       │                                    │
│  Frame 0: user annotates              │  ┌─────┐ mask_0   ┌──────────────┐ │
│  Frame 1-N: auto-propagated           │  │Video│ ───────► │ N frame masks│ │
│                                       │  └─────┘          └──────────────┘ │
│                                                                             │
│  BONUS: Text-based Tracking           │  "person" on video → track ALL     │
│  SAM3 Video PCS mode                  │  people across all frames          │
│                                       │                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SAM3 Function Locations

| Function | Path | Type | GPU VRAM |
|----------|------|------|----------|
| **SAM3 Unified (Recommended)** | `serverless/pytorch/facebookresearch/sam3-unified/` | interactor+detector+tracker | ~3.5GB |
| SAM3 Interactor (ONNX) | `serverless/onnx/facebookresearch/sam3/` | interactor | ~2GB |
| SAM3 PCS (PyTorch) | `serverless/pytorch/facebookresearch/sam3-pcs/` | detector | ~3.5GB |

### Memory-Efficient Deployment

Use the **unified function** for all three capabilities with a single model load:

```bash
# Recommended: Single function for interactor + detector + tracker
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3-unified
```

The unified function routes requests by `mode` parameter:
- `mode="encode"` → Interactor (returns embeddings for browser ONNX decoder)
- `mode="text-to-segment"` → Detector (returns masks + boxes)
- `mode="track/init"` → Video tracker init (Redis session)
- `mode="track/frame"` → Video tracker propagate (Redis state)

---

## SAM3 Architecture Overview

SAM3 is a unified model for segmentation with multiple prompt types. All encoders are **completely independent** (no parameter sharing).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SAM3 MODEL (~840M params)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │   VISION ENCODER    │    │    TEXT ENCODER     │                        │
│  │    (454M params)    │    │   (354M params)     │                        │
│  │  ViT + FPN neck     │    │  CLIP Text Model    │                        │
│  └──────────┬──────────┘    └──────────┬──────────┘                        │
│             │                          │                                    │
│             │   ┌──────────────────────┴────────────┐                      │
│             │   │  ┌─────────────────────────────┐  │                      │
│             │   │  │    GEOMETRY ENCODER (8M)   │  │  (optional: boxes)   │
│             │   │  └──────────────┬──────────────┘  │                      │
│             │   │     ┌───────────┴───────────┐     │                      │
│             │   └─────┤  combined_prompts     ├─────┘                      │
│             │         └───────────┬───────────┘                            │
│             │                     ▼                                        │
│             └────────►┌───────────────────────┐                            │
│                       │  DETR ENCODER (10M)   │  Cross-attention           │
│                       └───────────┬───────────┘                            │
│                                   ▼                                        │
│                       ┌───────────────────────┐                            │
│                       │  DETR DECODER (12M)   │  Object queries            │
│                       └───────────┬───────────┘                            │
│                                   │                                        │
│             ┌─────────────────────┼─────────────────────┐                  │
│             ▼                     ▼                     ▼                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│  │ DOT PRODUCT (1M)│   │    BOX HEAD     │   │ MASK DECODER(2M)│          │
│  │  pred_logits    │   │   pred_boxes    │   │   pred_masks    │          │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How SAM3 Components Fit Together

### Component Inventory

SAM3 consists of **8 independent components** with no shared parameters:

| Component | Parameters | Input | Output | Purpose |
|-----------|------------|-------|--------|---------|
| **Vision Encoder** | 454M | Image `[B,3,1008,1008]` | FPN features (3 levels) | Extract visual features |
| **Text Encoder** | 354M | Token IDs `[B,32]` | Text embeddings `[32,B,256]` | Encode text prompts |
| **Geometry Encoder** | 8M | Boxes/Points + FPN | Geometry embeddings | Encode spatial prompts |
| **DETR Encoder** | 10M | FPN + Prompts | Memory features | Fuse image & prompts |
| **DETR Decoder** | 12M | Memory + Queries | Object features | Generate object proposals |
| **Mask Decoder** | 2M | Object features + FPN | Masks `[B,N,H,W]` | Produce segmentation masks |
| **Scoring Head** | 1M | Object features + Text | Logits `[B,N,1]` | Match objects to prompts |
| **Box Head** | ~0.5M | Object features | Boxes `[B,N,4]` | Predict bounding boxes |

### Data Flow: Text-to-Segment Mode (PCS)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TEXT-TO-SEGMENT PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   IMAGE                              TEXT PROMPT                             │
│  [B,3,1008,1008]                    "a person"                               │
│       │                                  │                                   │
│       ▼                                  ▼                                   │
│  ┌─────────────┐                   ┌───────────┐                             │
│  │   VISION    │                   │   TEXT    │                             │
│  │   ENCODER   │                   │  ENCODER  │                             │
│  │  (ViT+FPN)  │                   │  (CLIP)   │                             │
│  └──────┬──────┘                   └─────┬─────┘                             │
│         │                                │                                   │
│         │  FPN Features:                 │  Text Features:                   │
│         │  - Level 0: [B,256,288,288]    │  - [32, B, 256]                   │
│         │  - Level 1: [B,256,144,144]    │  - text_mask: [B, 32]             │
│         │  - Level 2: [B,256,72,72]      │                                   │
│         │                                │                                   │
│         │         ┌──────────────────────┤                                   │
│         │         │  Optional:           │                                   │
│         │         │  ┌─────────────┐     │                                   │
│         │         │  │  GEOMETRY   │     │  Box prompts: [B,N,4]             │
│         │         │  │   ENCODER   │◄────┤  (for guided segmentation)        │
│         │         │  └──────┬──────┘     │                                   │
│         │         │         │            │                                   │
│         │         └─────────┼────────────┘                                   │
│         │                   │                                                │
│         │                   ▼                                                │
│         │         ┌─────────────────┐                                        │
│         │         │ CONCAT PROMPTS  │  [text_feats + geo_feats]              │
│         │         └────────┬────────┘                                        │
│         │                  │                                                 │
│         ▼                  ▼                                                 │
│  ┌──────────────────────────────────┐                                        │
│  │         DETR ENCODER             │  Cross-attention:                      │
│  │   (TransformerEncoderFusion)     │  - Image attends to prompts            │
│  │                                  │  - Only uses Level 2 (72x72)           │
│  └───────────────┬──────────────────┘                                        │
│                  │                                                           │
│                  │  Memory: [5184, B, 256]  (72*72 = 5184)                   │
│                  ▼                                                           │
│  ┌──────────────────────────────────┐                                        │
│  │         DETR DECODER             │  200 learnable object queries          │
│  │     (TransformerDecoder)         │  Cross-attend to memory + prompts      │
│  └───────────────┬──────────────────┘                                        │
│                  │                                                           │
│                  │  Object Features: [B, 200, 256]                           │
│                  │                                                           │
│    ┌─────────────┼─────────────┬─────────────┐                               │
│    │             │             │             │                               │
│    ▼             ▼             ▼             ▼                               │
│ ┌──────┐    ┌────────┐   ┌─────────┐   ┌──────────┐                          │
│ │SCORES│    │  BOX   │   │  MASK   │   │PRESENCE  │                          │
│ │(dot) │    │  HEAD  │   │ DECODER │   │  SCORE   │                          │
│ └──┬───┘    └───┬────┘   └────┬────┘   └────┬─────┘                          │
│    │            │             │             │                                │
│    ▼            ▼             ▼             ▼                                │
│ pred_logits  pred_boxes   pred_masks   obj_presence                          │
│ [B,200,1]    [B,200,4]    [B,200,H,W]   [B,200,1]                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Interactor Mode (Point/Box Clicks)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INTERACTOR PIPELINE                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   IMAGE                              CLICKS/BOX                              │
│  [B,3,1008,1008]                    point_coords, point_labels               │
│       │                                  │                                   │
│       ▼                                  │                                   │
│  ┌─────────────┐ SERVER                  │                                   │
│  │   VISION    │ (PyTorch)               │                                   │
│  │   ENCODER   │                         │                                   │
│  └──────┬──────┘                         │                                   │
│         │                                │                                   │
│         │  Cached Embeddings:            │                                   │
│         │  - image_embed: [B,256,72,72]  │                                   │
│         │  - high_res_0: [B,32,288,288]  │                                   │
│         │  - high_res_1: [B,64,144,144]  │                                   │
│         │                                │                                   │
│  ───────┼────────────────────────────────┼───────────────────────────────    │
│         │         BROWSER                │                                   │
│         │         (ONNX)                 │                                   │
│         ▼                                ▼                                   │
│  ┌──────────────────────────────────────────────┐                            │
│  │            PROMPT ENCODER + MASK DECODER     │                            │
│  │                 (16.3 MB ONNX)               │                            │
│  │                                              │                            │
│  │   1. Encode points/boxes → sparse_embed      │                            │
│  │   2. Encode prev mask → dense_embed          │                            │
│  │   3. Cross-attend to image features          │                            │
│  │   4. Upsample through high-res features      │                            │
│  │   5. Output 3 mask candidates + IoU scores   │                            │
│  └───────────────────┬──────────────────────────┘                            │
│                      │                                                       │
│                      ▼                                                       │
│               ┌────────────┐                                                 │
│               │ SELECT BEST│  Pick mask with highest IoU score               │
│               │    MASK    │  (or use stability-based selection)             │
│               └─────┬──────┘                                                 │
│                     │                                                        │
│                     ▼                                                        │
│              pred_mask [1008,1008]                                           │
│              low_res_mask [288,288] → cache for refinement                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Insights

#### 1. FPN Feature Usage

The backbone produces 3 FPN levels, but they're used differently:

| FPN Level | Resolution | Used By |
|-----------|------------|---------|
| Level 0 | 288×288 | Mask decoder (high-res details) |
| Level 1 | 144×144 | Mask decoder (mid-res features) |
| Level 2 | 72×72 | **DETR encoder** (main processing) |

**Key insight**: The DETR encoder (`num_feature_levels=1`) only processes the **smallest FPN level** (72×72). All fusion and attention happens at this resolution. The higher-resolution features are only used in the mask decoder for upsampling.

#### 2. Prompt Encoding

Prompts are encoded and concatenated before the DETR encoder:

```python
# Text prompts: [seq_len, B, 256]
text_features = text_encoder(input_ids)

# Geometry prompts (optional): [num_geo, B, 256]
geo_features = geometry_encoder(boxes, points, fpn_features)

# Combined: [seq_len + num_geo + 1, B, 256]
#           └─ text ─┘  └─ geo ─┘  └─ CLS token
prompt = concat([text_features, geo_features, cls_token])
```

#### 3. DETR Decoder Object Queries

The DETR decoder uses **200 learnable object queries** that cross-attend to:
1. **Memory** (fused image-prompt features from encoder)
2. **Prompts** (text + geometry features directly)

Each query proposes one potential object, producing:
- Object feature vector (256-dim)
- Predicted box (4 coords)
- Predicted mask (via mask decoder)
- Class logits (via dot product with text embeddings)
- Presence score (is this query valid?)

#### 4. Scoring via Dot Product

SAM3 uses **dot product scoring** instead of a classification head:

```python
# Object features: [B, 200, 256]
# Text features: [32, B, 256]
# Score = object_feat · text_feat
scores = torch.einsum("bqc,lbc->bql", object_features, text_features)
# Result: [B, 200, 32] → which text prompt matches which object
```

This enables **open-vocabulary detection** - no fixed class labels.

### File Locations

| Component | Official SAM3 | HuggingFace |
|-----------|---------------|-------------|
| Vision Encoder | `sam3/model/image_encoder/` | `transformers/models/sam3/` |
| Text Encoder | `sam3/model/language_backbone/` | `transformers/models/sam3/` |
| Geometry Encoder | `sam3/model/geometry_encoders.py` | (not separate) |
| DETR Encoder | `sam3/model/encoder.py` | (not separate) |
| DETR Decoder | `sam3/model/decoder.py` | (not separate) |
| Mask Decoder | `sam3/sam/mask_decoder.py` | `transformers/models/sam3/` |
| Full Pipeline | `sam3/model/sam3_image_predictor.py` | `Sam3Model` |

---

## Two Model Implementations

SAM3 has two PyTorch implementations with **different ONNX export capabilities**:

### 1. Official Facebook Repository (`sam3/`)

- **RoPE Implementation**: Uses `torch.view_as_complex()`
- **ONNX Export**: ❌ **Cannot export vision encoder** - `view_as_complex` not supported in ONNX
- **Use Case**: PyTorch inference, training, server-side processing

### 2. HuggingFace Transformers (`transformers.models.sam3`)

- **RoPE Implementation**: Pre-computes cos/sin as buffers (ONNX-compatible)
- **ONNX Export**: ✅ **Can export all components** including vision encoder
- **Use Case**: ONNX export for browser/edge deployment

**Key Difference**:
```python
# Official SAM3 (cannot export)
freqs_cis = torch.view_as_complex(freqs_cis)  # ❌ Not supported in ONNX

# HuggingFace (can export)
self.register_buffer("rope_embeddings_cos", inv_freq.cos())  # ✅ ONNX compatible
self.register_buffer("rope_embeddings_sin", inv_freq.sin())  # ✅ ONNX compatible
```

---

## ONNX Export Methods

### Method 1: usls Export Scripts (Full ONNX)

Uses HuggingFace Transformers to export all components.

**Location**: `usls/scripts/sam3-image/`

```bash
python export_v2.py --all \
  --model-path facebook/sam3 \
  --output-dir /tmp/sam3-onnx-export \
  --device cuda \
  --image-height 1008 --image-width 1008
```

**Output**:
| File | Size | Description |
|------|------|-------------|
| `vision-encoder.onnx` | 1.8 GB | ViT + FPN backbone |
| `text-encoder.onnx` | 1.4 GB | CLIP text encoder |
| `decoder.onnx` | 124 MB | Geometry + DETR + Mask decoder |

### Method 2: Custom Decoder Export (Interactor Mode)

For point/box prompts with mask refinement.

```bash
python export_decoder_with_mask_input.py \
  --output cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx
```

### Method 3: PyTorch Server-Side (Text-to-Segment)

Text prompts use full PyTorch pipeline on server - no ONNX export needed.

---

## Unified HuggingFace ONNX Export (Recommended)

This section describes the new **unified export approach** using HuggingFace Transformers for all ONNX components.

### Why Unified HuggingFace Export?

Previously, we used a hybrid approach:
- Vision encoder from external `onnx-community/sam3-tracker-ONNX` (32/64/256ch outputs - projections baked in)
- Our own tracker decoder (expects 32/64/256)
- PCS mode using PyTorch (expects 256/256/256)

**Problems with the hybrid approach:**
1. **No control**: External ONNX models have baked-in channel projections
2. **Inconsistency**: Vision encoder outputs differ between modes (32/64/256 vs 256/256/256)
3. **Maintenance**: Cannot easily update or modify external models

**New unified approach:**
- **Single vision encoder** from HuggingFace: outputs 256/256/256 at all FPN levels
- **Tracker decoder** includes conv_s0/conv_s1 projections internally
- **Full control** over all ONNX exports

### Export Script

**Location**: `serverless/pytorch/facebookresearch/sam3/nuclio/export_hf_onnx.py`

```bash
# Export all components
python export_hf_onnx.py --all --output-dir /tmp/sam3-onnx

# Or export individually
python export_hf_onnx.py --vision-encoder --output-dir /tmp/sam3-onnx
python export_hf_onnx.py --tracker-decoder --output-dir /tmp/sam3-onnx
python export_hf_onnx.py --text-encoder --output-dir /tmp/sam3-onnx
python export_hf_onnx.py --pcs-decoder --output-dir /tmp/sam3-onnx

# Verify exports match PyTorch
python export_hf_onnx.py --verify --output-dir /tmp/sam3-onnx
```

### Exported Models

| Model | Size | Description |
|-------|------|-------------|
| `vision-encoder.onnx` | ~1.8 GB | ViT backbone + FPN neck (256ch outputs) |
| `tracker-decoder.onnx` | ~16 MB | Prompt encoder + mask decoder (includes projections) |
| `text-encoder.onnx` | ~1.3 GB | CLIP text encoder + projection |
| `pcs-decoder.onnx` | ~123 MB | DETR encoder/decoder + scoring heads |

### Architecture: Unified Vision Encoder

The vision encoder outputs **256 channels at all FPN levels**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VISION ENCODER (1.8 GB)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: image [B, 3, 1008, 1008]                                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                    ViT Backbone (1024-dim)                          │    │
│   │   - Patch embeddings (14x14 stride)                                 │    │
│   │   - Position embeddings (interpolated)                              │    │
│   │   - Transformer layers with RoPE                                    │    │
│   └───────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     FPN Neck (256-dim)                              │    │
│   │   - Level 0: [B, 256, 288, 288] (4× upsample)                       │    │
│   │   - Level 1: [B, 256, 144, 144] (2× upsample)                       │    │
│   │   - Level 2: [B, 256, 72, 72]   (native)                            │    │
│   └───────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│   Outputs:                    │                                              │
│   - fpn_feat_0: [B, 256, 288, 288]                                           │
│   - fpn_feat_1: [B, 256, 144, 144]                                           │
│   - fpn_feat_2: [B, 256, 72, 72]                                             │
│   - fpn_pos_2:  [B, 256, 72, 72]  (position encoding)                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Architecture: Tracker Decoder (with Internal Projections)

The tracker decoder **includes conv_s0/conv_s1** projections, accepting 256ch inputs:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      TRACKER DECODER (16 MB)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Inputs:                                                                    │
│   - fpn_feat_0: [B, 256, 288, 288]  ───► conv_s0 ───► [B, 32, 288, 288]     │
│   - fpn_feat_1: [B, 256, 144, 144]  ───► conv_s1 ───► [B, 64, 144, 144]     │
│   - fpn_feat_2: [B, 256, 72, 72]    ───► + no_mem_embed                      │
│   - point_coords: [B, N, 2]                                                  │
│   - point_labels: [B, N]                                                     │
│   - mask_input: [B, 1, 288, 288]                                             │
│   - has_mask_input: [B]                                                      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                      Prompt Encoder                                 │    │
│   │   - Point embeddings (positional + type)                            │    │
│   │   - Mask downsampling (288→288 convolutions)                        │    │
│   └───────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                      Mask Decoder (multimask_output=True)           │    │
│   │   - Two-way transformer (prompt ↔ image cross-attention)            │    │
│   │   - 4 mask tokens → 3 output masks (token 0 discarded)              │    │
│   │   - Upsampling through high-res features                            │    │
│   └───────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│   Outputs:                    │                                              │
│   - masks: [B, 3, 1008, 1008]                                                │
│   - iou_predictions: [B, 3]                                                  │
│   - low_res_masks: [B, 3, 288, 288]                                          │
│   - object_score_logits: [B, 1]                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

#### 1. no_mem_embed Handling

The `no_mem_embed` tensor has shape `[1, 1, 256]` and must be reshaped for spatial broadcast:

```python
# In TrackerDecoderWrapper.forward():
no_mem_embed = self.no_mem_embed.view(1, 256, 1, 1)  # Reshape for broadcast
image_embed = image_embed + no_mem_embed  # [B, 256, 72, 72] + [1, 256, 1, 1]
```

#### 2. Multimask Output

The mask decoder uses `multimask_output=True`, which internally slices:
```python
# Mask decoder returns tokens 1, 2, 3 (skips token 0)
masks = masks[:, 1:, :, :]  # [B, 4, H, W] → [B, 3, H, W]
iou_pred = iou_pred[:, 1:]  # [B, 4] → [B, 3]
```

#### 3. Position Encoding

Position encoding is **pre-computed** as a buffer in the vision encoder wrapper:
```python
# Sinusoidal position encoding for 72×72 spatial grid
pos_enc_2 = compute_sine_position_encoding(shape=(1, 256, 72, 72))
self.register_buffer("pos_enc_2", pos_enc_2)
```

### Verification Results

All exports have been verified against PyTorch with Mean Absolute Error (MAE) < 0.001:

| Component | MAE | Max Diff | Status |
|-----------|-----|----------|--------|
| Vision Encoder (fpn_feat_0) | 0.00000005 | 0.00000517 | ✅ PASS |
| Vision Encoder (fpn_feat_1) | 0.00000370 | 0.00047082 | ✅ PASS |
| Vision Encoder (fpn_feat_2) | 0.00000263 | 0.00029105 | ✅ PASS |
| Vision Encoder (fpn_pos_2) | 0.00000000 | 0.00000000 | ✅ PASS |
| Text Encoder (text_features) | 0.00000305 | 0.00002313 | ✅ PASS |
| Text Encoder (text_mask) | 0.00000000 | 0.00000000 | ✅ PASS |

### Migration Notes

**For browser-side code:**
1. Update vision encoder to output 256ch features (not 32/64/256)
2. Use new tracker decoder that accepts 256ch inputs
3. No changes needed for mask selection logic (still pick highest IoU)

**For server-side code:**
1. Can use unified vision encoder for both interactor and PCS modes
2. PCS decoder can share same vision encoder outputs

---

## Current Working Implementation

| Component | Location | Status |
|-----------|----------|--------|
| **PyTorch Encoder** | `model_handler_pytorch.py` | ✅ Working |
| **ONNX Decoder** | `tracker-prompt-encoder-mask-decoder-with-mask-input.onnx` | ✅ Working (16.3 MB) |
| **Text-to-Segment** | `model_handler_pcs.py` (PyTorch) | ✅ Working |
| **Full ONNX Export** | `usls/scripts/sam3-image/` | ✅ Verified |

### Why Hybrid Approach?

**Interactor Mode** (point/box clicks):
- Vision encoder: PyTorch on server (official repo has view_as_complex issue)
- Decoder: ONNX in browser (fast interactive feedback)

**Text-to-Segment Mode**:
- Full PyTorch pipeline on server
- Returns complete masks (not embeddings)

### Key Implementation Details

1. **Model Loading** (CRITICAL):
   ```python
   # CORRECT: Use build_sam3_image_model with enable_inst_interactivity=True
   from sam3.model_builder import build_sam3_image_model
   model = build_sam3_image_model(
       device=device,
       eval_mode=True,
       load_from_HF=True,  # Loads checkpoint from HuggingFace
       enable_inst_interactivity=True,  # Creates tracker with loaded weights
   )

   # WRONG: build_tracker() does NOT load checkpoint weights!
   # from sam3.model_builder import build_tracker
   # tracker = build_tracker(...)  # Random weights, ~0.5 IoU, 90%+ mask coverage
   ```

2. **Channel Projections**:
   ```python
   # The backbone outputs 256ch features at all levels
   # conv_s0/conv_s1 project to 32ch/64ch for high-res features
   backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
   backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
   ```

3. **Embedding Shapes**:
   | Name | Shape | Description |
   |------|-------|-------------|
   | `high_res_feats_0` | `[B, 32, 288, 288]` | Level 0 high-res features |
   | `high_res_feats_1` | `[B, 64, 144, 144]` | Level 1 high-res features |
   | `image_embed` | `[B, 256, 72, 72]` | Main backbone embedding |

4. **ONNX Decoder Inputs/Outputs**:
   ```
   Inputs:
     - image_embed: [B, 256, 72, 72] FLOAT32
     - high_res_feats_0: [B, 32, 288, 288] FLOAT32
     - high_res_feats_1: [B, 64, 144, 144] FLOAT32
     - point_coords: [B, N, 2] FLOAT32 (in 1008x1008 space)
     - point_labels: [B, N] FLOAT32 (1=positive, 0=negative)
     - mask_input: [B, 1, 288, 288] FLOAT32 (previous low_res_mask)
     - has_mask_input: [B] FLOAT32 (1.0 if using mask refinement)

   Outputs:
     - masks: [B, 3, 1008, 1008] FLOAT32
     - iou_predictions: [B, 3] FLOAT32
     - low_res_masks: [B, 3, 288, 288] FLOAT32 (for refinement)
     - object_score_logits: [B, 1] FLOAT32
   ```

### Running Tests

```bash
cd cvat
conda activate grimme-tf2.18  # Or your SAM3 environment
python serverless/pytorch/facebookresearch/sam3/nuclio/test_sam3_multiclick.py --device cpu --save-viz
```

### Re-Exporting the Decoder

```bash
cd cvat
python serverless/pytorch/facebookresearch/sam3/nuclio/export_sam3_onnx.py \
    --export decoder \
    --output cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx
```

---

## Key Findings from Official SAM3 Implementation

### 1. Mask Selection with `multimask_output=True` (ONNX Decoder)

**IMPORTANT**: Our ONNX decoder uses `multimask_output=True`, which fundamentally changes how mask selection works.

#### What the Decoder Outputs

When `multimask_output=True`, the SAM3 mask decoder internally does this:
```python
# From sam3/sam/mask_decoder.py forward():
if multimask_output:
    masks = masks[:, 1:, :, :]  # Skip token 0, return tokens 1, 2, 3
    iou_pred = iou_pred[:, 1:]
```

This means our ONNX decoder returns **3 masks** that correspond to **tokens 1, 2, 3** (the "multi-object" tokens). Token 0 (the "single-object" token) is **NOT included**.

#### Correct Selection Logic

From the official SAM3 documentation (`sam1_task_predictor.py`):

> "For ambiguous input prompts (such as a single click), [multimask_output=True] will often produce better masks than a single prediction. **If only a single mask is needed, the model's predicted quality score can be used to select the best mask.**"

Therefore, the browser should simply **select the mask with the highest IoU score**:

```typescript
let bestIdx = 0;
let bestIou = iouData[0];
for (let i = 1; i < numMasks; i++) {
    if (iouData[i] > bestIou) {
        bestIou = iouData[i];
        bestIdx = i;
    }
}
```

This matches:
- The official SAM3 recommendation for `multimask_output=True`
- The usls Rust implementation (always picks best IoU)
- The SAM2 behavior when using multimask mode

#### What About `_dynamic_multimask_via_stability`?

The stability-based selection (`_dynamic_multimask_via_stability`) is **only used when `multimask_output=False`**:

```python
# From sam3/sam/mask_decoder.py forward():
if multimask_output:
    masks = masks[:, 1:, :, :]  # Return multi-masks
elif self.dynamic_multimask_via_stability and not self.training:
    masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)  # Only for single-mask mode
else:
    masks = masks[:, 0:1, :, :]
```

Since our ONNX decoder uses `multimask_output=True`, stability-based selection does not apply.

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

---

## Appendix: Key Discovery - HuggingFace Transformers vs Official SAM3

### Problem: Vision Encoder ONNX Export Fails

The official Facebook SAM3 repository uses `torch.view_as_complex()` in the vision encoder's Rotary Position Embedding (RoPE):

```python
# Official SAM3 (sam3/model/image_encoder/image_encoder.py)
freqs_cis = torch.view_as_complex(freqs_cis)  # ❌ Not supported in ONNX
```

This operation is **not supported in ONNX**, causing export to fail.

### Solution: HuggingFace Transformers Implementation

HuggingFace Transformers (v5.0.0+) includes an ONNX-compatible SAM3 implementation that pre-computes RoPE embeddings:

```python
# HuggingFace Transformers (transformers/models/sam3/modeling_sam3.py)
class Sam3ViTRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-compute and register as buffers (ONNX-compatible)
        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)
```

### Verified: Full ONNX Export Works

Using the `usls` export scripts with HuggingFace Transformers:

```bash
cd usls/scripts/sam3-image
python export_v2.py --all --model-path facebook/sam3 --output-dir /tmp/sam3-onnx

# Results:
# vision-encoder.onnx  1.8 GB  ✅ Verified with ONNX Runtime
# text-encoder.onnx    1.4 GB  ✅ Verified with ONNX Runtime
# decoder.onnx         124 MB  ✅ Verified with ONNX Runtime
```

### Implications for CVAT

| Mode | Current Implementation | Full ONNX Alternative |
|------|----------------------|----------------------|
| **Interactor** | PyTorch encoder + ONNX decoder | Full ONNX (usls export) |
| **Text-to-Segment** | Full PyTorch | Full ONNX possible |

For browser-side deployment or edge inference, the HuggingFace/usls ONNX models provide a complete solution.

---

## SAM3 Tracker Implementation Guide

SAM3 includes powerful video tracking capabilities through its memory-based architecture. This section describes how to implement SAM3 tracking for CVAT.

### Sam3TrackerVideo: Promptable Visual Segmentation on Videos

The `Sam3TrackerVideo` class performs **Promptable Visual Segmentation (PVS)** on videos:
- Takes interactive visual prompts (points, boxes, masks) on a **conditioning frame**
- Tracks the **specific object instance** across all video frames
- Uses **memory attention** to propagate information temporally

```python
from transformers import Sam3Model, Sam3Processor

# Load model
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")

# Process video frames
inputs = processor.images_to_sam3_tracker_video_inputs(video_frames)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Initialize with a prompt on frame 0
inference_session = model.init_sam3_tracker_video(inputs)

# Add prompts to conditioning frame
point_coords = torch.tensor([[[500, 375]]], device="cuda")  # [B, 1, 2]
point_labels = torch.tensor([[1]], device="cuda")  # 1 = positive

outputs_frame_0 = model(
    inference_session=inference_session,
    frame=inputs["pixel_values"][0],
    point_coords=point_coords,
    point_labels=point_labels,
)

# Propagate to all frames
video_segments = model.propagate_in_video(
    inference_session=inference_session,
    start_frame_idx=0,
    reverse=False,  # Forward propagation
)

# video_segments[frame_idx] contains {"masks": tensor, "scores": tensor}
```

### Memory Architecture

SAM3 Tracker uses a sophisticated **memory bank** for temporal propagation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SAM3 TRACKER MEMORY SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Frame 0 (Conditioning)    Frame 1           Frame 2           Frame N     │
│  ═══════════════════════   ═══════           ═══════           ═══════     │
│                                                                             │
│  ┌─────────────────┐      ┌─────────┐       ┌─────────┐       ┌─────────┐  │
│  │  User Prompt    │      │  Auto   │       │  Auto   │       │  Auto   │  │
│  │  (click/box)    │      │ Tracked │       │ Tracked │       │ Tracked │  │
│  └────────┬────────┘      └────┬────┘       └────┬────┘       └────┬────┘  │
│           │                    │                 │                 │        │
│           ▼                    ▼                 ▼                 ▼        │
│  ┌─────────────────┐      ┌─────────┐       ┌─────────┐       ┌─────────┐  │
│  │ Encode Memory   │──────│ Memory  │───────│ Memory  │───────│ Memory  │  │
│  │ (mask + feats)  │      │ Attn    │       │ Attn    │       │ Attn    │  │
│  └─────────────────┘      └─────────┘       └─────────┘       └─────────┘  │
│                                                                             │
│  Memory Bank: Up to 7 frames (1 conditioning + 6 recent)                   │
│  - maskmem_features: Spatial memory from predicted masks                    │
│  - obj_ptr: Object pointer tokens for cross-attention                       │
│  - maskmem_tpos_enc: Temporal position encoding                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Tracker Components

| Component | Purpose | Shape |
|-----------|---------|-------|
| `maskmem_backbone` | Encode mask + features into memory | SimpleMaskEncoder |
| `maskmem_features` | Spatial memory for attention | `[B, C, H, W]` |
| `maskmem_tpos_enc` | Temporal position encoding | `[num_maskmem, 1, 1, mem_dim]` |
| `obj_ptr` | Object pointer token | `[B, hidden_dim]` |
| `no_mem_embed` | "No memory" indicator | `[1, 1, hidden_dim]` |
| `no_obj_ptr` | "Object not present" indicator | `[1, hidden_dim]` |

### CVAT Tracker Interface

CVAT trackers follow a specific request/response protocol. Here's how SAM3 Tracker should implement it:

```python
# Request format (from CVAT)
{
    "image": "<base64-encoded-frame>",
    "shapes": [
        [xtl, ytl, xbr, ybr],  # Bounding box for each object
        ...
    ],
    "states": [
        {"<state_key>": "<encoded_value>", ...},  # Tracker state per object
        ...
    ]
}

# Response format (to CVAT)
{
    "shapes": [
        [xtl, ytl, xbr, ybr],  # Updated box or polygon points
        ...
    ],
    "states": [
        {"<state_key>": "<encoded_value>", ...},  # Updated state per object
        ...
    ]
}
```

### Implementing SAM3 Tracker for CVAT

```python
# model_handler_tracker.py

import torch
import jsonpickle
from sam3 import build_sam3_video_predictor

class SAM3TrackerHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = build_sam3_video_predictor("sam3.pt", device=self.device)

    def encode_state(self, inference_state):
        """Encode inference state for storage between frames."""
        state = {
            'obj_ids': jsonpickle.encode(inference_state.get('obj_ids', [])),
            'obj_ptr': jsonpickle.encode(inference_state.get('obj_ptr')),
            'maskmem_features': jsonpickle.encode(inference_state.get('maskmem_features')),
        }
        return state

    def decode_state(self, state):
        """Decode stored state for continuing tracking."""
        return {
            'obj_ids': jsonpickle.decode(state['obj_ids']),
            'obj_ptr': jsonpickle.decode(state['obj_ptr']),
            'maskmem_features': jsonpickle.decode(state['maskmem_features']),
        }

    def init_tracker(self, image, bbox):
        """Initialize tracking on first frame with bounding box."""
        xtl, ytl, xbr, ybr = bbox
        box = torch.tensor([[xtl, ytl, xbr, ybr]], device=self.device)

        inference_state = self.predictor.init_state(image)
        frame_idx, obj_ids, masks = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )
        return inference_state, masks[0]

    def track(self, image, state):
        """Track object in new frame using stored state."""
        inference_state = self.decode_state(state)

        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(inference_state):
            mask = masks[0] > 0
            if mask.any():
                ys, xs = torch.where(mask)
                bbox = [xs.min().item(), ys.min().item(),
                        xs.max().item(), ys.max().item()]
            else:
                bbox = None

        return bbox, self.encode_state(inference_state)

    def infer(self, image, shape, state):
        """Main entry point matching CVAT tracker interface."""
        if state is None:
            inference_state, mask = self.init_tracker(image, shape)
            bbox = self._mask_to_bbox(mask)
            state = self.encode_state(inference_state)
        else:
            bbox, state = self.track(image, state)

        return bbox, state
```

### SAM3 Video PCS: Text-Based Video Detection + Tracking

SAM3 also supports **text-based tracking** via the PCS Video mode:

```python
# Track "person" across entire video
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3")

# Process video
inputs = processor.images_to_sam3_video_inputs(video_frames)
inputs["input_text"] = [["person"]]  # Text prompt

# Run PCS on video - detects + tracks all people
outputs = model.forward_sam3_video(**inputs)

# outputs contains masks for all detected people in all frames
for frame_idx, frame_output in enumerate(outputs.masks):
    print(f"Frame {frame_idx}: {frame_output.shape[0]} people detected")
```

### Tracker Function YAML Template

```yaml
# serverless/pytorch/facebookresearch/sam3-tracker/nuclio/function-gpu.yaml
metadata:
  name: pth-facebookresearch-sam3-tracker
  namespace: cvat
  annotations:
    name: SAM3 Tracker
    version: 1
    type: tracker
    spec:
    help_message: |
      SAM3 Video Tracker for propagating masks across video frames.
      Initialize with a bounding box on the first frame.

spec:
  description: Video object tracking with SAM3 memory-based propagation
  runtime: 'python:3.10'
  handler: main:handler
  eventTimeout: 30s

  build:
    image: cvat.pth.facebookresearch.sam3.tracker:latest-gpu
    baseImage: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
    directives:
      preCopy:
        - kind: RUN
          value: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        - kind: RUN
          value: pip3 install git+https://github.com/facebookresearch/sam3.git
        - kind: RUN
          value: pip3 install jsonpickle numpy pillow

  triggers:
    myHttpTrigger:
      numWorkers: 2
      kind: 'http'

  resources:
    limits:
      nvidia.com/gpu: 1
```

### Comparison: SAM3 Tracker vs Existing CVAT Trackers

| Feature | SiamMask | TransT | SAM3 Tracker |
|---------|----------|--------|--------------|
| Output | Polygon | Box | Mask + Box |
| Occlusion handling | Limited | Limited | Object score |
| Re-detection | No | No | Memory attention |
| Multiple objects | Separate state | Separate state | Shared memory |
| Text prompts | No | No | Yes (PCS mode) |
| Speed (FPS) | ~55 | ~50 | ~44 |
| Model size | ~100MB | ~50MB | ~3.5GB |

---

## Summary: SAM3 Coverage of CVAT AI Tools

SAM3 is the first model to cover **all three CVAT AI tool categories** with a single architecture:

| Tool Type | SAM3 Mode | Implementation Status | Function Path |
|-----------|-----------|----------------------|---------------|
| **Interactor** | Sam3Tracker (PVS) | ✅ Complete | `onnx/.../sam3/` or `pytorch/.../sam3-unified/` |
| **Detector** | SAM3 PCS | ✅ Complete | `pytorch/.../sam3-pcs/` or `pytorch/.../sam3-unified/` |
| **Tracker** | Sam3TrackerVideo | 🔄 Planned | `pytorch/.../sam3-tracker/` |

### Recommended Deployment Strategy

For **memory-efficient** deployment with all three capabilities:

```bash
# Option 1: Unified function for interactor + detector (recommended)
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3-unified
# Plus separate tracker (video requires different state management)
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3-tracker

# Option 2: Separate functions (more isolation, higher VRAM)
./serverless/deploy_gpu.sh serverless/onnx/facebookresearch/sam3          # interactor
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3-pcs    # detector
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam3-tracker # tracker
```


---

## Session Notes: Key Discoveries & Resources

This section documents important discoveries, cleared misunderstandings, and reference resources from development sessions.

### Reference Implementations (Git Cloned)

The following repositories have been cloned locally for reference:

| Repository | Local Path | Purpose |
|------------|------------|---------|
| `facebookresearch/sam3` | `/home/jahs/GitHub/cvat/sam3/` | Official SAM3 PyTorch implementation (vision encoder can't export to ONNX) |
| `jamjamjon/usls` | `/home/jahs/GitHub/cvat/usls/` | Rust ONNX runtime with SAM3 export scripts (uses HuggingFace) |
| `mvaldi/cvat-yoloe-sam` | `/home/jahs/GitHub/cvat/cvat-yoloe-sam/` | Reference CVAT fork with YOLOE + SAM integration |
| `cvat-ai/cvat` | `/home/jahs/GitHub/cvat/` | Main CVAT repository (current workspace) |

**Note**: The `sam3/` and `usls/` directories are cloned **inside** the cvat workspace for convenience.

#### Repository Details

**facebookresearch/sam3** (Official SAM3):
```bash
cd /home/jahs/GitHub/cvat/sam3
git remote -v  # origin: https://github.com/facebookresearch/sam3
```
- Contains official PyTorch model code
- **Cannot export vision encoder to ONNX** due to `torch.view_as_complex()` in RoPE
- Use for reference and PyTorch-based inference only

**jamjamjon/usls** (Rust ONNX Runtime + Export Scripts):
```bash
cd /home/jahs/GitHub/cvat/usls
git remote -v  # origin: https://github.com/jamjamjon/usls.git
```
- Contains Python export scripts at `scripts/sam3-image/export_v2.py`
- Uses HuggingFace Transformers (which CAN export vision encoder)
- **Use this for ONNX export**

**mvaldi/cvat-yoloe-sam** (Reference Implementation):
```bash
cd /home/jahs/GitHub/cvat/cvat-yoloe-sam
git remote -v  # origin: https://github.com/mvaldi/cvat-yoloe-sam.git
                # upstream: https://github.com/cvat-ai/cvat.git
```
- CVAT fork with YOLOE + SAM integration example
- Useful reference for serverless function patterns
- Has `ai-models/` directory with model configurations

### Key Discovery: Two SAM3 Implementations

**Critical finding**: There are TWO different SAM3 implementations with different ONNX export capabilities:

| Implementation | Location | Vision Encoder ONNX | Why |
|----------------|----------|---------------------|-----|
| **Official Facebook** | `pip install git+https://github.com/facebookresearch/sam3.git` | ❌ **Cannot export** | Uses `torch.view_as_complex()` in RoPE |
| **HuggingFace Transformers** | `from transformers import Sam3Model` | ✅ **Can export** | Uses pre-computed RoPE buffers |

**The problematic code in official SAM3:**
```python
# sam3/model/image_encoder/image_encoder.py (Official Facebook)
freqs_cis = torch.view_as_complex(freqs_cis)  # ❌ ONNX doesn't support this
```

**The fixed code in HuggingFace:**
```python
# transformers/models/sam3/modeling_sam3.py (HuggingFace)
class Sam3ViTRotaryEmbedding(nn.Module):
    def __init__(self, config):
        # Pre-compute during __init__, no view_as_complex at forward time
        self.register_buffer("rope_embeddings_cos", inv_freq.cos())
        self.register_buffer("rope_embeddings_sin", inv_freq.sin())
```

### Working ONNX Export Method

**Use the `usls` export scripts with HuggingFace Transformers:**

```bash
cd ~/GitHub/usls/scripts/sam3-image
python export_v2.py --all --model-path facebook/sam3 --output-dir /tmp/sam3-onnx

# Successfully exports:
# - vision-encoder.onnx  (~1.8 GB)
# - text-encoder.onnx    (~1.4 GB)
# - decoder.onnx         (~124 MB)
```

**Prerequisites:**
- HuggingFace account with access to gated `facebook/sam3` model
- `huggingface-cli login` completed
- `pip install transformers>=5.0.0`

### Cleared Misunderstandings

#### 1. "SAM3 vision encoder can be exported to ONNX"
**Wrong**: Only HuggingFace Transformers version can. Official Facebook repo cannot due to `view_as_complex()`.

#### 2. "onnx-community/sam3-tracker-ONNX has all models"
**Partially correct**: It has vision encoder and decoder, but uses different input/output names than usls exports. Always verify tensor names match your code.

#### 3. "PCS decoder can be easily exported to ONNX"
**Wrong**: The PCS decoder is complex with:
- Tight coupling between encoder/decoder components
- Dynamic control flow for prompt types
- Geometry encoder that expects non-empty inputs during tracing

**Solution**: Keep PCS decoder in PyTorch on server. It's acceptable since text prompts aren't interactive like clicks.

#### 4. "SAM2 and SAM3 have the same architecture"
**Wrong**: Key differences:
| Parameter | SAM2 | SAM3 |
|-----------|------|------|
| Input image size | 1024×1024 | 1008×1008 |
| Backbone stride | 16 | 14 |
| Embedding size | 64×64 | 72×72 |
| Low-res mask size | 256×256 | 288×288 |
| Text encoder | None | CLIP (354M params) |

#### 5. "The 8 SAM3 components share parameters"
**Wrong**: All 8 components are **completely independent** with no parameter sharing:
- Vision Encoder (454M) - standalone
- Text Encoder (354M) - standalone
- Geometry Encoder (8M) - standalone
- DETR Encoder (10M) - standalone
- DETR Decoder (12M) - standalone
- Mask Decoder (2M) - standalone
- Scoring Head (1M) - standalone
- Box Head (~0.5M) - standalone

### Important File Locations

#### CVAT SAM3 Functions
```
serverless/
├── onnx/facebookresearch/sam3/nuclio/          # ONNX interactor
├── pytorch/facebookresearch/sam3/nuclio/       # Original PyTorch (legacy)
├── pytorch/facebookresearch/sam3-pcs/nuclio/   # Text-to-segment detector
├── pytorch/facebookresearch/sam3-unified/nuclio/ # Combined interactor+detector
└── pytorch/facebookresearch/sam3-tracker/nuclio/ # Video tracker (planned)
```

#### usls Export Scripts (inside cvat workspace)
```
/home/jahs/GitHub/cvat/usls/scripts/sam3-image/
├── export_v2.py          # Main export script (use this)
├── export.py             # Older version
└── README.md             # Documentation
```

#### Official SAM3 Model Code (inside cvat workspace)
```
/home/jahs/GitHub/cvat/sam3/sam3/
├── model/
│   ├── sam3_tracker_base.py    # Video tracking with memory
│   ├── sam1_task_predictor.py  # Image segmentation
│   ├── image_encoder/          # ViT backbone (has view_as_complex issue)
│   └── memory.py               # Memory encoder for tracking
└── sam/
    ├── mask_decoder.py         # SAM-style mask decoder
    └── prompt_encoder.py       # Point/box prompt encoder
```

#### Reference Implementation (cvat-yoloe-sam)
```
/home/jahs/GitHub/cvat/cvat-yoloe-sam/
├── ai-models/                  # Model configurations
├── serverless/                 # Nuclio function examples
├── cvat-ui/                    # UI modifications for YOLOE+SAM
└── zup.sh, zdown.sh            # Helper scripts for docker compose
```

### Conda Environment

For SAM3 development, use:
```bash
conda activate grimme-tf2.18  # Has transformers, torch, onnx, etc.
```

### HuggingFace Model Access

The `facebook/sam3` model is **gated**. To access:
1. Go to https://huggingface.co/facebook/sam3
2. Accept the license agreement
3. Run `huggingface-cli login` with your token

**Important**: Once models are exported to ONNX or baked into Docker images, no HuggingFace auth is needed at runtime.

### GitHub Issue Reference

- **GitHub Issue #224**: Discusses ONNX export challenges
- Key insight from issue: Use HuggingFace Transformers implementation for ONNX export

### API Differences: usls vs HuggingFace ONNX Models

The usls-exported models have different tensor names than onnx-community models:

**usls decoder inputs:**
```
input_points: [batch, 1, num_points, 2]
input_labels: [batch, 1, num_points]
input_boxes: [batch, num_boxes, 4]
image_embeddings.0: [batch, 32, 288, 288]
image_embeddings.1: [batch, 64, 144, 144]
image_embeddings.2: [batch, 256, 72, 72]
```

**usls decoder outputs:**
```
iou_scores: [batch, num_prompts, 3]
pred_masks: [batch, num_prompts, 3, 288, 288]
object_score_logits: [batch, num_prompts, 1]
```

### Version Information

Verified working versions as of February 2026:
- Python: 3.10+
- PyTorch: 2.x with CUDA 12.1
- Transformers: 5.0.0+
- ONNX: 1.14+
- ONNX Runtime: 1.16+

