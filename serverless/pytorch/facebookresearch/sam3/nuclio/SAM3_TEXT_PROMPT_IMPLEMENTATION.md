# SAM3 Text Prompt Support Implementation Plan

## Overview

This document tracks the implementation of text-to-segment (PCS mode) functionality for CVAT's SAM3 ONNX-based interactor.

## Constraints

**Key constraint**: The official SAM3 model on HuggingFace (`facebook/sam3`) is gated and requires permission from facebookresearch. Therefore:

1. **Development**: Use official PyTorch SAM3 with gated weights for development and verification
2. **Production**: Export standalone ONNX models that contain weights and don't require HuggingFace auth at runtime
3. **Verification**: Thoroughly test that ONNX exports match PyTorch outputs exactly
4. **No third-party ONNX**: Don't rely on random community ONNX exports - export our own so we can modify if needed

## Initial State

The CVAT SAM3 implementation supports:
- ✅ Point prompts (positive/negative clicks)
- ✅ Box prompts
- ✅ Mask refinement (iterative)
- ❌ Text prompts
- ❌ Combined text + geometry prompts
- ❌ Video tracking

## Target State

Add text-to-segment capability:
- Text prompts for concept-based segmentation (270K+ concepts)
- Combined text + box prompts for guided segmentation
- Multi-instance detection from text
- All using our own ONNX exports (not gated HuggingFace models)

## Architecture

### Required ONNX Models (Self-Exported)

1. **text_encoder.onnx** - Encodes text to embeddings
   - Input: `input_ids` [B, seq_len], `attention_mask` [B, seq_len]
   - Output: `text_features` [B, seq_len, 256], `text_mask` [B, seq_len]

2. **geometry_encoder.onnx** - Encodes box prompts with image context
   - Inputs: `boxes` [B, N, 4], `labels` [B, N], `fpn_feat_2`, `fpn_pos_2`
   - Outputs: `geo_features` [B, M, 256], `geo_mask` [B, M]

3. **pcs_decoder.onnx** - Transformer encoder/decoder + segmentation head
   - Inputs: fpn_feats (3 levels), prompt_features, prompt_mask
   - Outputs: masks, boxes, logits, presence

4. **Tokenizer files** - For text preprocessing (no model weights needed)
   - `vocab.json`, `merges.txt`, `byte_encoder.json`, `tokenizer_config.json`

### Implementation Steps

#### Phase 1: Model Export

1. [x] Create text encoder wrapper for ONNX export
2. [x] Export and verify text_encoder.onnx (1348 MB)
3. [x] Export tokenizer files
4. [ ] Create geometry encoder wrapper for ONNX export
5. [ ] Create PCS decoder wrapper for ONNX export
6. [ ] Verify all ONNX outputs match PyTorch exactly

#### Phase 2: Server-Side Handler (Interim PyTorch)

1. [x] Create `model_handler_pcs.py` with PyTorch SAM3 processor
2. [x] Create `main_pcs.py` Nuclio handler for text-to-segment endpoint
3. [x] Test with real images (bus.jpg: found 4 people + 1 bus)

#### Phase 3: ONNX-Only Handler

1. [ ] Create ONNX-based text-to-segment handler (no PyTorch/HuggingFace)
2. [ ] Verify ONNX handler matches PyTorch handler outputs
3. [ ] Performance comparison

#### Phase 4: Integration

1. [ ] Update function.yaml with ONNX model paths
2. [ ] Create text-to-segment API endpoint
3. [ ] End-to-end testing

## Reference Implementations

- **Official SAM3**: `sam3/sam3/model/` - PyTorch implementation (for verification)
- **usls**: `usls/src/models/vlm/sam3_image/` - Rust ONNX implementation (architecture reference)

## File Locations

- Export scripts: `serverless/pytorch/facebookresearch/sam3/nuclio/`
- Model handler: `serverless/pytorch/facebookresearch/sam3/nuclio/model_handler.py`
- Official SAM3: `sam3/sam3/`

---

## Progress Log

### 2026-02-01 - Session Start

- Created implementation plan document
- Initial analysis complete
- Starting Phase 1: ONNX model export
- Next: Explore SAM3 text encoder architecture and create export wrapper

### 2026-02-01 - Text Encoder Export Complete

- Created `export_text_encoder.py` script
- Fixed checkpoint prefix issue (was `backbone.language_backbone.`, should be `detector.backbone.language_backbone.`)
- Successfully exported `text_encoder.onnx` (1348 MB)
- Verified ONNX vs PyTorch: MAE < 0.000003 for all test texts
- Exported tokenizer files:
  - `vocab.json` (49408 tokens)
  - `merges.txt` (48894 BPE merges)
  - `byte_encoder.json`
  - `tokenizer_config.json`
- Next: Create PCS decoder export script for geometry encoder + transformer + segmentation head

### 2026-02-01 - Decision: Use Official SAM3 Repo Only

- Found usls has export scripts using HuggingFace transformers
- Decision: Stick with official SAM3 repo (sam3/) for consistency
- Continue PCS decoder export using official repo components:
  - `sam3.model.geometry_encoders.SequenceGeometryEncoder`
  - `sam3.model.encoder.TransformerEncoderFusion`
  - `sam3.model.decoder.TransformerDecoder`
  - `sam3.model.maskformer_segmentation.UniversalSegmentationHead`
- Next: Analyze PCS decoder data flow and create export wrapper

### 2026-02-01 - Architecture Analysis Complete

**SAM3 Architecture for PCS (Text-to-Segment):**

Verified data flow through processor:
1. **Vision encoder** (`set_image`):
   - Input: Image [B, 3, 1008, 1008]
   - Output: `backbone_fpn` (3 scales: 288×288, 144×144, 72×72), `vision_pos_enc`, `vision_features`

2. **Text encoder** (`forward_text`):
   - Input: Text strings
   - Output: `language_features` [32, B, 256], `language_mask` [B, 32], `language_embeds` [32, B, 1024]

3. **PCS decoder** (`forward_grounding`):
   - Complex multi-stage pipeline:
     - `_encode_prompt`: Geometry encoder + concatenate text/geometry features
     - `_run_encoder`: TransformerEncoderFusion (fuses image + prompt features)
     - `_run_decoder`: TransformerDecoder (generates 200 object queries)
     - `_run_segmentation_heads`: Produces masks, boxes, logits

**Decision: Server-Side PyTorch for PCS**

The PCS decoder is too complex to export cleanly to ONNX due to:
- Tight coupling between encoder/decoder components
- Dynamic control flow for prompt types
- Complex attention patterns with geometric prompts

**Revised Approach:**
1. ✅ Export text_encoder.onnx (done - 1348 MB)
2. 🔄 Keep PCS decoder in PyTorch on server
3. 📝 Create new server endpoint `/text-to-segment` that:
   - Accepts: image + text prompt(s)
   - Returns: masks + boxes + confidence scores
4. 📝 Browser sends text prompt to server, receives segmentation results

This is simpler than exporting the PCS decoder and matches the official processor API.

**Next Steps:**
1. Create `model_handler_pcs.py` for text-to-segment on server
2. Add `/text-to-segment` endpoint to `main.py`
3. Test end-to-end with real images

### 2026-02-01 - Server-Side PCS Handler Complete

**Created:**
- `model_handler_pcs.py`: Server-side handler using PyTorch SAM3 processor
  - `ModelHandlerPCS` class with:
    - `set_image()`: Cache vision features for reuse
    - `text_to_segment()`: Text prompt → masks/boxes/scores
    - `text_and_box_to_segment()`: Combined text + box guidance
  - Handles multiple text prompts in one call
  - Returns pixel-coordinate boxes (verified correct)

- `main_pcs.py`: Nuclio function handler for text-to-segment endpoint
  - Input: image (base64) + text_prompts + optional confidence_threshold + optional box
  - Output: detections with RLE-encoded masks, boxes, scores, labels
  - Uses pycocotools for efficient RLE mask encoding

**Test Results (bus.jpg, 810×1080):**
- Found 4 people (scores: 0.97, 0.97, 0.96, 0.94)
- Found 1 bus (score: 0.94)
- Latency: ~2.77s (includes model warm-up)

**Architecture Decision:**
Kept PCS decoder in PyTorch on server rather than exporting to ONNX because:
1. Complex multi-stage pipeline (geometry encoder → transformer encoder → transformer decoder → segmentation head)
2. Tight coupling between components with shared state
3. Dynamic control flow for different prompt types
4. Server-side execution is acceptable for text prompts (not interactive like clicks)

**Next Steps:**
1. Update function.yaml to support both endpoints (image-encode and text-to-segment)
2. Add frontend support for text prompt input
3. Performance optimization (batch inference, caching)
4. Integration testing with CVAT

### 2026-02-01 - ONNX Text Encoder Verification Complete

**Created test suite: `test_text_encoder.py`**

Comprehensive test suite that verifies ONNX text encoder matches PyTorch exactly:

**Test Results (43 tests, 100% pass rate):**
```
Total Tests: 43
Passed: 43
Failed: 0
Pass Rate: 100.0%

Statistics across all tests:
  MAE:     mean=0.00000246, max=0.00000288 (threshold: 0.0001)
  MaxDiff: mean=0.00001559, max=0.00002068 (threshold: 0.001)
  Corr:    mean=1.00000000, min=1.00000000 (threshold: 0.99999)
```

**Test categories:**
- Simple objects ("a person", "a car", "a dog", etc.)
- Descriptive phrases ("a red car", "a tall building", etc.)
- Complex phrases ("a person wearing a red shirt", "a dog running in the park", etc.)
- Edge cases (empty string, single char, uppercase, punctuation, whitespace)
- Long prompts (truncated to 32 token limit)
- Special concepts ("background", "foreground", "segment", etc.)

**BPE Tokenizer fix:**
The tokenizer had a bug where `</w>` suffix was added AFTER BPE processing instead of BEFORE.
Fixed `bpe()` function to match SimpleTokenizer: `word = tuple(token[:-1]) + (token[-1] + "</w>",)`

**Key verification:**
- Both PyTorch and ONNX use identical tokenization (our exported tokenizer files)
- Same input tensor → same output tensor (within float32 precision)
- Correlation is 1.0 for all tests

**Status:**
✅ Text encoder ONNX export verified
✅ Tokenizer implementation verified
✅ Ready for production use

**Next Steps:**
1. Complete ONNX-only text-to-segment handler (currently uses PyTorch for PCS decoder)
2. Add text prompt UI to CVAT frontend
3. Integration testing with CVAT backend

### 2026-02-01 - CVAT Integration Complete

**Added text prompt support to CVAT UI and backend:**

1. **Backend (views.py)**:
   - Added `supports_text_prompt` annotation parsing
   - Added `text_prompts` parameter support for detectors
   - Skip label mapping for text-based detectors (labels come from prompts)
   - Pass `threshold` parameter to function

2. **Frontend (detector-runner.tsx)**:
   - Added text prompt input for models with `supportsTextPrompt: true`
   - Text prompt split by commas into array of prompts
   - Shows label mapper OR text prompt based on model type

3. **Model (ml-model.ts)**:
   - Added `supportsTextPrompt` property getter

4. **Function config (function_pcs.yaml)**:
   - Added `supports_text_prompt: true` annotation
   - Type set to `detector` to use detector runner UI

5. **Handler (main_pcs.py)**:
   - Updated to return CVAT detector format: flat array of DetectedShape
   - Mask format: `[rle_counts..., xtl, ytl, xbr, ybr]`
   - Implemented `mask_to_cvat_rle()` function for proper mask encoding

**How it works:**
1. User selects SAM3 Text-to-Segment model in detector runner
2. User enters comma-separated text prompts (e.g., "person, car, dog")
3. Click "Annotate" sends request with `text_prompts` array
4. Backend passes image + text_prompts to Nuclio function
5. Function returns detected masks with labels from prompts
6. CVAT creates mask annotations

**Files changed:**
- `cvat/apps/lambda_manager/views.py` - Text prompt handling
- `cvat-core/src/core-types.ts` - SerializedModel interface
- `cvat-core/src/ml-model.ts` - supportsTextPrompt getter
- `cvat-ui/src/components/model-runner-modal/detector-runner.tsx` - Text input UI
- `cvat-ui/src/components/model-runner-modal/styles.scss` - Styling
- `serverless/.../function_pcs.yaml` - Function config
- `serverless/.../main_pcs.py` - CVAT detector format

**Status:**
✅ Backend support for text prompts
✅ Frontend text prompt UI
✅ CVAT detector response format
✅ Local handler test verified (5 detections on bus.jpg)
🔄 Integration testing needed

### Next Steps for Integration Testing

1. **Deploy the Nuclio function:**
   ```bash
   nuctl deploy --project-name cvat \
     --path /home/jahs/GitHub/cvat/serverless/pytorch/facebookresearch/sam3/nuclio \
     --file function_pcs.yaml \
     --platform local
   ```

2. **Build CVAT with changes:**
   ```bash
   # Frontend
   cd cvat-ui && yarn build

   # Backend has Python changes - restart server
   ```

3. **Test in CVAT:**
   - Open a task with images
   - Go to AI Tools → Detectors
   - Select "SAM3 Text-to-Segment" model
   - Enter text prompts like "person, car"
   - Click Annotate
   - Verify masks are created

4. **Verify labels:**
   - Masks should have labels matching the text prompts
   - For new labels, CVAT may need to create them or warn

### 2026-02-01 - ONNX Export Analysis

**Goal:** Export all SAM3 components to ONNX to avoid gated HuggingFace weights at runtime.

**Components analyzed:**

| Component | Size | Status | Notes |
|-----------|------|--------|-------|
| vision_encoder.onnx | ~1.7 GB | ✅ Available | onnx-community/sam3-tracker-ONNX |
| text_encoder.onnx | 1348 MB | ✅ Exported | Our own export, verified |
| pcs_decoder.onnx | 124 MB | ⚠️ Partial | Export succeeds but runtime issues |

**PCS Decoder Export Issue:**

Successfully exported `pcs_decoder.onnx` (124 MB), but ONNX runtime fails because:
- Empty geometry prompt `[0, B, 4]` is baked in as constant during tracing
- Later reshape operations expect non-zero dimensions

**Key Architecture Insight:**

SAM3 uses `num_feature_levels=1`, meaning:
- Transformer encoder ONLY uses the LAST FPN level (72×72)
- But segmentation head uses ALL FPN levels (288², 144², 72²) for upsampling

```
backbone_fpn[3 levels] ──┬──> [last level only] ──> Transformer Encoder ──> Transformer Decoder
                        │
                        └──> [all levels] ─────────────────────────────────> Segmentation Head
```

**Options to fix ONNX:**

1. **Text-only wrapper** - Skip geometry encoder entirely (loses box guidance)
2. **Dummy geometry** - Trace with valid dummy box, mask at inference
3. **Component export** - Export transformer/seghead separately
4. **Unified PyTorch** - Single function handles both modes

**Recommended approach: Unified PyTorch Handler**

Instead of duplicating model loading in two Nuclio functions, create ONE function that:
1. Loads SAM3 model once (with all components)
2. Handles `/encode` for interactor mode (returns embeddings for browser decode)
3. Handles `/text-to-segment` for detector mode (returns masks directly)
4. Gated HuggingFace weights only needed at Docker BUILD time (cached in image)

This approach:
- ✅ Avoids duplicate GPU memory usage
- ✅ Single deployment to manage
- ✅ Weights embedded in Docker image (no runtime auth needed)
- ✅ Can still use ONNX vision encoder for interactor mode

### 2026-02-XX - Key Discovery: HuggingFace vs Official SAM3

**Goal:** Investigate GitHub issue #224 to understand how existing encoder ONNX models were created.

**Key Finding:** There are TWO SAM3 implementations with different ONNX capabilities:

| Implementation | RoPE Method | Vision Encoder ONNX |
|----------------|-------------|---------------------|
| Official Facebook (`sam3/`) | `view_as_complex()` | ❌ Not supported |
| HuggingFace Transformers | Pre-computed buffers | ✅ Exportable |

**Why Official SAM3 Can't Export Vision Encoder:**
```python
# sam3/model/image_encoder/image_encoder.py
freqs_cis = torch.view_as_complex(freqs_cis)  # ❌ ONNX error
```

**Why HuggingFace Can Export:**
```python
# transformers/models/sam3/modeling_sam3.py
class Sam3ViTRotaryEmbedding(nn.Module):
    def __init__(self, config):
        # Pre-compute during init, no view_as_complex at runtime
        self.register_buffer("rope_embeddings_cos", inv_freq.cos())
        self.register_buffer("rope_embeddings_sin", inv_freq.sin())
```

**Verified Working Export (usls scripts + HuggingFace):**
```bash
cd usls/scripts/sam3-image
python export_v2.py --all --model-path facebook/sam3 --output-dir /tmp/sam3-onnx

# Successfully exported:
# vision-encoder.onnx  1.8 GB  ✅ Verified with ONNX Runtime
# text-encoder.onnx    1.4 GB  ✅ Verified with ONNX Runtime
# decoder.onnx         124 MB  ✅ Verified with ONNX Runtime
```

**SAM3 Architecture (All Encoders Independent):**
```
┌─────────────────────────────────────────────────────────────┐
│                      SAM3 (~840M params)                    │
├─────────────────────────────────────────────────────────────┤
│  Vision Encoder (454M)  ←─ independent ─→  Text Encoder (354M)  │
│         │                                        │          │
│         └────────────────┬───────────────────────┘          │
│                          ▼                                  │
│  [Optional Geometry Encoder (8M) for box prompts]           │
│                          ▼                                  │
│              DETR Encoder (10M) ← Cross-attention           │
│                          ▼                                  │
│              DETR Decoder (12M) ← Object queries            │
│                          ▼                                  │
│              Mask Decoder (2M) + Scoring (1M)               │
└─────────────────────────────────────────────────────────────┘
```

**Implications for CVAT:**

| Mode | Current | Alternative |
|------|---------|-------------|
| **Interactor** (clicks/boxes) | PyTorch encoder + ONNX decoder | Full ONNX (HuggingFace export) |
| **Text-to-Segment** | Full PyTorch on server | Full ONNX possible |

**Recommendations:**
1. For server-side inference: Continue using official SAM3 repo (better documented, maintained)
2. For browser/edge deployment: Use usls export scripts with HuggingFace Transformers
3. Document both approaches in `SAM3_ONNX_EXPORT_GUIDE.md`
┌─────────────────────────────────────────────────────────────┐
│                    Nuclio Function                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ main_pcs.py + model_handler_pcs.py                  │    │
│  │ - Loads SAM3 model with PyTorch                     │    │
│  │ - Runs text-to-segment pipeline                     │    │
│  │ - Returns: [{ type: "mask", label, mask: RLE }]     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```
