# SAM3-Tracker Plugin for CVAT

Interactive segmentation using Segment Anything 3 (SAM3-Tracker) with encoder/decoder split architecture.

## Architecture

Similar to SAM2, this plugin uses a two-stage architecture:

1. **Server-side Vision Encoder** (Nuclio function)
   - Runs on GPU server using ONNX Runtime
   - Processes entire image once per frame
   - Returns image embeddings (cached for subsequent clicks)

2. **Browser-side Prompt Decoder** (ONNX Runtime Web)
   - Runs in user's browser
   - Takes embeddings + click coordinates
   - Returns segmentation mask instantly (~50-100ms)

## Prerequisites

### 1. ONNX Models

The plugin requires pre-exported ONNX models. These are exported using `export_hf_onnx.py`
(requires HuggingFace authentication at export time, but NOT at runtime).

**Server-side models** (deployed with Nuclio, auto-downloaded from GitHub release):
- `vision_encoder.onnx` (1.74 GB) - Vision transformer encoder
- `text_encoder.onnx` (1.31 GB) - CLIP text encoder for PCS mode
- `pcs_decoder.onnx` (123 MB) - Promptable Cascade Segmentation decoder
- `tracker_decoder.onnx` (22 MB) - Point-based tracker decoder
- `memory_attention.onnx` (155 MB) - Memory conditioning for video propagation
- `memory_encoder.onnx` (5.3 MB) - Encodes masks into memory features
- `object_pointer.onnx` (776 KB) - Extracts object pointer from decoder
- `temporal_pos_enc.npy` (2 KB) - Temporal position encoding table

**Browser-side decoder** (in `cvat-ui/plugins/sam3/assets/`):
- `tracker_decoder.onnx` (22MB) - Decoder with mask refinement support

The unified decoder accepts 256-channel FPN features at all levels (matching the HuggingFace export).

**Model Hosting**: All models are hosted on GitHub releases at:
`https://github.com/dk-teknologisk-jahs/cvat/releases/tag/sam3`

### 2. Enable the Plugin

Add `plugins/sam3` to `CLIENT_PLUGINS` when building the UI:

```bash
export CLIENT_PLUGINS="plugins/sam2:plugins/sam3"
```

Or modify `cvat-ui/webpack.config.js`:
```javascript
const defaultPlugins = ['plugins/sam', 'plugins/sam2', 'plugins/sam3'];
```

### 3. Deploy the Nuclio Function

Deploy the ONNX unified function (supports interactor + detector modes):

```bash
cd serverless

# GPU version (recommended)
nuctl deploy --project-name cvat \
  onnx/facebookresearch/sam3-unified/nuclio \
  --platform local \
  --build-code-entry-type sourceCode

# Or CPU version (slower but no CUDA required)
nuctl deploy --project-name cvat \
  onnx/facebookresearch/sam3-unified/nuclio \
  --platform local \
  --build-code-entry-type sourceCode \
  --config-file function.yaml
```

For text-to-segment (detector) mode, deploy with the detector YAML:
```bash
nuctl deploy --project-name cvat \
  onnx/facebookresearch/sam3-unified/nuclio \
  --platform local \
  --config-file function-gpu-detector.yaml
```

## Model Details

### SAM3-Tracker vs SAM2

| Feature | SAM2 | SAM3-Tracker |
|---------|------|--------------|
| Input resolution | 1024×1024 | 1008×1008 |
| Encoder channels | 32/64/256 | 256/256/256 (unified) |
| Decoder size | ~16MB | ~22MB (unified) |
| Mask refinement | ✅ Supported | ✅ Supported |
| Text prompts | ❌ | ✅ (PCS mode) |

### Tensor Shapes (Unified Format)

**Encoder outputs:**
- `fpn_feat_0`: [1, 256, 288, 288] (high-res features)
- `fpn_feat_1`: [1, 256, 144, 144] (mid-res features)
- `fpn_feat_2`: [1, 256, 72, 72] (low-res features)
- `fpn_pos_2`: [1, 256, 72, 72] (positional encoding)

**Decoder inputs:**
- `fpn_feat_0/1/2`: from encoder
- `point_coords`: [B, num_objects, num_points, 2]
- `point_labels`: [B, num_objects, num_points]
- `mask_input`: [B, 1, 288, 288] (optional, for refinement)
- `has_mask_input`: [B] (float, 1.0 if mask_input valid)

**Decoder outputs:**
- `masks`: [B, 3, 1008, 1008] (3 mask predictions)
- `iou_predictions`: [B, 3]
- `low_res_masks`: [B, 3, 288, 288]
- `object_score_logits`: [B, 1]

## Mask Refinement

The unified decoder supports **iterative mask refinement** - feeding the previous mask prediction back to improve results with subsequent clicks:

1. **First click**: Decoder uses zero mask (`has_mask_input=0`)
2. **Subsequent clicks**: Previous low-res mask (288×288) is fed back
3. The model refines predictions based on previous output + new clicks

The plugin automatically detects if the decoder supports mask input:
- If `mask_input` found in inputs → refinement enabled
- Otherwise → single-pass inference (still functional)

## Video Propagation (Tracker Mode)

SAM3 supports **server-side video propagation** using the CVAT tracker interface. Memory components run on the server with Redis-backed state management.

### Architecture

**Why Server-Side?**
- Memory attention model is 155MB - too large for browser download
- GPU acceleration via CUDA on server
- Redis enables persistent state across requests
- Follows existing CVAT tracker pattern (SiamMask, TransT)

### Memory Components (Server-Side ONNX)

These models are deployed with the Nuclio function (NOT in browser):
- `memory_attention.onnx` (155MB) - Conditions current frame with past memory
- `memory_encoder.onnx` (5.3MB) - Encodes masks into memory features
- `object_pointer.onnx` (776KB) - Extracts object pointer from decoder
- `temporal_pos_enc.npy` (2KB) - Temporal position encoding table

### How It Works

```
Frame 1: init_tracking(image, objects)
  └── Encode image → decode box prompt → get initial mask
  └── Encode mask into memory bank
  └── Store in Redis: {memory_bank, memory_pos_bank, frame_count}

Frame 2+: track_frame(session_id, image, frame_idx)
  └── Load state from Redis
  └── memory_attention.onnx → condition current features with memory
  └── tracker_decoder.onnx → predict mask
  └── memory_encoder.onnx → encode new mask to memory
  └── Update memory bank (FIFO, max 7 frames)
  └── Store updated state in Redis
```

### CVAT Tracker Interface

The tracker uses CVAT's standard tracker interface:
- **Request**: `{image, shapes, states}` where `states` contains `session_id`
- **Response**: `{shapes, states}` with updated masks and `session_id`
- State is stored server-side in Redis (memory bank ~18MB per object)

### Export Memory Components

```bash
cd serverless/pytorch/facebookresearch/sam3/nuclio
python export_sam3_memory_components.py

# Output in onnx-memory-exports/:
# - memory_attention.onnx (155MB, 106 weights)
# - memory_encoder.onnx (5MB, 40 weights)
# - object_pointer.onnx (776KB, 6 weights)
# - temporal_pos_enc.npy (2KB)
```

All exports use opset 17. Models are hosted on GitHub releases.

## Operation Modes

### 1. Interactive (Interactor Mode)

Standard point-and-click segmentation:
- Click points → server encodes → browser decodes mask
- Multiple positive/negative points supported
- Box prompts supported
- ~50-100ms per decode after first encode

### 2. PCS Detector (Text-to-Segment)

Text-guided object detection and segmentation:
- Enter text prompts describing objects to find
- Server runs full PCS pipeline (CLIP encoding → memory search → cascade decoder)
- Returns all matching masks in one shot

## Usage

1. In CVAT, select **"Segment Anything 3"** from the AI Tools
2. For **interactive mode**: Click on objects to segment, right-click for negative points
3. For **detector mode**: Enter text prompts (requires detector function deployed)

## Development

### Building

```bash
cd cvat-ui
CLIENT_PLUGINS="plugins/sam2:plugins/sam3" npm run build -- --env API_URL=/api
```

### Testing

```bash
CLIENT_PLUGINS="plugins/sam2:plugins/sam3" npm run start
```

### Exporting Models

The models are exported using the HuggingFace-based exporter:

```bash
cd serverless/onnx/facebookresearch/sam3-unified/nuclio
python export_hf_onnx.py --output /path/to/export/dir
```

This creates all 4 ONNX models needed for full functionality.

## Acknowledgments

- [Facebook Research SAM3](https://github.com/facebookresearch/sam3)
- [HuggingFace SAM3](https://huggingface.co/facebook/sam3.1-hiera-large)
- [usls](https://github.com/jamjamjon/usls) - Alternative ONNX exports
