# SAM3-Tracker Plugin for CVAT

Interactive segmentation using Segment Anything 3 (SAM3-Tracker) with encoder/decoder split architecture.

## Architecture

Similar to SAM2, this plugin uses a two-stage architecture:

1. **Server-side Vision Encoder** (Nuclio function)
   - Runs on GPU server
   - Processes entire image once per frame
   - Returns image embeddings (cached for subsequent clicks)

2. **Browser-side Prompt Decoder** (ONNX Runtime Web)
   - Runs in user's browser
   - Takes embeddings + click coordinates
   - Returns segmentation mask instantly (~50-100ms)

## Prerequisites

### 1. Download ONNX Models

**Vision Encoder** (for Nuclio function):

The encoder is downloaded automatically by `function-gpu.yaml` from HuggingFace. For manual download:
```bash
cd serverless/pytorch/facebookresearch/sam3/nuclio
# Full precision (~1.8GB)
curl -L -o tracker-vision-encoder.onnx \
  "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx"
curl -L -o tracker-vision-encoder.onnx_data \
  "https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/main/onnx/vision_encoder.onnx_data"
```

**Prompt Decoder** (for browser):

The recommended decoder is the custom export with mask refinement support (see "Mask Refinement Support" section below for how to generate it):
- `tracker-prompt-encoder-mask-decoder-with-mask-input.onnx` (17MB, full precision, mask refinement) **default**

Alternatively, you can download pre-built decoders from usls (without mask refinement):
```bash
cd cvat-ui/plugins/sam3/assets
curl -L -o tracker-prompt-encoder-mask-decoder.onnx \
  "https://github.com/jamjamjon/assets/releases/download/sam3/tracker-prompt-encoder-mask-decoder.onnx"
```

Usls decoder variants (no mask refinement):
- `tracker-prompt-encoder-mask-decoder.onnx` (21MB, full precision)
- `tracker-prompt-encoder-mask-decoder-q8.onnx` (10MB, int8 quantized)
- `tracker-prompt-encoder-mask-decoder-q4f16.onnx` (5MB, 4-bit quantized)

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

```bash
cd serverless
nuctl deploy --project-name cvat \
  pytorch/facebookresearch/sam3/nuclio \
  --platform local
```

## Model Details

### SAM3-Tracker vs SAM2

| Feature | SAM2 | SAM3-Tracker |
|---------|------|--------------|
| Input resolution | 1024×1024 | 1008×1008 |
| Encoder outputs | 3 tensors | 3 tensors (image_embeddings.0/1/2) |
| Decoder size | ~16MB | ~21MB (full precision) |
| Mask refinement | ✅ Supported | ✅ Supported (custom export required) |

### Tensor Shapes

**Encoder outputs:**
- `image_embeddings.0`: [1, 32, 288, 288]
- `image_embeddings.1`: [1, 64, 144, 144]
- `image_embeddings.2`: [1, 256, 72, 72]

**Decoder inputs:**
- `input_points`: [B, 1, N, 2]
- `input_labels`: [B, 1, N]
- `input_boxes`: [B, M, 4]
- `image_embeddings.0`, `image_embeddings.1`, `image_embeddings.2`: from encoder

**Decoder outputs:**
- `pred_masks`: [B, 1, 3, H, W] (3 mask predictions)
- `iou_scores`: [B, 1, 3]
- `object_score_logits`: [B, 1, 1]

## Mask Refinement Support

### Overview

Like SAM2, this plugin now supports **iterative mask refinement** - feeding the previous mask prediction back to the decoder to improve results with each click. This requires using a custom decoder export.

### Standard vs Custom Decoder

| Decoder | Mask Refinement | Notes |
|---------|-----------------|-------|
| `tracker-prompt-encoder-mask-decoder-with-mask-input.onnx` (custom) | ✅ Yes | **Default**, full SAM2 parity |
| `tracker-prompt-encoder-mask-decoder.onnx` (usls) | ❌ No | No mask refinement |

### Exporting the Custom Decoder

The standard ONNX exports (both [usls](https://github.com/jamjamjon/usls) and [HuggingFace onnx-community](https://huggingface.co/onnx-community/sam3-tracker-ONNX)) do **not** include `mask_input` and `has_mask_input` inputs needed for refinement.

To export a custom decoder with mask input support:

```bash
# Requires PyTorch 2.x and SAM3 dependencies
cd serverless/pytorch/facebookresearch/sam3/nuclio
pip install torch>=2.0 onnx onnxruntime
pip install git+https://github.com/facebookresearch/sam3.git

# Run the export script (auto-downloads pretrained weights from HuggingFace)
python export_decoder_with_mask_input.py \
  --checkpoint auto \
  --output ../../../../../../cvat-ui/plugins/sam3/assets/tracker-prompt-encoder-mask-decoder-with-mask-input.onnx
```

### How It Works

1. **First click**: Decoder uses zero mask (`has_mask_input=0`)
2. **Subsequent clicks**: Previous low-res mask (288×288) is fed back as input
3. This allows the model to refine predictions based on previous output

### Automatic Detection

The plugin **automatically detects** whether the loaded decoder supports mask input:
- If `mask_input` is found in model inputs → mask refinement enabled
- Otherwise → standard single-pass inference (still works, just no refinement)

## Known Limitations

### No Official ONNX Export

As of the time of writing, there is no official SAM3 ONNX export from Facebook Research.
- See [facebookresearch/sam3#224](https://github.com/facebookresearch/sam3/issues/224) requesting official ONNX export scripts
- This plugin uses community exports from [usls](https://github.com/jamjamjon/usls) or a custom export script

### Standard Decoder Lacks Mask Refinement

If using the standard usls/HuggingFace decoder (not the custom export):
- Each click recalculates the mask from scratch using only point/box prompts
- Multiple positive/negative clicks still work and improve results
- Segmentation is fast (~50-100ms per click) since embeddings are cached

**Solution:** Use the custom decoder export (see "Mask Refinement Support" above)

## Usage

1. Select the "Segment Anything 3" interactor in CVAT
2. Click on an object to segment
3. Add positive points (left-click) or negative points (right-click) to refine
4. Draw a bounding box for better initial segmentation

## Development

### Building

```bash
cd cvat-ui
npm run build -- --env API_URL=/api
```

### Testing

```bash
# Run the development server with SAM3 enabled
CLIENT_PLUGINS="plugins/sam2:plugins/sam3" npm run start
```

## Acknowledgments

- [Facebook Research SAM3](https://github.com/facebookresearch/sam3)
- [onnx-community](https://huggingface.co/onnx-community/sam3-tracker-ONNX) - HuggingFace ONNX models (vision encoder)
- [usls](https://github.com/jamjamjon/usls) - Alternative ONNX exports
