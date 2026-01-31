// Copyright (C) 2024 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * SAM3-Tracker Inference Worker
 *
 * Runs the prompt encoder + mask decoder ONNX model in the browser.
 * Receives embeddings from server, decodes masks locally for fast interaction.
 *
 * ONNX Model: tracker-prompt-encoder-mask-decoder-with-mask-input.onnx (17 MB)
 *
 * Inputs (standard decoder - usls export, no mask input support):
 *   - input_points: [batch, 1, num_points, 2] FLOAT32
 *   - input_labels: [batch, 1, num_points] INT64 (1=pos, 0=neg)
 *   - input_boxes: [batch, num_boxes, 4] FLOAT32 (xyxy scaled to 1008)
 *   - image_embeddings.0: [batch, 32, 288, 288] FLOAT32
 *   - image_embeddings.1: [batch, 64, 144, 144] FLOAT32
 *   - image_embeddings.2: [batch, 256, 72, 72] FLOAT32
 *
 * Inputs (custom decoder with mask input support):
 *   - image_embed: [batch, 256, 72, 72] FLOAT32
 *   - high_res_feats_0: [batch, 32, 288, 288] FLOAT32
 *   - high_res_feats_1: [batch, 64, 144, 144] FLOAT32
 *   - point_coords: [batch, num_points, 2] FLOAT32 (includes box corners as points)
 *   - point_labels: [batch, num_points] FLOAT32 (0=neg, 1=pos, -1=pad, 2=box TL, 3=box BR)
 *   - mask_input: [batch, 1, 288, 288] FLOAT32 (previous low-res mask logits)
 *   - has_mask_input: [batch] FLOAT32 (1.0 if mask_input is valid)
 *
 * Outputs:
 *   - iou_scores/iou_predictions: [batch, 2] or [batch, 3] FLOAT32
 *   - pred_masks/masks: [batch, 2, H, W] or [batch, 3, H, W] FLOAT32
 *   - low_res_masks: [batch, 2, 288, 288] or [batch, 3, 288, 288] FLOAT32
 *   - object_score_logits: [batch, 1] FLOAT32
 */

import { InferenceSession, env, Tensor } from 'onnxruntime-web';

let decoder: InferenceSession | null = null;
let supportsMaskInput = false; // Detected at init time based on model inputs

env.wasm.wasmPaths = '/assets/';

export enum WorkerAction {
    INIT = 'init',
    DECODE = 'decode',
}

export interface InitBody {
    decoderURL: string;
}

// Standard decoder interface (usls export)
export interface DecodeBody {
    input_points: Tensor;
    input_labels: Tensor;
    input_boxes: Tensor;
    'image_embeddings.0': Tensor;
    'image_embeddings.1': Tensor;
    'image_embeddings.2': Tensor;
    // Mask refinement inputs (optional, for custom decoder with mask support)
    mask_input?: Tensor;        // [1, 1, 288, 288] previous low-res mask logits
    has_mask_input?: boolean;   // Whether mask_input is valid
    readonly [name: string]: Tensor | boolean | undefined;
}

export interface DecodeResult {
    maskData: Float32Array;  // Raw mask data (not Tensor - for proper postMessage serialization)
    maskH: number;
    maskW: number;
    iouScore: number;
    objectScore: number;
    xtl: number;
    ytl: number;
    xbr: number;
    ybr: number;
    lowResMaskData?: Float32Array;  // Raw low-res mask data for mask refinement
}

export interface WorkerOutput {
    action: WorkerAction;
    error?: string;
    payload?: DecodeResult | { supportsMaskInput: boolean };
}

export interface WorkerInput {
    action: WorkerAction;
    payload: InitBody | DecodeBody;
}

const errorToMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === 'string') {
        return error;
    }

    console.error(error);
    return 'Unknown error, please check console';
};

// eslint-disable-next-line no-restricted-globals
if ((self as any).importScripts) {
    onmessage = (e: MessageEvent<WorkerInput>) => {
        if (e.data.action === WorkerAction.INIT) {
            if (decoder) {
                postMessage({
                    action: WorkerAction.INIT,
                    payload: { supportsMaskInput },
                });
                return;
            }

            const body = e.data.payload as InitBody;
            InferenceSession.create(body.decoderURL).then((decoderSession) => {
                decoder = decoderSession;

                // Detect if the model supports mask input
                const inputNames = decoder.inputNames;
                supportsMaskInput = inputNames.includes('mask_input') || inputNames.includes('has_mask_input');

                console.log(`SAM3 decoder loaded. Input names: ${inputNames.join(', ')}`);
                console.log(`Mask input support: ${supportsMaskInput}`);

                postMessage({
                    action: WorkerAction.INIT,
                    payload: { supportsMaskInput },
                });
            }).catch((error: unknown) => {
                postMessage({ action: WorkerAction.INIT, error: errorToMessage(error) });
            });
        } else if (!decoder) {
            postMessage({
                action: e.data.action,
                error: 'Worker was not initialized',
            });
        } else if (e.data.action === WorkerAction.DECODE) {
            const rawInputs = e.data.payload as DecodeBody;
            const hasMaskFromCaller = rawInputs.has_mask_input && rawInputs.mask_input;

            // Reconstruct Tensors from serialized data (postMessage serializes Tensors as plain objects)
            const reconstructTensor = (serialized: any, name: string): Tensor => {
                // If it's already a proper Tensor, return it
                if (serialized instanceof Tensor) {
                    console.log(`  ${name}: already Tensor, dims=${serialized.dims}`);
                    return serialized;
                }
                // Reconstruct from serialized data
                const data = serialized.data;
                const dims = serialized.dims as number[];
                const type = serialized.type || 'float32';

                // Convert data to proper TypedArray if needed
                let typedData: Float32Array | BigInt64Array;
                if (data instanceof Float32Array) {
                    console.log(`  ${name}: data is Float32Array, length=${data.length}, dims=${dims}`);
                    typedData = data;
                } else if (data instanceof BigInt64Array) {
                    console.log(`  ${name}: data is BigInt64Array, length=${data.length}, dims=${dims}`);
                    typedData = data;
                } else if (typeof data === 'object' && data !== null) {
                    // Convert object-like representation to Float32Array
                    console.log(`  ${name}: data is object, converting...`);
                    const values = Object.values(data) as number[];
                    console.log(`  ${name}: converted to array, length=${values.length}`);
                    typedData = new Float32Array(values);
                } else {
                    throw new Error(`Cannot reconstruct Tensor ${name}: unknown data format (${typeof data})`);
                }

                return new Tensor(type as 'float32', typedData, dims);
            };

            console.log('SAM3 worker: reconstructing tensors...');

            // Reconstruct all input tensors
            const inputs = {
                input_points: reconstructTensor(rawInputs.input_points, 'input_points'),
                input_labels: reconstructTensor(rawInputs.input_labels, 'input_labels'),
                input_boxes: reconstructTensor(rawInputs.input_boxes, 'input_boxes'),
                'image_embeddings.0': reconstructTensor(rawInputs['image_embeddings.0'], 'image_embeddings.0'),
                'image_embeddings.1': reconstructTensor(rawInputs['image_embeddings.1'], 'image_embeddings.1'),
                'image_embeddings.2': reconstructTensor(rawInputs['image_embeddings.2'], 'image_embeddings.2'),
                mask_input: rawInputs.mask_input ? reconstructTensor(rawInputs.mask_input, 'mask_input') : undefined,
                has_mask_input: rawInputs.has_mask_input,
            };

            console.log('SAM3 worker: tensors reconstructed');

            // Prepare inputs based on decoder type
            let runInputs: Record<string, Tensor>;

            if (supportsMaskInput) {
                // Custom decoder with mask input support
                // Remap input names from standard format to custom format
                const emb0 = inputs['image_embeddings.0'];
                const emb1 = inputs['image_embeddings.1'];
                const emb2 = inputs['image_embeddings.2'];

                // Create mask input tensors
                const maskInputData = hasMaskFromCaller
                    ? inputs.mask_input!
                    : new Tensor('float32', new Float32Array(1 * 1 * 288 * 288).fill(0), [1, 1, 288, 288]);
                const hasMaskInputData = new Tensor(
                    'float32',
                    new Float32Array([hasMaskFromCaller ? 1.0 : 0.0]),
                    [1],
                );

                // Get existing points from input_points [1, 1, N, 2] and input_labels [1, 1, N]
                const existingPointsData = inputs.input_points.data as Float32Array;
                const existingLabelsData = inputs.input_labels.data as Float32Array;
                const numExistingPoints = inputs.input_points.dims[2] as number;

                // Get boxes from input_boxes [1, num_boxes, 4]
                // For custom decoder, convert boxes to point pairs with labels 2 (top-left) and 3 (bottom-right)
                const boxesData = inputs.input_boxes?.data as Float32Array | undefined;
                const numBoxes = (inputs.input_boxes?.dims?.[1] as number) || 0;

                // Total points = existing points + 2 points per box (top-left, bottom-right)
                const totalPoints = numExistingPoints + numBoxes * 2;

                // Create combined arrays
                const allPointCoords = new Float32Array(totalPoints * 2);
                const allPointLabels = new Float32Array(totalPoints);

                // Copy existing points
                for (let i = 0; i < numExistingPoints; i++) {
                    allPointCoords[i * 2] = existingPointsData[i * 2];
                    allPointCoords[i * 2 + 1] = existingPointsData[i * 2 + 1];
                    allPointLabels[i] = existingLabelsData[i];
                }

                // Add box corners as points
                if (boxesData && numBoxes > 0) {
                    for (let i = 0; i < numBoxes; i++) {
                        const boxOffset = i * 4;
                        const pointOffset = numExistingPoints + i * 2;

                        // Top-left corner (label 2)
                        allPointCoords[pointOffset * 2] = boxesData[boxOffset];  // x1
                        allPointCoords[pointOffset * 2 + 1] = boxesData[boxOffset + 1];  // y1
                        allPointLabels[pointOffset] = 2;

                        // Bottom-right corner (label 3)
                        allPointCoords[(pointOffset + 1) * 2] = boxesData[boxOffset + 2];  // x2
                        allPointCoords[(pointOffset + 1) * 2 + 1] = boxesData[boxOffset + 3];  // y2
                        allPointLabels[pointOffset + 1] = 3;
                    }
                }

                runInputs = {
                    image_embed: emb2,  // [1, 256, 72, 72]
                    high_res_feats_0: emb0,  // [1, 32, 288, 288]
                    high_res_feats_1: emb1,  // [1, 64, 144, 144]
                    point_coords: new Tensor('float32', allPointCoords, [1, totalPoints, 2]),
                    point_labels: new Tensor('float32', allPointLabels, [1, totalPoints]),
                    mask_input: maskInputData,
                    has_mask_input: hasMaskInputData,
                };
            } else {
                // Standard decoder (usls export) - no mask input
                // The usls decoder expects input_labels as INT64, so convert from float32
                const labelsFloat = inputs.input_labels.data as Float32Array;
                const labelsInt64 = new BigInt64Array(labelsFloat.length);
                for (let i = 0; i < labelsFloat.length; i++) {
                    labelsInt64[i] = BigInt(Math.round(labelsFloat[i]));
                }

                runInputs = {
                    input_points: inputs.input_points,
                    input_labels: new Tensor('int64', labelsInt64, inputs.input_labels.dims as number[]),
                    input_boxes: inputs.input_boxes,
                    'image_embeddings.0': inputs['image_embeddings.0'],
                    'image_embeddings.1': inputs['image_embeddings.1'],
                    'image_embeddings.2': inputs['image_embeddings.2'],
                };
            }

            console.log('SAM3 worker: running decoder...');
            console.log('  Input names:', Object.keys(runInputs));

            decoder.run(runInputs).then((results) => {
                console.log('SAM3 worker: decoder returned');
                console.log('  Output names:', Object.keys(results));

                // Get outputs - handle both naming conventions
                const iouScores = results.iou_scores || results.iou_predictions;
                const predMasks = results.pred_masks || results.masks;
                const objectScoreLogits = results.object_score_logits || null;
                const lowResMasks = results.low_res_masks || null;  // For mask refinement

                if (!iouScores || !predMasks) {
                    throw new Error(`Missing outputs: iou_scores=${!!iouScores}, pred_masks=${!!predMasks}`);
                }

                console.log('  iou_scores dims:', iouScores.dims);
                console.log('  pred_masks dims:', predMasks.dims);

                // Find best mask (highest IoU score)
                const iouData = iouScores.data as Float32Array;
                let bestIdx = 0;
                let bestIou = iouData[0];
                for (let i = 1; i < iouData.length; i++) {
                    if (iouData[i] > bestIou) {
                        bestIou = iouData[i];
                        bestIdx = i;
                    }
                }

                console.log('  Best mask index:', bestIdx, 'IoU:', bestIou);

                // Get the best mask
                // pred_masks shape: [1, 1, 3, H, W] or [1, 3, H, W]
                const maskDims = predMasks.dims;
                let maskH: number;
                let maskW: number;
                let numMasks: number;
                let maskOffset: number;

                // Handle different possible tensor shapes
                if (maskDims.length === 5) {
                    // Shape: [1, 1, 3, H, W]
                    numMasks = maskDims[2] as number;
                    maskH = maskDims[3] as number;
                    maskW = maskDims[4] as number;
                    maskOffset = bestIdx * maskH * maskW;
                } else if (maskDims.length === 4) {
                    // Shape: [1, 3, H, W]
                    numMasks = maskDims[1] as number;
                    maskH = maskDims[2] as number;
                    maskW = maskDims[3] as number;
                    maskOffset = bestIdx * maskH * maskW;
                } else {
                    throw new Error(`Unexpected mask tensor shape: ${maskDims}`);
                }

                const maskData = predMasks.data as Float32Array;

                // Extract the best mask
                const maskSize = maskH * maskW;
                const bestMaskLogits = new Float32Array(maskSize);
                for (let i = 0; i < maskSize; i++) {
                    bestMaskLogits[i] = maskData[maskOffset + i];
                }

                // Log mask logits stats
                const logitsMin = Math.min(...bestMaskLogits);
                const logitsMax = Math.max(...bestMaskLogits);
                console.log(`  Mask logits: min=${logitsMin.toFixed(3)}, max=${logitsMax.toFixed(3)}`);

                // Apply sigmoid to get probabilities (NOT thresholding yet - that happens after upsampling)
                const maskProbs = new Float32Array(maskSize);
                for (let i = 0; i < maskSize; i++) {
                    const v = bestMaskLogits[i];
                    maskProbs[i] = 1.0 / (1.0 + Math.exp(-Math.max(-50, Math.min(50, v))));
                }

                // Calculate bounding box from probabilities (use 0.5 threshold for bounds)
                let xtl = maskW, ytl = maskH, xbr = 0, ybr = 0;
                let hasPositivePixels = false;
                for (let y = 0; y < maskH; y++) {
                    for (let x = 0; x < maskW; x++) {
                        if (maskProbs[y * maskW + x] > 0.5) {
                            hasPositivePixels = true;
                            xtl = Math.min(xtl, x);
                            ytl = Math.min(ytl, y);
                            xbr = Math.max(xbr, x);
                            ybr = Math.max(ybr, y);
                        }
                    }
                }

                const positiveCount = Array.from(maskProbs).filter((v) => v > 0.5).length;
                console.log(`  Mask probs: ${positiveCount}/${maskSize} above 0.5 (${(100*positiveCount/maskSize).toFixed(1)}%)`);
                // If no positive pixels, set bounds to full mask
                if (!hasPositivePixels) {
                    xtl = 0;
                    ytl = 0;
                    xbr = maskW - 1;
                    ybr = maskH - 1;
                }

                // Get object score (optional output)
                const objScore = objectScoreLogits
                    ? 1.0 / (1.0 + Math.exp(-(objectScoreLogits.data as Float32Array)[0]))
                    : 1.0;  // Default to 1.0 if not available

                // Note: Low-res mask extraction moved to the postMessage block below

                // Extract low-res mask data for mask refinement (if available)
                let lowResMaskData: Float32Array | undefined;
                if (lowResMasks && supportsMaskInput) {
                    // low_res_masks shape: [1, 3, 288, 288] or [1, 1, 3, 288, 288]
                    const lowResFullData = lowResMasks.data as Float32Array;
                    const lowResH = 288;
                    const lowResW = 288;
                    const lowResMaskSize = lowResH * lowResW;
                    lowResMaskData = new Float32Array(lowResMaskSize);
                    const lowResOffset = bestIdx * lowResMaskSize;
                    for (let i = 0; i < lowResMaskSize; i++) {
                        lowResMaskData[i] = lowResFullData[lowResOffset + i];
                    }
                }

                // Send mask probabilities (NOT binary) for smooth upsampling on main thread
                // Thresholding will happen AFTER bilinear interpolation for smooth edges
                postMessage({
                    action: WorkerAction.DECODE,
                    payload: {
                        maskData: maskProbs,  // Probabilities [0,1], not binary
                        maskH,
                        maskW,
                        iouScore: bestIou,
                        objectScore: objScore,
                        xtl: xtl / maskW,  // Normalized coordinates
                        ytl: ytl / maskH,
                        xbr: xbr / maskW,
                        ybr: ybr / maskH,
                        lowResMaskData,  // Raw Float32Array for mask refinement
                    } as DecodeResult,
                });
            }).catch((error: unknown) => {
                postMessage({ action: WorkerAction.DECODE, error: errorToMessage(error) });
            });
        }
    };
}
