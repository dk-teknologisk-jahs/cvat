// Copyright (C) 2024 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * SAM3-Tracker Plugin for CVAT
 *
 * Provides interactive segmentation using Segment Anything 3 (SAM3-Tracker).
 * Server runs the vision encoder, browser runs the prompt decoder.
 *
 * Architecture:
 * 1. Server: tracker-vision-encoder.onnx → returns embeddings
 * 2. Browser: tracker-prompt-encoder-mask-decoder-with-mask-input.onnx → decodes clicks to masks
 *
 * This enables fast interactive refinement (<100ms per click).
 *
 * Video Propagation:
 * Uses CVAT tracker interface - server-side with Redis state management.
 * Memory components (memory_attention, memory_encoder, object_pointer) run on server.
 */

import { Tensor } from 'onnxruntime-web';
import { LRUCache } from 'lru-cache';
import { CVATCore, MLModel, Job } from 'cvat-core-wrapper';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import { InitBody, DecodeBody, WorkerAction } from './inference.worker';

interface SAM3Plugin {
    name: string;
    description: string;
    cvat: {
        lambda: {
            call: {
                enter: (
                    plugin: SAM3Plugin,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<null | APIWrapperEnterOptions>;
                leave: (
                    plugin: SAM3Plugin,
                    result: object,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<any>;
            };
        };
        jobs: {
            get: {
                leave: (
                    plugin: SAM3Plugin,
                    results: any[],
                    query: { jobID?: number }
                ) => Promise<any>;
            };
        };
    };
    data: {
        initialized: boolean;
        worker: Worker;
        core: CVATCore | null;
        jobs: Record<number, Job>;
        modelID: string;
        modelURL: string;
        // SAM3 uses 3 embedding tensors (emb0, emb1, emb2)
        emb0Cache: LRUCache<string, Tensor>;
        emb1Cache: LRUCache<string, Tensor>;
        emb2Cache: LRUCache<string, Tensor>;
        lastClicks: ClickType[];
        // Mask refinement support
        supportsMaskInput: boolean;
        lowResMaskCache: LRUCache<string, Tensor>;  // For mask refinement
    };
    callbacks: {
        onStatusChange: ((status: string) => void) | null;
    };
}

interface ClickType {
    clickType: 0 | 1 | 2 | 3;  // 0=neg point, 1=pos point, 2=box top-left, 3=box bottom-right
    x: number;
    y: number;
}

// SAM3 uses 1008x1008 input resolution
const SAM3_IMAGE_SIZE = 1008;

function getModelScale(w: number, h: number) {
    const scaleX = SAM3_IMAGE_SIZE / w;
    const scaleY = SAM3_IMAGE_SIZE / h;
    return { scaleX, scaleY, width: w, height: h };
}

function prepareDecoderInputs(
    {
        clicks,
        emb0Tensor,
        emb1Tensor,
        emb2Tensor,
        modelScale,
        maskInput,
        hasMaskInput,
    }: {
        clicks: ClickType[];
        emb0Tensor: Tensor;
        emb1Tensor: Tensor;
        emb2Tensor: Tensor;
        modelScale: { width: number; height: number; scaleX: number; scaleY: number };
        maskInput?: Tensor;
        hasMaskInput?: boolean;
    },
): DecodeBody {
    // Separate points and boxes
    const points: { x: number; y: number; label: number }[] = [];
    const boxes: number[][] = [];

    let boxTopLeft: { x: number; y: number } | null = null;

    for (const click of clicks) {
        if (click.clickType === 2) {
            // Box top-left
            boxTopLeft = { x: click.x, y: click.y };
        } else if (click.clickType === 3 && boxTopLeft) {
            // Box bottom-right - complete the box
            // NOTE: The ONNX decoder includes the prompt encoder which internally
            // adds +0.5 to shift from pixel corner to pixel center.
            // We scale raw pixel coordinates - the model handles the shift.
            const x1 = boxTopLeft.x * modelScale.scaleX;
            const y1 = boxTopLeft.y * modelScale.scaleY;
            const x2 = click.x * modelScale.scaleX;
            const y2 = click.y * modelScale.scaleY;
            boxes.push([x1, y1, x2, y2]);
            boxTopLeft = null;
        } else if (click.clickType === 0 || click.clickType === 1) {
            // Point: 0=negative, 1=positive
            // NOTE: The ONNX decoder includes the prompt encoder which internally
            // adds +0.5 to shift from pixel corner to pixel center.
            // We scale raw pixel coordinates - the model handles the shift.
            points.push({
                x: click.x * modelScale.scaleX,
                y: click.y * modelScale.scaleY,
                label: click.clickType,
            });
        }
    }

    // Create point tensors
    // SAM3-Tracker format: [batch, 1, num_points, 2] for coords, [batch, 1, num_points] for labels
    const numPoints = points.length || 1;  // At least 1 point (dummy if needed)
    const pointCoords = new Float32Array(numPoints * 2);
    const pointLabels = new Float32Array(numPoints);

    if (points.length > 0) {
        for (let i = 0; i < points.length; i++) {
            pointCoords[i * 2] = points[i].x;
            pointCoords[i * 2 + 1] = points[i].y;
            pointLabels[i] = points[i].label;
        }
    } else {
        // Dummy point with label -1 (ignored)
        pointCoords[0] = 0;
        pointCoords[1] = 0;
        pointLabels[0] = -1;
    }

    // Create box tensor
    // SAM3-Tracker format: [batch, num_boxes, 4]
    const numBoxes = boxes.length || 0;
    const boxCoords = new Float32Array(Math.max(numBoxes, 0) * 4);
    for (let i = 0; i < boxes.length; i++) {
        boxCoords[i * 4] = boxes[i][0];
        boxCoords[i * 4 + 1] = boxes[i][1];
        boxCoords[i * 4 + 2] = boxes[i][2];
        boxCoords[i * 4 + 3] = boxes[i][3];
    }

    return {
        input_points: new Tensor('float32', pointCoords, [1, 1, numPoints, 2]),
        input_labels: new Tensor('float32', pointLabels, [1, 1, numPoints]),
        input_boxes: new Tensor('float32', boxCoords, [1, numBoxes, 4]),
        // Legacy decoder names (32/64/256ch usls export)
        'image_embeddings.0': emb0Tensor,
        'image_embeddings.1': emb1Tensor,
        'image_embeddings.2': emb2Tensor,
        // Unified decoder names (256ch HuggingFace export)
        // Worker will use whichever the decoder supports
        fpn_feat_0: emb0Tensor,
        fpn_feat_1: emb1Tensor,
        fpn_feat_2: emb2Tensor,
        // Mask refinement inputs (optional)
        mask_input: maskInput,
        has_mask_input: hasMaskInput,
    };
}

const sam3Plugin: SAM3Plugin = {
    name: 'Segment Anything 3',
    description: 'Handles SAM3-Tracker serverless function for interactive segmentation',
    cvat: {
        jobs: {
            get: {
                async leave(
                    plugin: SAM3Plugin,
                    results: any[],
                    query: { jobID?: number },
                ): Promise<any> {
                    if (typeof query.jobID === 'number') {
                        [plugin.data.jobs[query.jobID]] = results;
                    }
                    return results;
                },
            },
        },
        lambda: {
            call: {
                async enter(
                    plugin: SAM3Plugin,
                    taskID: number,
                    model: MLModel,
                    { frame }: { frame: number },
                ): Promise<null | APIWrapperEnterOptions> {
                    return new Promise((resolve, reject) => {
                        function resolvePromise(): void {
                            const key = `${taskID}_${frame}`;
                            const hasAllFeatures = (
                                plugin.data.emb0Cache.has(key) &&
                                plugin.data.emb1Cache.has(key) &&
                                plugin.data.emb2Cache.has(key)
                            );
                            if (hasAllFeatures) {
                                resolve({ preventMethodCall: true });
                            } else {
                                resolve(null);
                            }
                        }

                        if (model.id === plugin.data.modelID) {
                            if (!plugin.data.initialized) {
                                plugin.data.worker.postMessage({
                                    action: WorkerAction.INIT,
                                    payload: {
                                        decoderURL: plugin.data.modelURL,
                                    } as InitBody,
                                });

                                plugin.data.worker.onmessage = (e: MessageEvent) => {
                                    if (e.data.action !== WorkerAction.INIT) {
                                        reject(new Error(
                                            `Caught unexpected action response from worker: ${e.data.action}`,
                                        ));
                                    }

                                    if (!e.data.error) {
                                        plugin.data.initialized = true;
                                        // Check if decoder supports mask input
                                        if (e.data.payload?.supportsMaskInput) {
                                            plugin.data.supportsMaskInput = true;
                                        }
                                        resolvePromise();
                                    } else {
                                        reject(new Error(`SAM3 worker was not initialized. ${e.data.error}`));
                                    }
                                };
                            } else {
                                resolvePromise();
                            }
                        } else {
                            resolve(null);
                        }
                    });
                },

                async leave(
                    plugin: SAM3Plugin,
                    result: any,
                    taskID: number,
                    model: MLModel,
                    {
                        frame, pos_points, neg_points, obj_bbox,
                    }: {
                        frame: number;
                        pos_points: number[][];
                        neg_points: number[][];
                        obj_bbox: number[][];
                    },
                ): Promise<{
                    mask: number[][];
                    bounds: [number, number, number, number];
                }> {
                    return new Promise((resolve, reject) => {
                        if (model.id !== plugin.data.modelID) {
                            resolve(result);
                            return;
                        }

                        const job = Object.values(plugin.data.jobs).find((_job) => (
                            _job.taskId === taskID && frame >= _job.startFrame && frame <= _job.stopFrame
                        )) as Job;

                        if (!job) {
                            reject(new Error('Could not find a job corresponding to the request'));
                            return;
                        }

                        plugin.data.jobs = {
                            [job.id]: job,
                        };

                        job.frames.get(frame)
                            .then(({ height: imHeight, width: imWidth }: { height: number; width: number }) => {
                                const key = `${taskID}_${frame}`;

                                // Process server response if we have new embeddings
                                if (result) {
                                    // Decode base64 embeddings from server
                                    // Server returns: high_res_feats_0, high_res_feats_1, image_embed
                                    const decodeEmbedding = (base64: string, shape: number[]): Tensor => {
                                        const binaryStr = window.atob(base64);
                                        const bytes = new Uint8Array(binaryStr.length);
                                        for (let i = 0; i < binaryStr.length; i++) {
                                            bytes[i] = binaryStr.charCodeAt(i);
                                        }
                                        const floatArray = new Float32Array(bytes.buffer);
                                        return new Tensor('float32', floatArray, shape);
                                    };

                                    plugin.data.emb0Cache.set(
                                        key,
                                        decodeEmbedding(result.high_res_feats_0, result.high_res_feats_0_shape),
                                    );
                                    plugin.data.emb1Cache.set(
                                        key,
                                        decodeEmbedding(result.high_res_feats_1, result.high_res_feats_1_shape),
                                    );
                                    plugin.data.emb2Cache.set(
                                        key,
                                        decodeEmbedding(result.image_embed, result.image_embed_shape),
                                    );
                                } else {
                                    // Using cached embeddings
                                }

                                const modelScale = getModelScale(imWidth, imHeight);

                                // Build clicks array
                                const clicks: ClickType[] = [];

                                // Add bounding box clicks
                                if (obj_bbox.length) {
                                    clicks.push({ clickType: 2, x: obj_bbox[0][0], y: obj_bbox[0][1] });
                                    clicks.push({ clickType: 3, x: obj_bbox[1][0], y: obj_bbox[1][1] });
                                }

                                // Add positive points
                                pos_points.forEach((point) => {
                                    clicks.push({ clickType: 1, x: point[0], y: point[1] });
                                });

                                // Add negative points
                                neg_points.forEach((point) => {
                                    clicks.push({ clickType: 0, x: point[0], y: point[1] });
                                });

                                // Prepare decoder inputs
                                // Check if we should use mask refinement:
                                // 1. Decoder must support mask input
                                // 2. We must have a previous mask cached
                                // 3. Current clicks (minus the last) must match previous clicks
                                //    This ensures we only refine when adding a new point, not when
                                //    the user starts a new annotation
                                const isLowResMaskSuitable = JSON.stringify(clicks.slice(0, -1)) ===
                                    JSON.stringify(plugin.data.lastClicks);
                                const useMaskRefinement = (
                                    plugin.data.supportsMaskInput &&
                                    plugin.data.lowResMaskCache.has(key) &&
                                    isLowResMaskSuitable
                                );

                                const inputs = prepareDecoderInputs({
                                    clicks,
                                    emb0Tensor: plugin.data.emb0Cache.get(key) as Tensor,
                                    emb1Tensor: plugin.data.emb1Cache.get(key) as Tensor,
                                    emb2Tensor: plugin.data.emb2Cache.get(key) as Tensor,
                                    modelScale,
                                    maskInput: useMaskRefinement
                                        ? plugin.data.lowResMaskCache.get(key)
                                        : undefined,
                                    hasMaskInput: useMaskRefinement,
                                });

                                // Run decoder in worker
                                plugin.data.worker.postMessage({
                                    action: WorkerAction.DECODE,
                                    payload: inputs,
                                });

                                plugin.data.worker.onmessage = (e: MessageEvent) => {
                                    if (e.data.action !== WorkerAction.DECODE) {
                                        reject(new Error(
                                            `Caught unexpected action response from worker: ${e.data.action}`,
                                        ));
                                        return;
                                    }

                                    if (e.data.error) {
                                        reject(new Error(`Decoder error: ${e.data.error}`));
                                        return;
                                    }

                                    const {
                                        maskData: rawMaskData, maskH, maskW, lowResMaskData,
                                    } = e.data.payload;

                                    // Ensure maskData is array-like (postMessage may serialize differently)
                                    let maskData: ArrayLike<number>;
                                    if (rawMaskData instanceof Float32Array) {
                                        maskData = rawMaskData;
                                    } else if (Array.isArray(rawMaskData)) {
                                        maskData = rawMaskData;
                                    } else if (typeof rawMaskData === 'object' && rawMaskData !== null) {
                                        // Convert object-like {0: val, 1: val, ...} to array
                                        maskData = Object.values(rawMaskData) as number[];
                                    } else {
                                        reject(new Error(`Invalid maskData type: ${typeof rawMaskData}`));
                                        return;
                                    }

                                    // Store low-res mask for future refinement (if available)
                                    // Convert raw Float32Array back to Tensor for cache storage
                                    if (lowResMaskData && plugin.data.supportsMaskInput) {
                                        const lowResMaskTensor = new Tensor(
                                            'float32',
                                            new Float32Array(lowResMaskData),
                                            [1, 1, 288, 288],
                                        );
                                        plugin.data.lowResMaskCache.set(key, lowResMaskTensor);
                                    }

                                    // Following official SAM3 and usls implementations:
                                    // Compute bbox on low-res mask, then interpolate only within that region
                                    // This is MUCH faster for large images (97%+ fewer pixels to process)
                                    const { mask: croppedMask, bounds: pixelBounds } = resizeMaskWithBbox(
                                        maskData,
                                        maskW,
                                        maskH,
                                        imWidth,
                                        imHeight,
                                    );

                                    plugin.data.lastClicks = clicks;

                                    resolve({
                                        mask: croppedMask,
                                        bounds: pixelBounds,
                                    });
                                };

                                plugin.data.worker.onerror = (error) => {
                                    reject(error);
                                };
                            })
                            .catch(reject);
                    });
                },
            },
        },
    },
    data: {
        initialized: false,
        core: null,
        worker: new Worker(new URL('./inference.worker', import.meta.url)),
        jobs: {},
        // Use ONNX detector function (main function with all models)
        // The detector deploys first alphabetically, ensuring it's always ready
        // Interactor mode is handled via interactor-proxy, but plugin calls detector directly
        modelID: 'onnx-facebookresearch-sam3-detector',
        // Decoder matching the vision encoder from export_hf_onnx.py
        // Supports 256ch inputs and mask refinement
        modelURL: '/assets/tracker_decoder.onnx',
        emb0Cache: new LRUCache({
            // [1, 256, 288, 288] float32 = ~75 MB per frame, max 8 frames = ~600 MB
            max: 8,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        emb1Cache: new LRUCache({
            // [1, 256, 144, 144] float32 = ~19 MB per frame
            max: 8,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        emb2Cache: new LRUCache({
            // [1, 256, 72, 72] float32 = ~5.3 MB per frame
            max: 8,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lastClicks: [],
        // Mask refinement support (auto-detected at init time)
        supportsMaskInput: false,
        lowResMaskCache: new LRUCache({
            // [1, 1, 288, 288] float32 = ~330 KB per frame, max 8 frames
            max: 8,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
    },
    callbacks: {
        onStatusChange: null,
    },
};

/**
 * Bilinear interpolation for a single point in the source mask.
 * Handles out-of-bounds coordinates by clamping to edge values.
 * This matches PyTorch's grid_sample with padding_mode='border'.
 */
function bilinearSample(
    maskData: ArrayLike<number>,
    srcW: number,
    srcH: number,
    x: number,
    y: number,
): number {
    // Clamp coordinates to valid range first (border padding mode)
    // This ensures we get correct interpolation weights at boundaries
    const xClamped = Math.max(0, Math.min(x, srcW - 1));
    const yClamped = Math.max(0, Math.min(y, srcH - 1));

    // Get integer and fractional parts from clamped coordinates
    const x0 = Math.floor(xClamped);
    const y0 = Math.floor(yClamped);
    const xFrac = xClamped - x0;
    const yFrac = yClamped - y0;

    // Clamp neighbor coordinates (x0+1, y0+1 might exceed bounds)
    const x1 = Math.min(x0 + 1, srcW - 1);
    const y1 = Math.min(y0 + 1, srcH - 1);

    // Sample the 4 corners
    const v00 = maskData[y0 * srcW + x0];
    const v10 = maskData[y0 * srcW + x1];
    const v01 = maskData[y1 * srcW + x0];
    const v11 = maskData[y1 * srcW + x1];

    // Bilinear interpolation
    const v0 = v00 * (1 - xFrac) + v10 * xFrac;
    const v1 = v01 * (1 - xFrac) + v11 * xFrac;
    return v0 * (1 - yFrac) + v1 * yFrac;
}

/**
 * Resize mask LOGITS to full image size using bilinear interpolation,
 * then crop to the bounding box region.
 *
 * Key insight: We interpolate LOGITS (not probabilities), then threshold at 0.
 * This matches the official SAM3 code (F.interpolate on logits, then > 0).
 * Interpolating logits produces smoother edges because:
 * - Logits have larger dynamic range near boundaries (e.g., -5 to +5)
 * - Probabilities are squeezed (0.007 to 0.993), losing interpolation precision
 *
 * Uses align_corners=False coordinate mapping (matching PyTorch's F.interpolate):
 * src_coord = (dst_coord + 0.5) * (src_size / dst_size) - 0.5
 *
 * SAM2's decoder has upsampling built-in and outputs at target resolution,
 * while SAM3's decoder outputs fixed 288x288 requiring external upsampling.
 *
 * Optimized mask resizing: compute bbox on low-res mask, then only interpolate within that region.
 * This avoids interpolating millions of pixels for large images.
 */
function resizeMaskWithBbox(
    maskLogits: ArrayLike<number>,  // LOGITS, not probabilities - threshold at 0
    srcW: number,
    srcH: number,
    dstW: number,
    dstH: number,
): { mask: number[][]; bounds: [number, number, number, number] } {
    // ========================================================================
    // STEP 1: Post-process LOW-RES LOGITS (matching official SAM3)
    // ========================================================================
    // Official SAM3 post-processes masks BEFORE upsampling, on 288x288 logits.
    // From sam3/model/utils/sam1_utils.py:
    //   - Fill holes: small background components → set to +10.0
    //   - Remove sprinkles: small foreground components → set to -10.0
    //
    // Official SAM3 defaults (sam1_utils.py SAM2Transforms):
    //   - max_hole_area=0, max_sprinkle_area=0 for image mode (DISABLED)
    //
    // We match the official defaults: no post-processing for interactive annotation.
    // Set maxHoleArea/maxSprinkleArea > 0 if you want to enable this.
    const logitsCopy = new Float32Array(maskLogits.length);
    for (let i = 0; i < maskLogits.length; i++) {
        logitsCopy[i] = maskLogits[i] as number;
    }

    // Match official SAM3 image mode: no hole filling or sprinkle removal
    // To enable, change these values (e.g., 8 for both as used in some video configs)
    fillHolesAndRemoveSprinkles(logitsCopy, srcW, srcH, 0, 0);

    // ========================================================================
    // STEP 2: Find bounding box on low-res mask (fast - only 288x288 pixels)
    // ========================================================================
    let srcMinX = srcW;
    let srcMinY = srcH;
    let srcMaxX = -1;
    let srcMaxY = -1;

    for (let y = 0; y < srcH; y++) {
        for (let x = 0; x < srcW; x++) {
            const idx = y * srcW + x;
            if (logitsCopy[idx] > 0) {
                srcMinX = Math.min(srcMinX, x);
                srcMinY = Math.min(srcMinY, y);
                srcMaxX = Math.max(srcMaxX, x);
                srcMaxY = Math.max(srcMaxY, y);
            }
        }
    }

    // If no positive pixels, return empty mask
    if (srcMaxX < 0) {
        return {
            mask: [[0]],
            bounds: [0, 0, 0, 0],
        };
    }

    // ========================================================================
    // STEP 3: Convert low-res bbox to high-res coordinates
    // ========================================================================
    // Using align_corners=False inverse mapping: dst = (src + 0.5) / scale - 0.5
    const scaleX = srcW / dstW;
    const scaleY = srcH / dstH;

    // Map source bbox corners to destination coordinates
    // Add generous padding (equivalent to ~10 source pixels) to avoid edge artifacts
    const paddingSrc = 10;

    // Convert with padding
    let dstMinX = Math.floor((srcMinX - paddingSrc + 0.5) / scaleX - 0.5);
    let dstMinY = Math.floor((srcMinY - paddingSrc + 0.5) / scaleY - 0.5);
    let dstMaxX = Math.ceil((srcMaxX + paddingSrc + 0.5) / scaleX - 0.5);
    let dstMaxY = Math.ceil((srcMaxY + paddingSrc + 0.5) / scaleY - 0.5);

    // Clamp to image bounds
    dstMinX = Math.max(0, dstMinX);
    dstMinY = Math.max(0, dstMinY);
    dstMaxX = Math.min(dstW - 1, dstMaxX);
    dstMaxY = Math.min(dstH - 1, dstMaxY);

    const croppedW = dstMaxX - dstMinX + 1;
    const croppedH = dstMaxY - dstMinY + 1;

    // ========================================================================
    // STEP 4: Upsample to high-res and threshold (matching official SAM3)
    // ========================================================================
    // Official SAM3: F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
    // Then threshold at 0 (logit > 0 = foreground)
    const result: number[][] = Array(croppedH).fill(0).map(() => Array(croppedW).fill(0));

    for (let y = 0; y < croppedH; y++) {
        const dstY = y + dstMinY;
        for (let x = 0; x < croppedW; x++) {
            const dstX = x + dstMinX;

            // Map destination coords to source mask coords using align_corners=False
            const srcX = (dstX + 0.5) * scaleX - 0.5;
            const srcY = (dstY + 0.5) * scaleY - 0.5;

            // Bilinear interpolation of LOGITS (with holes already filled)
            const logit = bilinearSample(logitsCopy, srcW, srcH, srcX, srcY);

            // Threshold at 0 (equivalent to sigmoid > 0.5)
            result[y][x] = logit > 0 ? 255 : 0;
        }
    }

    return {
        mask: result,
        bounds: [dstMinX, dstMinY, dstMaxX, dstMaxY],
    };
}

/**
 * Fill small holes and remove small sprinkles in a LOGIT mask (before upsampling).
 *
 * This EXACTLY matches the official SAM3 implementation from sam3/model/utils/sam1_utils.py:
 *
 * For hole filling (max_hole_area > 0):
 * ```python
 * labels, areas = connected_components((mask_flat <= self.mask_threshold).to(torch.uint8))
 * is_hole = (labels > 0) & (areas <= self.max_hole_area)
 * masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)
 * ```
 *
 * For sprinkle removal (max_sprinkle_area > 0):
 * ```python
 * labels, areas = connected_components((mask_flat > self.mask_threshold).to(torch.uint8))
 * is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
 * masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
 * ```
 *
 * Key insight: `labels > 0` means ANY connected component except the largest one (label 0).
 * This is different from checking if a component touches the border.
 *
 * Official SAM3 defaults:
 *   - max_hole_area=0, max_sprinkle_area=0 for image mode (no post-processing)
 *   - max_hole_area=8, max_sprinkle_area=8 used in some video configs
 *
 * @param logits - Float32Array of mask logits (modified in place)
 * @param width - mask width
 * @param height - mask height
 * @param maxHoleArea - maximum hole area to fill (0 = disabled)
 * @param maxSprinkleArea - maximum sprinkle area to remove (0 = disabled)
 */
function fillHolesAndRemoveSprinkles(
    logits: Float32Array,
    width: number,
    height: number,
    maxHoleArea: number = 0,
    maxSprinkleArea: number = 0,
): void {
    if (width <= 2 || height <= 2) return;
    if (maxHoleArea <= 0 && maxSprinkleArea <= 0) return;  // Nothing to do

    const size = width * height;

    // Helper: Connected component labeling using union-find
    const findComponents = (isForeground: (idx: number) => boolean): { labels: Int32Array; areas: Int32Array } => {
        const parent = new Int32Array(size);
        const rank = new Int32Array(size);

        // Initialize
        for (let i = 0; i < size; i++) {
            parent[i] = i;
        }

        // Find with path compression
        const find = (x: number): number => {
            if (parent[x] !== x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };

        // Union by rank
        const union = (x: number, y: number): void => {
            const rootX = find(x);
            const rootY = find(y);
            if (rootX === rootY) return;
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        };

        // First pass: union adjacent pixels of the same class
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                if (!isForeground(idx)) continue;

                // Union with left neighbor
                if (x > 0 && isForeground(idx - 1)) {
                    union(idx, idx - 1);
                }
                // Union with top neighbor
                if (y > 0 && isForeground(idx - width)) {
                    union(idx, idx - width);
                }
            }
        }

        // Second pass: compute component areas
        const componentArea = new Map<number, number>();
        const labels = new Int32Array(size);

        for (let i = 0; i < size; i++) {
            if (!isForeground(i)) {
                labels[i] = -1;  // Not part of any component
                continue;
            }
            const root = find(i);
            labels[i] = root;
            componentArea.set(root, (componentArea.get(root) || 0) + 1);
        }

        // Find the largest component (this becomes "label 0" conceptually)
        let largestRoot = -1;
        let largestArea = 0;
        for (const [root, area] of componentArea) {
            if (area > largestArea) {
                largestArea = area;
                largestRoot = root;
            }
        }

        // Convert to dense labels (largest = 0, others = 1, 2, 3, ...)
        const rootToLabel = new Map<number, number>();
        let nextLabel = 1;
        rootToLabel.set(largestRoot, 0);  // Largest component gets label 0

        const finalLabels = new Int32Array(size);
        const areas = new Int32Array(size);

        for (let i = 0; i < size; i++) {
            if (labels[i] === -1) {
                finalLabels[i] = -1;
                areas[i] = 0;
                continue;
            }
            const root = labels[i];
            if (!rootToLabel.has(root)) {
                rootToLabel.set(root, nextLabel++);
            }
            finalLabels[i] = rootToLabel.get(root)!;
            areas[i] = componentArea.get(root) || 0;
        }

        return { labels: finalLabels, areas };
    };

    // Fill holes: small background components (mask <= 0)
    if (maxHoleArea > 0) {
        const { labels, areas } = findComponents((idx) => logits[idx] <= 0);
        for (let i = 0; i < size; i++) {
            // labels > 0 means it's NOT the largest background component
            // This matches: is_hole = (labels > 0) & (areas <= self.max_hole_area)
            if (labels[i] > 0 && areas[i] <= maxHoleArea) {
                logits[i] = 10.0;  // mask_threshold + 10.0 = 0 + 10 = 10
            }
        }
    }

    // Remove sprinkles: small foreground components (mask > 0)
    if (maxSprinkleArea > 0) {
        const { labels, areas } = findComponents((idx) => logits[idx] > 0);
        for (let i = 0; i < size; i++) {
            // labels > 0 means it's NOT the largest foreground component
            // This matches: is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            if (labels[i] > 0 && areas[i] <= maxSprinkleArea) {
                logits[i] = -10.0;  // mask_threshold - 10.0 = 0 - 10 = -10
            }
        }
    }
}

const builder: ComponentBuilder = ({ core }) => {
    sam3Plugin.data.core = core;
    core.plugins.register(sam3Plugin);

    return {
        name: sam3Plugin.name,
        destructor: () => {},
    };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } })
            .cvatUI.registerComponent(builder);
    }
}

window.addEventListener('plugins.ready', register, { once: true });
