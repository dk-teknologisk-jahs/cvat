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
// Decoder output mask size (from usls: 288x288 for the feature map)
const SAM3_MASK_SIZE = 256;

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
            const x1 = boxTopLeft.x * modelScale.scaleX;
            const y1 = boxTopLeft.y * modelScale.scaleY;
            const x2 = click.x * modelScale.scaleX;
            const y2 = click.y * modelScale.scaleY;
            boxes.push([x1, y1, x2, y2]);
            boxTopLeft = null;
        } else if (click.clickType === 0 || click.clickType === 1) {
            // Point: 0=negative, 1=positive
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
        'image_embeddings.0': emb0Tensor,
        'image_embeddings.1': emb1Tensor,
        'image_embeddings.2': emb2Tensor,
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
                                            console.log('SAM3 decoder supports mask refinement');
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
                                        decodeEmbedding(result.image_embeddings_0, result.image_embeddings_0_shape),
                                    );
                                    plugin.data.emb1Cache.set(
                                        key,
                                        decodeEmbedding(result.image_embeddings_1, result.image_embeddings_1_shape),
                                    );
                                    plugin.data.emb2Cache.set(
                                        key,
                                        decodeEmbedding(result.image_embeddings_2, result.image_embeddings_2_shape),
                                    );
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
                                        maskData: rawMaskData, maskH, maskW, xtl, ytl, xbr, ybr, lowResMaskData,
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

                                    // Debug: log mask stats
                                    const maskArray = Array.from(maskData);
                                    const positivePixels = maskArray.filter((v) => v > 0).length;
                                    console.log(`SAM3 decode result: maskH=${maskH}, maskW=${maskW}, ` +
                                        `positivePixels=${positivePixels}/${maskArray.length}, ` +
                                        `bounds=(${xtl.toFixed(3)},${ytl.toFixed(3)})-(${xbr.toFixed(3)},${ybr.toFixed(3)})`);

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

                                    // maskData is already a Float32Array (or array-like from postMessage)

                                    // Convert normalized bounds to pixel coordinates
                                    const pixelBounds: [number, number, number, number] = [
                                        Math.round(xtl * imWidth),
                                        Math.round(ytl * imHeight),
                                        Math.round(xbr * imWidth),
                                        Math.round(ybr * imHeight),
                                    ];

                                    // Ensure bounds are valid
                                    const [left, top, right, bottom] = pixelBounds;
                                    const cropW = right - left + 1;
                                    const cropH = bottom - top + 1;

                                    // Resize mask probabilities using bilinear interpolation, then threshold
                                    // This produces smooth edges (like usls does)
                                    const croppedMask = resizeMaskToCropBilinear(
                                        maskData,
                                        maskW,
                                        maskH,
                                        imWidth,
                                        imHeight,
                                        left,
                                        top,
                                        cropW,
                                        cropH,
                                    );

                                    // Debug: log final result
                                    const croppedPositive = croppedMask.flat().filter((v) => v > 0).length;
                                    console.log(`SAM3 final result: image=${imWidth}x${imHeight}, ` +
                                        `croppedMask=${croppedMask[0]?.length || 0}x${croppedMask.length}, ` +
                                        `positivePixels=${croppedPositive}/${cropW * cropH}, ` +
                                        `bounds=[${pixelBounds.join(',')}]`);

                                    plugin.data.lastClicks = clicks;

                                    const result = {
                                        mask: croppedMask,
                                        bounds: pixelBounds,
                                    };
                                    console.log('SAM3 resolving with:', result);
                                    resolve(result);
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
        modelID: 'pth-facebookresearch-sam3-tracker',
        // Use usls decoder (has trained weights) - mask refinement not yet supported
        modelURL: '/assets/tracker-prompt-encoder-mask-decoder.onnx',
        emb0Cache: new LRUCache({
            // [1, 32, 288, 288] float32 = ~38 MB per frame, max 8 frames = ~300 MB
            max: 8,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        emb1Cache: new LRUCache({
            // [1, 64, 144, 144] float32 = ~5.3 MB per frame
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
        // Mask refinement support
        supportsMaskInput: false,  // Detected at init time based on decoder model
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
 * Returns interpolated probability value.
 */
function bilinearSample(
    maskData: ArrayLike<number>,
    srcW: number,
    srcH: number,
    x: number,
    y: number,
): number {
    // Clamp coordinates
    const x0 = Math.max(0, Math.min(Math.floor(x), srcW - 1));
    const y0 = Math.max(0, Math.min(Math.floor(y), srcH - 1));
    const x1 = Math.min(x0 + 1, srcW - 1);
    const y1 = Math.min(y0 + 1, srcH - 1);

    // Fractional parts
    const xFrac = x - x0;
    const yFrac = y - y0;

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
 * Resize mask probabilities from decoder resolution to a cropped region using bilinear interpolation.
 * Thresholds AFTER interpolation for smooth edges (like usls does).
 */
function resizeMaskToCropBilinear(
    maskProbs: ArrayLike<number>,
    srcW: number,
    srcH: number,
    imageW: number,
    imageH: number,
    cropLeft: number,
    cropTop: number,
    cropW: number,
    cropH: number,
): number[][] {
    const result: number[][] = Array(cropH).fill(0).map(() => Array(cropW).fill(0));

    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            // Map crop coords to image coords
            const imgX = cropLeft + x;
            const imgY = cropTop + y;

            // Map image coords to source (decoder) coords - use floating point for interpolation
            const srcX = (imgX / imageW) * srcW;
            const srcY = (imgY / imageH) * srcH;

            // Bilinear interpolation of probability
            const prob = bilinearSample(maskProbs, srcW, srcH, srcX, srcY);

            // Threshold AFTER interpolation for smooth edges
            result[y][x] = prob > 0.5 ? 255 : 0;
        }
    }

    return result;
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
