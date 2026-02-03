#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Tracker Proxy

Lightweight proxy that forwards tracker requests to the SAM3 detector function.
This allows registering SAM3 as a tracker in CVAT without loading models twice.

No models are loaded here - just HTTP forwarding.
Memory usage: ~50MB (Python + requests only)

IMPORTANT: The SAM3 detector function must be deployed and running first!
This proxy will return 503 errors if the detector is not available.
"""

import json
import os
import time
import requests

# Target detector function URL (the main SAM3 function with models loaded)
# In Docker/nuclio, functions are accessible via container name
DETECTOR_FUNCTION_URL = os.environ.get(
    "SAM3_DETECTOR_URL",
    "http://nuclio-nuclio-onnx-facebookresearch-sam3-detector:8080"
)

TIMEOUT = int(os.environ.get("SAM3_PROXY_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("SAM3_PROXY_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("SAM3_PROXY_RETRY_DELAY", "1.0"))


def init_context(context):
    """
    Initialize the proxy.

    NOTE: We do NOT check if the detector is running here because:
    1. Nuclio requires init_context to complete within ~120s or marks function as failed
    2. The detector should already be deployed (deploys first alphabetically)
    3. Connection check is done lazily on first request with retries
    """
    context.logger.info("SAM3 Tracker Proxy initialized")
    context.logger.info(f"Will forward requests to: {DETECTOR_FUNCTION_URL}")
    context.logger.info(f"Timeout: {TIMEOUT}s, Max retries: {MAX_RETRIES}")
    context.logger.warn(
        "NOTE: Make sure the SAM3 detector function "
        "(onnx-facebookresearch-sam3-detector) is deployed and running!"
    )


def handler(context, event):
    """Forward tracker requests to the SAM3 detector function with retry logic."""
    try:
        # Parse incoming request
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        # Determine tracker sub-mode based on request content
        # The detector handler will auto-detect, but we can be explicit
        if data.get("shapes") and not data.get("states"):
            data["mode"] = "track/init"
        elif data.get("states"):
            data["mode"] = "track/frame"
        # track/clear would also work

        # Forward to detector function with retry logic
        context.logger.info(f"Forwarding tracker request (mode={data.get('mode')}) to detector function")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    DETECTOR_FUNCTION_URL,
                    json=data,
                    timeout=TIMEOUT,
                    headers={"Content-Type": "application/json"},
                )

                # Return the response from detector function
                return context.Response(
                    body=response.text,
                    headers={},
                    content_type="application/json",
                    status_code=response.status_code,
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    context.logger.warn(
                        f"Connection failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)

        # All retries exhausted
        context.logger.error(f"Failed to connect to detector function after {MAX_RETRIES} attempts: {last_error}")
        return context.Response(
            body=json.dumps({
                "error": f"Failed to connect to SAM3 detector function at {DETECTOR_FUNCTION_URL}. "
                         "Make sure the detector function (onnx-facebookresearch-sam3-detector) "
                         "is deployed and running. Deploy it first, then restart this proxy."
            }),
            headers={},
            content_type="application/json",
            status_code=503,
        )
    except requests.exceptions.Timeout as e:
        context.logger.error(f"Request to detector function timed out: {e}")
        return context.Response(
            body=json.dumps({
                "error": f"Request timed out after {TIMEOUT}s. The detector may be overloaded or processing a large video."
            }),
            headers={},
            content_type="application/json",
            status_code=504,
        )
    except Exception as e:
        context.logger.error(f"Proxy error: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
