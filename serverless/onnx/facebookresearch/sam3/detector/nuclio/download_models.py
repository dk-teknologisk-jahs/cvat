#!/usr/bin/env python3
# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 ONNX Model Download/Export Script

Downloads pre-exported ONNX models from GitHub releases, or exports them
from HuggingFace if a token is provided.

Usage:
    # Download from GitHub release (default, no auth needed)
    python download_models.py --output-dir /opt/nuclio/sam3/models

    # Export from HuggingFace (requires token for gated model)
    python download_models.py --output-dir ./models --hf-token YOUR_TOKEN

    # Use environment variables
    HF_TOKEN=xxx python download_models.py --output-dir ./models --use-hf

Environment Variables:
    SAM3_GITHUB_RELEASE_URL - Override GitHub release base URL
    HF_TOKEN - HuggingFace token for exporting models
    SAM3_MODEL_SOURCE - 'github' (default) or 'huggingface'
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default GitHub release URL for pre-exported ONNX models
DEFAULT_GITHUB_RELEASE_URL = "https://github.com/dk-teknologisk-jahs/cvat/releases/download/sam3"

# Model files and their expected sizes (for verification)
MODEL_FILES = {
    "vision_encoder.onnx": {"size_mb": 1740, "required": True},
    "tracker_decoder.onnx": {"size_mb": 21, "required": True},
    "text_encoder.onnx": {"size_mb": 1310, "required": True},
    "pcs_decoder.onnx": {"size_mb": 123, "required": True},
}


def download_from_github(
    output_dir: Path,
    base_url: str = DEFAULT_GITHUB_RELEASE_URL,
    force: bool = False,
) -> bool:
    """
    Download pre-exported ONNX models from GitHub releases.

    Args:
        output_dir: Directory to save models
        base_url: GitHub release base URL
        force: Re-download even if files exist

    Returns:
        True if all models downloaded successfully
    """
    import urllib.request
    import hashlib

    output_dir.mkdir(parents=True, exist_ok=True)
    success = True

    for filename, info in MODEL_FILES.items():
        output_path = output_dir / filename
        url = f"{base_url}/{filename}"

        if output_path.exists() and not force:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            if size_mb > info["size_mb"] * 0.9:  # Allow 10% tolerance
                logger.info(f"  {filename}: Already exists ({size_mb:.1f} MB), skipping")
                continue
            else:
                logger.warning(f"  {filename}: Exists but too small ({size_mb:.1f} MB), re-downloading")

        logger.info(f"  Downloading {filename} from {url}...")
        try:
            # Download with progress reporting
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 / total_size)
                    mb_downloaded = block_num * block_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
            print()  # Newline after progress

            # Verify download
            size_mb = output_path.stat().st_size / (1024 * 1024)
            if size_mb < info["size_mb"] * 0.9:
                logger.error(f"  {filename}: Downloaded file too small ({size_mb:.1f} MB < {info['size_mb']} MB)")
                output_path.unlink()
                success = False
            else:
                logger.info(f"  {filename}: Downloaded successfully ({size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"  {filename}: Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            success = False

    return success


def export_from_huggingface(
    output_dir: Path,
    hf_token: Optional[str] = None,
    device: str = "cpu",
) -> bool:
    """
    Export ONNX models from HuggingFace (requires transformers and torch).

    Args:
        output_dir: Directory to save exported models
        hf_token: HuggingFace token (required for gated models)
        device: Device for export (cpu or cuda)

    Returns:
        True if export succeeded
    """
    try:
        # Import the export script from the pytorch directory
        import importlib.util

        # Find the export script
        export_script_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "pytorch" / "facebookresearch" / "sam3" / "nuclio" / "export_hf_onnx.py",
            Path("/opt/nuclio/sam3/export_hf_onnx.py"),
        ]

        export_script = None
        for path in export_script_paths:
            if path.exists():
                export_script = path
                break

        if export_script is None:
            logger.error("export_hf_onnx.py not found. Cannot export from HuggingFace.")
            logger.error("Searched paths:")
            for p in export_script_paths:
                logger.error(f"  - {p}")
            return False

        logger.info(f"Using export script: {export_script}")

        # Load the export module
        spec = importlib.util.spec_from_file_location("export_hf_onnx", export_script)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to load module spec from {export_script}")
            return False
        export_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_module)

        # Import required modules (only needed for HF export)
        import torch  # type: ignore
        from transformers import Sam3Model, Sam3TrackerModel  # type: ignore

        output_dir.mkdir(parents=True, exist_ok=True)

        # Set token in environment if provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            logger.info("Using provided HuggingFace token")

        token = hf_token or os.environ.get("HF_TOKEN")

        # Load models
        logger.info("Loading Sam3TrackerModel from HuggingFace...")
        tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=token)
        tracker_model = tracker_model.to(device).eval()

        logger.info("Loading Sam3Model from HuggingFace...")
        pcs_model = Sam3Model.from_pretrained("facebook/sam3", token=token)
        pcs_model = pcs_model.to(device).eval()

        # Export all components
        logger.info("Exporting vision encoder...")
        export_module.export_vision_encoder(tracker_model, output_dir, device)

        logger.info("Exporting tracker decoder...")
        export_module.export_tracker_decoder(tracker_model, output_dir, device)

        logger.info("Exporting text encoder...")
        export_module.export_text_encoder(pcs_model, output_dir, device)

        logger.info("Exporting PCS decoder...")
        export_module.export_pcs_decoder(pcs_model, output_dir, device)

        logger.info("Export complete!")
        return True

    except ImportError as e:
        logger.error(f"Missing dependencies for HuggingFace export: {e}")
        logger.error("Install with: pip install torch transformers")
        return False
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_models(model_dir: Path) -> bool:
    """Verify all required models exist and have reasonable sizes."""
    all_ok = True

    for filename, info in MODEL_FILES.items():
        path = model_dir / filename
        if not path.exists():
            if info["required"]:
                logger.error(f"  Missing required model: {filename}")
                all_ok = False
            else:
                logger.warning(f"  Missing optional model: {filename}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb < info["size_mb"] * 0.5:  # Allow 50% tolerance for compression differences
                logger.warning(f"  {filename}: Size ({size_mb:.1f} MB) much smaller than expected ({info['size_mb']} MB)")
            else:
                logger.info(f"  {filename}: OK ({size_mb:.1f} MB)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download or export SAM3 ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/nuclio/sam3/models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--source",
        choices=["github", "huggingface", "auto"],
        default="auto",
        help="Model source: github (download pre-exported), huggingface (export from HF), auto (try github first)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Force HuggingFace export (shortcut for --source huggingface)",
    )
    parser.add_argument(
        "--github-url",
        type=str,
        default=None,
        help="Override GitHub release URL",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for HuggingFace export (cpu or cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/export even if files exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Handle source selection
    source = args.source
    if args.use_hf:
        source = "huggingface"
    elif os.environ.get("SAM3_MODEL_SOURCE"):
        source = os.environ.get("SAM3_MODEL_SOURCE")

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Get GitHub URL
    github_url = args.github_url or os.environ.get("SAM3_GITHUB_RELEASE_URL", DEFAULT_GITHUB_RELEASE_URL)

    logger.info("=" * 60)
    logger.info("SAM3 ONNX Model Setup")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Source: {source}")

    if args.verify_only:
        logger.info("\nVerifying existing models...")
        if verify_models(output_dir):
            logger.info("\nAll models verified successfully!")
            return 0
        else:
            logger.error("\nModel verification failed!")
            return 1

    success = False

    if source == "auto":
        # Try GitHub first, fall back to HuggingFace
        logger.info("\nTrying GitHub download first...")
        success = download_from_github(output_dir, github_url, args.force)

        if not success and hf_token:
            logger.info("\nGitHub download failed, trying HuggingFace export...")
            success = export_from_huggingface(output_dir, hf_token, args.device)

    elif source == "github":
        logger.info(f"\nDownloading from GitHub: {github_url}")
        success = download_from_github(output_dir, github_url, args.force)

    elif source == "huggingface":
        if not hf_token:
            logger.error("HuggingFace token required for export. Set HF_TOKEN or use --hf-token")
            return 1
        logger.info("\nExporting from HuggingFace...")
        success = export_from_huggingface(output_dir, hf_token, args.device)

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("Verifying downloaded/exported models...")
        if verify_models(output_dir):
            logger.info("\nSetup complete! All models ready.")
            return 0
        else:
            logger.warning("\nSome models may have issues, but continuing...")
            return 0
    else:
        logger.error("\nModel setup failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
