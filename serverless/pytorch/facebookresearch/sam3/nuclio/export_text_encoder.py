#!/usr/bin/env python3
"""
SAM3 Text Encoder ONNX Export Script

Exports the SAM3 text encoder (VETextEncoder) to ONNX format for text-to-segment functionality.

The text encoder converts tokenized text to embeddings that can be used with the PCS decoder
for text-based segmentation.

Architecture:
- Input: Tokenized text (input_ids, attention_mask)
- TextTransformer: 24-layer transformer with width=1024, heads=16
- Output: text_features [B, seq_len, 256], text_mask [B, seq_len]

Usage:
    python export_text_encoder.py --output text_encoder.onnx
    python export_text_encoder.py --output text_encoder.onnx --verify
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# SAM3 Constants
SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_TEXT_WIDTH = 1024
SAM3_TEXT_HEADS = 16
SAM3_TEXT_LAYERS = 24
SAM3_D_MODEL = 256


class TextEncoderWrapper(nn.Module):
    """
    Wrapper for SAM3 VETextEncoder for ONNX export.

    Takes pre-tokenized input (input_ids, attention_mask) and returns
    text features suitable for the PCS decoder.

    This wrapper exposes only the tensor operations needed for ONNX export,
    excluding the tokenizer which must run separately.

    Inputs:
        input_ids: [B, seq_len] int64 - tokenized text
        attention_mask: [B, seq_len] float32 - 1.0 for valid tokens, 0.0 for padding

    Outputs:
        text_features: [B, seq_len, 256] float32 - resized text embeddings
        text_mask: [B, seq_len] bool - inverted attention mask for transformer
    """

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.encoder = text_encoder.encoder  # TextTransformer
        self.resizer = text_encoder.resizer  # Linear(1024, 256)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokenized text to features.

        Args:
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask (1=valid, 0=padding)

        Returns:
            text_features: [B, seq_len, 256] resized embeddings (transposed for decoder)
            text_mask: [B, seq_len] inverted mask for transformer attention
        """
        # Get text embeddings from transformer
        # Returns (pooled, tokens) when output_tokens=True
        _, text_memory = self.encoder(input_ids)  # [B, seq_len, 1024]

        # Resize to decoder dimension
        # Note: The original code transposes to [seq_len, B, 256] but we keep [B, seq_len, 256]
        # for easier handling in ONNX/browser. The decoder will transpose as needed.
        text_features = self.resizer(text_memory)  # [B, seq_len, 256]

        # Invert attention mask for transformer (True = masked/padding)
        # Original: text_attention_mask = text_attention_mask.ne(1)
        text_mask = attention_mask == 0  # [B, seq_len]

        return text_features, text_mask


def load_text_encoder(device: str = "cpu") -> nn.Module:
    """Load the SAM3 text encoder from HuggingFace checkpoint."""
    import pkg_resources
    from sam3.model_builder import _create_text_encoder, download_ckpt_from_hf

    # Get BPE path
    bpe_path = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
    )

    # Create text encoder
    text_encoder = _create_text_encoder(bpe_path)

    # Load checkpoint
    checkpoint_path = download_ckpt_from_hf()
    print(f"Loading checkpoint from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    # Filter text encoder weights
    # The checkpoint uses prefix "detector.backbone.language_backbone."
    text_encoder_state = {}
    prefix = "detector.backbone.language_backbone."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            text_encoder_state[new_key] = v

    if not text_encoder_state:
        # Fallback to alternative prefix
        prefix = "backbone.language_backbone."
        for k, v in ckpt.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                text_encoder_state[new_key] = v

    print(f"Found {len(text_encoder_state)} text encoder weights")

    # Load weights
    missing, unexpected = text_encoder.load_state_dict(text_encoder_state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    text_encoder = text_encoder.to(device).eval()
    return text_encoder


def export_text_encoder(
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """
    Export SAM3 text encoder to ONNX.

    Args:
        output_path: Output ONNX file path
        opset_version: ONNX opset version
        device: Device for export
    """
    print(f"\n{'='*60}")
    print("Exporting SAM3 Text Encoder")
    print(f"{'='*60}")

    # Load text encoder
    text_encoder = load_text_encoder(device)

    # Create wrapper
    wrapper = TextEncoderWrapper(text_encoder).to(device).eval()

    # Dummy inputs
    batch_size = 1
    seq_len = SAM3_CONTEXT_LENGTH

    dummy_input_ids = torch.randint(0, SAM3_VOCAB_SIZE, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["text_features", "text_mask"]

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "text_features": {0: "batch_size"},
        "text_mask": {0: "batch_size"},
    }

    print(f"Input shapes:")
    print(f"  input_ids: {list(dummy_input_ids.shape)}")
    print(f"  attention_mask: {list(dummy_attention_mask.shape)}")

    with torch.no_grad():
        # Test forward pass
        text_features, text_mask = wrapper(dummy_input_ids, dummy_attention_mask)
        print(f"\nOutput shapes:")
        print(f"  text_features: {list(text_features.shape)}")
        print(f"  text_mask: {list(text_mask.shape)}")

        # Export
        print(f"\nExporting to {output_path}...")
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )

    print(f"✓ Text encoder exported to {output_path}")

    # Verify
    _verify_onnx(output_path)


def _verify_onnx(model_path: str) -> None:
    """Verify exported ONNX model."""
    import onnx

    print(f"\nVerifying {model_path}...")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    print("  Inputs:")
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {dims}")

    print("  Outputs:")
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {dims}")

    # Get file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    print("✓ ONNX model verified")


def verify_text_encoder(
    onnx_path: str,
    device: str = "cpu",
) -> None:
    """
    Verify ONNX text encoder against PyTorch implementation.

    Args:
        onnx_path: Path to ONNX model
        device: Device for testing
    """
    import onnxruntime as ort

    print(f"\n{'='*60}")
    print("Verifying Text Encoder ONNX vs PyTorch")
    print(f"{'='*60}")

    # Load PyTorch model
    text_encoder = load_text_encoder(device)
    wrapper = TextEncoderWrapper(text_encoder).to(device).eval()

    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # Test with sample text
    import pkg_resources
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe_path = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
    )
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    test_texts = ["a person", "a red car", "coffee cup on table"]

    for text in test_texts:
        print(f"\nTesting: '{text}'")

        # Tokenize
        input_ids = tokenizer([text], context_length=SAM3_CONTEXT_LENGTH)
        attention_mask = (input_ids != 0).float()

        input_ids_np = input_ids.numpy().astype(np.int64)
        attention_mask_np = attention_mask.numpy().astype(np.float32)

        # PyTorch forward
        with torch.no_grad():
            input_ids_pt = input_ids.to(device)
            attention_mask_pt = attention_mask.to(device)
            pt_features, pt_mask = wrapper(input_ids_pt, attention_mask_pt)
            pt_features = pt_features.cpu().numpy()
            pt_mask = pt_mask.cpu().numpy()

        # ONNX forward
        onnx_features, onnx_mask = ort_session.run(
            None,
            {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
        )

        # Compare
        features_mae = np.abs(pt_features - onnx_features).mean()
        features_max = np.abs(pt_features - onnx_features).max()
        mask_match = np.all(pt_mask == onnx_mask)

        print(f"  Features MAE: {features_mae:.6f}")
        print(f"  Features Max Diff: {features_max:.6f}")
        print(f"  Mask Match: {mask_match}")

        if features_mae > 0.001:
            print("  ⚠ Warning: MAE > 0.001")
        else:
            print("  ✓ OK")


def export_tokenizer(output_dir: str) -> None:
    """
    Export tokenizer files for browser-side tokenization.

    Creates JSON files that can be used with a JavaScript tokenizer implementation.
    """
    import json
    import gzip
    import pkg_resources

    print(f"\n{'='*60}")
    print("Exporting Tokenizer Files")
    print(f"{'='*60}")

    bpe_path = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
    )

    from sam3.model.tokenizer_ve import SimpleTokenizer, bytes_to_unicode

    # Load tokenizer
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    # Export vocab
    vocab = {v: k for k, v in tokenizer.encoder.items()}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(tokenizer.encoder, f)
    print(f"  Vocab saved to {vocab_path} ({len(tokenizer.encoder)} tokens)")

    # Export BPE merges
    with gzip.open(bpe_path, 'rt', encoding='utf-8') as f:
        bpe_data = f.read().split('\n')

    merges = bpe_data[1:49152-256-2+1]  # Same as SimpleTokenizer
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w") as f:
        f.write('\n'.join(merges))
    print(f"  Merges saved to {merges_path} ({len(merges)} merges)")

    # Export byte encoder
    byte_encoder = bytes_to_unicode()
    byte_encoder_path = os.path.join(output_dir, "byte_encoder.json")
    with open(byte_encoder_path, "w") as f:
        json.dump(byte_encoder, f)
    print(f"  Byte encoder saved to {byte_encoder_path}")

    # Export tokenizer config
    config = {
        "context_length": SAM3_CONTEXT_LENGTH,
        "vocab_size": SAM3_VOCAB_SIZE,
        "pad_token": "<end_of_text>",
        "pad_token_id": tokenizer.encoder["<end_of_text>"],
        "start_token": "<start_of_text>",
        "start_token_id": tokenizer.encoder["<start_of_text>"],
        "end_token": "<end_of_text>",
        "end_token_id": tokenizer.encoder["<end_of_text>"],
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    print("✓ Tokenizer files exported")


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 Text Encoder to ONNX")
    parser.add_argument("--output", type=str, default="text_encoder.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for tokenizer files")
    parser.add_argument("--opset", type=int, default=17,
                       help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device for export")
    parser.add_argument("--verify", action="store_true",
                       help="Verify ONNX model against PyTorch")
    parser.add_argument("--export-tokenizer", action="store_true",
                       help="Export tokenizer files for browser-side tokenization")

    args = parser.parse_args()

    # Export text encoder
    export_text_encoder(
        output_path=args.output,
        opset_version=args.opset,
        device=args.device,
    )

    # Verify if requested
    if args.verify:
        verify_text_encoder(args.output, device=args.device)

    # Export tokenizer if requested
    if args.export_tokenizer:
        export_tokenizer(args.output_dir)


if __name__ == "__main__":
    main()
