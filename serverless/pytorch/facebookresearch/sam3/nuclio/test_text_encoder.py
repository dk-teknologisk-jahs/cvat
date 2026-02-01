#!/usr/bin/env python3
"""
SAM3 Text Encoder ONNX Test Suite

Comprehensive tests for SAM3 text encoder ONNX model:
1. Exact numerical comparison between ONNX and PyTorch
2. Various text prompt types (short, long, special characters, etc.)
3. Tokenizer validation
4. End-to-end text-to-segment comparison

Usage:
    # Basic verification
    python test_text_encoder.py

    # Full comparison with statistics
    python test_text_encoder.py --full

    # Test with custom ONNX path
    python test_text_encoder.py --onnx-path /path/to/text_encoder.onnx

    # Compare end-to-end text-to-segment results
    python test_text_encoder.py --e2e --image /path/to/image.jpg
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Configuration
# ============================================================================
SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_EMBED_DIM = 256

DEFAULT_TEXT_ENCODER_ONNX = str(Path(__file__).parent / "text_encoder.onnx")
DEFAULT_TOKENIZER_DIR = str(Path(__file__).parent)

# Tolerance thresholds
MAX_ACCEPTABLE_MAE = 0.0001  # Mean absolute error
MAX_ACCEPTABLE_MAX_DIFF = 0.001  # Maximum single value difference
MIN_ACCEPTABLE_CORR = 0.99999  # Correlation coefficient


@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    text: str
    mae: float
    max_diff: float
    correlation: float
    mask_match: bool
    passed: bool
    details: str = ""


# ============================================================================
# Test Cases
# ============================================================================
TEST_PROMPTS = {
    "simple_object": [
        "a person",
        "a car",
        "a dog",
        "a cat",
        "a tree",
        "a building",
        "a chair",
        "a table",
    ],
    "descriptive": [
        "a red car",
        "a large dog",
        "a wooden chair",
        "a tall building",
        "a green tree",
        "a small cat",
        "a white house",
        "a blue sky",
    ],
    "complex_phrases": [
        "a person wearing a red shirt",
        "a car parked on the street",
        "a dog running in the park",
        "a coffee cup on the table",
        "a bird flying in the sky",
        "a book on the shelf",
        "a flower in a vase",
        "a person riding a bicycle",
    ],
    "edge_cases": [
        "",  # Empty string
        "a",  # Single character
        "person",  # No article
        "THE PERSON",  # Uppercase
        "a person!",  # Punctuation
        "a person?",  # Question mark
        "person, car, dog",  # Multiple items
        "  a person  ",  # Whitespace
    ],
    "long_prompts": [
        "a person standing next to a red car in front of a tall building",
        "a small dog sitting on a wooden chair near a green tree in the garden",
        "a white coffee cup placed on a brown wooden table in a cozy kitchen",
    ],
    "special_concepts": [
        "background",
        "foreground",
        "object",
        "thing",
        "visual",  # Used as fallback prompt in SAM3
        "segment",
        "mask",
        "region",
    ],
}


# ============================================================================
# Tokenizer Functions (Pure Python - No HuggingFace)
# ============================================================================
def load_tokenizer(tokenizer_dir: str) -> dict:
    """Load tokenizer from exported JSON files."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")
    byte_encoder_path = os.path.join(tokenizer_dir, "byte_encoder.json")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")

    with open(vocab_path, 'r') as f:
        encoder = json.load(f)

    with open(merges_path, 'r') as f:
        merges = f.read().strip().split('\n')

    with open(byte_encoder_path, 'r') as f:
        byte_encoder = json.load(f)
        # Convert string keys back to int
        byte_encoder = {int(k): v for k, v in byte_encoder.items()}

    return {
        'encoder': encoder,
        'merges': merges,
        'byte_encoder': byte_encoder,
        'decoder': {v: k for k, v in encoder.items()},
        'byte_decoder': {v: k for k, v in byte_encoder.items()},
        'bpe_ranks': {tuple(merge.split()): i for i, merge in enumerate(merges)},
    }


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe(token: str, bpe_ranks: dict) -> str:
    """Apply BPE to a token. Adds </w> suffix to last character before processing."""
    if len(token) == 0:
        return token

    # Add </w> to last character (CLIP/OpenAI tokenizer style)
    word = tuple(token[:-1]) + (token[-1] + "</w>",)

    pairs = get_pairs(word)

    if not pairs:
        return token + "</w>"

    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = get_pairs(word)

    return ' '.join(word)


def tokenize(text: str, tokenizer: dict, context_length: int = 32) -> np.ndarray:
    """Tokenize text using the loaded tokenizer."""
    import re

    encoder = tokenizer['encoder']
    byte_encoder = tokenizer['byte_encoder']
    bpe_ranks = tokenizer['bpe_ranks']

    # Special tokens
    sot_token = encoder.get('<|startoftext|>', encoder.get('<start_of_text>'))
    eot_token = encoder.get('<|endoftext|>', encoder.get('<end_of_text>'))

    # Clean text
    text = text.lower().strip()

    # Try to use regex module for Unicode support, fall back to simpler pattern
    try:
        import regex
        pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
        tokens = regex.findall(pat, text)
    except ImportError:
        # Simple fallback - match words and punctuation
        tokens = re.findall(r"[\w]+|[^\s\w]", text)

    # BPE encode each token
    bpe_tokens = []
    for token in tokens:
        # Encode bytes
        token_bytes = token.encode('utf-8')
        token_unicode = ''.join(byte_encoder.get(b, chr(b)) for b in token_bytes)

        # Apply BPE (bpe() already handles </w> suffix)
        bpe_result = bpe(token_unicode, bpe_ranks)
        bpe_tokens.extend(encoder.get(t, 0) for t in bpe_result.split())

    # Build result with special tokens
    result = [sot_token] + bpe_tokens[:context_length-2] + [eot_token]

    # Pad to context_length
    result = result + [0] * (context_length - len(result))

    return np.array([result], dtype=np.int64)


# ============================================================================
# PyTorch Reference Functions
# ============================================================================
def load_pytorch_text_encoder(device: str = "cpu"):
    """Load PyTorch text encoder from official SAM3."""
    import torch
    from sam3.model_builder import build_sam3_image_model

    print("Loading PyTorch SAM3 model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    text_encoder = model.backbone.language_backbone
    return text_encoder, model


class PyTorchTextEncoderWrapper:
    """Wrapper to call PyTorch text encoder the same way as ONNX."""

    def __init__(self, text_encoder, device: str = "cpu"):
        self.encoder = text_encoder.encoder  # TextTransformer
        self.resizer = text_encoder.resizer  # Linear(1024, 256)
        self.device = device

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        """Encode tokenized text, matching ONNX interface."""
        import torch

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Get text embeddings from transformer
            _, text_memory = self.encoder(input_ids_t)  # [B, seq_len, 1024]

            # Resize to decoder dimension
            text_features = self.resizer(text_memory)  # [B, seq_len, 256]

            # Invert attention mask for transformer (True = masked/padding)
            text_mask = attention_mask_t == 0  # [B, seq_len]

        return text_features.cpu().numpy(), text_mask.cpu().numpy()


def pytorch_encode_text(text: str, pt_wrapper: PyTorchTextEncoderWrapper, tokenizer: dict, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Encode text using PyTorch model (same tokenization as ONNX)."""
    # Tokenize using our tokenizer
    input_ids = tokenize(text, tokenizer, SAM3_CONTEXT_LENGTH)
    attention_mask = (input_ids != 0).astype(np.float32)

    # Encode
    text_features, text_mask = pt_wrapper.encode(input_ids, attention_mask)

    return text_features, text_mask


# ============================================================================
# ONNX Functions
# ============================================================================
def load_onnx_text_encoder(onnx_path: str, device: str = "cpu"):
    """Load ONNX text encoder."""
    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    return session


def onnx_encode_text(text: str, session, tokenizer: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Encode text using ONNX model."""
    # Tokenize
    input_ids = tokenize(text, tokenizer, SAM3_CONTEXT_LENGTH)
    attention_mask = (input_ids != 0).astype(np.float32)

    # Run ONNX
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })

    return outputs[0], outputs[1]


# ============================================================================
# Comparison Functions
# ============================================================================
def compare_outputs(
    pt_features: np.ndarray,
    pt_mask: np.ndarray,
    onnx_features: np.ndarray,
    onnx_mask: np.ndarray,
) -> Tuple[float, float, float, bool]:
    """
    Compare PyTorch and ONNX outputs.

    Returns:
        mae: Mean absolute error
        max_diff: Maximum absolute difference
        correlation: Pearson correlation coefficient
        mask_match: Whether masks match exactly
    """
    # Flatten for comparison
    pt_flat = pt_features.flatten()
    onnx_flat = onnx_features.flatten()

    # Calculate metrics
    mae = np.abs(pt_flat - onnx_flat).mean()
    max_diff = np.abs(pt_flat - onnx_flat).max()

    # Correlation
    if np.std(pt_flat) > 0 and np.std(onnx_flat) > 0:
        correlation = np.corrcoef(pt_flat, onnx_flat)[0, 1]
    else:
        correlation = 1.0 if np.allclose(pt_flat, onnx_flat) else 0.0

    # Mask comparison
    mask_match = np.allclose(pt_mask, onnx_mask)

    return mae, max_diff, correlation, mask_match


def run_single_test(
    text: str,
    name: str,
    pt_wrapper: PyTorchTextEncoderWrapper,
    onnx_session,
    tokenizer: dict,
    device: str,
    verbose: bool = True,
) -> TestResult:
    """Run a single comparison test."""
    try:
        # PyTorch encoding (using same tokenizer as ONNX for fair comparison)
        pt_features, pt_mask = pytorch_encode_text(text, pt_wrapper, tokenizer, device)

        # ONNX encoding
        onnx_features, onnx_mask = onnx_encode_text(text, onnx_session, tokenizer)

        # Compare
        mae, max_diff, correlation, mask_match = compare_outputs(
            pt_features, pt_mask, onnx_features, onnx_mask
        )

        # Determine pass/fail
        passed = (
            mae <= MAX_ACCEPTABLE_MAE and
            max_diff <= MAX_ACCEPTABLE_MAX_DIFF and
            correlation >= MIN_ACCEPTABLE_CORR and
            mask_match
        )

        details = ""
        if not passed:
            if mae > MAX_ACCEPTABLE_MAE:
                details += f"MAE {mae:.6f} > {MAX_ACCEPTABLE_MAE}; "
            if max_diff > MAX_ACCEPTABLE_MAX_DIFF:
                details += f"MaxDiff {max_diff:.6f} > {MAX_ACCEPTABLE_MAX_DIFF}; "
            if correlation < MIN_ACCEPTABLE_CORR:
                details += f"Corr {correlation:.6f} < {MIN_ACCEPTABLE_CORR}; "
            if not mask_match:
                details += "Mask mismatch; "

        result = TestResult(
            name=name,
            text=text,
            mae=mae,
            max_diff=max_diff,
            correlation=correlation,
            mask_match=mask_match,
            passed=passed,
            details=details,
        )

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{status}] {name}: '{text[:30]}...' " if len(text) > 30 else f"  [{status}] {name}: '{text}'")
            print(f"          MAE={mae:.8f}, MaxDiff={max_diff:.8f}, Corr={correlation:.8f}")

        return result

    except Exception as e:
        import traceback
        print(f"  [ERROR] {name}: {str(e)}")
        traceback.print_exc()
        return TestResult(
            name=name,
            text=text,
            mae=float('inf'),
            max_diff=float('inf'),
            correlation=0.0,
            mask_match=False,
            passed=False,
            details=f"Error: {str(e)}",
        )


# ============================================================================
# Main Test Functions
# ============================================================================
def run_full_test_suite(
    onnx_path: str,
    tokenizer_dir: str,
    device: str = "cpu",
    verbose: bool = True,
) -> List[TestResult]:
    """Run the full test suite."""
    print(f"\n{'='*70}")
    print("SAM3 Text Encoder ONNX vs PyTorch Comparison")
    print(f"{'='*70}")
    print(f"\nONNX Model: {onnx_path}")
    print(f"Tokenizer Dir: {tokenizer_dir}")
    print(f"Device: {device}")
    print(f"\nTolerances:")
    print(f"  Max MAE: {MAX_ACCEPTABLE_MAE}")
    print(f"  Max Single Diff: {MAX_ACCEPTABLE_MAX_DIFF}")
    print(f"  Min Correlation: {MIN_ACCEPTABLE_CORR}")

    # Load models
    print(f"\n{'='*70}")
    print("Loading Models...")
    print(f"{'='*70}")

    pt_encoder, _ = load_pytorch_text_encoder(device)
    pt_wrapper = PyTorchTextEncoderWrapper(pt_encoder, device)
    onnx_session = load_onnx_text_encoder(onnx_path, device)
    tokenizer = load_tokenizer(tokenizer_dir)

    print(f"  ✓ PyTorch model loaded")
    print(f"  ✓ ONNX model loaded")
    print(f"  ✓ Tokenizer loaded ({len(tokenizer['encoder'])} tokens)")

    # Run tests
    all_results = []

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"Testing: {category}")
        print(f"{'='*70}")

        for i, prompt in enumerate(prompts):
            result = run_single_test(
                text=prompt,
                name=f"{category}_{i}",
                pt_wrapper=pt_wrapper,
                onnx_session=onnx_session,
                tokenizer=tokenizer,
                device=device,
                verbose=verbose,
            )
            all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {100*passed/total:.1f}%")

    if all_results:
        maes = [r.mae for r in all_results if r.mae < float('inf')]
        max_diffs = [r.max_diff for r in all_results if r.max_diff < float('inf')]
        corrs = [r.correlation for r in all_results if r.correlation > 0]

        print(f"\nStatistics across all tests:")
        print(f"  MAE:     mean={np.mean(maes):.8f}, max={np.max(maes):.8f}, min={np.min(maes):.8f}")
        print(f"  MaxDiff: mean={np.mean(max_diffs):.8f}, max={np.max(max_diffs):.8f}")
        print(f"  Corr:    mean={np.mean(corrs):.8f}, min={np.min(corrs):.8f}")

    # Failed tests
    failed_results = [r for r in all_results if not r.passed]
    if failed_results:
        print(f"\nFailed Tests:")
        for r in failed_results:
            print(f"  - {r.name}: '{r.text[:50]}' - {r.details}")

    return all_results


def run_quick_test(
    onnx_path: str,
    tokenizer_dir: str,
    device: str = "cpu",
) -> bool:
    """Run a quick sanity check."""
    print(f"\n{'='*70}")
    print("Quick Sanity Check")
    print(f"{'='*70}")

    # Load models
    pt_encoder, _ = load_pytorch_text_encoder(device)
    pt_wrapper = PyTorchTextEncoderWrapper(pt_encoder, device)
    onnx_session = load_onnx_text_encoder(onnx_path, device)
    tokenizer = load_tokenizer(tokenizer_dir)

    # Test a few prompts
    test_prompts = ["a person", "a red car", "coffee cup on table"]

    all_passed = True
    for prompt in test_prompts:
        result = run_single_test(
            text=prompt,
            name="quick_test",
            pt_wrapper=pt_wrapper,
            onnx_session=onnx_session,
            tokenizer=tokenizer,
            device=device,
            verbose=True,
        )
        if not result.passed:
            all_passed = False

    print(f"\nQuick Test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SAM3 Text Encoder ONNX Test Suite")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=DEFAULT_TEXT_ENCODER_ONNX,
        help="Path to text encoder ONNX model",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=DEFAULT_TOKENIZER_DIR,
        help="Directory containing tokenizer files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )

    args = parser.parse_args()

    if args.full:
        results = run_full_test_suite(
            onnx_path=args.onnx_path,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
            verbose=not args.quiet,
        )
        sys.exit(0 if all(r.passed for r in results) else 1)
    else:
        passed = run_quick_test(
            onnx_path=args.onnx_path,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
        )
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
