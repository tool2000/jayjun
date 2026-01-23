"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.

Bilingual support: Use --en-ratio to mix English and Korean data.
Example: --en-ratio=0.7 means 70% English, 30% Korean
"""

import os
import time
import argparse
import random
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import (
    parquets_iter_batched,
    parquets_iter_batched_bilingual,
    list_parquet_files_ko,
)

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max-chars",
    type=int,
    default=10_000_000_000,
    help="Maximum characters to train on (default: 10B)",
)
parser.add_argument(
    "--doc-cap",
    type=int,
    default=10_000,
    help="Maximum characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Vocabulary size (default: 32768 = 2^15)",
)
parser.add_argument(
    "--en-ratio",
    type=float,
    default=1.0,
    help="English ratio for bilingual training (1.0 = English only, 0.7 = 70%% English + 30%% Korean)",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for bilingual mixing"
)
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"en_ratio: {args.en_ratio:.2f}")

# Check if bilingual mode is requested
is_bilingual = args.en_ratio < 1.0
if is_bilingual:
    ko_files = list_parquet_files_ko()
    if not ko_files:
        print(
            "WARNING: No Korean data found. Run 'python -m nanochat.dataset --korean' first."
        )
        print("Falling back to English only.")
        is_bilingual = False
    else:
        print(
            f"Bilingual mode: {args.en_ratio * 100:.0f}% English, {(1 - args.en_ratio) * 100:.0f}% Korean"
        )

# -----------------------------------------------------------------------------
# Text iterator


def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters

    Supports bilingual mode: mixes English and Korean based on en_ratio.
    """
    nchars = 0

    if is_bilingual:
        # Use bilingual iterator
        for batch in parquets_iter_batched_bilingual(
            split="train", en_ratio=args.en_ratio, seed=args.seed
        ):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[: args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return
    else:
        # Original English-only iterator
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[: args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check (English + Korean)
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# Korean sanity check
test_text_ko = """안녕하세요! 저는 제이준입니다.
오늘 날씨가 좋네요. 한국어 테스트입니다.
숫자: 123, 4567, 89
특수문자: @#$%^&*()"""
encoded_ko = tokenizer.encode(test_text_ko)
decoded_ko = tokenizer.decode(encoded_ko)
assert decoded_ko == test_text_ko, f"Korean decode mismatch: {decoded_ko}"
print(
    f"Korean text tokens: {len(encoded_ko)} (chars: {len(test_text_ko)}, ratio: {len(test_text_ko) / len(encoded_ko):.2f} chars/token)"
)

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[
        token_id
    ]  # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(
            token_str.encode("utf-8")
        )  # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report

token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(
    section="Tokenizer training",
    data=[
        vars(args),  # argparse command line arguments
        {"train_time": train_time},
        {"num_special_tokens": len(special_set)},
        {
            "token_bytes_min": int(token_bytes_nonzero.min().item()),
            "token_bytes_max": int(token_bytes_nonzero.max().item()),
            "token_bytes_mean": token_bytes_nonzero.mean().item(),
            "token_bytes_std": token_bytes_nonzero.std().item(),
        },
    ],
)
