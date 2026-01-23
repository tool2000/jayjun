"""
Build a mixed English/Korean dataset in parquet shards.

English source: existing FineWeb-Edu shards in base_data.
Korean source: eliceai/korean-fineweb-edu-demo from HuggingFace.

The output shards follow the nanochat convention:
- shard_00000.parquet ... shard_XXXXX.parquet
- the final shard is treated as val by the dataloader.
"""

import argparse
import os
import random
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched


def iter_english_docs(data_dir: str, max_chars: Optional[int]) -> Iterable[str]:
    total_chars = 0
    for batch in parquets_iter_batched(split="train", data_dir=data_dir):
        for doc in batch:
            if not isinstance(doc, str) or not doc:
                continue
            total_chars += len(doc)
            if max_chars is not None and total_chars > max_chars:
                return
            yield doc


def iter_english_val_docs(data_dir: str, max_chars: Optional[int]) -> Iterable[str]:
    total_chars = 0
    for batch in parquets_iter_batched(split="val", data_dir=data_dir):
        for doc in batch:
            if not isinstance(doc, str) or not doc:
                continue
            total_chars += len(doc)
            if max_chars is not None and total_chars > max_chars:
                return
            yield doc


def pick_text_key(dataset) -> str:
    if hasattr(dataset, "features") and dataset.features:
        if "text" in dataset.features:
            return "text"
        for key, feature in dataset.features.items():
            if getattr(feature, "dtype", None) == "string":
                return key
    return "text"


def korean_stream(split: str, streaming: bool) -> Iterable[str]:
    dataset = load_dataset(
        "eliceai/korean-fineweb-edu-demo", split=split, streaming=streaming
    )
    text_key = pick_text_key(dataset)
    for row in dataset:
        text = row.get(text_key)
        if isinstance(text, str) and text:
            yield text


def ensure_empty_dir(path: str, overwrite: bool) -> None:
    os.makedirs(path, exist_ok=True)
    existing = [name for name in os.listdir(path) if name.endswith(".parquet")]
    if existing and not overwrite:
        raise RuntimeError(
            f"Output directory already has parquet files: {path}. "
            "Use --overwrite to replace them."
        )
    if existing and overwrite:
        for name in existing:
            os.remove(os.path.join(path, name))


def write_shard(output_dir: str, shard_idx: int, docs: list[str]) -> None:
    table = pa.Table.from_pydict({"text": docs})
    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
    pq.write_table(table, path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix FineWeb-Edu with Korean data")
    parser.add_argument("--korean-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--max-chars-per-shard", type=int, default=250_000_000)
    parser.add_argument("--val-max-chars", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--english-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--no-repeat-korean", dest="repeat_korean", action="store_false", default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.korean_ratio < 1.0):
        raise ValueError("--korean-ratio must be between 0 and 1")

    base_dir = get_base_dir()
    english_dir = args.english_dir or os.path.join(base_dir, "base_data")
    output_dir = args.output_dir or os.path.join(base_dir, "base_data_ko10")
    ensure_empty_dir(output_dir, args.overwrite)

    random.seed(args.seed)
    korean_probability = args.korean_ratio / (1.0 - args.korean_ratio)

    korean_iter = iter(korean_stream(split="train", streaming=True))

    def next_korean() -> Optional[str]:
        nonlocal korean_iter
        try:
            return next(korean_iter)
        except StopIteration:
            if not args.repeat_korean:
                return None
            korean_iter = iter(korean_stream(split="train", streaming=True))
            try:
                return next(korean_iter)
            except StopIteration:
                return None

    shard_idx = 0
    shard_docs: list[str] = []
    shard_chars = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_docs, shard_chars
        if not shard_docs:
            return
        write_shard(output_dir, shard_idx, shard_docs)
        shard_idx += 1
        shard_docs = []
        shard_chars = 0

    for doc in iter_english_docs(english_dir, max_chars=args.max_train_chars):
        shard_docs.append(doc)
        shard_chars += len(doc)
        if random.random() < korean_probability:
            korean_doc = next_korean()
            if korean_doc:
                shard_docs.append(korean_doc)
                shard_chars += len(korean_doc)
        if shard_chars >= args.max_chars_per_shard:
            flush_shard()

    flush_shard()

    val_docs: list[str] = []
    val_chars = 0
    korean_val_iter = iter(korean_stream(split="train", streaming=True))

    def next_korean_val() -> Optional[str]:
        nonlocal korean_val_iter
        try:
            return next(korean_val_iter)
        except StopIteration:
            if not args.repeat_korean:
                return None
            korean_val_iter = iter(korean_stream(split="train", streaming=True))
            try:
                return next(korean_val_iter)
            except StopIteration:
                return None

    for doc in iter_english_val_docs(english_dir, max_chars=args.val_max_chars):
        val_docs.append(doc)
        val_chars += len(doc)
        if random.random() < korean_probability:
            korean_doc = next_korean_val()
            if korean_doc:
                val_docs.append(korean_doc)
                val_chars += len(korean_doc)
        if val_chars >= args.val_max_chars:
            break

    if val_docs:
        write_shard(output_dir, shard_idx, val_docs)

    print(f"Wrote mixed shards to {output_dir}")


if __name__ == "__main__":
    main()
