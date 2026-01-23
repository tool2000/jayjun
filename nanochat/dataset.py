"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.

Bilingual Support (English + Korean):
- English: karpathy/fineweb-edu-100b-shuffle (default)
- Korean: eliceai/korean-fineweb-edu-demo (HuggingFace datasets)
- Use NANOCHAT_LANG_RATIO environment variable to set mixing ratio (default: 1.0 = English only)
  Example: NANOCHAT_LANG_RATIO=0.7 means 70% English, 30% Korean
"""

import os
import argparse
import time
import random
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# English dataset: karpathy/fineweb-edu-100b-shuffle
BASE_URL_EN = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_SHARD_EN = 1822  # the last datashard is shard_01822.parquet
index_to_filename_en = lambda index: f"shard_{index:05d}.parquet"

# Korean dataset: eliceai/korean-fineweb-edu-demo (HuggingFace datasets)
KOREAN_DATASET_NAME = "eliceai/korean-fineweb-edu-demo"

# Legacy alias for backward compatibility
BASE_URL = BASE_URL_EN
MAX_SHARD = MAX_SHARD_EN
index_to_filename = index_to_filename_en

base_dir = get_base_dir()
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR")
if DATA_DIR is None:
    DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Korean data directory
DATA_DIR_KO = os.environ.get("NANOCHAT_DATA_DIR_KO")
if DATA_DIR_KO is None:
    DATA_DIR_KO = os.path.join(base_dir, "base_data_ko")

# Language mixing ratio: 1.0 = English only, 0.7 = 70% English + 30% Korean
LANG_RATIO_EN = float(os.environ.get("NANOCHAT_LANG_RATIO", "1.0"))

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def list_parquet_files_ko():
    """Lists Korean dataset parquet files."""
    return list_parquet_files(data_dir=DATA_DIR_KO)


def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def parquets_iter_batched_bilingual(split, start=0, step=1, en_ratio=0.7, seed=42):
    """
    Iterate through English and Korean datasets with specified mixing ratio.

    Args:
        split: "train" or "val"
        start/step: for DDP sharding
        en_ratio: ratio of English samples (0.7 = 70% English, 30% Korean)
        seed: random seed for reproducible mixing

    Yields:
        texts: list of text strings (mixed English and Korean)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert 0.0 <= en_ratio <= 1.0, "en_ratio must be between 0.0 and 1.0"

    rng = random.Random(seed)

    # Get parquet files for both languages
    en_paths = list_parquet_files(data_dir=DATA_DIR)
    ko_paths = list_parquet_files(data_dir=DATA_DIR_KO)

    if not en_paths:
        raise RuntimeError(f"No English parquet files found in {DATA_DIR}")
    if not ko_paths and en_ratio < 1.0:
        raise RuntimeError(
            f"No Korean parquet files found in {DATA_DIR_KO}. Run download_korean_dataset() first."
        )

    # Split train/val (last file is val)
    en_paths = en_paths[:-1] if split == "train" else en_paths[-1:]
    if ko_paths:
        ko_paths = ko_paths[:-1] if split == "train" else ko_paths[-1:]

    # Create iterators for both languages
    def make_iterator(paths, text_column="text"):
        for filepath in paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column(text_column).to_pylist()
                for text in texts:
                    yield text

    en_iter = make_iterator(en_paths, "text")
    ko_iter = make_iterator(ko_paths, "text") if ko_paths else iter([])

    # Mix documents according to ratio
    en_buffer = []
    ko_buffer = []
    batch_size = 1000  # number of documents per batch

    while True:
        # Refill English buffer
        while len(en_buffer) < batch_size:
            try:
                en_buffer.append(next(en_iter))
            except StopIteration:
                break

        # Refill Korean buffer
        if en_ratio < 1.0:
            while len(ko_buffer) < batch_size:
                try:
                    ko_buffer.append(next(ko_iter))
                except StopIteration:
                    break

        # Check if we have any data left
        if not en_buffer and not ko_buffer:
            return

        # Build mixed batch
        mixed_batch = []
        target_size = min(batch_size, len(en_buffer) + len(ko_buffer))

        for _ in range(target_size):
            # Decide language based on ratio and availability
            use_english = True
            if en_buffer and ko_buffer:
                use_english = rng.random() < en_ratio
            elif ko_buffer:
                use_english = False
            elif not en_buffer:
                break

            if use_english and en_buffer:
                mixed_batch.append(en_buffer.pop(0))
            elif ko_buffer:
                mixed_batch.append(ko_buffer.pop(0))
            elif en_buffer:
                mixed_batch.append(en_buffer.pop(0))

        if mixed_batch:
            yield mixed_batch


# -----------------------------------------------------------------------------
def download_single_file(index):
    """Downloads a single English file index, with some backoff"""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename_en(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL_EN}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_korean_dataset(max_samples=None):
    """
    Downloads the Korean dataset from HuggingFace and saves as parquet files.

    Args:
        max_samples: Maximum number of samples to download (None = all)

    Returns:
        Path to the Korean data directory
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    os.makedirs(DATA_DIR_KO, exist_ok=True)

    print(f"Downloading Korean dataset: {KOREAN_DATASET_NAME}")
    print(f"Target directory: {DATA_DIR_KO}")

    # Load the dataset from HuggingFace
    dataset = load_dataset(KOREAN_DATASET_NAME, split="train")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # Save as parquet files (split into shards for consistency with English data)
    shard_size = 10000  # samples per shard
    num_shards = (len(dataset) + shard_size - 1) // shard_size

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))

        filename = f"shard_{shard_idx:05d}.parquet"
        filepath = os.path.join(DATA_DIR_KO, filename)

        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            continue

        # Convert to pandas and save as parquet
        df = shard.to_pandas()
        df.to_parquet(filepath, index=False)
        print(f"Saved {filename} ({end_idx - start_idx} samples)")

    print(
        f"Korean dataset download complete: {num_shards} shards saved to {DATA_DIR_KO}"
    )
    return DATA_DIR_KO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT dataset shards (English and/or Korean)"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of English shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--korean",
        action="store_true",
        help="Download Korean dataset (eliceai/korean-fineweb-edu-demo)",
    )
    parser.add_argument(
        "--korean-samples",
        type=int,
        default=None,
        help="Maximum number of Korean samples to download (default: all)",
    )
    args = parser.parse_args()

    # Download English dataset
    if args.num_files != -1 or (not args.korean):
        num = (
            MAX_SHARD_EN + 1
            if args.num_files == -1
            else min(args.num_files, MAX_SHARD_EN + 1)
        )
        if num > 0:
            ids_to_download = list(range(num))
            print(
                f"Downloading {len(ids_to_download)} English shards using {args.num_workers} workers..."
            )
            print(f"Target directory: {DATA_DIR}")
            print()
            with Pool(processes=args.num_workers) as pool:
                results = pool.map(download_single_file, ids_to_download)

            # Report results
            successful = sum(1 for success in results if success)
            print(
                f"Done! Downloaded: {successful}/{len(ids_to_download)} English shards to {DATA_DIR}"
            )

    # Download Korean dataset
    if args.korean:
        print()
        print("=" * 60)
        print("Downloading Korean dataset...")
        print("=" * 60)
        download_korean_dataset(max_samples=args.korean_samples)
