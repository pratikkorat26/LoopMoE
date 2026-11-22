"""Zero-bottleneck pre-tokenized uint16 memory-mapped dataset loaders."""

import time
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import numpy as np


def prepare_dataset(
    data_dir: str,
    dataset_key: str = "tinystories",
    num_train_docs: int = 25000,
    num_val_docs: int = 1000,
) -> Tuple[Path, Path]:
    """Download & pre-tokenize TinyStories or FineWeb-Edu (crawled web pages) into uint16 memmap."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if dataset_key == "tinystories" else f"{dataset_key}_"
    train_path = out_dir / f"{prefix}train.bin"
    val_path = out_dir / f"{prefix}val.bin"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    t0 = time.perf_counter()
    if dataset_key == "fineweb":
        print(f"[data] Streaming FineWeb-Edu (crawled web pages) -> {out_dir}/...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
        train_toks, val_toks = [], []
        total_docs = num_train_docs + num_val_docs
        for idx, ex in enumerate(ds):
            if idx >= total_docs:
                break
            text = ex["text"].strip()
            if not text:
                continue
            toks = enc.encode_ordinary(text)
            toks.append(eot)
            if idx < num_val_docs:
                val_toks.extend(toks)
            else:
                train_toks.extend(toks)
        np.array(train_toks, dtype=np.uint16).tofile(train_path)
        np.array(val_toks, dtype=np.uint16).tofile(val_path)
        n_train_toks, n_val_toks = len(train_toks), len(val_toks)
    else:
        print("[data] Preparing pre-tokenized memmap dataset from roneneldan/TinyStories...")

        def _tokenize_split(split_name: str, limit: int, dest: Path) -> int:
            ds = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
            tokens = []
            for idx, ex in enumerate(ds):
                if idx >= limit:
                    break
                text = ex["text"].strip()
                if text:
                    tokens.extend(enc.encode_ordinary(text))
                    tokens.append(eot)
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(dest)
            return len(arr)

        n_train_toks = _tokenize_split("train", num_train_docs, train_path)
        n_val_toks = _tokenize_split("validation", num_val_docs, val_path)

    dt = time.perf_counter() - t0
    print(
        f"[data] Tokenized {n_train_toks:,} train tokens & {n_val_toks:,} val tokens "
        f"in {dt:.1f}s -> {out_dir}/"
    )
    return train_path, val_path


class MemmapDataset:
    """Zero-copy memory-mapped token dataset with O(1) random batch sampling."""

    def __init__(self, path: Path, seed: int = 42):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        self.rng = np.random.default_rng(seed)

    def get_batch(self, batch_size: int, seq_len: int) -> Tuple[mx.array, mx.array]:
        max_start = self.n_tokens - seq_len - 1
        starts = self.rng.integers(0, max_start, size=batch_size)
        offsets = starts[:, None] + np.arange(seq_len + 1)[None, :]
        chunk = self.data[offsets].astype(np.int32)
        x = mx.array(chunk[:, :-1], dtype=mx.int32)
        y = mx.array(chunk[:, 1:], dtype=mx.int32)
        return x, y
