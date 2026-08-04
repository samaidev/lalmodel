#!/usr/bin/env python3
"""tokenize_data.py - SentencePiece BPE tokenizer -> LAL binary training data.

Reads one or more JSONL files, extracts the "text" field of each line,
tokenizes it with a SentencePiece BPE model, and writes a compact binary
file that is consumed by runtime/lal_data_loader.h.

Binary file format (little-endian):
    [magic     : 4 bytes  = "LALT"]
    [n_samples : int32]
    [n_vocab   : int32]
    then n_samples records of:
        [n_tokens  : int32]
        [token_ids : int32 * n_tokens]

Usage:
    python3 tokenize_data.py \\
        --input  data/train.jsonl data/stages/stage1_math.jsonl \\
        --output data/train_tokens.bin \\
        --tokenizer tokenizer/chinese_bpe.model \\
        --max_len 512

Options:
    --input      One or more JSONL files (each line: {"text": "...", ...}).
    --output     Output binary file path.
    --tokenizer  SentencePiece model path (.model).
    --max_len    Max tokens per sample; longer samples are truncated (default 512).
    --skip_long  Skip samples longer than max_len instead of truncating them.
    --add_eos    Append the EOS token id to every sample.
"""
import argparse
import json
import os
import struct
import sys

import sentencepiece as spm

MAGIC = b"LALT"


def iter_texts(jsonl_paths):
    """Yield the 'text' field from each non-empty line of the given JSONL files.

    Also prints per-file / overall progress to stderr.
    """
    n_lines = 0
    n_ok = 0
    for path in jsonl_paths:
        if not os.path.isfile(path):
            print(f"[!] File not found, skipping: {path}", file=sys.stderr)
            continue
        file_lines = 0
        file_ok = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                file_lines += 1
                n_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[!] {path}:{line_no}: invalid JSON, skipped",
                          file=sys.stderr)
                    continue
                # Accept either a JSON object with a "text" key, or a bare string.
                if isinstance(obj, dict):
                    text = obj.get("text")
                elif isinstance(obj, str):
                    text = obj
                else:
                    text = None
                if not text:
                    continue
                file_ok += 1
                n_ok += 1
                yield text
        print(f"[*] {path}: {file_lines} lines, {file_ok} with text")
    print(f"[*] Total: {n_lines} JSONL lines, {n_ok} with text")


def main():
    ap = argparse.ArgumentParser(
        description="Tokenize JSONL text with SentencePiece BPE -> LAL binary.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more JSONL files (each line: {\"text\": ...}).")
    ap.add_argument("--output", required=True,
                    help="Output binary file path.")
    ap.add_argument("--tokenizer", required=True,
                    help="SentencePiece model path (.model).")
    ap.add_argument("--max_len", type=int, default=512,
                    help="Max tokens per sample (default 512).")
    ap.add_argument("--skip_long", action="store_true",
                    help="Skip samples longer than max_len instead of truncating.")
    ap.add_argument("--add_eos", action="store_true",
                    help="Append EOS token to each sample.")
    args = ap.parse_args()

    if args.max_len <= 0:
        print("[!] --max_len must be positive", file=sys.stderr)
        sys.exit(2)

    # ---- load tokenizer ----
    if not os.path.isfile(args.tokenizer):
        print(f"[!] Tokenizer model not found: {args.tokenizer}", file=sys.stderr)
        sys.exit(1)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    vocab_size = sp.GetPieceSize()
    eos_id = sp.eos_id()
    unk_id = sp.unk_id()
    print(f"[*] Loaded tokenizer: {args.tokenizer}")
    print(f"    vocab_size={vocab_size}  eos_id={eos_id}  unk_id={unk_id}")

    # ---- tokenize ----
    samples = []          # list[list[int]]
    total_tokens = 0
    used_ids = set()
    n_truncated = 0
    n_skipped = 0
    max_len = args.max_len

    for text in iter_texts(args.input):
        ids = sp.EncodeAsIds(text)
        if args.add_eos and eos_id is not None and eos_id >= 0:
            ids = ids + [eos_id]
        if len(ids) > max_len:
            if args.skip_long:
                n_skipped += 1
                continue
            ids = ids[:max_len]
            n_truncated += 1
        if not ids:
            continue
        samples.append(ids)
        total_tokens += len(ids)
        used_ids.update(ids)

    n_samples = len(samples)
    if n_samples == 0:
        print("[!] No samples produced; nothing to write.", file=sys.stderr)
        sys.exit(1)

    # ---- write binary ----
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", n_samples))
        f.write(struct.pack("<i", vocab_size))
        for ids in samples:
            f.write(struct.pack("<i", len(ids)))
            if ids:
                f.write(struct.pack(f"<{len(ids)}i", *ids))

    file_size = os.path.getsize(args.output)

    # ---- statistics ----
    avg_len = total_tokens / n_samples
    vocab_usage = len(used_ids) / vocab_size * 100.0 if vocab_size else 0.0
    max_seen = max(len(s) for s in samples)
    min_seen = min(len(s) for s in samples)

    print()
    print("=" * 60)
    print("Tokenization statistics")
    print("=" * 60)
    print(f"  Output file       : {args.output}")
    print(f"  Tokenizer         : {args.tokenizer}")
    print(f"  Vocab size        : {vocab_size}")
    print(f"  Max length        : {max_len}")
    print(f"  Mode              : {'skip_long' if args.skip_long else 'truncate'}")
    print(f"  Total samples     : {n_samples}")
    print(f"  Truncated samples : {n_truncated}")
    print(f"  Skipped samples   : {n_skipped}")
    print(f"  Total tokens      : {total_tokens}")
    print(f"  Avg length        : {avg_len:.2f}")
    print(f"  Min / Max length  : {min_seen} / {max_seen}")
    print(f"  Unique token ids  : {len(used_ids)}")
    print(f"  Vocab usage       : {vocab_usage:.2f}% "
          f"({len(used_ids)}/{vocab_size})")
    print(f"  Binary file size  : {file_size} bytes "
          f"({file_size / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
