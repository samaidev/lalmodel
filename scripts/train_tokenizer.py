#!/usr/bin/env python3
"""train_tokenizer.py - Train a SentencePiece tokenizer with high CJK coverage.

The previous tokenizer only covered 3.8% of CJK characters (799/20992),
causing 24% of training tokens to be byte-fallback. This script trains
a new tokenizer with character_coverage=0.9995 to fix this.

Usage:
    python3 scripts/train_tokenizer.py \\
        --input  data/large_merged.jsonl data/cognitive_foundation.jsonl \\
        --output tokenizer/chinese_bpe_v2.model \\
        --vocab_size 32768
"""
import argparse
import os
import sys
import tempfile

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer with high CJK coverage.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="Input JSONL/text files for training the tokenizer.")
    ap.add_argument("--output", required=True,
                    help="Output .model file path (prefix, no extension needed).")
    ap.add_argument("--vocab_size", type=int, default=32768,
                    help="Vocabulary size (default 32768).")
    ap.add_argument("--model_type", default="unigram",
                    choices=["unigram", "bpe"],
                    help="Model type (default unigram).")
    ap.add_argument("--character_coverage", type=float, default=0.9995,
                    help="Character coverage (default 0.9995).")
    ap.add_argument("--max_sentence_length", type=int, default=8192,
                    help="Max sentence length (default 8192).")
    args = ap.parse_args()

    # Create a temporary concatenated text file (SentencePiece needs raw text input)
    # Extract text from JSONL files
    tmp_text = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8")
    
    import json
    n_lines = 0
    for fpath in args.input:
        if not os.path.isfile(fpath):
            print(f"[!] File not found: {fpath}", file=sys.stderr)
            continue
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
                except json.JSONDecodeError:
                    text = line
                if text:
                    tmp_text.write(text + "\n")
                    n_lines += 1
    tmp_text.close()
    print(f"[*] Wrote {n_lines} lines to temp file: {tmp_text.name}")

    # Determine model prefix (strip .model extension if present)
    model_prefix = args.output
    if model_prefix.endswith(".model"):
        model_prefix = model_prefix[:-6]
    
    out_dir = os.path.dirname(os.path.abspath(model_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Training SentencePiece tokenizer:")
    print(f"    model_type={args.model_type}")
    print(f"    vocab_size={args.vocab_size}")
    print(f"    character_coverage={args.character_coverage}")
    print(f"    output_prefix={model_prefix}")
    print()

    spm.SentencePieceTrainer.train(
        input=tmp_text.name,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        max_sentence_length=args.max_sentence_length,
        # Byte fallback: encode unknown chars as UTF-8 bytes
        byte_fallback=True,
        # Split digits and whitespace
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        # Normalize: NFKC for consistent encoding
        normalization_rule_name="nmt_nfkc",
        # Don't add dummy prefix (we handle it ourselves)
        add_dummy_prefix=True,
        # Special tokens
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        # Limit on number of unique characters
        # (with character_coverage=0.9995, this should include almost all CJK)
        train_extremely_large_corpus=False,
    )

    os.unlink(tmp_text.name)

    # Verify the new tokenizer
    print()
    print("=" * 60)
    print("Tokenizer verification")
    print("=" * 60)
    
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    vocab_size = sp.GetPieceSize()
    print(f"Vocab size: {vocab_size}")

    # Count CJK coverage
    cjk_count = 0
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        clean = piece.replace("▁", "")
        for ch in clean:
            if "\u4e00" <= ch <= "\u9fff":
                cjk_count += 1
                break
    print(f"Tokens with CJK: {cjk_count} ({cjk_count/vocab_size*100:.1f}%)")

    # Test common words
    test_words = ["火", "水", "太阳", "苹果", "中医", "花", "猫", "月亮",
                  "下雨", "你好", "码", "的", "是", "人", "他", "她",
                  "感", "接", "教", "技", "中医", "编程", "学习"]
    print("\nTokenization test:")
    total_bf = 0
    total_toks = 0
    for w in test_words:
        ids = sp.EncodeAsIds(w)
        pieces = sp.EncodeAsPieces(w)
        bf = any(sp.IsUnknown(i) or (len(p) > 2 and p.startswith("<0x")) 
                 for i, p in zip(ids, pieces))
        bf_count = sum(1 for i in ids if 3 <= i <= 258)
        total_bf += bf_count
        total_toks += len(ids)
        status = "BYTE-FALLBACK" if bf_count > 0 else "OK"
        print(f"  {w}: {ids} -> {pieces} [{status}]")
    
    print(f"\nByte-fallback rate (test): {total_bf}/{total_toks} = {total_bf/total_toks*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
