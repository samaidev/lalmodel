#!/usr/bin/env python3
"""合并所有中文训练数据, 拼接短句成长段落.
目标: 让模型见过 128+ token 的连贯文本, 而不是孤立短句.
"""
import json, os, random
import sentencepiece as spm

random.seed(42)

all_texts = []

# 1. lalmodel 认知数据
for fn in ['data/cognitive_foundation.jsonl', 'data/concept_boundary.jsonl', 'data/stage_commonsense.jsonl']:
    if os.path.exists(fn):
        with open(fn) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    t = d.get('text', '').strip()
                    if t and len(t) >= 4:
                        all_texts.append(t)
                except: pass

# 2. lal 短句数据
for fn in ['lal/data/train.jsonl', 'lal/data/stages/stage0_basics.jsonl']:
    if os.path.exists(fn):
        with open(fn) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    t = d.get('text', '').strip()
                    if t and len(t) >= 4:
                        all_texts.append(t)
                except: pass

# 3. lal 长推理数据 (编程问答)
prog_texts = []
for fn in ['lal/data/stages/prog_stage1.jsonl', 'lal/data/stages/prog_stage2.jsonl']:
    if os.path.exists(fn):
        with open(fn) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    t = d.get('text', '').strip()
                    if t and len(t) >= 20:
                        prog_texts.append(t)
                except: pass

print(f"短句样本: {len(all_texts)}")
print(f"长推理样本: {len(prog_texts)}")

sp = spm.SentencePieceProcessor()
sp.Load('tokenizer/chinese_bpe.model')

# 拼接短句成 ~128 token 段落
TARGET_TOKENS = 128
random.shuffle(all_texts)
merged_texts = []
current = []
current_tokens = 0
for text in all_texts:
    toks = sp.EncodeAsIds(text)
    if current_tokens + len(toks) > TARGET_TOKENS:
        if current:
            merged_texts.append(' '.join(current))
        current = [text]
        current_tokens = len(toks)
    else:
        current.append(text)
        current_tokens += len(toks)
if current:
    merged_texts.append(' '.join(current))

merged_texts.extend(prog_texts)
random.shuffle(merged_texts)

all_lens = [len(sp.EncodeAsIds(t)) for t in merged_texts]
print(f"\n拼接后样本: {len(merged_texts)}")
print(f"总 tokens: {sum(all_lens)} ({sum(all_lens)/1000:.0f}K)")
print(f"平均: {sum(all_lens)//len(all_lens)}, 最长: {max(all_lens)}")

with open('data/large_merged.jsonl', 'w') as f:
    for i, t in enumerate(merged_texts):
        f.write(json.dumps({"id": f"lm-{i:05d}", "text": t}, ensure_ascii=False) + '\n')
print(f"写入 data/large_merged.jsonl")
