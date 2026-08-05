#!/bin/bash
# lalmodel 训练脚本 — v14: 新tokenizer + 修复残差归一化
# 用法: bash train.sh
# 前提: 已安装 gcc, python3, pip3 install sentencepiece

set -e
cd "$(dirname "$0")"

echo "=== LAL Model Training (v14: fixed tokenizer + residual norm) ==="
echo "时间: $(date)"
echo ""

# Tokenizer 和数据路径
TOKENIZER_MODEL="tokenizer/chinese_bpe_v2.model"
TOKENIZER_VOCAB="tokenizer/chinese_bpe_v2.vocab"
DATA_BIN="data/general_v2_bpe.bin"

# 1. 编译
echo "[1/4] 编译..."
make clean 2>/dev/null || true
make 2>&1 | tail -2
echo ""

# 2. 生成训练数据(如果不存在)
if [ ! -f "$DATA_BIN" ] || [ ! -f "$TOKENIZER_MODEL" ]; then
    echo "[2/4] 生成新 tokenizer + BPE 训练数据..."
    # 训练新 tokenizer (character_coverage=0.9995, CJK覆盖率 3.8% -> 84.8%)
    if [ ! -f "$TOKENIZER_MODEL" ]; then
        echo "  训练新 tokenizer..."
        python3 scripts/train_tokenizer.py \
            --input data/large_merged.jsonl data/cognitive_foundation.jsonl \
                   data/concept_boundary.jsonl data/stage_commonsense.jsonl \
            --output tokenizer/chinese_bpe_v2 \
            --vocab_size 12227 \
            --model_type unigram \
            --character_coverage 0.9995
    fi
    # 重新编码训练数据
    if [ ! -f "$DATA_BIN" ]; then
        echo "  重新编码训练数据..."
        python3 scripts/tokenize_data.py \
            --input data/large_merged.jsonl data/cognitive_foundation.jsonl \
                   data/concept_boundary.jsonl data/stage_commonsense.jsonl \
            --output "$DATA_BIN" \
            --tokenizer "$TOKENIZER_MODEL" \
            --max_len 64 \
            --add_eos
    fi
fi
echo ""

# 3. 训练 Phase 1 (35M) — 真实语料
echo "[3/4] 训练 Phase 1 (35M) on real corpus..."
echo "  配置: batch_size=4, lr=0.001, 500步"
echo "  Tokenizer: $TOKENIZER_MODEL (12227 vocab, 0.3% byte-fallback)"
echo ""

rm -f model.ste

# 第一阶段: lr=0.001, 200步
echo "  --- 阶段1: lr=0.001, 200步 ---"
./ste_train --steps 200 --batch-size 4 --lr 0.001 --logic-lr 1.0 \
    --phase 1 --vocab 32768 --data "$DATA_BIN" \
    --tokenizer "$TOKENIZER_VOCAB" \
    --no-generate --save model.ste 2>&1 | grep -E "step.*loss=|STE training done|Loaded|residual"

# 第二阶段: lr=0.0005, 200步
echo "  --- 阶段2: lr=0.0005, 200步 ---"
./ste_train --steps 200 --batch-size 4 --lr 0.0005 --logic-lr 1.0 \
    --phase 1 --vocab 32768 --data "$DATA_BIN" \
    --tokenizer "$TOKENIZER_VOCAB" \
    --no-generate --resume model.ste --save model.ste 2>&1 | grep -E "step.*loss=|STE training done|Loaded|residual"

# 第三阶段: lr=0.0002, 100步
echo "  --- 阶段3: lr=0.0002, 100步 ---"
./ste_train --steps 100 --batch-size 4 --lr 0.0002 --logic-lr 1.0 \
    --phase 1 --vocab 32768 --data "$DATA_BIN" \
    --tokenizer "$TOKENIZER_VOCAB" \
    --no-generate --resume model.ste --save model.ste 2>&1 | grep -E "step.*loss=|STE training done|Loaded|residual"

echo ""

# 4. 生成测试
echo "[4/4] 生成测试..."
echo ""

for p in "火" "水" "太阳" "苹果" "中医" "花" "猫" "月亮" "下雨" "你好"; do
    IDS=$(python3 -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(); sp.Load('$TOKENIZER_MODEL'); print(','.join(map(str, sp.EncodeAsIds('$p'))))" 2>/dev/null)
    echo -n "  '$p' → "
    ./ste_train --diagnose-only --phase 1 --vocab 32768 \
        --tokenizer "$TOKENIZER_VOCAB" \
        --resume model.ste --prompt-ids "$IDS" \
        --max-gen 30 --temp 0.01 2>/dev/null | grep "输出:" | head -1 | sed 's/输出: //'
    echo ""
done

echo ""
echo "=== 训练完成 $(date) ==="
echo "模型: model.ste"
