#!/bin/bash
# lalmodel 训练脚本 — 在服务器上运行
# 用法: bash train.sh
# 前提: 已安装 gcc, python3, pip3 install sentencepiece

set -e
cd "$(dirname "$0")"

echo "=== LAL Model Training ==="
echo "时间: $(date)"
echo ""

# 1. 编译
echo "[1/4] 编译..."
make clean 2>/dev/null || true
make 2>&1 | tail -2
echo ""

# 2. 生成训练数据(如果不存在)
if [ ! -f data/general_bpe.bin ]; then
    echo "[2/4] 生成 BPE 训练数据..."
    # 克隆 laldata 获取真实语料(如果不存在)
    if [ ! -f data/general_corpus.jsonl ]; then
        echo "  下载真实语料..."
        git clone https://github.com/samaidev/laldata.git /tmp/laldata 2>/dev/null || true
        cp /tmp/laldata/data/general/general_corpus.jsonl data/ 2>/dev/null || true
        cp /tmp/laldata/data/stage_grounding/concept_boundary.jsonl data/ 2>/dev/null || true
        cp /tmp/laldata/data/commonsense/stage_commonsense.jsonl data/ 2>/dev/null || true
    fi
    python3 scripts/tokenize_data.py \
        --input data/general_corpus.jsonl \
        --output data/general_bpe.bin \
        --tokenizer tokenizer/chinese_bpe.model \
        --max_len 64
fi
echo ""

# 3. 训练 Phase 1 (35M) — 真实语料
echo "[3/4] 训练 Phase 1 (35M) on real corpus..."
echo "  配置: batch_size=4, lr=0.001, 500步"
echo ""

rm -f model.ste

# 第一阶段: lr=0.001, 200步
echo "  --- 阶段1: lr=0.001, 200步 ---"
./ste_train --steps 200 --batch-size 4 --lr 0.001 --logic-lr 0.002 \
    --phase 1 --vocab 32768 --data data/general_bpe.bin \
    --no-generate --save model.ste 2>&1 | grep -E "step.*loss=|STE training done"

# 第二阶段: lr=0.0005, 200步
echo "  --- 阶段2: lr=0.0005, 200步 ---"
./ste_train --steps 200 --batch-size 4 --lr 0.0005 --logic-lr 0.002 \
    --phase 1 --vocab 32768 --data data/general_bpe.bin \
    --no-generate --resume model.ste --save model.ste 2>&1 | grep -E "step.*loss=|STE training done"

# 第三阶段: lr=0.0002, 100步
echo "  --- 阶段3: lr=0.0002, 100步 ---"
./ste_train --steps 100 --batch-size 4 --lr 0.0002 --logic-lr 0.002 \
    --phase 1 --vocab 32768 --data data/general_bpe.bin \
    --no-generate --resume model.ste --save model.ste 2>&1 | grep -E "step.*loss=|STE training done"

echo ""

# 4. 生成测试
echo "[4/4] 生成测试..."
echo ""

for p in "火" "水" "太阳" "苹果" "中医" "花" "猫" "月亮" "下雨" "你好"; do
    IDS=$(python3 -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(); sp.Load('tokenizer/chinese_bpe.model'); print(','.join(map(str, sp.EncodeAsIds('$p'))))" 2>/dev/null)
    echo -n "  '$p' → "
    ./ste_train --diagnose-only --phase 1 --vocab 32768 \
        --resume model.ste --prompt-ids "$IDS" \
        --max-gen 30 --temp 0.01 2>/dev/null | grep "输出:" | head -1 | sed 's/输出: //'
    echo ""
done

echo ""
echo "=== 训练完成 $(date) ==="
echo "模型: model.ste"
