#!/usr/bin/env bash
# train_dialogue.sh — LAL 长句对话训练 · 唯一正确的路 (固化流程, 勿改参数)
#
# 为什么存在: 这个仓库的训练容易走岔路(参数拼错/数据用错/忘记原生推理),
# 导致浪费大量时间. 本脚本把 v5 验证过的正确流程写死, 克隆仓库的人只需:
#     bash train_dialogue.sh            # 训练 (默认 4000 步)
#     bash train_dialogue.sh test       # 用默认模型测双形态(句子层+概念链层)
#     bash train_dialogue.sh train 2000 # 自定义步数训练
#     bash train_dialogue.sh resume model_dialogue_v5.ste 2000 4000 6000 0.0005 0.003
#                                         # 从 v5 续训 2000 步 (start=4000 total=6000)
# 不要手动拼 ./ste_train 参数 —— 除非你明确知道在做什么.
#
# 固化参数 (v5 → v6 验证, 别改):
#   batch-size=4  lr=0.001 (新训) / 0.0005 (续训)  logic-lr=0.005 / 0.003
#   phase=0 (Seed 8L/448d)  vocab=32768  data=data/dialogue_bpe.bin  native_chain=ON
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-train}"
MODEL="model_dialogue.ste"

echo "=================================================="
echo " LAL 长句对话训练 — 唯一正确的路"
echo " 模式=$MODE  模型=$MODEL  数据=data/dialogue_bpe.bin"
echo "=================================================="

# ---- 阶段1: 确保数据存在 (对话+认知合并) ----
if [ ! -f data/dialogue_bpe.bin ]; then
    echo "[*] 生成对话数据..."
    python3 data/gen_dialogue_data.py --out data/dialogue_train.jsonl
    python3 - <<'PY'
srcs = ["data/cognitive_foundation.jsonl", "data/dialogue_train.jsonl"]
with open("data/train_dialogue_combined.jsonl", "w", encoding="utf-8") as fo:
    for s in srcs:
        try:
            with open(s, encoding="utf-8") as fi:
                for line in fi:
                    line = line.strip()
                    if line: fo.write(line + "\n")
        except FileNotFoundError:
            print(f"[!] 缺少 {s}, 跳过")
PY
    echo "[*] 分词为 dialogue_bpe.bin (vocab=32768, BPE)..."
    python3 scripts/tokenize_data.py \
        --input data/train_dialogue_combined.jsonl \
        --output data/dialogue_bpe.bin \
        --tokenizer tokenizer/chinese_bpe.model \
        --max_len 64
fi
echo "[*] 数据就绪: $(ls -la data/dialogue_bpe.bin | awk '{print $5}') bytes"

# ---- 阶段2: 编译 (如需要) ----
if [ ! -x ste_train ] || [ models/ste_train.c -nt ste_train ]; then
    echo "[*] 编译 ste_train..."
    make ste_train
fi

# ---- 阶段3: 训练 (参数固化, 勿改) ----
case "$MODE" in
    train)
        STEPS="${2:-4000}"
        echo "[*] 训练 $STEPS 步 (lr=0.001 logic-lr=0.005 phase=0 vocab=32768)..."
        ./ste_train --steps "$STEPS" --batch-size 4 --lr 0.001 --logic-lr 0.005 \
            --phase 0 --vocab 32768 --data data/dialogue_bpe.bin \
            --no-generate --save "$MODEL"
        echo "[*] 训练完成 -> $MODEL"
        ;;
    resume)
        # 用法: bash train_dialogue.sh resume <resume.ste> [steps] [start-step] [total-steps] [lr] [logic-lr]
        RESUME="${2:?usage: bash train_dialogue.sh resume <model.ste> [steps] [start] [total] [lr] [logic-lr]}"
        STEPS="${3:-2000}"
        START="${4:-4000}"
        TOTAL="${5:-6000}"
        LR="${6:-0.0005}"
        LLR="${7:-0.003}"
        echo "[*] 续训 $RESUME → $MODEL  steps=$STEPS  start=$START  total=$TOTAL  lr=$LR  logic-lr=$LLR"
        ./ste_train --resume "$RESUME" --steps "$STEPS" --start-step "$START" \
            --total-steps "$TOTAL" --batch-size 4 --lr "$LR" --logic-lr "$LLR" \
            --phase 0 --vocab 32768 --data data/dialogue_bpe.bin \
            --no-generate --save "$MODEL"
        echo "[*] 续训完成 -> $MODEL"
        ;;
    test)
        : # 在阶段4统一执行
        ;;
    *)
        echo "未知模式: $MODE  (合法: train | resume | test)" >&2
        exit 2
        ;;
esac

# ---- 阶段4: 测试三形态 (句子层 + 概念链层 + 关系推理层) ----
if [ "$MODE" = "test" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "########## 句子层 (transformer 自回归) ##########"
    for q in "你好" "鸟为什么天上飞？" "什么是火？"; do
        echo "--- Q: $q ---"
        ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
            --prompt "$q" --max-gen 30 --temp 0.4 2>&1 | sed -n '/=== Generation/,/^$/p' | head -5
    done
    echo ""
    echo "########## 概念链层 (LAL 原生推理, 默认开启) ##########"
    for q in "鸟为什么天上飞？" "什么是火？" "天为什么是蓝色的？"; do
        echo "--- Q: $q ---"
        ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
            --prompt "$q" --native-topk 3 --native-depth 2 --no-reason --no-generate 2>&1 \
            | sed -n '/=== LAL Native/,/嵌入空间检索/p' | head -16
    done
    echo ""
    echo "########## 关系推理层 (概念边界 + 相互关系 + 推演) ##########"
    echo "## 智慧 = 掌握概念边界和相互关系进行推理的能力"
    for q in "鸟为什么天上飞？" "什么是火？" "太阳怎么发光？"; do
        echo "--- Q: $q ---"
        ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
            --prompt "$q" --reason --reason-depth 3 --no-generate 2>&1 \
            | sed -n '/=== LAL Relationship/,/验证方式/p' | head -50
    done
fi
echo ""
echo "[*] Done. 模型=$MODEL"
