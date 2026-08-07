#!/usr/bin/env bash
# test_dialogue.sh — 长句对话能力双形态测试
#   1) 句子层: transformer 自回归 (generate_text)
#   2) 概念链层: LAL 原生推理 (wte 余弦, --native-chain)
# 用法: bash test_dialogue.sh <ste模型路径> [ste_train二进制] [max_gen]
set -u
MODEL="${1:-model_dialogue_v2.ste}"
BIN="${2:-./ste_train}"
MAXGEN="${3:-40}"

echo "########################################################"
echo "# 模型: $MODEL   二进制: $BIN"
echo "########################################################"

echo ""
echo "########## [A] 句子层 (transformer 自回归, 完整中文答复) ##########"
for q in "你好" "鸟为什么天上飞？" "什么是火？" "天为什么是蓝色的？" "水为什么会结冰？" "怎么喝水？"; do
  echo ""
  echo "===== Q: $q ====="
  timeout 90 "$BIN" --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
    --prompt "$q" --max-gen "$MAXGEN" --temp 0.4 2>&1 \
    | sed -n '/=== Generation/,/^$/p' | head -6
done

echo ""
echo "########## [B] 概念链层 (LAL 原生推理, 不走前向) ##########"
for q in "鸟为什么天上飞？" "什么是火？" "天为什么是蓝色的？" "水为什么会结冰？" "猫是什么？"; do
  echo ""
  echo "===== Q: $q ====="
  timeout 60 "$BIN" --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
    --prompt "$q" --native-chain --native-topk 3 --native-depth 2 --no-generate 2>&1 \
    | sed -n '/=== LAL Native/,/嵌入空间检索/p' | head -20
done
