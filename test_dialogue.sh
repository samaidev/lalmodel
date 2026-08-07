#!/usr/bin/env bash
# test_dialogue.sh — 长句对话能力三形态测试
#   1) 句子层: transformer 自回归 (generate_text)
#   2) 概念链层: LAL 原生推理 (wte 余弦, --native-chain)
#   3) 关系推理层: LAL 关系推理引擎 (概念边界 + 相互关系 + 推演, --reason)
# 用法: bash test_dialogue.sh <ste模型路径> [ste_train二进制] [max_gen]
set -u
MODEL="${1:-model_dialogue.ste}"
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

echo ""
echo "########## [C] 关系推理层 (概念边界 + 相互关系 + 推演) ##########"
echo "## 智慧 = 掌握概念边界和相互关系进行推理的能力"
echo "## 用因果/并行/主被动/串行等关系类型, 结合 CORE 激活验证"
for q in "鸟为什么天上飞？" "什么是火？" "太阳怎么发光？" "天为什么是蓝色的？" "什么是水？"; do
  echo ""
  echo "===== Q: $q ====="
  timeout 60 "$BIN" --diagnose-only --phase 0 --vocab 32768 --resume "$MODEL" \
    --prompt "$q" --reason --reason-depth 3 --no-generate 2>&1 \
    | sed -n '/=== LAL Relationship/,/\[*\] Done/p' | head -60
done
