# LAL 长句对话训练 — 总结（v1→v4）

## 目标
训练 LAL 中文模型的长句对话能力：
- 句子层：`你好`→`你好`、`鸟为什么天上飞？`→`因为鸟有翅膀可以飞`
- 概念链层（LAL 原生推理，不走 transformer 前向）：`鸟→动物`、`火→热/光`

## 已完成
1. **仓库同步**：`lalmodel`(训练副本) → `lal`(主仓)，回灌 `ste_train.c`(v2 tokenizer+关系探针)、
   `merge_data.py`、认知种子数据，补 STE 训练 Makefile 目标，编译验证通过。
2. **数据管线**：`gen_dialogue_data.py` 生成 200 条长句对话（问候/因果/定义/方式/常识/闲聊），
   与 6004 条认知数据合并 → `tokenize_data.py` → `dialogue_bpe.bin`（6204 样本, vocab=32768）。
3. **双形态推理**：
   - 句子层：既有 `generate_text`（transformer 自回归）。
   - 概念链层：新增 `--native-chain`（wte 余弦直接检索，无前向），在 `models/ste_train.c`。
4. **关系监督注入（v15）**：`lal_whitebox_probe.h` 新增 `relation_logic_regularization`，
   对 16 组关系对（鸟→动物/火→热光/水→冷蓝/太阳→亮/冰雪→冷/雨→水/风→云…）在 wte 空间拉近相关、推远无关。
5. **关键 BUG 修复（v15b）**：`rel_get_wte_row` 原误取 `bpe_ids[0]=259`（▁ 共享前缀），
   改成多 token 取 `bpe_ids[1]`（真实概念 token），单 token 取 `bpe_ids[0]`。修复后关系正则真正生效。

## 训练演进
| 版本 | 步数 | logic-lr | RELATION PROBE | 概念链层 |
|------|------|----------|----------------|----------|
| v1 | 300 | 0.002 | FAIL | 全 0.5 噪声 |
| v2 | 1000 | 0.005 | WEAK/FAIL | 全 0.5 噪声 |
| v3 | 1500 | 0.005 | WEAK/FAIL | 全 0.5 噪声（关系正则未生效，BUG 所致）|
| **v4** | **1500** | **0.005** | **STRONG** ✅ | **火→热(0.81)/光(0.80)、鸟→动物(0.78)、蓝→水(0.80)/冷(0.60)** ✅ |

产物：`model_dialogue_v4.ste`（154MB）。

## 当前能力
- **LAL 原生推理概念链（纯 wte 余弦）真正可用**：问"鸟/火/蓝"等概念，能检索出正确关联，
  完全不走 transformer 前向，符合"用 LAL 原生推理"的要求。
- 句子层自回归能学到中文局部字频（"的"高频），但**尚未组成完整可读长句**。

## 剩余短板
1. **句子层**：CE loss 仍停在 avg≈3，自回归未产出完整句子。需更长课程学习（数千步）或
   把对话 Q&A 的因果结构也纳入监督。
2. **对立对边界**（热vs冷）仍弱：旧 `logic_guided_regularization` 未充分推开，与关系对是正交机制。

## 群内同步
已在 AICQ 房间（invite f0af80e3）同步：初始进展汇报 + v4 BUG 修复验证更新。
房间当前另一成员 `1000008` 尚未发言。脚本：`aicq_join.py`(join+问候)、`aicq_post.py`(发消息+拉增量)。

## 复现命令
```bash
cd lal
make ste_train                       # 编译（含原生推理 + 关系正则）
python3 data/gen_dialogue_data.py   # 生成对话数据
# 合并+分词见 gen 脚本说明，产物 data/dialogue_bpe.bin
./ste_train --steps 1500 --batch-size 4 --lr 0.001 --logic-lr 0.005 \
    --phase 0 --vocab 32768 --data data/dialogue_bpe.bin \
    --no-generate --save model_dialogue_v4.ste
# 概念链层测试
./ste_train --diagnose-only --phase 0 --vocab 32768 --resume model_dialogue_v4.ste \
    --prompt "鸟为什么天上飞？" --native-chain --native-topk 3 --native-depth 2 --no-generate
# 句子层测试
./ste_train --diagnose-only --phase 0 --vocab 32768 --resume model_dialogue_v4.ste \
    --prompt "你好" --max-gen 30 --temp 0.4
```
