# 关系监督注入报告（v15）

## 问题
原 `logic_guided_regularization` 只监控 14 个**对立**概念对（热/冷、大/小…），
在 CORE 激活层把对立概念推开，但**不改 wte**、也**不推动相关概念靠近**。
导致 wte 余弦空间里 "鸟→动物" "火→光" 这类关系永远学不出来，
原生推理概念链退化为随机噪声（所有相似度挤在 0.5 附近）。

## 方案
在 `runtime/lal_whitebox_probe.h` 新增 `relation_logic_regularization()`：
- 定义 `relation_regs[]`：16 组（相关对 A-B，无关对照 C），覆盖
  类别（鸟/猫/鱼→动物，花/树→植物）、属性（火→热/光，水→冷/蓝，
  太阳→亮，冰/雪→冷）、因果（雨→水，风→云，山→花，人→山）。
- 直接在 **wte 空间**对概念行施加余弦监督：
  - 相关对 (A,B)：若 `cos(A,B) < 0.85` 则沿 `(B - cos·A)` 方向拉近。
  - 无关对 (A,C)：若 `cos(A,C) > 0.30` 则反向推远。
- BPE 模式下单概念 token（▁鸟=1051 等）独立，不像 byte-fallback 共享首字节，
  故直接改 wte 行安全（不同于 BUG #18 的 byte-level 污染场景）。

## 接入
`models/ste_train.c` 的 `logic_guided_regularization` 调用处（每 5 步执行）追加：
```c
float rel_loss = relation_logic_regularization(m, logic_lr * 0.5f);
```
lr 取 logic_lr 的 0.5 倍（直接在 wte 上加梯度需谨慎）。

## 验证
- 编译通过，`make ste_train` 产出 362KB 二进制。
- 8 步 smoke 训练无 NaN/崩溃。
- 正式训练 `model_dialogue_v3.ste`（1500 步, logic-lr=0.005）进行中。

## 预期效果
训练后原生推理概念链层应能给出有意义关联：
  鸟 → 动物（而非随机）
  火 → 热/光
  水 → 冷/蓝
  猫 → 动物
这恰是 LAL 原生推理（不走 transformer 前向）能体现"长句对话"语义关联的方式；
同时句子层（自回归）经更长步数后应能从"的"字高频进化为可读短句。
