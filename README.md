# lalmodel — LAL 中文 LLM 训练与推理

专门用于 LAL (Logic-Augmented Learning) 中文语言模型的训练,包含完整的代码和数据。

## 快速开始

```bash
# 1. 编译
make

# 2. 生成训练数据(BPE tokenize)
make data/cognitive_bpe.bin

# 3. 训练 500 步
make train

# 4. 生成文本
make generate
```

## 训练

```bash
# 从头训练
./ste_train --steps 500 --batch-size 4 --lr 0.001 --logic-lr 0.002 \
    --phase 0 --vocab 32768 --data data/cognitive_bpe.bin \
    --no-generate --save model.ste

# 继续训练(从检查点)
./ste_train --steps 500 --batch-size 4 --lr 0.0005 --logic-lr 0.002 \
    --phase 0 --vocab 32768 --data data/cognitive_bpe.bin \
    --no-generate --resume model.ste --save model_1000.ste
```

## 推理

```bash
# 生成文本(prompt 用 BPE token id)
./ste_train --diagnose-only --phase 0 --vocab 32768 \
    --resume model.ste --prompt-ids "32646" --max-gen 20 --temp 0.01

# 获取 prompt 的 BPE id
python3 -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(); sp.Load('tokenizer/chinese_bpe.model'); print(','.join(map(str, sp.EncodeAsIds('火'))))"
```

## 数据

### 认知基础数据 (`data/cognitive_foundation.jsonl`)

1946 条独特样本,13 个类别,按认知发展顺序组织:

| 类别 | 数量 | 示例 |
|------|------|------|
| counting | 720 | 一个苹果就是1个苹果 |
| number | 306 | 一就是1。1就是一。 |
| math | 303 | 一加一等于二。二就是2。 |
| antonym | 150 | 大的反义词是小 |
| sentence | 131 | 这是苹果。苹果是红色的 |
| color | 85 | 苹果是红色的 |
| comparison | 52 | 一比二少。1比2少 |
| property | 46 | 苹果是红的、圆的、甜的 |
| commonsense | 40 | 火很热，不能用手摸 |
| qa | 36 | 问：火是什么？火是热的 |
| spatial | 32 | 鸟在上面飞，鱼在下面游 |
| action | 25 | 我吃饭。饭在碗里 |
| nature | 20 | 下雨了。雨从天上落下来 |

### 重新生成数据

```bash
python3 scripts/gen_cognitive_data.py
```

## 模型架构

- **Phase 0**: 8层, 448d, ~22M 参数
- **Phase 1**: 10层, 512d, ~35M 参数
- **BPE vocab**: 32768 (每个中文字独立 token)
- **STE**: 前向用浮点,反向用 Straight-Through Estimator
- **LAL-aware Adam**: CORE 神经元 3x 学习率,分组二阶矩
- **CORE/BINARY/PRUNE**: 三类神经元逻辑引导

## 训练方法

```
前向:  CORE = w_float * core_gain * K     (精确计算)
       BINARY = sign(w) * alpha * K       (二值符号)
       PRUNE = 0                           (静默剪枝)
反向:  STE through w_float (CORE+BINARY)
更新:  LAL-aware Adam (分组二阶矩, CORE 3x lr)
引导:  CORE → 差异化反义词 | BINARY → 收敛共性 | PRUNE → 静默
```

## 生成策略

- **重复惩罚**: 已生成 token 的 logit 除以 1.5
- **Top-p 采样**: 累积概率达到 0.9 的 token 集合内采样
- **argmax 模式**: temp=0.01 时用 argmax + 重复惩罚

## 目录结构

```
lalmodel/
├── Makefile                    # 编译和训练入口
├── README.md                   # 本文档
├── runtime/                    # LAL 运行时(C 语言)
│   ├── lal_runtime.h/c         # 核心运行时(模型前向/反向/Adam)
│   ├── lal_data_loader.h       # LALT 二进制数据加载器
│   ├── lal_model_growth.h      # 模型生长阶段配置
│   ├── lal_semantic_logic.h    # CORE/BINARY/PRUNE 逻辑分配
│   ├── lal_whitebox_probe.h    # 白箱探针(语义分析)
│   └── lal_inference_tracer.h  # 推理追踪
├── models/
│   └── ste_train.c             # STE 训练 + 推理入口
├── tokenizer/
│   ├── chinese_bpe.model       # 32k BPE 中文分词器
│   └── chinese_bpe.vocab       # 分词器词表
├── data/
│   ├── cognitive_foundation.jsonl  # 1946 条认知基础数据
│   ├── concept_boundary.jsonl      # 189 条概念边界数据
│   └── stage_commonsense.jsonl     # 883 条常识数据
└── scripts/
    ├── tokenize_data.py            # JSONL → BPE 二进制
    └── gen_cognitive_data.py       # 生成认知数据
```

## 训练进展

| 步数 | loss | 生成示例 |
|------|------|----------|
| 100 | 3.55 | "燃烧生成"(新组合) |
| 500 | 2.55 | "产生水""桌子很明亮""wǒ huà" |
| 700 | 3.06 | "是""很""电阻""爱老师""寓于" |

## 依赖

- gcc (支持 -march=native)
- Python 3 + sentencepiece
- 无需 GPU,纯 CPU 训练
