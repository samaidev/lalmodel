# LAL — Logic-Assembly Language

A minimal CPU LLM inference engine in ~2000 lines of C. No dependencies, no abstractions, no GPU. Just hand-written SIMD kernels.

**84% of llama.cpp's speed, 2% of the code.**

---

## 30-second quickstart (Qwen2.5-7B-Instruct)

```bash
git clone https://github.com/samaidev/lal.git && cd lal
make qwen7b-server                    # build (auto-detects AVX2/AVX512)
./prebuilt/qwen7b_server --weights prebuilt/qwen7b_q4k_weights.bin \
  --prompt "Hello" --n 40 --threads 2  # run
```

Pre-built weights (7.5 GB) are **not** in the repo. Convert from HuggingFace:
```bash
python3 scripts/convert_qwen7b_q4k.py  # reads /root/qwen7b, outputs prebuilt/qwen7b_q4k_weights.bin
```

---

## What LAL can do

- ✅ Run Qwen2.5-7B-Instruct at **1.4 tok/s** on 2-vCPU Xeon (no GPU)
- ✅ 5 quantization formats: Q8, Q4_0, Q8_0, Q4_0A, Q4_K (llama.cpp-compatible)
- ✅ Hand-tuned AVX2 kernels with 256-bit maddubs, prefetch, 8-row parallelism
- ✅ Full BPE tokenizer, chat template, top-k sampling, repetition penalty
- ✅ Zero external dependencies (only libc + libm + libgomp)
- ✅ ~200KB binary, mmap-based memory, <100ms cold start
- ✅ **从零训练**: 语义结构驱动的课程学习 (CORE/BINARY/PRUNE 逻辑引导层)
- ✅ **LAL-aware Adam**: 分组归一化优化器，防止 CORE/BINARY/PRUNE 结构退化
- ✅ **白箱监控**: 每 10 步直接观测逻辑电路（概念边界/CORE分化/PRUNE静默）
- ✅ **渐进式生长**: 22M → 35M → 50M → 68M 参数，随语义理解增长
- ✅ **长上下文**: 滑动窗口注意力 + Attention Sinks，支持 1M+ token
- ✅ **仿海马体记忆**: 三层记忆系统（工作/情景/语义）+ 艾宾浩斯遗忘曲线
- ✅ **元学习**: LoRA 适配器层，边推理边学习，基础权重冻结

## What LAL cannot do (be honest)

- ❌ Beat llama.cpp in speed (we're 84%, not 100%)
- ❌ Support GPU/Metal/NPU (CPU only, by design)
- ❌ Support 100+ models (only Qwen2.5-7B + GPT-2 out of the box)
- ❌ Continuous batching / concurrent requests (single-stream inference)
- ❌ KV cache quantization (448MB KV cache is F32)
- ❌ Importance matrix (imatrix) quantization (simple min-max only)

**LAL is not a llama.cpp replacement. It's a minimal, readable, hackable reference.**

---

## Training from Scratch — 语义结构驱动训练

LAL 不只是推理引擎——它还能从零训练模型。训练围绕 LAL 的核心设计：
**CORE/BINARY/PRUNE 逻辑引导层本身就是语义结构机制**。

### CORE / BINARY / PRUNE 三层语义逻辑

| 状态 | 精度 | 语义角色 |
|------|------|----------|
| **CORE** | 全精度浮点 | 核心概念（热/冷、大/小）——需要精确区分 |
| **BINARY** | ±1 二值近似 | 一般语义关系——够用即可 |
| **PRUNE** | 剪枝归零 | 噪声/无关连接——直接移除 |

训练中渐进式激活：早期 40% PRUNE（极度稀疏，只学核心）→ 后期 10% PRUNE（接近全容量）。

### 快速开始训练

```bash
# 编译训练脚本（LAL-aware Adam + 白箱监控 + 语义正则化）
make web-train

# 启动语义门控课程学习
./web_model_v3 --curriculum --generate --batch-size 4

# 仅生成文本
./web_model_v3 --generate --weights /tmp/lal_web_model_v3.bin --prompt "火"
```

### 课程学习阶段

```
Stage -1: 感知 (Grounding)  → 物理/感官/因果 (5000步, LR=0.0008)
Stage  0: 启蒙 (Basics)      → 基础文字/数字   (4000步, LR=0.0005)
Stage  1: 小学 (Primary)     → 语文/数学入门   (4000步, LR=0.0004)
Stage  2: 初中 (Middle)      → 理化生/历史     (4000步, LR=0.0003)
Stage  3: 高中 (Senior)      → 深度学科        (3500步, LR=0.0002)
Stage  4: 大学 (College)     → 专业/编程       (2500步, LR=0.0001)
```

每个阶段必须通过**语义门控**才能进入下一阶段（5 级评估：字节连贯 → 字符识别 → 概念边界 → 语义关系 → 常识结构）。

### 模型渐进式生长

| 阶段 | 层数 | 嵌入维度 | 参数量 | CORE/BINARY/PRUNE |
|------|------|----------|--------|-------------------|
| Phase 0 | 8 | 448 | ~22M | 10% / 50% / 40% |
| Phase 1 | 10 | 512 | ~35M | 15% / 60% / 25% |
| Phase 2 | 12 | 576 | ~50M | 20% / 65% / 15% |
| Phase 3 | 14 | 640 | ~68M | 20% / 70% / 10% |

### 长句对话训练 · 唯一正确的路（v5/v6 验证流程，参数已固化）

```bash
# 一行命令搞定一切（生成数据 → 训练 → 双形态测试）
bash train_dialogue.sh            # 默认 4000 步从头训
bash train_dialogue.sh train 2000 # 自定义步数从头训

# 从已有模型续训（v6 验证：低 lr 续训不会再回退）
bash train_dialogue.sh resume model_dialogue_v5.ste 2000 4000 6000 0.0005 0.003

# 用最终模型跑双形态验证（句子层 + 概念链层原生推理都已默认开启）
bash train_dialogue.sh test
```

固化参数（v5/v6 实证，别改）：batch=4 · phase=0 (Seed 8L/448d) · vocab=32768 ·
data=`data/dialogue_bpe.bin` · native_chain=ON。
详细演进记录见 [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)。

📖 **完整训练方法论请阅读 [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**

---

## Code structure

```
runtime/
  lal_q4k_kernel.h        — Q4_K matmul kernel (256-bit maddubs, 8-row parallel, prefetch)
  lal_q8_kernel.h         — Q8 matmul + LM head (sign-trick int8 dot product)
  lal_tokenizer.h         — BPE tokenizer (byte-level, GPT-2 style)
  lal_runtime.c           — weight loading, forward/backward, logic-guided binarization
  lal_runtime.h           — ModelConfig, global flags, API declarations
  lal_weight_utils.h      — per-row quantization helpers
  lal_data_loader.h       — LALT binary training data loader
  lal_semantic_gate.h     — 5-level semantic evaluation gate system
  lal_model_growth.h      — progressive model growth (22M→68M)
  lal_whitebox_probe.h    — whitebox probing + semantic regularization (CORE/BINARY guidance)
  lal_semantic_logic.h    — CORE/BINARY/PRUNE ratio configuration
  lal_memory.h            — hippocampus-inspired 3-tier memory system
  lal_meta_learn.h        — LoRA-style online learning adapter layers
  lal_search.h            — tree search / beam search for inference

models/
  web_model_v3.c          — main training script (semantic-gated curriculum)
  web_model_v2.c          — batch training with curriculum learning
  meta_learn.c            — meta-learning model (few-shot online adaptation)
  gpt2.c                  — GPT-2 inference
  coder.c                 — code generation model
  train_optimized.c       — optimized training with distillation

scripts/
  convert_qwen7b_q4k.py   — BF16 safetensors → LAL Q4_K format
  gen_grounding_data.py   — physical world grounding data generation
  gen_concept_boundary.py — concept boundary data (热/冷, 大/小 pairs)
  combine_lal_data.py     — merge multiple LALT data files
  q4k_unit_test.c         — correctness test for Q4_K kernel
```

---

## Adding a new model (template)

1. **Copy `qwen7b_server.c` → `yourmodel_server.c`**
2. Change these constants at the top:
   ```c
   #define N_LAYER  28    // your model's layer count
   #define N_EMBD   3584  // hidden dim
   #define N_HEAD   28    // attention heads
   #define N_KV_HEAD 4    // GQA kv heads
   #define HEAD_DIM  128  // 3584/28
   #define MLP_DIM   18944 // intermediate size
   #define VOCAB_SIZE 152064
   ```
3. Adjust the chat template in `encode_prompt()` (each model has different special tokens)
4. Write a converter: `scripts/convert_yourmodel_q4k.py` (copy from `convert_qwen7b_q4k.py`)
5. Add a Makefile target:
   ```makefile
   yourmodel-server: prebuilt/yourmodel_server
   prebuilt/yourmodel_server: tools/server/yourmodel_server.c runtime/*.h
       $(CC) $(CFLAGS) -fopenmp -o $@ tools/server/yourmodel_server.c runtime/lal_runtime.c -lm -lgomp
   ```

See `docs/ADDING_MODELS.md` for a detailed walkthrough.

---

## Porting to new hardware (guide)

LAL is designed to be portable. The only platform-specific code is in the kernel headers.

### Step 1: Identify your SIMD
```c
// In each kernel header, check for platform:
#if defined(__AVX2__) && defined(__F16C__)
    // x86 with AVX2
#elif defined(__ARM_NEON)
    // ARM with NEON
#elif defined(__riscv_v)
    // RISC-V vector extension
#else
    // scalar fallback (always works, ~10x slower)
#endif
```

### Step 2: Implement 3 core operations
You only need to implement these for your platform:

1. **`lal_matmul_q4_k`** in `lal_q4k_kernel.h` — the Q4_K matmul kernel
2. **`lal_lm_head_int8_range`** in `lal_q8_kernel.h` — the LM head dot product
3. **`unpack_scales_6bit`** — 12-byte packed scales → 16 uint8

Everything else (attention, RoPE, sampling, tokenizer) is platform-independent C.

### Step 3: Test
```bash
gcc -O3 -march=yourarch -I. -o test scripts/q4k_unit_test.c -lm
./test  # should print PASS
```

### Porting effort estimate
| Platform | Effort | Notes |
|----------|--------|-------|
| x86 AVX2 | ✅ Done | Current implementation |
| x86 AVX512 | ✅ Done | Kernel exists, slower due to downclocking |
| ARM NEON | ~1 day | Similar to AVX2, use `vmlal_s8` |
| RISC-V RVV | ~3 days | Vector extension, no maddubs equivalent |
| Custom NPU | 1-2 weeks | Depends on instruction set |

---

## Key Training Lessons (实战经验)

1. **LAL 的语义结构不是外挂的** — CORE/BINARY/PRUNE 就是模型本身的语义机制
2. **STE 模式训练（非纯浮点）** — `g_use_ste=1`，训练=推理，直接学二值逻辑
3. **LAL-aware Adam 防退化** — 分组归一化保留 CORE/BINARY 梯度差异（默认开启）
4. **语义正则化要够强** — alpha=2.0, lr=0.002, sqrt 归一化（曾因梯度 4e-8 停滞）
5. **白箱监控每 10 步** — LAL 是白箱，直接观测逻辑电路而非只看 loss
6. **先学简单概念再学复杂知识** — 语义门控确保模型在掌握基础前不冒进
7. **参数要可变生长** — 固定大小模型无法从大量数据中收敛
8. **residual_scale 必须为 1.0** — 0.5 导致 8 层模型信号衰减为 0.8%
9. **多位置预测** — 每样本 4 个预测点，梯度信号 4 倍
10. **逻辑掩码渐进激活** — 早期高 PRUNE → 后期低 PRUNE
11. **全 softmax 防模式崩溃** — 不用采样 softmax
12. **wte 嵌入必须更新** — 否则概念边界无法形成
13. **KV 缓存必须在 model_load 前分配** — 否则注意力退化为 V-copy
14. **字节级分词器需要 max_pos ≥ 64** — 中文字符占 3 字节

---

## Documentation

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** — 完整训练方法论（语义结构驱动课程学习）
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Why the code is written this way (optimization journey, design decisions)
- **[PITFALLS.md](docs/PITFALLS.md)** — Bugs we hit and how to avoid them (uint64 overflow, packing formats, AVX512 downclocking...)
- **[MINIMAL.md](docs/MINIMAL.md)** — How to add a new quantization format by changing 1 file
- **[HARDWARE_TEST_REPORT.md](docs/HARDWARE_TEST_REPORT.md)** — Benchmark results vs llama.cpp
- **[ADDING_MODELS.md](docs/ADDING_MODELS.md)** — Step-by-step guide for new models

---

## Project status

⚠️ **Experimental. Author does not guarantee continued maintenance.**

This is a research/educational project. It works, it's tested, but it's not production-hardened.

- **Forks welcome.** If you want to maintain a fork, go ahead.
- **PRs welcome.** They'll be merged if they pass tests and don't break existing formats.
- **Issues.** Bug reports prioritized over feature requests.
- **No co-maintainers actively sought**, but if you've contributed meaningfully and want write access, ask.

**If this project dies, that's okay.** The code is here, the docs explain why, the git history tells the story. Future developers can fork it, learn from it, or ignore it. That's fine.

---

## License

MIT. Do whatever you want with it. No warranty. If you build something useful, a mention is appreciated but not required.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — for proving CPU LLM inference is viable, and for the Q4_K format design
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) — for the Q8 sign-trick SIMD pattern
- [Qwen Team](https://qwenlm.github.io/) — for the excellent Qwen2.5 model
- [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) — for attention sinks concept
- [Net2Net](https://arxiv.org/abs/1511.05641) — for function-preserving network growth
