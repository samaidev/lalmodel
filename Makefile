# LAL — Logic-Assembly Language
PYTHON ?= python3
CC ?= gcc

# Auto-detect SIMD flags: AVX2+FMA on x86_64, NEON on ARM, nothing elsewhere.
# The gpt2_server.c file uses a portable v8f wrapper that picks the right
# intrinsics per platform (see top of that file).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  # Try -march=native first (enables all CPU-specific tuning), fall back to
  # generic AVX2 if native is not supported (cross-compilation, old gcc)
  NATIVE_OK := $(shell echo 'int main(){return 0;}' | $(CC) -march=native -x c - -o /dev/null 2>/dev/null && echo yes)
  ifeq ($(NATIVE_OK),yes)
    SIMD_FLAGS ?= -march=native
  else
    SIMD_FLAGS ?= -mavx2 -mfma -mf16c
  endif
else ifeq ($(UNAME_M),i386)
  SIMD_FLAGS ?= -msse4.1
else ifneq (,$(filter $(UNAME_M),arm armv7l armv7-a))
  # ARMv7 has NEON; -mfpu=neon is needed on 32-bit
  SIMD_FLAGS ?= -march=armv7-a -mfpu=neon -mfloat-abi=softfp
else ifneq (,$(filter $(UNAME_M),aarch64 arm64))
  # AArch64 has NEON + FMA by default
  SIMD_FLAGS ?=
else
  SIMD_FLAGS ?=
endif

CFLAGS ?= -O3 $(SIMD_FLAGS) -Wall -ldl
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all train server qwen-server qwen7b-server float-subset demos verify verify-qwen7b-lal mini-filter clean test-memory test-sparse meta-train meta-train-mem

all: demos train

# === GPT-2 training (binary mode, no PyTorch) ===
train: prebuilt/gpt2_train

prebuilt/gpt2_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -o $@ models/gpt2.c runtime/lal_runtime.c -lm

# === GPT-2 training on GPU via CUDA backend (requires NVIDIA CUDA Toolkit / nvcc) ===
NVCC ?= nvcc
cuda-train: prebuilt/gpt2_cuda_train
prebuilt/gpt2_cuda_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_cuda.cu runtime/lal_cuda.h runtime/lal_runtime.h
	$(CC) -O3 -DLAL_CUDA -I. -c models/gpt2.c -o prebuilt/gpt2_cuda_train_gpt2.o
	$(CC) -O3 -DLAL_CUDA -I. -c runtime/lal_runtime.c -o prebuilt/gpt2_cuda_train_runtime.o
	$(NVCC) -O3 -DLAL_CUDA -I. -c runtime/lal_cuda.cu -o prebuilt/gpt2_cuda_train_cuda.o
	$(CC) prebuilt/gpt2_cuda_train_gpt2.o prebuilt/gpt2_cuda_train_runtime.o prebuilt/gpt2_cuda_train_cuda.o -o $@ -lcudart -lm
	@echo "[*] built gpt2_cuda_train (GPU training via CUDA backend)"
	@echo "[*] run: ./prebuilt/gpt2_cuda_train --cuda --data <text> --steps N"

# === OpenBLAS auto-detection for the float server ===
# gpt2_server.c auto-enables BLAS matmul (cblas_sgemv) when <cblas.h> is
# present at compile time, via __has_include. We auto-link -lopenblas here so
# `make server` uses it out of the box on systems with libopenblas-dev.
# On systems without it, BLAS_LIBS is empty and the server silently falls back
# to hand-written SIMD. Override: make server BLAS_LIBS=-lopenblas (or = to disable)
BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(BLAS_LIBS),)
  # Fallback: detect -lopenblas by probing whether <cblas.h> preprocesses.
	BLAS_LIBS := $(shell echo '#include <cblas.h>' | $(CC) -E -x c - >/dev/null 2>&1 && echo -lopenblas)
endif

# === GPT-2 web server (float mode, auto OpenBLAS, with browser frontend) ===
# Float mode + (OpenBLAS if available, else hand-written SIMD). ~5× faster
# than the original scalar server (96 ms/token vs. 490 ms/token baseline on
# x86_64). Works on x86_64 (AVX2), ARMv7 (NEON), and AArch64 (NEON+FMA).
server: prebuilt/gpt2_server

prebuilt/gpt2_server: tools/server/gpt2_server.c tools/server/frontend.html \
	runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	-o $@ tools/server/gpt2_server.c runtime/lal_runtime.c -lm -lpthread $(BLAS_LIBS)
	@if [ -n "$(BLAS_LIBS)" ]; then \
	        echo "[*] built with OpenBLAS acceleration ($(BLAS_LIBS))"; \
	else \
	        echo "[*] built with hand-written SIMD (install libopenblas-dev for ~2-3x speedup)"; \
	fi

# server-blas: legacy alias. OpenBLAS is now auto-detected by `make server`.
# Kept for backward compatibility with existing docs and scripts.
server-blas: server

# === Float subset extractor for --mixed-precision on memory-constrained devices ===
# Extracts only layers 0 and 11 (24 tensors, ~54 MB) from the full 498 MB
# gpt2_weights.bin, in the same GPW2 format, so the tablet can run
# --mixed-precision without downloading the full float file.
#   LAL_FLOAT_SUBSET=prebuilt/gpt2_float_subset.bin ./gpt2_server --mixed-precision
float-subset: scripts/extract_float_subset prebuilt/gpt2_float_subset.bin

scripts/extract_float_subset: scripts/extract_float_subset.c
	$(CC) -O2 -o $@ $<

prebuilt/gpt2_float_subset.bin: scripts/extract_float_subset prebuilt/gpt2_weights.bin
	./scripts/extract_float_subset prebuilt/gpt2_weights.bin $@

# === Compile LAL demos ===
demos: prebuilt/demos/demo

prebuilt/demos/demo: demos/basic/demo.lal compiler/lal.py
	@mkdir -p prebuilt/demos
	$(LALC) demos/basic/demo.lal classify prebuilt/demos/demo.c
	$(CC) $(CFLAGS) -o $@ prebuilt/demos/demo.c -lm

# === Verify ===
verify: verify-steer verify-skip verify-qwen7b-lal

verify-steer: build/verify_steer
	./build/verify_steer

build/verify_steer: tests/verify_steer.c compiler/lal.py
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_steer.c -lm

verify-skip: build/verify_skip build/verify_skip_cond build/verify_skip_uncond prebuilt/mini_skip.so prebuilt/mini_skip_cond.so
	./build/verify_skip

build/verify_skip: tests/verify_skip.c
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_skip.c

build/verify_skip_cond: tests/verify_skip_cond.c
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_skip_cond.c -ldl

build/verify_skip_uncond: tests/verify_skip_uncond.c
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_skip_uncond.c -ldl

# Contract test for the 7B flagship's LAL three-layer fusion bridge (no 7.5GB
# weights needed — exercises the same generic .so through the server's contracts).
verify-qwen7b-lal: build/verify_qwen7b_lal prebuilt/mini_antirepeat.so prebuilt/mini_skip.so
	./build/verify_qwen7b_lal

build/verify_qwen7b_lal: tests/verify_qwen7b_lal.c
	@mkdir -p build
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_qwen7b_lal.c -ldl

# === End-to-end: real generation through the mini server (.lal .so hot-loaded) ===
e2e: prebuilt/mini_server prebuilt/mini_steer.so prebuilt/mini_steer_neg.so prebuilt/mini_skip.so prebuilt/mini_skip_cond.so prebuilt/mini_antirepeat.so
	bash scripts/e2e_test.sh

# Build the LAL logic-layer sampling filter .so (level-2 fusion: constraints the
# top-k sampling pool via declarative ban_* rules — no retraining, no rebuild).
mini-filter: prebuilt/mini_antirepeat.so

prebuilt/mini_antirepeat.so: demos/mini_antirepeat.lal compiler/lal.py
	bash scripts/build_lal_mini_filter.sh

clean:
	rm -rf build/ prebuilt/demos/*.c prebuilt/gpt2_server prebuilt/qwen_server

# === Qwen2.5-0.5B inference server (Q8 quantization, GQA, RoPE, SwiGLU) ===
qwen-server: prebuilt/qwen_server

prebuilt/qwen_server: tools/server/qwen_server.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/server/qwen_server.c runtime/lal_runtime.c -lm -lpthread
	@echo "[*] built qwen_server (Qwen2.5-0.5B, Q8 default)"

# === Qwen2.5-7B inference server (Q8, GQA, GPQ8 pre-quantized) ===
qwen7b-server: prebuilt/qwen7b_server

prebuilt/qwen7b_server: tools/server/qwen7b_server.c runtime/lal_runtime.c runtime/lal_runtime.h runtime/lal_q8_kernel.h runtime/lal_sampling.h runtime/lal_dequant.h runtime/lal_tokenizer.h
	$(CC) $(CFLAGS) -fopenmp -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/server/qwen7b_server.c runtime/lal_runtime.c -lm -lpthread -lgomp
	@echo "[*] built qwen7b_server (Qwen2.5-7B, Q8, GPQ8, OpenMP)"

# === Mini local transformer (real model, no internet) for LAL level-1 live test ===
# Trains a tiny char-LM, exports C-loadable weights + a REAL steering direction.
mini-train: prebuilt/mini_model.bin

prebuilt/mini_model.bin: tools/mini_train.py
	$(PYTHON) tools/mini_train.py

prebuilt/mini_server: tools/mini_server.c
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/mini_server.c -lm -ldl
	@echo "[*] built mini_server (tiny real transformer, level-1 hook enabled)"

# Build the real-direction steering .so for mini_server (and qwen_server).
mini-steer: prebuilt/mini_steer.so

prebuilt/mini_steer.so: demos/mini_steer.lal compiler/lal.py
	bash scripts/build_lal_mini_steer.sh

# Build the logic-driven layer-skip .so (level-1 acceleration: early-exit / skip).
mini-skip: prebuilt/mini_skip.so

prebuilt/mini_skip.so: demos/mini_skip.lal compiler/lal.py
	bash scripts/build_lal_skip.sh

# Build the CONDITIONAL layer-skip .so (only skips when hidden-state confidence
# exceeds the .lal `when` threshold — preserves quality on hard tokens).
mini-skip-cond: prebuilt/mini_skip_cond.so

prebuilt/mini_skip_cond.so: demos/mini_skip_cond.lal compiler/lal.py
	bash scripts/build_lal_skip_cond.sh

# === Sparse Attention + Stateful Inference Test ===
test-sparse: build/test_sparse_attn
	./build/test_sparse_attn

build/test_sparse_attn: tests/test_sparse_attn.c runtime/lal_runtime.c runtime/lal_runtime.h
	@mkdir -p build
	$(CC) $(CFLAGS) -I. -o $@ tests/test_sparse_attn.c runtime/lal_runtime.c -lm
	@echo "[*] built test_sparse_attn (sparse attention + stateful inference)"

# === Semantic-Gated Curriculum Training (LAL-aware Adam + Whitebox Probing) ===
web-train: web_model_v3

web_model_v3: models/web_model_v3.c runtime/lal_runtime.c runtime/lal_runtime.h \
              runtime/lal_model_growth.h runtime/lal_semantic_gate.h \
              runtime/lal_whitebox_probe.h runtime/lal_data_loader.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
		-o $@ models/web_model_v3.c runtime/lal_runtime.c -lm
	@echo "[*] built web_model_v3 (LAL-aware Adam + whitebox probing + semantic regularization)"
	@echo "[*] usage: ./web_model_v3 --curriculum --batch-size 4"

# === 100M Programming Model (Curriculum Training) ===
coder-train: prebuilt/coder_train

prebuilt/coder_train: models/coder.c runtime/lal_runtime.c runtime/lal_runtime.h runtime/lal_data_loader.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
		-o $@ models/coder.c runtime/lal_runtime.c -lm
	@echo "[*] built coder_train (100M programming model, curriculum training)"
	@echo "[*] usage: ./prebuilt/coder_train --stage 0 --steps 5000 --ste --adam"

# === Meta-Learning: Dynamic Adapter Layers (Frozen Base + Online Learning) ===
meta-data:
	$(PYTHON) scripts/gen_meta_data.py --n_tasks 5000 --out data/meta_train.jsonl --split train
	$(PYTHON) scripts/gen_meta_data.py --n_tasks 1000 --out data/meta_test.jsonl --split test

meta-train: prebuilt/meta_learn
	./prebuilt/meta_learn --episodes 1000 --rank 16 --adapter-lr 0.001

meta-train-mem: prebuilt/meta_learn
	./prebuilt/meta_learn --episodes 1000 --rank 16 --adapter-lr 0.001 --use-memory --mem-write-freq 2

prebuilt/meta_learn: models/meta_learn.c runtime/lal_runtime.c runtime/lal_runtime.h runtime/lal_meta_learn.h runtime/lal_memory.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
		-o $@ models/meta_learn.c runtime/lal_runtime.c -lm
	@echo "[*] built meta_learn (dynamic adapter layers + memory notebook)"
	@echo "[*] usage: ./prebuilt/meta_learn --episodes 1000 --rank 16 --adapter-lr 0.001 --use-memory"

# === Memory Notebook System Test (Brain-Inspired Memory) ===
test-memory: tests/test_memory
	./tests/test_memory

tests/test_memory: tests/test_memory.c runtime/lal_memory.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_memory.c -lm
	@echo "[*] built test_memory (hippocampus-inspired memory notebook)"

# === Optimized Training (Multi-token loss + EMA + Cosine Restart) ===
opt-train: prebuilt/train_optimized
	./prebuilt/train_optimized 5000 0.001 --ste --adam --multitoken --accum 4 --ema --warmup 100

prebuilt/train_optimized: models/train_optimized.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
		-o $@ models/train_optimized.c runtime/lal_runtime.c -lm
	@echo "[*] built train_optimized (multi-token loss + EMA + cosine restart)"

# ============================================================================
# === LAL STE 训练流程 (从 lalmodel 移植: 中文 BPE 训练 + 白箱监督 + 关系探针) ===
# 独立于上面的 gpt2/qwen 目标, 不破坏原有全功能构建.
# ============================================================================

# STE 训练器 (白箱监督 + BPE + 认知数据)
ste_train: models/ste_train.c runtime/lal_runtime.c runtime/lal_runtime.h \
           runtime/lal_data_loader.h runtime/lal_model_growth.h \
           runtime/lal_semantic_logic.h runtime/lal_whitebox_probe.h \
           runtime/lal_inference_tracer.h
	$(CC) $(CFLAGS) -o $@ models/ste_train.c runtime/lal_runtime.c -lm
	@echo "[*] built ste_train (LAL STE + BPE + cognitive training)"

# 生成对话 BPE 训练数据 (对话+认知合并, 唯一正确数据)
# 注意: 不把 cognitive_foundation.jsonl 列为显式依赖, 避免触发其重建规则
# (gen_cognitive_data.py 含他人机器硬编码路径, 克隆后会报错). 该文件应已随仓库提供.
data/dialogue_bpe.bin: data/gen_dialogue_data.py tokenizer/chinese_bpe.model
	@if [ ! -f data/cognitive_foundation.jsonl ]; then \
	    echo "[!] 缺少 data/cognitive_foundation.jsonl (应随仓库提供), 无法生成对话数据"; exit 1; fi
	python3 data/gen_dialogue_data.py --out data/dialogue_train.jsonl
	python3 -c "srcs=['data/cognitive_foundation.jsonl','data/dialogue_train.jsonl']; \
open('data/train_dialogue_combined.jsonl','w',encoding='utf-8').writelines( \
line for s in srcs for line in open(s,encoding='utf-8') if line.strip())"
	python3 scripts/tokenize_data.py \
	    --input data/train_dialogue_combined.jsonl \
	    --output data/dialogue_bpe.bin \
	    --tokenizer tokenizer/chinese_bpe.model \
	    --max_len 64
	@echo "[*] dialogue_bpe.bin ready"

# ============================================================================
# 长句对话训练 — 唯一正确的路 (固化参数, 勿改)
#   make dialogue      完整流程: 生成数据→训练4000步→测双形态
#   make dialogue-train 仅训练(默认4000步, 可用 make dialogue-train STEPS=2000 覆盖)
#   make dialogue-test  仅测双形态(句子层+概念链层原生推理)
# 参数已固化(v5验证): steps=4000 lr=0.001 logic-lr=0.005 phase=0 vocab=32768
# 原生推理概念链默认开启, 全探针默认输出.
# ============================================================================
DIALOGUE_MODEL ?= model_dialogue.ste
DIALOGUE_STEPS ?= 4000

dialogue: dialogue-data dialogue-train dialogue-test
	@echo "[*] 对话训练全流程完成 -> $(DIALOGUE_MODEL)"

dialogue-data: data/dialogue_bpe.bin

dialogue-train: ste_train dialogue-data
	./ste_train --steps $(DIALOGUE_STEPS) --batch-size 4 --lr 0.001 --logic-lr 0.005 \
	    --phase 0 --vocab 32768 --data data/dialogue_bpe.bin \
	    --no-generate --save $(DIALOGUE_MODEL)
	@echo "[*] $(DIALOGUE_MODEL) trained ($(DIALOGUE_STEPS) steps)"

dialogue-test: ste_train
	@echo "########## 句子层 (transformer 自回归) ##########"
	@for q in "你好" "鸟为什么天上飞？" "什么是火？"; do \
	    echo "--- Q: $$q ---"; \
	    ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume $(DIALOGUE_MODEL) \
	        --prompt "$$q" --max-gen 30 --temp 0.4 2>&1 | sed -n '/=== Generation/,/^$$/p' | head -5; \
	done
	@echo "########## 概念链层 (LAL 原生推理, 默认开启) ##########"
	@for q in "鸟为什么天上飞？" "什么是火？" "天为什么是蓝色的？"; do \
	    echo "--- Q: $$q ---"; \
	    ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume $(DIALOGUE_MODEL) \
	        --prompt "$$q" --native-topk 3 --native-depth 2 --no-generate 2>&1 \
	        | sed -n '/=== LAL Native/,/嵌入空间检索/p' | head -16; \
	done

# 旧 ste-train 目标保留兼容, 但指向正确数据(对话合并)以防走岔路
ste-train: dialogue-train
ste-generate: dialogue-test

.PHONY: ste_train dialogue dialogue-data dialogue-train dialogue-test ste-train ste-generate
