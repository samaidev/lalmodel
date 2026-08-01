# lalmodel Makefile — 编译训练和推理工具

CC ?= gcc
CFLAGS ?= -O3 -march=native -ldl -Wno-unused-function -Wno-unused-variable -I.

all: ste_train

ste_train: models/ste_train.c runtime/lal_runtime.c runtime/lal_runtime.h \
           runtime/lal_data_loader.h runtime/lal_model_growth.h \
           runtime/lal_semantic_logic.h runtime/lal_whitebox_probe.h \
           runtime/lal_inference_tracer.h
	$(CC) $(CFLAGS) -o $@ models/ste_train.c runtime/lal_runtime.c -lm
	@echo "[*] built ste_train (LAL STE + BPE + cognitive training)"
	@echo "[*] usage: ./ste_train --steps 500 --vocab 32768 --data data/cognitive_bpe.bin"

# 生成 BPE 训练数据
data/cognitive_bpe.bin: data/cognitive_foundation.jsonl tokenizer/chinese_bpe.model
	python3 scripts/tokenize_data.py \
	    --input data/cognitive_foundation.jsonl \
	    --output data/cognitive_bpe.bin \
	    --tokenizer tokenizer/chinese_bpe.model \
	    --max_len 64

# 重新生成认知数据
data/cognitive_foundation.jsonl: scripts/gen_cognitive_data.py
	python3 scripts/gen_cognitive_data.py

train: ste_train data/cognitive_bpe.bin
	./ste_train --steps 500 --batch-size 4 --lr 0.001 --logic-lr 0.002 \
	    --phase 0 --vocab 32768 --data data/cognitive_bpe.bin \
	    --no-generate --save model.ste

generate: ste_train
	./ste_train --diagnose-only --phase 0 --vocab 32768 \
	    --resume model.ste --prompt-ids "32646" --max-gen 20 --temp 0.01

clean:
	rm -f ste_train *.o

.PHONY: all train generate clean
