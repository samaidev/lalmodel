# 仓库同步报告：lalmodel → lal

## 背景
工作区初始为空（仅 `.codebuddy`）。两个仓库 `samaidev/lal` 与 `samaidev/lalmodel`
经 git clone 拉取到 `/workspace`。

## 角色判定
- **`lal`**：全功能主仓（287 文件，含 CUDA/demos/tests/docs/prebuilt/全量 runtime 头文件）。
- **`lalmodel`**：训练工作副本（93 文件，更活跃，含训练脚本、v2 中文 tokenizer、认知数据、各版本训练日志）。

git 历史证实同步方向为 **lalmodel → lal**（lal 的提交信息为 "sync ... from lalmodel"）。

## 差异分析（共有文件 checksum 比对）
仅 5 个共有文件内容不同：
| 文件 | 差异 | 处理 |
|------|------|------|
| `models/ste_train.c` | lalmodel 增加 `--tokenizer`、v2 tokenizer id（火=1164）、v14 关系探针 | **回灌到 lal**（更新版） |
| `scripts/merge_data.py` | lalmodel 修正相对路径（`lal/data/...`） | **回灌到 lal** |
| `Makefile` | lal 全功能版；lalmodel 精简训练版 | **保留 lal 全功能**，并追加 lalmodel 的 STE 训练目标（`ste_train`/`ste-data`/`ste-train`/`ste-generate`） |
| `README.md` | 语义不同（全引擎 vs 中文训练） | 各自保留，不互覆盖 |
| `.gitignore` | 忽略规则不同 | 保留 lal 版 |

lalmodel 独有的训练资产（`cognitive_foundation.jsonl`、`tokenizer/chinese_bpe.*`、`scripts/tokenize_data.py` 等）
中，仅 `data/cognitive_foundation.jsonl` 在 lal 缺失，已补入 lal。

## 已执行动作
1. 备份被覆盖文件至 `/root/.codebuddy/artifact/.../lal_backup/`。
2. `lalmodel/models/ste_train.c` → `lal/models/ste_train.c` 且 `lal/ste_train.c`（两者原相同，一并覆盖），三方 md5 一致。
3. `lalmodel/scripts/merge_data.py` → `lal/scripts/merge_data.py`。
4. `lalmodel/data/cognitive_foundation.jsonl` → `lal/data/cognitive_foundation.jsonl`。
5. `lal/Makefile` 追加 STE 训练目标（不破坏原有 gpt2/qwen 目标）。
6. 验证：`cd lal && make ste_train` 编译通过（产物 358KB）。

## 新增（长句对话训练相关，本阶段工作）
- `data/gen_dialogue_data.py`：生成长句对话训练数据（问候/因果/定义/方式/常识/闲聊）。
- `data/dialogue_train.jsonl` + `data/train_dialogue_combined.jsonl` + `data/dialogue_bpe.bin`：对话数据管线。
- `models/ste_train.c` 新增 `--native-chain` / `--native-topk` / `--native-depth`：LAL 原生推理概念链（wte 余弦，不走 transformer 前向）。
- 训练产物：`model_dialogue.ste`（v1, 300 步）、`model_dialogue_v2.ste`（v2, 1000 步, logic-lr=0.005）。
