/* lal_runtime.h — LAL Universal Runtime
 *
 * A model-agnostic C library for binary neural network inference + training.
 * Any transformer model (GPT-2, BERT, LLaMA, Qwen, ...) can build on top.
 *
 * Three API levels:
 *   1. Operator level: bin_forward, layer_norm, gelu, softmax (building blocks)
 *   2. Layer level: transformer_layer_forward/backward (one call = one layer)
 *   3. Model level: model_load, model_forward, model_train (full model)
 *
 * Architecture:
 *   lal_runtime.h/c     — this file (model-agnostic, all 3 levels)
 *   models/gpt2.c       — GPT-2 (just config + weight keys, ~30 lines)
 *   models/qwen.c       — Qwen (just config + weight keys, ~30 lines)
 */
#ifndef LAL_RUNTIME_H
#define LAL_RUNTIME_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Level 0: Types & Enums
 * ======================================================================== */

typedef enum {
    NORM_LAYER = 0,   /* GPT-2 style: LayerNorm */
    NORM_RMS    = 1,  /* LLaMA/Qwen style: RMSNorm */
} NormType;

typedef enum {
    ATTN_LEARNED = 0, /* GPT-2: learned positional embeddings */
    ATTN_ROPE    = 1, /* LLaMA/Qwen: Rotary Position Embedding */
} AttnType;

typedef enum {
    ACT_GELU   = 0,   /* GPT-2: GELU */
    ACT_SWIGLU = 1,   /* LLaMA/Qwen: SwiGLU (gate * SiLU(up)) */
    ACT_SILU   = 2,   /* SiLU/Swish */
} ActType;

/* ========================================================================
 * Level 0: Model Configuration
 * ======================================================================== */
typedef struct {
    int n_layer;
    int n_embd;
    int n_head;
    int n_ctx;
    int vocab_size;
    int mlp_dim;       /* MLP hidden dim (GPT-2: 4*embd, LLaMA: ~2.7*embd) */
    NormType norm_type;
    AttnType attn_type;
    ActType  act_type;
    float residual_scale;  /* 0.5 for stability, 1.0 for standard */
    int   qkv_merged;      /* 1 = GPT-2 (c_attn = merged QKV), 0 = LLaMA (separate Q/K/V/O) */
    int   sliding_window;  /* SWA window size (0 = full attention, >0 = sparse) */
    int   n_sinks;         /* Attention sink tokens (StreamingLLM, default 4) */
} ModelConfig;

/* ========================================================================
 * Level 1: Binary Weight Layer (operator level)
 * ======================================================================== */
typedef struct {
    uint64_t *wbits;
    uint64_t *wbits_T;
    uint64_t *zbits;    /* Ternary zero mask (same shape as wbits): 1=weight is 0, 0=active.
                         * NULL when ternary mode is off (pure BWN, all weights ±1).
                         * Combined with wbits (sign): ternary value = sign * (1 - zbit) ∈ {-1,0,+1}. */
    float    *alpha;
    float    *bias;
    float    *w_float;   /* STE: full-precision weights for gradient update */
    float    *w_core;    /* Logic-guided: CORE weights kept as float (not binarized) */
    float    *m_adam;    /* Adam first moment, same shape as w_float [out*in] */
    float    *v_adam;    /* Adam second moment */
    float    *grad_accum;     /* Batch: accumulated weight gradients [out*in] */
    float    *bias_grad_accum;/* Batch: accumulated bias gradients [out] */
    uint8_t  *logic_mask; /* Per-output: 0=CORE(float), 1=BINARY(sign+alpha), 2=PRUNE(zero) */
    float     ternary_delta; /* Per-layer Δ threshold for ternary binarization. 0=BWN (no zeroing).
                              * >0 means weights with |W|<=Δ are zeroed → ternary {-1,0,+1}. */
    int       n_core;     /* Count of CORE outputs */
    int       n_prune;    /* Count of PRUNE outputs */
    int       in_dim, out_dim, n_words, n_words_T;
} BinLayer;

void bin_layer_init(BinLayer *bl, const float *W, const float *bias,
                    int in_dim, int out_dim);

/* Logic-guided binarization: initialize with a per-output logic_mask.
 * mask[j]: 0=CORE (keep float in w_core), 1=BINARY (sign+alpha), 2=PRUNE (zero).
 * This implements PHONE's "logic extraction at binarization time":
 * core logic weights stay float, noise is pruned, rest is binarized. */
void bin_layer_init_logic(BinLayer *bl, const float *W, const float *bias,
                          int in_dim, int out_dim, const uint8_t *logic_mask);
void bin_layer_free(BinLayer *bl);

/* Ternary Weight Network (TWN) mode.
 * When g_use_ternary is set, BINARY rows in logic-guided layers use ternary
 * weights {-1, 0, +1} instead of binary {-1, +1}. Weights with |W| <= Δ are
 * zeroed (Δ = g_ternary_delta_factor * mean(|W_row|), default 0.7 per TWN paper).
 * This triples representational capacity at ~1.58 bits/weight while keeping
 * bit-parallel friendliness (sign stored in wbits, zero-mask in zbits).
 *
 * STE backward flows through ALL weights including zeroed ones — a zeroed
 * weight can "wake up" if gradient pushes |W| past Δ, and an active weight
 * can be zeroed if |W| drops below Δ. This is the key TWN training dynamic. */
extern int   g_use_ternary;
extern float g_ternary_delta_factor;  /* Δ = factor * mean(|W|), default 0.7 */

/* Recompute zbits from |w_float| vs Δ. Called after STE update when ternary
 * mode is on. Updates alpha (mean|W| over active weights) too. */
void bin_layer_repack_ternary(BinLayer *bl);

/* bin_forward (BWN, default): x stays float, only W is binarized.
 * Matches Python STE training (tools/train_binary_gpt2.py) so train/inference
 * distributions are aligned. Applies XNOR-Net K-norm scaling
 *   K = ||x||_1 / in_dim
 * to preserve input magnitude information.
 *
 * y[j] = (sum_i sign(W[j,i]) * x[i]) * alpha[j] * K + bias[j] */
void bin_forward(float *y, const float *x, const BinLayer *bl);

/* bin_forward_bnn (legacy fast path): binarizes BOTH x and W via XNOR+popcount.
 * ~64x faster than BWN on long vectors but loses input magnitude → quality
 * collapse after a few layers. Kept as opt-in for max-speed-low-quality mode.
 * Use ONLY when you can prove BNN quality is acceptable for your task. */
void bin_forward_bnn(float *y, const float *x, const BinLayer *bl);

/* bin_forward_float: same as BWN but without K-norm (legacy interface).
 * Kept for backward compat with callers that don't want K scaling. */
void bin_forward_float(float *y, const float *x, const BinLayer *bl);

void bin_backward(float *grad_x, const float *grad_y, const float *x,
                  BinLayer *bl, float lr);

/* STE (Straight-Through Estimator) backward pass.
 * Updates w_float using gradient, then re-binarizes wbits from sign(w_float).
 * This is the key to recovering accuracy lost by binarization:
 *   - Forward uses sign(w) (binary)
 *   - Backward treats sign() as identity, so gradient flows to w_float
 *   - After update, wbits = sign(w_float) is recomputed
 * Call this instead of bin_backward for STE fine-tuning. */
void bin_backward_ste(float *grad_x, const float *grad_y, const float *x,
                      BinLayer *bl, float lr);

/* Global flag: set to 1 to use STE backward in trans_layer_backward.
 * Models can set this before calling model_backward(). */
extern int g_use_ste;
extern int g_use_logic_binarization;  /* norm-based auto logic mask in model_load */

/* Semantic logic mask ratios (set by training script per curriculum phase) */
extern float g_logic_core_ratio;   /* 0 = use default 20% */
extern float g_logic_prune_ratio;  /* 0 = use default 10% */

/* Global flag: set to 1 to dispatch bin_forward/bin_backward_ste to the CUDA
 * GPU backend (runtime/lal_cuda.cu). The binary must be built with -DLAL_CUDA
 * (see `make cuda-train`) and an NVIDIA GPU + CUDA driver must be present at
 * runtime. Only the standard BWN path (no logic mask) is GPU-accelerated;
 * logic-guided / ternary rows automatically fall back to the CPU path. */
extern int g_use_cuda;

/* Global flag: set to 1 to use Adam optimizer inside bin_backward_ste.
 * Adam uses m_adam/v_adam per-param moments (allocated in bin_layer_init).
 * Helps stabilize STE on bit-space (gradients are extremely noisy with SGD).
 * Sets g_opt_step automatically per model_backward call for bias correction. */
extern int g_use_adam;
extern int g_opt_step;
extern float g_adam_beta1, g_adam_beta2, g_adam_eps;

/* Cosine LR schedule with linear warmup.
 *   step < warmup          : lr = base_lr * (step+1) / warmup   (linear ramp)
 *   warmup <= step < total : lr = base_lr * 0.5 * (1 + cos(pi * progress))
 *   step >= total          : lr = base_lr * 0.01 (floor — don't go to zero)
 * Prevents early mode-collapse from large initial gradient + stabilizes late
 * training with decay. Pass warmup=0 for pure cosine, total=0 for constant lr. */
float lr_schedule(int step, int warmup_steps, int total_steps, float base_lr);

/* Pure float forward (no binarization): y[j] = sum_i w_float[j*in+i] * x[i] + bias[j].
 * Used for the teacher model in distillation. The teacher's w_float holds the
 * original GPT-2 weights (loaded once, never updated), so this is a true
 * full-precision forward pass. */
void bin_forward_pure_float(float *y, const float *x, const BinLayer *bl);

/* Global flag: set to 1 to use the legacy BNN fast path (bin_forward_bnn) in
 * trans_layer_forward instead of the BWN default. Off by default — BNN causes
 * train/inference mismatch and quality collapse. Only enable if you have a
 * very tight latency budget AND can verify quality is still acceptable. */
extern int g_use_bnn_fast_path;
extern float g_core_lr_multiplier;
extern int g_use_lal_adam;
extern float g_prune_decay;
extern float g_prune_freeze_thresh;
void bin_layer_repack(BinLayer *bl);

/* Global flag: set to 1 to use real causal multi-head self-attention with
 * KV cache in trans_layer_forward (replaces the degenerate V-copy).
 * Off by default for backward compatibility. When on, Model.k_cache and
 * Model.v_cache must be allocated — call model_kv_cache_alloc() or set the
 * flag before model_load(). */
extern int g_use_real_attention;

/* Global flag: set to 1 to use pure float (full-precision) forward and backward
 * instead of binary weight network (BWN). When on:
 *   - trans_layer_forward dispatches to bin_forward_pure_float (uses w_float)
 *   - bin_backward_ste uses w_float for grad_x, skips W_CLIP and repack
 * Off by default (BWN mode). */
extern int g_use_pure_float;

/* Global flag: set to 1 to accumulate gradients instead of applying them
 * immediately in bin_backward_ste. Used by batch training to collect
 * gradients across multiple sequences before doing one weight update. */
extern int g_accumulate_gradients;

/* ========================================================================
 * Level 1: Standard NN Operations (operator level)
 * ======================================================================== */
void layer_norm(float *out, const float *x, const float *w, const float *b, int n);
void rms_norm(float *out, const float *x, const float *w, int n);
void layer_norm_backward(float *grad_x, const float *grad_y, const float *x,
                         const float *w, float mean, float std_inv, int n,
                         float *grad_w, float *grad_b);
void rms_norm_backward(float *grad_x, const float *grad_y, const float *x,
                       const float *w, int n, float *grad_w);
float gelu(float x);
float gelu_grad(float x);
float silu(float x);
float silu_grad(float x);
void softmax(float *x, int n);

/* Universal normalization dispatch (calls LayerNorm or RMSNorm based on type) */
void norm_forward(float *out, const float *x, const float *w, const float *b,
                  NormType type, int n);
void norm_backward(float *grad_x, const float *grad_y, const float *x,
                   const float *w, const float *cached, NormType type, int n,
                   float *grad_w, float *grad_b);

/* Universal activation dispatch */
float act_forward(float x, ActType type);
float act_grad(float x, ActType type);

/* Cross-entropy (sampled softmax for efficient training) */
float cross_entropy_sampled(const float *hidden, const float *wte,
                            int target, int vocab_size, int n_embd,
                            int n_samples, unsigned int *seed);
void cross_entropy_grad(float *grad_hidden, const float *hidden, const float *wte,
                        int target, int vocab_size, int n_embd,
                        int n_samples, unsigned int *seed);

/* Tensor I/O (GPW2 format) */
typedef struct { char key[128]; int ndim; int shape[4]; float *data; } Tensor;
Tensor *tensor_load_all(const char *path, int *n_tensors);
float *tensor_get(Tensor *tensors, int n, const char *key);
void tensor_free_all(Tensor *tensors, int n);

/* Variable-size training from scratch: write a GPW2 weight file with
 * Gaussian-random init for ANY ModelConfig (GPT-2 or LLaMA key layout,
 * selected by cfg.qkv_merged / cfg.act_type). Lets you train an arbitrary
 * parameter-count model without a pretrained checkpoint. */
void gen_random_gpw2(const char *path, ModelConfig cfg);

/* Utility */
void clip_array(float *x, int n, float clip_val);
/* Cross-entropy loss + gradient (full softmax over entire vocab) */
float cross_entropy_full(const float *hidden, const float *wte,
                         int target, int vocab_size, int n_embd,
                         float *logits_scratch);
void cross_entropy_full_grad(float *grad_hidden, const float *hidden, const float *wte,
                             int target, int vocab_size, int n_embd,
                             float *logits_scratch);
void compute_mean_std(const float *x, int n, float *mean, float *std_inv);

/* RoPE (Rotary Position Embedding) — for LLaMA/Qwen */
void apply_rope(float *q, float *k, int seq_len, int n_head, int head_dim, int n_embd);

/* ========================================================================
 * Level 2: Transformer Layer (building block — one call = one layer)
 * ======================================================================== */

/* A transformer layer's binary weights */
typedef struct {
    BinLayer attn_q;      /* Q projection (or merged QKV if qkv_merged) */
    BinLayer attn_k;      /* K projection (unused if qkv_merged) */
    BinLayer attn_v;      /* V projection (unused if qkv_merged) */
    BinLayer attn_o;      /* output projection */
    BinLayer mlp_gate;    /* MLP gate (SwiGLU) or c_fc (GELU) */
    BinLayer mlp_up;      /* MLP up (SwiGLU only, unused for GELU) */
    BinLayer mlp_down;    /* MLP down projection */
    float *norm1_w, *norm1_b;  /* first norm weight/bias */
    float *norm2_w, *norm2_b;  /* second norm weight/bias */
    float *grad_norm1_w, *grad_norm1_b;  /* norm1 weight/bias gradients */
    float *grad_norm2_w, *grad_norm2_b;  /* norm2 weight/bias gradients */
    /* KV cache pointers (set by model_load when g_use_real_attention is on).
     * Each points to Model-owned [n_ctx * n_embd] float array.
     * NULL in legacy V-copy mode. */
    float *_kv_k;
    float *_kv_v;
} TransLayer;

/* Per-layer activation cache (for backward) */
typedef struct {
    float *x_pre_norm1;   /* x before first norm */
    float *norm1_out;     /* after first norm */
    float norm1_cache[4]; /* cached mean, std_inv (LN) or just w (RMS) */
    float *q, *k, *v;     /* attention projections */
    float *attn_out;      /* after attention */
    float *proj_out;      /* after output projection */
    float *x_pre_norm2;   /* x before second norm */
    float *norm2_out;     /* after second norm */
    float norm2_cache[4];
    float *mlp_hidden;    /* MLP hidden state (after activation) */
    float *mlp_out;       /* MLP output */
    /* BUG #45 FIX: cache gate/up for SwiGLU backward.
     * Forward: hidden = silu(gate) * up. Backward needs gate (not hidden)
     * to compute silu_grad(gate) * up. Without caching, backward used
     * silu_grad(hidden) which is mathematically wrong.
     * For GELU mode, these are NULL (unused). */
    float *swiglu_gate;   /* gate = W_gate · norm2 (pre-SiLU) */
    float *swiglu_up;     /* up = W_up · norm2 */
    int   seq_pos;        /* position of the cached forward pass (for attention bwd) */
} TransAct;

/* Initialize a transformer layer from tensors (model-agnostic) */
void trans_layer_init(TransLayer *tl, Tensor *tensors, int n_tensors,
                      ModelConfig *cfg, int layer_idx,
                      const char *qkv_key, const char *q_key, const char *k_key,
                      const char *v_key, const char *o_key,
                      const char *gate_key, const char *up_key, const char *down_key,
                      const char *norm1_w_key, const char *norm1_b_key,
                      const char *norm2_w_key, const char *norm2_b_key);

/* Free a transformer layer */
void trans_layer_free(TransLayer *tl, ModelConfig *cfg);

/* Forward: one transformer layer (x modified in-place, activations cached) */
void trans_layer_forward(float *x, TransLayer *tl, TransAct *act,
                         ModelConfig *cfg, int seq_pos);

/* Pure-float forward: same as trans_layer_forward but uses bin_forward_pure_float
 * for every matmul (no sign binarization anywhere). Used by the teacher model
 * in distillation. Activations cache is the same shape as the student's. */
void trans_layer_forward_pure_float(float *x, TransLayer *tl, TransAct *act,
                                    ModelConfig *cfg, int seq_pos);

/* Backward: one transformer layer (updates weights, computes grad_x) */
void trans_layer_backward(float *grad_x, TransLayer *tl, TransAct *act,
                          ModelConfig *cfg, float lr);

/* Allocate/free activation cache */
TransAct *trans_act_alloc(ModelConfig *cfg);
void trans_act_free(TransAct *acts, int n_layer);

/* ========================================================================
 * Level 2: Causal Multi-Head Self-Attention (KV cache)
 * ========================================================================
 * Replaces the degenerate `memcpy(attn_out, v, n)` in trans_layer_forward.
 * Mirrors tools/server/gpt2_server.c:real_attention semantics:
 *   - Stores K, V at current position into per-layer cache
 *   - Multi-head QK^T dot product with 1/sqrt(head_dim) scaling
 *   - Causal mask (only attend to positions 0..seq_pos)
 *   - Softmax with max subtraction (numerical stability)
 *   - Weighted sum of V
 *
 * KV cache is owned by Model (k_cache/v_cache arrays, [n_layer][n_ctx*n_embd]).
 * Allocated in model_load when g_use_real_attention is on, or via
 * model_kv_cache_alloc(). Freed in model_free.
 *
 * Backward: NOT YET IMPLEMENTED. trans_layer_backward still does pass-through
 *   (memcpy g_qkv+2n, g_attn). This means gradient flow through attention
 *   is incorrect — Q, K gradients are zero. Sufficient for inference but
 *   training with attention enabled will not learn Q/K properly. TODO. */
void attention_forward(float *attn_out, const float *qkv,
                       int n_embd, int n_head,
                       int seq_pos,
                       float *k_cache_layer, float *v_cache_layer);

/* Attention backward: given grad w.r.t. attn_out at the current position,
 * compute grad w.r.t. Q, K, V at the current position. Cached K/V at
 * positions 0..seq_pos-1 are treated as constants (no gradient flows to
 * them — only the current token's QKV projection learns, consistent with
 * the single-position activation cache in model_forward/backward).
 *
 * qkv:        [3*n_embd] current Q|K|V (contiguous, both merged & separate)
 * grad_qkv:   [3*n_embd] output grad Q|K|V (caller-allocated, zeroed here)
 * k/v_cache:  [n_ctx*n_embd] — read for scores/weights and current-position K,V */
void attention_backward(float *grad_qkv, const float *grad_attn_out,
                        const float *qkv, int n_embd, int n_head, int seq_pos,
                        const float *k_cache_layer, const float *v_cache_layer);

/* Forward decl — model_kv_cache_alloc/free defined after Model struct below */
struct Model;
void model_kv_cache_alloc(struct Model *m);
void model_kv_cache_free(struct Model *m);

/* ========================================================================
 * Level 3: Full Model (highest level — just config + weight keys)
 * ======================================================================== */

typedef struct Model {
    ModelConfig cfg;
    Tensor *tensors;
    int n_tensors;
    TransLayer *layers;
    TransAct *acts;
    float *wte, *wpe;       /* token + position embeddings */
    float *ln_f_w, *ln_f_b; /* final norm */
    /* Batch training: gradient accumulation + Adam state for embeddings/norms */
    float *grad_wte_accum;  /* [vocab * n_embd] */
    float *m_wte, *v_wte;   /* Adam state for wte */
    float *grad_wpe_accum;  /* [n_ctx * n_embd] — position embedding grads */
    float *m_wpe, *v_wpe;   /* Adam state for wpe */
    float *grad_ln_f_w_accum, *grad_ln_f_b_accum; /* final norm grads */
    float *m_ln_f_w, *v_ln_f_w, *m_ln_f_b, *v_ln_f_b; /* Adam state */
    float *final_ln;         /* cached final norm output */
    float *x_before_final;   /* cached for backward */
    float final_mean, final_std_inv;
    /* KV cache for real causal attention — [n_layer] pointers, each
     * [n_ctx * n_embd] floats. NULL when g_use_real_attention is 0. */
    float **k_cache;
    float **v_cache;
} Model;
/* Load a model from a GPW2 weight file.
 * key_prefix: "h." for GPT-2, "model.layers." for LLaMA/Qwen
 * This is the ONLY function a new model needs to customize. */
void model_load(Model *m, const char *weight_path, ModelConfig cfg,
                const char *layer_prefix,  /* e.g. "h.%d." or "model.layers.%d." */
                int qkv_merged);            /* 1=GPT-2, 0=LLaMA/Qwen */

/* Forward pass (returns loss) */
float model_forward(Model *m, const int *tokens, int n_tokens);

/* Backward pass (updates all weights) */
void model_backward(Model *m, const int *tokens, int n_tokens, float lr);

/* Compute full vocab logits at the target position using pure float forward
 * (no binarization). Used as the teacher signal in distillation.
 * logits_out must be vocab_size floats, caller-allocated. */
void model_forward_float_logits(Model *m, const int *tokens, int n_tokens,
                                float *logits_out);

/* Backward with distillation: combines hard CE (next-token target) with soft
 * KL(teacher/T || student/T) gradient. The hard CE path matches model_backward.
 * The KL path adds: T * (softmax(student_logits/T) - softmax(teacher_logits/T))
 *   to the gradient at the student's pre-softmax hidden.
 * distill_alpha: weight on the KL term (0 = pure CE, 1 = pure KL).
 * teacher_logits: full vocab logits from model_forward_float_logits (same tokens). */
void model_backward_distill(Model *m, const int *tokens, int n_tokens, float lr,
                            const float *teacher_logits,
                            float distill_alpha, float distill_T);

/* Free model */
void model_free(Model *m);


/* ========================================================================
 * Level 3.5: Sparse Attention + Stateful Continuous Inference
 * ========================================================================
 * Sliding Window Attention (SWA) with Attention Sinks (StreamingLLM-style).
 * Enables efficient long-context inference with O(window) attention per token
 * instead of O(seq_len). Combined with a circular KV cache buffer, the model
 * can process arbitrarily long sequences without memcpy shifts.
 *
 * Key concepts:
 *   - Sliding window: each token attends to last W tokens (configurable)
 *   - Attention sinks: first S tokens always stay in attention window
 *   - Circular buffer: KV cache wraps around n_ctx, no shifting needed
 *   - Stateful inference: token-by-token generation with persistent KV state
 */

/* Stateful inference context — persists across model_stateful_forward calls */
typedef struct {
    int    active;
    int    kv_pos;       /* circular buffer write position (wraps mod n_ctx) */
    int    total_pos;    /* absolute sequence position (monotonic) */
    float *x;            /* working buffer [n_embd] */
    float *logits;       /* output logits [vocab_size] */
} StatefulContext;

extern StatefulContext g_sctx;

/* Sparse sliding-window attention forward.
 * Replaces attention_forward() when sliding_window > 0.
 * Uses circular buffer: KV stored at (seq_pos % n_ctx), no shifting. */
void attention_forward_sliding(float *attn_out, const float *qkv,
                               int n_embd, int n_head,
                               int seq_pos,
                               float *k_cache_layer, float *v_cache_layer,
                               int n_ctx, int window_size, int n_sinks);

/* Sparse sliding-window attention backward (gradient w.r.t. current Q/K/V). */
void attention_backward_sliding(float *grad_qkv, const float *grad_attn_out,
                                const float *qkv, int n_embd, int n_head,
                                int seq_pos,
                                const float *k_cache_layer, const float *v_cache_layer,
                                int n_ctx, int window_size, int n_sinks);

/* Transformer layer forward with sliding window attention. */
void trans_layer_forward_sliding(float *x, TransLayer *tl, TransAct *act,
                                  ModelConfig *cfg, int cache_pos, int abs_pos,
                                  int window, int n_sinks);

/* Stateful inference: begin a new generation session. */
void model_stateful_begin(Model *m);

/* Stateful inference: feed one token, get logits back (uses sliding window). */
const float *model_stateful_forward_sliding(Model *m, int token);

/* Stateful inference: reset KV cache (keep model weights). */
void model_stateful_reset(Model *m);

/* Configure sliding window at runtime. */
void model_set_sliding_window(Model *m, int window, int n_sinks);


/* ========================================================================
 * Level 3: Batch Training (multiple sequences per weight update)
 * ========================================================================
 * True mini-batch training with gradient accumulation:
 *   1. model_batch_begin()  — zero all gradient accumulation buffers
 *   2. For each sample i in batch:
 *        model_batch_forward()  — forward pass (returns loss)
 *        model_batch_backward() — accumulate gradients (no weight update)
 *   3. model_batch_apply()  — apply averaged gradients with Adam/SGD
 *
 * This gives the same gradient as processing B samples in parallel
 * (averaged loss), which reduces gradient variance and stabilizes training
 * compared to single-sample updates. */

/* Allocate/free gradient accumulation buffers for all layers in a model.
 * Must be called after model_load(). Safe to call multiple times. */
void model_batch_alloc(Model *m);

/* Zero all gradient accumulation buffers. Call at the start of each batch. */
void model_batch_begin(Model *m);

/* Forward pass for one sample in the batch (same as model_forward).
 * Returns cross-entropy loss at the last position. */
float model_batch_forward(Model *m, const int *tokens, int n_tokens);

/* Backward pass for one sample: accumulates gradients without updating
 * weights. Sets g_accumulate_gradients=1 internally. */
void model_batch_backward(Model *m, const int *tokens, int n_tokens);

/* Apply accumulated gradients with Adam/SGD, divided by batch_size.
 * Increments g_opt_step once (one optimizer step per batch). */
void model_batch_apply(Model *m, float lr, int batch_size);

#endif /* LAL_RUNTIME_H */
