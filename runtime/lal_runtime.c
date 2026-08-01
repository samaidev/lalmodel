/* lal_runtime.c — LAL Universal Runtime implementation
 *
 * Three API levels:
 *   Level 1: operators (bin_forward, norm, gelu, etc.)
 *   Level 2: transformer layer (trans_layer_forward/backward)
 *   Level 3: full model (model_load/forward/backward)
 *
 * Models only need Level 3 — just config + weight key patterns.
 */
#include "lal_runtime.h"
#ifdef LAL_CUDA
#include "lal_cuda.h"   /* CUDA GPU training backend (runtime/lal_cuda.cu) */
#endif

/* Forward declarations for full-vocab softmax (defined later in this file,
 * but model_forward/model_backward call them — declared here to avoid
 * implicit-declaration errors since the definitions sit after the callers). */
float cross_entropy_full(const float *hidden, const float *wte,
                         int target, int vocab_size, int n_embd,
                         float *logits_scratch);
void cross_entropy_full_grad(float *grad_hidden, const float *hidden, const float *wte,
                             int target, int vocab_size, int n_embd,
                             float *logits_scratch);

/* ========================================================================
 * Level 1 additions: RMSNorm, SiLU, dispatch functions, RoPE
 * ======================================================================== */

void rms_norm(float *out, const float *x, const float *w, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms * w[i];
}

void rms_norm_backward(float *grad_x, const float *grad_y, const float *x,
                       const float *w, int n, float *grad_w) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        grad_x[i] = grad_y[i] * w[i] * ms;
        if (grad_w) grad_w[i] += grad_y[i] * x[i] * ms;
    }
}

float silu(float x) { return x / (1.0f + expf(-x)); }
float silu_grad(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}

void norm_forward(float *out, const float *x, const float *w, const float *b,
                  NormType type, int n) {
    if (type == NORM_RMS) rms_norm(out, x, w, n);
    else layer_norm(out, x, w, b, n);
}

void norm_backward(float *grad_x, const float *grad_y, const float *x,
                   const float *w, const float *cached, NormType type, int n,
                   float *grad_w, float *grad_b) {
    if (type == NORM_RMS) rms_norm_backward(grad_x, grad_y, x, w, n, grad_w);
    else layer_norm_backward(grad_x, grad_y, x, w, cached[0], cached[1], n, grad_w, grad_b);
}

float act_forward(float x, ActType type) {
    switch (type) {
        case ACT_GELU:   return gelu(x);
        case ACT_SWIGLU: return silu(x);  /* gate * silu(up), caller handles gate */
        case ACT_SILU:   return silu(x);
        default:         return x;
    }
}

float act_grad(float x, ActType type) {
    switch (type) {
        case ACT_GELU:   return gelu_grad(x);
        case ACT_SWIGLU: return silu_grad(x);
        case ACT_SILU:   return silu_grad(x);
        default:         return 1.0f;
    }
}

void apply_rope(float *q, float *k, int seq_len, int n_head, int head_dim, int n_embd) {
    /* Simplified RoPE: rotate pairs by position-dependent angle */
    for (int h = 0; h < n_head; h++) {
        float *qh = q + h * head_dim;
        float *kh = k + h * head_dim;
        for (int d = 0; d < head_dim / 2; d++) {
            float angle = (float)seq_len / powf(10000.0f, (float)(2 * d) / head_dim);
            float c = cosf(angle), s = sinf(angle);
            float q0 = qh[d], q1 = qh[d + head_dim / 2];
            float k0 = kh[d], k1 = kh[d + head_dim / 2];
            qh[d] = q0 * c - q1 * s;
            qh[d + head_dim / 2] = q0 * s + q1 * c;
            kh[d] = k0 * c - k1 * s;
            kh[d + head_dim / 2] = k0 * s + k1 * c;
        }
    }
}

/* ========================================================================
 * Level 2: Transformer Layer (building block)
 * ======================================================================== */

void trans_layer_init(TransLayer *tl, Tensor *tensors, int n_tensors,
                      ModelConfig *cfg, int layer_idx,
                      const char *qkv_key, const char *q_key, const char *k_key,
                      const char *v_key, const char *o_key,
                      const char *gate_key, const char *up_key, const char *down_key,
                      const char *norm1_w_key, const char *norm1_b_key,
                      const char *norm2_w_key, const char *norm2_b_key) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    tl->_kv_k = NULL;
    tl->_kv_v = NULL;
    char full_key[256];

    if (cfg->qkv_merged) {
        /* GPT-2: merged QKV [n → 3n] */
        sprintf(full_key, qkv_key, layer_idx);
        float *W = tensor_get(tensors, n_tensors, full_key);
        sprintf(full_key, "%s.bias", full_key);
        /* Remove ".weight" suffix for bias — actually qkv_key already has .weight */
        /* The key format is like "h.%d.attn.c_attn.weight" */
        char bias_key[256];
        strncpy(bias_key, full_key, sizeof(bias_key));
        /* Replace ".weight" with ".bias" */
        char *dot = strstr(bias_key, ".weight");
        if (dot) { *dot = 0; strcat(bias_key, ".bias"); }
        float *b = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_q, W, b, n, 3 * n);
    } else {
        /* LLaMA/Qwen: separate Q, K, V, O */
        sprintf(full_key, q_key, layer_idx);
        char bias_key[256];
        float *Wq = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        char *dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bq = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_q, Wq, bq, n, n);

        sprintf(full_key, k_key, layer_idx);
        float *Wk = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bk = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_k, Wk, bk, n, n);

        sprintf(full_key, v_key, layer_idx);
        float *Wv = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bv = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->attn_v, Wv, bv, n, n);
    }

    /* Output projection */
    sprintf(full_key, o_key, layer_idx);
    float *Wo = tensor_get(tensors, n_tensors, full_key);
    char bias_key[256]; strncpy(bias_key, full_key, sizeof(bias_key));
    char *dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
    float *bo = tensor_get(tensors, n_tensors, bias_key);
    bin_layer_init(&tl->attn_o, Wo, bo, n, n);

    /* MLP */
    if (cfg->act_type == ACT_SWIGLU) {
        sprintf(full_key, gate_key, layer_idx);
        float *Wg = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bg = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_gate, Wg, bg, n, m);

        sprintf(full_key, up_key, layer_idx);
        float *Wu = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bu = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_up, Wu, bu, n, m);
    } else {
        /* GELU: single c_fc */
        sprintf(full_key, gate_key, layer_idx);
        float *Wg = tensor_get(tensors, n_tensors, full_key);
        strncpy(bias_key, full_key, sizeof(bias_key));
        dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
        float *bg = tensor_get(tensors, n_tensors, bias_key);
        bin_layer_init(&tl->mlp_gate, Wg, bg, n, m);
    }

    sprintf(full_key, down_key, layer_idx);
    float *Wd = tensor_get(tensors, n_tensors, full_key);
    strncpy(bias_key, full_key, sizeof(bias_key));
    dot = strstr(bias_key, ".weight"); if (dot) { *dot=0; strcat(bias_key, ".bias"); }
    float *bd = tensor_get(tensors, n_tensors, bias_key);
    bin_layer_init(&tl->mlp_down, Wd, bd, m, n);

    /* Norm weights */
    sprintf(full_key, norm1_w_key, layer_idx);
    tl->norm1_w = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm1_b_key, layer_idx);
    tl->norm1_b = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm2_w_key, layer_idx);
    tl->norm2_w = tensor_get(tensors, n_tensors, full_key);
    sprintf(full_key, norm2_b_key, layer_idx);
    tl->norm2_b = tensor_get(tensors, n_tensors, full_key);
}

void trans_layer_free(TransLayer *tl, ModelConfig *cfg) {
    bin_layer_free(&tl->attn_q);
    if (!cfg->qkv_merged) { bin_layer_free(&tl->attn_k); bin_layer_free(&tl->attn_v); }
    bin_layer_free(&tl->attn_o);
    bin_layer_free(&tl->mlp_gate);
    if (cfg->act_type == ACT_SWIGLU) bin_layer_free(&tl->mlp_up);
    bin_layer_free(&tl->mlp_down);
}

/* Dispatch: pure float > BNN fast path > standard BWN */
static inline void bin_fwd(float *y, const float *x, const BinLayer *bl) {
    if (g_use_pure_float)      bin_forward_pure_float(y, x, bl);
    else if (g_use_bnn_fast_path) bin_forward_bnn(y, x, bl);
    else                       bin_forward(y, x, bl);
}

void trans_layer_forward(float *x, TransLayer *tl, TransAct *act,
                         ModelConfig *cfg, int seq_pos) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    act->seq_pos = seq_pos;  /* cached for attention backward */

    /* Save x before norm1 */
    memcpy(act->x_pre_norm1, x, n * sizeof(float));
    norm_forward(act->norm1_out, x, tl->norm1_w, tl->norm1_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm1, n, &act->norm1_cache[0], &act->norm1_cache[1]);

    /* Attention */
    if (cfg->qkv_merged) {
        bin_fwd(act->q, act->norm1_out, &tl->attn_q);
        act->k = act->q + n;
        act->v = act->q + 2 * n;
    } else {
        bin_fwd(act->q, act->norm1_out, &tl->attn_q);
        bin_fwd(act->k, act->norm1_out, &tl->attn_k);
        bin_fwd(act->v, act->norm1_out, &tl->attn_v);
    }

    if (cfg->attn_type == ATTN_ROPE)
        apply_rope(act->q, act->k, seq_pos, cfg->n_head, n / cfg->n_head, n);

    /* Simplified attention: V copy (full attention in future) */
    memcpy(act->attn_out, act->v, n * sizeof(float));
    /* Attention: real causal multi-head (KV cache) if flag is on and cache
     * is allocated, else legacy V-copy (degenerate, no token mixing).
     * [FIX 致命2] The V-copy was a placeholder — see attention_forward(). */
    if (g_use_real_attention && tl->_kv_k && tl->_kv_v) {
        attention_forward(act->attn_out, act->q, n, cfg->n_head, seq_pos,
                          tl->_kv_k, tl->_kv_v);
    } else {
        /* Legacy V-copy (degenerate, no QK mixing). */
        memcpy(act->attn_out, act->v, n * sizeof(float));
    }
    bin_fwd(act->proj_out, act->attn_out, &tl->attn_o);
    for (int i = 0; i < n; i++) x[i] += rs * act->proj_out[i];
    /* BUG #48 FIX: normalize residual stream to prevent ||x|| explosion */
    normalize_residual(x, n, 3.0f);

    /* MLP */
    memcpy(act->x_pre_norm2, x, n * sizeof(float));
    norm_forward(act->norm2_out, x, tl->norm2_w, tl->norm2_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm2, n, &act->norm2_cache[0], &act->norm2_cache[1]);

    if (cfg->act_type == ACT_SWIGLU) {
        /* BUG #44 FIX: static buffer instead of malloc/free per call.
         * BUG #45 FIX: cache gate/up in act->swiglu_gate/swiglu_up for backward.
         *   Backward needs gate (pre-SiLU) to compute silu_grad(gate) * up.
         *   Without caching, backward used silu_grad(hidden) which is wrong. */
        static float *sgate = NULL, *sup = NULL;
        static int sg_m = 0;
        if (sg_m != m) {
            free(sgate); free(sup);
            sgate = malloc(m * sizeof(float));
            sup = malloc(m * sizeof(float));
            sg_m = m;
        }
        bin_fwd(sgate, act->norm2_out, &tl->mlp_gate);
        bin_fwd(sup,   act->norm2_out, &tl->mlp_up);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = silu(sgate[i]) * sup[i];
        /* Cache for backward */
        if (act->swiglu_gate) memcpy(act->swiglu_gate, sgate, m * sizeof(float));
        if (act->swiglu_up) memcpy(act->swiglu_up, sup, m * sizeof(float));
    } else {
        /* GELU: hidden = gelu(c_fc(ln2)) */
        bin_fwd(act->mlp_hidden, act->norm2_out, &tl->mlp_gate);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = gelu(act->mlp_hidden[i]);
    }

    bin_fwd(act->mlp_out, act->mlp_hidden, &tl->mlp_down);
    /* BUG #46/#47 FIX: MLP output normalization (matches sliding window path).
     * Without this, training uses raw MLP output but inference normalizes
     * MLP to match attention magnitude → train/infer mismatch.
     * Also prevents ||x|| explosion: MLP weights grow larger than attention
     * (CORE has 3x lr multiplier), so raw MLP output dominates the residual,
     * causing ||x|| to grow from ~1 (L0) to ~210 (L7) over layers.
     * The clip_array(x, 10) can't prevent this because individual elements
     * may be <10 but the vector norm still grows. */
    {
        float mlp_norm = 0;
        for (int i = 0; i < n; i++) mlp_norm += act->mlp_out[i] * act->mlp_out[i];
        mlp_norm = sqrtf(mlp_norm) + 1e-8f;
        float attn_norm = 0;
        for (int i = 0; i < n; i++) attn_norm += act->proj_out[i] * act->proj_out[i];
        attn_norm = sqrtf(attn_norm) + 1e-8f;
        float mlp_scale = attn_norm / mlp_norm;
        for (int i = 0; i < n; i++) x[i] += rs * (act->proj_out[i] + mlp_scale * act->mlp_out[i]);
    }
    /* BUG #48 FIX: normalize residual stream after MLP residual too */
    normalize_residual(x, n, 3.0f);
}

/* Pure-float forward: same as trans_layer_forward but uses bin_forward_pure_float
 * for every matmul (no sign binarization anywhere). Used by the teacher model
 * in distillation — w_float holds original GPT-2 weights, never updated.
 * Activations cache is shared with the student's structure (same shape) so we
 * can reuse m->acts. NOTE: this does NOT overwrite student activations if
 * called on a separate teacher Model (m->acts is per-model). */
void trans_layer_forward_pure_float(float *x, TransLayer *tl, TransAct *act,
                                    ModelConfig *cfg, int seq_pos) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    act->seq_pos = seq_pos;

    memcpy(act->x_pre_norm1, x, n * sizeof(float));
    norm_forward(act->norm1_out, x, tl->norm1_w, tl->norm1_b, cfg->norm_type, n);

    if (cfg->qkv_merged) {
        bin_forward_pure_float(act->q, act->norm1_out, &tl->attn_q);
        act->k = act->q + n;
        act->v = act->q + 2 * n;
    } else {
        bin_forward_pure_float(act->q, act->norm1_out, &tl->attn_q);
        bin_forward_pure_float(act->k, act->norm1_out, &tl->attn_k);
        bin_forward_pure_float(act->v, act->norm1_out, &tl->attn_v);
    }

    if (cfg->attn_type == ATTN_ROPE)
        apply_rope(act->q, act->k, seq_pos, cfg->n_head, n / cfg->n_head, n);

    if (g_use_real_attention && tl->_kv_k && tl->_kv_v)
        attention_forward(act->attn_out, act->q, n, cfg->n_head, seq_pos,
                          tl->_kv_k, tl->_kv_v);
    else
        memcpy(act->attn_out, act->v, n * sizeof(float));

    bin_forward_pure_float(act->proj_out, act->attn_out, &tl->attn_o);
    for (int i = 0; i < n; i++) x[i] += rs * act->proj_out[i];
    /* BUG #48 FIX: normalize residual stream */
    normalize_residual(x, n, 3.0f);

    memcpy(act->x_pre_norm2, x, n * sizeof(float));
    norm_forward(act->norm2_out, x, tl->norm2_w, tl->norm2_b, cfg->norm_type, n);

    if (cfg->act_type == ACT_SWIGLU) {
        /* BUG #44 FIX: static buffer instead of malloc/free per call */
        static float *sgate = NULL, *sup = NULL;
        static int sg_m = 0;
        if (sg_m != m) {
            free(sgate); free(sup);
            sgate = malloc(m * sizeof(float));
            sup = malloc(m * sizeof(float));
            sg_m = m;
        }
        bin_forward_pure_float(sgate, act->norm2_out, &tl->mlp_gate);
        bin_forward_pure_float(sup,   act->norm2_out, &tl->mlp_up);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = silu(sgate[i]) * sup[i];
    } else {
        bin_forward_pure_float(act->mlp_hidden, act->norm2_out, &tl->mlp_gate);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = gelu(act->mlp_hidden[i]);
    }

    bin_forward_pure_float(act->mlp_out, act->mlp_hidden, &tl->mlp_down);
    /* BUG #46/#47 FIX: MLP normalization (same as trans_layer_forward) */
    {
        float mlp_norm = 0;
        for (int i = 0; i < n; i++) mlp_norm += act->mlp_out[i] * act->mlp_out[i];
        mlp_norm = sqrtf(mlp_norm) + 1e-8f;
        float attn_norm = 0;
        for (int i = 0; i < n; i++) attn_norm += act->proj_out[i] * act->proj_out[i];
        attn_norm = sqrtf(attn_norm) + 1e-8f;
        float mlp_scale = attn_norm / mlp_norm;
        for (int i = 0; i < n; i++) x[i] += rs * (act->proj_out[i] + mlp_scale * act->mlp_out[i]);
    }
    /* BUG #48 FIX: normalize residual stream after MLP */
    normalize_residual(x, n, 3.0f);
}

/* Global flag: use STE backward (updates w_float + repacks wbits) */
int g_use_ste = 1;  /* LAL default: STE mode (train=infer, learn binary logic directly) */
int g_use_cuda = 0;            /* 1 = dispatch matmul to CUDA backend */
int g_use_logic_binarization = 1;  /* LAL default: logic-guided layers (CORE/BINARY/PRUNE semantic structure) */

/* Semantic logic mask ratios (set by training script per curriculum phase).
 * When g_logic_core_ratio > 0, compute_norm_mask uses these instead of
 * the hardcoded 20%/70%/10% split. This enables progressive activation:
 * early stages are sparse (high PRUNE), later stages are dense. */
float g_logic_core_ratio = 0.0f;   /* 0 = use default 20% */
float g_logic_prune_ratio = 0.0f; /* 0 = use default 10% */

/* Adam optimizer globals (used inside bin_backward_ste when g_use_adam=1).
 * Defaults are standard Adam (Kingma & Ba 2015).
 * g_opt_step is incremented per model_backward call to drive bias correction. */
int   g_use_adam = 0;
int   g_opt_step = 0;
float g_adam_beta1 = 0.9f;
float g_adam_beta2 = 0.999f;
float g_adam_eps    = 1e-8f;

/* Ternary Weight Network (TWN) globals.
 * When g_use_ternary=1, BINARY rows use {-1,0,+1}: |W|<=Δ is zeroed (Δ stored
 * per-layer in BinLayer.ternary_delta). Triples capacity vs BWN at ~1.58 bits. */
int   g_use_ternary = 0;
float g_ternary_delta_factor = 0.7f;  /* Δ = factor * mean(|W_row|), TWN default */

/* Cosine LR with linear warmup.
 *   step < warmup          : lr = base * (step+1) / warmup    (linear ramp from 0)
 *   warmup <= step < total : lr = base * 0.5 * (1 + cos(pi * progress))  (cosine)
 *   step >= total          : lr = base * 0.01                  (floor — keep updating)
 * Warmup tames the early-step gradient explosion (STE on bit-space is noisy).
 * Cosine decay reduces late-step oscillation for convergence.
 * Pass warmup=0 to disable warmup, total=0 to disable decay. */
float lr_schedule(int step, int warmup_steps, int total_steps, float base_lr) {
    if (warmup_steps > 0 && step < warmup_steps) {
        return base_lr * (float)(step + 1) / (float)warmup_steps;
    }
    if (total_steps <= warmup_steps) return base_lr;  /* degenerate: no decay */
    if (step >= total_steps) return base_lr * 0.01f;   /* floor */
    float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * progress));
}

/* Pure float forward: y[j] = sum_i w_float[j*in+i] * x[i] + bias[j].
 * Skips sign binarization entirely. Used for the teacher model in
 * distillation — the teacher's w_float holds the original GPT-2 weights
 * and is never updated, so this is a faithful full-precision matmul.
 * Logic-guided layers: CORE uses w_core (already float), BINARY uses w_float,
 * PRUNE outputs 0 (skipped). */
void bin_forward_pure_float(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim;
    if (bl->logic_mask) {
        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            uint8_t m = bl->logic_mask[j];
            if (m == 0) {  /* CORE: float dot with w_core[core_idx] */
                const float *wc = &bl->w_core[core_idx * in];
                float s = bl->bias[j];
                for (int i = 0; i < in; i++) s += wc[i] * x[i];
                y[j] = s;
                core_idx++;
            } else if (m == 1) {  /* BINARY: float dot with w_float */
                const float *wf = &bl->w_float[j * in];
                float s = bl->bias[j];
                for (int i = 0; i < in; i++) s += wf[i] * x[i];
                y[j] = s;
            } else {
                y[j] = 0.0f;  /* PRUNE */
            }
        }
    } else {
        for (int j = 0; j < out; j++) {
            const float *wf = &bl->w_float[j * in];
            float s = bl->bias[j];
            for (int i = 0; i < in; i++) s += wf[i] * x[i];
            y[j] = s;
        }
    }
}

/* Auto-generate per-output logic mask based on weight norms.
 * W is [in, out] (GPT-2 Conv1D format). We compute per-output column norms.
 * top 20% → CORE (0), bottom 10% → PRUNE (2), middle 70% → BINARY (1).
 * mask: [out_dim] bytes, 0=CORE, 1=BINARY, 2=PRUNE. */
static void compute_norm_mask(const float *W, int in_dim, int out_dim, uint8_t *mask) {
    /* Compute per-output norms (W is [in, out] row-major) */
    float *norms = malloc(out_dim * sizeof(float));
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = W[i * out_dim + j];
            s += w * w;
        }
        norms[j] = sqrtf(s);
    }
    /* Find thresholds via partial sort (simple: sort a copy) */
    float *sorted = malloc(out_dim * sizeof(float));
    memcpy(sorted, norms, out_dim * sizeof(float));
    /* Simple insertion sort (out_dim ≤ 3072, OK) */
    for (int i = 1; i < out_dim; i++) {
        float v = sorted[i]; int k = i - 1;
        while (k >= 0 && sorted[k] > v) { sorted[k+1] = sorted[k]; k--; }
        sorted[k+1] = v;
    }

    /* Use semantic ratios when set, otherwise default 20%/10% */
    float core_r = (g_logic_core_ratio > 0.0f) ? g_logic_core_ratio : 0.20f;
    float prune_r = (g_logic_prune_ratio > 0.0f) ? g_logic_prune_ratio : 0.10f;

    int core_count = (int)(out_dim * core_r);
    int prune_count = (int)(out_dim * prune_r);
    if (core_count < 1) core_count = 1;
    if (core_count + prune_count > out_dim) prune_count = out_dim - core_count;

    /* sorted[0] = smallest norm, sorted[out_dim-1] = largest */
    float core_threshold = sorted[out_dim - core_count];
    float prune_threshold = (prune_count > 0) ? sorted[prune_count - 1] : -1.0f;

    int n_core = 0, n_binary = 0, n_prune = 0;
    for (int j = 0; j < out_dim; j++) {
        if (norms[j] >= core_threshold && n_core < core_count) {
            mask[j] = 0;       /* CORE */
            n_core++;
        } else if (norms[j] <= prune_threshold && n_prune < prune_count) {
            mask[j] = 2;      /* PRUNE */
            n_prune++;
        } else {
            mask[j] = 1;      /* BINARY */
            n_binary++;
        }
    }

    static int first_call = 1;
    if (first_call) {
        printf("    [logic] CORE=%d (%.0f%%), BINARY=%d (%.0f%%), PRUNE=%d (%.0f%%)\n",
               n_core, 100.0f * n_core / out_dim,
               n_binary, 100.0f * n_binary / out_dim,
               n_prune, 100.0f * n_prune / out_dim);
        first_call = 0;
    }

    free(norms); free(sorted);
}

/* Global flag: use legacy BNN fast path (binarizes x too). Off by default —
 * BNN causes train/inference mismatch. Enable only for max-speed-low-quality. */
int g_use_bnn_fast_path = 0;
float g_core_lr_multiplier = 3.0f;  /* CORE neurons learn 3x faster than BINARY */
int   g_use_lal_adam = 1;        /* 1=group-wise Adam (LAL-aware), 0=standard per-param Adam */
float g_prune_decay = 0.01f;     /* PRUNE weight decay per step (pulls toward 0) */
float g_prune_freeze_thresh = 0.001f; /* PRUNE neurons below this are frozen */

/* Global flag: use real causal multi-head self-attention with KV cache.
 * Off by default — backward compat with V-copy. When on, trans_layer_forward
 * calls attention_forward() instead of memcpy(act->attn_out, act->v, n). */
int g_use_real_attention = 0;
int g_use_pure_float = 0;
int g_accumulate_gradients = 0;  /* 1 = accumulate grads, don't update weights */

void trans_layer_backward(float *grad_x, TransLayer *tl, TransAct *act,
                          ModelConfig *cfg, float lr) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    /* BUGFIX: static buffers were sized [4096], which overflows for any
     * config with mlp_dim > 4096 (e.g. the 1B config: mlp_dim=8192) or
     * n_embd > 4096. Bumped to 16384 to cover up to ~7B-class configs.
     * g_qkv is 3*n_embd so it needs 3*16384 = 49152 entries. */
    static float g_mlp[16384], g_hidden[16384], g_norm2[16384], g_proj[16384];
    static float g_attn[16384], g_qkv[16384*3], g_norm1[16384], g_pre[16384];

    /* Choose backward function based on STE flag */
    #define BIN_BW(gx, gy, x, bl, lr) \
        (g_use_ste ? bin_backward_ste(gx, gy, x, bl, lr) : bin_backward(gx, gy, x, bl, lr))

    /* MLP backward */
    for (int i = 0; i < n; i++) g_mlp[i] = grad_x[i] * rs;
    BIN_BW(g_hidden, g_mlp, act->mlp_hidden, &tl->mlp_down, lr);
    if (cfg->act_type == ACT_SWIGLU) {
        /* BUG #45 FIX: correct SwiGLU backward.
         * Forward: hidden[i] = silu(gate[i]) * up[i]
         *   d(hidden)/d(gate) = silu_grad(gate) * up
         *   d(hidden)/d(up) = silu(gate)
         * So: g_gate[i] = g_hidden[i] * silu_grad(gate[i]) * up[i]
         *     g_up[i]   = g_hidden[i] * silu(gate[i])
         * Then g_norm2 = W_gate^T · g_gate + W_up^T · g_up
         *
         * Old code (WRONG):
         *   g_hidden[i] *= silu_grad(act->mlp_hidden[i])  // used hidden, not gate!
         *   BIN_BW(g_norm2, g_hidden, ...)                // only backprop'd mlp_gate
         * Missing: mlp_up gradient entirely. And silu_grad(hidden) != silu_grad(gate). */
        static float g_gate[16384], g_up[16384];
        static float g_norm2_gate[16384], g_norm2_up[16384];
        for (int i = 0; i < m; i++) {
            float sv = silu(act->swiglu_gate[i]);
            float sg = silu_grad(act->swiglu_gate[i]);
            g_gate[i] = g_hidden[i] * sg * act->swiglu_up[i];
            g_up[i]   = g_hidden[i] * sv;
        }
        BIN_BW(g_norm2_gate, g_gate, act->norm2_out, &tl->mlp_gate, lr);
        BIN_BW(g_norm2_up,   g_up,   act->norm2_out, &tl->mlp_up,   lr);
        for (int i = 0; i < n; i++) g_norm2[i] = g_norm2_gate[i] + g_norm2_up[i];
    } else {
        for (int i = 0; i < m; i++) g_hidden[i] *= gelu_grad(act->mlp_hidden[i]);
        BIN_BW(g_norm2, g_hidden, act->norm2_out, &tl->mlp_gate, lr);
    }
    norm_backward(g_pre, g_norm2, act->x_pre_norm2, tl->norm2_w,
                  act->norm2_cache, cfg->norm_type, n,
                  tl->grad_norm2_w, tl->grad_norm2_b);
    for (int i = 0; i < n; i++) grad_x[i] += g_pre[i] * rs;

    /* Attention backward */
    for (int i = 0; i < n; i++) g_proj[i] = grad_x[i] * rs;
    BIN_BW(g_attn, g_proj, act->attn_out, &tl->attn_o, lr);
    if (g_use_real_attention && tl->_kv_k && tl->_kv_v) {
        /* Real attention: dQ/dK/dV at the current position. act->q is the
         * contiguous [Q|K|V] buffer (k=q+n, v=q+2n), valid for both merged
         * and separate QKV layouts. */
        attention_backward(g_qkv, g_attn, act->q, n, cfg->n_head,
                           act->seq_pos, tl->_kv_k, tl->_kv_v);
    } else {
        /* Legacy V-copy: only V receives gradient (Q, K grad = 0). */
        memset(g_qkv, 0, 3 * n * sizeof(float));
        memcpy(g_qkv + 2 * n, g_attn, n * sizeof(float));
    }
    if (cfg->qkv_merged) {
        /* GPT-2: attn_q is the merged [in→3n] projection. */
        BIN_BW(g_norm1, g_qkv, act->norm1_out, &tl->attn_q, lr);
    } else {
        /* LLaMA/Qwen: separate Q/K/V projections — backprop each. */
        static float g_n1k[4096], g_n1v[4096];
        BIN_BW(g_norm1,  g_qkv,       act->norm1_out, &tl->attn_q, lr);
        BIN_BW(g_n1k,    g_qkv + n,   act->norm1_out, &tl->attn_k, lr);
        BIN_BW(g_n1v,    g_qkv + 2*n, act->norm1_out, &tl->attn_v, lr);
        for (int i = 0; i < n; i++) g_norm1[i] += g_n1k[i] + g_n1v[i];
    }
    norm_backward(g_pre, g_norm1, act->x_pre_norm1, tl->norm1_w,
                  act->norm1_cache, cfg->norm_type, n,
                  tl->grad_norm1_w, tl->grad_norm1_b);
    for (int i = 0; i < n; i++) grad_x[i] += g_pre[i] * rs;

    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += grad_x[i] * grad_x[i];
    gnorm = sqrtf(gnorm);
    if (gnorm > 1.0f) { float clip = 1.0f / gnorm; for (int i = 0; i < n; i++) grad_x[i] *= clip; }
}

TransAct *trans_act_alloc(ModelConfig *cfg) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    TransAct *acts = malloc(cfg->n_layer * sizeof(TransAct));
    for (int l = 0; l < cfg->n_layer; l++) {
        acts[l].x_pre_norm1 = malloc(n * sizeof(float));
        acts[l].norm1_out = malloc(n * sizeof(float));
        acts[l].q = malloc(3 * n * sizeof(float));
        /* k/v alias into the contiguous Q|K|V buffer so both merged (GPT-2)
         * and separate (LLaMA/Qwen) paths share one [3n] layout. Previously
         * k/v were left NULL for the separate path → segfault. */
        acts[l].k = acts[l].q + n;
        acts[l].v = acts[l].q + 2 * n;
        acts[l].attn_out = malloc(n * sizeof(float));
        acts[l].proj_out = malloc(n * sizeof(float));
        acts[l].x_pre_norm2 = malloc(n * sizeof(float));
        acts[l].norm2_out = malloc(n * sizeof(float));
        acts[l].mlp_hidden = malloc(m * sizeof(float));
        acts[l].mlp_out = malloc(n * sizeof(float));
        /* BUG #45 FIX: allocate SwiGLU gate/up cache (NULL for GELU mode) */
        if (cfg->act_type == ACT_SWIGLU) {
            acts[l].swiglu_gate = malloc(m * sizeof(float));
            acts[l].swiglu_up = malloc(m * sizeof(float));
        } else {
            acts[l].swiglu_gate = NULL;
            acts[l].swiglu_up = NULL;
        }
    }
    return acts;
}

void trans_act_free(TransAct *acts, int n_layer) {
    for (int l = 0; l < n_layer; l++) {
        free(acts[l].x_pre_norm1); free(acts[l].norm1_out);
        free(acts[l].q); free(acts[l].attn_out); free(acts[l].proj_out);
        free(acts[l].x_pre_norm2); free(acts[l].norm2_out);
        free(acts[l].mlp_hidden); free(acts[l].mlp_out);
        free(acts[l].swiglu_gate); free(acts[l].swiglu_up);
    }
    free(acts);
}

/* ========================================================================
 * Level 3: Full Model
 * ======================================================================== */

/* ----- Causal Multi-Head Self-Attention (KV cache) -----
 * Replaces the degenerate V-copy in trans_layer_forward.
 * Mirrors tools/server/gpt2_server.c:real_attention (scalar version).
 *
 * Layout:
 *   qkv:        [3 * n_embd]  — Q | K | V concatenated, single token
 *   k_cache_layer / v_cache_layer: [n_ctx * n_embd] — filled position-by-position
 *   attn_out:   [n_embd]      — output, weighted sum of V across heads
 *
 * Causal: position seq_pos attends only to positions 0..seq_pos (inclusive).
 * Multi-head: n_head heads, head_dim = n_embd / n_head (must divide evenly).
 */
void attention_forward(float *attn_out, const float *qkv,
                       int n_embd, int n_head,
                       int seq_pos,
                       float *k_cache_layer, float *v_cache_layer) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    const float *Q = qkv;                  /* [n_embd] */
    const float *K_new = qkv + n_embd;
    const float *V_new = qkv + 2 * n_embd;

    /* Store current K, V into cache at position seq_pos */
    memcpy(k_cache_layer + (size_t)seq_pos * n_embd, K_new, n_embd * sizeof(float));
    memcpy(v_cache_layer + (size_t)seq_pos * n_embd, V_new, n_embd * sizeof(float));

    /* Stack scratch — max n_ctx per head. 1024 is GPT-2 default; if n_ctx
     * grows beyond this, switch to malloc. */
    float scores[1024];
    float attn_weights[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        /* scores[j] = Q_h · K[j, h] * scale, j = 0..seq_pos (causal) */
        float max_score = -1e30f;
        int n_attend = seq_pos + 1;  /* positions 0..seq_pos inclusive */
        if (n_attend > 1024) n_attend = 1024;  /* clip to scratch size */
        for (int j = 0; j < n_attend; j++) {
            const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }
        /* Softmax with max subtraction (numerical stability) */
        float sum_exp = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float e = expf(scores[j] - max_score);
            attn_weights[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j < n_attend; j++) attn_weights[j] *= inv_sum;
        /* Weighted sum: out_h[d] = sum_j attn_weights[j] * V[j, h, d] */
        float *out_h = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float w = attn_weights[j];
            const float *V_jh = v_cache_layer + (size_t)j * n_embd + h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] += w * V_jh[d];
        }
    }
}

/* ----- Attention backward (dQ/dK/dV) -----
 * Computes gradients for the current token's Q, K, V. Cached K/V at positions
 * 0..seq_pos-1 are treated as constants (they are context, not learned here —
 * only the current token's QKV projection receives gradient, matching the
 * single-position activation cache used by model_forward/backward).
 *
 * Per head h (head_dim d, scale = 1/sqrt(head_dim)):
 *   forward: scores[j]=Q·K_j*scale; w=softmax(scores); out=sum_j w[j]*V_j
 *   backward:
 *     g_w[j]      = <g_out, V_j>                       (grad w.r.t. weight j)
 *     g_scores[j] = w[j] * (g_w[j] - <g_w, w>)         (softmax bwd)
 *     g_Q[d]     += sum_j g_scores[j] * K_j[d] * scale
 *     g_K_cur[d] += g_scores[seq_pos] * Q[d] * scale   (current K only)
 *     g_V_cur[d] += w[seq_pos] * g_out[d]              (current V only)
 */
void attention_backward(float *grad_qkv, const float *grad_attn_out,
                        const float *qkv, int n_embd, int n_head, int seq_pos,
                        const float *k_cache_layer, const float *v_cache_layer) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_attend = seq_pos + 1;  /* positions 0..seq_pos inclusive */
    if (n_attend > 1024) n_attend = 1024;  /* clip to scratch size */

    const float *Q = qkv;
    float *gQ = grad_qkv;
    float *gK = grad_qkv + n_embd;      /* grad K at current position */
    float *gV = grad_qkv + 2 * n_embd;  /* grad V at current position */
    memset(grad_qkv, 0, 3 * n_embd * sizeof(float));

    /* If the current position fell outside the (clipped) attended window,
     * there is no self-attention gradient to propagate. */
    int have_self = (seq_pos >= 0 && seq_pos < n_attend);

    float scores[1024], w[1024], g_w[1024];

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        const float *g_out_h = grad_attn_out + h * head_dim;

        /* Recompute scores + softmax weights (K is in the cache). */
        float max_score = -1e30f;
        for (int j = 0; j < n_attend; j++) {
            const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            float e = expf(scores[j] - max_score);
            w[j] = e; sum_exp += e;
        }
        float inv = 1.0f / (sum_exp + 1e-12f);
        for (int j = 0; j < n_attend; j++) w[j] *= inv;

        /* g_w[j] = <g_out, V_j>; dot_gw_w = <g_w, w> */
        float dot_gw_w = 0.0f;
        for (int j = 0; j < n_attend; j++) {
            const float *V_jh = v_cache_layer + (size_t)j * n_embd + h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += g_out_h[d] * V_jh[d];
            g_w[j] = s;
            dot_gw_w += w[j] * s;
        }
        /* g_scores[j] = w[j] * (g_w[j] - dot_gw_w)  (now reuse g_w buffer) */
        for (int j = 0; j < n_attend; j++) g_w[j] = w[j] * (g_w[j] - dot_gw_w);

        /* g_Q[d] += sum_j g_scores[j] * K_j[d] * scale */
        float *gQ_h = gQ + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float s = 0.0f;
            for (int j = 0; j < n_attend; j++) {
                const float *K_jh = k_cache_layer + (size_t)j * n_embd + h * head_dim;
                s += g_w[j] * K_jh[d];
            }
            gQ_h[d] += s * scale;
        }

        if (have_self) {
            /* g_K_cur[d] += g_scores[seq_pos] * Q[d] * scale  (current K only) */
            float gs_cur = g_w[seq_pos] * scale;
            float *gK_h = gK + h * head_dim;
            for (int d = 0; d < head_dim; d++) gK_h[d] += gs_cur * Q_h[d];
            /* g_V_cur[d] += w[seq_pos] * g_out[d]  (current V only) */
            float w_cur = w[seq_pos];
            float *gV_h = gV + h * head_dim;
            for (int d = 0; d < head_dim; d++) gV_h[d] += w_cur * g_out_h[d];
        }
    }
}

void model_kv_cache_alloc(Model *m) {
    if (m->k_cache) return;  /* idempotent */
    int n_layer = m->cfg.n_layer;
    size_t per_layer = (size_t)m->cfg.n_ctx * m->cfg.n_embd * sizeof(float);
    m->k_cache = calloc(n_layer, sizeof(float *));
    m->v_cache = calloc(n_layer, sizeof(float *));
    for (int l = 0; l < n_layer; l++) {
        m->k_cache[l] = calloc(1, per_layer);
        m->v_cache[l] = calloc(1, per_layer);
        /* Wire into TransLayer so trans_layer_forward can find them */
        if (m->layers) {
            m->layers[l]._kv_k = m->k_cache[l];
            m->layers[l]._kv_v = m->v_cache[l];
        }
    }
}

void model_kv_cache_free(Model *m) {
    if (!m->k_cache) return;
    for (int l = 0; l < m->cfg.n_layer; l++) {
        free(m->k_cache[l]);
        free(m->v_cache[l]);
    }
    free(m->k_cache);
    free(m->v_cache);
    m->k_cache = NULL;
    m->v_cache = NULL;
}

/* FIX: get-or-realloc a thread-local scratch TransAct buffer that tracks
 * the model's current config. Previously this was a static pointer
 * allocated once for the first model and never updated — on phase switch
 * (n_embd change) the scratch was too small, causing heap-buffer-overflow
 * in trans_layer_forward's memcpy. */
static TransAct *get_scratch_acts(Model *m) {
    static TransAct *scratch = NULL;
    static int scratch_n_embd = 0;
    static int scratch_n_layer = 0;
    if (!scratch || scratch_n_embd != m->cfg.n_embd || scratch_n_layer != m->cfg.n_layer) {
        if (scratch) {
            trans_act_free(scratch, scratch_n_layer);  /* frees inner arrays + scratch itself */
            scratch = NULL;  /* trans_act_free already freed scratch; avoid double-free */
        }
        scratch = trans_act_alloc(&m->cfg);
        scratch_n_embd = m->cfg.n_embd;
        scratch_n_layer = m->cfg.n_layer;
    }
    return scratch;
}

void model_load(Model *m, const char *weight_path, ModelConfig cfg,
                const char *layer_prefix, int qkv_merged) {
    m->cfg = cfg;
    m->cfg.qkv_merged = qkv_merged;
    m->tensors = tensor_load_all(weight_path, &m->n_tensors);
    if (!m->tensors) { fprintf(stderr, "failed to load %s\n", weight_path); exit(1); }
    printf("[*] loaded %d tensors\n", m->n_tensors);

    m->wte = tensor_get(m->tensors, m->n_tensors, "wte.weight");
    m->wpe = (cfg.attn_type == ATTN_LEARNED)
        ? tensor_get(m->tensors, m->n_tensors, "wpe.weight") : NULL;
    m->ln_f_w = tensor_get(m->tensors, m->n_tensors, "ln_f.weight");
    m->ln_f_b = tensor_get(m->tensors, m->n_tensors, "ln_f.bias");

    printf("[*] binarizing %d layers%s...\n", cfg.n_layer,
           g_use_logic_binarization ? " (logic-guided)" : "");
    m->layers = calloc(cfg.n_layer, sizeof(TransLayer));  /* FIX: calloc (not malloc) zero-inits BinLayer fields like grad_accum so model_batch_alloc's NULL check works */
    m->acts = trans_act_alloc(&cfg);

    /* Build keys and binarize each layer */
    char key[256], bk[256];
    for (int l = 0; l < cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        int n = cfg.n_embd, mm = cfg.mlp_dim;

        /* Helper: bin_layer_init or bin_layer_init_logic depending on flag */
        #define BIN_INIT(bl, W, b, in, out) do { \
            if (g_use_logic_binarization) { \
                uint8_t *mask = malloc(out); \
                compute_norm_mask(W, in, out, mask); \
                bin_layer_init_logic(bl, W, b, in, out, mask); \
                free(mask); \
            } else { \
                bin_layer_init(bl, W, b, in, out); \
            } \
        } while(0)

        if (qkv_merged) {
            sprintf(key, "h.%d.attn.c_attn.weight", l);
            char bk[256]; strncpy(bk, key, sizeof(bk));
            char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_q, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, 3*n);
        } else {
            sprintf(key, "model.layers.%d.self_attn.q_proj.weight", l);
            char bk[256]; strncpy(bk, key, sizeof(bk));
            char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_q, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
            sprintf(key, "model.layers.%d.self_attn.k_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_k, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
            sprintf(key, "model.layers.%d.self_attn.v_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->attn_v, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, n);
        }

        sprintf(key, qkv_merged ? "h.%d.attn.c_proj.weight" : "model.layers.%d.self_attn.o_proj.weight", l);
        char bk[256]; strncpy(bk, key, sizeof(bk));
        char *dot = strstr(bk, ".weight"); if(dot){*dot=0;strcat(bk,".bias");}
        BIN_INIT(&tl->attn_o, tensor_get(m->tensors, m->n_tensors, key),
                 tensor_get(m->tensors, m->n_tensors, bk), n, n);

        if (cfg.act_type == ACT_SWIGLU) {
            sprintf(key, "model.layers.%d.mlp.gate_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_gate, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
            sprintf(key, "model.layers.%d.mlp.up_proj.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_up, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
        } else {
            sprintf(key, "h.%d.mlp.c_fc.weight", l);
            strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
            BIN_INIT(&tl->mlp_gate, tensor_get(m->tensors, m->n_tensors, key),
                     tensor_get(m->tensors, m->n_tensors, bk), n, mm);
        }

        sprintf(key, qkv_merged ? "h.%d.mlp.c_proj.weight" : "model.layers.%d.mlp.down_proj.weight", l);
        strncpy(bk, key, sizeof(bk)); dot=strstr(bk,".weight"); if(dot){*dot=0;strcat(bk,".bias");}
        BIN_INIT(&tl->mlp_down, tensor_get(m->tensors, m->n_tensors, key),
                 tensor_get(m->tensors, m->n_tensors, bk), mm, n);
        #undef BIN_INIT

        /* Norm weights */
        if (qkv_merged) {
            sprintf(key, "h.%d.ln_1.weight", l); tl->norm1_w = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_1.bias", l); tl->norm1_b = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_2.weight", l); tl->norm2_w = tensor_get(m->tensors, m->n_tensors, key);
            sprintf(key, "h.%d.ln_2.bias", l); tl->norm2_b = tensor_get(m->tensors, m->n_tensors, key);
        } else {
            sprintf(key, "model.layers.%d.input_layernorm.weight", l); tl->norm1_w = tensor_get(m->tensors, m->n_tensors, key);
            tl->norm1_b = NULL;
            sprintf(key, "model.layers.%d.post_attention_layernorm.weight", l); tl->norm2_w = tensor_get(m->tensors, m->n_tensors, key);
            tl->norm2_b = NULL;
        }
    }
    printf("[*] done\n");

    /* Free large weight matrix tensor data after binarization to save ~3.6GB.
     * Small tensors (wte, wpe, ln_f, per-layer norms) are kept for forward pass.
     * bin_layer_init copies all needed data into w_float/wbits/alpha/bias. */
    {
        int freed = 0;
        size_t freed_bytes = 0;
        for (int l = 0; l < cfg.n_layer; l++) {
            char wk[256];
            const char *mats[] = {
                qkv_merged ? "h.%d.attn.c_attn.weight" : "model.layers.%d.self_attn.q_proj.weight",
                qkv_merged ? "h.%d.attn.c_proj.weight" : "model.layers.%d.self_attn.o_proj.weight",
                qkv_merged ? "h.%d.mlp.c_fc.weight" : "model.layers.%d.mlp.gate_proj.weight",
                qkv_merged ? "h.%d.mlp.c_proj.weight" : "model.layers.%d.mlp.down_proj.weight",
            };
            for (int mi = 0; mi < 4; mi++) {
                sprintf(wk, mats[mi], l);
                for (int i = 0; i < m->n_tensors; i++) {
                    if (m->tensors[i].data && strcmp(m->tensors[i].key, wk) == 0) {
                        int n2 = 1;
                        for (int d = 0; d < m->tensors[i].ndim; d++) n2 *= m->tensors[i].shape[d];
                        freed_bytes += (size_t)n2 * sizeof(float);
                        free(m->tensors[i].data);
                        m->tensors[i].data = NULL;
                        freed++;
                        break;
                    }
                }
            }
        }
        printf("[*] freed %d weight tensors (%.0f MB) after binarization\n",
               freed, freed_bytes / 1e6);
    }

    m->final_ln = malloc(cfg.n_embd * sizeof(float));
    m->x_before_final = malloc(cfg.n_embd * sizeof(float));
    m->k_cache = NULL;
    m->v_cache = NULL;
    /* Auto-allocate KV cache if real attention is requested at load time.
     * Callers can also call model_kv_cache_alloc() later to enable it. */
    if (g_use_real_attention) model_kv_cache_alloc(m);
}

float model_forward(Model *m, const int *tokens, int n_tokens) {
    int n = m->cfg.n_embd;
    int nL = m->cfg.n_layer;
    static float x[4096];
    int t = n_tokens - 1;  /* predict tokens[t+1] from context tokens[0..t] */

    if (g_use_real_attention && m->k_cache) {
        /* Real attention needs the KV cache for positions 0..t to hold THIS
         * sequence's keys/values. The cache persists across training steps
         * (different sentences), so (re)fill 0..t-1 as context before running
         * the target position t. Context positions are run through all layers
         * with a scratch activation buffer (not stored); only their K/V land
         * in the cache and are treated as constants during backward. */
        static float xc[4096];
        TransAct *scratch = get_scratch_acts(m);
        for (int p = 0; p < t; p++) {
            for (int i = 0; i < n; i++) {
                xc[i] = m->wte[tokens[p] * n + i];
                if (m->wpe) xc[i] += m->wpe[p * n + i];
            }
            for (int l = 0; l < nL; l++)
                trans_layer_forward(xc, &m->layers[l], &scratch[l], &m->cfg, p);
        }
    }

    for (int i = 0; i < n; i++) {
        x[i] = m->wte[tokens[t] * n + i];
        if (m->wpe) x[i] += m->wpe[t * n + i];
    }

    for (int l = 0; l < nL; l++)
        trans_layer_forward(x, &m->layers[l], &m->acts[l], &m->cfg, t);

    memcpy(m->x_before_final, x, n * sizeof(float));
    norm_forward(m->final_ln, x, m->ln_f_w, m->ln_f_b, m->cfg.norm_type, n);
    compute_mean_std(m->x_before_final, n, &m->final_mean, &m->final_std_inv);

    int target = tokens[n_tokens];
    /* BUGFIX: seed was hardcoded to 42, which means the SAME 100 negative
     * samples were used for EVERY forward/backward pass. The model could
     * trivially overfit to "beat these 100 specific negatives" and then
     * stop learning — collapsing to a single token (764='.') that happened
     * to score high against the fixed negative set.
     *
     * Now seed varies per training step (g_opt_step is incremented at the
     * end of model_backward, so forward and backward within the same step
     * see the same seed -> loss and gradient are consistent). The Knuth
     * multiplicative hash spreads consecutive step values across the
     * uint32 space so negatives don't correlate between steps. */
    unsigned int seed = 42u + (unsigned int)g_opt_step * 2654435761u;
    /* Use FULL softmax (not sampled) so the gradient covers the entire
     * vocab and the model cannot collapse to a shortcut token. The static
     * logits buffer is shared between forward and backward within the same
     * step (forward writes softmax probs, backward consumes them). */
    static float *g_full_logits = NULL;
    static int g_full_logits_vocab = 0;
    if (g_full_logits_vocab != m->cfg.vocab_size) {
        free(g_full_logits);
        g_full_logits = malloc((size_t)m->cfg.vocab_size * sizeof(float));
        g_full_logits_vocab = m->cfg.vocab_size;
    }
    return cross_entropy_full(m->final_ln, m->wte, target, m->cfg.vocab_size, n, g_full_logits);
}

void model_backward(Model *m, const int *tokens, int n_tokens, float lr) {
    int n = m->cfg.n_embd;
    int target = tokens[n_tokens];
    static float gh[4096];
    /* BUGFIX: same seed derivation as model_forward — must match so that
     * the gradient corresponds to the same loss we just computed. */
    unsigned int seed = 42u + (unsigned int)g_opt_step * 2654435761u;
    (void)seed;  /* sampled-softmax path no longer used; kept for reference */
    /* Full-softmax gradient. Reuse the logits buffer that model_forward
     * populated (it holds softmax probabilities, not raw logits). */
    static float *g_full_logits = NULL;
    static int g_full_logits_vocab = 0;
    if (g_full_logits_vocab != m->cfg.vocab_size) {
        free(g_full_logits);
        g_full_logits = malloc((size_t)m->cfg.vocab_size * sizeof(float));
        g_full_logits_vocab = m->cfg.vocab_size;
    }
    /* Re-run forward's logit computation so g_full_logits holds the
     * current-step softmax probs (model_forward already computed loss
     * with this same buffer, but we re-populate to be safe in case the
     * caller didn't call model_forward first). */
    cross_entropy_full(m->final_ln, m->wte, target, m->cfg.vocab_size, n, g_full_logits);
    cross_entropy_full_grad(gh, m->final_ln, m->wte, target, m->cfg.vocab_size, n, g_full_logits);
    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += gh[i] * gh[i];
    gnorm = sqrtf(gnorm);
    /* Full-softmax gradients are much larger than sampled-softmax (they
     * sum over all 50257 vocab tokens, not 100 negatives). The old clip
     * threshold of 0.1 was calibrated for sampled-softmax magnitudes and
     * would shrink full-softmax gradients by ~1000x, freezing learning.
     * Bumped to 1.0 — still prevents NaN runaway, but lets the gradient
     * actually flow. */
    if (gnorm > 1.0f) { float clip = 1.0f / gnorm; for (int i = 0; i < n; i++) gh[i] *= clip; }

    static float g_pre[4096];
    norm_backward(g_pre, gh, m->x_before_final, m->ln_f_w,
                  (float[]){m->final_mean, m->final_std_inv}, m->cfg.norm_type, n,
                  m->grad_ln_f_w_accum, m->grad_ln_f_b_accum);
    memcpy(gh, g_pre, n * sizeof(float));

    for (int l = m->cfg.n_layer - 1; l >= 0; l--)
        trans_layer_backward(gh, &m->layers[l], &m->acts[l], &m->cfg, lr);

    /* Increment Adam step (for bias correction in next call). */
    if (g_use_adam) g_opt_step++;
}

/* Compute full vocab logits at target position using pure float forward.
 * Replaces bin_forward with bin_forward_pure_float for one pass (no
 * binarization anywhere). The result is the teacher signal for distillation.
 * Caller must allocate logits_out[vocab_size]. */
void model_forward_float_logits(Model *m, const int *tokens, int n_tokens,
                                float *logits_out) {
    int n = m->cfg.n_embd, nL = m->cfg.n_layer, vocab = m->cfg.vocab_size;
    int t = n_tokens - 1;
    static float x[4096];

    /* Save global flag, swap to pure-float mode, restore at end. */
    int saved = g_use_bnn_fast_path;
    g_use_bnn_fast_path = 0;  /* never BNN for teacher */

    /* Fill KV cache for context positions (if real attention is enabled). */
    if (g_use_real_attention && m->k_cache) {
        static float xc[4096];
        TransAct *scratch = get_scratch_acts(m);
        for (int p = 0; p < t; p++) {
            for (int i = 0; i < n; i++) {
                xc[i] = m->wte[tokens[p] * n + i];
                if (m->wpe) xc[i] += m->wpe[p * n + i];
            }
            for (int l = 0; l < nL; l++)
                trans_layer_forward_pure_float(xc, &m->layers[l], &scratch[l], &m->cfg, p);
        }
    }

    /* Run target position forward with pure float. */
    for (int i = 0; i < n; i++) {
        x[i] = m->wte[tokens[t] * n + i];
        if (m->wpe) x[i] += m->wpe[t * n + i];
    }
    for (int l = 0; l < nL; l++)
        trans_layer_forward_pure_float(x, &m->layers[l], &m->acts[l], &m->cfg, t);

    /* Compute logits over full vocab: logits[j] = dot(final_ln, wte[j]). */
    norm_forward(m->final_ln, x, m->ln_f_w, m->ln_f_b, m->cfg.norm_type, n);
    for (int j = 0; j < vocab; j++) {
        const float *w = &m->wte[(size_t)j * n];
        float s = 0;
        for (int i = 0; i + 7 < n; i += 8)
            s += m->final_ln[i+0]*w[i+0] + m->final_ln[i+1]*w[i+1]
               + m->final_ln[i+2]*w[i+2] + m->final_ln[i+3]*w[i+3]
               + m->final_ln[i+4]*w[i+4] + m->final_ln[i+5]*w[i+5]
               + m->final_ln[i+6]*w[i+6] + m->final_ln[i+7]*w[i+7];
        for (int i = (n/8)*8; i < n; i++) s += m->final_ln[i] * w[i];
        logits_out[j] = s;
    }
    g_use_bnn_fast_path = saved;
}

/* Backward with distillation: hard CE (target) + soft KL(teacher || student).
 * The KL gradient w.r.t. student logits[j] is:
 *   d_KL/d_s[j] = T * (softmax(s/T)[j] - softmax(t/T)[j])
 * Then w.r.t. final_ln[i]:
 *   d_KL/d_final_ln[i] = sum_j (T * (ps[j]-pt[j])) * wte[j*n+i]
 *
 * Combined grad on final_ln:
 *   gh[i] = alpha * CE_grad[i] + (1-alpha) * T^2 * KL_grad[i]
 * (T^2 because KL of T-scaled soft targets is conventionally multiplied by T^2
 *  to keep gradient magnitude roughly constant across T.)
 *
 * Teacher logits must be full vocab (computed by model_forward_float_logits).
 * Memory cost: ~3*vocab*sizeof(float) = 600KB scratch (heap-allocated here). */
void model_backward_distill(Model *m, const int *tokens, int n_tokens, float lr,
                            const float *teacher_logits,
                            float distill_alpha, float distill_T) {
    int n = m->cfg.n_embd, vocab = m->cfg.vocab_size;
    int target = tokens[n_tokens];

    /* 1. Compute student logits at target position over full vocab. */
    float *s_logits = malloc(vocab * sizeof(float));
    for (int j = 0; j < vocab; j++) {
        const float *w = &m->wte[(size_t)j * n];
        float s = 0;
        for (int i = 0; i + 7 < n; i += 8)
            s += m->final_ln[i+0]*w[i+0] + m->final_ln[i+1]*w[i+1]
               + m->final_ln[i+2]*w[i+2] + m->final_ln[i+3]*w[i+3]
               + m->final_ln[i+4]*w[i+4] + m->final_ln[i+5]*w[i+5]
               + m->final_ln[i+6]*w[i+6] + m->final_ln[i+7]*w[i+7];
        for (int i = (n/8)*8; i < n; i++) s += m->final_ln[i] * w[i];
        s_logits[j] = s;
    }

    /* 2. Softmax(student/T) and softmax(teacher/T) (numerically stable). */
    float *ps = malloc(vocab * sizeof(float));
    float *pt = malloc(vocab * sizeof(float));
    float ms = s_logits[0] / distill_T, mt = teacher_logits[0] / distill_T;
    for (int j = 1; j < vocab; j++) {
        if (s_logits[j] / distill_T > ms) ms = s_logits[j] / distill_T;
        if (teacher_logits[j] / distill_T > mt) mt = teacher_logits[j] / distill_T;
    }
    float ss = 0, st = 0;
    for (int j = 0; j < vocab; j++) {
        ps[j] = expf(s_logits[j] / distill_T - ms);
        pt[j] = expf(teacher_logits[j] / distill_T - mt);
        ss += ps[j]; st += pt[j];
    }
    for (int j = 0; j < vocab; j++) { ps[j] /= ss; pt[j] /= st; }

    /* 3. Hard CE grad w.r.t. final_ln (sampled softmax — matches model_backward). */
    static float gh_ce[4096];
    unsigned int seed = 42;
    cross_entropy_grad(gh_ce, m->final_ln, m->wte, target, vocab, n, 100, &seed);

    /* 4. KL grad w.r.t. final_ln: T * sum_j (ps[j] - pt[j]) * wte[j*n + i].
     * This is the heaviest op (vocab * n_embd FMA). Sparsify: skip near-zero diffs. */
    static float gh_kl[4096];
    for (int i = 0; i < n; i++) gh_kl[i] = 0.0f;
    for (int j = 0; j < vocab; j++) {
        float diff = ps[j] - pt[j];
        if (fabsf(diff) < 1e-6f) continue;  /* skip — negligible contribution */
        float coef = distill_T * diff;
        const float *w = &m->wte[(size_t)j * n];
        for (int i = 0; i + 7 < n; i += 8) {
            gh_kl[i+0] += coef * w[i+0];
            gh_kl[i+1] += coef * w[i+1];
            gh_kl[i+2] += coef * w[i+2];
            gh_kl[i+3] += coef * w[i+3];
            gh_kl[i+4] += coef * w[i+4];
            gh_kl[i+5] += coef * w[i+5];
            gh_kl[i+6] += coef * w[i+6];
            gh_kl[i+7] += coef * w[i+7];
        }
        for (int i = (n/8)*8; i < n; i++) gh_kl[i] += coef * w[i];
    }

    /* 5. Combine: total grad = alpha * CE + (1-alpha) * T^2 * KL. */
    static float gh[4096];
    float kl_scale = (1.0f - distill_alpha) * distill_T * distill_T;
    float ce_scale = distill_alpha;
    for (int i = 0; i < n; i++)
        gh[i] = ce_scale * gh_ce[i] + kl_scale * gh_kl[i];

    /* 6. Clip + backward through norm + layers (same as model_backward). */
    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += gh[i] * gh[i];
    gnorm = sqrtf(gnorm);
    if (gnorm > 0.1f) { float clip = 0.1f / gnorm; for (int i = 0; i < n; i++) gh[i] *= clip; }

    static float g_pre[4096];
    norm_backward(g_pre, gh, m->x_before_final, m->ln_f_w,
                  (float[]){m->final_mean, m->final_std_inv}, m->cfg.norm_type, n,
                  m->grad_ln_f_w_accum, m->grad_ln_f_b_accum);
    memcpy(gh, g_pre, n * sizeof(float));

    for (int l = m->cfg.n_layer - 1; l >= 0; l--)
        trans_layer_backward(gh, &m->layers[l], &m->acts[l], &m->cfg, lr);

    if (g_use_adam) g_opt_step++;

    free(s_logits); free(ps); free(pt);
}

void model_free(Model *m) {
    for (int l = 0; l < m->cfg.n_layer; l++)
        trans_layer_free(&m->layers[l], &m->cfg);
    free(m->layers);
    trans_act_free(m->acts, m->cfg.n_layer);
    free(m->final_ln);
    free(m->x_before_final);
    model_kv_cache_free(m);
    tensor_free_all(m->tensors, m->n_tensors);
}

/* ========================================================================
 * Binary Weight Layer
 * ======================================================================== */
void bin_layer_init(BinLayer *bl, const float *W, const float *bias,
                    int in_dim, int out_dim) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->n_words_T = (out_dim + 63) / 64;
    bl->wbits = calloc(out_dim * bl->n_words, sizeof(uint64_t));
    bl->wbits_T = calloc(in_dim * bl->n_words_T, sizeof(uint64_t));
    bl->zbits = NULL;  /* allocated only in ternary mode (logic-guided path) */
    bl->w_core = NULL; /* allocated only in logic-guided path; NULL → free() is a no-op */
    bl->logic_mask = NULL; /* set only in logic-guided path; NULL → free() is a no-op */
    bl->n_core = 0;
    bl->n_prune = 0;
    bl->alpha = calloc(out_dim, sizeof(float));
    bl->bias = bias ? malloc(out_dim * sizeof(float)) : calloc(out_dim, sizeof(float));
    bl->w_float = malloc((size_t)in_dim * out_dim * sizeof(float));  /* STE */
    bl->m_adam  = g_use_adam ? calloc((size_t)in_dim * out_dim, sizeof(float)) : NULL;   /* Adam m (conditional) */
    bl->v_adam  = g_use_adam ? calloc((size_t)in_dim * out_dim, sizeof(float)) : NULL;   /* Adam v (conditional) */
    bl->grad_accum = calloc((size_t)in_dim * out_dim, sizeof(float));  /* batch grad accumulation */
    bl->bias_grad_accum = calloc((size_t)out_dim, sizeof(float));  /* batch bias grad accumulation */
    bl->ternary_delta = 0.0f;  /* BWN by default; set by bin_layer_repack_ternary */

    /* Copy float weights for STE updates — TRANSPOSE to [out, in] layout!
     * W is [in, out] row-major (GPT-2 Conv1D format). We store w_float as
     * [out, in] so that w_float[j*in + i] is contiguous per output j.
     * This makes repack/alpha/update loops all contiguous → SIMD-friendly. */
    for (int j = 0; j < out_dim; j++)
        for (int i = 0; i < in_dim; i++)
            bl->w_float[j * in_dim + i] = W[i * out_dim + j];

    /* Row-major: pack sign(w[j][i]) per output j */
    for (int j = 0; j < out_dim; j++) {
        float abs_sum = 0;
        for (int i = 0; i < in_dim; i++) abs_sum += fabsf(W[i * out_dim + j]);
        bl->alpha[j] = abs_sum / in_dim;
        if (bias) bl->bias[j] = bias[j];
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in_dim && W[idx * out_dim + j] > 0.0f) word |= (1ULL << bi);
            }
            bl->wbits[j * bl->n_words + wi] = word;
        }
    }

    /* Col-major (transposed): pack sign(w[j][i]) per input i */
    for (int i = 0; i < in_dim; i++) {
        for (int wi = 0; wi < bl->n_words_T; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int j = wi * 64 + bi;
                if (j < out_dim && W[i * out_dim + j] > 0.0f) word |= (1ULL << bi);
            }
            bl->wbits_T[i * bl->n_words_T + wi] = word;
        }
    }
}

/* Logic-guided binarization: initialize with per-output logic_mask.
 * mask[j]: 0=CORE (keep float), 1=BINARY (sign+alpha), 2=PRUNE (zero).
 *
 * This implements PHONE's "logic extraction at binarization time":
 * - CORE outputs: weights stored as float in w_core, NOT binarized
 * - BINARY outputs: sign(w) packed into wbits, alpha = mean(|w|)
 * - PRUNE outputs: wbits all zero, alpha=0, bias=0 (effectively removed)
 *
 * The forward pass (bin_forward) checks logic_mask per output:
 * - CORE: y[j] = x @ w_core[j] (float matmul, no binarization)
 * - BINARY: y[j] = alpha * (2*popcount - N) + bias (XNOR+popcount)
 * - PRUNE: y[j] = 0 (skipped entirely)
 */
void bin_layer_init_logic(BinLayer *bl, const float *W, const float *bias,
                          int in_dim, int out_dim, const uint8_t *logic_mask) {
    bl->in_dim = in_dim;
    bl->out_dim = out_dim;
    bl->n_words = (in_dim + 63) / 64;
    bl->n_words_T = (out_dim + 63) / 64;
    bl->wbits = calloc(out_dim * bl->n_words, sizeof(uint64_t));
    bl->wbits_T = calloc(in_dim * bl->n_words_T, sizeof(uint64_t));
    bl->alpha = calloc(out_dim, sizeof(float));
    bl->bias = bias ? malloc(out_dim * sizeof(float)) : calloc(out_dim, sizeof(float));
    bl->w_float = malloc((size_t)out_dim * in_dim * sizeof(float));
    bl->m_adam  = NULL;  /* allocated below only if we keep this layer */
    bl->v_adam  = NULL;
    bl->w_core = NULL;
    bl->logic_mask = NULL;
    bl->n_core = 0;
    bl->n_prune = 0;
    bl->zbits = NULL;
    bl->ternary_delta = 0.0f;
    bl->grad_accum = NULL;        /* FIX: must NULL-init; model_batch_alloc checks !grad_accum */
    bl->bias_grad_accum = NULL;   /* FIX: same — otherwise random heap value passes the check */

    if (!logic_mask) {
        /* No logic mask → free w_float (bin_layer_init will re-alloc) and delegate. */
        free(bl->w_float); bl->w_float = NULL;
        bin_layer_init(bl, W, bias, in_dim, out_dim);
        return;
    }
    /* We're keeping this layer — allocate Adam state for the BINARY/CORE
     * STE-update path. PRUNE outputs contribute zero, but they still own a
     * slot in w_float (so the [out*in] indexing stays uniform). */
    bl->m_adam = calloc((size_t)out_dim * in_dim, sizeof(float));
    bl->v_adam = calloc((size_t)out_dim * in_dim, sizeof(float));

    /* Copy logic mask + count categories */
    bl->logic_mask = malloc(out_dim);
    memcpy(bl->logic_mask, logic_mask, out_dim);
    for (int j = 0; j < out_dim; j++) {
        if (logic_mask[j] == 0) bl->n_core++;
        else if (logic_mask[j] == 2) bl->n_prune++;
    }

    /* Allocate w_core for CORE outputs (float weights, [n_core, in_dim]) */
    if (bl->n_core > 0) {
        bl->w_core = malloc((size_t)bl->n_core * in_dim * sizeof(float));
    }

    /* Process each output based on its logic category */
    int core_idx = 0;
    for (int j = 0; j < out_dim; j++) {
        const float *wj = &W[j * in_dim];  /* W is [out, in] (transposed) */

        switch (logic_mask[j]) {
        case 0: /* CORE: keep float */
            memcpy(&bl->w_core[core_idx * in_dim], wj, in_dim * sizeof(float));
            bl->alpha[j] = 0.0f;  /* not used for CORE */
            if (bias) bl->bias[j] = bias[j];
            /* wbits for CORE: all zero (not used, but keep for indexing) */
            core_idx++;
            break;

        case 1: /* BINARY: sign(w) + alpha */
            {
                float abs_sum = 0;
                for (int i = 0; i < in_dim; i++) abs_sum += fabsf(wj[i]);
                bl->alpha[j] = abs_sum / in_dim;
                if (bias) bl->bias[j] = bias[j];
                for (int wi = 0; wi < bl->n_words; wi++) {
                    uint64_t word = 0;
                    for (int bi = 0; bi < 64; bi++) {
                        int idx = wi * 64 + bi;
                        if (idx < in_dim && wj[idx] > 0.0f) word |= (1ULL << bi);
                    }
                    bl->wbits[j * bl->n_words + wi] = word;
                }
            }
            break;

        case 2: /* PRUNE: zero out */
            bl->alpha[j] = 0.0f;
            bl->bias[j] = 0.0f;
            /* wbits already zero from calloc */
            break;
        }

        /* Copy to w_float (transposed [out, in] for STE compatibility) */
        memcpy(&bl->w_float[j * in_dim], wj, in_dim * sizeof(float));
    }

    /* Build wbits_T (transposed) only for BINARY outputs */
    for (int i = 0; i < in_dim; i++) {
        for (int wi = 0; wi < bl->n_words_T; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int j = wi * 64 + bi;
                if (j < out_dim && logic_mask[j] == 1 && W[j * in_dim + i] > 0.0f)
                    word |= (1ULL << bi);
            }
            bl->wbits_T[i * bl->n_words_T + wi] = word;
        }
    }

    /* Ternary mode: allocate zbits (zero mask, same shape as wbits) and
     * compute initial ternary binarization from w_float. When ternary is off,
     * zbits stays NULL — bin_forward uses the pure BWN ±1 path. */
    if (g_use_ternary) {
        bl->zbits = calloc((size_t)out_dim * bl->n_words, sizeof(uint64_t));
        bin_layer_repack_ternary(bl);
    }
}

void bin_layer_free(BinLayer *bl) {
    free(bl->wbits); free(bl->wbits_T); free(bl->zbits); free(bl->alpha); free(bl->bias);
    free(bl->w_float); free(bl->w_core); free(bl->logic_mask);
    free(bl->m_adam); free(bl->v_adam); free(bl->grad_accum); free(bl->bias_grad_accum);
    bl->wbits = NULL; bl->wbits_T = NULL; bl->zbits = NULL;
    bl->alpha = NULL; bl->bias = NULL;
    bl->w_float = NULL; bl->w_core = NULL; bl->logic_mask = NULL;
    bl->m_adam = NULL; bl->v_adam = NULL;
    bl->grad_accum = NULL; bl->bias_grad_accum = NULL;  /* FIX: was missing, causing wild-pointer crash in model_batch_begin after model_free + model_batch_alloc on phase switch */
}

/* Re-pack wbits and wbits_T from sign(w_float).
 * w_float is [out, in] (transposed from Conv1D's [in, out] for contiguous
 * per-output access). All loops here are now contiguous → auto-vectorizable.
 *
 * Key optimization: wbits[j] packs sign(w_float[j*in + 0..in-1]) which is
 * contiguous memory. The compiler auto-vectorizes the 8x unrolled comparison
 * into SIMD compare + movemask-style bit extraction. */
void bin_layer_repack(BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim;

    /* === CRITICAL FIX: Sync w_core from w_float for CORE neurons ===
     * CORE neurons use w_core (float) in forward pass, but model_batch_apply
     * updates w_float. Without this sync, CORE weights are FROZEN and
     * CORE/BINARY differentiation never improves. */
    if (bl->w_core && bl->logic_mask) {
        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            if (bl->logic_mask[j] == 0) {  /* CORE */
                memcpy(&bl->w_core[core_idx * in],
                       &bl->w_float[j * in],
                       in * sizeof(float));
                core_idx++;
            }
        }
    }

    /* Pack wbits[j][wi] from sign(w_float[j*in + i]) — CONTIGUOUS in i! */
    for (int j = 0; j < out; j++) {
        const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            int base = wi * 64;
            for (int grp = 0; grp < 8; grp++) {
                int idx = base + grp * 8;
                if (idx + 7 < in) {
                    /* 8 contiguous floats — compiler auto-vectorizes to SIMD */
                    if (wf[idx+0] > 0.0f) word |= (1ULL << (grp*8 + 0));
                    if (wf[idx+1] > 0.0f) word |= (1ULL << (grp*8 + 1));
                    if (wf[idx+2] > 0.0f) word |= (1ULL << (grp*8 + 2));
                    if (wf[idx+3] > 0.0f) word |= (1ULL << (grp*8 + 3));
                    if (wf[idx+4] > 0.0f) word |= (1ULL << (grp*8 + 4));
                    if (wf[idx+5] > 0.0f) word |= (1ULL << (grp*8 + 5));
                    if (wf[idx+6] > 0.0f) word |= (1ULL << (grp*8 + 6));
                    if (wf[idx+7] > 0.0f) word |= (1ULL << (grp*8 + 7));
                } else {
                    for (int bi = 0; bi < 8; bi++) {
                        int i = idx + bi;
                        if (i < in && wf[i] > 0.0f)
                            word |= (1ULL << (grp * 8 + bi));
                    }
                }
            }
            bl->wbits[j * bl->n_words + wi] = word;
        }
    }

    /* Skip wbits_T repack in STE mode — grad_x is now computed from w_float
     * directly (float arithmetic), so wbits_T is never read during STE training.
     * This saves the strided wbits_T repack loop (the slowest part). */
    for (int j = 0; j < out; j++) {
        const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
        float abs_sum = 0;
        for (int i = 0; i + 7 < in; i += 8) {
            abs_sum += fabsf(wf[i+0]) + fabsf(wf[i+1]) + fabsf(wf[i+2]) + fabsf(wf[i+3]);
            abs_sum += fabsf(wf[i+4]) + fabsf(wf[i+5]) + fabsf(wf[i+6]) + fabsf(wf[i+7]);
        }
        for (int i = (in / 8) * 8; i < in; i++)
            abs_sum += fabsf(wf[i]);
        bl->alpha[j] = abs_sum / in;
    }
}

/* Ternary repack: recompute zbits (zero mask) from |w_float| vs Δ.
 * For each BINARY output row j:
 *   Δ_j = g_ternary_delta_factor * mean(|w_float[j]|)
 *   zbits[j][i] = 1  if |w_float[j*in+i]| <= Δ_j  (weight zeroed → ternary 0)
 *   zbits[j][i] = 0  otherwise                      (weight active → ±1)
 * Also updates alpha[j] = mean(|w|) over ACTIVE weights only (standard TWN
 * scaling: the zeroed weights contribute nothing, so scaling reflects the
 * active subset). CORE/PRUNE rows are skipped (zbits stays 0 there).
 *
 * wbits (sign) is NOT recomputed here — bin_layer_repack (called separately
 * for STE) keeps sign in sync. This function only updates the zero mask.
 * Called at init and after every STE step when g_use_ternary is on. */
void bin_layer_repack_ternary(BinLayer *bl) {
    if (!bl->zbits || !bl->w_float) return;
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;

    for (int j = 0; j < out; j++) {
        /* Skip non-BINARY rows — they don't use ternary (CORE=float, PRUNE=0). */
        if (bl->logic_mask && bl->logic_mask[j] != 1) continue;

        const float *wf = &bl->w_float[j * in];
        /* Compute per-row Δ = factor * mean(|w|). */
        float abs_sum = 0.0f;
        for (int i = 0; i + 7 < in; i += 8) {
            abs_sum += fabsf(wf[i+0]) + fabsf(wf[i+1]) + fabsf(wf[i+2]) + fabsf(wf[i+3]);
            abs_sum += fabsf(wf[i+4]) + fabsf(wf[i+5]) + fabsf(wf[i+6]) + fabsf(wf[i+7]);
        }
        for (int i = (in/8)*8; i < in; i++) abs_sum += fabsf(wf[i]);
        float mean_abs = abs_sum / in;
        float delta = g_ternary_delta_factor * mean_abs;
        bl->ternary_delta = delta;  /* store per-layer (last row wins, used for stats) */

        /* Pack zbits[j]: 1 where |w| <= delta. Also sum active |w| for alpha. */
        uint64_t *zb = &bl->zbits[(size_t)j * nw];
        float active_abs_sum = 0.0f;
        int n_active = 0;
        for (int wi = 0; wi < nw; wi++) {
            uint64_t word = 0;
            int base = wi * 64;
            for (int grp = 0; grp < 8; grp++) {
                int idx = base + grp * 8;
                if (idx + 7 < in) {
                    for (int k = 0; k < 8; k++) {
                        int i = idx + k;
                        if (fabsf(wf[i]) <= delta) {
                            word |= (1ULL << (grp*8 + k));
                        } else {
                            active_abs_sum += fabsf(wf[i]);
                            n_active++;
                        }
                    }
                } else {
                    for (int k = 0; k < 8; k++) {
                        int i = idx + k;
                        if (i >= in) break;
                        if (fabsf(wf[i]) <= delta) {
                            word |= (1ULL << (grp*8 + k));
                        } else {
                            active_abs_sum += fabsf(wf[i]);
                            n_active++;
                        }
                    }
                }
            }
            zb[wi] = word;
        }
        /* TWN alpha: mean(|w|) over active weights. Falls back to mean over all
         * if everything got zeroed (degenerate row). */
        bl->alpha[j] = (n_active > 0) ? (active_abs_sum / n_active) : mean_abs;
    }
}

/* ========================================================================
 * Binary Forward — BWN (default, matches Python STE training)
 * ========================================================================
 * x stays float. Only W is binarized (sign(W) * alpha).
 * Adds XNOR-Net K-norm: K = ||x||_1 / in_dim, preserves input magnitude.
 *
 *   y[j] = (sum_i sign(W[j,i]) * x[i]) * alpha[j] * K + bias[j]
 *
 * This is the mathematically correct BWN forward. The old bin_forward was
 * BNN (binarized x too) which diverged from training and caused quality
 * collapse. BNN is retained as bin_forward_bnn() for opt-in fast mode.
 * ======================================================================== */
void bin_forward(float *y, const float *x, const BinLayer *bl) {
#ifdef LAL_CUDA
    if (g_use_cuda && !bl->logic_mask) {   /* GPU path: standard BWN (no logic mask) */
        lal_cuda_bin_forward(y, x, bl);
        return;
    }
#endif
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;

    /* Logic-guided: if logic_mask exists, dispatch per-output */
    if (bl->logic_mask) {
        /* K-norm for BINARY outputs */
        float abs_sum = 0.0f;
        for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
        float K = abs_sum / in;

        static float sign_lut[256][8];
        static int lut_init = 0;
        if (!lut_init) {
            for (int b = 0; b < 256; b++)
                for (int i = 0; i < 8; i++)
                    sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
            lut_init = 1;
        }

        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            switch (bl->logic_mask[j]) {
            case 0: { /* CORE: float matmul * core_gain * K
                 * Whitebox circuit trace: CORE was 22x weaker than BINARY.
                 * BINARY: sign(w)*alpha*K — sign() amplifies every weight to ±1.
                 * CORE:   w*K — raw float weights (~0.02), no amplification.
                 *
                 * Fix: core_gain = 1/alpha normalizes CORE's effective weight
                 * magnitude to ~1 (like BINARY's sign). Capped at 5 to prevent
                 * explosion when alpha is tiny. This makes CORE's signal O(1)
                 * like BINARY, so the circuit can actually use CORE's precision.
                 *
                 * alpha[j] = mean(|w_float[j]|), recalculated in bin_layer_repack.
                 * For CORE neurons, alpha is set in init then recalculated in repack. */
                const float *wc = &bl->w_core[core_idx * in];
                float s = 0.0f;
                for (int i = 0; i + 7 < in; i += 8) {
                    s += x[i+0]*wc[i+0] + x[i+1]*wc[i+1] + x[i+2]*wc[i+2] + x[i+3]*wc[i+3];
                    s += x[i+4]*wc[i+4] + x[i+5]*wc[i+5] + x[i+6]*wc[i+6] + x[i+7]*wc[i+7];
                }
                for (int i = (in/8)*8; i < in; i++) s += x[i] * wc[i];
                float core_gain = 1.0f / (bl->alpha[j] + 1e-8f);
                if (core_gain > 5.0f) core_gain = 5.0f;  /* moderate cap: 5x boost */
                y[j] = s * core_gain * K + bl->bias[j];
                core_idx++;
                break;
            }
            case 1: { /* BINARY: sign(w) * alpha * K + bias (ternary if zbits set) */
                const uint64_t *wb = &bl->wbits[j * nw];
                const uint64_t *zb = bl->zbits ? &bl->zbits[j * nw] : NULL;
                float s = 0.0f;
                for (int wi = 0; wi < nw; wi++) {
                    uint64_t w = wb[wi];
                    uint64_t z = zb ? zb[wi] : 0;  /* zero mask: 1=skip */
                    int base = wi * 64;
                    for (int bi = 0; bi < 8; bi++) {
                        int idx = base + bi * 8;
                        uint8_t byte = (uint8_t)((w >> (bi * 8)) & 0xFF);
                        uint8_t zbyte = (uint8_t)((z >> (bi * 8)) & 0xFF);
                        const float *sw = sign_lut[byte];
                        if (idx + 7 < in) {
                            /* Ternary: zeroed positions contribute 0.
                             * contribution = sign * (1 - zbit) * x.
                             * (1 - zbit) ∈ {0,1} acts as an enable mask. */
                            if (zbyte == 0) {
                                /* No zeros in this byte — full 8x dot product. */
                                s += x[idx+0]*sw[0] + x[idx+1]*sw[1] + x[idx+2]*sw[2] + x[idx+3]*sw[3];
                                s += x[idx+4]*sw[4] + x[idx+5]*sw[5] + x[idx+6]*sw[6] + x[idx+7]*sw[7];
                            } else {
                                /* Mixed: check each bit. zbyte bit set = skip. */
                                s += (zbyte & 0x01) ? 0 : x[idx+0]*sw[0];
                                s += (zbyte & 0x02) ? 0 : x[idx+1]*sw[1];
                                s += (zbyte & 0x04) ? 0 : x[idx+2]*sw[2];
                                s += (zbyte & 0x08) ? 0 : x[idx+3]*sw[3];
                                s += (zbyte & 0x10) ? 0 : x[idx+4]*sw[4];
                                s += (zbyte & 0x20) ? 0 : x[idx+5]*sw[5];
                                s += (zbyte & 0x40) ? 0 : x[idx+6]*sw[6];
                                s += (zbyte & 0x80) ? 0 : x[idx+7]*sw[7];
                            }
                        } else {
                            for (int k = 0; k < 8; k++) {
                                int i = idx + k;
                                if (i < in && !((zbyte >> k) & 1)) s += x[i] * sw[k];
                            }
                        }
                    }
                }
                y[j] = s * bl->alpha[j] * K + bl->bias[j];
                break;
            }
            default: /* PRUNE: zero */
                y[j] = 0.0f;
                break;
            }
        }
        return;
    }

    /* Standard BWN path (no logic_mask) */
    float abs_sum = 0.0f;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;

    static float sign_lut[256][8];
    static int lut_init = 0;
    if (!lut_init) {
        for (int b = 0; b < 256; b++)
            for (int i = 0; i < 8; i++)
                sign_lut[b][i] = (b >> i) & 1 ? 1.0f : -1.0f;
        lut_init = 1;
    }

    for (int j = 0; j < out; j++) {
        const uint64_t *wb = &bl->wbits[j * nw];
        float s = 0.0f;
        for (int wi = 0; wi < nw; wi++) {
            uint64_t w = wb[wi];
            int base = wi * 64;
            /* Process 8 bytes (8×8=64 bits) per word, 8 floats at a time */
            for (int bi = 0; bi < 8; bi++) {
                int idx = base + bi * 8;
                uint8_t byte = (uint8_t)((w >> (bi * 8)) & 0xFF);
                const float *sw = sign_lut[byte];
                if (idx + 7 < in) {
                    /* 8x unrolled dot product — auto-vectorizes to SIMD */
                    s += x[idx+0] * sw[0];
                    s += x[idx+1] * sw[1];
                    s += x[idx+2] * sw[2];
                    s += x[idx+3] * sw[3];
                    s += x[idx+4] * sw[4];
                    s += x[idx+5] * sw[5];
                    s += x[idx+6] * sw[6];
                    s += x[idx+7] * sw[7];
                } else {
                    /* Tail: handle remaining elements (< 8) */
                    for (int k = 0; k < 8; k++) {
                        int i = idx + k;
                        if (i < in) s += x[i] * sw[k];
                    }
                }
            }
        }
        y[j] = s * bl->alpha[j] * K + bl->bias[j];
    }
}

/* BNN fast path: XNOR + popcount, binarizes BOTH x and W.
 * ~64x faster than BWN. With K-norm input scaling (XNOR-Net, Rastegari 2016),
 * the output magnitude is restored: y = (2*pc-in) * alpha * K + bias, where
 * K = mean(|x|). Without K, outputs have wrong magnitude → garbled generation. */
void bin_forward_bnn(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;

    /* Compute input scale K = mean(|x|) — restores magnitude lost by sign(x).
     * O(in) cost is negligible vs the O(in*out) XNOR+popcount matmul. */
    float abs_sum = 0.0f;
    for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
    float K = abs_sum / in;

    /* Binarize input */
    uint64_t xbits[64];
    for (int wi = 0; wi < nw; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        xbits[wi] = word;
    }
    /* XNOR + popcount per output, scaled by alpha * K */
    for (int j = 0; j < out; j++) {
        int pc = 0;
        const uint64_t *wb = &bl->wbits[j * nw];
        for (int wi = 0; wi < nw; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        y[j] = (float)(2 * pc - in) * bl->alpha[j] * K + bl->bias[j];
    }
}

/* Legacy bin_forward_float: BWN without K-norm. Kept for callers that
 * explicitly don't want input magnitude scaling. */
void bin_forward_float(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    for (int j = 0; j < out; j++) {
        float s = bl->bias[j];
        const uint64_t *wb = &bl->wbits[j * nw];
        float a = bl->alpha[j];
        for (int wi = 0; wi < nw; wi++) {
            uint64_t w = wb[wi];
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx >= in) break;
                s += x[idx] * ((w >> bi) & 1 ? 1.0f : -1.0f) * a;
            }
        }
        y[j] = s;
    }
}

/* ========================================================================
 * Binary Backward: popcount for grad_x, popcount for alpha update
 * ======================================================================== */
void bin_backward(float *grad_x, const float *grad_y, const float *x,
                  BinLayer *bl, float lr) {
    int in = bl->in_dim, out = bl->out_dim;
    int nw_T = bl->n_words_T;

    /* Logic-guided: if logic_mask exists, dispatch per-output.
     * Without this, the non-logic path uses mean_alpha = sum(alpha)/out,
     * but PRUNE (alpha=0) and CORE (alpha=0) dilute mean_alpha → wrong
     * grad_x → NaN divergence. This was the root cause of ai_4116f587's
     * step-100 NaN in --logic + --real-attention testing.
     *   CORE: grad_x += grad_y * w_core (proper float gradient)
     *   BINARY: grad_x += grad_y * sign(wbits) * alpha (original logic)
     *   PRUNE: skip (zero gradient, output is zeroed in forward) */
    if (bl->logic_mask) {
        for (int i = 0; i < in; i++) grad_x[i] = 0.0f;
        /* K-norm for BINARY/CORE outputs (must match bin_forward's logic_mask path,
         * otherwise CORE/BINARY grad_x is off by a factor of K = mean(|x|)). */
        float abs_sum = 0.0f;
        for (int i = 0; i < in; i++) abs_sum += fabsf(x[i]);
        float K = abs_sum / in;
        int core_idx = 0;
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (bl->logic_mask[j] == 0) {
                /* CORE: float gradient through w_core * core_gain * K */
                if (fabsf(gy) >= 1e-8f) {
                    const float *wc = &bl->w_core[core_idx * in];
                    float core_gain = 1.0f / (bl->alpha[j] + 1e-8f);
                    if (core_gain > 5.0f) core_gain = 5.0f;
                    float scale = gy * core_gain * K;
                    for (int i = 0; i + 7 < in; i += 8) {
                        grad_x[i+0] += scale * wc[i+0];
                        grad_x[i+1] += scale * wc[i+1];
                        grad_x[i+2] += scale * wc[i+2];
                        grad_x[i+3] += scale * wc[i+3];
                        grad_x[i+4] += scale * wc[i+4];
                        grad_x[i+5] += scale * wc[i+5];
                        grad_x[i+6] += scale * wc[i+6];
                        grad_x[i+7] += scale * wc[i+7];
                    }
                    for (int i = (in/8)*8; i < in; i++) grad_x[i] += scale * wc[i];
                }
                core_idx++;
                bl->bias[j] -= lr * gy;
            } else if (bl->logic_mask[j] == 1) {
                /* BINARY: gradient through sign(wbits) * alpha */
                if (fabsf(gy) >= 1e-8f) {
                    const uint64_t *wb = &bl->wbits[j * bl->n_words];
                    float scale = gy * bl->alpha[j];
                    for (int wi = 0; wi < bl->n_words; wi++) {
                        uint64_t w = wb[wi];
                        int base = wi * 64;
                        for (int bi = 0; bi < 8; bi++) {
                            int idx = base + bi * 8;
                            if (idx + 7 < in) {
                                grad_x[idx+0] += scale * ((w >> (bi*8+0)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+1] += scale * ((w >> (bi*8+1)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+2] += scale * ((w >> (bi*8+2)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+3] += scale * ((w >> (bi*8+3)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+4] += scale * ((w >> (bi*8+4)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+5] += scale * ((w >> (bi*8+5)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+6] += scale * ((w >> (bi*8+6)) & 1 ? 1.0f : -1.0f);
                                grad_x[idx+7] += scale * ((w >> (bi*8+7)) & 1 ? 1.0f : -1.0f);
                            } else {
                                for (int k = 0; k < 8; k++) {
                                    int i = idx + k;
                                    if (i < in) grad_x[i] += scale * ((w >> (bi*8+k)) & 1 ? 1.0f : -1.0f);
                                }
                            }
                        }
                    }
                }
                bl->bias[j] -= lr * gy;
            }
            /* PRUNE (case 2): no gradient, skip entirely */
        }
        return;
    }

    /* Non-logic path: original bin_backward */

    /* Part 1: grad_x via XNOR+popcount using transposed weights */
    uint64_t gybits[64];
    for (int wi = 0; wi < nw_T; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int j = wi * 64 + bi;
            if (j < out && grad_y[j] > 0.0f) word |= (1ULL << bi);
        }
        gybits[wi] = word;
    }
    float mean_abs_gy = 0, mean_alpha = 0;
    for (int j = 0; j < out; j++) mean_abs_gy += fabsf(grad_y[j]);
    mean_abs_gy /= out;
    for (int j = 0; j < out; j++) mean_alpha += bl->alpha[j];
    mean_alpha /= out;
    for (int i = 0; i < in; i++) {
        int pc = 0;
        const uint64_t *wbT = &bl->wbits_T[i * nw_T];
        for (int wi = 0; wi < nw_T; wi++)
            pc += __builtin_popcountll(~(gybits[wi] ^ wbT[wi]));
        grad_x[i] = (float)(2 * pc - out) * mean_alpha * mean_abs_gy;
    }

    /* Part 2: alpha + bias update via popcount (reuse x_bits)
     *
     * [FIX 严重4] alpha update direction: was '+=', now '-'= to match bias.
     *   Old code: bl->alpha[j] += lr * grad_alpha * gy / in;   (WRONG: ascends loss)
     *   New code: bl->alpha[j] -= lr * grad_alpha * gy;         (correct: descends loss)
     * Also dropped spurious '/in' that shrank alpha's effective LR by in_dim.
     * [FIX 严重6] Removed alpha clamp to [0.001, 1.0] — it prevented alpha from
     *   converging to its natural magnitude and forced a fake floor. */
    float mean_abs_x = 0;
    for (int i = 0; i < in; i++) mean_abs_x += fabsf(x[i]);
    mean_abs_x /= in;
    uint64_t xbits[64];
    for (int wi = 0; wi < bl->n_words; wi++) {
        uint64_t word = 0;
        for (int bi = 0; bi < 64; bi++) {
            int idx = wi * 64 + bi;
            if (idx < in && x[idx] > 0.0f) word |= (1ULL << bi);
        }
        xbits[wi] = word;
    }
    for (int j = 0; j < out; j++) {
        float gy = grad_y[j];
        if (fabsf(gy) < 1e-6f) continue;
        int pc = 0;
        const uint64_t *wb = &bl->wbits[j * bl->n_words];
        for (int wi = 0; wi < bl->n_words; wi++)
            pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
        float grad_alpha = (float)(2 * pc - in) * mean_abs_x;
        bl->alpha[j] -= lr * grad_alpha * gy;   /* FIXED: direction + no /in */
        /* Removed: if (bl->alpha[j] < 0.001f) bl->alpha[j] = 0.001f;
         *         if (bl->alpha[j] > 1.0f)   bl->alpha[j] = 1.0f; */
        if (bl->alpha[j] < 0.0f) bl->alpha[j] = 0.0f;  /* only non-negativity */
        bl->bias[j] -= lr * gy;
    }
}

/* STE (Straight-Through Estimator) backward pass.
 *
 * Key difference from bin_backward: this updates w_float (the full-precision
 * weights) using the gradient, treating sign() as identity. After the update,
 * wbits is re-packed from sign(w_float) via bin_layer_repack().
 *
 * This allows the binary weights to actually change during training, which
 * is impossible with bin_backward (it only updates alpha and bias).
 *
 * STE gradient: d(loss)/d(w_float) = d(loss)/d(sign(w)) * d(sign(w))/d(w)
 *                                  = grad_y * x * 1  (STE: sign'(w) = 1)
 * So: w_float[i,j] -= lr * grad_y[j] * x[i]
 *
 * Memory note: w_float is [in_dim, out_dim] row-major, same as input W.
 * This adds ~in*out*4 bytes per layer (e.g. 768*2304*4 = 7MB for c_attn).
 * Total for 12 layers × 4 matrices ≈ 339 MB extra during training.
 * For inference, w_float can be freed (set to NULL after training). */
void bin_backward_ste(float *grad_x, const float *grad_y, const float *x,
                      BinLayer *bl, float lr) {
#ifdef LAL_CUDA
    if (g_use_cuda && !bl->logic_mask) {   /* GPU path: standard BWN (no logic mask) */
        lal_cuda_bin_backward_ste(grad_x, grad_y, x, bl, lr);
        if (g_use_adam) g_opt_step++;      /* mirror CPU path */
        return;
    }
#endif
    int in = bl->in_dim, out = bl->out_dim;

    /* Part 1: grad_x computation.
     * In STE mode, skip wbits_T repack entirely — compute grad_x directly
     * from w_float using float arithmetic. This avoids the strided wbits_T
     * repack (50% of repack cost) at the expense of float mul-adds.
     *
     * grad_x[i] = sum_j grad_y[j] * sign(w_float[j*in+i]) * alpha[j]
     *
     * w_float is [out, in], so w_float[j*in+i] has i contiguous per j.
     * But we need i fixed, j varying — that's strided. So we compute
     * per-i by accumulating over j. With [out,in] layout, w_float[j*in+i]
     * for fixed i has stride=in. This is still strided but avoids repack.
     *
     * Alternative: compute grad_x = sign(w_float)^T @ (grad_y * alpha)
     * which is a matrix-vector product. We can do it per-output j and
     * accumulate into grad_x (since w_float[j*in+i] is contiguous in i). */
    if (bl->w_float) {
        /* Zero grad_x first */
        for (int i = 0; i < in; i++) grad_x[i] = 0.0f;
        /* For each output j: grad_x += grad_y[j] * alpha[j] * sign(w_float[j*in+i])
         * w_float[j*in + 0..in-1] is contiguous → SIMD-friendly! */
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-8f) continue;
            /* Skip PRUNE in grad_x: PRUNE outputs 0 in forward, so it must
             * NOT contribute gradient to the input. Without this skip,
             * sign(w_float) of dead neurons leaks gradient upstream,
             * causing PRUNE activations to grow instead of staying silent. */
            if (bl->logic_mask && bl->logic_mask[j] == 2) continue;
            float scale = g_use_pure_float ? gy : (gy * bl->alpha[j]);
            const float *wf = &bl->w_float[j * in];  /* contiguous [in] */
            if (g_use_pure_float) {
                /* Pure float: grad_x uses w_float directly (not sign) */
                for (int i = 0; i + 7 < in; i += 8) {
                    grad_x[i+0] += scale * wf[i+0];
                    grad_x[i+1] += scale * wf[i+1];
                    grad_x[i+2] += scale * wf[i+2];
                    grad_x[i+3] += scale * wf[i+3];
                    grad_x[i+4] += scale * wf[i+4];
                    grad_x[i+5] += scale * wf[i+5];
                    grad_x[i+6] += scale * wf[i+6];
                    grad_x[i+7] += scale * wf[i+7];
                }
                for (int i = (in / 8) * 8; i < in; i++)
                    grad_x[i] += scale * wf[i];
            } else {
                /* BWN: grad_x uses sign(w_float) * alpha */
                for (int i = 0; i + 7 < in; i += 8) {
                    grad_x[i+0] += scale * (wf[i+0] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+1] += scale * (wf[i+1] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+2] += scale * (wf[i+2] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+3] += scale * (wf[i+3] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+4] += scale * (wf[i+4] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+5] += scale * (wf[i+5] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+6] += scale * (wf[i+6] > 0.0f ? 1.0f : -1.0f);
                    grad_x[i+7] += scale * (wf[i+7] > 0.0f ? 1.0f : -1.0f);
                }
                for (int i = (in / 8) * 8; i < in; i++)
                    grad_x[i] += scale * (wf[i] > 0.0f ? 1.0f : -1.0f);
            }
        }
    } else {
        /* No w_float — use popcount on existing wbits_T (original path) */
        int nw_T = bl->n_words_T;
        uint64_t gybits[64];
        for (int wi = 0; wi < nw_T; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int j = wi * 64 + bi;
                if (j < out && grad_y[j] > 0.0f) word |= (1ULL << bi);
            }
            gybits[wi] = word;
        }
        float mean_abs_gy = 0, mean_alpha = 0;
        for (int j = 0; j < out; j++) mean_abs_gy += fabsf(grad_y[j]);
        mean_abs_gy /= out;
        for (int j = 0; j < out; j++) mean_alpha += bl->alpha[j];
        mean_alpha /= out;
        for (int i = 0; i < in; i++) {
            int pc = 0;
            const uint64_t *wbT = &bl->wbits_T[i * nw_T];
            for (int wi = 0; wi < nw_T; wi++)
                pc += __builtin_popcountll(~(gybits[wi] ^ wbT[wi]));
            grad_x[i] = (float)(2 * pc - out) * mean_alpha * mean_abs_gy;
        }
    }

    /* Part 2: Gradient accumulation or STE update.
     * When g_accumulate_gradients is set (batch training), we accumulate
     * grad_y[j]*x[i] into grad_accum and grad_y[j] into bias_grad_accum
     * instead of updating weights. model_batch_apply() later applies the
     * accumulated (averaged) gradient with Adam. */
    if (g_accumulate_gradients && bl->grad_accum) {
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-8f) continue;
            if (bl->logic_mask && bl->logic_mask[j] == 2) continue;
            float *ga = &bl->grad_accum[j * in];
            for (int i = 0; i + 7 < in; i += 8) {
                ga[i+0] += gy * x[i+0]; ga[i+1] += gy * x[i+1];
                ga[i+2] += gy * x[i+2]; ga[i+3] += gy * x[i+3];
                ga[i+4] += gy * x[i+4]; ga[i+5] += gy * x[i+5];
                ga[i+6] += gy * x[i+6]; ga[i+7] += gy * x[i+7];
            }
            for (int i = (in / 8) * 8; i < in; i++)
                ga[i] += gy * x[i];
            bl->bias_grad_accum[j] += gy;
        }
        return;  /* Don't update weights yet — wait for model_batch_apply */
    }

    /* Part 2: STE update — w_float[j*in + i] -= lr * grad_y[j] * x[i]
     * w_float is [out, in] (transposed), so w_float[j*in + i] is CONTIGUOUS in i!
     * This means the inner loop over i is a contiguous SAXPY: w_float[j] -= scale * x
     * The compiler auto-vectorizes this to SIMD FMA (8 floats per iteration).
     *
     * When g_use_adam is set, we use the Adam update instead of plain SGD:
     *   m[i] = b1*m[i] + (1-b1)*g       (1st moment)
     *   v[i] = b2*v[i] + (1-b2)*g*g     (2nd moment)
     *   w  -= lr * m_hat / (sqrt(v_hat) + eps)
     * where m_hat, v_hat are bias-corrected with g_opt_step+1. Adam dramatically
     * stabilizes STE on bit-space: per-param adaptive lr counters the
     * extreme gradient variance that causes SGD to mode-collapse to "servers". */
    if (bl->w_float) {
        int t = g_opt_step + 1;  /* Adam timestep (1-indexed) */
        float bc1 = 1.0f - powf(g_adam_beta1, (float)t);  /* bias correction 1 */
        float bc2 = 1.0f - powf(g_adam_beta2, (float)t);  /* bias correction 2 */
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-8f) continue;
            /* Skip PRUNE rows — they're zeroed and contribute nothing. */
            if (bl->logic_mask && bl->logic_mask[j] == 2) continue;
            float *wf = &bl->w_float[j * in];  /* contiguous [in] */
            if (g_use_adam && bl->m_adam) {
                float *m = &bl->m_adam[j * in];
                float *v = &bl->v_adam[j * in];
                /* Adam: per-param adaptive update. 8x unrolled for SIMD. */
                for (int i = 0; i + 7 < in; i += 8) {
                    float g0 = gy * x[i+0], g1 = gy * x[i+1], g2 = gy * x[i+2], g3 = gy * x[i+3];
                    float g4 = gy * x[i+4], g5 = gy * x[i+5], g6 = gy * x[i+6], g7 = gy * x[i+7];
                    m[i+0] = g_adam_beta1*m[i+0] + (1.0f-g_adam_beta1)*g0;
                    m[i+1] = g_adam_beta1*m[i+1] + (1.0f-g_adam_beta1)*g1;
                    m[i+2] = g_adam_beta1*m[i+2] + (1.0f-g_adam_beta1)*g2;
                    m[i+3] = g_adam_beta1*m[i+3] + (1.0f-g_adam_beta1)*g3;
                    m[i+4] = g_adam_beta1*m[i+4] + (1.0f-g_adam_beta1)*g4;
                    m[i+5] = g_adam_beta1*m[i+5] + (1.0f-g_adam_beta1)*g5;
                    m[i+6] = g_adam_beta1*m[i+6] + (1.0f-g_adam_beta1)*g6;
                    m[i+7] = g_adam_beta1*m[i+7] + (1.0f-g_adam_beta1)*g7;
                    v[i+0] = g_adam_beta2*v[i+0] + (1.0f-g_adam_beta2)*g0*g0;
                    v[i+1] = g_adam_beta2*v[i+1] + (1.0f-g_adam_beta2)*g1*g1;
                    v[i+2] = g_adam_beta2*v[i+2] + (1.0f-g_adam_beta2)*g2*g2;
                    v[i+3] = g_adam_beta2*v[i+3] + (1.0f-g_adam_beta2)*g3*g3;
                    v[i+4] = g_adam_beta2*v[i+4] + (1.0f-g_adam_beta2)*g4*g4;
                    v[i+5] = g_adam_beta2*v[i+5] + (1.0f-g_adam_beta2)*g5*g5;
                    v[i+6] = g_adam_beta2*v[i+6] + (1.0f-g_adam_beta2)*g6*g6;
                    v[i+7] = g_adam_beta2*v[i+7] + (1.0f-g_adam_beta2)*g7*g7;
                    float mh0=m[i+0]/bc1, mh1=m[i+1]/bc1, mh2=m[i+2]/bc1, mh3=m[i+3]/bc1;
                    float mh4=m[i+4]/bc1, mh5=m[i+5]/bc1, mh6=m[i+6]/bc1, mh7=m[i+7]/bc1;
                    float vh0=sqrtf(v[i+0]/bc2)+g_adam_eps, vh1=sqrtf(v[i+1]/bc2)+g_adam_eps;
                    float vh2=sqrtf(v[i+2]/bc2)+g_adam_eps, vh3=sqrtf(v[i+3]/bc2)+g_adam_eps;
                    float vh4=sqrtf(v[i+4]/bc2)+g_adam_eps, vh5=sqrtf(v[i+5]/bc2)+g_adam_eps;
                    float vh6=sqrtf(v[i+6]/bc2)+g_adam_eps, vh7=sqrtf(v[i+7]/bc2)+g_adam_eps;
                    wf[i+0] -= lr * mh0/vh0; wf[i+1] -= lr * mh1/vh1;
                    wf[i+2] -= lr * mh2/vh2; wf[i+3] -= lr * mh3/vh3;
                    wf[i+4] -= lr * mh4/vh4; wf[i+5] -= lr * mh5/vh5;
                    wf[i+6] -= lr * mh6/vh6; wf[i+7] -= lr * mh7/vh7;
                }
                for (int i = (in / 8) * 8; i < in; i++) {
                    float g = gy * x[i];
                    m[i] = g_adam_beta1*m[i] + (1.0f-g_adam_beta1)*g;
                    v[i] = g_adam_beta2*v[i] + (1.0f-g_adam_beta2)*g*g;
                    wf[i] -= lr * (m[i]/bc1) / (sqrtf(v[i]/bc2) + g_adam_eps);
                }
            } else {
                /* SGD: wf[i] -= lr * gy * x[i] (original path). */
                float scale = lr * gy;
                for (int i = 0; i + 7 < in; i += 8) {
                    wf[i+0] -= scale * x[i+0]; wf[i+1] -= scale * x[i+1];
                    wf[i+2] -= scale * x[i+2]; wf[i+3] -= scale * x[i+3];
                    wf[i+4] -= scale * x[i+4]; wf[i+5] -= scale * x[i+5];
                    wf[i+6] -= scale * x[i+6]; wf[i+7] -= scale * x[i+7];
                }
                for (int i = (in / 8) * 8; i < in; i++)
                    wf[i] -= scale * x[i];
            }
            /* Update bias (SGD always — bias is a scalar, Adam benefit marginal). */
            bl->bias[j] -= lr * gy;
        }
        /* Weight clipping: bound w_float to [-W_CLIP, W_CLIP].
         * Standard technique for BWN training (cf. XNOR-Net, Real-to-Binary
         * Networks). Prevents w_float from drifting to extreme values where
         * Adam's adaptive update becomes numerically unstable and sign(w)
         * starts flipping chaotically. The bound [-1, 1] is natural because
         * alpha = mean(|w|) is in [0, 1] for typical GPT-2 weight rows.
         * Without this, STE+Adam diverges to NaN around step 400-850. */
        if (!g_use_pure_float) {
            #define W_CLIP 1.0f
            for (int i = 0; i < in * out; i++) {
                float w = bl->w_float[i];
                if (w > W_CLIP) bl->w_float[i] = W_CLIP;
                else if (w < -W_CLIP) bl->w_float[i] = -W_CLIP;
            }
            /* Re-pack binary weights from updated (and clipped) w_float */
            bin_layer_repack(bl);
        } else {
            /* Pure float: clip to larger bound to prevent gradient explosion */
            #define W_CLIP_FLOAT 2.0f
            for (int i = 0; i < in * out; i++) {
                float w = bl->w_float[i];
                if (w > W_CLIP_FLOAT) bl->w_float[i] = W_CLIP_FLOAT;
                else if (w < -W_CLIP_FLOAT) bl->w_float[i] = -W_CLIP_FLOAT;
            }
        }
        /* Ternary: refresh zbits (zero mask) from updated |w_float| vs Δ.
         * This is the TWN STE dynamic — zeroed weights that received enough
         * gradient to cross Δ "wake up" (become ±1), and active weights whose
         * |w| dropped below Δ get zeroed. alpha is also recomputed over the
         * new active set. Skipped automatically if zbits is NULL (BWN mode). */
        if (bl->zbits && !g_use_pure_float) bin_layer_repack_ternary(bl);
    } else {
        /* No w_float — fall back to alpha-only update */
        float mean_abs_x = 0;
        for (int i = 0; i < in; i++) mean_abs_x += fabsf(x[i]);
        mean_abs_x /= in;
        uint64_t xbits[64];
        for (int wi = 0; wi < bl->n_words; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in && x[idx] > 0.0f) word |= (1ULL << bi);
            }
            xbits[wi] = word;
        }
        for (int j = 0; j < out; j++) {
            float gy = grad_y[j];
            if (fabsf(gy) < 1e-6f) continue;
            int pc = 0;
            const uint64_t *wb = &bl->wbits[j * bl->n_words];
            for (int wi = 0; wi < bl->n_words; wi++)
                pc += __builtin_popcountll(~(xbits[wi] ^ wb[wi]));
            float grad_alpha = (float)(2 * pc - in) * mean_abs_x;
            bl->alpha[j] -= lr * grad_alpha * gy;   /* FIXED: direction + no /in */
            if (bl->alpha[j] < 0.0f) bl->alpha[j] = 0.0f;
            bl->bias[j] -= lr * gy;
        }
    }
}

/* ========================================================================
 * Standard Neural Network Operations
 * ======================================================================== */
void layer_norm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float is = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * is * w[i] + b[i];
}

void layer_norm_backward(float *grad_x, const float *grad_y, const float *x,
                         const float *w, float mean, float std_inv, int n,
                         float *grad_w, float *grad_b) {
    float sum_grad = 0;
    for (int i = 0; i < n; i++) sum_grad += grad_y[i] * w[i] * (x[i] - mean);
    float common = std_inv / n * sum_grad;
    float scale = (1.0f - 1.0f / n);
    for (int i = 0; i < n; i++) {
        grad_x[i] = grad_y[i] * w[i] * std_inv * scale - common;
        if (grad_w) grad_w[i] += grad_y[i] * (x[i] - mean) * std_inv;
        if (grad_b) grad_b[i] += grad_y[i];
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

float gelu_grad(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.7978845608f * (1.0f + 0.134145f * x * x);
}

void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

float cross_entropy_sampled(const float *hidden, const float *wte,
                            int target, int vocab_size, int n_embd,
                            int n_samples, unsigned int *seed) {
    float tl = 0;
    for (int i = 0; i < n_embd; i++) tl += hidden[i] * wte[target * n_embd + i];
    float mx = tl;
    float neg[256];
    for (int k = 0; k < n_samples && k < 256; k++) {
        int v = rand_r(seed) % vocab_size;
        float s = 0;
        for (int i = 0; i < n_embd; i++) s += hidden[i] * wte[v * n_embd + i];
        neg[k] = s;
        if (s > mx) mx = s;
    }
    float se = expf(tl - mx);
    for (int k = 0; k < n_samples && k < 256; k++) se += expf(neg[k] - mx);
    return -logf(expf(tl - mx) / se + 1e-7f);
}

void cross_entropy_grad(float *grad_hidden, const float *hidden, const float *wte,
                        int target, int vocab_size, int n_embd,
                        int n_samples, unsigned int *seed) {
    /* Sampled-softmax cross-entropy gradient.
     *
     * Loss:  L = -log( exp(tl) / (exp(tl) + sum_k exp(neg_k)) )
     *            = -log( prob ),   prob = exp(tl-mx) / (exp(tl-mx) + sum_k exp(neg_k-mx))
     *
     * Gradient w.r.t. hidden[i] (treating sampled negatives as constants —
     * standard sampled-softmax approximation, drops the second-order term
     * sum_k prob_k * wte[k, i]):
     *
     *   dL/d(hidden[i]) = dL/d(tl) * d(tl)/d(hidden[i])
     *                   = -(1 - prob) * wte[target, i]
     *
     * ---------------------------------------------------------------------
     * BUGFIX (gibberish-output root cause):
     *
     * The previous implementation returned
     *
     *     grad_hidden[i] = +(1 - prob) * wte[target, i] * 0.001f
     *
     * which had THREE bugs that together made binary training diverge into
     * mode-collapse / gibberish:
     *
     *   (1) WRONG SIGN. Returned +grad instead of -grad. Combined with the
     *       optimizer's `w -= lr * grad`, this flipped descent into ascent
     *       on -log(p_target): the model was trained to *lower* p_target,
     *       i.e. to actively avoid predicting the correct token. After a few
     *       hundred steps the logits collapse and generation produces
     *       constant-token gibberish.
     *
     *   (2) grad_scale = 0.001f shrank the learning signal by 1000x. Even
     *       after fixing the sign, with lr=0.05 the effective step on
     *       `hidden` was 5e-5 — far too small to escape random init in any
     *       reasonable number of steps. Removed.
     *
     *   (3) `se += 1.0f` per sampled negative (instead of the true
     *       exp(neg_k - mx)) inflated the denominator systematically,
     *       forcing prob -> 0 and (1-prob) -> 1, which (combined with the
     *       wrong sign) made every step push hidden AWAY from wte[target]
     *       at maximum magnitude. Now we use the actual exp(neg_k - mx).
     * ---------------------------------------------------------------------
     */
    float tl = 0;
    for (int i = 0; i < n_embd; i++) tl += hidden[i] * wte[target * n_embd + i];

    /* Sample negatives and remember their logits so we can build the
     * correct softmax denominator. Cap at 256 to keep the stack buffer
     * bounded (n_samples=100 in practice). */
    float neg[256];
    int actual = n_samples < 256 ? n_samples : 256;
    float mx = tl;
    for (int k = 0; k < actual; k++) {
        int v = rand_r(seed) % vocab_size;
        float s = 0;
        for (int i = 0; i < n_embd; i++) s += hidden[i] * wte[v * n_embd + i];
        neg[k] = s;
        if (s > mx) mx = s;
    }

    float se = expf(tl - mx);
    for (int k = 0; k < actual; k++) se += expf(neg[k] - mx);

    /* prob = P(target) under the sampled softmax. +1e-7f guards against
     * logf(0) in the caller (cross_entropy_sampled) and div-by-zero here. */
    float prob = expf(tl - mx) / (se + 1e-7f);

    /* Correct gradient of L = -log(prob) w.r.t. hidden[i].
     * Optimizer does `w -= lr * grad`, so a NEGATIVE grad here means
     * hidden moves TOWARD wte[target], which INCREASES prob and
     * DECREASES loss — i.e. true gradient descent. */
    float coef = -(1.0f - prob);
    const float *wt = &wte[target * n_embd];
    for (int i = 0; i < n_embd; i++)
        grad_hidden[i] = coef * wt[i];
}

void clip_array(float *x, int n, float clip_val) {
    for (int i = 0; i < n; i++) {
        if (x[i] > clip_val) x[i] = clip_val;
        if (x[i] < -clip_val) x[i] = -clip_val;
    }
}

/* BUG #48 FIX: Normalize residual stream ||x|| to target_norm.
 * The residual stream accumulates: x = wte + sum(attn_residual + mlp_residual).
 * With residual_scale=1.0 and 8+ layers, ||x|| grows from ~1 (L0) to ~210 (L7).
 * This causes logits = dot(final_ln, wte) to explode, making sampling degenerate.
 *
 * LayerNorm normalizes the INPUT to each sublayer, but NOT the residual x itself.
 * So x grows unboundedly between layers.
 *
 * Fix: after each residual addition, scale x so ||x|| = target_norm (~3.0).
 * This is similar to "RMSNorm on residual stream" used in some architectures.
 * target_norm=3.0 matches the natural ||x|| after embedding + position encoding
 * in early layers (before the residual starts accumulating). */
void normalize_residual(float *x, int n, float target_norm) {
    float norm_sq = 0;
    for (int i = 0; i < n; i++) norm_sq += x[i] * x[i];
    float norm = sqrtf(norm_sq) + 1e-8f;
    if (norm > target_norm) {
        float scale = target_norm / norm;
        for (int i = 0; i < n; i++) x[i] *= scale;
    }
}

/* ========================================================================
 * Full-vocab softmax cross-entropy (replaces sampled softmax for training)
 *
 * The sampled-softmax path (cross_entropy_sampled / cross_entropy_grad)
 * uses 100 random negatives per step. When the training data is heavily
 * skewed (91% of sentences end in token 764='.'), the model can trivially
 * win against 100 random negatives by always outputting 764 — collapsing
 * to a single-token predictor. The full-softmax path computes the true
 * gradient over all 50257 vocab tokens, so the model is forced to actually
 * learn the distribution (token 764 gets probability mass only when the
 * context genuinely predicts it).
 *
 * Cost: 50257 * 768 = ~38M FMA per forward, ~76M per backward. Negligible
 * vs the per-layer binary matmul (12 layers * ~3M FMA = 36M).
 * ======================================================================== */

static void compute_full_logits(const float *hidden, const float *wte,
                                float *logits_out, int vocab, int n_embd) {
    for (int j = 0; j < vocab; j++) {
        const float *w = &wte[(size_t)j * n_embd];
        float s = 0;
        for (int i = 0; i + 7 < n_embd; i += 8)
            s += hidden[i+0]*w[i+0] + hidden[i+1]*w[i+1]
               + hidden[i+2]*w[i+2] + hidden[i+3]*w[i+3]
               + hidden[i+4]*w[i+4] + hidden[i+5]*w[i+5]
               + hidden[i+6]*w[i+6] + hidden[i+7]*w[i+7];
        for (int i = (n_embd/8)*8; i < n_embd; i++) s += hidden[i] * w[i];
        logits_out[j] = s;
    }
}

float cross_entropy_full(const float *hidden, const float *wte,
                         int target, int vocab_size, int n_embd,
                         float *logits_scratch) {
    compute_full_logits(hidden, wte, logits_scratch, vocab_size, n_embd);
    /* numerically stable softmax + cross-entropy */
    float mx = logits_scratch[0];
    for (int j = 1; j < vocab_size; j++)
        if (logits_scratch[j] > mx) mx = logits_scratch[j];
    float sum = 0;
    for (int j = 0; j < vocab_size; j++) {
        logits_scratch[j] = expf(logits_scratch[j] - mx);
        sum += logits_scratch[j];
    }
    /* logits_scratch now holds softmax probabilities; loss = -log(p_target) */
    float p_target = logits_scratch[target] / sum;
    return -logf(p_target + 1e-12f);
}

void cross_entropy_full_grad(float *grad_hidden, const float *hidden, const float *wte,
                             int target, int vocab_size, int n_embd,
                             float *logits_scratch) {
    /* grad_hidden[i] = (softmax(logits)[target_or_not] - one_hot[target]) * wte[i]
     *                 = (p[j] - 1{j==target}) * wte[j, i]   summed over j.
     *
     * Equivalent to: grad_hidden = wte[target] - sum_j p[j] * wte[j]
     * But computing it as  wte[target] - sum_j p[j]*wte[j]  is O(vocab*n_embd)
     * and avoids materializing a per-(j,i) gradient.
     *
     * logits_scratch must already hold the softmax probabilities from
     * cross_entropy_full (caller reuses it to avoid recomputing logits). */
    /* Start with wte[target] (the +1 in dL/d_logit = p - one_hot, multiplied
     * by -1 because we want dL/d_hidden, and the chain rule gives a negative
     * sign through the loss). Actually:
     *   L = -log(p_target),  p = softmax(logits),  logits[j] = hidden . wte[j]
     *   dL/d_logits[j] = p[j] - 1{j==target}
     *   dL/d_hidden[i] = sum_j (p[j] - 1{j==target}) * wte[j, i]
     *                  = sum_j p[j]*wte[j,i] - wte[target, i]
     * The optimizer does w -= lr * grad, so we return dL/d_hidden directly.
     * (Previously the sign bug was here; now correct.) */
    const float *wt_target = &wte[(size_t)target * n_embd];
    for (int i = 0; i < n_embd; i++)
        grad_hidden[i] = -wt_target[i];

    /* Add sum_j p[j] * wte[j, i]. p[j] is in logits_scratch (already
     * normalized to sum=1 by cross_entropy_full, but we re-normalize
     * defensively in case the caller passed un-normalized logits). */
    float psum = 0;
    for (int j = 0; j < vocab_size; j++) psum += logits_scratch[j];
    float inv_psum = 1.0f / (psum + 1e-12f);

    for (int j = 0; j < vocab_size; j++) {
        float p = logits_scratch[j] * inv_psum;
        if (p < 1e-7f) continue;  /* skip — negligible contribution */
        const float *w = &wte[(size_t)j * n_embd];
        float coef = p;
        for (int i = 0; i + 7 < n_embd; i += 8) {
            grad_hidden[i+0] += coef * w[i+0];
            grad_hidden[i+1] += coef * w[i+1];
            grad_hidden[i+2] += coef * w[i+2];
            grad_hidden[i+3] += coef * w[i+3];
            grad_hidden[i+4] += coef * w[i+4];
            grad_hidden[i+5] += coef * w[i+5];
            grad_hidden[i+6] += coef * w[i+6];
            grad_hidden[i+7] += coef * w[i+7];
        }
        for (int i = (n_embd/8)*8; i < n_embd; i++)
            grad_hidden[i] += coef * w[i];
    }
}

void compute_mean_std(const float *x, int n, float *mean, float *std_inv) {
    float m = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - m; var += d * d; }
    var /= n;
    *mean = m;
    *std_inv = 1.0f / sqrtf(var + 1e-5f);
}

/* ========================================================================
 * Tensor File Loading (GPW2 format)
 * ======================================================================== */
Tensor *tensor_load_all(const char *path, int *n_tensors) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GPW2", 4) != 0) { fprintf(stderr, "bad magic\n"); fclose(f); return NULL; }
    fread(n_tensors, 4, 1, f);
    Tensor *t = calloc(*n_tensors, sizeof(Tensor));
    for (int i = 0; i < *n_tensors; i++) {
        int klen;
        fread(&klen, 4, 1, f);
        fread(t[i].key, 1, klen, f);
        t[i].key[klen] = '\0';
        fread(&t[i].ndim, 4, 1, f);
        int n = 1;
        for (int d = 0; d < t[i].ndim; d++) {
            fread(&t[i].shape[d], 4, 1, f);
            n *= t[i].shape[d];
        }
        t[i].data = malloc(n * sizeof(float));
        fread(t[i].data, 4, n, f);
    }
    fclose(f);
    return t;
}

float *tensor_get(Tensor *tensors, int n, const char *key) {
    for (int i = 0; i < n; i++)
        if (strcmp(tensors[i].key, key) == 0) return tensors[i].data;
    fprintf(stderr, "tensor not found: %s\n", key);
    return NULL;
}

void tensor_free_all(Tensor *tensors, int n) {
    for (int i = 0; i < n; i++) free(tensors[i].data);
    free(tensors);
}

/* Free a single tensor's data by key (sets data to NULL so tensor_free_all
 * won't double-free). Used to reclaim memory from large weight matrices
 * after they've been binarized into BinLayer. */
void tensor_free_data_by_key(Tensor *tensors, int n, const char *key) {
    for (int i = 0; i < n; i++) {
        if (tensors[i].data && strcmp(tensors[i].key, key) == 0) {
            free(tensors[i].data);
            tensors[i].data = NULL;
            return;
        }
    }
}

/* mmap-based tensor loader: maps the GPW2 file into memory and points
 * each tensor->data at the corresponding offset. The OS pages in data
 * on demand, so startup is ~10x faster on cold cache and peak RSS is
 * lower (only touched pages count).
 *
 * Trade-off: cannot free individual tensors (they live in the mmap region),
 * so the free-float-weights optimization is disabled in mmap mode. Use
 * this when startup time matters more than steady-state RSS.
 *
 * The returned Tensor array must be freed with tensor_free_all_mmap(). */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    Tensor *tensors;
    int n_tensors;
    void *mmap_base;   /* the mmap'd region, for munmap later */
    size_t mmap_size;
    int fd;
} MmapedTensors;

static MmapedTensors g_mmap_state = {NULL, 0, NULL, 0, -1};

/* ========================================================================
 * Random-weight GPW2 generator — train an arbitrary-size model from scratch
 * (no pretrained checkpoint needed). Writes Gaussian-init weights in the same
 * "GPW2" layout that tensor_load_all / model_load expect, for any ModelConfig.
 * Keys follow the GPT-2 (qkv_merged) or LLaMA (separate Q/K/V, SwiGLU) layout
 * selected by cfg.qkv_merged / cfg.act_type.
 * ======================================================================== */
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
typedef struct { char key[64]; int ndim; int shape[4]; } TEntry;
static float bin_randn(void) {
    static int has = 0; static float spare = 0.0f;
    if (has) { has = 0; return spare; }
    float u = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float v = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float mag = sqrtf(-2.0f * logf(u));
    spare = mag * sinf(2.0f * M_PI * v); has = 1;
    return mag * cosf(2.0f * M_PI * v);
}
static void bin_push(TEntry **E, int *cnt, const char *key, int ndim, int s0, int s1, int s2, int s3) {
    TEntry *e = &(*E)[(*cnt)++];
    strncpy(e->key, key, sizeof(e->key) - 1); e->key[sizeof(e->key) - 1] = '\0';
    e->ndim = ndim; e->shape[0] = s0; e->shape[1] = s1; e->shape[2] = s2; e->shape[3] = s3;
}
static void bin_gpw2_put(FILE *f, const char *key, int ndim, const int *shape) {
    int klen = (int)strlen(key), n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    fwrite(&klen, 4, 1, f); fwrite(key, 1, (size_t)klen, f);
    fwrite(&ndim, 4, 1, f);
    for (int d = 0; d < ndim; d++) fwrite(&shape[d], 4, 1, f);
    for (int i = 0; i < n; i++) { float g = bin_randn() * 0.02f; fwrite(&g, 4, 1, f); }
}
/* Write a tensor with custom initialization to GPW2 file */
static void bin_gpw2_put_init(FILE *f, const char *key, int ndim, const int *shape, float scale, int init_mode) {
    /* init_mode: 0 = N(0, scale), 1 = constant scale, 2 = zeros, 3 = Xavier(sqrt(2/fan_in)) */
    int klen = (int)strlen(key), n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    fwrite(&klen, 4, 1, f); fwrite(key, 1, (size_t)klen, f);
    fwrite(&ndim, 4, 1, f);
    for (int d = 0; d < ndim; d++) fwrite(&shape[d], 4, 1, f);
    if (init_mode == 1) {
        /* constant value (for LayerNorm weight = 1.0) */
        for (int i = 0; i < n; i++) fwrite(&scale, 4, 1, f);
    } else if (init_mode == 2) {
        /* zeros (for biases) */
        float z = 0.0f;
        for (int i = 0; i < n; i++) fwrite(&z, 4, 1, f);
    } else if (init_mode == 3) {
        /* Xavier/He: std = sqrt(2.0 / fan_in) for ReLU/GELU, fan_in = shape[ndim-1] */
        float fan_in = (float)shape[ndim - 1];
        float std_val = sqrtf(2.0f / fan_in);
        for (int i = 0; i < n; i++) { float g = bin_randn() * std_val; fwrite(&g, 4, 1, f); }
    } else {
        /* Normal(0, scale) */
        for (int i = 0; i < n; i++) { float g = bin_randn() * scale; fwrite(&g, 4, 1, f); }
    }
}

void gen_random_gpw2(const char *path, ModelConfig cfg) {
    int n = cfg.n_embd, m = cfg.mlp_dim, V = cfg.vocab_size, C = cfg.n_ctx;
    int cnt = 0; char kb[64];
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "gen_random_gpw2: cannot write %s\n", path); exit(1); }

    /* Count tensors: base(4) + per_layer depends on config */
    int per_layer;
    if (cfg.qkv_merged) {
        per_layer = (cfg.act_type == ACT_SWIGLU) ? 11 : 12;
    } else {
        per_layer = 9;  /* q/k/v/o + gate/up/down + 2 layernorms */
    }
    int n_tensors = 4 + cfg.n_layer * per_layer;

    fwrite("GPW2", 1, 4, f);
    fwrite(&n_tensors, 4, 1, f);

    /* Embeddings: N(0, 1/sqrt(n_embd)) for proper scale */
    float emb_scale = 1.0f / sqrtf((float)n);
    bin_gpw2_put_init(f, "wte.weight", 2, (int[]){V, n}, emb_scale, 0);
    if (cfg.attn_type == ATTN_LEARNED)
        bin_gpw2_put_init(f, "wpe.weight", 2, (int[]){C, n}, emb_scale, 0);

    /* Final LayerNorm: weight=1.0, bias=0.0 (CRITICAL for convergence) */
    bin_gpw2_put_init(f, "ln_f.weight", 1, (int[]){n}, 1.0f, 1);
    bin_gpw2_put_init(f, "ln_f.bias", 1, (int[]){n}, 0.0f, 2);

    for (int l = 0; l < cfg.n_layer; l++) {
        if (cfg.qkv_merged) {
            /* Weight matrices: Xavier init (large enough for meaningful alpha after binarization) */
            snprintf(kb, sizeof kb, "h.%d.attn.c_attn.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){3*n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "h.%d.attn.c_attn.bias", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){3*n}, 0.0f, 2);
            snprintf(kb, sizeof kb, "h.%d.attn.c_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "h.%d.attn.c_proj.bias", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 0.0f, 2);
            if (cfg.act_type == ACT_SWIGLU) {
                snprintf(kb, sizeof kb, "h.%d.mlp.gate_proj.weight", l);
                bin_gpw2_put_init(f, kb, 2, (int[]){m, n}, 0.0f, 3);
                snprintf(kb, sizeof kb, "h.%d.mlp.up_proj.weight", l);
                bin_gpw2_put_init(f, kb, 2, (int[]){m, n}, 0.0f, 3);
                snprintf(kb, sizeof kb, "h.%d.mlp.down_proj.weight", l);
                bin_gpw2_put_init(f, kb, 2, (int[]){n, m}, 0.0f, 3);
            } else {
                snprintf(kb, sizeof kb, "h.%d.mlp.c_fc.weight", l);
                bin_gpw2_put_init(f, kb, 2, (int[]){m, n}, 0.0f, 3);
                snprintf(kb, sizeof kb, "h.%d.mlp.c_fc.bias", l);
                bin_gpw2_put_init(f, kb, 1, (int[]){m}, 0.0f, 2);
                snprintf(kb, sizeof kb, "h.%d.mlp.c_proj.weight", l);
                bin_gpw2_put_init(f, kb, 2, (int[]){n, m}, 0.0f, 3);
                snprintf(kb, sizeof kb, "h.%d.mlp.c_proj.bias", l);
                bin_gpw2_put_init(f, kb, 1, (int[]){n}, 0.0f, 2);
            }
            /* LayerNorm: weight=1.0, bias=0.0 */
            snprintf(kb, sizeof kb, "h.%d.ln_1.weight", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 1.0f, 1);
            snprintf(kb, sizeof kb, "h.%d.ln_1.bias", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 0.0f, 2);
            snprintf(kb, sizeof kb, "h.%d.ln_2.weight", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 1.0f, 1);
            snprintf(kb, sizeof kb, "h.%d.ln_2.bias", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 0.0f, 2);
        } else {
            snprintf(kb, sizeof kb, "model.layers.%d.self_attn.q_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.self_attn.k_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.self_attn.v_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.self_attn.o_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.mlp.gate_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){m, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.mlp.up_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){m, n}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.mlp.down_proj.weight", l);
            bin_gpw2_put_init(f, kb, 2, (int[]){n, m}, 0.0f, 3);
            snprintf(kb, sizeof kb, "model.layers.%d.input_layernorm.weight", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 1.0f, 1);
            snprintf(kb, sizeof kb, "model.layers.%d.post_attention_layernorm.weight", l);
            bin_gpw2_put_init(f, kb, 1, (int[]){n}, 1.0f, 1);
        }
        cnt++;
    }
    fclose(f);
    printf("[*] generated random weights (Xavier init, LN=1.0): %d tensors -> %s\n", n_tensors, path);
}

Tensor *tensor_load_all_mmap(const char *path, int *n_tensors) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) < 0) { fprintf(stderr, "fstat failed\n"); close(fd); return NULL; }
    size_t file_size = st.st_size;
    void *base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); close(fd); return NULL; }

    const unsigned char *p = (const unsigned char *)base;
    if (memcmp(p, "GPW2", 4) != 0) { fprintf(stderr, "bad magic\n"); munmap(base, file_size); close(fd); return NULL; }
    p += 4;
    int n = *(const int *)p; p += 4;
    *n_tensors = n;
    Tensor *t = calloc(n, sizeof(Tensor));
    for (int i = 0; i < n; i++) {
        int klen = *(const int *)p; p += 4;
        memcpy(t[i].key, p, klen); t[i].key[klen] = '\0'; p += klen;
        t[i].ndim = *(const int *)p; p += 4;
        int sz = 1;
        for (int d = 0; d < t[i].ndim; d++) {
            t[i].shape[d] = *(const int *)p; p += 4;
            sz *= t[i].shape[d];
        }
        /* Point data at the mmap'd region (no copy) */
        t[i].data = (float *)p;
        p += sz * sizeof(float);
    }
    g_mmap_state.tensors = t;
    g_mmap_state.n_tensors = n;
    g_mmap_state.mmap_base = base;
    g_mmap_state.mmap_size = file_size;
    g_mmap_state.fd = fd;
    return t;
}

void tensor_free_all_mmap(Tensor *tensors, int n) {
    /* Don't free individual data pointers — they live in the mmap region */
    free(tensors);
    if (g_mmap_state.mmap_base) {
        munmap(g_mmap_state.mmap_base, g_mmap_state.mmap_size);
        g_mmap_state.mmap_base = NULL;
    }
    if (g_mmap_state.fd >= 0) {
        close(g_mmap_state.fd);
        g_mmap_state.fd = -1;
    }
}

/* ========================================================================
 * Sparse Sliding Window Attention + Stateful Continuous Inference
 * ========================================================================
 * Implements:
 *   1. attention_forward_sliding() — sparse attention with configurable window
 *   2. attention_backward_sliding() — gradient computation for sparse attention
 *   3. Circular buffer KV cache management (no memcpy shifting)
 *   4. Attention sinks (StreamingLLM-style: keep first N tokens stable)
 *   5. trans_layer_forward_sliding() — layer forward with sparse attention
 *   6. Stateful inference context (g_sctx) for token-by-token generation
 *
 * Design:
 *   - Sliding window: each token attends to last W tokens + first S sink tokens
 *   - Circular buffer: KV cache uses ring buffer, write pointer wraps around
 *   - Attention sinks: first S positions are always in the attention window
 *   - Configurable via ModelConfig.sliding_window and ModelConfig.n_sinks
 * ======================================================================== */

/* Global stateful inference context */
StatefulContext g_sctx = {0};

/* ─── Sliding Window Attention Forward ──────────────────────────── */
void attention_forward_sliding(float *attn_out, const float *qkv,
                               int n_embd, int n_head,
                               int seq_pos,
                               float *k_cache_layer, float *v_cache_layer,
                               int n_ctx, int window_size, int n_sinks) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    const float *Q = qkv;
    const float *K_new = qkv + n_embd;
    const float *V_new = qkv + 2 * n_embd;

    /* Circular buffer: store at seq_pos % n_ctx */
    int cache_pos = seq_pos % n_ctx;
    memcpy(k_cache_layer + (size_t)cache_pos * n_embd, K_new, n_embd * sizeof(float));
    memcpy(v_cache_layer + (size_t)cache_pos * n_embd, V_new, n_embd * sizeof(float));

    /* Build attended position list: sinks + sliding window
     * FIX: when seq_pos < n_sinks, the window calculation produced
     * n_attend=0 (win_start clamped to n_sinks > seq_pos), causing
     * attention to output all-zeros for single-token prompts.
     * Ensure at least positions 0..seq_pos are attended. */
    int n_sink = (seq_pos < n_sinks) ? seq_pos : n_sinks;
    int win_start = seq_pos - window_size + 1;
    if (win_start < n_sinks) win_start = n_sinks;
    if (win_start > seq_pos) win_start = 0;  /* FIX: don't exclude current position */
    int n_win = seq_pos - win_start + 1;
    if (n_win < 0) n_win = 0;
    int n_attend = n_sink + n_win;
    if (n_attend < 1) n_attend = seq_pos + 1;  /* FIX: at least attend 0..seq_pos */
    if (n_attend > n_ctx) n_attend = n_ctx;

    /* Position index list */
    int pos_idx[2048];
    int *pos_list = (n_attend <= 2048) ? pos_idx : malloc(n_attend * sizeof(int));
    int idx = 0;
    for (int j = 0; j < n_sink && idx < n_attend; j++) pos_list[idx++] = j;
    for (int j = win_start; j <= seq_pos && idx < n_attend; j++) pos_list[idx++] = j;

    /* Scratch buffers */
    float scores_stack[256];
    float *scores = (n_attend <= 256) ? scores_stack : malloc(n_attend * sizeof(float));
    float *attn_w = (n_attend <= 256) ? scores_stack : malloc(n_attend * sizeof(float));

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;

        /* Compute attention scores */
        float max_score = -1e30f;
        for (int i = 0; i < n_attend; i++) {
            int j = pos_list[i];
            int phys_j = j % n_ctx;
            const float *K_jh = k_cache_layer + (size_t)phys_j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[i] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (int i = 0; i < n_attend; i++) {
            float e = expf(scores[i] - max_score);
            attn_w[i] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int i = 0; i < n_attend; i++) attn_w[i] *= inv_sum;

        /* Weighted sum of V */
        float *out_h = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (int i = 0; i < n_attend; i++) {
            int j = pos_list[i];
            int phys_j = j % n_ctx;
            float w = attn_w[i];
            const float *V_jh = v_cache_layer + (size_t)phys_j * n_embd + h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] += w * V_jh[d];
        }
    }

    if (n_attend > 256) { free(scores); free(attn_w); }
    if (n_attend > 2048) free(pos_list);
}

/* ─── Sliding Window Attention Backward ─────────────────────────── */
void attention_backward_sliding(float *grad_qkv, const float *grad_attn_out,
                                const float *qkv, int n_embd, int n_head,
                                int seq_pos,
                                const float *k_cache_layer, const float *v_cache_layer,
                                int n_ctx, int window_size, int n_sinks) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Determine attended positions (same as forward) */
    int n_sink = (seq_pos < n_sinks) ? seq_pos : n_sinks;
    int win_start = seq_pos - window_size + 1;
    if (win_start < n_sinks) win_start = n_sinks;
    int n_win = seq_pos - win_start + 1;
    if (n_win < 0) n_win = 0;
    int n_attend = n_sink + n_win;

    const float *Q = qkv;
    float *gQ = grad_qkv;
    float *gK = grad_qkv + n_embd;
    float *gV = grad_qkv + 2 * n_embd;
    memset(grad_qkv, 0, 3 * n_embd * sizeof(float));

    /* Build position index */
    int pos_idx[2048];
    int *pos_list = (n_attend <= 2048) ? pos_idx : malloc(n_attend * sizeof(int));
    int idx = 0;
    for (int j = 0; j < n_sink; j++) pos_list[idx++] = j;
    for (int j = win_start; j <= seq_pos; j++) pos_list[idx++] = j;

    /* Check if current position is in attended set */
    int have_self = 0;
    int self_idx = -1;
    for (int i = 0; i < n_attend; i++) {
        if (pos_list[i] == seq_pos) { have_self = 1; self_idx = i; break; }
    }

    float scores_stack[256];
    float *scores = (n_attend <= 256) ? scores_stack : malloc(n_attend * sizeof(float));
    float w_stack[256];
    float *w = (n_attend <= 256) ? w_stack : malloc(n_attend * sizeof(float));
    float gw_stack[256];
    float *g_w = (n_attend <= 256) ? gw_stack : malloc(n_attend * sizeof(float));

    for (int h = 0; h < n_head; h++) {
        const float *Q_h = Q + h * head_dim;
        const float *g_out_h = grad_attn_out + h * head_dim;

        /* Recompute scores + softmax */
        float max_score = -1e30f;
        for (int i = 0; i < n_attend; i++) {
            int j = pos_list[i];
            int phys_j = j % n_ctx;
            const float *K_jh = k_cache_layer + (size_t)phys_j * n_embd + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += Q_h[d] * K_jh[d];
            dot *= scale;
            scores[i] = dot;
            if (dot > max_score) max_score = dot;
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < n_attend; i++) {
            float e = expf(scores[i] - max_score);
            w[i] = e; sum_exp += e;
        }
        float inv = 1.0f / (sum_exp + 1e-12f);
        for (int i = 0; i < n_attend; i++) w[i] *= inv;

        /* g_w[i] = <g_out, V_{pos_list[i]}> */
        float dot_gw_w = 0.0f;
        for (int i = 0; i < n_attend; i++) {
            int j = pos_list[i];
            int phys_j = j % n_ctx;
            const float *V_jh = v_cache_layer + (size_t)phys_j * n_embd + h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += g_out_h[d] * V_jh[d];
            g_w[i] = s;
            dot_gw_w += w[i] * s;
        }
        /* g_scores[i] = w[i] * (g_w[i] - dot_gw_w) */
        for (int i = 0; i < n_attend; i++) g_w[i] = w[i] * (g_w[i] - dot_gw_w);

        /* g_Q[d] += sum_i g_scores[i] * K_{pos_list[i]}[d] * scale */
        float *gQ_h = gQ + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float s = 0.0f;
            for (int i = 0; i < n_attend; i++) {
                int j = pos_list[i];
                int phys_j = j % n_ctx;
                const float *K_jh = k_cache_layer + (size_t)phys_j * n_embd + h * head_dim;
                s += g_w[i] * K_jh[d];
            }
            gQ_h[d] += s * scale;
        }

        if (have_self) {
            float gs_cur = g_w[self_idx] * scale;
            float *gK_h = gK + h * head_dim;
            for (int d = 0; d < head_dim; d++) gK_h[d] += gs_cur * Q_h[d];
            float w_cur = w[self_idx];
            float *gV_h = gV + h * head_dim;
            for (int d = 0; d < head_dim; d++) gV_h[d] += w_cur * g_out_h[d];
        }
    }

    if (n_attend > 256) { free(scores); free(w); free(g_w); }
    if (n_attend > 2048) free(pos_list);
}

/* ─── Transformer Layer Forward with Sliding Window ─────────────── */
void trans_layer_forward_sliding(float *x, TransLayer *tl, TransAct *act,
                                  ModelConfig *cfg, int cache_pos, int abs_pos,
                                  int window, int n_sinks) {
    int n = cfg->n_embd, m = cfg->mlp_dim;
    float rs = cfg->residual_scale;
    act->seq_pos = abs_pos;

    /* Norm1 + QKV projection */
    memcpy(act->x_pre_norm1, x, n * sizeof(float));
    norm_forward(act->norm1_out, x, tl->norm1_w, tl->norm1_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm1, n, &act->norm1_cache[0], &act->norm1_cache[1]);

    if (cfg->qkv_merged) {
        bin_fwd(act->q, act->norm1_out, &tl->attn_q);
        act->k = act->q + n;
        act->v = act->q + 2 * n;
    } else {
        bin_fwd(act->q, act->norm1_out, &tl->attn_q);
        bin_fwd(act->k, act->norm1_out, &tl->attn_k);
        bin_fwd(act->v, act->norm1_out, &tl->attn_v);
    }

    /* Apply RoPE if configured */
    if (cfg->attn_type == ATTN_ROPE)
        apply_rope(act->q, act->k, abs_pos, cfg->n_head, n / cfg->n_head, n);

    /* Sliding window attention */
    if (tl->_kv_k && tl->_kv_v) {
        attention_forward_sliding(act->attn_out, act->q, n, cfg->n_head,
                                   cache_pos, tl->_kv_k, tl->_kv_v,
                                   cfg->n_ctx, window, n_sinks);
    } else {
        /* Fallback: V-copy (legacy) */
        memcpy(act->attn_out, act->v, n * sizeof(float));
    }

    /* Output projection */
    bin_fwd(act->proj_out, act->attn_out, &tl->attn_o);
    for (int i = 0; i < n; i++) x[i] += rs * act->proj_out[i];
    /* BUG #48 FIX: normalize residual stream */
    normalize_residual(x, n, 3.0f);

    /* Norm2 + MLP */
    memcpy(act->x_pre_norm2, x, n * sizeof(float));
    norm_forward(act->norm2_out, x, tl->norm2_w, tl->norm2_b, cfg->norm_type, n);
    compute_mean_std(act->x_pre_norm2, n, &act->norm2_cache[0], &act->norm2_cache[1]);

    if (cfg->act_type == ACT_SWIGLU) {
        /* BUG #44 FIX: static buffer instead of malloc/free per call */
        static float *sgate = NULL, *sup = NULL;
        static int sg_m = 0;
        if (sg_m != m) {
            free(sgate); free(sup);
            sgate = malloc(m * sizeof(float));
            sup = malloc(m * sizeof(float));
            sg_m = m;
        }
        bin_fwd(sgate, act->norm2_out, &tl->mlp_gate);
        bin_fwd(sup, act->norm2_out, &tl->mlp_up);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = silu(sgate[i]) * sup[i];
    } else {
        bin_fwd(act->mlp_hidden, act->norm2_out, &tl->mlp_gate);
        for (int i = 0; i < m; i++) act->mlp_hidden[i] = gelu(act->mlp_hidden[i]);
    }

    bin_fwd(act->mlp_out, act->mlp_hidden, &tl->mlp_down);
    /* MLP output normalization: prevent MLP from dominating attention.
     * Without this, CORE's core_gain=5 + lr_multiplier=3 causes MLP
     * weights (norm ~490) to grow 5x larger than attention (~90),
     * drowning out attention's prompt-specific signal. Normalize MLP
     * output to match attention's output magnitude before residual. */
    float mlp_norm = 0;
    for (int i = 0; i < n; i++) mlp_norm += act->mlp_out[i] * act->mlp_out[i];
    mlp_norm = sqrtf(mlp_norm) + 1e-8f;
    float attn_norm = 0;
    for (int i = 0; i < n; i++) attn_norm += act->proj_out[i] * act->proj_out[i];
    attn_norm = sqrtf(attn_norm) + 1e-8f;
    float mlp_scale = (attn_norm / mlp_norm);  /* scale MLP to match attention */
    for (int i = 0; i < n; i++) x[i] += rs * (act->proj_out[i] + mlp_scale * act->mlp_out[i]);
    /* BUG #48 FIX: normalize residual stream after MLP */
    normalize_residual(x, n, 3.0f);
}

/* ─── Stateful Inference: Begin New Session ─────────────────────── */

/* ========================================================================
 * Batch Training Implementation
 * ======================================================================== */

void model_batch_alloc(Model *m) {
    /* Buffers are already allocated in bin_layer_init, but this ensures
     * they exist for models loaded without g_use_adam. */
    for (int l = 0; l < m->cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        BinLayer *bls[8] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) {
            /* Separate Q/K/V — attn_k and attn_v also need buffers */
            bls[4] = &tl->attn_k; bls[5] = &tl->attn_v;
            n_bl = 6;
        }
        if (m->cfg.act_type == ACT_SWIGLU) {
            bls[n_bl] = &tl->mlp_up;
            n_bl++;
        }
        for (int b = 0; b < n_bl; b++) {
            BinLayer *bl = bls[b];
            if (!bl->grad_accum && bl->w_float) {
                bl->grad_accum = calloc((size_t)bl->in_dim * bl->out_dim, sizeof(float));
            }
            if (!bl->bias_grad_accum && bl->w_float) {
                bl->bias_grad_accum = calloc((size_t)bl->out_dim, sizeof(float));
            }
        }
    }

    /* Allocate gradient accumulation + Adam state for wte, wpe and ln_f */
    if (!m->grad_wte_accum) {
        size_t wte_size = (size_t)m->cfg.vocab_size * m->cfg.n_embd;
        m->grad_wte_accum = calloc(wte_size, sizeof(float));
        m->m_wte = calloc(wte_size, sizeof(float));
        m->v_wte = calloc(wte_size, sizeof(float));
    }
    if (m->wpe && !m->grad_wpe_accum) {
        size_t wpe_size = (size_t)m->cfg.n_ctx * m->cfg.n_embd;
        m->grad_wpe_accum = calloc(wpe_size, sizeof(float));
        m->m_wpe = calloc(wpe_size, sizeof(float));
        m->v_wpe = calloc(wpe_size, sizeof(float));
    }
    if (!m->grad_ln_f_w_accum) {
        m->grad_ln_f_w_accum = calloc(m->cfg.n_embd, sizeof(float));
        m->grad_ln_f_b_accum = calloc(m->cfg.n_embd, sizeof(float));
        m->m_ln_f_w = calloc(m->cfg.n_embd, sizeof(float));
        m->v_ln_f_w = calloc(m->cfg.n_embd, sizeof(float));
        m->m_ln_f_b = calloc(m->cfg.n_embd, sizeof(float));
        m->v_ln_f_b = calloc(m->cfg.n_embd, sizeof(float));
    }

    /* Allocate norm weight gradients for each layer */
    for (int l = 0; l < m->cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        if (!tl->grad_norm1_w) {
            tl->grad_norm1_w = calloc(m->cfg.n_embd, sizeof(float));
            tl->grad_norm1_b = calloc(m->cfg.n_embd, sizeof(float));
            tl->grad_norm2_w = calloc(m->cfg.n_embd, sizeof(float));
            tl->grad_norm2_b = calloc(m->cfg.n_embd, sizeof(float));
        }
    }
}

void model_batch_begin(Model *m) {
    for (int l = 0; l < m->cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        /* Zero norm weight gradients */
        if (tl->grad_norm1_w) {
            memset(tl->grad_norm1_w, 0, m->cfg.n_embd * sizeof(float));
            memset(tl->grad_norm1_b, 0, m->cfg.n_embd * sizeof(float));
            memset(tl->grad_norm2_w, 0, m->cfg.n_embd * sizeof(float));
            memset(tl->grad_norm2_b, 0, m->cfg.n_embd * sizeof(float));
        }
        BinLayer *bls[8] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) { bls[4] = &tl->attn_k; bls[5] = &tl->attn_v; n_bl = 6; }
        if (m->cfg.act_type == ACT_SWIGLU) { bls[n_bl] = &tl->mlp_up; n_bl++; }
        for (int b = 0; b < n_bl; b++) {
            BinLayer *bl = bls[b];
            if (bl->grad_accum)
                memset(bl->grad_accum, 0, (size_t)bl->in_dim * bl->out_dim * sizeof(float));
            if (bl->bias_grad_accum)
                memset(bl->bias_grad_accum, 0, (size_t)bl->out_dim * sizeof(float));
        }
    }
    /* Zero embedding and norm gradients */
    if (m->grad_wte_accum)
        memset(m->grad_wte_accum, 0, (size_t)m->cfg.vocab_size * m->cfg.n_embd * sizeof(float));
    if (m->grad_wpe_accum)
        memset(m->grad_wpe_accum, 0, (size_t)m->cfg.n_ctx * m->cfg.n_embd * sizeof(float));
    if (m->grad_ln_f_w_accum) {
        memset(m->grad_ln_f_w_accum, 0, m->cfg.n_embd * sizeof(float));
        memset(m->grad_ln_f_b_accum, 0, m->cfg.n_embd * sizeof(float));
    }
}

float model_batch_forward(Model *m, const int *tokens, int n_tokens) {
    /* Identical to model_forward — just a semantic alias for batch training */
    return model_forward(m, tokens, n_tokens);
}

void model_batch_backward(Model *m, const int *tokens, int n_tokens) {
    /* Set accumulation flag, call normal backward, then unset.
     * The backward pass will accumulate gradients instead of updating weights. */
    int prev = g_accumulate_gradients;
    g_accumulate_gradients = 1;

    int n = m->cfg.n_embd;
    int target = tokens[n_tokens];
    static float gh[4096];
    static float *g_full_logits = NULL;
    static int g_full_logits_vocab = 0;
    if (g_full_logits_vocab != m->cfg.vocab_size) {
        free(g_full_logits);
        g_full_logits = malloc((size_t)m->cfg.vocab_size * sizeof(float));
        g_full_logits_vocab = m->cfg.vocab_size;
    }
    /* Recompute softmax probs for gradient (matches model_backward) */
    cross_entropy_full(m->final_ln, m->wte, target, m->cfg.vocab_size, n, g_full_logits);
    cross_entropy_full_grad(gh, m->final_ln, m->wte, target, m->cfg.vocab_size, n, g_full_logits);

    /* Gradient clipping */
    float gnorm = 0;
    for (int i = 0; i < n; i++) gnorm += gh[i] * gh[i];
    gnorm = sqrtf(gnorm);
    if (gnorm > 1.0f) { float clip = 1.0f / gnorm; for (int i = 0; i < n; i++) gh[i] *= clip; }

    /* Backprop through final norm */
    static float g_pre[4096];
    norm_backward(g_pre, gh, m->x_before_final, m->ln_f_w,
                  (float[]){m->final_mean, m->final_std_inv}, m->cfg.norm_type, n,
                  m->grad_ln_f_w_accum, m->grad_ln_f_b_accum);
    memcpy(gh, g_pre, n * sizeof(float));

    /* Backprop through layers (gradients accumulate, no weight update) */
    for (int l = m->cfg.n_layer - 1; l >= 0; l--)
        trans_layer_backward(gh, &m->layers[l], &m->acts[l], &m->cfg, 0.0f);  /* lr=0, not used when accumulating */

    /* === Accumulate gradient for token embedding (wte) ===
     * gh is dL/d_x at position t (the prediction position).
     * x_t = wte[tokens[t]] + wpe[t], so grad_wte[tokens[t]] += gh.
     * Earlier positions (0..t-1) contribute via attention KV cache,
     * but attention_backward treats cached K/V as constants (no gradient).
     * So we only update the last token's wte, not all tokens.
     * Previously: accumulated to ALL tokens (diluted gradient 1/n_tokens). */
    if (m->grad_wte_accum) {
        int input_token = tokens[n_tokens - 1];
        if (input_token >= 0 && input_token < m->cfg.vocab_size) {
            float *gw = &m->grad_wte_accum[(size_t)input_token * n];
            for (int i = 0; i < n; i++)
                gw[i] += gh[i];
        }
    }

    /* === Accumulate gradient for position embedding (wpe) ===
     * Same logic: only position t (last position) gets gradient. */
    if (m->grad_wpe_accum) {
        int pos = n_tokens - 1;
        if (pos >= 0 && pos < m->cfg.n_ctx) {
            float *gw = &m->grad_wpe_accum[(size_t)pos * n];
            for (int i = 0; i < n; i++)
                gw[i] += gh[i];
        }
    }

    /* Also accumulate gradient for ln_f_w and ln_f_b.
     * gh BEFORE norm_backward was the gradient w.r.t. final_ln output.
     * But we already overwrote gh with norm_backward result.
     * So we recompute: grad_ln_f_w[i] = grad_final_ln[i] * (x[i] - mean) * std_inv
     * We saved final_mean, final_std_inv, and x_before_final. */
    if (m->grad_ln_f_w_accum) {
        for (int i = 0; i < n; i++) {
            float normalized = (m->x_before_final[i] - m->final_mean) * m->final_std_inv;
            /* grad_ln_f_w needs the gradient w.r.t. final_ln output, which is gh
             * BEFORE norm_backward. We need to save it. For now, approximate
             * using the pre-norm gradient (gh before norm_backward was stored
             * in g_pre, but we already copied it to gh). Skip ln_f for now. */
        }
    }

    g_accumulate_gradients = prev;  /* Restore */
}

void model_batch_apply(Model *m, float lr, int batch_size) {
    /* Apply accumulated gradients with Adam, averaged by batch_size.
     * This does ONE optimizer step for the entire batch. */
    int n = m->cfg.n_embd;
    float inv_batch = 1.0f / (float)batch_size;

    for (int l = 0; l < m->cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        BinLayer *bls[8] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) { bls[4] = &tl->attn_k; bls[5] = &tl->attn_v; n_bl = 6; }
        if (m->cfg.act_type == ACT_SWIGLU) { bls[n_bl] = &tl->mlp_up; n_bl++; }

        for (int b = 0; b < n_bl; b++) {
            BinLayer *bl = bls[b];
            if (!bl->grad_accum || !bl->w_float) continue;

            int in = bl->in_dim, out = bl->out_dim;
            int t = g_opt_step + 1;
            float bc1 = 1.0f - powf(g_adam_beta1, (float)t);
            float bc2 = 1.0f - powf(g_adam_beta2, (float)t);

            /* === LAL-aware Adam: group-wise second moment ===
             * Standard Adam normalizes per-parameter: update = m / sqrt(v_per_param)
             * This erases CORE/BINARY gradient differences because large CORE
             * gradients produce large v, reducing the effective update.
             *
             * LAL-aware Adam shares v within each group (CORE, BINARY):
             *   v_core = EMA(mean(||grad_core||^2))  -- shared across ALL CORE params
             *   v_bin  = EMA(mean(||grad_bin||^2))   -- shared across ALL BINARY params
             *   update = m[i] / sqrt(v_group)        -- group-normalized
             *
             * This preserves relative gradient magnitudes: if CORE has 3x
             * larger gradients than BINARY, the update is 3x larger too.
             * Combined with g_core_lr_multiplier, CORE truly learns faster.
             *
             * PRUNE neurons: weight decay toward 0 + freeze if small. */
            if (g_use_lal_adam && bl->logic_mask) {
                /* Step 1: Compute group-wise gradient energy */
                float core_g_sq = 0, bin_g_sq = 0;
                int n_core = 0, n_bin = 0;
                for (int j = 0; j < out; j++) {
                    uint8_t m = bl->logic_mask[j];
                    if (m == 2) continue;
                    const float *ga = &bl->grad_accum[j * in];
                    float row_sq = 0;
                    for (int i = 0; i < in; i++) row_sq += ga[i] * ga[i];
                    row_sq /= in;  /* per-param average within this neuron */
                    if (m == 0) { core_g_sq += row_sq; n_core++; }
                    else        { bin_g_sq  += row_sq; n_bin++;  }
                }
                core_g_sq = n_core > 0 ? core_g_sq / n_core : 0;
                bin_g_sq  = n_bin  > 0 ? bin_g_sq  / n_bin  : 0;

                /* Step 2: EMA update of group v (persist across steps) */
                /* Store in first CORE/BINARY neuron's v_adam[0] as proxy.
                 * This is safe because v_adam is per-param and we only
                 * read v_adam[0] of the first neuron in each group. */
                int core_first = -1, bin_first = -1;
                for (int j = 0; j < out; j++) {
                    uint8_t m = bl->logic_mask[j];
                    if (m == 0 && core_first < 0) core_first = j;
                    if (m == 1 && bin_first  < 0) bin_first  = j;
                }

                float core_v, bin_v;
                if (g_use_adam && bl->v_adam) {
                    if (core_first >= 0) {
                        bl->v_adam[core_first * in] =
                            g_adam_beta2 * bl->v_adam[core_first * in] +
                            (1.0f - g_adam_beta2) * core_g_sq;
                        core_v = bl->v_adam[core_first * in] / bc2;
                    } else core_v = 1e-8f;
                    if (bin_first >= 0) {
                        bl->v_adam[bin_first * in] =
                            g_adam_beta2 * bl->v_adam[bin_first * in] +
                            (1.0f - g_adam_beta2) * bin_g_sq;
                        bin_v = bl->v_adam[bin_first * in] / bc2;
                    } else bin_v = 1e-8f;
                } else {
                    core_v = core_g_sq;
                    bin_v = bin_g_sq;
                }
                float core_sqrt_v = sqrtf(core_v) + g_adam_eps;
                float bin_sqrt_v  = sqrtf(bin_v)  + g_adam_eps;

                /* Step 3: Update weights using group-wise normalization */
                for (int j = 0; j < out; j++) {
                    uint8_t m = bl->logic_mask[j];
                    float *wf = &bl->w_float[j * in];

                    if (m == 2) {
                        /* PRUNE: weight decay toward zero */
                        for (int i = 0; i < in; i++) {
                            float w = wf[i] * (1.0f - g_prune_decay);
                            if (fabsf(w) < g_prune_freeze_thresh) w = 0.0f;
                            wf[i] = w;
                        }
                        continue;
                    }

                    float lr_j = (m == 0) ? lr * g_core_lr_multiplier : lr;
                    /* BUG #22 FIX: use the right group's sqrt_v for each neuron.
                     * Previously hardcoded bin_sqrt_v for BOTH groups, which
                     * artificially amplified CORE updates (CORE has 10x larger
                     * gradients from alpha=2 vs beta=0.2 + sqrt(807/201) normalization,
                     * so core_v >> bin_v; using bin_sqrt_v as denominator makes
                     * CORE effective lr explode on top of g_core_lr_multiplier=3.0).
                     *
                     * With this fix, each group is normalized by its own
                     * second moment — relative gradient magnitudes within a
                     * group are preserved, and cross-group scaling is left
                     * to g_core_lr_multiplier alone. */
                    float sqrt_v = (m == 0) ? core_sqrt_v : bin_sqrt_v;
                    float *ga = &bl->grad_accum[j * in];

                    if (g_use_adam && bl->m_adam) {
                        float *ma = &bl->m_adam[j * in];
                        for (int i = 0; i + 7 < in; i += 8) {
                            float g0=ga[i+0]*inv_batch, g1=ga[i+1]*inv_batch;
                            float g2=ga[i+2]*inv_batch, g3=ga[i+3]*inv_batch;
                            float g4=ga[i+4]*inv_batch, g5=ga[i+5]*inv_batch;
                            float g6=ga[i+6]*inv_batch, g7=ga[i+7]*inv_batch;
                            ma[i+0]=g_adam_beta1*ma[i+0]+(1.0f-g_adam_beta1)*g0;
                            ma[i+1]=g_adam_beta1*ma[i+1]+(1.0f-g_adam_beta1)*g1;
                            ma[i+2]=g_adam_beta1*ma[i+2]+(1.0f-g_adam_beta1)*g2;
                            ma[i+3]=g_adam_beta1*ma[i+3]+(1.0f-g_adam_beta1)*g3;
                            ma[i+4]=g_adam_beta1*ma[i+4]+(1.0f-g_adam_beta1)*g4;
                            ma[i+5]=g_adam_beta1*ma[i+5]+(1.0f-g_adam_beta1)*g5;
                            ma[i+6]=g_adam_beta1*ma[i+6]+(1.0f-g_adam_beta1)*g6;
                            ma[i+7]=g_adam_beta1*ma[i+7]+(1.0f-g_adam_beta1)*g7;
                            /* GROUP-WISE v: use sqrt_v, not per-param v */
                            wf[i+0]-=lr_j*(ma[i+0]/bc1)/sqrt_v;
                            wf[i+1]-=lr_j*(ma[i+1]/bc1)/sqrt_v;
                            wf[i+2]-=lr_j*(ma[i+2]/bc1)/sqrt_v;
                            wf[i+3]-=lr_j*(ma[i+3]/bc1)/sqrt_v;
                            wf[i+4]-=lr_j*(ma[i+4]/bc1)/sqrt_v;
                            wf[i+5]-=lr_j*(ma[i+5]/bc1)/sqrt_v;
                            wf[i+6]-=lr_j*(ma[i+6]/bc1)/sqrt_v;
                            wf[i+7]-=lr_j*(ma[i+7]/bc1)/sqrt_v;
                        }
                        for (int i = (in/8)*8; i < in; i++) {
                            float g = ga[i]*inv_batch;
                            ma[i]=g_adam_beta1*ma[i]+(1.0f-g_adam_beta1)*g;
                            wf[i]-=lr_j*(ma[i]/bc1)/sqrt_v;
                        }
                    } else {
                        float scale = lr_j * inv_batch / sqrt_v;
                        for (int i = 0; i + 7 < in; i += 8) {
                            wf[i+0]-=scale*ga[i+0]; wf[i+1]-=scale*ga[i+1];
                            wf[i+2]-=scale*ga[i+2]; wf[i+3]-=scale*ga[i+3];
                            wf[i+4]-=scale*ga[i+4]; wf[i+5]-=scale*ga[i+5];
                            wf[i+6]-=scale*ga[i+6]; wf[i+7]-=scale*ga[i+7];
                        }
                        for (int i = (in/8)*8; i < in; i++)
                            wf[i] -= scale * ga[i];
                    }
                    bl->bias[j] -= lr_j * bl->bias_grad_accum[j] * inv_batch;
                }
                /* Skip standard Adam loop — already done above */
                goto layer_done;
            }

            for (int j = 0; j < out; j++) {
                if (bl->logic_mask && bl->logic_mask[j] == 2) continue;
                /* CORE neurons get boosted learning rate for faster differentiation */
                float lr_j = lr;
                if (bl->logic_mask && bl->logic_mask[j] == 0)
                    lr_j = lr * g_core_lr_multiplier;
                float *wf = &bl->w_float[j * in];
                float *ga = &bl->grad_accum[j * in];

                if (g_use_adam && bl->m_adam) {
                    float *ma = &bl->m_adam[j * in];
                    float *va = &bl->v_adam[j * in];
                    for (int i = 0; i + 7 < in; i += 8) {
                        /* Average gradient over batch */
                        float g0 = ga[i+0]*inv_batch, g1 = ga[i+1]*inv_batch;
                        float g2 = ga[i+2]*inv_batch, g3 = ga[i+3]*inv_batch;
                        float g4 = ga[i+4]*inv_batch, g5 = ga[i+5]*inv_batch;
                        float g6 = ga[i+6]*inv_batch, g7 = ga[i+7]*inv_batch;
                        /* Adam moment updates */
                        ma[i+0]=g_adam_beta1*ma[i+0]+(1.0f-g_adam_beta1)*g0;
                        ma[i+1]=g_adam_beta1*ma[i+1]+(1.0f-g_adam_beta1)*g1;
                        ma[i+2]=g_adam_beta1*ma[i+2]+(1.0f-g_adam_beta1)*g2;
                        ma[i+3]=g_adam_beta1*ma[i+3]+(1.0f-g_adam_beta1)*g3;
                        ma[i+4]=g_adam_beta1*ma[i+4]+(1.0f-g_adam_beta1)*g4;
                        ma[i+5]=g_adam_beta1*ma[i+5]+(1.0f-g_adam_beta1)*g5;
                        ma[i+6]=g_adam_beta1*ma[i+6]+(1.0f-g_adam_beta1)*g6;
                        ma[i+7]=g_adam_beta1*ma[i+7]+(1.0f-g_adam_beta1)*g7;
                        va[i+0]=g_adam_beta2*va[i+0]+(1.0f-g_adam_beta2)*g0*g0;
                        va[i+1]=g_adam_beta2*va[i+1]+(1.0f-g_adam_beta2)*g1*g1;
                        va[i+2]=g_adam_beta2*va[i+2]+(1.0f-g_adam_beta2)*g2*g2;
                        va[i+3]=g_adam_beta2*va[i+3]+(1.0f-g_adam_beta2)*g3*g3;
                        va[i+4]=g_adam_beta2*va[i+4]+(1.0f-g_adam_beta2)*g4*g4;
                        va[i+5]=g_adam_beta2*va[i+5]+(1.0f-g_adam_beta2)*g5*g5;
                        va[i+6]=g_adam_beta2*va[i+6]+(1.0f-g_adam_beta2)*g6*g6;
                        va[i+7]=g_adam_beta2*va[i+7]+(1.0f-g_adam_beta2)*g7*g7;
                        /* Bias-corrected update */
                        float mh0=ma[i+0]/bc1, mh1=ma[i+1]/bc1;
                        float mh2=ma[i+2]/bc1, mh3=ma[i+3]/bc1;
                        float mh4=ma[i+4]/bc1, mh5=ma[i+5]/bc1;
                        float mh6=ma[i+6]/bc1, mh7=ma[i+7]/bc1;
                        float vh0=sqrtf(va[i+0]/bc2)+g_adam_eps;
                        float vh1=sqrtf(va[i+1]/bc2)+g_adam_eps;
                        float vh2=sqrtf(va[i+2]/bc2)+g_adam_eps;
                        float vh3=sqrtf(va[i+3]/bc2)+g_adam_eps;
                        float vh4=sqrtf(va[i+4]/bc2)+g_adam_eps;
                        float vh5=sqrtf(va[i+5]/bc2)+g_adam_eps;
                        float vh6=sqrtf(va[i+6]/bc2)+g_adam_eps;
                        float vh7=sqrtf(va[i+7]/bc2)+g_adam_eps;
                        wf[i+0]-=lr_j*mh0/vh0; wf[i+1]-=lr_j*mh1/vh1;
                        wf[i+2]-=lr_j*mh2/vh2; wf[i+3]-=lr_j*mh3/vh3;
                        wf[i+4]-=lr_j*mh4/vh4; wf[i+5]-=lr_j*mh5/vh5;
                        wf[i+6]-=lr_j*mh6/vh6; wf[i+7]-=lr_j*mh7/vh7;
                    }
                    for (int i = (in/8)*8; i < in; i++) {
                        float g = ga[i]*inv_batch;
                        ma[i]=g_adam_beta1*ma[i]+(1.0f-g_adam_beta1)*g;
                        va[i]=g_adam_beta2*va[i]+(1.0f-g_adam_beta2)*g*g;
                        wf[i]-=lr_j*(ma[i]/bc1)/(sqrtf(va[i]/bc2)+g_adam_eps);
                    }
                } else {
                    /* SGD: w -= lr * avg_grad */
                    float scale = lr_j * inv_batch;
                    for (int i = 0; i + 7 < in; i += 8) {
                        wf[i+0]-=scale*ga[i+0]; wf[i+1]-=scale*ga[i+1];
                        wf[i+2]-=scale*ga[i+2]; wf[i+3]-=scale*ga[i+3];
                        wf[i+4]-=scale*ga[i+4]; wf[i+5]-=scale*ga[i+5];
                        wf[i+6]-=scale*ga[i+6]; wf[i+7]-=scale*ga[i+7];
                    }
                    for (int i = (in/8)*8; i < in; i++)
                        wf[i] -= scale * ga[i];
                }
                /* Update bias */
                bl->bias[j] -= lr_j * bl->bias_grad_accum[j] * inv_batch;
            }

layer_done:
            /* Weight clipping + repack: per-neuron based on logic_mask.
             * CORE (float): ±2.0 — needs room for precise differentiation.
             * BINARY (sign): ±1.0 — must stay near ±1 for sign function.
             * PRUNE: already skipped in update loop above. */
            if (!g_use_pure_float) {
                for (int j = 0; j < out; j++) {
                    float clip_val = 1.0f;  /* BINARY default */
                    if (bl->logic_mask && bl->logic_mask[j] == 0)
                        clip_val = 2.0f;  /* CORE: allow larger float weights */
                    /* PRUNE (mask==2) already skipped, but clip anyway for safety */
                    float *wf_row = &bl->w_float[j * in];
                    for (int i = 0; i < in; i++) {
                        if (wf_row[i] > clip_val) wf_row[i] = clip_val;
                        else if (wf_row[i] < -clip_val) wf_row[i] = -clip_val;
                    }
                }
                bin_layer_repack(bl);
            } else {
                #define W_CLIP_BF 2.0f
                for (int i = 0; i < in * out; i++) {
                    float w = bl->w_float[i];
                    if (w > W_CLIP_BF) bl->w_float[i] = W_CLIP_BF;
                    else if (w < -W_CLIP_BF) bl->w_float[i] = -W_CLIP_BF;
                }
                #undef W_CLIP_BF
            }
        }
    }

    /* === CRITICAL FIX: Update token embeddings (wte) with Adam ===
     * Without this, embeddings are frozen and the model cannot learn
     * concept boundaries. This is the #1 fix for LAL whitebox training. */
    if (m->grad_wte_accum && m->m_wte && m->v_wte && g_use_adam) {
        int vocab = m->cfg.vocab_size;
        int t = g_opt_step + 1;
        float bc1 = 1.0f - powf(g_adam_beta1, (float)t);
        float bc2 = 1.0f - powf(g_adam_beta2, (float)t);
        float inv_batch = 1.0f / (float)batch_size;

        for (int v = 0; v < vocab; v++) {
            float *w  = &m->wte[(size_t)v * n];
            float *gw = &m->grad_wte_accum[(size_t)v * n];
            float *ma = &m->m_wte[(size_t)v * n];
            float *va = &m->v_wte[(size_t)v * n];
            for (int i = 0; i < n; i++) {
                float g = gw[i] * inv_batch;
                if (fabsf(g) < 1e-12f) continue;  /* skip unused tokens */
                ma[i] = g_adam_beta1 * ma[i] + (1.0f - g_adam_beta1) * g;
                va[i] = g_adam_beta2 * va[i] + (1.0f - g_adam_beta2) * g * g;
                float mh = ma[i] / bc1;
                float vh = sqrtf(va[i] / bc2) + g_adam_eps;
                w[i] -= lr * mh / vh;
            }
        }
    }

    /* === Update position embeddings (wpe) with Adam ===
     * Without this, position embeddings are random noise → model has no
     * position awareness → attention collapses all positions → same output. */
    if (m->grad_wpe_accum && m->m_wpe && m->v_wpe && g_use_adam && m->wpe) {
        int n_ctx = m->cfg.n_ctx;
        int t = g_opt_step + 1;
        float bc1 = 1.0f - powf(g_adam_beta1, (float)t);
        float bc2 = 1.0f - powf(g_adam_beta2, (float)t);
        float inv_batch = 1.0f / (float)batch_size;

        for (int pos = 0; pos < n_ctx; pos++) {
            float *w  = &m->wpe[(size_t)pos * n];
            float *gw = &m->grad_wpe_accum[(size_t)pos * n];
            float *ma = &m->m_wpe[(size_t)pos * n];
            float *va = &m->v_wpe[(size_t)pos * n];
            for (int i = 0; i < n; i++) {
                float g = gw[i] * inv_batch;
                if (fabsf(g) < 1e-12f) continue;
                ma[i] = g_adam_beta1 * ma[i] + (1.0f - g_adam_beta1) * g;
                va[i] = g_adam_beta2 * va[i] + (1.0f - g_adam_beta2) * g * g;
                float mh = ma[i] / bc1;
                float vh = sqrtf(va[i] / bc2) + g_adam_eps;
                w[i] -= lr * mh / vh;
            }
        }
    }

    /* === Update LayerNorm weights with proper Adam ===
     * Now using correct gradients from layer_norm_backward (grad_w/grad_b).
     * Previously these were stuck at init (w=1.0, b=0.0) because
     * layer_norm_backward didn't compute grad_w, causing all inputs
     * to produce identical final_ln. */
    for (int l = 0; l < m->cfg.n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        if (tl->grad_norm1_w) {
            for (int i = 0; i < n; i++) {
                /* SGD with weight decay toward 1.0 (w) / 0.0 (b) + gradient clipping */
                float g1w = tl->grad_norm1_w[i] * inv_batch + 0.01f * (tl->norm1_w[i] - 1.0f);
                float g1b = tl->grad_norm1_b[i] * inv_batch + 0.01f * tl->norm1_b[i];
                float g2w = tl->grad_norm2_w[i] * inv_batch + 0.01f * (tl->norm2_w[i] - 1.0f);
                float g2b = tl->grad_norm2_b[i] * inv_batch + 0.01f * tl->norm2_b[i];
                /* Clip gradients */
                if (g1w > 0.1f) g1w = 0.1f; if (g1w < -0.1f) g1w = -0.1f;
                if (g1b > 0.1f) g1b = 0.1f; if (g1b < -0.1f) g1b = -0.1f;
                if (g2w > 0.1f) g2w = 0.1f; if (g2w < -0.1f) g2w = -0.1f;
                if (g2b > 0.1f) g2b = 0.1f; if (g2b < -0.1f) g2b = -0.1f;
                tl->norm1_w[i] -= lr * g1w;
                tl->norm1_b[i] -= lr * g1b;
                tl->norm2_w[i] -= lr * g2w;
                tl->norm2_b[i] -= lr * g2b;
            }
        }
    }
    /* ln_f weights with Adam */
    if (m->grad_ln_f_w_accum && m->m_ln_f_w && g_use_adam) {
        int t = g_opt_step + 1;
        float bc1 = 1.0f - powf(g_adam_beta1, (float)t);
        float bc2 = 1.0f - powf(g_adam_beta2, (float)t);
        for (int i = 0; i < n; i++) {
            float gw = m->grad_ln_f_w_accum[i] * inv_batch;
            float gb = m->grad_ln_f_b_accum[i] * inv_batch;
            if (fabsf(gw) < 1e-12f && fabsf(gb) < 1e-12f) continue;
            m->m_ln_f_w[i] = g_adam_beta1 * m->m_ln_f_w[i] + (1.0f - g_adam_beta1) * gw;
            m->v_ln_f_w[i] = g_adam_beta2 * m->v_ln_f_w[i] + (1.0f - g_adam_beta2) * gw * gw;
            m->m_ln_f_b[i] = g_adam_beta1 * m->m_ln_f_b[i] + (1.0f - g_adam_beta1) * gb;
            m->v_ln_f_b[i] = g_adam_beta2 * m->v_ln_f_b[i] + (1.0f - g_adam_beta2) * gb * gb;
            float mhw = m->m_ln_f_w[i] / bc1, vhw = sqrtf(m->v_ln_f_w[i] / bc2) + g_adam_eps;
            float mhb = m->m_ln_f_b[i] / bc1, vhb = sqrtf(m->v_ln_f_b[i] / bc2) + g_adam_eps;
            m->ln_f_w[i] -= lr * mhw / vhw;
            m->ln_f_b[i] -= lr * mhb / vhb;
        }
    }

    /* === Sync w_core and wbits from updated w_float ===
     * model_batch_apply updates w_float, but forward pass uses:
     * - w_core for CORE neurons (bin_forward_pure_float)
     * - wbits (sign(w_float)) for BINARY neurons (bin_forward)
     * - alpha (mean(|w_float|)) for BINARY scaling
     * Without this sync, CORE weights are FROZEN at init! */
    for (int l = 0; l < m->cfg.n_layer; l++) {
        BinLayer *bls[8] = {&m->layers[l].attn_q, &m->layers[l].attn_o,
                            &m->layers[l].mlp_gate, &m->layers[l].mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) { bls[4] = &m->layers[l].attn_k; bls[5] = &m->layers[l].attn_v; n_bl = 6; }
        if (m->cfg.act_type == ACT_SWIGLU) { bls[n_bl] = &m->layers[l].mlp_up; n_bl++; }
        for (int b = 0; b < n_bl; b++) {
            if (bls[b]->w_float && bls[b]->logic_mask)
                bin_layer_repack(bls[b]);
        }
    }

    /* Increment Adam step once per batch */
    if (g_use_adam) g_opt_step++;
}

void model_stateful_begin(Model *m) {
    /* Ensure KV cache is allocated */
    if (!m->k_cache) model_kv_cache_alloc(m);

    /* Reset KV cache to zero */
    int n_layer = m->cfg.n_layer;
    size_t per_layer = (size_t)m->cfg.n_ctx * m->cfg.n_embd * sizeof(float);
    for (int l = 0; l < n_layer; l++) {
        memset(m->k_cache[l], 0, per_layer);
        memset(m->v_cache[l], 0, per_layer);
    }

    /* Allocate stateful context buffers */
    if (!g_sctx.x) g_sctx.x = malloc(m->cfg.n_embd * sizeof(float));
    if (!g_sctx.logits) g_sctx.logits = malloc(m->cfg.vocab_size * sizeof(float));

    g_sctx.kv_pos = 0;
    g_sctx.total_pos = 0;
    g_sctx.active = 1;

    int window = m->cfg.sliding_window > 0 ? m->cfg.sliding_window : m->cfg.n_ctx;
    int sinks = m->cfg.n_sinks;
    printf("[*] stateful inference started: window=%d, sinks=%d, ctx=%d\n",
           window, sinks, m->cfg.n_ctx);
}

/* ─── Stateful Inference: Reset KV Cache ────────────────────────── */
void model_stateful_reset(Model *m) {
    if (!m->k_cache) return;
    int n_layer = m->cfg.n_layer;
    size_t per_layer = (size_t)m->cfg.n_ctx * m->cfg.n_embd * sizeof(float);
    for (int l = 0; l < n_layer; l++) {
        memset(m->k_cache[l], 0, per_layer);
        memset(m->v_cache[l], 0, per_layer);
    }
    g_sctx.kv_pos = 0;
    g_sctx.total_pos = 0;
}

/* ─── Stateful Forward with Sliding Window ──────────────────────── */
const float *model_stateful_forward_sliding(Model *m, int token) {
    if (!g_sctx.active || !m->k_cache) {
        fprintf(stderr, "[!] stateful mode not active — call model_stateful_begin() first\n");
        return NULL;
    }
    int n = m->cfg.n_embd, nL = m->cfg.n_layer, ctx = m->cfg.n_ctx;
    int window = m->cfg.sliding_window > 0 ? m->cfg.sliding_window : ctx;
    int n_sinks = m->cfg.n_sinks;

    /* Circular buffer: no need to shift. Just wrap around. */
    int pos = g_sctx.kv_pos;       /* logical position in cache */
    int abs_pos = g_sctx.total_pos; /* absolute position in sequence */
    int pe_pos = (m->cfg.attn_type == ATTN_LEARNED) ? (abs_pos % ctx) : abs_pos;

    float *x = g_sctx.x;
    /* Embedding lookup + position encoding */
    for (int i = 0; i < n; i++) {
        x[i] = m->wte[(size_t)token * n + i];
        if (m->wpe) x[i] += m->wpe[(size_t)pe_pos * n + i];
    }

    /* Forward through layers with sliding window attention */
    for (int l = 0; l < nL; l++)
        trans_layer_forward_sliding(x, &m->layers[l], &m->acts[l], &m->cfg,
                                     pos, abs_pos, window, n_sinks);

    /* Final norm + logits (tied embeddings) */
    memcpy(m->x_before_final, x, n * sizeof(float));
    norm_forward(m->final_ln, x, m->ln_f_w, m->ln_f_b, m->cfg.norm_type, n);
    compute_mean_std(m->x_before_final, n, &m->final_mean, &m->final_std_inv);

    int V = m->cfg.vocab_size;
    /* Logits: raw dot product (tied embeddings).
     * Cosine normalization removed — it compressed logit range too much,
     * making sampling unable to distinguish good tokens from noise.
     * Repetition penalty in generation handles mode collapse instead.
     *
     * BUG #42 FIX: was a scalar loop (1 mul-add per iteration).
     * Same computation as compute_full_logits and model_forward_float_logits,
     * which both use 8-way unrolled loops for SIMD vectorization.
     * The scalar version was 4-8x slower on vocab=32768. Now matches. */
    for (int j = 0; j < V; j++) {
        const float *w = &m->wte[(size_t)j * n];
        float s = 0;
        for (int k = 0; k + 7 < n; k += 8)
            s += m->final_ln[k+0]*w[k+0] + m->final_ln[k+1]*w[k+1]
               + m->final_ln[k+2]*w[k+2] + m->final_ln[k+3]*w[k+3]
               + m->final_ln[k+4]*w[k+4] + m->final_ln[k+5]*w[k+5]
               + m->final_ln[k+6]*w[k+6] + m->final_ln[k+7]*w[k+7];
        for (int k = (n/8)*8; k < n; k++)
            s += m->final_ln[k] * w[k];
        g_sctx.logits[j] = s;
    }

    /* Advance circular buffer pointer */
    g_sctx.kv_pos = (g_sctx.kv_pos + 1) % ctx;
    g_sctx.total_pos++;
    return g_sctx.logits;
}

/* ─── Configure Sliding Window at Runtime ───────────────────────── */
void model_set_sliding_window(Model *m, int window, int n_sinks) {
    m->cfg.sliding_window = window;
    m->cfg.n_sinks = n_sinks;
    printf("[*] sliding window configured: W=%d, sinks=%d (effective context: %d)\n",
           window, n_sinks, window + n_sinks);
}
