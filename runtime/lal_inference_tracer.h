/* lal_inference_tracer.h — LAL Whitebox Inference Circuit Tracer
 *
 * LAL的推理也是白箱。直接追踪推理时每一层逻辑电路的激活路径。
 * 使用模型自身的推理路径(model_stateful_forward_sliding)，然后
 * 检查缓存的激活值(m->acts)来显示逻辑电路状态。
 */
#ifndef LAL_INFERENCE_TRACER_H
#define LAL_INFERENCE_TRACER_H

#include "lal_runtime.h"
#include <math.h>

/* ============================================================
 * Trace one BinLayer's output: show top-k CORE/BINARY activations
 * ============================================================ */
static void trace_binlayer(const char *name, const float *y,
                           const BinLayer *bl, int top_k) {
    int out = bl->out_dim;
    if (!bl->logic_mask) return;

    int n_core = 0, n_bin = 0, n_prune = 0;
    float core_pos = 0, core_neg = 0, bin_pos = 0, bin_neg = 0;
    float prune_max = 0;

    for (int j = 0; j < out; j++) {
        float ay = y[j];
        switch (bl->logic_mask[j]) {
        case 0: n_core++;
            if (ay > 0) core_pos += ay; else core_neg += ay;
            break;
        case 1: n_bin++;
            if (ay > 0) bin_pos += ay; else bin_neg += ay;
            break;
        case 2: n_prune++;
            if (fabsf(ay) > prune_max) prune_max = fabsf(ay);
            break;
        }
    }

    printf("    %s: CORE=%d(+%.3f -%.3f) BIN=%d(+%.3f -%.3f) PRUNE=%d(max%.4f)\n",
           name, n_core, core_pos, core_neg,
           n_bin, bin_pos, bin_neg, n_prune, prune_max);
}

/* ============================================================
 * COMPACT real inference trace — for frequent training monitoring.
 *
 * This is the TRUE whitebox: runs model_stateful_forward_sliding
 * (the actual inference path), then reads m->acts (real cached
 * activations). NOT a separate matmul approximation.
 * ============================================================ */
static float inference_trace_compact(Model *m, const char *prompt,
                                     const char *label) {
    int n = m->cfg.n_embd;
    int nL = m->cfg.n_layer;
    int V = m->cfg.vocab_size;
    int mlp_dim = m->cfg.mlp_dim;

    int tokens[512];
    int n_tok = 0;

    /* BUG #26 FIX: use BPE tokenization (prompt_tokenize) when vocab > 256.
     * Old code treated prompt as raw UTF-8 bytes — for Chinese prompt "热"
     * (3 UTF-8 bytes) it produced [231, 131, 173] (byte-fallback tokens)
     * instead of [32226] (the single BPE token for 热). The model then
     * processed 3 wrong tokens, and the trace showed garbage activations.
     *
     * prompt_tokenize is declared static in ste_train.c (the only file that
     * includes this header), so it's visible here via forward declaration. */
    if (V > 256) {
        n_tok = prompt_tokenize(prompt, tokens, 512);
    }
    if (n_tok == 0) {
        /* Byte-level fallback (byte mode or BPE tokenization failed) */
        for (int i = 0; prompt[i] && n_tok < 512; i++)
            tokens[n_tok++] = (unsigned char)prompt[i];
    }
    if (n_tok == 0) return -1.0f;

    /* BUG #27 FIX: use g_use_pure_float=1 to match training mode.
     * Old code set g_use_pure_float=0, switching to binary forward
     * (sign(w) * alpha * XNOR popcount). The activations seen by the
     * trace were from a completely different forward path than training
     * (which uses pure_float=1). CORE/BINARY values were meaningless.
     *
     * BUG #28 FIX: save/restore global flags. Old code set globals but
     * never restored them → after trace at step 50, g_use_pure_float
     * stayed at 0 for the rest of training, corrupting all subsequent
     * forward passes (binary mode instead of pure float). */
    int saved_pure_float = g_use_pure_float;
    int saved_real_attn = g_use_real_attention;
    int saved_ste = g_use_ste;

    g_use_real_attention = 1;
    g_use_pure_float = 1;  /* match training mode */
    g_use_ste = 1;

    model_stateful_begin(m);
    model_set_sliding_window(m, m->cfg.sliding_window, m->cfg.n_sinks);

    const float *logits = NULL;
    for (int i = 0; i < n_tok; i++)
        logits = model_stateful_forward_sliding(m, tokens[i]);

    /* BUG #28 FIX: restore flags even on early return */
    if (!logits) {
        g_use_pure_float = saved_pure_float;
        g_use_real_attention = saved_real_attn;
        g_use_ste = saved_ste;
        return -1.0f;
    }

    /* Read REAL activations from m->acts */
    printf("  [TRACE] \"%s\": ", label);

    int sample_layers[] = {0, nL/2, nL-1};
    int n_sample = (nL <= 3) ? nL : 3;

    for (int si = 0; si < n_sample; si++) {
        int l = sample_layers[si];
        TransAct *act = &m->acts[l];
        TransLayer *tl = &m->layers[l];

        float core_pos = 0, core_neg = 0, bin_pos = 0, bin_neg = 0;
        float prune_max = 0;
        int nc = 0, nb = 0, np = 0;
        uint8_t *mask = tl->mlp_gate.logic_mask;

        if (mask) {
            for (int j = 0; j < mlp_dim; j++) {
                float h = act->mlp_hidden[j];
                switch (mask[j]) {
                case 0:
                    if (h > 0) core_pos += h; else core_neg += h;
                    nc++;
                    break;
                case 1:
                    if (h > 0) bin_pos += h; else bin_neg += h;
                    nb++;
                    break;
                case 2:
                    if (fabsf(h) > prune_max) prune_max = fabsf(h);
                    np++;
                    break;
                }
            }
        }

        float xn = 0;
        for (int i = 0; i < n; i++)
            xn += act->x_pre_norm1[i] * act->x_pre_norm1[i];
        xn = sqrtf(xn);

        printf("L%d[||x||=%.2f C%d(+%.2f-%.2f) B%d(+%.2f-%.2f) P%d(max%.3f)] ",
               l, xn, nc, core_pos, core_neg, nb, bin_pos, bin_neg, np, prune_max);
    }

    /* Logit entropy from REAL forward pass */
    float lmax = -1e10f;
    for (int j = 0; j < V; j++)
        if (logits[j] > lmax) lmax = logits[j];
    float esum = 0;
    for (int j = 0; j < V; j++) esum += expf(logits[j] - lmax);
    float entropy = 0;
    for (int j = 0; j < V; j++) {
        float p = expf(logits[j] - lmax) / esum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    float eb = entropy / logf(2.0f);
    float mb = logf(V) / logf(2.0f);

    int top3[3] = {-1,-1,-1};
    for (int rank = 0; rank < 3; rank++) {
        int best = -1; float bv = -1e10f;
        for (int j = 0; j < V; j++) {
            int skip = 0;
            for (int r = 0; r < rank; r++)
                if (top3[r] == j) { skip = 1; break; }
            if (skip) continue;
            if (logits[j] > bv) { bv = logits[j]; best = j; }
        }
        top3[rank] = best;
    }

    printf("| H=%.2f/%.0f top3=", eb, mb);
    for (int r = 0; r < 3; r++)
        printf("%d ", top3[r]);
    printf("\n");

    /* BUG #28 FIX: restore global flags to pre-trace state */
    g_use_pure_float = saved_pure_float;
    g_use_real_attention = saved_real_attn;
    g_use_ste = saved_ste;

    return eb;
}

/* ============================================================
 * COMPACT compare: trace two concepts through REAL forward path,
 * show per-layer CORE/BINARY differentiation from m->acts.
 * ============================================================ */
static void inference_compare_compact(Model *m, const char *pa, const char *la,
                                      const char *pb, const char *lb) {
    int n = m->cfg.n_embd;
    int nL = m->cfg.n_layer;
    int mlp_dim = m->cfg.mlp_dim;
    int V = m->cfg.vocab_size;

    const char *prompts[] = {pa, pb};
    const char *labels[] = {la, lb};

    float layer_core[2][32];
    float layer_bin[2][32];
    float layer_xnorm[2][32];

    /* BUG #27/#28 FIX: save flags, use pure_float=1 (match training), restore after */
    int saved_pure_float = g_use_pure_float;
    int saved_real_attn = g_use_real_attention;
    int saved_ste = g_use_ste;

    g_use_real_attention = 1;
    g_use_pure_float = 1;  /* match training mode */
    g_use_ste = 1;

    for (int inp = 0; inp < 2; inp++) {
        int tokens[512];
        int n_tok = 0;
        /* BUG #26 FIX: use BPE tokenization when vocab > 256 */
        if (V > 256) {
            n_tok = prompt_tokenize(prompts[inp], tokens, 512);
        }
        if (n_tok == 0) {
            for (int i = 0; prompts[inp][i] && n_tok < 512; i++)
                tokens[n_tok++] = (unsigned char)prompts[inp][i];
        }

        model_stateful_begin(m);
        model_set_sliding_window(m, m->cfg.sliding_window, m->cfg.n_sinks);

        const float *logits = NULL;
        for (int i = 0; i < n_tok; i++)
            logits = model_stateful_forward_sliding(m, tokens[i]);
        if (!logits) continue;

        for (int l = 0; l < nL && l < 32; l++) {
            TransAct *act = &m->acts[l];
            TransLayer *tl = &m->layers[l];
            uint8_t *mask = tl->mlp_gate.logic_mask;

            float xn = 0, ca = 0, ba = 0;
            int nc = 0, nb = 0;
            for (int i = 0; i < n; i++)
                xn += act->x_pre_norm1[i] * act->x_pre_norm1[i];
            layer_xnorm[inp][l] = sqrtf(xn);

            if (mask) {
                for (int j = 0; j < mlp_dim; j++) {
                    float h = act->mlp_hidden[j];
                    if (mask[j] == 0) { ca += fabsf(h); nc++; }
                    else if (mask[j] == 1) { ba += fabsf(h); nb++; }
                }
            }
            layer_core[inp][l] = nc > 0 ? ca / nc : 0;
            layer_bin[inp][l] = nb > 0 ? ba / nb : 0;
        }
    }

    /* BUG #28 FIX: restore global flags */
    g_use_pure_float = saved_pure_float;
    g_use_real_attention = saved_real_attn;
    g_use_ste = saved_ste;

    printf("  [CMP] %s vs %s:\n", la, lb);
    printf("  [CMP]  L  | CORE_a  CORE_b  diff  | BIN_a   BIN_b   diff  | ||x||_a ||x||_b\n");
    for (int l = 0; l < nL && l < 32; l++) {
        float cd = fabsf(layer_core[0][l] - layer_core[1][l]);
        float bd = fabsf(layer_bin[0][l] - layer_bin[1][l]);
        printf("  [CMP] L%-2d | %.4f  %.4f  %.4f | %.4f  %.4f  %.4f | %.3f   %.3f\n",
               l, layer_core[0][l], layer_core[1][l], cd,
               layer_bin[0][l], layer_bin[1][l], bd,
               layer_xnorm[0][l], layer_xnorm[1][l]);
    }
    printf("  [CMP] CORE diff should > BIN diff for differentiation\n");
}

/* ============================================================
 * Full whitebox inference trace for one input.
 * ============================================================ */
static float inference_circuit_trace(Model *m, const char *prompt,
                                     const char *label) {
    int n = m->cfg.n_embd;
    int nL = m->cfg.n_layer;
    int V = m->cfg.vocab_size;
    int mlp_dim = m->cfg.mlp_dim;

    int tokens[512];
    int n_tok = 0;
    /* BUG #26 FIX: use BPE tokenization when vocab > 256 */
    if (V > 256) {
        n_tok = prompt_tokenize(prompt, tokens, 512);
    }
    if (n_tok == 0) {
        for (int i = 0; prompt[i] && n_tok < 512; i++)
            tokens[n_tok++] = (unsigned char)prompt[i];
    }
    if (n_tok == 0) return -1.0f;

    printf("\n  === Trace: \"%s\" (%d tokens) ===\n", label, n_tok);

    /* BUG #27/#28 FIX: save flags, use pure_float=1 (match training), restore after */
    int saved_pure_float = g_use_pure_float;
    int saved_real_attn = g_use_real_attention;
    int saved_ste = g_use_ste;

    g_use_real_attention = 1;
    g_use_pure_float = 1;  /* match training mode */
    g_use_ste = 1;

    model_stateful_begin(m);
    model_set_sliding_window(m, m->cfg.sliding_window, m->cfg.n_sinks);

    const float *logits = NULL;
    for (int i = 0; i < n_tok; i++)
        logits = model_stateful_forward_sliding(m, tokens[i]);

    /* BUG #28 FIX: restore flags even on early return */
    if (!logits) {
        g_use_pure_float = saved_pure_float;
        g_use_real_attention = saved_real_attn;
        g_use_ste = saved_ste;
        return -1.0f;
    }

    printf("  [INPUT] last token=%d\n", tokens[n_tok-1]);

    for (int l = 0; l < nL; l++) {
        TransAct *act = &m->acts[l];
        TransLayer *tl = &m->layers[l];

        float xn = 0;
        for (int i = 0; i < n; i++)
            xn += act->x_pre_norm1[i] * act->x_pre_norm1[i];
        xn = sqrtf(xn);

        float abs_sum = 0;
        for (int i = 0; i < n; i++) abs_sum += fabsf(act->norm1_out[i]);
        float K = abs_sum / n;
        if (K > 1.0f) K = 1.0f;

        printf("\n  --- L%d: ||x||=%.4f K=%.4f ---\n", l, xn, K);
        trace_binlayer("GATE", act->mlp_hidden, &tl->mlp_gate, 3);

        int n_active = 0;
        float active_sum = 0;
        for (int i = 0; i < mlp_dim; i++) {
            if (act->mlp_hidden[i] > 0.01f) {
                n_active++;
                active_sum += act->mlp_hidden[i];
            }
        }
        printf("    GELU: %d/%d active (%.1f%%) sum=%.3f\n",
               n_active, mlp_dim, 100.0f*n_active/mlp_dim, active_sum);
    }

    float lmax = -1e10f;
    for (int j = 0; j < V; j++)
        if (logits[j] > lmax) lmax = logits[j];
    float esum = 0;
    for (int j = 0; j < V; j++) esum += expf(logits[j] - lmax);
    float entropy = 0;
    for (int j = 0; j < V; j++) {
        float p = expf(logits[j] - lmax) / esum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    float eb = entropy / logf(2.0f);
    float mb = logf(V) / logf(2.0f);

    printf("\n  [OUT] entropy=%.2f/%.2f bits (%.1f%% random)\n", eb, mb, 100.0f*eb/mb);

    /* BUG #28 FIX: restore global flags */
    g_use_pure_float = saved_pure_float;
    g_use_real_attention = saved_real_attn;
    g_use_ste = saved_ste;

    return eb;
}

#endif /* LAL_INFERENCE_TRACER_H */
