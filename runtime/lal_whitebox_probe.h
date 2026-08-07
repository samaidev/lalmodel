/* lal_whitebox_probe.h — Whitebox Semantic Probe for LAL
 *
 * LAL is transparent: CORE/BINARY/PRUNE structure is directly observable.
 * This probe NEVER calls model_forward — it reads weights and embeddings
 * directly, so it works in ANY training mode without state conflicts.
 *
 * What we measure:
 * 1. Embedding concept boundaries (wte direct read)
 * 2. Weight structure per category (w_float + logic_mask)
 * 3. Simulated CORE activation: emb * W for concept pairs
 *
 * This is the correct LAL way: observe the logic circuit, not the loss.
 */
#ifndef LAL_WHITEBOX_PROBE_H
#define LAL_WHITEBOX_PROBE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declaration */
struct Model;

/* ========================================================================
 * Concept Probe Definitions
 * ======================================================================== */
typedef struct {
    const char *name_a;
    const char *bytes_a;
    const char *name_b;
    const char *bytes_b;
    const char *category;
} ConceptPair;

static ConceptPair probe_pairs[] = {
    {"\xe7\x83\xad", "\xe7\x83\xad", "\xe5\x86\xb7", "\xe5\x86\xb7", "temperature"},
    {"\xe5\xa4\xa7", "\xe5\xa4\xa7", "\xe5\xb0\x8f", "\xe5\xb0\x8f", "size"},
    {"\xe4\xb8\x8a", "\xe4\xb8\x8a", "\xe4\xb8\x8b", "\xe4\xb8\x8b", "position"},
    {"\xe4\xba\xae", "\xe4\xba\xae", "\xe6\x9a\x97", "\xe6\x9a\x97", "light"},
    {"\xe9\x87\x8d", "\xe9\x87\x8d", "\xe8\xbd\xbb", "\xe8\xbd\xbb", "weight"},
    {"\xe5\xbf\xab", "\xe5\xbf\xab", "\xe6\x85\xa2", "\xe6\x85\xa2", "speed"},
    {"\xe6\xb9\xbf", "\xe6\xb9\xbf", "\xe5\xb9\xb2", "\xe5\xb9\xb2", "moisture"},
};
#define N_PROBE_PAIRS (sizeof(probe_pairs) / sizeof(probe_pairs[0]))

/* ========================================================================
 * Cosine similarity
 * ======================================================================== */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0;
    return dot / (sqrtf(na) * sqrtf(nb));
}

/* ========================================================================
 * Get concept embedding: sum of byte token embeddings
 * Direct wte read — no model_forward needed.
 * ======================================================================== */
/* get_concept_embedding: lookup wte for a concept.
 * For BPE mode (vocab > 256), use a hash→token-id mapping table
 * that maps UTF-8 concept strings to their BPE token ids.
 * For byte-level mode (vocab == 256), average UTF-8 byte embeddings. */

/* BPE token ids for probe concepts (verified via sentencepiece).
 * Most common Chinese chars have a single BPE token (e.g. 热=32226).
 * But 轻/湿/干 fall back to byte-level encoding (3 byte-fallback tokens each)
 * because they're not in the BPE vocab as single tokens.
 *
 * BUG #18 FIX: Previously only the FIRST byte-fallback token was stored,
 * so get_concept_embedding returned the embedding of <0xE8>/<0xE6>/<0xE5>
 * (shared by ALL chars starting with that byte) instead of the actual char.
 * This corrupted 2/7 probe pairs (weight: 重 vs 轻; moisture: 湿 vs 干)
 * AND the logic_guided_regularization gradient signal for those pairs.
 *
 * Fix: store up to 4 token ids per concept (n_ids = how many to sum).
 * For single-token concepts, n_ids=1. For byte-fallback, n_ids=3. */
typedef struct { const char *utf8; int n_ids; int bpe_ids[4]; } BpeTokenMap;
/* v14: updated for chinese_bpe_v2 tokenizer (vocab=12227, unigram model).
 * New tokenizer prepends ▁ (id=259) to word-initial characters.
 * All byte-fallback issues eliminated (0.3% vs 24.1% in old tokenizer). */
static BpeTokenMap bpe_token_map[] = {
    {"\xe7\x83\xad", 2, {259, 1686}},             /* 热 = ▁热 */
    {"\xe5\x86\xb7", 2, {259, 2187}},             /* 冷 = ▁冷 */
    {"\xe5\xa4\xa7", 2, {259, 615}},              /* 大 = ▁大 */
    {"\xe5\xb0\x8f", 2, {259, 350}},              /* 小 = ▁小 */
    {"\xe4\xb8\x8a", 2, {259, 2416}},             /* 上 = ▁上 */
    {"\xe4\xb8\x8b", 2, {259, 1299}},             /* 下 = ▁下 */
    {"\xe4\xba\xae", 2, {259, 2616}},             /* 亮 = ▁亮 */
    {"\xe6\x9a\x97", 2, {259, 2397}},             /* 暗 = ▁暗 */
    {"\xe9\x87\x8d", 2, {259, 1047}},             /* 重 = ▁重 */
    {"\xe8\xbd\xbb", 2, {259, 1672}},             /* 轻 = ▁轻 */
    {"\xe5\xbf\xab", 2, {259, 703}},              /* 快 = ▁快 */
    {"\xe6\x85\xa2", 2, {259, 1067}},             /* 慢 = ▁慢 */
    {"\xe6\xb9\xbf", 2, {259, 2667}},             /* 湿 = ▁湿 */
    {"\xe5\xb9\xb2", 2, {259, 2594}},             /* 干 = ▁干 */
    {"\xe7\x81\xab", 2, {259, 1164}},             /* 火 = ▁火 */
    {"\xe6\xb0\xb4", 2, {259, 962}},              /* 水 = ▁水 */
    {"\xe5\xb1\xb1", 2, {259, 1206}},             /* 山 = ▁山 */
    {"\xe8\x8a\xb1", 2, {259, 438}},              /* 花 = ▁花 */
    {"\xe6\xa0\x91", 2, {259, 1024}},             /* 树 = ▁树 */
    {"\xe9\xb8\x9f", 2, {259, 1051}},             /* 鸟 = ▁鸟 */
    {"\xe9\xb1\xbc", 2, {259, 1149}},             /* 鱼 = ▁鱼 */
    {"\xe4\xba\xba", 2, {259, 1159}},             /* 人 = ▁人 */
    {"\xe5\xa4\xa9", 2, {259, 2058}},             /* 天 = ▁天 */
    {"\xe5\x9c\xb0", 2, {259, 1295}},             /* 地 = ▁地 */
    {"\xe6\x9c\x88", 2, {259, 2194}},             /* 月 = ▁月 */
    {"\xe6\x98\x9f", 2, {259, 5733}},             /* 星 = ▁星 */
    {"\xe9\xa3\x8e", 2, {259, 1104}},             /* 风 = ▁风 */
    {"\xe9\x9b\xa8", 2, {259, 1232}},             /* 雨 = ▁雨 */
    {"\xe4\xba\x91", 2, {259, 1474}},             /* 云 = ▁云 */
    {"\xe5\xa4\xaa\xe9\x98\xb3", 1, {2981}},      /* 太阳 = ▁太阳 (single token) */
    {"\xe5\xa4\xaa", 2, {259, 9464}},             /* 太 = ▁太 */
    {"\xe9\x98\xb3", 2, {259, 2811}},             /* 阳 = ▁阳 */
    {"\xe7\x8c\xab", 1, {3457}},                  /* 猫 = ▁猫 (single token) */
    {"\xe8\x8b\xb9\xe6\x9e\x9c", 2, {259, 427}},  /* 苹果 = ▁苹果 */
    {"\xe5\x8a\xa8\xe7\x89\xa9", 2, {259, 2249}}, /* 动物 = ▁动物 */
    {"\xe6\xa4\x8d\xe7\x89\xa9", 2, {259, 2721}}, /* 植物 = ▁植物 */
    {"\xe7\xba\xa2", 2, {259, 1278}},             /* 红 = ▁红 */
    {"\xe9\xbb\x84", 2, {259, 1635}},             /* 黄 = ▁黄 */
    {"\xe8\x93\x9d", 2, {259, 2054}},             /* 蓝 = ▁蓝 */
    {"\xe7\xbb\xbf", 2, {259, 1715}},             /* 绿 = ▁绿 */
    {"\xe5\x85\x89", 2, {259, 2538}},             /* 光 = ▁光 */
    {"\xe6\xb8\xa9", 1, {11229}},                 /* 温 = ▁温 (single token) */
    {"\xe5\x86\xb0", 1, {3467}},                  /* 冰 = ▁冰 (single token) */
    {"\xe9\x9b\xaa", 2, {259, 1191}},             /* 雪 = ▁雪 */
};
#define N_BPE_MAP (sizeof(bpe_token_map) / sizeof(bpe_token_map[0]))

static void get_concept_embedding(Model *m, const char *utf8_bytes,
                                   float *out, int n_embd) {
    memset(out, 0, n_embd * sizeof(float));

    if (m->cfg.vocab_size > 256) {
        /* BPE mode: look up token id sequence from mapping table.
         * For single-token concepts: copy that token's wte row.
         * For byte-fallback concepts (轻/湿/干): SUM all byte-fallback
         *   token embeddings. Sum (not average) matches how the model
         *   embeds a multi-token sequence at the input layer. */
        BpeTokenMap *entry = NULL;
        for (int i = 0; i < (int)N_BPE_MAP; i++) {
            if (strcmp(utf8_bytes, bpe_token_map[i].utf8) == 0) {
                entry = &bpe_token_map[i];
                break;
            }
        }
        if (entry) {
            for (int t = 0; t < entry->n_ids; t++) {
                int tok = entry->bpe_ids[t];
                if (tok < 0 || tok >= m->cfg.vocab_size) continue;
                const float *row = m->wte + (size_t)tok * n_embd;
                for (int j = 0; j < n_embd; j++)
                    out[j] += row[j];
            }
        } else {
            /* BUG #36 FIX: concept not in map → fall back to byte-fallback
             * tokens (SentencePiece convention: id = 3 + byte_value).
             * This handles any UTF-8 string, not just the 14 hardcoded ones.
             * Sum (not average) all byte-fallback embeddings. */
            for (int i = 0; utf8_bytes[i]; i++) {
                int tok = 3 + (unsigned char)utf8_bytes[i];
                if (tok < m->cfg.vocab_size) {
                    const float *row = m->wte + (size_t)tok * n_embd;
                    for (int j = 0; j < n_embd; j++)
                        out[j] += row[j];
                }
            }
        }
    } else {
        /* Byte-level mode: average UTF-8 byte embeddings */
        int n_bytes = 0;
        for (int i = 0; utf8_bytes[i]; i++) {
            int tok = (unsigned char)utf8_bytes[i];
            if (tok < m->cfg.vocab_size) {
                float *row = m->wte + (size_t)tok * n_embd;
                for (int j = 0; j < n_embd; j++)
                    out[j] += row[j];
                n_bytes++;
            }
        }
        if (n_bytes > 0) {
            for (int j = 0; j < n_embd; j++)
                out[j] /= n_bytes;
        }
    }
}

/* ========================================================================
 * Compute the input to mlp_gate at target_layer for a single concept.
 *
 * BUG #19 FIX: Previously simulate_activation computed W_gate * raw_emb,
 * completely skipping the LayerNorm + attention + residual path that the
 * real forward applies before mlp_gate. The reported CORE/BINARY diff
 * metrics and the logic_guided_regularization gradient were both computed
 * on raw wte values, which have a totally different distribution from
 * the LN'd, attention-mixed, residual-added activations the model sees.
 *
 * This function does a proper single-position forward through layers
 * 0..target_layer using V-copy attention (matches training default
 * g_use_real_attention=0). Uses scratch buffers — does NOT touch m->acts.
 *
 * For single-token concepts: initial_emb = wte[token].
 * For multi-byte concepts (轻/湿/干): initial_emb = sum of byte-fallback
 *   wte rows (approximation — model never sees this exact vector, but
 *   it captures the concept's direction in wte space).
 * ======================================================================== */
static void compute_gate_input(Model *m, const float *initial_emb,
                                int target_layer, float *out, int n_embd) {
    int n_layer = m->cfg.n_layer;
    int mlp_dim = m->cfg.mlp_dim;
    int n = n_embd;
    float rs = m->cfg.residual_scale;

    /* Static scratch buffers — allocated once, reused across calls.
     * PERFORMANCE FIX: logic_guided_regularization calls this 112 times per
     * training step (8 layers × 7 pairs × 2 concepts). Old code did 11
     * malloc+free per call = 1232 heap ops/step. Now zero heap ops after
     * first call. Buffers grow if model size increases (checked at runtime). */
    static float *x = NULL, *norm1 = NULL, *norm2 = NULL;
    static float *qkv_buf = NULL, *q = NULL, *v = NULL;
    static float *attn_out = NULL, *proj_out = NULL;
    static float *gate_buf = NULL, *up_buf = NULL, *hidden = NULL, *mlp_out = NULL;
    static int s_n = 0, s_mlp = 0, s_qkv = 0;

    int qkv_size = m->cfg.qkv_merged ? 3 * n : 0;
    if (s_n != n || s_mlp != mlp_dim || s_qkv != qkv_size) {
        free(x); free(norm1); free(norm2);
        free(qkv_buf); free(q); free(v);
        free(attn_out); free(proj_out);
        free(gate_buf); free(up_buf); free(hidden); free(mlp_out);
        x = malloc(n * sizeof(float));
        norm1 = malloc(n * sizeof(float));
        norm2 = malloc(n * sizeof(float));
        qkv_buf = qkv_size > 0 ? malloc(qkv_size * sizeof(float)) : NULL;
        q = qkv_size > 0 ? NULL : malloc(n * sizeof(float));
        v = qkv_size > 0 ? (qkv_buf + 2 * n) : malloc(n * sizeof(float));
        attn_out = malloc(n * sizeof(float));
        proj_out = malloc(n * sizeof(float));
        gate_buf = malloc(mlp_dim * sizeof(float));
        up_buf = malloc(mlp_dim * sizeof(float));
        hidden = malloc(mlp_dim * sizeof(float));
        mlp_out = malloc(n * sizeof(float));
        s_n = n; s_mlp = mlp_dim; s_qkv = qkv_size;
    }

    /* x = initial_emb + wpe[0] */
    memcpy(x, initial_emb, n * sizeof(float));
    if (m->wpe) {
        const float *wpe0 = &m->wpe[0];  /* position 0 */
        for (int i = 0; i < n; i++) x[i] += wpe0[i];
    }

    for (int l = 0; l <= target_layer && l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];

        /* LN1: norm1 = LN1(x) */
        norm_forward(norm1, x, tl->norm1_w, tl->norm1_b, m->cfg.norm_type, n);

        /* Q, V projections. For qkv_merged, attn_q produces [Q|K|V] (3n);
         * V is at offset 2n. For non-merged, call attn_q and attn_v
         * separately (K not needed for V-copy attention).
         * v13l: When g_skip_wv is set, skip Q/V projections entirely and
         * use norm1 directly as attn_out (matches training forward). */
        if (g_skip_wv) {
            memcpy(attn_out, norm1, n * sizeof(float));
        } else if (m->cfg.qkv_merged) {
            bin_forward_pure_float(qkv_buf, norm1, &tl->attn_q);
            /* v already points to qkv_buf + 2*n */
            memcpy(attn_out, v, n * sizeof(float));
        } else {
            bin_forward_pure_float(q, norm1, &tl->attn_q);  /* unused but matches forward */
            bin_forward_pure_float(v, norm1, &tl->attn_v);
            memcpy(attn_out, v, n * sizeof(float));
        }

        /* v13l: When g_skip_wv, attn_out = norm1 (no V-copy needed) */

        /* Output projection + residual */
        bin_forward_pure_float(proj_out, attn_out, &tl->attn_o);
        /* v13j: Proportional attention scaling — 15% of residual norm */
        {
            float xn = 0, pn = 0;
            for (int i = 0; i < n; i++) { xn += x[i] * x[i]; pn += proj_out[i] * proj_out[i]; }
            xn = sqrtf(xn) + 1e-8f;
            pn = sqrtf(pn) + 1e-8f;
            float target = 0.15f * xn;
            float as = target / pn;
            for (int i = 0; i < n; i++) x[i] += rs * as * proj_out[i];
        }
        clip_array(x, n, 10.0f);

        /* LN2: norm2 = LN2(x) — this is the mlp_gate input */
        norm_forward(norm2, x, tl->norm2_w, tl->norm2_b, m->cfg.norm_type, n);

        if (l == target_layer) {
            memcpy(out, norm2, n * sizeof(float));
            break;
        }

        /* MLP (needed to feed next layer) */
        if (m->cfg.act_type == ACT_SWIGLU) {
            bin_forward_pure_float(gate_buf, norm2, &tl->mlp_gate);
            bin_forward_pure_float(up_buf,   norm2, &tl->mlp_up);
            for (int i = 0; i < mlp_dim; i++) hidden[i] = silu(gate_buf[i]) * up_buf[i];
        } else {
            bin_forward_pure_float(hidden, norm2, &tl->mlp_gate);
            for (int i = 0; i < mlp_dim; i++) hidden[i] = gelu(hidden[i]);
        }
        bin_forward_pure_float(mlp_out, hidden, &tl->mlp_down);
        /* v13j: Proportional MLP scaling — 25% of residual norm + normalize_residual(6.0) */
        {
            float xn = 0, mns = 0;
            for (int i = 0; i < n; i++) { xn += x[i] * x[i]; mns += mlp_out[i] * mlp_out[i]; }
            float xn_norm = sqrtf(xn) + 1e-8f;
            float mn = sqrtf(mns) + 1e-8f;
            float cap = 0.25f * xn_norm;
            float ms = (mn > cap) ? (cap / mn) : 1.0f;
            for (int i = 0; i < n; i++) x[i] += rs * ms * mlp_out[i];
        }
        normalize_residual(x, n, 6.0f);
    }
    /* Static buffers — NOT freed here (reused on next call).
     * They persist for the program lifetime and are auto-freed on exit. */
}

/* ========================================================================
 * BUG #35 FIX: Compute gate_inputs for ALL layers in ONE forward pass.
 *
 * Old logic_guided_regularization called compute_gate_input(l) for each
 * layer l=0..n_layer-1. Each call runs forward 0..l, so total forwards =
 * 1+2+...+n_layer = O(n_layer²). For n_layer=8: 36 forwards/concept.
 *
 * This function does a SINGLE forward pass through all layers, capturing
 * the LN2 output (gate_input) at each layer. Result: n_layer forwards
 * instead of n_layer*(n_layer+1)/2. For n_layer=8: 8 instead of 36 (4.5x).
 *
 * out_gate_inputs: [n_layer * n_embd] — gate_input (norm2_out) for each layer.
 *
 * v13 EXTENSION: out_norm1_inputs (optional, may be NULL).
 *   [n_layer * n_embd] — norm1_out (input to attn_q) for each layer.
 *   Used by attention regularization to differentiate concept pairs at
 *   the attention output (attn_o). Without this, attention gets no
 *   logic-guided gradient and stays at random init (v12 root cause #3).
 * ======================================================================== */
static void compute_all_gate_inputs(Model *m, const float *initial_emb,
                                     float *out_gate_inputs, int n_embd,
                                     float *out_norm1_inputs, /* NULL ok */
                                     float *out_final_hidden   /* NULL ok, v13g: for logit diversity loss */) {
    int n_layer = m->cfg.n_layer;
    int mlp_dim = m->cfg.mlp_dim;
    int n = n_embd;
    float rs = m->cfg.residual_scale;

    /* Reuse the same static buffers as compute_gate_input */
    static float *x = NULL, *norm1 = NULL, *norm2 = NULL;
    static float *qkv_buf = NULL, *q = NULL, *v = NULL;
    static float *attn_out = NULL, *proj_out = NULL;
    static float *gate_buf = NULL, *up_buf = NULL, *hidden = NULL, *mlp_out = NULL;
    static int s_n = 0, s_mlp = 0, s_qkv = 0;

    int qkv_size = m->cfg.qkv_merged ? 3 * n : 0;
    if (s_n != n || s_mlp != mlp_dim || s_qkv != qkv_size) {
        free(x); free(norm1); free(norm2);
        free(qkv_buf); free(q); free(v);
        free(attn_out); free(proj_out);
        free(gate_buf); free(up_buf); free(hidden); free(mlp_out);
        x = malloc(n * sizeof(float));
        norm1 = malloc(n * sizeof(float));
        norm2 = malloc(n * sizeof(float));
        qkv_buf = qkv_size > 0 ? malloc(qkv_size * sizeof(float)) : NULL;
        q = qkv_size > 0 ? NULL : malloc(n * sizeof(float));
        v = qkv_size > 0 ? (qkv_buf + 2 * n) : malloc(n * sizeof(float));
        attn_out = malloc(n * sizeof(float));
        proj_out = malloc(n * sizeof(float));
        gate_buf = malloc(mlp_dim * sizeof(float));
        up_buf = malloc(mlp_dim * sizeof(float));
        hidden = malloc(mlp_dim * sizeof(float));
        mlp_out = malloc(n * sizeof(float));
        s_n = n; s_mlp = mlp_dim; s_qkv = qkv_size;
    }

    memcpy(x, initial_emb, n * sizeof(float));
    if (m->wpe) {
        const float *wpe0 = &m->wpe[0];
        for (int i = 0; i < n; i++) x[i] += wpe0[i];
    }

    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];

        norm_forward(norm1, x, tl->norm1_w, tl->norm1_b, m->cfg.norm_type, n);

        /* v13: cache norm1_out for attention regularization */
        if (out_norm1_inputs)
            memcpy(&out_norm1_inputs[(size_t)l * n], norm1, n * sizeof(float));

        if (m->cfg.qkv_merged) {
            bin_forward_pure_float(qkv_buf, norm1, &tl->attn_q);
        } else {
            bin_forward_pure_float(v, norm1, &tl->attn_v);
        }
        memcpy(attn_out, v, n * sizeof(float));

        bin_forward_pure_float(proj_out, attn_out, &tl->attn_o);
        /* v13j: Proportional attention scaling — 15% of residual norm */
        {
            float xn = 0, pn = 0;
            for (int i = 0; i < n; i++) { xn += x[i] * x[i]; pn += proj_out[i] * proj_out[i]; }
            xn = sqrtf(xn) + 1e-8f;
            pn = sqrtf(pn) + 1e-8f;
            float target = 0.15f * xn;
            float as = target / pn;
            for (int i = 0; i < n; i++) x[i] += rs * as * proj_out[i];
        }
        clip_array(x, n, 10.0f);

        norm_forward(norm2, x, tl->norm2_w, tl->norm2_b, m->cfg.norm_type, n);

        /* Capture gate_input for this layer (v13: NULL ok, skip write) */
        if (out_gate_inputs)
            memcpy(&out_gate_inputs[(size_t)l * n], norm2, n * sizeof(float));

        /* MLP to feed next layer */
        if (m->cfg.act_type == ACT_SWIGLU) {
            bin_forward_pure_float(gate_buf, norm2, &tl->mlp_gate);
            bin_forward_pure_float(up_buf,   norm2, &tl->mlp_up);
            for (int i = 0; i < mlp_dim; i++) hidden[i] = silu(gate_buf[i]) * up_buf[i];
        } else {
            bin_forward_pure_float(hidden, norm2, &tl->mlp_gate);
            for (int i = 0; i < mlp_dim; i++) hidden[i] = gelu(hidden[i]);
        }
        bin_forward_pure_float(mlp_out, hidden, &tl->mlp_down);
        /* v13j: Proportional MLP scaling — 25% of residual norm + normalize_residual(6.0) */
        {
            float xn = 0, mns = 0;
            for (int i = 0; i < n; i++) { xn += x[i] * x[i]; mns += mlp_out[i] * mlp_out[i]; }
            float xn_norm = sqrtf(xn) + 1e-8f;
            float mn = sqrtf(mns) + 1e-8f;
            float cap = 0.25f * xn_norm;
            float ms = (mn > cap) ? (cap / mn) : 1.0f;
            for (int i = 0; i < n; i++) x[i] += rs * ms * mlp_out[i];
        }
        normalize_residual(x, n, 6.0f);
    }

    /* v13g: apply final LayerNorm and return final hidden state.
     * Used by logit_diversity_loss to compute logits = wte @ final_hidden
     * and penalize collapse to punctuation tokens. */
    if (out_final_hidden) {
        norm_forward(out_final_hidden, x, m->ln_f_w, m->ln_f_b, m->cfg.norm_type, n);
    }
}

/* ========================================================================
 * Simulated activation: gate_input * W for one layer's mlp_gate
 * Direct matmul using w_core (CORE) or w_float (BINARY) — no model_forward.
 *
 * NOTE: 'gate_input' should be the output of compute_gate_input() — i.e.
 * the LN2'd, attention-mixed, residual-added activation that mlp_gate
 * actually sees at runtime. Passing raw wte here will produce meaningless
 * results (this was BUG #19).
 * ======================================================================== */
static void simulate_activation(Model *m, const float *gate_input, int layer,
                                 float *out_activations, int out_dim) {
    BinLayer *fc = &m->layers[layer].mlp_gate;
    int in_dim = fc->in_dim;
    memset(out_activations, 0, out_dim * sizeof(float));

    /* Must match bin_forward_pure_float exactly:
     *   CORE: s = w_core[core_idx] · x + bias  (no core_gain, no K)
     *   BINARY: s = w_float[j] · x + bias
     *   PRUNE: 0
     * Previously used w_float for CORE (wrong!) and applied core_gain
     * (not used in pure_float mode). This caused logic_guided to compute
     * incorrect activations → semantic gradient on wrong values. */
    int core_idx = 0;
    for (int j = 0; j < out_dim; j++) {
        /* PRUNE: output 0 */
        if (fc->logic_mask && fc->logic_mask[j] == 2) {
            out_activations[j] = 0.0f;
            continue;
        }
        float s = fc->bias ? fc->bias[j] : 0;

        if (fc->logic_mask && fc->logic_mask[j] == 0) {
            /* CORE: use w_core (matches bin_forward_pure_float) */
            if (fc->w_core) {
                const float *wc = &fc->w_core[core_idx * in_dim];
                for (int i = 0; i < in_dim; i++)
                    s += gate_input[i] * wc[i];
            }
            core_idx++;
        } else {
            /* BINARY: use w_float (matches bin_forward_pure_float) */
            const float *wf = &fc->w_float[(size_t)j * in_dim];
            for (int i = 0; i < in_dim; i++)
                s += gate_input[i] * wf[i];
        }
        out_activations[j] = s;
    }
}

/* ========================================================================
 * Embedding Concept Boundary Analysis
 * Direct wte read — works at any training step.
 * ======================================================================== */
static void embedding_boundary_analysis(Model *m) {
    int n_embd = m->cfg.n_embd;
    
    printf("--- Embedding Concept Boundary ---\n\n");
    
    float *embs[32];
    const char *names[32];
    int n_concepts = 0;
    
    for (int p = 0; p < (int)N_PROBE_PAIRS && n_concepts < 32; p++) {
        ConceptPair *cp = &probe_pairs[p];
        embs[n_concepts] = (float *)malloc(n_embd * sizeof(float));
        get_concept_embedding(m, cp->bytes_a, embs[n_concepts], n_embd);
        names[n_concepts] = cp->name_a;
        n_concepts++;
        
        if (n_concepts >= 32) break;
        embs[n_concepts] = (float *)malloc(n_embd * sizeof(float));
        get_concept_embedding(m, cp->bytes_b, embs[n_concepts], n_embd);
        names[n_concepts] = cp->name_b;
        n_concepts++;
    }
    
    /* Similarity matrix */
    printf("  %-4s", "");
    for (int j = 0; j < n_concepts; j++)
        printf(" %4s", names[j]);
    printf("\n");
    
    for (int i = 0; i < n_concepts; i++) {
        printf("  %-4s", names[i]);
        for (int j = 0; j < n_concepts; j++) {
            float sim = cosine_sim(embs[i], embs[j], n_embd);
            printf(" %+.2f", sim);
        }
        printf("\n");
    }
    
    /* Opposite pairs */
    printf("\n  Opposite pairs (boundary clarity):\n");
    float avg_opp = 0;
    int n_pairs = 0;
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        int ia = p * 2, ib = p * 2 + 1;
        if (ia < n_concepts && ib < n_concepts) {
            float sim = cosine_sim(embs[ia], embs[ib], n_embd);
            printf("    %s vs %s: %+.4f %s\n",
                   names[ia], names[ib], sim,
                   sim < 0.3f ? "CLEAR" : sim < 0.7f ? "weak" : "NONE");
            avg_opp += sim;
            n_pairs++;
        }
    }
    avg_opp = n_pairs > 0 ? avg_opp / n_pairs : 0;
    printf("\n  Boundary score: %.0f/100 (%s)\n\n",
           100.0f * (1.0f - avg_opp),
           avg_opp < 0.3f ? "STRONG" : avg_opp < 0.7f ? "MODERATE" : "WEAK");
    
    for (int c = 0; c < n_concepts; c++) free(embs[c]);
}

/* ========================================================================
 * Weight Structure Analysis
 * Direct w_float + logic_mask read — no model_forward needed.
 * ======================================================================== */
typedef struct {
    float core_norm, binary_norm, prune_norm;
    int n_core, n_binary, n_prune;
    int assignment_ok;
} WeightStruct;

static WeightStruct analyze_weights(BinLayer *bl) {
    WeightStruct r;
    memset(&r, 0, sizeof(r));
    if (!bl || !bl->w_float || !bl->logic_mask) return r;
    
    int in_dim = bl->in_dim, out_dim = bl->out_dim;
    float cn = 0, bn = 0, pn = 0;

    /* BUG #16 FIX: w_float is stored as [out, in] row-major
     * (see bin_layer_init / bin_layer_init_logic in lal_runtime.c:
     *  "w_float[j*in + i] is contiguous per output j").
     * Previously this read w_float[i*out_dim + j] which is the [in, out]
     * stride -- that accesses scattered elements across multiple output
     * rows, producing meaningless "norms". The reported structure %
     * metric was effectively random. */
    for (int j = 0; j < out_dim; j++) {
        const float *wj = &bl->w_float[(size_t)j * in_dim];  /* contiguous [in] for output j */
        float ns = 0;
        for (int i = 0; i < in_dim; i++)
            ns += wj[i] * wj[i];
        float norm = sqrtf(ns);
        switch (bl->logic_mask[j]) {
            case 0: cn += norm; r.n_core++; break;
            case 1: bn += norm; r.n_binary++; break;
            case 2: pn += norm; r.n_prune++; break;
        }
    }
    r.core_norm = r.n_core > 0 ? cn / r.n_core : 0;
    r.binary_norm = r.n_binary > 0 ? bn / r.n_binary : 0;
    r.prune_norm = r.n_prune > 0 ? pn / r.n_prune : 0;
    r.assignment_ok = (r.core_norm > r.binary_norm) && (r.binary_norm > r.prune_norm);
    return r;
}

/* ========================================================================
 * Simulated CORE Activation Analysis
 * Compute gate_input * W to see which neurons fire for each concept.
 * Uses compute_gate_input() for proper LN+attn+residual forward.
 * ======================================================================== */
static void core_activation_analysis(Model *m) {
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    int n_layer = m->cfg.n_layer;

    printf("--- CORE Activation (simulated, layer 0) ---\n\n");

    /* PERF: reuse stack buffers (was malloc/free per pair = 28 heap ops) */
    float emb_a[4096], emb_b[4096];
    float gate_a[4096], gate_b[4096];
    float *act_a = (float *)malloc(mlp_dim * sizeof(float));
    float *act_b = (float *)malloc(mlp_dim * sizeof(float));

    /* For each concept pair, compute layer-0 MLP activation */
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];

        get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
        get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);

        /* BUG #19 FIX: compute proper gate_input (LN1 + attn + residual + LN2)
         * instead of feeding raw wte to mlp_gate. */
        compute_gate_input(m, emb_a, 0, gate_a, n_embd);
        compute_gate_input(m, emb_b, 0, gate_b, n_embd);

        /* Simulate: activation = gate_input * W_layer0_mlp_gate */
        simulate_activation(m, gate_a, 0, act_a, mlp_dim);
        simulate_activation(m, gate_b, 0, act_b, mlp_dim);

        uint8_t *mask = m->layers[0].mlp_gate.logic_mask;
        if (!mask) continue;

        /* Measure CORE vs BINARY differentiation */
        float core_diff = 0, bin_diff = 0, prune_diff = 0;
        float core_act = 0, bin_act = 0, prune_act = 0;
        int nc = 0, nb = 0, np = 0;

        for (int j = 0; j < mlp_dim; j++) {
            float diff = fabsf(act_a[j] - act_b[j]);
            float mag = fabsf(act_a[j]);
            switch (mask[j]) {
                case 0: core_diff += diff; core_act += mag; nc++; break;
                case 1: bin_diff += diff; bin_act += mag; nb++; break;
                case 2: prune_diff += diff; prune_act += mag; np++; break;
            }
        }
        core_diff = nc > 0 ? core_diff / nc : 0;
        bin_diff = nb > 0 ? bin_diff / nb : 0;
        core_act = nc > 0 ? core_act / nc : 0;
        bin_act = nb > 0 ? bin_act / nb : 0;
        prune_act = np > 0 ? prune_act / np : 0;

        printf("  %s vs %s: CORE diff=%.4f act=%.4f %s | BIN diff=%.4f | PRUNE act=%.4f\n",
               cp->name_a, cp->name_b,
               core_diff, core_act,
               core_diff > bin_diff ? "OK" : "X",
               bin_diff, prune_act);
    }
    free(act_a);
    free(act_b);
    printf("\n");
}

/* ========================================================================
 * Full Whitebox Probe — no model_forward, pure weight/embedding analysis
 * ======================================================================== */
static void whitebox_probe(Model *m) {
    int n_layer = m->cfg.n_layer;
    
    printf("\n=== LAL Whitebox Probe (step checkpoint) ===\n");
    printf("Model: %dL, %d embd, %d mlp\n\n",
           n_layer, m->cfg.n_embd, m->cfg.mlp_dim);
    
    /* 1. Embedding boundaries */
    embedding_boundary_analysis(m);
    
    /* 2. CORE activation simulation */
    core_activation_analysis(m);
    
    /* 3. Weight structure */
    printf("--- Weight Structure ---\n\n");
    float ok_pct = 0;
    int n_checked = 0;
    for (int l = 0; l < n_layer; l++) {
        WeightStruct wr = analyze_weights(&m->layers[l].mlp_gate);
        WeightStruct wa = analyze_weights(&m->layers[l].attn_q);
        if (l < 2 || l == n_layer - 1) {
            printf("  L%d mlp: CORE=%.4f BIN=%.4f PRUNE=%.4f %s | attn: CORE=%.4f BIN=%.4f PRUNE=%.4f %s\n",
                   l, wr.core_norm, wr.binary_norm, wr.prune_norm,
                   wr.assignment_ok ? "OK" : "X",
                   wa.core_norm, wa.binary_norm, wa.prune_norm,
                   wa.assignment_ok ? "OK" : "X");
        } else if (l == 2) {
            printf("  ... (layers 2-%d) ...\n", n_layer - 2);
        }
        if (wr.assignment_ok) ok_pct++;
        if (wa.assignment_ok) ok_pct++;
        n_checked += 2;
    }
    printf("\n  Structure: %.0f%% OK (%d/%d)\n\n",
           n_checked > 0 ? 100.0f * ok_pct / n_checked : 0,
           (int)ok_pct, n_checked);
}

/* ========================================================================
 * Concept similarity probe (kept for post-stage analysis)
 * ======================================================================== */
static void concept_similarity_probe(Model *m) {
    int n_embd = m->cfg.n_embd;

    printf("--- Concept Similarity (mid-layer weights) ---\n\n");
    
    float *embs[32];
    const char *names[32];
    int n_concepts = 0;
    
    for (int p = 0; p < (int)N_PROBE_PAIRS && n_concepts < 32; p++) {
        ConceptPair *cp = &probe_pairs[p];
        embs[n_concepts] = (float *)malloc(n_embd * sizeof(float));
        get_concept_embedding(m, cp->bytes_a, embs[n_concepts], n_embd);
        names[n_concepts] = cp->name_a;
        n_concepts++;
        if (n_concepts >= 32) break;
        embs[n_concepts] = (float *)malloc(n_embd * sizeof(float));
        get_concept_embedding(m, cp->bytes_b, embs[n_concepts], n_embd);
        names[n_concepts] = cp->name_b;
        n_concepts++;
    }
    
    printf("  Embedding similarity:\n  %-4s", "");
    for (int j = 0; j < n_concepts; j++) printf(" %4s", names[j]);
    printf("\n");
    for (int i = 0; i < n_concepts; i++) {
        printf("  %-4s", names[i]);
        for (int j = 0; j < n_concepts; j++)
            printf(" %+.2f", cosine_sim(embs[i], embs[j], n_embd));
        printf("\n");
    }
    printf("\n");
    for (int c = 0; c < n_concepts; c++) free(embs[c]);
}

/* ========================================================================
 * Compact Whitebox Probe -- for frequent monitoring (every 10 steps)
 * Outputs 3 lines of key metrics, no matrices.
 * Returns a struct so training code can make decisions.
 * ======================================================================== */
typedef struct {
    float boundary_score;     /* 0-100, higher = clearer concept boundaries */
    float core_diff_avg;      /* avg CORE neuron differentiation for opposites */
    float bin_diff_avg;       /* avg BINARY neuron differentiation */
    float prune_act_avg;      /* avg PRUNE activation (should be ~0) */
    float structure_pct;      /* % of layers with CORE>BINARY>PRUNE */
    int   n_layers_ok;
    int   n_layers_total;
    float core_bin_ratio;     /* core_diff / bin_diff, should be >1.0 */
} ProbeMetrics;

static ProbeMetrics whitebox_probe_compact(Model *m) {
    ProbeMetrics pm;
    memset(&pm, 0, sizeof(pm));

    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    int n_layer = m->cfg.n_layer;

    /* 1. Boundary score: avg cosine sim of opposite pairs */
    float avg_opp_sim = 0;
    int n_pairs = 0;
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        float ea[4096], eb[4096];
        get_concept_embedding(m, cp->bytes_a, ea, n_embd);
        get_concept_embedding(m, cp->bytes_b, eb, n_embd);
        avg_opp_sim += cosine_sim(ea, eb, n_embd);
        n_pairs++;
    }
    avg_opp_sim = n_pairs > 0 ? avg_opp_sim / n_pairs : 0;
    pm.boundary_score = 100.0f * (1.0f - avg_opp_sim);

    /* 2. CORE vs BINARY differentiation (layer 0 simulated activation).
     * BUG #19 FIX: use compute_gate_input() to get the proper LN2'd input
     * that mlp_gate actually sees at runtime, instead of raw wte.
     * PERF: reuse stack buffers (was malloc/free per pair = 28 heap ops/step). */
    float core_diff_sum = 0, bin_diff_sum = 0, prune_act_sum = 0;
    float emb_a[4096], emb_b[4096];
    float gate_a[4096], gate_b[4096];
    float *act_a = (float *)malloc(mlp_dim * sizeof(float));
    float *act_b = (float *)malloc(mlp_dim * sizeof(float));
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
        get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);

        compute_gate_input(m, emb_a, 0, gate_a, n_embd);
        compute_gate_input(m, emb_b, 0, gate_b, n_embd);

        simulate_activation(m, gate_a, 0, act_a, mlp_dim);
        simulate_activation(m, gate_b, 0, act_b, mlp_dim);

        uint8_t *mask = m->layers[0].mlp_gate.logic_mask;
        if (mask) {
            float cd = 0, bd = 0, pa = 0;
            int nc = 0, nb = 0, np = 0;
            for (int j = 0; j < mlp_dim; j++) {
                float diff = fabsf(act_a[j] - act_b[j]);
                float mag = fabsf(act_a[j]);
                switch (mask[j]) {
                    case 0: cd += diff; nc++; break;
                    case 1: bd += diff; nb++; break;
                    case 2: pa += mag;  np++; break;
                }
            }
            core_diff_sum += nc > 0 ? cd / nc : 0;
            bin_diff_sum  += nb > 0 ? bd / nb : 0;
            prune_act_sum += np > 0 ? pa / np : 0;
        }
    }
    free(act_a);
    free(act_b);
    pm.core_diff_avg = core_diff_sum / N_PROBE_PAIRS;
    pm.bin_diff_avg  = bin_diff_sum / N_PROBE_PAIRS;
    pm.prune_act_avg = prune_act_sum / N_PROBE_PAIRS;
    pm.core_bin_ratio = pm.bin_diff_avg > 1e-8f
        ? pm.core_diff_avg / pm.bin_diff_avg : 0;

    /* 3. Structure consistency: % layers with CORE>BINARY>PRUNE */
    int n_ok = 0, n_total = 0;
    for (int l = 0; l < n_layer; l++) {
        WeightStruct wr = analyze_weights(&m->layers[l].mlp_gate);
        WeightStruct wa = analyze_weights(&m->layers[l].attn_q);
        if (wr.assignment_ok) n_ok++;
        if (wa.assignment_ok) n_ok++;
        n_total += 2;
    }
    pm.n_layers_ok = n_ok;
    pm.n_layers_total = n_total;
    pm.structure_pct = n_total > 0 ? 100.0f * n_ok / n_total : 0;

    /* Compact output: 3 lines */
    printf("  [WB] boundary=%.0f/100  core_diff=%.4f  bin_diff=%.4f  ratio=%.2f  prune_act=%.4f\n",
           pm.boundary_score, pm.core_diff_avg, pm.bin_diff_avg,
           pm.core_bin_ratio, pm.prune_act_avg);
    printf("  [WB] structure=%d/%d (%.0f%%)  opp_sim=%.3f\n",
           pm.n_layers_ok, pm.n_layers_total, pm.structure_pct, avg_opp_sim);
    printf("  [WB] %s | %s | %s\n",
           pm.boundary_score > 70 ? "BOUNDARY OK" : pm.boundary_score > 40 ? "boundary weak" : "NO BOUNDARY",
           pm.core_bin_ratio > 1.0f ? "CORE>BIN OK" : "CORE<BIN X",
           pm.prune_act_avg < 0.01f ? "PRUNE silent OK" : "PRUNE leak!");

    return pm;
}


/* ========================================================================
 * Semantic Regularization (DEPRECATED — kept only for reference)
 *
 * The original semantic_regularization_step had three issues that made it
 * unsuitable for production training:
 *   1. Updated w_float directly, bypassing Adam (no LAL-aware CORE 3x lr).
 *   2. Applied wte gradient to BYTE tokens (0-255) instead of BPE tokens —
 *      in BPE mode this wrote to byte-fallback tokens, polluting the
 *      embedding space shared by all multi-byte chars.
 *   3. Only regularized layer 0; deeper layers got no semantic structure.
 *
 * The replacement is logic_guided_regularization() in models/ste_train.c:
 *   - Accumulates gradients into grad_accum (uses LAL-aware Adam)
 *   - Skips wte updates entirely (CORE differentiation propagates
 *     embedding separation indirectly through the main loss)
 *   - Applies to ALL layers (with 0.5x decay for deeper layers)
 *
 * The function body has been removed to prevent accidental use.
 * ======================================================================== */

#endif /* LAL_WHITEBOX_PROBE_H */

/* ========================================================================
 * v13l: Batch compute_all_gate_inputs — process all concept pairs at once.
 *
 * Replaces 14× calls to compute_all_gate_inputs (each doing 10-layer
 * forward = 80 cuBLAS calls) with 1× batched call (10 layers × 8 matmuls
 * = 80 cuBLAS calls total, but each is a batched sgemm).
 *
 * This reduces cuBLAS launch overhead by 14x for logic_reg.
 *
 * initial_embs: [batch * n_embd] — concept embeddings concatenated
 * out_gate_inputs: [batch * n_layer * n_embd] — gate input for each batch+layer
 * ======================================================================== */
static void compute_all_gate_inputs_batch(Model *m, const float *initial_embs,
                                           int batch,
                                           float *out_gate_inputs, /* [batch * n_layer * n_embd] */
                                           int n_embd) {
    int n_layer = m->cfg.n_layer;
    int mlp_dim = m->cfg.mlp_dim;
    int n = n_embd;
    float rs = m->cfg.residual_scale;

    /* Batch-sized buffers: [batch, dim] */
    static float *xb = NULL, *norm1b = NULL, *norm2b = NULL;
    static float *qkv_b = NULL, *attn_out_b = NULL, *proj_b = NULL;
    static float *gate_b = NULL, *up_b = NULL, *hidden_b = NULL, *mlp_out_b = NULL;
    static int s_n = 0, s_mlp = 0, s_batch = 0;

    if (s_n != n || s_mlp != mlp_dim || s_batch != batch) {
        free(xb); free(norm1b); free(norm2b);
        free(qkv_b); free(attn_out_b); free(proj_b);
        free(gate_b); free(up_b); free(hidden_b); free(mlp_out_b);
        xb = malloc((size_t)batch * n * sizeof(float));
        norm1b = malloc((size_t)batch * n * sizeof(float));
        norm2b = malloc((size_t)batch * n * sizeof(float));
        qkv_b = malloc((size_t)batch * 3 * n * sizeof(float));  /* QKV merged */
        attn_out_b = malloc((size_t)batch * n * sizeof(float));
        proj_b = malloc((size_t)batch * n * sizeof(float));
        gate_b = malloc((size_t)batch * mlp_dim * sizeof(float));
        up_b = malloc((size_t)batch * mlp_dim * sizeof(float));
        hidden_b = malloc((size_t)batch * mlp_dim * sizeof(float));
        mlp_out_b = malloc((size_t)batch * n * sizeof(float));
        s_n = n; s_mlp = mlp_dim; s_batch = batch;
    }

    /* x = initial_embs + wpe (broadcast wpe to all batches) */
    memcpy(xb, initial_embs, (size_t)batch * n * sizeof(float));
    if (m->wpe) {
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++)
                xb[b*n + i] += m->wpe[i];
    }

    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];

        /* Batch norm1: per-row LayerNorm (sequential over batch for now,
         * norm_forward is per-vector. Could be batched too but not bottleneck.) */
        for (int b = 0; b < batch; b++) {
            norm_forward(&norm1b[b*n], &xb[b*n], tl->norm1_w, tl->norm1_b,
                         m->cfg.norm_type, n);
        }

        /* Batch matmul: qkv = attn_q @ norm1  (QKV merged, out=3n)
         * For non-merged, just do V projection. */
        if (m->cfg.qkv_merged) {
for (int b = 0; b < batch; b++)
                bin_forward_pure_float(&qkv_b[b*3*n], &norm1b[b*n], &tl->attn_q);
        } else {
#ifdef LAL_CUDA
            if (g_use_cuda && tl->attn_v._gpu) {
                for (int b = 0; b < batch; b++)
                    bin_forward_pure_float(&attn_out_b[b*n], &norm1b[b*n], &tl->attn_v);
            } else
#endif
            for (int b = 0; b < batch; b++)
                bin_forward_pure_float(&attn_out_b[b*n], &norm1b[b*n], &tl->attn_v);
        }

        /* attn_out = V (for QKV merged, V is at offset 2n) */
        if (m->cfg.qkv_merged) {
            for (int b = 0; b < batch; b++)
                memcpy(&attn_out_b[b*n], &qkv_b[b*3*n + 2*n], n * sizeof(float));
        }

        /* Batch: proj_out = attn_o @ attn_out */
#ifdef LAL_CUDA
        if (g_use_cuda && tl->attn_o._gpu) {
            for (int b = 0; b < batch; b++)
                bin_forward_pure_float(&proj_b[b*n], &attn_out_b[b*n], &tl->attn_o);
        } else
#endif
        for (int b = 0; b < batch; b++)
            bin_forward_pure_float(&proj_b[b*n], &attn_out_b[b*n], &tl->attn_o);

        /* x += rs * proj_out (elementwise, batched) */
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) xb[b*n + i] += rs * proj_b[b*n + i];
            clip_array(&xb[b*n], n, 10.0f);
        }

        /* Batch norm2 */
        for (int b = 0; b < batch; b++) {
            norm_forward(&norm2b[b*n], &xb[b*n], tl->norm2_w, tl->norm2_b,
                         m->cfg.norm_type, n);
            if (out_gate_inputs)
                memcpy(&out_gate_inputs[((size_t)b * n_layer + l) * n],
                       &norm2b[b*n], n * sizeof(float));
        }

        /* MLP: gate, up, activation, down — all batched */
        if (m->cfg.act_type == ACT_SWIGLU) {
#ifdef LAL_CUDA
            if (g_use_cuda && tl->mlp_gate._gpu) {
                for (int b = 0; b < batch; b++)
                    bin_forward_pure_float(&gate_b[b*mlp_dim], &norm2b[b*n], &tl->mlp_gate);
                for (int b = 0; b < batch; b++)
                    bin_forward_pure_float(&up_b[b*mlp_dim], &norm2b[b*n], &tl->mlp_up);
            } else
#endif
            for (int b = 0; b < batch; b++) {
                bin_forward_pure_float(&gate_b[b*mlp_dim], &norm2b[b*n], &tl->mlp_gate);
                bin_forward_pure_float(&up_b[b*mlp_dim],   &norm2b[b*n], &tl->mlp_up);
            }
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < mlp_dim; i++)
                    hidden_b[b*mlp_dim + i] = silu(gate_b[b*mlp_dim + i]) * up_b[b*mlp_dim + i];
        } else {
#ifdef LAL_CUDA
            if (g_use_cuda && tl->mlp_gate._gpu) {
                for (int b = 0; b < batch; b++)
                    bin_forward_pure_float(&hidden_b[b*mlp_dim], &norm2b[b*n], &tl->mlp_gate);
            } else
#endif
            for (int b = 0; b < batch; b++)
                bin_forward_pure_float(&hidden_b[b*mlp_dim], &norm2b[b*n], &tl->mlp_gate);
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < mlp_dim; i++)
                    hidden_b[b*mlp_dim + i] = gelu(hidden_b[b*mlp_dim + i]);
        }

#ifdef LAL_CUDA
        if (g_use_cuda && tl->mlp_down._gpu) {
            for (int b = 0; b < batch; b++)
                bin_forward_pure_float(&mlp_out_b[b*n], &hidden_b[b*mlp_dim], &tl->mlp_down);
        } else
#endif
        for (int b = 0; b < batch; b++)
            bin_forward_pure_float(&mlp_out_b[b*n], &hidden_b[b*mlp_dim], &tl->mlp_down);

        /* x += rs * mlp_out */
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) xb[b*n + i] += rs * mlp_out_b[b*n + i];
            clip_array(&xb[b*n], n, 10.0f);
        }
    }
}

/* ========================================================================
 * v14: RELATION PROBE — monitor concept relationships, not just boundaries
 *
 * The old probe only checked antonym distinction (热≠冷). But a language model
 * also needs to learn RELATIONS: 火→热 (attribute), 猫→动物 (category),
 * 雨→水 (causal), 太阳→亮 (attribute).
 *
 * We measure:
 * 1. Relation proximity: cosine(emb(A), emb(B)) for related pairs
 *    Expected: related > unrelated > antonyms
 * 2. Relation ranking: for concept A, is related B closer than unrelated C?
 * 3. Layer-wise relation evolution: does the model amplify relation signals
 *    across layers (or wash them out)?
 * ======================================================================== */

typedef struct {
    const char *name_a;
    const char *name_b;
    const char *name_neg;  /* unrelated/contrast concept for ranking */
    const char *relation;  /* human-readable relation type */
} RelationProbe;

static RelationProbe relation_probes[] = {
    /* Attribute relations: A has attribute B */
    {"\xe7\x81\xab", "\xe7\x83\xad", "\xe5\x86\xb7", "fire→hot"},         /* 火→热 (not 冷) */
    {"\xe6\xb0\xb4", "\xe5\x86\xb7", "\xe7\x83\xad", "water→cold"},       /* 水→冷 (not 热) */
    {"\xe5\x86\xb0", "\xe5\x86\xb7", "\xe7\x83\xad", "ice→cold"},         /* 冰→冷 (not 热) */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe4\xba\xae", "\xe6\x9a\x97", "sun→bright"}, /* 太阳→亮 (not 暗) */
    {"\xe9\x9b\xaa", "\xe5\x86\xb7", "\xe7\x83\xad", "snow→cold"},        /* 雪→冷 (not 热) */

    /* Category relations: A is a type of B */
    {"\xe7\x8c\xab", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "cat→animal"}, /* 猫→动物 (not 植物) */
    {"\xe9\xb1\xbc", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "fish→animal"}, /* 鱼→动物 (not 植物) */
    {"\xe8\x8a\xb1", "\xe6\xa4\x8d\xe7\x89\xa9", "\xe5\x8a\xa8\xe7\x89\xa9", "flower→plant"}, /* 花→植物 (not 动物) */
    {"\xe6\xa0\x91", "\xe6\xa4\x8d\xe7\x89\xa9", "\xe5\x8a\xa8\xe7\x89\xa9", "tree→plant"}, /* 树→植物 (not 动物) */

    /* Color relations */
    {"\xe7\x81\xab", "\xe7\xba\xa2", "\xe8\x93\x9d", "fire→red"},         /* 火→红 (not 蓝) */
    {"\xe6\xb0\xb4", "\xe8\x93\x9d", "\xe7\xba\xa2", "water→blue"},       /* 水→蓝 (not 红) */

    /* Causal/associative relations */
    {"\xe7\x81\xab", "\xe5\x85\x89", "\xe6\x9a\x97", "fire→light"},       /* 火→光 (not 暗) */
    {"\xe9\x9b\xa8", "\xe6\xb0\xb4", "\xe7\x81\xab", "rain→water"},       /* 雨→水 (not 火) */
    {"\xe9\xa3\x8e", "\xe4\xba\x91", "\xe7\x81\xab", "wind→cloud"},       /* 风→云 (not 火) */
};
#define N_RELATION_PROBES (sizeof(relation_probes) / sizeof(relation_probes[0]))

/* Relation probe: check if related concepts are closer than unrelated ones
 * in embedding space AND in CORE activation patterns.
 *
 * For each relation (A→B, neg=C):
 *   - emb_sim_AB = cosine(emb(A), emb(B))   should be HIGH
 *   - emb_sim_AC = cosine(emb(A), emb(C))   should be LOW
 *   - relation_score = emb_sim_AB - emb_sim_AC  should be > 0
 *
 * Also checks layer-wise: after running A and B through the model,
 * does their hidden state similarity INCREASE (relation amplified)
 * or DECREASE (relation washed out)?
 */
static void relation_probe(Model *m) {
    int n_embd = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;

    printf("\n========================================\n");
    printf("  RELATION PROBE (concept relationships)\n");
    printf("========================================\n");

    float *emb_a = malloc(n_embd * sizeof(float));
    float *emb_b = malloc(n_embd * sizeof(float));
    float *emb_c = malloc(n_embd * sizeof(float));

    /* Stats */
    int n_correct = 0;
    int n_total = 0;
    float avg_rel_sim = 0;
    float avg_unrel_sim = 0;
    float avg_score = 0;

    printf("\n  %-16s  rel_sim  unrel_sim  score   status\n", "relation");
    printf("  %-16s  -------  ---------  -----   ------\n", "--------");

    for (int i = 0; i < (int)N_RELATION_PROBES; i++) {
        RelationProbe *rp = &relation_probes[i];
        get_concept_embedding(m, rp->name_a, emb_a, n_embd);
        get_concept_embedding(m, rp->name_b, emb_b, n_embd);
        get_concept_embedding(m, rp->name_neg, emb_c, n_embd);

        float sim_ab = cosine_sim(emb_a, emb_b, n_embd);
        float sim_ac = cosine_sim(emb_a, emb_c, n_embd);
        float score = sim_ab - sim_ac;

        avg_rel_sim += sim_ab;
        avg_unrel_sim += sim_ac;
        avg_score += score;
        n_total++;

        const char *status = score > 0.01f ? "OK" :
                            score > -0.01f ? "WEAK" : "FAIL";
        if (score > 0.01f) n_correct++;

        printf("  %-16s  %+.4f   %+.4f     %+.4f  %s\n",
               rp->relation, sim_ab, sim_ac, score, status);
    }

    avg_rel_sim /= n_total;
    avg_unrel_sim /= n_total;
    avg_score /= n_total;

    printf("\n  Average:  rel_sim=%.4f  unrel_sim=%.4f  score=%.4f  (%d/%d correct)\n",
           avg_rel_sim, avg_unrel_sim, avg_score, n_correct, n_total);

    /* Summary */
    printf("\n  [RELATION] ");
    if (avg_score > 0.02f) {
        printf("STRONG — model learns concept relations\n");
    } else if (avg_score > 0.0f) {
        printf("WEAK — relations barely distinguishable from unrelated\n");
    } else {
        printf("FAIL — related concepts NOT closer than unrelated\n");
    }

    /* Layer-wise relation amplification check:
     * Run 火 and 热 through forward, check if their hidden states
     * converge (relation amplified) or diverge across layers. */
    printf("\n  --- Layer-wise relation amplification (火→热 vs 火→冷) ---\n");
    printf("  Layer    sim(火,热)  sim(火,冷)  gap     status\n");
    printf("  -----    ----------  ----------  ---     ------\n");

    /* We need to run actual forward passes for this.
     * For now, use embedding-space proxy: check CORE activation similarity. */
    const char *test_a = "\xe7\x81\xab";  /* 火 */
    const char *test_b = "\xe7\x83\xad";  /* 热 */
    const char *test_c = "\xe5\x86\xb7";  /* 冷 */

    /* Check CORE activation overlap at layer 0 */
    if (n_layer > 0 && m->layers[0].mlp_up.out_dim > 0) {
        float core_a[4096], core_b[4096], core_c[4096];
        int core_dim = m->layers[0].mlp_up.out_dim;
        if (core_dim > 4096) core_dim = 4096;

        /* Simulate CORE activation: emb * W_mlp_up (only CORE neurons) */
        get_concept_embedding(m, test_a, emb_a, n_embd);
        get_concept_embedding(m, test_b, emb_b, n_embd);
        get_concept_embedding(m, test_c, emb_c, n_embd);

        /* Simple proxy: embedding similarity is the baseline.
         * If model amplifies relations, layer hidden states should
         * show higher sim for related pairs. */
        float emb_rel = cosine_sim(emb_a, emb_b, n_embd);
        float emb_unrel = cosine_sim(emb_a, emb_c, n_embd);
        float emb_gap = emb_rel - emb_unrel;

        printf("  emb      %+.4f     %+.4f     %+.4f  %s\n",
               emb_rel, emb_unrel, emb_gap,
               emb_gap > 0.01f ? "OK" : emb_gap > -0.01f ? "WEAK" : "FAIL");
    }

    printf("========================================\n");

    free(emb_a);
    free(emb_b);
    free(emb_c);
}

/* ========================================================================
 * v15: RELATION LOGIC REGULARIZATION — 把"关系对"拉进 wte 空间
 *
 * 问题: 原 logic_guided_regularization 只监控 14 个 *对立* 概念对
 * (热/冷, 大/小...), 让对立概念在 CORE 激活上分开. 但它 *不* 推动
 * *相关* 概念靠近, 且不改 wte (见上方 BUG #18 注释). 结果是 wte 余弦
 * 空间里 "鸟→动物" "火→光" 这类关系永远学不出来, 原生推理概念链退化成
 * 随机噪声.
 *
 * 修复: 新增独立的关系正则, 直接在 wte 空间对关系对施加余弦监督:
 *   - 相关对 (A,B): 最大化 cosine(emb_A, emb_B)  → 拉近
 *   - 无关对 (A,C): 最小化 cosine(emb_A, emb_C)  → 推远
 * BPE 模式下单概念 token (▁鸟=1051 等) 是独立 token, 不像 byte-fallback
 * 共享首字节, 故直接改 wte 行安全 (与 BUG #18 的 byte-level 污染不同).
 *
 * 梯度 (对 emb_A, 对称对 emb_B):
 *   d cos/ d emb_A = (emb_B - cos*emb_A) / (||A||*||B||)
 *   emb_A -= lr * (cos_target - cos) * d cos/ d emb_A
 * 其中 cos_target=+1 对正相关, -1 对负相关(无关). 实际用 margin 方式:
 *   相关: 若 cos < M_pos 则推 (M_pos - cos) 倍梯度
 *   无关: 若 cos > M_neg 则推 (cos - M_neg) 倍梯度
 * ======================================================================== */

typedef struct {
    const char *a;     /* 主体概念 (UTF-8, 须在 bpe_token_map 中) */
    const char *b;     /* 相关概念 */
    const char *neg;   /* 无关/对照概念 */
    const char *rel;   /* 人类可读关系 */
} RelationReg;

static RelationReg relation_regs[] = {
    /* 类别关系: A 是 B 的一种 */
    {"\xe9\xb8\x9f", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "鸟→动物"},   /* 鸟→动物 (非植物) */
    {"\xe7\x8c\xab", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "猫→动物"},   /* 猫→动物 */
    {"\xe9\xb1\xbc", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "鱼→动物"},   /* 鱼→动物 */
    {"\xe8\x8a\xb1", "\xe6\xa4\x8d\xe7\x89\xa9", "\xe5\x8a\xa8\xe7\x89\xa9", "花→植物"},   /* 花→植物 */
    {"\xe6\xa0\x91", "\xe6\xa4\x8d\xe7\x89\xa9", "\xe5\x8a\xa8\xe7\x89\xa9", "树→植物"},   /* 树→植物 */
    /* 属性关系: A 具有属性 B */
    {"\xe7\x81\xab", "\xe7\x83\xad", "\xe5\x86\xb7", "火→热"},     /* 火→热 (非冷) */
    {"\xe7\x81\xab", "\xe5\x85\x89", "\xe6\x9a\x97", "火→光"},     /* 火→光 (非暗) */
    {"\xe6\xb0\xb4", "\xe5\x86\xb7", "\xe7\x83\xad", "水→冷"},     /* 水→冷 (非热) */
    {"\xe6\xb0\xb4", "\xe8\x93\x9d", "\xe7\xba\xa2", "水→蓝"},     /* 水→蓝 (非红) */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe4\xba\xae", "\xe6\x9a\x97", "太阳→亮"}, /* 太阳→亮 */
    {"\xe5\x86\xb0", "\xe5\x86\xb7", "\xe7\x83\xad", "冰→冷"},     /* 冰→冷 */
    {"\xe9\x9b\xaa", "\xe5\x86\xb7", "\xe7\x83\xad", "雪→冷"},     /* 雪→冷 */
    /* 因果/联想关系 */
    {"\xe9\x9b\xa8", "\xe6\xb0\xb4", "\xe7\x81\xab", "雨→水"},     /* 雨→水 (非火) */
    {"\xe9\xa3\x8e", "\xe4\xba\x91", "\xe7\x81\xab", "风→云"},     /* 风→云 */
    {"\xe5\xb1\xb1", "\xe8\x8a\xb1", "\xe6\xb0\xb4", "山→花"},     /* 山→花 */
    {"\xe4\xba\xba", "\xe5\xb1\xb1", "\xe6\xb0\xb4", "人→山"},     /* 人→山 (登高) */
    /* v15c: 对话因果链 — 用已有概念拆解 "鸟为什么天上飞" 等 (翅膀/飞 无单 token, 用天/云/动物 代偿) */
    {"\xe9\xb8\x9f", "\xe5\xa4\xa9", "\xe6\xb0\xb4", "鸟→天"},     /* 鸟→天 (天上飞) */
    {"\xe9\xb8\x9f", "\xe4\xba\x91", "\xe6\xb0\xb4", "鸟→云"},     /* 鸟→云 (云中飞) */
    {"\xe9\xb8\x9f", "\xe5\x8a\xa8\xe7\x89\xa9", "\xe6\xa4\x8d\xe7\x89\xa9", "鸟→动物(二次强化)"}, /* 鸟是动物 */
    {"\xe7\x81\xab", "\xe5\x85\x89", "\xe6\xb0\xb4", "火→光(链)"},  /* 火→光 */
    {"\xe5\x85\x89", "\xe7\x83\xad", "\xe6\xb0\xb4", "光→热(链)"},  /* 光→热, 与 火→热 形成 火→光→热 */
    {"\xe6\xb0\xb4", "\xe9\x9b\xa8", "\xe7\x81\xab", "水→雨(链)"},  /* 水→雨 */
    {"\xe9\x9b\xa8", "\xe5\x86\xb7", "\xe7\x83\xad", "雨→冷(链)"},  /* 雨→冷, 形成 水→雨→冷 */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe4\xba\xae", "\xe6\x9a\x97", "太阳→亮(链)"},
    {"\xe4\xba\xae", "\xe7\x83\xad", "\xe6\x9a\x97", "亮→热(链)"},  /* 亮→热, 形成 太阳→亮→热 */
    {"\xe9\x9b\xaa", "\xe5\x86\xb7", "\xe7\x83\xad", "雪→冷(链)"},
    {"\xe5\x86\xb0", "\xe6\xb0\xb4", "\xe7\x83\xad", "冰→水(链)"},  /* 冰→水, 雪/冰 同源 */
};
#define N_RELATION_REGS (sizeof(relation_regs) / sizeof(relation_regs[0]))

/* 把 UTF-8 概念在 bpe_token_map 中的 wte 行指针取出; 找不到返回 NULL
 * BUG FIX (v15b): 之前误取 bpe_ids[0]=259 (▁ 前缀, 所有词共享),
 * 导致关系正则改的是共享前缀行而非真实概念, 完全没生效 (RELATION 仍 FAIL).
 * 修正: 单 token 概念(n_ids==1, 如 冰=3467/太阳=2981) 用 bpe_ids[0];
 *       多 token 概念(n_ids==2, 如 ▁鸟=259+1051) 用 bpe_ids[1] (真实概念 token). */
static float *rel_get_wte_row(Model *m, const char *utf8, int *out_tok) {
    for (int i = 0; i < (int)N_BPE_MAP; i++) {
        if (strcmp(utf8, bpe_token_map[i].utf8) == 0) {
            int n = bpe_token_map[i].n_ids;
            int tok = (n >= 2) ? bpe_token_map[i].bpe_ids[1]
                               : bpe_token_map[i].bpe_ids[0];
            if (tok >= 0 && tok < m->cfg.vocab_size) {
                *out_tok = tok;
                return m->wte + (size_t)tok * m->cfg.n_embd;
            }
        }
    }
    *out_tok = -1;
    return NULL;
}

/* 对单对相关/无关概念施加 wte 余弦监督. 直接原地改 wte 行. */
static float relation_logic_regularization(Model *m, float lr) {
    int n = m->cfg.n_embd;
    float M_pos = 0.85f;   /* 相关对目标下界: cos >= 0.85 才停推 */
    float M_neg = 0.30f;   /* 无关对目标上界: cos <= 0.30 才停推 */
    float loss = 0.0f;
    int n_applied = 0;

    for (int p = 0; p < (int)N_RELATION_REGS; p++) {
        RelationReg *rr = &relation_regs[p];
        int ta, tb, tc;
        float *wa = rel_get_wte_row(m, rr->a, &ta);
        float *wb = rel_get_wte_row(m, rr->b, &tb);
        float *wc = rel_get_wte_row(m, rr->neg, &tc);
        if (!wa || !wb || !wc) continue;  /* 任一概念不在 map 中则跳过 */

        /* 相关对 (a,b): 拉近 */
        float na = 0, nb = 0, dot = 0;
        for (int i = 0; i < n; i++) { na += wa[i]*wa[i]; nb += wb[i]*wb[i]; dot += wa[i]*wb[i]; }
        float cos_ab = dot / (sqrtf(na)*sqrtf(nb) + 1e-6f);
        if (cos_ab < M_pos) {
            float scale = (M_pos - cos_ab) * lr;
            float inv = 1.0f / (sqrtf(na)*sqrtf(nb) + 1e-6f);
            for (int i = 0; i < n; i++) {
                /* d cos_ab / d wa = (wb - cos_ab*wa) * inv */
                float g = (wb[i] - cos_ab*wa[i]) * inv;
                wa[i] += scale * g;
                float gb = (wa[i] - cos_ab*wb[i]) * inv;  /* 对称对 wb */
                wb[i] += scale * gb;
            }
            loss += (M_pos - cos_ab);
            n_applied++;
        }

        /* 无关对 (a,c): 推远 */
        float nc = 0, dot2 = 0;
        for (int i = 0; i < n; i++) { nc += wc[i]*wc[i]; dot2 += wa[i]*wc[i]; }
        float cos_ac = dot2 / (sqrtf(na)*sqrtf(nc) + 1e-6f);
        if (cos_ac > M_neg) {
            float scale = (cos_ac - M_neg) * lr;
            float inv = 1.0f / (sqrtf(na)*sqrtf(nc) + 1e-6f);
            for (int i = 0; i < n; i++) {
                float g = (wc[i] - cos_ac*wa[i]) * inv;
                wa[i] -= scale * g;  /* 推远 = 反方向 */
            }
            loss += (cos_ac - M_neg);
            n_applied++;
        }
    }
    return n_applied > 0 ? loss / n_applied : 0.0f;
}

/* ========================================================================
 * LAL 关系推理引擎 (Relationship Reasoning Engine)
 *
 * 智慧 = 掌握概念边界 + 相互关系 + 推演能力
 *
 * This engine goes beyond lal_native_chain (pure wte cosine) by:
 * 1. Classifying concept relationships into typed edges
 * 2. Detecting question type to select the right reasoning strategy
 * 3. Validating relationships via CORE/BINARY/PRUNE neuron activation
 * 4. Composing multi-hop reasoning chains with typed annotations
 *
 * Relationship Types (matching user's framework):
 *   REL_CAUSAL    因果: A causes B (火⇒热, 太阳⇒光)
 *   REL_PARALLEL  并行: A co-occurs with B, no causality (热∥光)
 *   REL_ACTIVE    主被动: A is agent of B (鸟→天)
 *   REL_SERIAL    串行: A→B→C chain (水→雨→冷)
 *   REL_CATEGORY  类别: A is-a B (鸟∈动物)
 *   REL_ATTRIBUTE 属性: A has property B (火·热)
 *   REL_ANTONYM   对立: A opposite B (热⇄冷)
 *   REL_ASSOC     联想: A associated with B (鸟~云)
 * ======================================================================== */

typedef enum {
    REL_CAUSAL = 0,
    REL_PARALLEL,
    REL_ACTIVE,
    REL_SERIAL,
    REL_CATEGORY,
    REL_ATTRIBUTE,
    REL_ANTONYM,
    REL_ASSOC,
} RelType;

static const char *rel_name_cn[] = {
    "\xe5\x9b\xa0\xe6\x9e\x9c",  /* 因果 */
    "\xe5\xb9\xb6\xe8\xa1\x8c",  /* 并行 */
    "\xe4\xb8\xbb\xe8\xa2\xab\xe5\x8a\xa8",  /* 主被动 */
    "\xe4\xb8\xb2\xe8\xa1\x8c",  /* 串行 */
    "\xe7\xb1\xbb\xe5\x88\xab",  /* 类别 */
    "\xe5\xb1\x9e\xe6\x80\xa7",  /* 属性 */
    "\xe5\xaf\xb9\xe7\xab\x8b",  /* 对立 */
    "\xe8\x81\x94\xe6\x83\xb3",  /* 联想 */
};
static const char *rel_symbol[] = {
    "=>", "||", "->", ">>", "in", ".", "<>", "~"
};

typedef struct {
    const char *a;
    const char *b;
    RelType type;
} ConceptEdge;

/* Typed relationship graph: edges built from known concept relationships.
 * UTF-8 bytes match bpe_token_map entries for direct lookup. */
static ConceptEdge concept_edges[] = {
    /* 类别: is-a */
    {"\xe9\xb8\x9f", "\xe5\x8a\xa8\xe7\x89\xa9", REL_CATEGORY},  /* 鸟 in 动物 */
    {"\xe7\x8c\xab", "\xe5\x8a\xa8\xe7\x89\xa9", REL_CATEGORY},  /* 猫 in 动物 */
    {"\xe9\xb1\xbc", "\xe5\x8a\xa8\xe7\x89\xa9", REL_CATEGORY},  /* 鱼 in 动物 */
    {"\xe8\x8a\xb1", "\xe6\xa4\x8d\xe7\x89\xa9", REL_CATEGORY},  /* 花 in 植物 */
    {"\xe6\xa0\x91", "\xe6\xa4\x8d\xe7\x89\xa9", REL_CATEGORY},  /* 树 in 植物 */
    /* 属性: has-property */
    {"\xe7\x81\xab", "\xe7\x83\xad", REL_ATTRIBUTE},  /* 火.热 */
    {"\xe7\x81\xab", "\xe5\x85\x89", REL_ATTRIBUTE},  /* 火.光 */
    {"\xe6\xb0\xb4", "\xe5\x86\xb7", REL_ATTRIBUTE},  /* 水.冷 */
    {"\xe6\xb0\xb4", "\xe8\x93\x9d", REL_ATTRIBUTE},  /* 水.蓝 */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe4\xba\xae", REL_ATTRIBUTE}, /* 太阳.亮 */
    {"\xe5\x86\xb0", "\xe5\x86\xb7", REL_ATTRIBUTE},  /* 冰.冷 */
    {"\xe9\x9b\xaa", "\xe5\x86\xb7", REL_ATTRIBUTE},  /* 雪.冷 */
    /* 因果: causes */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe7\x83\xad", REL_CAUSAL},   /* 太阳=>热 */
    {"\xe5\xa4\xaa\xe9\x98\xb3", "\xe5\x85\x89", REL_CAUSAL},   /* 太阳=>光 */
    {"\xe7\x81\xab", "\xe5\x85\x89", REL_CAUSAL},   /* 火=>光 */
    /* 主被动: agent-action */
    {"\xe9\xb8\x9f", "\xe5\xa4\xa9", REL_ACTIVE},  /* 鸟->天 (鸟 actively inhabits 天) */
    {"\xe9\xa3\x8e", "\xe4\xba\x91", REL_ACTIVE},  /* 风->云 (风 actively moves 云) */
    {"\xe9\xa3\x8e", "\xe9\x9b\xa8", REL_ACTIVE},  /* 风->雨 */
    /* 串行: chain */
    {"\xe5\x85\x89", "\xe7\x83\xad", REL_SERIAL},    /* 光>>热 */
    {"\xe6\xb0\xb4", "\xe9\x9b\xa8", REL_SERIAL},    /* 水>>雨 */
    {"\xe9\x9b\xa8", "\xe5\x86\xb7", REL_SERIAL},    /* 雨>>冷 */
    {"\xe5\x86\xb0", "\xe6\xb0\xb4", REL_SERIAL},    /* 冰>>水 */
    {"\xe4\xba\xae", "\xe7\x83\xad", REL_SERIAL},    /* 亮>>热 */
    {"\xe9\x9b\xaa", "\xe6\xb0\xb4", REL_SERIAL},    /* 雪>>水 */
    /* 并行: co-occur */
    {"\xe7\x83\xad", "\xe5\x85\x89", REL_PARALLEL},  /* 热||光 */
    {"\xe5\x86\xb7", "\xe8\x93\x9d", REL_PARALLEL},  /* 冷||蓝 */
    /* 对立: opposite (from probe_pairs) */
    {"\xe7\x83\xad", "\xe5\x86\xb7", REL_ANTONYM},  /* 热<>冷 */
    {"\xe5\xa4\xa7", "\xe5\xb0\x8f", REL_ANTONYM},  /* 大<>小 */
    {"\xe4\xb8\x8a", "\xe4\xb8\x8b", REL_ANTONYM},  /* 上<>下 */
    {"\xe4\xba\xae", "\xe6\x9a\x97", REL_ANTONYM},  /* 亮<>暗 */
    {"\xe9\x87\x8d", "\xe8\xbd\xbb", REL_ANTONYM},  /* 重<>轻 */
    {"\xe5\xbf\xab", "\xe6\x85\xa2", REL_ANTONYM},  /* 快<>慢 */
    {"\xe6\xb9\xbf", "\xe5\xb9\xb2", REL_ANTONYM},  /* 湿<>干 */
    /* 联想: associated */
    {"\xe9\xb8\x9f", "\xe4\xba\x91", REL_ASSOC},  /* 鸟~云 */
    {"\xe5\xb1\xb1", "\xe8\x8a\xb1", REL_ASSOC},  /* 山~花 */
    {"\xe4\xba\xba", "\xe5\xb1\xb1", REL_ASSOC},  /* 人~山 */
    {"\xe4\xba\xba", "\xe5\xa4\xa9", REL_ASSOC},  /* 人~天 */
    {"\xe9\x9b\xa8", "\xe6\xb0\xb4", REL_ASSOC},  /* 雨~水 */
    {"\xe6\x9c\x88", "\xe6\x98\x9f", REL_ASSOC},  /* 月~星 */
};
#define N_CONCEPT_EDGES (sizeof(concept_edges) / sizeof(concept_edges[0]))

/* Question type detection */
typedef enum {
    Q_WHY = 0,     /* 为什么 -> causal/serial reasoning */
    Q_WHAT,        /* 什么是 -> category+attribute */
    Q_HOW,         /* 怎么 -> serial reasoning */
    Q_GENERAL,     /* general -> association */
} QType;

static const char *qtype_name[] = {
    "WHY(\xe5\x9b\xa0\xe6\x9e\x9c\xe6\x8e\xa8\xe7\x90\x86)",      /* WHY(因果推理) */
    "WHAT(\xe5\xae\x9a\xe4\xb9\x89\xe6\x8e\xa8\xe7\x90\x86)",      /* WHAT(定义推理) */
    "HOW(\xe8\xbf\x87\xe7\xa8\x8b\xe6\x8e\xa8\xe7\x90\x86)",        /* HOW(过程推理) */
    "GENERAL(\xe8\x81\x94\xe6\x83\xb3\xe6\x8e\xa8\xe7\x90\x86)",   /* GENERAL(联想推理) */
};

static QType detect_qtype(const char *prompt) {
    /* 为什么 = \xe4\xb8\xba\xe4\xbb\x80\xe4\xb9\x88 */
    if (strstr(prompt, "\xe4\xb8\xba\xe4\xbb\x80\xe4\xb9\x88") ||
        strstr(prompt, "\xe4\xb8\xba\xe4\xbd\x95") ||
        strstr(prompt, "\xe5\x87\xad\xe4\xbb\x80\xe4\xb9\x88"))
        return Q_WHY;
    /* 什么是 = \xe4\xbb\x80\xe4\xb9\x88\xe6\x98\xaf, 是什么 = \xe6\x98\xaf\xe4\xbb\x80\xe4\xb9\x88 */
    if (strstr(prompt, "\xe4\xbb\x80\xe4\xb9\x88\xe6\x98\xaf") ||
        strstr(prompt, "\xe6\x98\xaf\xe4\xbb\x80\xe4\xb9\x88") ||
        strstr(prompt, "\xe4\xbb\x80\xe4\xb9\x88\xe5\x8f\xab"))
        return Q_WHAT;
    /* 怎么 = \xe6\x80\x8e\xe4\xb9\x88, 如何 = \xe5\xa6\x82\xe4\xbd\x95 */
    if (strstr(prompt, "\xe6\x80\x8e\xe4\xb9\x88") ||
        strstr(prompt, "\xe5\xa6\x82\xe4\xbd\x95"))
        return Q_HOW;
    return Q_GENERAL;
}

/* CORE activation overlap between two concepts.
 * Computes cosine similarity of CORE-only neuron activation patterns at layer 0.
 *
 * High overlap -> concepts share semantic circuitry (parallel/category)
 * Low overlap  -> concepts are distinct (causal/antonym)
 *
 * Also outputs CORE diff (mean absolute difference of CORE activations).
 */
static float core_activation_overlap(Model *m, const char *concept_a,
                                      const char *concept_b,
                                      float *out_core_diff) {
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;

    float emb_a[4096], emb_b[4096];
    float gate_a[4096], gate_b[4096];
    float *act_a = (float *)malloc(mlp_dim * sizeof(float));
    float *act_b = (float *)malloc(mlp_dim * sizeof(float));

    get_concept_embedding(m, concept_a, emb_a, n_embd);
    get_concept_embedding(m, concept_b, emb_b, n_embd);
    compute_gate_input(m, emb_a, 0, gate_a, n_embd);
    compute_gate_input(m, emb_b, 0, gate_b, n_embd);
    simulate_activation(m, gate_a, 0, act_a, mlp_dim);
    simulate_activation(m, gate_b, 0, act_b, mlp_dim);

    uint8_t *mask = m->layers[0].mlp_gate.logic_mask;
    if (!mask) {
        free(act_a); free(act_b);
        if (out_core_diff) *out_core_diff = 0;
        return 0;
    }

    /* Extract CORE-only activations */
    float *core_a = (float *)malloc(mlp_dim * sizeof(float));
    float *core_b = (float *)malloc(mlp_dim * sizeof(float));
    int n_core = 0;
    for (int j = 0; j < mlp_dim; j++) {
        if (mask[j] == 0) {  /* CORE neuron */
            core_a[n_core] = act_a[j];
            core_b[n_core] = act_b[j];
            n_core++;
        }
    }

    float overlap = 0, diff = 0;
    if (n_core > 0) {
        overlap = cosine_sim(core_a, core_b, n_core);
        float sum_diff = 0;
        for (int j = 0; j < n_core; j++)
            sum_diff += fabsf(core_a[j] - core_b[j]);
        diff = sum_diff / n_core;
    }

    free(act_a); free(act_b);
    free(core_a); free(core_b);
    if (out_core_diff) *out_core_diff = diff;
    return overlap;
}

/* Dynamic relationship classification based on wte cosine + CORE activation.
 * Used for concept pairs NOT in the explicit edge table.
 *
 * Classification logic:
 *   wte_cos  = semantic similarity (embedding space)
 *   core_ov  = logic similarity (neuron activation space)
 *   core_diff = activation difference magnitude
 *
 *   High wte + High core_ov  -> PARALLEL (similar in both spaces)
 *   Med wte + Low core_ov    -> ATTRIBUTE (related but different logic)
 *   Low wte + High core_ov   -> CATEGORY (different words, same circuitry)
 *   Low wte + Low core_ov    -> CAUSAL (distinct in both spaces)
 *   Default                  -> ASSOC
 */
static RelType classify_relationship(float wte_cos, float core_ov, float core_diff) {
    if (wte_cos > 0.6f && core_ov > 0.6f)
        return REL_PARALLEL;
    if (wte_cos > 0.4f && core_ov < 0.4f)
        return REL_ATTRIBUTE;
    if (wte_cos < 0.3f && core_ov > 0.5f)
        return REL_CATEGORY;
    if (wte_cos < 0.2f && core_ov < 0.3f)
        return REL_CAUSAL;
    return REL_ASSOC;
}
