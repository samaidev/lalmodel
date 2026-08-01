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
static BpeTokenMap bpe_token_map[] = {
    {"\xe7\x83\xad", 1, {32226}},                 /* 热 */
    {"\xe5\x86\xb7", 1, {32551}},                 /* 冷 */
    {"\xe5\xa4\xa7", 1, {31974}},                 /* 大 */
    {"\xe5\xb0\x8f", 1, {31928}},                 /* 小 */
    {"\xe4\xb8\x8a", 1, {31926}},                 /* 上 */
    {"\xe4\xb8\x8b", 1, {31968}},                 /* 下 */
    {"\xe4\xba\xae", 1, {32545}},                 /* 亮 */
    {"\xe6\x9a\x97", 1, {32723}},                 /* 暗 */
    {"\xe9\x87\x8d", 1, {31995}},                 /* 重 */
    {"\xe8\xbd\xbb", 3, {235, 192, 190}},         /* 轻 = <0xE8><0xBD><0xBB> */
    {"\xe5\xbf\xab", 1, {32391}},                 /* 快 */
    {"\xe6\x85\xa2", 1, {32356}},                 /* 慢 */
    {"\xe6\xb9\xbf", 3, {233, 188, 194}},         /* 湿 = <0xE6><0xB9><0xBF> */
    {"\xe5\xb9\xb2", 3, {232, 188, 181}},         /* 干 = <0xE5><0xB9><0xB2> */
    /* BUG #36 FIX: deep_whitebox_diagnosis uses 火/水 but they weren't in
     * the map → get_concept_embedding returned all-zeros → embedding norms
     * showed 0.000, cosine sim was meaningless, diagnosis was wrong.
     * Added common concepts used in diagnosis and generation testing. */
    {"\xe7\x81\xab", 1, {32646}},                 /* 火 */
    {"\xe6\xb0\xb4", 1, {31940}},                 /* 水 */
    {"\xe5\xb1\xb1", 1, {32170}},                 /* 山 */
    {"\xe8\x8a\xb1", 1, {32223}},                 /* 花 */
    {"\xe6\xa0\x91", 1, {32121}},                 /* 树 */
    {"\xe9\xb8\x9f", 1, {32765}},                 /* 鸟 */
    {"\xe9\xb1\xbc", 1, {32766}},                 /* 鱼 */
    {"\xe4\xba\xba", 1, {31920}},                 /* 人 */
    {"\xe5\xa4\xa9", 1, {31925}},                 /* 天 */
    {"\xe5\x9c\xb0", 1, {31989}},                 /* 地 */
    {"\xe6\x9c\x88", 1, {31967}},                 /* 月 */
    {"\xe6\x98\x9f", 1, {32224}},                 /* 星 */
    {"\xe9\xa3\x8e", 1, {32171}},                 /* 风 */
    {"\xe9\x9b\xa8", 1, {32561}},                 /* 雨 */
    {"\xe4\xba\x91", 1, {32604}},                 /* 云 */
    {"\xe5\xa4\xaa\xe9\x98\xb3", 1, {348}},       /* 太阳 = single BPE token */
    {"\xe5\xa4\xaa", 1, {31924}},                 /* 太 */
    {"\xe9\x98\xb3", 1, {32039}},                 /* 阳 */
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
         * separately (K not needed for V-copy attention). */
        if (m->cfg.qkv_merged) {
            bin_forward_pure_float(qkv_buf, norm1, &tl->attn_q);
            /* v already points to qkv_buf + 2*n */
        } else {
            bin_forward_pure_float(q, norm1, &tl->attn_q);  /* unused but matches forward */
            bin_forward_pure_float(v, norm1, &tl->attn_v);
        }

        /* V-copy attention (matches training default g_use_real_attention=0):
         *   attn_out = v
         * For single token this is exact. For multi-token concepts we're
         * already collapsing to one position, so this is still the right
         * degenerate attention. */
        memcpy(attn_out, v, n * sizeof(float));

        /* Output projection + residual */
        bin_forward_pure_float(proj_out, attn_out, &tl->attn_o);
        for (int i = 0; i < n; i++) x[i] += rs * proj_out[i];
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
        for (int i = 0; i < n; i++) x[i] += rs * mlp_out[i];
        clip_array(x, n, 10.0f);
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
 * out_gate_inputs: [n_layer * n_embd] — gate_input for each layer.
 * ======================================================================== */
static void compute_all_gate_inputs(Model *m, const float *initial_emb,
                                     float *out_gate_inputs, int n_embd) {
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

        if (m->cfg.qkv_merged) {
            bin_forward_pure_float(qkv_buf, norm1, &tl->attn_q);
        } else {
            bin_forward_pure_float(v, norm1, &tl->attn_v);
        }
        memcpy(attn_out, v, n * sizeof(float));

        bin_forward_pure_float(proj_out, attn_out, &tl->attn_o);
        for (int i = 0; i < n; i++) x[i] += rs * proj_out[i];
        clip_array(x, n, 10.0f);

        norm_forward(norm2, x, tl->norm2_w, tl->norm2_b, m->cfg.norm_type, n);

        /* Capture gate_input for this layer */
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
        for (int i = 0; i < n; i++) x[i] += rs * mlp_out[i];
        clip_array(x, n, 10.0f);
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
