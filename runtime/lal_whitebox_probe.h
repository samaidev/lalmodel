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

/* BPE token ids for probe concepts (precomputed via SentencePiece) */
typedef struct { const char *utf8; int bpe_id; } BpeTokenMap;
static BpeTokenMap bpe_token_map[] = {
    {"\xe7\x83\xad", 32226},  /* 热 */
    {"\xe5\x86\xb7", 32551},  /* 冷 */
    {"\xe5\xa4\xa7", 31974},  /* 大 */
    {"\xe5\xb0\x8f", 31928},  /* 小 */
    {"\xe4\xb8\x8a", 31926},  /* 上 */
    {"\xe4\xb8\x8b", 31968},  /* 下 */
    {"\xe4\xba\xae", 32545},  /* 亮 */
    {"\xe6\x9a\x97", 32723},  /* 暗 */
    {"\xe9\x87\x8d", 31995},  /* 重 */
    {"\xe8\xbd\xbb", 235},    /* 轻 */
    {"\xe5\xbf\xab", 32391},  /* 快 */
    {"\xe6\x85\xa2", 32356},  /* 慢 */
    {"\xe6\xb9\xbf", 233},    /* 湿 */
    {"\xe5\xb9\xb2", 232},    /* 干 */
};
#define N_BPE_MAP (sizeof(bpe_token_map) / sizeof(bpe_token_map[0]))

static void get_concept_embedding(Model *m, const char *utf8_bytes,
                                   float *out, int n_embd) {
    memset(out, 0, n_embd * sizeof(float));

    if (m->cfg.vocab_size > 256) {
        /* BPE mode: look up token id from mapping table */
        int tok = -1;
        for (int i = 0; i < (int)N_BPE_MAP; i++) {
            if (strcmp(utf8_bytes, bpe_token_map[i].utf8) == 0) {
                tok = bpe_token_map[i].bpe_id;
                break;
            }
        }
        if (tok < 0 || tok >= m->cfg.vocab_size) return;
        float *row = m->wte + (size_t)tok * n_embd;
        for (int j = 0; j < n_embd; j++)
            out[j] = row[j];
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
 * Simulated activation: emb * W for one layer's mlp_gate
 * Direct matmul using w_float — no model_forward needed.
 * This shows what CORE/BINARY/PRUNE neurons would fire.
 * ======================================================================== */
static void simulate_activation(Model *m, const float *emb, int layer,
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
                    s += emb[i] * wc[i];
            }
            core_idx++;
        } else {
            /* BINARY: use w_float (matches bin_forward_pure_float) */
            const float *wf = &fc->w_float[(size_t)j * in_dim];
            for (int i = 0; i < in_dim; i++)
                s += emb[i] * wf[i];
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
    
    for (int j = 0; j < out_dim; j++) {
        float ns = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = bl->w_float[i * out_dim + j];
            ns += w * w;
        }
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
 * Compute emb * W directly to see which neurons fire for each concept.
 * No model_forward — pure weight analysis.
 * ======================================================================== */
static void core_activation_analysis(Model *m) {
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    int n_layer = m->cfg.n_layer;
    
    printf("--- CORE Activation (simulated, layer 0) ---\n\n");
    
    /* For each concept pair, compute layer-0 MLP activation */
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        
        float emb_a[4096], emb_b[4096];
        get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
        get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);
        
        /* Simulate: activation = emb * W_layer0_mlp_gate */
        float *act_a = (float *)malloc(mlp_dim * sizeof(float));
        float *act_b = (float *)malloc(mlp_dim * sizeof(float));
        simulate_activation(m, emb_a, 0, act_a, mlp_dim);
        simulate_activation(m, emb_b, 0, act_b, mlp_dim);
        
        uint8_t *mask = m->layers[0].mlp_gate.logic_mask;
        if (!mask) { free(act_a); free(act_b); continue; }
        
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
        
        free(act_a);
        free(act_b);
    }
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
    int mid_layer = m->cfg.n_layer / 2;
    
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

    /* 2. CORE vs BINARY differentiation (layer 0 simulated activation) */
    float core_diff_sum = 0, bin_diff_sum = 0, prune_act_sum = 0;
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        float emb_a[4096], emb_b[4096];
        get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
        get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);

        float *act_a = (float *)malloc(mlp_dim * sizeof(float));
        float *act_b = (float *)malloc(mlp_dim * sizeof(float));
        simulate_activation(m, emb_a, 0, act_a, mlp_dim);
        simulate_activation(m, emb_b, 0, act_b, mlp_dim);

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
        free(act_a);
        free(act_b);
    }
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
 * Semantic Regularization: Contrastive loss to guide CORE/BINARY roles
 *
 * CORE neurons should DIFFERENTIATE opposite concepts (热 vs 冷)
 * BINARY neurons should CONVERGE for opposite concepts (coarse logic)
 *
 * Loss: L = -alpha * mean(|CORE_act_a - CORE_act_b|)
 *       + beta  * mean((BIN_act_a - BIN_act_b)^2)
 *
 * Gradient applied directly to w_float and wte for layer 0 mlp_gate.
 * ======================================================================== */

/* Apply one semantic regularization step.
 * Returns the regularization loss for logging. */
static float semantic_regularization_step(Model *m, float lr) {
    int n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim;
    BinLayer *fc = &m->layers[0].mlp_gate;
    uint8_t *mask = fc->logic_mask;
    if (!mask) return 0.0f;

    float alpha = 2.0f;  /* CORE differentiation weight (was 0.5 -- too weak) */
    float beta = 0.2f;   /* BINARY convergence weight (was 0.1) */
    float total_loss = 0.0f;

    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];

        /* Get embeddings for both concepts */
        float emb_a[4096], emb_b[4096];
        get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
        get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);

        /* Compute current activations (using CORRECT [out,in] layout) */
        float *act_a = (float *)malloc(mlp_dim * sizeof(float));
        float *act_b = (float *)malloc(mlp_dim * sizeof(float));
        simulate_activation(m, emb_a, 0, act_a, mlp_dim);
        simulate_activation(m, emb_b, 0, act_b, mlp_dim);

        /* Compute loss and gradients */
        int nc = 0, nb = 0;
        for (int j = 0; j < mlp_dim; j++) {
            if (mask[j] == 2) continue;  /* skip PRUNE */
            float diff = act_a[j] - act_b[j];
            float adiff = fabsf(diff);

            if (mask[j] == 0) {  /* CORE: maximize |diff| */
                total_loss -= alpha * adiff;
                nc++;
            } else {  /* BINARY: minimize diff^2 */
                total_loss += beta * diff * diff;
                nb++;
            }
        }

        /* Apply gradient to w_float for layer 0 mlp_gate */
        /* For CORE: dL/dw[j,i] = -alpha * sign(diff) * (emb_a[i] - emb_b[i]) */
        /* For BIN:  dL/dw[j,i] = beta * 2 * diff * (emb_a[i] - emb_b[i]) */
        float inv_nc = nc > 0 ? 1.0f / sqrtf((float)nc) : 0;  /* sqrt norm: 24x stronger than 1/nc */
        float inv_nb = nb > 0 ? 1.0f / sqrtf((float)nb) : 0;  /* sqrt norm: 24x stronger than 1/nb */

        for (int j = 0; j < mlp_dim; j++) {
            if (mask[j] == 2) continue;
            float diff = act_a[j] - act_b[j];
            float *wf = &fc->w_float[(size_t)j * n_embd];
            float clip_val = (mask[j] == 0) ? 2.0f : 1.0f;

            if (mask[j] == 0) {  /* CORE */
                float s = diff > 0 ? 1.0f : -1.0f;
                float grad_scale = -alpha * s * inv_nc * lr;
                for (int i = 0; i < n_embd; i++) {
                    float g = grad_scale * (emb_a[i] - emb_b[i]);
                    wf[i] += g;
                    if (wf[i] > clip_val) wf[i] = clip_val;
                    else if (wf[i] < -clip_val) wf[i] = -clip_val;
                }
            } else {  /* BINARY */
                float grad_scale = beta * 2.0f * diff * inv_nb * lr;
                for (int i = 0; i < n_embd; i++) {
                    float g = grad_scale * (emb_a[i] - emb_b[i]);
                    wf[i] -= g;
                    if (wf[i] > clip_val) wf[i] = clip_val;
                    else if (wf[i] < -clip_val) wf[i] = -clip_val;
                }
            }
        }

        /* Also apply gradient to wte (push concept embeddings apart for CORE) */
        /* dL/demb_a[i] = sum_j (grad_w[j] * w[j,i]) */
        for (int i = 0; i < n_embd; i++) {
            float grad_a = 0, grad_b = 0;
            for (int j = 0; j < mlp_dim; j++) {
                if (mask[j] == 2) continue;
                float diff = act_a[j] - act_b[j];
                float wf = fc->w_float[(size_t)j * n_embd + i];
                if (mask[j] == 0) {
                    float s = diff > 0 ? 1.0f : -1.0f;
                    grad_a += -alpha * s * inv_nc * wf;
                    grad_b += alpha * s * inv_nc * wf;
                } else {
                    grad_a += beta * 2.0f * diff * inv_nb * wf;
                    grad_b -= beta * 2.0f * diff * inv_nb * wf;
                }
            }
            /* Apply to token embeddings for each byte of the concept */
            float scale = lr * 0.5f;  /* larger step for embeddings */
            int n_bytes_a = 0, n_bytes_b = 0;
            for (int b = 0; cp->bytes_a[b]; b++) n_bytes_a++;
            for (int b = 0; cp->bytes_b[b]; b++) n_bytes_b++;

            for (int b = 0; cp->bytes_a[b]; b++) {
                int tok = (unsigned char)cp->bytes_a[b];
                if (tok < m->cfg.vocab_size)
                    m->wte[(size_t)tok * n_embd + i] += scale * grad_a / n_bytes_a;
            }
            for (int b = 0; cp->bytes_b[b]; b++) {
                int tok = (unsigned char)cp->bytes_b[b];
                if (tok < m->cfg.vocab_size)
                    m->wte[(size_t)tok * n_embd + i] += scale * grad_b / n_bytes_b;
            }
        }

        free(act_a);
        free(act_b);
    }

    /* Repack to sync w_core and wbits */
    bin_layer_repack(fc);

    return total_loss / N_PROBE_PAIRS;
}

#endif /* LAL_WHITEBOX_PROBE_H */
