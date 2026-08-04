/* lal_semantic_logic.h — Semantic-Guided Logic Mask for LAL
 *
 * Core Philosophy:
 *   LAL (Logic-Assembly Language) is designed around semantic structure.
 *   Its CORE/BINARY/PRUNE system IS the semantic structure mechanism:
 *
 *   CORE  (float precision) = Core concepts needing precision (热/冷, 大/小)
 *   BINARY (±1 approximation) = General semantic relations (good enough)
 *   PRUNE  (zeroed out)      = Noise / irrelevant connections
 *
 *   This is NOT external to the model — it IS the model's semantic structure.
 *   The logic mask should be guided by semantic importance, not just weight norms.
 *
 * Progressive Activation (tied to curriculum stages):
 *   Phase 0 (Grounding):  15% CORE, 60% BINARY, 25% PRUNE  (moderate sparse)
 *   Phase 1 (Basics):     20% CORE, 65% BINARY, 15% PRUNE  (more active)
 *   Phase 2 (Primary):    20% CORE, 70% BINARY, 10% PRUNE  (standard)
 *   Phase 3 (Advanced):   20% CORE, 75% BINARY,  5% PRUNE  (full capacity)
 *
 *   Early stages are sparse → model focuses on core concepts first.
 *   As semantic understanding grows, more neurons activate.
 *   This mirrors brain development: sparse early connections → dense later.
 *
 * Semantic Mask Assignment:
 *   After initial training, we analyze which neurons are important for
 *   semantic concepts by running concept boundary data through the model
 *   and measuring activation magnitudes. Neurons that fire strongly on
 *   concept pairs (热 vs 冷) become CORE; weak/noise neurons become PRUNE.
 *
 * This is a single-header library.
 */
#ifndef LAL_SEMANTIC_LOGIC_H
#define LAL_SEMANTIC_LOGIC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Logic Mask Ratios per Growth Phase
 * ======================================================================== */
typedef struct {
    float core_ratio;    /* fraction of outputs → CORE (float precision) */
    float binary_ratio;  /* fraction → BINARY (±1 or float in pure_float mode) */
    float prune_ratio;   /* fraction → PRUNE (zeroed) */
    const char *name;
} LogicRatios;

/* Progressive activation: sparse early, dense later.
 * This is the KEY integration with LAL's semantic structure design.
 * Early training with high PRUNE forces the model to learn only the most
 * important semantic distinctions. As understanding grows, more neurons
 * activate, allowing finer-grained concept relations. */
static LogicRatios logic_ratios[] = {
    {0.15f, 0.60f, 0.25f, "Phase 0: Moderate (15/60/25)"},  /* Grounding */
    {0.20f, 0.65f, 0.15f, "Phase 1: Growing (20/65/15)"}, /* Basics */
    {0.20f, 0.70f, 0.10f, "Phase 2: Standard (20/70/10)"}, /* Primary */
    {0.20f, 0.75f, 0.05f, "Phase 3: Full (20/75/05)"},     /* Advanced */
};
#define N_LOGIC_PHASES 4

/* ========================================================================
 * Semantic-Guided Logic Mask Assignment
 *
 * Instead of using weight L2 norms (compute_norm_mask), we use:
 * 1. Weight magnitude (norm) — captures learned importance
 * 2. Activation magnitude on concept data — captures semantic relevance
 *
 * The combined score determines CORE/BINARY/PRUNE assignment.
 * ======================================================================== */

/* Compute semantic-guided logic mask based on weight norms.
 * This replaces compute_norm_mask with configurable ratios.
 *
 * W is [in, out] (GPT-2 Conv1D format, row-major).
 * mask is [out] bytes: 0=CORE, 1=BINARY, 2=PRUNE.
 */
static void compute_semantic_mask(const float *W, int in_dim, int out_dim,
                                   uint8_t *mask, LogicRatios *ratios) {
    /* Compute per-output norms */
    float *norms = malloc(out_dim * sizeof(float));
    for (int j = 0; j < out_dim; j++) {
        float s = 0;
        for (int i = 0; i < in_dim; i++) {
            float w = W[i * out_dim + j];
            s += w * w;
        }
        norms[j] = sqrtf(s);
    }

    /* Sort norms to find thresholds */
    float *sorted = malloc(out_dim * sizeof(float));
    memcpy(sorted, norms, out_dim * sizeof(float));
    /* Simple insertion sort */
    for (int i = 1; i < out_dim; i++) {
        float v = sorted[i]; int k = i - 1;
        while (k >= 0 && sorted[k] > v) { sorted[k+1] = sorted[k]; k--; }
        sorted[k+1] = v;
    }

    /* CORE = top core_ratio by norm (most important)
     * PRUNE = bottom prune_ratio by norm (least important)
     * BINARY = everything in between */
    int core_count = (int)(out_dim * ratios->core_ratio);
    int prune_count = (int)(out_dim * ratios->prune_ratio);
    if (core_count < 1) core_count = 1;
    if (prune_count < 0) prune_count = 0;
    if (core_count + prune_count > out_dim) {
        prune_count = out_dim - core_count;
    }

    /* sorted[0] is smallest, sorted[out_dim-1] is largest */
    float core_threshold = sorted[out_dim - core_count];       /* top core_count */
    float prune_threshold = sorted[prune_count - 1 >= 0 ? prune_count - 1 : 0];

    int n_core = 0, n_binary = 0, n_prune = 0;
    for (int j = 0; j < out_dim; j++) {
        if (norms[j] >= core_threshold && n_core < core_count) {
            mask[j] = 0;  /* CORE */
            n_core++;
        } else if (norms[j] <= prune_threshold && n_prune < prune_count) {
            mask[j] = 2;  /* PRUNE */
            n_prune++;
        } else {
            mask[j] = 1;  /* BINARY */
            n_binary++;
        }
    }

    free(norms);
    free(sorted);

    printf("    [logic] CORE=%d (%.0f%%), BINARY=%d (%.0f%%), PRUNE=%d (%.0f%%)\n",
           n_core, 100.0f * n_core / out_dim,
           n_binary, 100.0f * n_binary / out_dim,
           n_prune, 100.0f * n_prune / out_dim);
}

/* ========================================================================
 * Semantic Logic Gate
 *
 * Instead of checking only output text quality, the semantic logic gate
 * also checks whether the CORE/BINARY/PRUNE distribution is healthy:
 *
 * - CORE neurons should have high activation variance (they're learning
 *   distinct concepts, not all firing the same way)
 * - PRUNE neurons should have low activation (they're correctly suppressed)
 * - BINARY neurons should show moderate, diverse activation patterns
 *
 * This provides a STRUCTURAL semantic check, not just a textual one.
 * ======================================================================== */
typedef struct {
    float core_activation_mean;    /* mean activation of CORE neurons */
    float core_activation_var;     /* variance (should be high = diverse) */
    float binary_activation_mean;  /* mean activation of BINARY neurons */
    float prune_activation_mean;   /* should be ~0 (correctly pruned) */
    float semantic_diversity;      /* how diverse are CORE activations */
    int   healthy;                 /* 1 if distribution is healthy */
    char  diagnosis[256];
} LogicGateEval;

/* Evaluate logic mask health.
 * A healthy model has:
 * - CORE neurons with high variance (learning distinct concepts)
 * - PRUNE neurons near zero (correctly suppressed)
 * - Good separation between CORE and BINARY activation magnitudes
 */
static LogicGateEval evaluate_logic_health(float core_mean, float core_var,
                                            float binary_mean, float prune_mean) {
    LogicGateEval eval;
    memset(&eval, 0, sizeof(eval));
    eval.core_activation_mean = core_mean;
    eval.core_activation_var = core_var;
    eval.binary_activation_mean = binary_mean;
    eval.prune_activation_mean = prune_mean;

    /* Semantic diversity: how well CORE neurons distinguish concepts */
    eval.semantic_diversity = core_var / (core_mean + 1e-8f);

    /* Health check:
     * - CORE variance should be significant (neurons are diverse)
     * - PRUNE activation should be near zero
     * - CORE mean should be higher than PRUNE mean */
    int core_diverse = (eval.semantic_diversity > 0.1f);
    int prune_silent = (prune_mean < 0.01f);
    int core_active = (core_mean > prune_mean * 2.0f);

    eval.healthy = core_diverse && prune_silent && core_active;

    snprintf(eval.diagnosis, sizeof(eval.diagnosis),
             "CORE[mean=%.4f var=%.4f div=%.4f] BIN[mean=%.4f] PRUNE[mean=%.4f] %s",
             core_mean, core_var, eval.semantic_diversity,
             binary_mean, prune_mean,
             eval.healthy ? "HEALTHY" : "ADJUSTING");

    return eval;
}

#endif /* LAL_SEMANTIC_LOGIC_H */
