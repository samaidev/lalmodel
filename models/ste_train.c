/* ste_train.c — LAL 纯 STE 白箱监督训练(单阶段,无切换)

这是 LAL 的核心训练方法演示:
- STE (Straight-Through Estimator): 前向用 sign(w) 二值化,反向用 w_float 浮点梯度
- 白箱监督: 每 10 步用 lal_whitebox_probe.h 检查 CORE/BINARY/PRUNE 逻辑电路
- 语义正则化: 用 semantic_regularization_step 主动引导 CORE 差异化反义词
- 不切换阶段/不切换模型 → 避免阶段切换的内存 bug,纯净展示 STE 训练效果

Build: gcc -O3 -march=native -o ste_train models/ste_train.c runtime/lal_runtime.c -lm
Run:   ./ste_train --steps 200 --batch-size 4
*/
#define _POSIX_C_SOURCE 200809L  /* for strdup */
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>

/* 前向声明(定义在后面) */
static char *g_pieces[40000];
static int g_n_pieces;
static void decode_token(int id);
static int prompt_tokenize(const char *text, int *tokens, int max_len);

#include "../runtime/lal_runtime.h"
#define LAL_DATA_LOADER_IMPLEMENTATION
#include "../runtime/lal_data_loader.h"
#include "../runtime/lal_model_growth.h"
#include "../runtime/lal_semantic_logic.h"
#include "../runtime/lal_whitebox_probe.h"
#include "../runtime/lal_inference_tracer.h"

/* === 全层 CORE/BINARY/PRUNE 逻辑引导 ===
 *
 * LAL 的核心:每层权重按范数分为三类,各自承担不同语义角色:
 *   CORE  (mask=0): 高范数,浮点精确计算 → 引导其最大化反义词对的激活差异
 *   BINARY(mask=1): 中范数,二值符号计算 → 引导其收敛反义词对的激活(共性归纳)
 *   PRUNE (mask=2): 低范数,剪枝静默   → 不更新,保持静默
 *
 * 原始 semantic_regularization_step 只引导 layer 0。本函数对所有层的 mlp_gate
 * 做引导,让逻辑电路在深度方向也形成语义结构。
 */
static float logic_guided_regularization(Model *m, float lr) {
    int n_embd = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    float alpha = 2.0f;   /* CORE 差异化权重 */
    float beta = 0.2f;    /* BINARY 收敛权重 */
    float total_loss = 0.0f;
    int n_guided_layers = 0;

    /* 统计三类神经元总数 */
    int total_core = 0, total_binary = 0, total_prune = 0;

    for (int l = 0; l < n_layer; l++) {
        BinLayer *fc = &m->layers[l].mlp_gate;
        uint8_t *mask = fc->logic_mask;
        if (!mask) continue;

        int mlp_dim = fc->out_dim;
        int in_dim = fc->in_dim;

        /* 统计该层三类神经元数 */
        int nc = 0, nb = 0, np = 0;
        for (int j = 0; j < mlp_dim; j++) {
            if (mask[j] == 0) nc++;
            else if (mask[j] == 1) nb++;
            else np++;
        }
        total_core += nc;
        total_binary += nb;
        total_prune += np;

        /* 对每个反义词对,计算激活并引导 */
        for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
            ConceptPair *cp = &probe_pairs[p];

            float *emb_a = (float *)malloc(n_embd * sizeof(float));
            float *emb_b = (float *)malloc(n_embd * sizeof(float));
            get_concept_embedding(m, cp->bytes_a, emb_a, n_embd);
            get_concept_embedding(m, cp->bytes_b, emb_b, n_embd);

            /* BUG #19 FIX: 用 compute_gate_input 算出该层 mlp_gate 真实输入
             * (经过 LN1 + attn + residual + LN2),不再用原始 wte 喂 mlp_gate。
             * 这让 simulate_activation 算出的激活和真实 forward 一致,
             * 梯度也是在"真实输入空间"上算的,不是原始 embedding 空间。 */
            float *gate_a = (float *)malloc(n_embd * sizeof(float));
            float *gate_b = (float *)malloc(n_embd * sizeof(float));
            compute_gate_input(m, emb_a, l, gate_a, n_embd);
            compute_gate_input(m, emb_b, l, gate_b, n_embd);

            /* 模拟该层激活:gate_input * W (CORE 用 w_core, BINARY 用 w_float) */
            float *act_a = (float *)malloc(mlp_dim * sizeof(float));
            float *act_b = (float *)malloc(mlp_dim * sizeof(float));
            simulate_activation(m, gate_a, l, act_a, mlp_dim);
            simulate_activation(m, gate_b, l, act_b, mlp_dim);

            /* 计算损失: CORE 最大化 |diff|, BINARY 最小化 diff^2
             * BUG #30 FIX: 旧代码对所有层都乘 * 0.5f (注释说"深层权重衰减"但实际
             *   连 layer 0 也乘了)。layer_lr_scale 已经在梯度里处理了深层衰减
             *   (layer 0 = 1.0, 深层 = 0.5),损失里不该再乘。现在去掉 * 0.5f,
             *   让损失反映真实的 CORE 差异化量级,梯度衰减完全由 layer_lr_scale 控制。 */
            int layer_nc = 0, layer_nb = 0;
            for (int j = 0; j < mlp_dim; j++) {
                if (mask[j] == 2) continue;  /* PRUNE: 跳过,保持静默 */
                float diff = act_a[j] - act_b[j];
                float adiff = fabsf(diff);
                if (mask[j] == 0) {  /* CORE: 最大化 |diff| */
                    total_loss -= alpha * adiff;
                    layer_nc++;
                } else {  /* BINARY: 最小化 diff^2 */
                    total_loss += beta * diff * diff;
                    layer_nb++;
                }
            }

            /* === 将语义引导梯度累加到 grad_accum,由 LAL-aware Adam 统一更新 ===
             * 之前直接改 w_float 绕过了 Adam,导致:
             * 1. 没有 g_core_lr_multiplier=3.0 的 CORE 加成
             * 2. 没有分组二阶矩归一化
             * 3. 与主训练循环的 Adam 更新冲突
             * 现在改为:梯度 → grad_accum → model_batch_apply(LAL-aware Adam)
             *
             * BUG #19: 梯度公式 dL/dw[j,i] = grad_scale * (gate_a[i] - gate_b[i])
             *   不是 (emb_a[i] - emb_b[i])!因为 s = w · gate_input,不是 w · emb。
             *   之前用 emb 算梯度,相当于在错误的输入空间上优化,引导方向偏了。
             *
             * BUG #23 FIX: 旧代码把 *lr 乘进 grad_scale,但 model_batch_apply
             *   再用 lr 缩放 grad_accum → 逻辑梯度被乘了 lr^2 (≈2.5e-6 而非 0.002)。
             *   现在去掉 *lr,让 logic_lr 表示"逻辑梯度相对于主训练梯度的倍率"
             *   (logic_lr=1.0 表示逻辑梯度与主梯度同量级,>1 表示逻辑引导更强)。
             *   实际有效 lr = logic_lr * base_lr (例如 4.0 * 0.0005 = 0.002)。 */

            float inv_nc = layer_nc > 0 ? 1.0f / sqrtf((float)layer_nc) : 0;
            float inv_nb = layer_nb > 0 ? 1.0f / sqrtf((float)layer_nb) : 0;

            for (int j = 0; j < mlp_dim; j++) {
                if (mask[j] == 2) continue;  /* PRUNE: 无梯度 */
                float diff = act_a[j] - act_b[j];
                float *ga = &fc->grad_accum[(size_t)j * in_dim];  /* 累加到 grad_accum */
                /* 深层 lr 衰减:layer 0 全量,深层 0.5x */
                float layer_lr_scale = (l == 0) ? 1.0f : 0.5f;

                if (mask[j] == 0) {  /* CORE: 梯度推动差异化 */
                    /* Loss L = -alpha * |diff|, minimize → dL/dw = -alpha*sign(diff)*(gate_a-gate_b)
                     * w -= lr*grad_accum → w += lr*alpha*sign(diff)*(gate_a-gate_b) → diff 增大 ✓
                     *
                     * BUG #23 FIX: 旧代码在 grad_scale 中乘 *lr (=logic_lr=0.002),
                     *   但 model_batch_apply 又乘一次 main_lr=0.0005 → 逻辑梯度被乘 lr^2
                     *   (逻辑梯度 ≈ 1e-6, 主梯度 ≈ 5e-4, 相差 500x,逻辑引导几乎无效).
                     *   而且由于 LAL-aware Adam 的 group_v 会 EMA grad^2, 逻辑梯度的缩放会被
                     *   sqrt_v 抵消,使得 *lr 参数实际上对 CORE/BINARY 梯度无影响.
                     *   现在用 lr 作为相对倍率 (1.0 = 同主梯度量级),默认 4.0 (4x 主梯度). */
                    float s = diff > 0 ? 1.0f : -1.0f;
                    float grad_scale = -alpha * s * inv_nc * layer_lr_scale * lr;
                    for (int i = 0; i < in_dim; i++) {
                        float g = grad_scale * (gate_a[i] - gate_b[i]);  /* BUG #19: gate_input, not emb */
                        ga[i] += g;  /* 累加,不直接改 w_float */
                    }
                } else {  /* BINARY: 梯度推动收敛 */
                    /* BUG #21 FIX: Loss L = +beta * diff^2, minimize → dL/dw = +beta*2*diff*(gate_a-gate_b)
                     * w -= lr*grad_accum → w -= lr*beta*2*diff*(gate_a-gate_b) → diff^2 减小 ✓
                     *
                     * 旧代码用 ga[i] -= grad_scale * ... (= grad_accum -= beta*2*diff*(gate_a-gate_b))
                     * 这相当于 grad_accum 取负,update 时 w += lr*beta*2*diff*(gate_a-gate_b)
                     * → diff^2 被最大化了!收敛目标完全反了。
                     * 这解释了为什么 BIN diff 一直和 CORE diff 差不多大 (ratio≈0.99):
                     * BINARY 神经元没在收敛,而是在被推开。
                     *
                     * BUG #23 FIX: lr 现在是相对倍率 (见 CORE 注释). */
                    float grad_scale = beta * 2.0f * diff * inv_nb * layer_lr_scale * lr;
                    for (int i = 0; i < in_dim; i++) {
                        float g = grad_scale * (gate_a[i] - gate_b[i]);  /* BUG #19: gate_input, not emb */
                        ga[i] += g;  /* BUG #21: was -=, should be += (sign was reversed) */
                    }
                }
            }

            free(emb_a); free(emb_b);
            free(gate_a); free(gate_b);
            free(act_a); free(act_b);
        }

        /* 不再调 bin_layer_repack — 由 model_batch_apply 的 LAL-aware Adam 统一更新 */
        n_guided_layers++;
    }

    /* 注:wte 推开试验过,但 byte-level 下反义词共享首 byte(热=0xE783AD, 冷=0xE586B7),
     * 推开单个 byte embedding 会污染所有以该 byte 开头的字符,适得其反。
     * embedding 分化应通过 CORE 引导间接实现,不直接改 wte。 */

    /* 返回时通过静态变量传递统计(简化接口) */
    static int last_core = 0, last_binary = 0, last_prune = 0;
    last_core = total_core;
    last_binary = total_binary;
    last_prune = total_prune;

    return total_loss / (N_PROBE_PAIRS * n_guided_layers);
}

/* 获取上次逻辑引导的三类神经元统计 */
static void get_logic_stats(Model *m, int *core, int *binary, int *prune) {
    int tc = 0, tb = 0, tp = 0;
    for (int l = 0; l < m->cfg.n_layer; l++) {
        BinLayer *fc = &m->layers[l].mlp_gate;
        if (!fc->logic_mask) continue;
        for (int j = 0; j < fc->out_dim; j++) {
            if (fc->logic_mask[j] == 0) tc++;
            else if (fc->logic_mask[j] == 1) tb++;
            else tp++;
        }
    }
    *core = tc; *binary = tb; *prune = tp;
}

/* === 深度白箱诊断:解释为什么生成是乱码 ===
 *
 * 不只看 boundary score,而是检查:
 * 1. embedding 空间是否真的分化(无关概念不应高相似)
 * 2. CORE 激活差异的绝对量级是否足以驱动 logits
 * 3. 三类权重范数是否真的分层(CORE > BINARY > PRUNE)
 * 4. logits 分布是否坍缩(熵太低 → 模式坍缩 → 乱码重复)
 */
static void deep_whitebox_diagnosis(Model *m) {
    int n_embd = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int vocab = m->cfg.vocab_size;

    printf("\n========================================\n");
    printf("  DEEP WHITEBOX DIAGNOSIS\n");
    printf("  (explaining WHY generation is garbage)\n");
    printf("========================================\n\n");

    /* === 1. Embedding 空间分化检查 === */
    printf("--- 1. Embedding Space Differentiation ---\n\n");
    const char *names[] = {"热","冷","大","小","上","下","亮","暗","火","水"};
    const char *bytes[] = {"\xe7\x83\xad","\xe5\x86\xb7","\xe5\xa4\xa7","\xe5\xb0\x8f",
                           "\xe4\xb8\x8a","\xe4\xb8\x8b","\xe4\xba\xae","\xe6\x9a\x97",
                           "\xe7\x81\xab","\xe6\xb0\xb4"};
    int n_concepts = 10;

    static float embs[10][4096];
    for (int i = 0; i < n_concepts; i++)
        get_concept_embedding(m, bytes[i], embs[i], n_embd);

    /* embedding 范数 */
    printf("  Embedding norms:\n    ");
    for (int i = 0; i < n_concepts; i++) {
        float norm = 0;
        for (int j = 0; j < n_embd; j++) norm += embs[i][j] * embs[i][j];
        printf("%s=%.3f ", names[i], sqrtf(norm));
    }
    printf("\n\n");

    /* 反义词 vs 无关概念 平均相似度 */
    float opp_sim = 0, unrelated_sim = 0;
    int n_opp = 0, n_unrel = 0;
    int opp_pairs[][2] = {{0,1},{2,3},{4,5},{6,7},{8,9}};
    for (int p = 0; p < 5; p++) {
        opp_sim += cosine_sim(embs[opp_pairs[p][0]], embs[opp_pairs[p][1]], n_embd);
        n_opp++;
    }
    for (int i = 0; i < n_concepts; i++) {
        for (int j = i+1; j < n_concepts; j++) {
            int is_opp = 0;
            for (int p = 0; p < 5; p++)
                if (opp_pairs[p][0]==i && opp_pairs[p][1]==j) is_opp = 1;
            if (!is_opp) {
                unrelated_sim += cosine_sim(embs[i], embs[j], n_embd);
                n_unrel++;
            }
        }
    }
    opp_sim /= n_opp;
    unrelated_sim /= n_unrel;

    printf("  Avg cosine similarity:\n");
    printf("    Opposite pairs (should be LOW):  %.4f\n", opp_sim);
    printf("    Unrelated pairs (should be ~0):  %.4f\n", unrelated_sim);
    printf("    Gap (unrelated - opposite):       %.4f  ", unrelated_sim - opp_sim);
    if (unrelated_sim - opp_sim < 0.1f && opp_sim > 0.1f) {
        printf("[FAIL] embeddings clustered — concepts not separated\n");
        printf("           → generation will produce near-random tokens\n");
    } else if (opp_sim > 0.2f) {
        printf("[WEAK] opposites still too similar\n");
    } else if (unrelated_sim > 0.3f) {
        printf("[WEAK] unrelated concepts too similar — space not spread out\n");
    } else {
        printf("[OK]\n");
    }
    printf("\n");

    /* === 2. CORE 激活绝对量级 === */
    printf("--- 2. CORE Activation Magnitude ---\n\n");
    int mlp_dim = m->cfg.mlp_dim;
    BinLayer *fc0 = &m->layers[0].mlp_gate;
    if (fc0->logic_mask) {
        /* BUG #19 FIX: 用 compute_gate_input 算 mlp_gate 真实输入 */
        float *gate_hot = malloc(n_embd * sizeof(float));
        float *gate_cold = malloc(n_embd * sizeof(float));
        compute_gate_input(m, embs[0], 0, gate_hot, n_embd);
        compute_gate_input(m, embs[1], 0, gate_cold, n_embd);

        float *act_hot = malloc(mlp_dim * sizeof(float));
        float *act_cold = malloc(mlp_dim * sizeof(float));
        simulate_activation(m, gate_hot, 0, act_hot, mlp_dim);
        simulate_activation(m, gate_cold, 0, act_cold, mlp_dim);

        float core_max = -1e10, core_avg = 0;
        float bin_max = -1e10, bin_avg = 0;
        int nc = 0, nb = 0;
        for (int j = 0; j < mlp_dim; j++) {
            float a = fabsf(act_hot[j]);
            if (fc0->logic_mask[j] == 0) { if(a>core_max)core_max=a; core_avg+=a; nc++; }
            else if (fc0->logic_mask[j] == 1) { if(a>bin_max)bin_max=a; bin_avg+=a; nb++; }
        }
        core_avg /= nc; bin_avg /= nb;

        printf("  Layer 0 mlp activation (for '热'):\n");
        printf("    CORE  : max=%.4f avg=%.4f  (n=%d)\n", core_max, core_avg, nc);
        printf("    BINARY: max=%.4f avg=%.4f  (n=%d)\n", bin_max, bin_avg, nb);

        float core_diff = 0;
        for (int j = 0; j < mlp_dim; j++)
            if (fc0->logic_mask[j] == 0)
                core_diff += fabsf(act_hot[j] - act_cold[j]);
        core_diff /= nc;
        printf("    CORE diff (热 vs 冷): %.4f (%.1f%% of avg)\n",
               core_diff, core_avg > 0 ? 100*core_diff/core_avg : 0);
        if (core_diff < 0.05f) {
            printf("    [FAIL] CORE diff too small — cannot drive distinct logits\n");
        } else if (core_diff < 0.2f) {
            printf("    [WEAK] CORE diff marginal — weak discrimination\n");
        } else {
            printf("    [OK]\n");
        }
        printf("\n");
        free(act_hot); free(act_cold);
        free(gate_hot); free(gate_cold);
    }

    /* === 3. 三类权重范数分层 === */
    printf("--- 3. Weight Norm Layering (CORE > BINARY > PRUNE?) ---\n\n");
    int all_ok = 1;
    for (int l = 0; l < n_layer; l++) {
        BinLayer *bl = &m->layers[l].mlp_gate;
        if (!bl->logic_mask) continue;
        float core_norm = 0, bin_norm = 0, prune_norm = 0;
        int nc = 0, nb = 0, np = 0;
        for (int j = 0; j < bl->out_dim; j++) {
            float row_norm = 0;
            for (int i = 0; i < bl->in_dim; i++)
                row_norm += bl->w_float[j*bl->in_dim + i] * bl->w_float[j*bl->in_dim + i];
            row_norm = sqrtf(row_norm);
            if (bl->logic_mask[j] == 0) { core_norm += row_norm; nc++; }
            else if (bl->logic_mask[j] == 1) { bin_norm += row_norm; nb++; }
            else { prune_norm += row_norm; np++; }
        }
        core_norm /= nc; bin_norm /= nb; prune_norm /= np;
        printf("  L%d: CORE=%.4f BIN=%.4f PRUNE=%.4f", l, core_norm, bin_norm, prune_norm);
        if (core_norm > bin_norm * 1.1f && bin_norm > prune_norm * 1.1f) {
            printf("  [OK layered]\n");
        } else if (core_norm > bin_norm * 1.05f) {
            printf("  [WEAK]\n"); all_ok = 0;
        } else {
            printf("  [FAIL: not layered]\n"); all_ok = 0;
        }
    }
    if (!all_ok) printf("\n  → Weight norms not differentiated → logic mask is cosmetic, not functional\n");
    printf("\n");

    /* === 4. Logits 分布 === */
    printf("--- 4. Logits Distribution (mode collapse check) ---\n\n");
    /* BPE 模式用 token id,byte 模式用 byte */
    int tokens[16];
    int n_tok = prompt_tokenize("\xe7\x83\xad", tokens, 16);  /* "热" */
    if (n_tok == 0) {  /* fallback */
        tokens[0] = (unsigned char)'\xe7'; tokens[1] = (unsigned char)'\x83'; tokens[2] = (unsigned char)'\xad';
        n_tok = 3;
    }
    float *logits = malloc(vocab * sizeof(float));
    model_forward_float_logits(m, tokens, n_tok, logits);

    /* top-5 */
    printf("  Prompt '热' → top-5 next-token logits:\n");
    /* used 标记数组要覆盖整个 vocab */
    static int used[40000];
    memset(used, 0, vocab * sizeof(int));
    for (int k = 0; k < 5; k++) {
        int best = -1;
        float bestv = -1e10;
        for (int i = 0; i < vocab; i++) {
            if (!used[i] && logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        if (best < 0) break;
        used[best] = 1;
        printf("    token %d '", best);
        decode_token(best);
        printf("' logit=%.4f\n", logits[best]);
    }

    float max_l = logits[0], min_l = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > max_l) max_l = logits[i];
        if (logits[i] < min_l) min_l = logits[i];
    }
    float exp_sum = 0;
    for (int i = 0; i < vocab; i++) exp_sum += expf(logits[i] - max_l);
    float entropy = 0;
    for (int i = 0; i < vocab; i++) {
        float p = expf(logits[i] - max_l) / exp_sum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    printf("\n  Logit range: %.4f (max=%.4f min=%.4f)\n", max_l-min_l, max_l, min_l);
    printf("  Softmax entropy: %.4f bits (uniform would be %.2f)\n",
           entropy/logf(2), logf(256)/logf(2));
    if (entropy < 1.0f) {
        printf("  [FAIL] entropy %.2f bits — MODE COLLAPSE, generation will repeat same token\n", entropy/logf(2));
    } else if (max_l - min_l < 2.0f) {
        printf("  [FAIL] logit range %.2f too flat — near-uniform, generation is random noise\n", max_l-min_l);
    } else if (entropy < 3.0f) {
        printf("  [WEAK] entropy %.2f bits — low diversity, generation will be repetitive\n", entropy/logf(2));
    } else {
        printf("  [OK]\n");
    }
    printf("\n");
    free(logits);

    /* === 总结 === */
    printf("========================================\n");
    printf("  DIAGNOSIS SUMMARY\n");
    printf("========================================\n");
    printf("  If generation is garbage, likely causes:\n");
    printf("  1. Embeddings not spread (Section 1) → tokens sampled near-randomly\n");
    printf("  2. CORE activation too weak (Section 2) → model can't distinguish inputs\n");
    printf("  3. Weight norms not layered (Section 3) → logic mask is cosmetic\n");
    printf("  4. Logits collapsed/flat (Section 4) → repetitive or random output\n");
    printf("  Fix: train more steps, or increase logic_guided_regularization lr\n");
    printf("========================================\n\n");
}

/* === 权重保存/加载(用于断点续训) ===
 * 自定义二进制格式 .ste:
 *   magic(4) = "STEW"
 *   n_layer, n_embd, mlp_dim, vocab (各 4 字节)
 *   wte[vocab * n_embd] (float)
 *   ln_f_w[n_embd], ln_f_b[n_embd] (float)
 *   每层:
 *     mlp_gate: w_float[in*out], bias[out], alpha[out], logic_mask[out]
 *     attn_q:   w_float, bias, alpha, logic_mask
 *     attn_o:   w_float, bias, alpha, logic_mask
 *     (attn_k/v if !qkv_merged)
 *     mlp_down: w_float, bias, alpha, logic_mask
 *     (mlp_up if SwiGLU)
 *     norm1_w/b, norm2_w/b
 */
static int ste_save(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    fwrite("STEW", 1, 4, f);
    int n_layer = m->cfg.n_layer, n_embd = m->cfg.n_embd;
    int mlp_dim = m->cfg.mlp_dim, vocab = m->cfg.vocab_size;
    fwrite(&n_layer, 4, 1, f);
    fwrite(&n_embd, 4, 1, f);
    fwrite(&mlp_dim, 4, 1, f);
    fwrite(&vocab, 4, 1, f);

    /* wte */
    fwrite(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    /* wpe */
    if (m->wpe) {
        fwrite(m->wpe, sizeof(float), (size_t)512 * n_embd, f);  /* n_ctx=512 */
    }
    /* ln_f */
    fwrite(m->ln_f_w, sizeof(float), n_embd, f);
    fwrite(m->ln_f_b, sizeof(float), n_embd, f);

    /* 每层 */
    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        /* 保存所有 BinLayer */
        BinLayer *bls[8] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) { bls[4] = &tl->attn_k; bls[5] = &tl->attn_v; n_bl = 6; }
        if (m->cfg.act_type == ACT_SWIGLU) { bls[n_bl] = &tl->mlp_up; n_bl++; }

        for (int b = 0; b < n_bl; b++) {
            BinLayer *bl = bls[b];
            fwrite(bl->w_float, sizeof(float), (size_t)bl->in_dim * bl->out_dim, f);
            fwrite(bl->bias, sizeof(float), bl->out_dim, f);
            fwrite(bl->alpha, sizeof(float), bl->out_dim, f);
            if (bl->logic_mask)
                fwrite(bl->logic_mask, 1, bl->out_dim, f);
        }
        /* norm */
        fwrite(tl->norm1_w, sizeof(float), n_embd, f);
        fwrite(tl->norm1_b, sizeof(float), n_embd, f);
        fwrite(tl->norm2_w, sizeof(float), n_embd, f);
        fwrite(tl->norm2_b, sizeof(float), n_embd, f);
    }
    fclose(f);
    return 0;
}

static int ste_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "STEW", 4) != 0) { fclose(f); return -1; }

    int n_layer, n_embd, mlp_dim, vocab;
    fread(&n_layer, 4, 1, f);
    fread(&n_embd, 4, 1, f);
    fread(&mlp_dim, 4, 1, f);
    fread(&vocab, 4, 1, f);

    if (n_layer != m->cfg.n_layer || n_embd != m->cfg.n_embd ||
        mlp_dim != m->cfg.mlp_dim || vocab != m->cfg.vocab_size) {
        fprintf(stderr, "[!] ste_load: config mismatch\n");
        fclose(f);
        return -1;
    }

    fread(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    /* wpe */
    if (m->wpe) {
        fread(m->wpe, sizeof(float), (size_t)512 * n_embd, f);
    }
    fread(m->ln_f_w, sizeof(float), n_embd, f);
    fread(m->ln_f_b, sizeof(float), n_embd, f);

    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        BinLayer *bls[8] = {&tl->attn_q, &tl->attn_o, &tl->mlp_gate, &tl->mlp_down};
        int n_bl = 4;
        if (!m->cfg.qkv_merged) { bls[4] = &tl->attn_k; bls[5] = &tl->attn_v; n_bl = 6; }
        if (m->cfg.act_type == ACT_SWIGLU) { bls[n_bl] = &tl->mlp_up; n_bl++; }

        for (int b = 0; b < n_bl; b++) {
            BinLayer *bl = bls[b];
            fread(bl->w_float, sizeof(float), (size_t)bl->in_dim * bl->out_dim, f);
            fread(bl->bias, sizeof(float), bl->out_dim, f);
            fread(bl->alpha, sizeof(float), bl->out_dim, f);
            if (bl->logic_mask) {
                fread(bl->logic_mask, 1, bl->out_dim, f);
                /* 重建 w_core 和 wbits */
                bin_layer_repack(bl);
            }
        }
        fread(tl->norm1_w, sizeof(float), n_embd, f);
        fread(tl->norm1_b, sizeof(float), n_embd, f);
        fread(tl->norm2_w, sizeof(float), n_embd, f);
        fread(tl->norm2_b, sizeof(float), n_embd, f);
    }
    fclose(f);
    return 0;
}

/* === BPE vocab 加载(id → UTF-8 piece) ===
 * SentencePiece .vocab 格式:每行 "piece\tscore",score 是浮点数。
 * token id = 行号(0-indexed)。 */
/* g_pieces 和 g_n_pieces 已在前向声明 */

static void load_vocab(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[!] cannot load vocab %s\n", path); return; }
    char line[512];
    int line_no = 0;  /* 行号 = token id (SentencePiece .vocab 格式: piece\tscore) */
    while (fgets(line, sizeof(line), f)) {
        if (line_no >= 40000) break;
        /* 格式: piece\tscore (score 是浮点数或负数,id 是行号) */
        char *tab = strchr(line, '\t');
        if (!tab) { line_no++; continue; }
        *tab = '\0';
        /* 去掉行尾换行 */
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        g_pieces[line_no] = strdup(line);
        line_no++;
    }
    g_n_pieces = line_no;
    fclose(f);
    printf("[*] Loaded %d vocab pieces from %s\n", g_n_pieces, path);
}

static void decode_token(int id) {
    if (id < 0 || id >= g_n_pieces || !g_pieces[id]) {
        /* byte fallback: id 0-255 对应 byte */
        if (id >= 0 && id < 256) putchar(id);
        else { putchar(0xEF); putchar(0xBF); putchar(0xBD); }  /* U+FFFD */
        return;
    }

    /* BUG #32 FIX: SentencePiece byte-fallback tokens (id 3-258) are stored
     * as literal strings like "<0xE4>" in g_pieces. If we fputs them, the
     * output shows the 6-char string "<0xE4>" instead of the actual byte 0xE4.
     * This corrupts multi-byte character output: a 3-byte CJK char encoded as
     * [<0xE4>, <0xB8>, <0xAD>] would print "<0xE4><0xB8><0xAD>" instead of "中".
     *
     * Fix: detect byte-fallback tokens by their "<0xNN>" format and output
     * the actual byte value. */
    const char *piece = g_pieces[id];
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>' && piece[6] == '\0') {
        /* Parse hex: <0xAB> → byte 0xAB */
        int hi = (piece[3] >= 'a') ? (piece[3] - 'a' + 10) :
                 (piece[3] >= 'A') ? (piece[3] - 'A' + 10) : (piece[3] - '0');
        int lo = (piece[4] >= 'a') ? (piece[4] - 'a' + 10) :
                 (piece[4] >= 'A') ? (piece[4] - 'A' + 10) : (piece[4] - '0');
        if (hi >= 0 && hi <= 15 && lo >= 0 && lo <= 15) {
            putchar(hi * 16 + lo);
            return;
        }
    }

    /* 正常 piece: 输出 UTF-8 字节 */
    fputs(piece, stdout);
}

/* 用 vocab 把文本编码成 token id。
 * BUG #31 FIX: 旧代码逐字符查 vocab,但 SentencePiece BPE 会合并常见词
 * (如 "是什么" 是单个 token id=354)。逐字符编码产出 [是,什,么] 而非 [是什么],
 * 与训练数据的 BPE 编码不一致 → 生成时模型看到的 token 序列和训练时不同。
 *
 * 修复:用真正的 SentencePiece 编码(如果可用)。否则回退到逐字符。 */
static int prompt_tokenize(const char *text, int *tokens, int max_len) {
    int n = 0;
    /* 如果 vocab 未加载(byte 模式),回退到 byte tokenize */
    if (g_n_pieces == 0) {
        for (int i = 0; text[i] && n < max_len; i++)
            tokens[n++] = (unsigned char)text[i];
        return n;
    }

    /* BUG #31 FIX: 先尝试用 SentencePiece 编码(通过外部 python 脚本预处理过的
     * vocab 不支持,但我们可以用贪心最长匹配来近似 BPE 合并)。
     * 对每个位置,尝试匹配最长的 vocab piece (最多 8 字节)。 */
    int text_len = (int)strlen(text);
    int i = 0;
    while (i < text_len && n < max_len) {
        /* 提取当前 UTF-8 字符 */
        unsigned char c = (unsigned char)text[i];
        int chlen = 1;
        if (c >= 0xF0) chlen = 4;
        else if (c >= 0xE0) chlen = 3;
        else if (c >= 0xC0) chlen = 2;

        /* 贪心最长匹配:尝试从最长到最短,但不能超过剩余文本长度。
         * BPE pieces 可以长达 12 字节 (4 个 CJK 字符合并),所以 max=12。 */
        int max_try = text_len - i;
        if (max_try > 12) max_try = 12;
        int found = -1;
        int found_len = 0;
        for (int try_len = max_try; try_len >= 1; try_len--) {
            char piece[13] = {0};
            memcpy(piece, &text[i], try_len);
            piece[try_len] = '\0';
            /* 线性搜索 vocab (慢,但只在生成时调用一次,可接受) */
            for (int id = 0; id < g_n_pieces; id++) {
                if (g_pieces[id] && strcmp(g_pieces[id], piece) == 0) {
                    found = id;
                    found_len = try_len;
                    break;
                }
            }
            if (found >= 0) break;
        }

        if (found >= 0) {
            tokens[n++] = found;
            i += found_len;
        } else {
            /* 找不到匹配,用 byte-fallback (3 bytes for CJK) */
            for (int b = 0; b < chlen && n < max_len; b++) {
                /* byte-fallback token id = 3 + byte_value (SentencePiece 约定) */
                int byte_tok = 3 + (unsigned char)text[i + b];
                if (byte_tok < g_n_pieces) tokens[n++] = byte_tok;
            }
            i += chlen;
        }
    }
    return n;
}

/* === Byte-Level Tokenizer (LAL 风格,vocab=256) === */
static int byte_tokenize(const char *text, int *tokens, int max_len) {
    int n = 0;
    for (int i = 0; text[i] && n < max_len; i++)
        tokens[n++] = (unsigned char)text[i];
    return n;
}

static int sample_argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_s = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > best_s) { best_s = logits[i]; best = i; }
    return best;
}

/* === STE 训练:单阶段,固定模型 === */
static float ste_train(Model *m, DataLoader *dl, int n_steps, float base_lr,
                       int warmup, int batch_size, int max_pos, int eval_interval,
                       float logic_lr) {
    printf("\n=== LAL STE Training (B=%d, max_pos=%d, steps=%d, logic_lr=%.4f) ===\n",
           batch_size, max_pos, n_steps, logic_lr);
    printf("[*] STE: forward=sign(w), backward=w_float gradient\n");
    printf("[*] Whitebox probe every %d steps\n", eval_interval);
    printf("[*] Semantic regularization every step\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int **batch_tokens = malloc(batch_size * sizeof(int*));
    int *batch_lens = malloc(batch_size * sizeof(int));
    for (int i = 0; i < batch_size; i++) batch_tokens[i] = malloc(512 * sizeof(int));

    #define WINDOW_SIZE 64
    float loss_window[WINDOW_SIZE];
    int loss_idx = 0;
    float best_recent_loss = 1e10f;

    for (int step = 0; step < n_steps; step++) {
        float lr = lr_schedule(step, warmup, n_steps, base_lr);
        model_batch_begin(m);

        float batch_loss = 0;
        int n_valid = 0;

        for (int b = 0; b < batch_size; b++) {
            int idx = rand() % dl->n_samples;
            int n_tok = dataloader_get(dl, idx, batch_tokens[b], 512);
            if (n_tok < 20) continue;

            int mp = n_tok < max_pos ? n_tok : max_pos;
            if (mp < 10) continue;

            /* Multi-position prediction: 4 positions per sample */
            int n_preds = 4;
            int stride = (mp - 6) / n_preds;
            if (stride < 1) stride = 1;

            for (int p = 0; p < n_preds; p++) {
                int pred_pos = 5 + p * stride;
                if (pred_pos >= mp - 1) break;
                float loss = model_batch_forward(m, batch_tokens[b], pred_pos + 1);
                if (!isnan(loss) && !isinf(loss)) {
                    model_batch_backward(m, batch_tokens[b], pred_pos + 1);
                    batch_loss += loss;
                    n_valid++;
                }
            }
        }

        if (n_valid > 0) {
            /* 在 apply 之前加语义引导梯度,与训练梯度一起更新 */
            float logic_loss = logic_guided_regularization(m, logic_lr);
            model_batch_apply(m, lr, n_valid);
            /* g_opt_step is incremented inside model_batch_apply — don't double-increment */
            if (step % 10 == 0) {
                printf("  [LOGIC] loss=%.4f\n", logic_loss);
            }
        }

        float avg_loss = n_valid > 0 ? batch_loss / n_valid : 0;
        loss_window[loss_idx] = avg_loss;
        loss_idx = (loss_idx + 1) % WINDOW_SIZE;
        int n_win = step + 1 < WINDOW_SIZE ? step + 1 : WINDOW_SIZE;
        float recent_loss = 0;
        for (int i = 0; i < n_win; i++) recent_loss += loss_window[i];
        recent_loss /= n_win;

        if (step % eval_interval == 0 || step == n_steps - 1) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
            printf("  step %5d/%d  loss=%.4f  avg=%.4f  lr=%.6f  B=%d  %.0fms\n",
                   step, n_steps, avg_loss, recent_loss, lr, n_valid,
                   dt / (step + 1) * 1000);
            fflush(stdout);

            if (recent_loss < best_recent_loss) best_recent_loss = recent_loss;
        }

        /* 白箱探针:每 10 步检查 CORE/BINARY/PRUNE 逻辑电路 */
        if (step % 10 == 0) {
            ProbeMetrics pm = whitebox_probe_compact(m);
            printf("  [WB] boundary=%.0f/100  core_diff=%.4f  bin_diff=%.4f  ratio=%.2f  prune_act=%.4f\n",
                   pm.boundary_score, pm.core_diff_avg, pm.bin_diff_avg,
                   pm.core_bin_ratio, pm.prune_act_avg);
            printf("  [WB] structure=%d/%d (%.0f%%)  opp_sim=%.3f\n",
                   pm.n_layers_ok, pm.n_layers_total,
                   pm.n_layers_total > 0 ? 100.0f * pm.n_layers_ok / pm.n_layers_total : 0,
                   1.0f - pm.boundary_score / 100.0f);
            printf("  [WB] %s | %s | %s\n",
                   pm.boundary_score > 70 ? "BOUNDARY OK" : "BOUNDARY weak",
                   pm.core_bin_ratio > 1.0f ? "CORE>BIN OK" : "CORE<BIN!",
                   pm.prune_act_avg < 0.01f ? "PRUNE silent OK" : "PRUNE noisy");
        }

        /* 逻辑引导已在 model_batch_apply 前累加梯度 */

        /* 每 50 步做一次推理 trace */
        if (step > 0 && step % 50 == 0) {
            const char *trace_prompts[] = {
                "\xe7\x83\xad", "\xe5\x86\xb7", "\xe7\x81\xab",
                "\xe6\xb0\xb4", "\xe5\xa4\xa7", "\xe5\xb0\x8f"
            };
            const char *trace_labels[] = {"热","冷","火","水","大","小"};
            int ti = (step / 50) % 6;
            inference_trace_compact(m, trace_prompts[ti], trace_labels[ti]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("\n[*] STE training done in %.1fs (%.1f ms/step)\n", dt, dt/n_steps*1000);
    printf("[*] Best recent loss: %.4f\n\n", best_recent_loss);

    for (int i = 0; i < batch_size; i++) free(batch_tokens[i]);
    free(batch_tokens);
    free(batch_lens);
    return best_recent_loss;
}

/* === 生成文本 (用 stateful sliding window inference) === */
static void generate_text(Model *m, const char *prompt, const int *prompt_ids,
                          int n_prompt_ids, int max_gen, float temp, int top_k) {
    printf("\n=== Generation (temp=%.2f, top_k=%d, max=%d tokens) ===\n",
           temp, top_k, max_gen);
    g_use_adam = 0;
    g_use_pure_float = 1;  /* 生成用 pure_float(完整浮点),避免 bin_forward 二值化丢失输入区分度 */
    g_use_real_attention = 1;

    int prompt_tokens[512];
    int prompt_len;
    if (n_prompt_ids > 0) {
        /* 直接用 BPE token id */
        prompt_len = n_prompt_ids < 512 ? n_prompt_ids : 512;
        memcpy(prompt_tokens, prompt_ids, prompt_len * sizeof(int));
        printf("[*] prompt ids (%d):", prompt_len);
        for (int i = 0; i < prompt_len; i++) printf(" %d", prompt_tokens[i]);
        printf("\n");
    } else {
        prompt_len = prompt_tokenize(prompt, prompt_tokens, 512);
        printf("[*] prompt '%s' → %d tokens\n", prompt, prompt_len);
    }

    model_stateful_begin(m);
    model_set_sliding_window(m, m->cfg.sliding_window, m->cfg.n_sinks);

    const float *logits = NULL;
    for (int i = 0; i < prompt_len; i++) {
        logits = model_stateful_forward_sliding(m, prompt_tokens[i]);
    }
    if (!logits) { printf("[!] no logits\n"); return; }

    printf("输出: ");
    if (n_prompt_ids > 0) {
        for (int i = 0; i < prompt_len; i++) decode_token(prompt_tokens[i]);
    } else {
        printf("%s", prompt);
    }
    fflush(stdout);

    int out_tokens[2048];
    int out_len = 0;
    for (int i = 0; i < prompt_len; i++) out_tokens[out_len++] = prompt_tokens[i];

    int vocab = m->cfg.vocab_size;
    /* 重复惩罚:对已生成的 token 降权 */
    static float penalty[40000];
    memset(penalty, 0, vocab * sizeof(float));

    for (int step = 0; step < max_gen && out_len < 2000; step++) {
        int next;

        /* 应用重复惩罚:已生成的 token logit 除以 1.5 */
        static float adjusted_logits[40000];
        memcpy(adjusted_logits, logits, vocab * sizeof(float));

        /* BUG #24 FIX: mask special tokens during generation.
         * SentencePiece vocab reserves ids 0-2 for <unk>, <s>, </s>.
         * These should NEVER be generated (the model can emit them due to
         * untrained bias, polluting output with literal strings like
         * "</s>" or "<s>"). Set their logits to -INFINITY.
         * Byte-fallback tokens (id 3-258, format <0xNN>) are kept because
         * the model legitimately needs them for OOV characters. */
        adjusted_logits[0] = -1e30f;  /* <unk> */
        adjusted_logits[1] = -1e30f;  /* <s> */
        adjusted_logits[2] = -1e30f;  /* </s> */

        for (int j = 0; j < out_len; j++) {
            if (out_tokens[j] >= 0 && out_tokens[j] < vocab) {
                if (adjusted_logits[out_tokens[j]] > 0)
                    adjusted_logits[out_tokens[j]] /= 1.5f;
                else
                    adjusted_logits[out_tokens[j]] *= 1.5f;
            }
        }
        
        if (temp <= 0.01f) {
            next = sample_argmax(adjusted_logits, vocab);
        } else {
            /* top-p (nucleus) 采样:选概率累积到 p 的 token 集合 */
            static int indices[40000];
            for (int j = 0; j < vocab; j++) indices[j] = j;
            /* 部分排序:按 logit 降序 */
            int k_limit = top_k < vocab ? top_k : vocab;
            for (int j = 0; j < k_limit; j++) {
                int max_idx = j;
                for (int k = j + 1; k < vocab; k++)
                    if (adjusted_logits[indices[k]] > adjusted_logits[indices[max_idx]]) max_idx = k;
                /* BUG #25 FIX: 旧代码写错成 indices[j] = tmp (无操作,没真正交换)
                 * → indices 保持 0,1,2,...,k_limit-1 → 被mask的 token 0 被当 max_l,
                 * 其他 token 的 expf 溢出成 +inf → sum=inf, r=inf → 采样永远 fallback
                 * 到 indices[0]=0 → 一直输出 <unk>。
                 * 正确的交换是 indices[max_idx] = tmp。 */
                int tmp = indices[j]; indices[j] = indices[max_idx]; indices[max_idx] = tmp;
            }
            float max_l = adjusted_logits[indices[0]];
            static float probs[40000];
            float sum = 0;
            for (int j = 0; j < k_limit; j++) {
                probs[j] = expf((adjusted_logits[indices[j]] - max_l) / temp);
                sum += probs[j];
            }
            /* top-p: 找累积概率达到 0.9 的截断点 */
            float cum_thresh = 0.9f * sum;
            int p_limit = k_limit;
            float cum = 0;
            for (int j = 0; j < k_limit; j++) {
                cum += probs[j];
                if (cum > cum_thresh) { p_limit = j + 1; break; }
            }
            /* 在 top-p 范围内采样 */
            float r = (float)rand() / RAND_MAX * cum;
            cum = 0;
            next = indices[0];
            for (int j = 0; j < p_limit; j++) {
                cum += probs[j];
                if (r < cum) { next = indices[j]; break; }
            }
        }

        out_tokens[out_len++] = next;
        decode_token(next);
        fflush(stdout);
        /* BUG #24: stop on </s> (id=2) instead of <unk> (id=0).
         * <unk> is now masked so never generated; </s> is the proper EOS. */
        if (next == 2) break;
        logits = model_stateful_forward_sliding(m, next);
        if (!logits) break;
    }
    printf("\n");
}

/* === Main === */
int main(int argc, char **argv) {
    int n_steps = 200;
    int batch_size = 4;
    float base_lr = 0.0005f;
    float logic_lr = 1.0f;  /* --logic-lr 调整:相对倍率 (1.0=同主梯度量级) */
    int phase_idx = 0;
    int vocab_size = 256;  /* --vocab 256=byte, 32768=BPE */
    const char *data_path = "data/curriculum/stage_grounding_combined.bin";
    const char *weights_path = "/tmp/lal_ste_model.bin";
    const char *save_path = NULL;   /* --save: 训练后保存 */
    const char *resume_path = NULL; /* --resume: 加载已训练权重续训 */
    const char *prompt = "\xe7\x81\xab\xe6\x98\xaf\xe4\xbb\x80\xe4\xb9\x88";  /* 火是什么 */
    int prompt_ids[64];      /* --prompt-ids 直接传 BPE token id */
    int n_prompt_ids = 0;    /* >0 表示用 --prompt-ids 而非 --prompt */
    int do_generate = 1;
    int max_gen = 100;
    float temp = 0.4f;
    int top_k = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-size") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) base_lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--logic-lr") && i+1 < argc) logic_lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--phase") && i+1 < argc) phase_idx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i+1 < argc) data_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--prompt-ids") && i+1 < argc) {
            /* 逗号分隔的 token id,如 "32646,31888,31923,31922" */
            char *p = argv[++i];
            n_prompt_ids = 0;
            while (*p && n_prompt_ids < 64) {
                prompt_ids[n_prompt_ids++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        else if (!strcmp(argv[i], "--max-gen") && i+1 < argc) max_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-generate")) do_generate = 0;
        else if (!strcmp(argv[i], "--save") && i+1 < argc) save_path = argv[++i];
        else if (!strcmp(argv[i], "--resume") && i+1 < argc) resume_path = argv[++i];
        else if (!strcmp(argv[i], "--diagnose-only")) { n_steps = 0; }
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: ste_train [options]\n"
                   "  --steps N         Training steps (default 200)\n"
                   "  --batch-size N    Batch size (default 4)\n"
                   "  --lr F            Base learning rate (default 0.0005)\n"
                   "  --logic-lr F      Logic guidance multiplier (default 1.0 = same as main grad)\n"
                   "  --data PATH       Data .bin path\n"
                   "  --save PATH       Save trained weights to .ste file\n"
                   "  --resume PATH     Load .ste weights and continue training\n"
                   "  --diagnose-only   Skip training, just load + diagnose + generate\n"
                   "  --prompt TEXT     Generation prompt\n"
                   "  --max-gen N       Max generation tokens (default 100)\n"
                   "  --temp F          Sampling temperature (default 0.4)\n"
                   "  --no-generate     Skip generation\n");
            return 0;
        }
    }

    srand(time(NULL));
    setvbuf(stdout, NULL, _IONBF, 0);

    /* BPE 模式加载 vocab */
    if (vocab_size > 256) {
        load_vocab("tokenizer/chinese_bpe.vocab");
    }

    printf("=== LAL STE Whitebox-Supervised Training ===\n\n");
    printf("[*] Method: Straight-Through Estimator + CORE/BINARY/PRUNE Logic Guidance\n");
    printf("[*]   forward:  CORE = w_float * core_gain * K     (精确计算)\n");
    printf("[*]             BINARY = sign(w) * alpha * K       (二值符号)\n");
    printf("[*]             PRUNE = 0                           (静默剪枝)\n");
    printf("[*]   backward: STE through w_float (CORE+BINARY), PRUNE 无梯度\n");
    printf("[*]   update:   Adam on w_float\n");
    printf("[*] Logic Guidance (每步,全层):\n");
    printf("[*]             CORE  → maximize |act(a) - act(b)|  (差异化反义词)\n");
    printf("[*]             BINARY → minimize (act(a) - act(b))^2 (收敛共性)\n");
    printf("[*]             PRUNE → skip (保持静默)\n");
    printf("[*] Whitebox probe: every 10 steps observe CORE/BINARY/PRUNE circuit\n\n");

    /* 加载数据(diagnose-only 模式跳过) */
    DataLoader dl;
    memset(&dl, 0, sizeof(dl));
    if (n_steps > 0) {
        if (dataloader_init(&dl, data_path) != 0) {
            fprintf(stderr, "[!] Failed to load %s\n", data_path);
            return 1;
        }
        printf("[*] Loaded %d samples from %s\n", dl.n_samples, data_path);
    }

    /* 模型 Phase 选择: 0=8L/448d(22M), 1=10L/512d(35M), 2=12L/576d(50M), 3=14L/640d(68M) */
    if (phase_idx < 0 || phase_idx >= N_GROWTH_PHASES) phase_idx = 0;
    printf("[*] Using %s (~%ldM params, vocab=%d)\n", growth_phases[phase_idx].name, growth_phases[phase_idx].est_params, vocab_size);
    gen_phase_weights(weights_path, &growth_phases[phase_idx], vocab_size);

    ModelConfig cfg = growth_phase_config(&growth_phases[phase_idx], vocab_size);
    g_use_adam = 1;
    g_use_lal_adam = 1;        /* LAL-aware Adam: 分组二阶矩 */
    g_core_lr_multiplier = 3.0f; /* CORE 学习率 3x */
    g_use_pure_float = 1;  /* 前向用浮点(与推理一致),反向仍用 STE(g_use_ste=1) */
    g_use_ste = 1;         /* STE 反向:梯度通过 w_float 回传 */
    g_use_real_attention = 1;
    g_use_logic_binarization = 1;

    /* Logic mask: 按阶段递进 CORE 比例 */
    int lp = phase_idx < N_LOGIC_PHASES ? phase_idx : N_LOGIC_PHASES - 1;
    g_logic_core_ratio = logic_ratios[lp].core_ratio;
    g_logic_prune_ratio = logic_ratios[lp].prune_ratio;
    printf("[*] Logic mask: %s (CORE=%.0f%% PRUNE=%.0f%%)\n",
           logic_ratios[lp].name, g_logic_core_ratio * 100, g_logic_prune_ratio * 100);

    Model model;
    memset(&model, 0, sizeof(model));
    model_load(&model, weights_path, cfg, "h.%d.", cfg.qkv_merged);
    model_batch_alloc(&model);
    printf("[*] Model loaded: %d layers, %d embd, k_cache=%s\n",
           cfg.n_layer, cfg.n_embd, model.k_cache ? "YES" : "NO");

    /* 如果指定 --resume,加载已训练权重 */
    if (resume_path) {
        printf("[*] Resuming from %s...\n", resume_path);
        if (ste_load(&model, resume_path) == 0) {
            printf("[*] Resumed weights loaded successfully\n\n");
        } else {
            fprintf(stderr, "[!] Failed to load resume weights, starting fresh\n\n");
        }
    } else {
        printf("\n");
    }

    /* STE 训练 */
    if (n_steps > 0) {
        ste_train(&model, &dl, n_steps, base_lr, 50, batch_size, 64, 10, logic_lr);
    }

    /* 保存训练后权重 */
    if (save_path) {
        printf("[*] Saving trained weights to %s...\n", save_path);
        if (ste_save(&model, save_path) == 0) {
            FILE *sf = fopen(save_path, "rb");
            if (sf) { fseek(sf, 0, SEEK_END); printf("[*] Saved %ld bytes\n", (long)ftell(sf)); fclose(sf); }
        } else {
            fprintf(stderr, "[!] Failed to save weights\n");
        }
    }

    /* 训练后白箱分析 */
    printf("=== Post-Training Whitebox Analysis (surface) ===\n");
    whitebox_probe(&model);
    embedding_boundary_analysis(&model);
    concept_similarity_probe(&model);

    /* 深度白箱诊断:解释生成质量 */
    deep_whitebox_diagnosis(&model);

    /* 生成 */
    if (do_generate) {
        generate_text(&model, prompt, prompt_ids, n_prompt_ids, max_gen, temp, top_k);
    }

    printf("\n[*] Done\n");
    model_free(&model);
    dataloader_free(&dl);
    return 0;
}
