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
#include "../runtime/lal_cuda.h"  /* v13p: GPU forward declarations */
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
    int mlp_dim = m->cfg.mlp_dim;
    /* OPTIMIZATION v2: 调试 Agent 建议提高 alpha (2.0→4.0) 增强 CORE 差异化动力 */
    float alpha = 4.0f;   /* CORE 差异化权重 (v1: 2.0, v2: 4.0) */
    float beta = 0.2f;    /* BINARY 收敛权重 */
    float total_loss = 0.0f;
    float logic_loss = 0.0f;
    int n_guided_layers = 0;

    /* 统计三类神经元总数 */
    int total_core = 0, total_binary = 0, total_prune = 0;
    for (int l = 0; l < n_layer; l++) {
        BinLayer *fc = &m->layers[l].mlp_gate;
        if (!fc->logic_mask) continue;
        for (int j = 0; j < fc->out_dim; j++) {
            if (fc->logic_mask[j] == 0) total_core++;
            else if (fc->logic_mask[j] == 1) total_binary++;
            else total_prune++;
        }
        n_guided_layers++;
    }
    if (n_guided_layers == 0) return 0.0f;

    /* BUG #35 FIX: 旧代码结构是 for(layer) for(pair) { compute_gate_input(layer) }
     * 这导致:
     *   1. get_concept_embedding 被调用 n_layer × N_PROBE_PAIRS × 2 = 112 次/步
     *      (实际只需 14 次,emb 不随 layer 变化)
     *   2. compute_gate_input(l) 内部跑 layer 0..l 的完整 forward。
     *      对 layer 7,每次调用跑 8 层 forward。n_layer 个 layer 总共跑
     *      1+2+...+8 = 36 次 forward/pair = 36×14 = 504 次 forward/步 (O(n²))
     *
     * 修复:外层循环 pair,内层循环 layer。对每个 pair,只做一次 0→n_layer-1 的
     * forward,在每层捕获 gate_input。forward 次数从 504 降到 14 (36x 加速)。
     * emb 计算从 112 降到 14 (8x 加速)。
     *
     * 需要缓存每层的 gate_input:gate_a[layer][n_embd], gate_b[layer][n_embd] */

    /* 静态缓存:gate_inputs[pair][concept][layer][n_embd]
     * N_PROBE_PAIRS=7, 2 concepts, n_layer≤32, n_embd≤4096
     * 用静态指针数组,按需分配 */
    static float *gate_cache_a = NULL;  /* v13l: [N_PROBE_PAIRS * n_layer * n_embd] */
    static float *gate_cache_b = NULL;  /* v13l: [N_PROBE_PAIRS * n_layer * n_embd] */
    static int gc_n = 0, gc_layers = 0;
    if (gc_n != n_embd || gc_layers != n_layer) {
        free(gate_cache_a); free(gate_cache_b);
        gate_cache_a = malloc((size_t)N_PROBE_PAIRS * n_layer * n_embd * sizeof(float));
        gate_cache_b = malloc((size_t)N_PROBE_PAIRS * n_layer * n_embd * sizeof(float));
        gc_n = n_embd; gc_layers = n_layer;
    }

    /* 静态 emb 缓存 (避免每对重复计算) */
    static float *emb_cache_a = NULL;  /* [N_PROBE_PAIRS * n_embd] */
    static float *emb_cache_b = NULL;
    static int ec_n = 0;
    if (ec_n != n_embd) {
        free(emb_cache_a); free(emb_cache_b);
        emb_cache_a = malloc((size_t)N_PROBE_PAIRS * n_embd * sizeof(float));
        emb_cache_b = malloc((size_t)N_PROBE_PAIRS * n_embd * sizeof(float));
        ec_n = n_embd;
    }

    /* 预计算所有 emb (14 次,不是 112 次) */
    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        get_concept_embedding(m, cp->bytes_a, &emb_cache_a[(size_t)p * n_embd], n_embd);
        get_concept_embedding(m, cp->bytes_b, &emb_cache_b[(size_t)p * n_embd], n_embd);
    }

    /* BUG FIX: gate_cache 必须填充实际的 gate_input (LN2 输出)。
     * 之前 gate_cache_a/b 只分配未填充,CUDA 和 CPU 路径都读到未初始化内存,
     * 导致 simulate_activation 产生垃圾激活值,逻辑损失爆炸到 1e25。
     * 使用 compute_all_gate_inputs 一次 forward 捕获所有层 gate_input (O(n) not O(n²))。 */
    {
        static float *tmp_gate = NULL;
        static int tg_sz = 0;
        int need_sz = n_layer * n_embd;
        if (tg_sz != need_sz) {
            free(tmp_gate);
            tmp_gate = malloc((size_t)need_sz * sizeof(float));
            tg_sz = need_sz;
        }
        int batch = (int)N_PROBE_PAIRS;
        for (int p = 0; p < batch; p++) {
            compute_all_gate_inputs(m, &emb_cache_a[(size_t)p * n_embd],
                                    tmp_gate, n_embd, NULL, NULL);
            for (int l = 0; l < n_layer; l++) {
                memcpy(&gate_cache_a[((size_t)l * batch + p) * n_embd],
                       &tmp_gate[(size_t)l * n_embd], n_embd * sizeof(float));
            }
            compute_all_gate_inputs(m, &emb_cache_b[(size_t)p * n_embd],
                                    tmp_gate, n_embd, NULL, NULL);
            for (int l = 0; l < n_layer; l++) {
                memcpy(&gate_cache_b[((size_t)l * batch + p) * n_embd],
                       &tmp_gate[(size_t)l * n_embd], n_embd * sizeof(float));
            }
        }
    }

    /* v13r: GPU logic_reg — simulate_activation + grad_accum on GPU */
#ifdef LAL_CUDA
    if (g_use_cuda) {
        logic_loss = lal_cuda_logic_reg(m, gate_cache_a, gate_cache_b, lr);
    } else
#endif
    {
        /* CPU fallback: original logic_reg loop */
        int batch = (int)N_PROBE_PAIRS;
        float *act_a = malloc(mlp_dim * sizeof(float));
        float *act_b = malloc(mlp_dim * sizeof(float));
        for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
            for (int l = 0; l < n_layer; l++) {
                BinLayer *fc = &m->layers[l].mlp_gate;
                uint8_t *mask = fc->logic_mask;
                if (!mask) continue;
                int layer_in_dim = fc->in_dim;
                float *gate_a = &gate_cache_a[((size_t)l * batch + p) * n_embd];
                float *gate_b = &gate_cache_b[((size_t)l * batch + p) * n_embd];
                simulate_activation(m, gate_a, l, act_a, mlp_dim);
                simulate_activation(m, gate_b, l, act_b, mlp_dim);
                int layer_nc = 0, layer_nb = 0;
                for (int j = 0; j < mlp_dim; j++) {
                    if (mask[j] == 2) continue;
                    float diff = act_a[j] - act_b[j];
                    float adiff = fabsf(diff);
                    if (mask[j] == 0) {
                        total_loss -= alpha * tanhf(adiff * 0.5f);
                        float gamma = 0.3f;
                        total_loss += gamma * (fabsf(act_a[j]) + fabsf(act_b[j]));
                        layer_nc++;
                    } else {
                        total_loss += beta * diff * diff;
                        layer_nb++;
                    }
                }
                float inv_nc = layer_nc > 0 ? 1.0f / sqrtf((float)layer_nc) : 0;
                float inv_nb = layer_nb > 0 ? 1.0f / sqrtf((float)layer_nb) : 0;
                float layer_lr_scale = (l == 0) ? 1.0f : 0.5f;
                for (int j = 0; j < mlp_dim; j++) {
                    if (mask[j] == 2) continue;
                    float diff = act_a[j] - act_b[j];
                    float *ga = &fc->grad_accum[(size_t)j * layer_in_dim];
                    if (mask[j] == 0) {
                        float s = diff > 0 ? 1.0f : -1.0f;
                        float tanh_adiff = tanhf(fabsf(diff) * 0.5f);
                        float diff_grad = -alpha * 0.5f * s * (1.0f - tanh_adiff * tanh_adiff);
                        float gamma = 0.3f;
                        float mag_grad_a = gamma * (act_a[j] > 0 ? 1.0f : -1.0f);
                        float mag_grad_b = gamma * (act_b[j] > 0 ? 1.0f : -1.0f);
                        float grad_scale = (diff_grad + mag_grad_a) * inv_nc * layer_lr_scale * lr;
                        for (int i = 0; i < layer_in_dim; i++)
                            ga[i] += grad_scale * (gate_a[i] - gate_b[i]) + mag_grad_b * inv_nc * layer_lr_scale * lr * gate_b[i];
                        if (fc->bias_grad_accum)
                            fc->bias_grad_accum[j] += grad_scale + mag_grad_b * inv_nc * layer_lr_scale * lr;
                    } else {
                        float grad_scale = beta * 2.0f * diff * inv_nb * layer_lr_scale * lr;
                        for (int i = 0; i < layer_in_dim; i++)
                            ga[i] += grad_scale * (gate_a[i] - gate_b[i]);
                        if (fc->bias_grad_accum)
                            fc->bias_grad_accum[j] += grad_scale;
                    }
                }
            }
        }
        free(act_a); free(act_b);
        logic_loss = total_loss / (N_PROBE_PAIRS * n_guided_layers);
    }
    

    /* 注:wte 推开试验过,但 byte-level 下反义词共享首 byte(热=0xE783AD, 冷=0xE586B7),
     * 推开单个 byte embedding 会污染所有以该 byte 开头的字符,适得其反。
     * embedding 分化应通过 CORE 引导间接实现,不直接改 wte。 */

    /* 返回时通过静态变量传递统计(简化接口) */
    static int last_core = 0, last_binary = 0, last_prune = 0;
    last_core = total_core;
    last_binary = total_binary;
    last_prune = total_prune;

    return logic_loss;
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

/* === v13: Attention 逻辑引导 ===
 *
 * v12 根因 #3: logic_guided_regularization 只迭代 mlp_gate, attention 的
 * QKV/O 完全没有逻辑引导信号. 配合 base_lr=0.001 vs logic_lr=1.0,
 * MLP 获得 ~1000x stronger gradient, attention 留在 random init (mean=0.000073).
 *
 * 本函数对 attn_o (output projection) 做 CORE/BINARY/PRUNE 差异化引导:
 *   - 对每对反义词 (a, b), 在每层计算 proj_out_a vs proj_out_b
 *   - CORE (mask=0): 最大化 |proj_a - proj_b| (让 attention 区分概念)
 *   - BINARY (mask=1): 最小化 (proj_a - proj_b)^2 (共性归纳)
 *   - PRUNE (mask=2): skip
 *
 * attn_o 的 logic_mask 在 model_load 时已通过 compute_norm_mask 自动生成
 * (与 mlp_gate 同样的逻辑).
 *
 * 梯度路径: proj_out = W_o · V, V = W_q[2n:] · norm1
 *   d proj_a / d W_o = V_a  (V 是 attn_q 的 V 切片, attn_q 的输出 [Q|K|V])
 *   d proj_b / d W_o = V_b
 *   对 CORE: grad_W_o = -alpha * 0.5 * sign(diff) * (1-tanh²) * (V_a - V_b)
 *   注意: 我们只更新 W_o, 不更新 W_q (避免污染 Q/K 学习).
 *   V_a, V_b 作为常量从 norm1_a, norm1_b 通过 attn_q 算出.
 */
static float attn_logic_regularization(Model *m, float lr) {
    int n_embd = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int n = n_embd;
    /* attention 引导用更温和的 alpha (attention 输出量级小于 MLP gate) */
    float alpha = 2.0f;   /* CORE 差异化权重 (比 MLP 的 4.0 小, 防 attention 爆炸) */
    float beta = 0.1f;    /* BINARY 收敛权重 */
    float total_loss = 0.0f;
    int n_guided_layers = 0;

    /* 统计 */
    int total_core = 0, total_binary = 0;
    for (int l = 0; l < n_layer; l++) {
        BinLayer *ao = &m->layers[l].attn_o;
        if (!ao->logic_mask) continue;
        for (int j = 0; j < ao->out_dim; j++) {
            if (ao->logic_mask[j] == 0) total_core++;
            else if (ao->logic_mask[j] == 1) total_binary++;
        }
        n_guided_layers++;
    }
    if (n_guided_layers == 0) return 0.0f;

    /* 缓存: norm1_inputs[pair][concept][layer][n_embd]
     * V_outputs[pair][concept][layer][n_embd] (V 切片 of QKV merged output) */
    static float *norm1_cache_a = NULL;  /* [n_layer * n_embd] */
    static float *norm1_cache_b = NULL;
    static float *v_cache_a = NULL;      /* [n_layer * n_embd] — V 切片 */
    static float *v_cache_b = NULL;
    static float *proj_a = NULL;         /* [n_embd] — attn_o output */
    static float *proj_b = NULL;
    static int gc_n = 0, gc_layers = 0;
    if (gc_n != n_embd || gc_layers != n_layer) {
        free(norm1_cache_a); free(norm1_cache_b);
        free(v_cache_a); free(v_cache_b);
        free(proj_a); free(proj_b);
        norm1_cache_a = malloc((size_t)n_layer * n_embd * sizeof(float));
        norm1_cache_b = malloc((size_t)n_layer * n_embd * sizeof(float));
        v_cache_a = malloc((size_t)n_layer * n_embd * sizeof(float));
        v_cache_b = malloc((size_t)n_layer * n_embd * sizeof(float));
        proj_a = malloc(n_embd * sizeof(float));
        proj_b = malloc(n_embd * sizeof(float));
        gc_n = n_embd; gc_layers = n_layer;
    }

    /* 静态 emb 缓存 */
    static float *emb_cache_a = NULL;
    static float *emb_cache_b = NULL;
    static int ec_n = 0;
    if (ec_n != n_embd) {
        free(emb_cache_a); free(emb_cache_b);
        emb_cache_a = malloc((size_t)N_PROBE_PAIRS * n_embd * sizeof(float));
        emb_cache_b = malloc((size_t)N_PROBE_PAIRS * n_embd * sizeof(float));
        ec_n = n_embd;
    }

    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        get_concept_embedding(m, cp->bytes_a, &emb_cache_a[(size_t)p * n_embd], n_embd);
        get_concept_embedding(m, cp->bytes_b, &emb_cache_b[(size_t)p * n_embd], n_embd);
    }

    /* QKV merged 临时缓冲: [3*n] */
    float *qkv_a = malloc(3 * n * sizeof(float));
    float *qkv_b = malloc(3 * n * sizeof(float));

    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        /* 一次 forward 拿到所有层 norm1_out, 同时也拿到 V (QKV 的 V 切片) */
        compute_all_gate_inputs(m, &emb_cache_a[(size_t)p * n_embd],
                                /*out_gate_inputs*/NULL, n_embd,
                                /*out_norm1_inputs*/norm1_cache_a, /*final*/NULL);
        compute_all_gate_inputs(m, &emb_cache_b[(size_t)p * n_embd],
                                /*out_gate_inputs*/NULL, n_embd,
                                /*out_norm1_inputs*/norm1_cache_b, /*final*/NULL);

        /* 对每层, 用 norm1 重新算 V (QKV merged 的 V 切片) 和 proj_out */
        for (int l = 0; l < n_layer; l++) {
            TransLayer *tl = &m->layers[l];
            BinLayer *ao = &tl->attn_o;
            uint8_t *mask = ao->logic_mask;
            if (!mask) continue;

            float *n1a = &norm1_cache_a[(size_t)l * n];
            float *n1b = &norm1_cache_b[(size_t)l * n];

            /* 算 QKV merged: qkv = attn_q · norm1, 然后 V = qkv[2n:3n] */
            bin_forward_pure_float(qkv_a, n1a, &tl->attn_q);
            bin_forward_pure_float(qkv_b, n1b, &tl->attn_q);
            float *V_a = qkv_a + 2 * n;  /* V 切片 */
            float *V_b = qkv_b + 2 * n;

            /* 算 attn_o output: proj = attn_o · V */
            bin_forward_pure_float(proj_a, V_a, ao);
            bin_forward_pure_float(proj_b, V_b, ao);

            int layer_nc = 0, layer_nb = 0;
            int out_dim = ao->out_dim;  /* = n */
            int in_dim = ao->in_dim;    /* = n */

            /* 计算 loss + 统计 (与 logic_guided_regularization 同结构) */
            for (int j = 0; j < out_dim; j++) {
                if (mask[j] == 2) continue;
                float diff = proj_a[j] - proj_b[j];
                float adiff = fabsf(diff);
                if (mask[j] == 0) {
                    total_loss -= alpha * tanhf(adiff * 0.5f);
                    /* 幅度惩罚: 防 attention 对所有概念都高激活但不区分 */
                    float gamma = 0.15f;  /* 比 MLP 小 (0.3), attention 信号本来就小 */
                    total_loss += gamma * (fabsf(proj_a[j]) + fabsf(proj_b[j]));
                    layer_nc++;
                } else {
                    total_loss += beta * diff * diff;
                    layer_nb++;
                }
            }

            float inv_nc = layer_nc > 0 ? 1.0f / sqrtf((float)layer_nc) : 0;
            float inv_nb = layer_nb > 0 ? 1.0f / sqrtf((float)layer_nb) : 0;
            float layer_lr_scale = (l == 0) ? 1.0f : 0.5f;

            /* 梯度: d Loss / d W_o
             * 对 CORE: grad = -alpha*0.5*sign(diff)*(1-tanh²) * (V_a - V_b) + gamma*sign(proj)*V
             * 对 BINARY: grad = beta*2*diff * (V_a - V_b)
             * 累加到 ao->grad_accum (与 logic_guided_regularization 同结构) */
            for (int j = 0; j < out_dim; j++) {
                if (mask[j] == 2) continue;
                float diff = proj_a[j] - proj_b[j];
                float *ga = &ao->grad_accum[(size_t)j * in_dim];

                if (mask[j] == 0) {
                    float s = diff > 0 ? 1.0f : -1.0f;
                    float tanh_adiff = tanhf(fabsf(diff) * 0.5f);
                    float diff_grad = -alpha * 0.5f * s * (1.0f - tanh_adiff * tanh_adiff);
                    float gamma = 0.15f;
                    float mag_grad_a = gamma * (proj_a[j] > 0 ? 1.0f : -1.0f);
                    float mag_grad_b = gamma * (proj_b[j] > 0 ? 1.0f : -1.0f);
                    float grad_scale = (diff_grad + mag_grad_a) * inv_nc * layer_lr_scale * lr;
                    /* d proj_a/dW = V_a, d proj_b/dW = V_b
                     * diff_grad 对 a 用 +, 对 b 用 -, 与 MLP 一致 */
                    for (int i = 0; i < in_dim; i++)
                        ga[i] += grad_scale * (V_a[i] - V_b[i]) + mag_grad_b * inv_nc * layer_lr_scale * lr * V_b[i];
                    if (ao->bias_grad_accum)
                        ao->bias_grad_accum[j] += grad_scale + mag_grad_b * inv_nc * layer_lr_scale * lr;
                } else {
                    float grad_scale = beta * 2.0f * diff * inv_nb * layer_lr_scale * lr;
                    for (int i = 0; i < in_dim; i++)
                        ga[i] += grad_scale * (V_a[i] - V_b[i]);
                    if (ao->bias_grad_accum)
                        ao->bias_grad_accum[j] += grad_scale;
                }
            }
        }
    }

    free(qkv_a); free(qkv_b);
    return total_loss / (N_PROBE_PAIRS * n_guided_layers);
}

/* === v13: 残差流多样性损失 (Anti-Collapse Loss) ===
 *
 * 即使 MLP gate 的 CORE diff 很好 (1.02), 信号在残差流中仍会被淹没.
 * 原因: MLP 输出量级 (~2.0) >> 输入量级 (~0.9), MLP 主导残差方向.
 * CE loss 推动模型对不同输入产生相同输出 (mode collapse to frequent tokens).
 *
 * 本函数直接惩罚残差流的 cosine 塌缩:
 *   对每对反义词 (a, b), 在每层计算 cosine(h_a, h_b)
 *   如果 cosine > threshold (0.5), 加损失 (cosine - 0.5) * weight
 *   梯度直接推开 h_a 和 h_b 的方向
 *
 * 这是最直接的 anti-collapse 机制 — 不依赖 MLP gate 的间接信号,
 * 而是直接监控残差流本身.
 *
 * 实现细节: 梯度对 h_a 和 h_b 各推开 (cosine - threshold) 的方向分量.
 *   d cosine / d h_a = (h_b - cosine * h_a) / (||h_a|| * ||h_b||)
 *   push apart: h_a -= lr * d_cosine_dh_a, h_b -= lr * d_cosine_dh_b
 *   v13l: 重写为带梯度的多样性损失。对每对反义词 (a,b)，在每层计算
 *   cosine(x_a, x_b)，当超过阈值时直接对 attn_o 和 mlp_down 的 grad_accum
 *   累加推开梯度。梯度方向：让 x_a 和 x_b 的残差方向互相远离。
 *
 *   梯度推导：L = max(0, cos(x_a, x_b) - threshold)
 *   dL/dx_a = (x_b / (||x_a|| ||x_b||)) - cos * (x_a / ||x_a||²)   (当 cos > threshold)
 *   dL/dx_b = (x_a / (||x_a|| ||x_b||)) - cos * (x_b / ||x_b||²)
 *   这个梯度通过残差路径传回 W_o 和 W_down。
 */
static float residual_diversity_loss(Model *m, float lr) {
    int n_embd = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int n = n_embd;
    float threshold = 0.3f;  /* 惩罚阈值：cosine超过0.3就开始推开 */
    float weight = 2.0f;     /* 损失权重 */
    float total_loss = 0.0f;

    static float *emb_a = NULL, *emb_b = NULL;
    static float *xa = NULL, *xb = NULL;
    static float *grad_a = NULL, *grad_b = NULL;
    static int ec_n = 0;
    if (ec_n != n_embd) {
        free(emb_a); free(emb_b); free(xa); free(xb); free(grad_a); free(grad_b);
        emb_a = malloc(n * sizeof(float));
        emb_b = malloc(n * sizeof(float));
        xa = malloc(n * sizeof(float));
        xb = malloc(n * sizeof(float));
        grad_a = malloc(n * sizeof(float));
        grad_b = malloc(n * sizeof(float));
        ec_n = n;
    }

    float max_cos = 0;
    int n_collapsed = 0;

    for (int p = 0; p < (int)N_PROBE_PAIRS; p++) {
        ConceptPair *cp = &probe_pairs[p];
        get_concept_embedding(m, cp->bytes_a, emb_a, n);
        get_concept_embedding(m, cp->bytes_b, emb_b, n);

        /* Forward: 跑完所有层，缓存每层后的 x_a, x_b */
        /* 用静态数组缓存每层的 x_a, x_b */
        static float *xa_layers = NULL, *xb_layers = NULL;
        static int xl_n = 0, xl_layers = 0;
        if (xl_n != n_embd || xl_layers != n_layer) {
            free(xa_layers); free(xb_layers);
            xa_layers = malloc((size_t)(n_layer + 1) * n * sizeof(float));
            xb_layers = malloc((size_t)(n_layer + 1) * n * sizeof(float));
            xl_n = n_embd; xl_layers = n_layer;
        }

        memcpy(xa, emb_a, n * sizeof(float));
        memcpy(xb, emb_b, n * sizeof(float));
        if (m->wpe) {
            for (int i = 0; i < n; i++) {
                xa[i] += m->wpe[i];
                xb[i] += m->wpe[i];
            }
        }
        memcpy(&xa_layers[0], xa, n * sizeof(float));
        memcpy(&xb_layers[0], xb, n * sizeof(float));

        for (int l = 0; l < n_layer; l++) {
            trans_layer_forward(xa, &m->layers[l], &m->acts[l], &m->cfg, 0);
            trans_layer_forward(xb, &m->layers[l], &m->acts[l], &m->cfg, 0);
            memcpy(&xa_layers[(size_t)(l + 1) * n], xa, n * sizeof(float));
            memcpy(&xb_layers[(size_t)(l + 1) * n], xb, n * sizeof(float));
        }

        /* Backward: 从最后一层往回传梯度 */
        memset(grad_a, 0, n * sizeof(float));
        memset(grad_b, 0, n * sizeof(float));

        for (int l = n_layer - 1; l >= 0; l--) {
            float *xal = &xa_layers[(size_t)(l + 1) * n];
            float *xbl = &xb_layers[(size_t)(l + 1) * n];

            /* 计算 cosine(x_a, x_b) */
            float dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) {
                dot += xal[i] * xbl[i];
                na += xal[i] * xal[i];
                nb += xbl[i] * xbl[i];
            }
            float norm_a = sqrtf(na) + 1e-8f;
            float norm_b = sqrtf(nb) + 1e-8f;
            float cos = dot / (norm_a * norm_b);
            if (cos > max_cos) max_cos = cos;

            if (cos > threshold) {
                total_loss += weight * (cos - threshold);
                if (cos > 0.9f) n_collapsed++;

                /* dL/dx_a = weight * (x_b / (||a|| ||b||) - cos * x_a / ||a||²) */
                float inv_ab = 1.0f / (norm_a * norm_b);
                float cos_over_na = cos / na;
                float cos_over_nb = cos / nb;
                for (int i = 0; i < n; i++) {
                    grad_a[i] += weight * (xbl[i] * inv_ab - xal[i] * cos_over_na);
                    grad_b[i] += weight * (xal[i] * inv_ab - xbl[i] * cos_over_nb);
                }

                /* 将梯度传到 attn_o 和 mlp_down 的 grad_accum */
                /* x = x_pre + rs * attn_scale * proj_out + rs * mlp_scale * mlp_out */
                /* 我们近似：对 attn_o 的 grad_accum 加 grad * norm1_out^T */
                TransLayer *tl = &m->layers[l];
                float rs = m->cfg.residual_scale;
                /* attn_o: grad_W_o += (grad * rs * attn_scale) × norm1_out^T */
                float attn_scale = m->acts[l].attn_scale;
                float grad_attn_scale = rs * attn_scale;
                int in_dim = tl->attn_o.in_dim;
                int out_dim = tl->attn_o.out_dim;
                for (int j = 0; j < out_dim && j < n; j++) {
                    float ga = grad_a[j] * grad_attn_scale * lr;
                    float gb = grad_b[j] * grad_attn_scale * lr;
                    if (fabsf(ga) > 1e-10f || fabsf(gb) > 1e-10f) {
                        float *wga = &tl->attn_o.grad_accum[(size_t)j * in_dim];
                        const float *n1 = m->acts[l].norm1_out;
                        for (int i = 0; i < in_dim && i < n; i++)
                            wga[i] += ga * n1[i] + gb * n1[i];  /* 两个概念都推开 */
                        if (tl->attn_o.bias_grad_accum) {
                            tl->attn_o.bias_grad_accum[j] += ga + gb;
                        }
                    }
                }

                /* mlp_down: grad_W_down += (grad * rs * mlp_scale) × hidden^T */
                float mlp_scale = m->acts[l].mlp_scale;
                float grad_mlp_scale = rs * mlp_scale;
                int md_in = tl->mlp_down.in_dim;
                int md_out = tl->mlp_down.out_dim;
                for (int j = 0; j < md_out && j < n; j++) {
                    float ga = grad_a[j] * grad_mlp_scale * lr;
                    float gb = grad_b[j] * grad_mlp_scale * lr;
                    if (fabsf(ga) > 1e-10f || fabsf(gb) > 1e-10f) {
                        float *wga = &tl->mlp_down.grad_accum[(size_t)j * md_in];
                        const float *hid = m->acts[l].mlp_hidden;
                        for (int i = 0; i < md_in; i++)
                            wga[i] += ga * hid[i] + gb * hid[i];
                        if (tl->mlp_down.bias_grad_accum) {
                            tl->mlp_down.bias_grad_accum[j] += ga + gb;
                        }
                    }
                }
            }

            /* 传播梯度到前一层 (简化：通过 W_o^T 和 W_down^T 传回) */
            /* grad_x_pre = grad * (W_o^T * attn_scale + W_down^T * mlp_scale) */
            /* 为了效率，只传播到 x_pre (跳过 LN 链) */
            TransLayer *tl = &m->layers[l];  /* re-declare for this scope */
            float *new_grad_a = grad_a;  /* 原地操作 */
            float *new_grad_b = grad_b;
            /* 实际上这里需要存储中间结果，用一个临时buffer */
            static float *tmp_ga = NULL, *tmp_gb = NULL;
            static int tg_n = 0;
            if (tg_n != n) {
                free(tmp_ga); free(tmp_gb);
                tmp_ga = malloc(n * sizeof(float));
                tmp_gb = malloc(n * sizeof(float));
                tg_n = n;
            }
            memcpy(tmp_ga, grad_a, n * sizeof(float));
            memcpy(tmp_gb, grad_b, n * sizeof(float));
            memset(grad_a, 0, n * sizeof(float));
            memset(grad_b, 0, n * sizeof(float));
            /* grad_x_pre = W_o^T * (grad * attn_scale) + W_down^T * (grad * mlp_scale) */
            /* 用 bin_forward_pure_float 近似反向传播（转置乘法） */
            /* 简化：只传 attn_o 部分，因为 attn 是主要坍缩源 */
            float as = m->acts[l].attn_scale * m->cfg.residual_scale;
            float ms = m->acts[l].mlp_scale * m->cfg.residual_scale;
            for (int j = 0; j < n; j++) {
                float ga_j = tmp_ga[j] * as;
                float gb_j = tmp_gb[j] * as;
                /* W_o^T * grad: 对每个输入 i, sum_j W_o[j*n+i] * grad[j] */
                const float *wo = tl->attn_o.w_float;
                if (wo) {
                    for (int i = 0; i < n; i++) {
                        grad_a[i] += wo[(size_t)j * n + i] * ga_j;
                        grad_b[i] += wo[(size_t)j * n + i] * gb_j;
                    }
                }
            }
            /* 加上 mlp_down 的贡献 */
            for (int j = 0; j < n; j++) {
                float ga_j = tmp_ga[j] * ms;
                float gb_j = tmp_gb[j] * ms;
                const float *wd = tl->mlp_down.w_float;
                if (wd) {
                    int md_in = tl->mlp_down.in_dim;
                    for (int i = 0; i < n && i < md_in; i++) {
                        grad_a[i] += wd[(size_t)j * md_in + i] * ga_j;
                        grad_b[i] += wd[(size_t)j * md_in + i] * gb_j;
                    }
                }
            }
        }
    }

    if (n_collapsed > 0 && g_opt_step % 10 == 0) {
        printf("  [DIVERSITY] max_cos=%.4f n_collapsed=%d/%d layers\n",
               max_cos, n_collapsed, (int)(N_PROBE_PAIRS * n_layer));
    }
    return total_loss / N_PROBE_PAIRS;
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
        /* BUG #43 FIX: guard against division by zero when nc or nb is 0.
         * With standard logic ratios (CORE≥15%, BINARY≥60%) this shouldn't
         * happen, but defensive coding prevents NaN in edge cases (e.g.
         * custom logic mask with all-PRUNE or all-CORE layers). */
        if (nc > 0) core_avg /= nc; else core_avg = 0;
        if (nb > 0) bin_avg /= nb; else bin_avg = 0;

        printf("  Layer 0 mlp activation (for '热'):\n");
        printf("    CORE  : max=%.4f avg=%.4f  (n=%d)\n", core_max, core_avg, nc);
        printf("    BINARY: max=%.4f avg=%.4f  (n=%d)\n", bin_max, bin_avg, nb);

        float core_diff = 0;
        for (int j = 0; j < mlp_dim; j++)
            if (fc0->logic_mask[j] == 0)
                core_diff += fabsf(act_hot[j] - act_cold[j]);
        /* BUG #43: guard against nc=0 (same fix as above) */
        if (nc > 0) core_diff /= nc;
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
        /* BUG #43: guard against division by zero when nc/nb/np is 0 */
        if (nc > 0) core_norm /= nc; else core_norm = 0;
        if (nb > 0) bin_norm /= nb; else bin_norm = 0;
        if (np > 0) prune_norm /= np; else prune_norm = 0;
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

    /* top-5 (BUG #38 FIX: mask special tokens id 0-2, same as generation) */
    printf("  Prompt '热' → top-5 next-token logits:\n");
    /* used 标记数组要覆盖整个 vocab */
    static int used[40000];
    memset(used, 0, vocab * sizeof(int));
    used[0] = 1;  /* <unk> */
    used[1] = 1;  /* <s> */
    used[2] = 1;  /* </s> */
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
    /* BUG #37 FIX: 旧代码硬编码 logf(256)/logf(2) = 8 bits 作为"均匀分布熵",
     *   但 BPE 模式 vocab=32768,均匀分布熵应该是 log2(32768)=15 bits。
     *   这导致诊断报告显示"uniform would be 8.00"而非 15.00,让 8 bits 的熵
     *   看起来"接近均匀",实际上只有均匀分布的 53% (8/15),模型远未充分分散。 */
    float uniform_bits = logf((float)vocab) / logf(2.0f);
    printf("  Softmax entropy: %.4f bits (uniform would be %.2f)\n",
           entropy/logf(2), uniform_bits);
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
    /* BUG #41 FIX: g_opt_step is appended at END of file (after all weights)
     * for backward compatibility with old .ste files. */

    /* wte */
    fwrite(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    /* wpe */
    if (m->wpe) {
        fwrite(m->wpe, sizeof(float), (size_t)m->cfg.n_ctx * n_embd, f);
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
    /* BUG #41 FIX: append g_opt_step at END of file for resume continuity */
    fwrite(&g_opt_step, sizeof(int), 1, f);
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

    /* BUG #41: g_opt_step is saved at END of file (not here) for backward
     * compatibility with old .ste files that don't have it. See ste_load
     * epilogue after all weights are read. */

    fread(m->wte, sizeof(float), (size_t)vocab * n_embd, f);
    /* wpe */
    if (m->wpe) {
        fread(m->wpe, sizeof(float), (size_t)m->cfg.n_ctx * n_embd, f);
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
                /* FIX: recompute n_core and reallocate w_core — the loaded
                 * logic_mask may have a different CORE count than the one
                 * computed during model_load (different weight distribution).
                 * Without this, bin_layer_repack writes beyond w_core buffer,
                 * causing memory corruption and NaN in forward pass. */
                int new_n_core = 0;
                for (int j = 0; j < bl->out_dim; j++) {
                    if (bl->logic_mask[j] == 0) new_n_core++;
                }
                if (new_n_core != bl->n_core) {
                    bl->n_core = new_n_core;
                    free(bl->w_core);
                    bl->w_core = (new_n_core > 0)
                        ? malloc((size_t)new_n_core * bl->in_dim * sizeof(float))
                        : NULL;
                }
                /* 重建 w_core 和 wbits */
                bin_layer_repack(bl);
            }
        }
        fread(tl->norm1_w, sizeof(float), n_embd, f);
        fread(tl->norm1_b, sizeof(float), n_embd, f);
        fread(tl->norm2_w, sizeof(float), n_embd, f);
        fread(tl->norm2_b, sizeof(float), n_embd, f);
    }
    /* BUG #41 FIX: try to read g_opt_step from END of file.
     * Old .ste files don't have this — fread returns 0, g_opt_step stays 0.
     * New files have it — resume preserves Adam step count. */
    {
        int saved_opt_step = 0;
        size_t rc = fread(&saved_opt_step, sizeof(int), 1, f);
        if (rc == 1) {
            g_opt_step = saved_opt_step;
            printf("[*] Resumed g_opt_step = %d (Adam bias correction preserved)\n", g_opt_step);
        }
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

/* === STE 训练:单阶段,固定模型 ===
 * start_step: 用于续训时正确恢复 LR schedule (cosine decay 从 start_step 开始计算)
 * total_schedule_steps: LR schedule 的总步数 (分块训练时保持 cosine 连续)
 */
static float ste_train(Model *m, DataLoader *dl, int n_steps, float base_lr,
                       int warmup, int batch_size, int max_pos, int eval_interval,
                       float logic_lr, int start_step, int total_schedule_steps,
                       int logic_only) {  /* v13g: logic_only mode for curriculum stage 1 */
    printf("\n=== LAL STE Training (B=%d, max_pos=%d, steps=%d, start_step=%d, total_sched=%d, logic_lr=%.4f, logic_only=%d) ===\n",
           batch_size, max_pos, n_steps, start_step, total_schedule_steps, logic_lr, logic_only);
    printf("[*] STE: forward=sign(w), backward=w_float gradient\n");
    printf("[*] Whitebox probe every %d steps\n", eval_interval);
    printf("[*] Semantic regularization every step\n");
    if (logic_only) printf("[*] LOGIC-ONLY MODE: skipping CE loss, only concept-pair regularization (curriculum stage 1)\n\n");

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
        int global_step = start_step + step;
        /* 用 total_schedule_steps 让分块训练的 cosine LR 连续 */
        float lr = lr_schedule(global_step, warmup, total_schedule_steps, base_lr);
        model_batch_begin(m);

        float batch_loss = 0;
        int n_valid = 0;
        struct timespec t_ce_start, t_logic_start, t_apply_start, t_ce_end;
        clock_gettime(CLOCK_MONOTONIC, &t_ce_start);

        if (!logic_only) {
            /* v13g: logic-only mode 跳过 CE 训练, 只做概念对正则化 */
            for (int b = 0; b < batch_size; b++) {
                int idx = rand() % dl->n_samples;
                int n_tok = dataloader_get(dl, idx, batch_tokens[b], 512);
                if (n_tok < 20) continue;

                int mp = n_tok < max_pos ? n_tok : max_pos;
                if (mp < 10) continue;

                /* Multi-position prediction: 2 positions per sample (v13i: was 4, reduced for max_pos=256) */
                int n_preds = 2;
                int stride = (mp - 6) / n_preds;
                if (stride < 1) stride = 1;

                for (int p = 0; p < n_preds; p++) {
                    int pred_pos = 5 + p * stride;
                    if (pred_pos >= mp - 1) break;
                    int target = batch_tokens[b][pred_pos + 1];
                    int predicted = 0;
                    float grad_hidden[4096];
                    float loss;
#ifdef LAL_CUDA
                    if (g_use_cuda) {
                        loss = lal_cuda_full_forward(m, batch_tokens[b], pred_pos + 1,
                                                      target, grad_hidden, &predicted);
                        lal_cuda_full_backward(m, batch_tokens[b], pred_pos + 1, target);
                    } else
#endif
                    {
                        loss = model_batch_forward(m, batch_tokens[b], pred_pos + 1);
                        model_batch_backward(m, batch_tokens[b], pred_pos + 1);
                    }
                    if (!isnan(loss) && !isinf(loss)) {
                        batch_loss += loss;
                        n_valid++;
                    }
                }
            }
        } /* end if (!logic_only) */
        clock_gettime(CLOCK_MONOTONIC, &t_logic_start);

        if (n_valid > 0 || logic_only) {
            /* v13k: 只在每 5 步做 logic_reg, 减少 80% 的 logic_reg 计算量
             * logic_reg 是 GPU 瓶颈 (13.5s/step), 每 5 步做一次可降到 ~2.7s
             * logic_lr 乘 5 补偿频率降低 */
            int do_logic = (step % 5 == 0) || logic_only;
            float effective_logic_lr = do_logic ? logic_lr * 5.0f : 0.0f;
            float effective_attn_lr = do_logic ? logic_lr * 2.5f : 0.0f;
            float logic_loss = 0, attn_loss = 0, div_loss = 0;
            if (do_logic) {
                logic_loss = logic_guided_regularization(m, effective_logic_lr);
                attn_loss = attn_logic_regularization(m, effective_attn_lr);
                /* v13l: 多样性损失 — 直接惩罚概念对间隐藏状态余弦相似度 */
                div_loss = residual_diversity_loss(m, effective_logic_lr);
                /* v15: 关系监督 — 在 wte 空间拉近相关概念/推远无关概念,
                 * 让原生推理概念链(鸟→动物, 火→光)有意义. 直接用 wte 行梯度,
                 * lr 取 logic_lr 量级, BPE 单 token 独立安全. */
                float rel_loss = relation_logic_regularization(m, logic_lr * 0.5f);
                (void)rel_loss;
            }
            /* v13g: logic-only 模式下 n_valid=0, 用 batch_size=1 做 apply */
            int apply_batch = logic_only ? 1 : n_valid;
            model_batch_apply(m, lr, apply_batch);
            /* g_opt_step is incremented inside model_batch_apply — don't double-increment */
            if (step % 10 == 0) {
                printf("  [LOGIC] mlp=%.4f attn=%.4f div=%.4f\n", logic_loss, attn_loss, div_loss);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t_apply_start);

        /* v13k: timing breakdown — find GPU bottleneck */
        if (step % 10 == 0) {
            double ce_ms = (t_logic_start.tv_sec-t_ce_start.tv_sec)*1000 + (t_logic_start.tv_nsec-t_ce_start.tv_nsec)/1e6;
            double logic_ms = (t_apply_start.tv_sec-t_logic_start.tv_sec)*1000 + (t_apply_start.tv_nsec-t_logic_start.tv_nsec)/1e6;
            printf("  [TIME] CE=%.0fms logic_reg=%.0fms\n", ce_ms, logic_ms);
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
            printf("  step %5d/%d (global=%d)  loss=%.4f  avg=%.4f  lr=%.6f  B=%d  %.0fms\n",
                   step, n_steps, global_step, avg_loss, recent_loss, lr, n_valid,
                   dt / (step + 1) * 1000);
            fflush(stdout);

            if (recent_loss < best_recent_loss) best_recent_loss = recent_loss;
        }

        /* 白箱探针:每 10 步检查 CORE/BINARY/PRUNE 逻辑电路 */
        if (step % 10 == 0) {
            whitebox_probe_compact(m);
        }

        /* 逻辑引导已在 model_batch_apply 前累加梯度 */

        /* 每 50 步做一次推理 trace */
        if (step > 0 && step % 50 == 0) {
            const char *trace_prompts[] = {
                "\xe7\x83\xad", "\xe5\x86\xb7", "\xe7\x81\xab",
                "\xe6\xb0\xb4", "\xe5\xa4\xa7", "\xe5\xb0\x8f"
            };
            const char *trace_labels[] = {"热","冷","火","水","大","小"};
            int ti = (global_step / 50) % 6;
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

/* === LAL 原生推理: 概念链检索 (不走 transformer 前向/采样) ===
 * 直接读 wte 余弦相似度, 对问句中命中的概念在概念表内做 top-k 检索,
 * 串成 "概念链" 输出. 这是 LAL 的原生推理形态, 与 generate_text 的自回归
 * 句子生成互补:
 *   - 句子层(generate_text): 逐 token 自回归, 产出完整中文答复
 *   - 概念链层(lal_native_chain): wte 余弦, 直接给出关联概念(火→热→光)
 * 概念表复用白箱探针的 bpe_token_map(已 include 自 lal_whitebox_probe.h).
 */
static void lal_native_chain(Model *m, const char *prompt, int top_k, int depth) {
    int n_embd = m->cfg.n_embd;
    if (top_k <= 0) top_k = 3;
    if (depth <= 0) depth = 1;

    printf("\n=== LAL Native Reasoning — Concept Chain (wte cosine, NO forward) ===\n");
    printf("[*] query: %s\n", prompt);

    /* 1) 在概念表中命中问句里出现的概念 */
    float *q_emb = malloc(n_embd * sizeof(float));
    float *c_emb = malloc(n_embd * sizeof(float));
    int hits[64]; int n_hits = 0;
    for (int i = 0; i < (int)N_BPE_MAP && n_hits < 64; i++) {
        if (strstr(prompt, bpe_token_map[i].utf8)) {
            hits[n_hits++] = i;
        }
    }
    if (n_hits == 0) {
        printf("[!] 问句中未命中已知概念表, 无法用原生推理检索.\n");
        printf("    已知概念: 火 水 山 花 树 鸟 鱼 人 天 地 月 星 风 雨 云 太阳 猫 苹果 动物 植物 红 黄 蓝 绿 光 温 冰 雪 等\n");
        free(q_emb); free(c_emb);
        return;
    }

    printf("[*] 命中概念(%d): ", n_hits);
    for (int h = 0; h < n_hits; h++) printf("%s ", bpe_token_map[hits[h]].utf8);
    printf("\n");

    /* 2) 逐层展开概念链: 每步对当前概念取 top-k 最相似概念(排除已访问) */
    char visited[256]; memset(visited, 0, sizeof(visited));
    for (int h = 0; h < n_hits; h++) visited[hits[h]] = 1;

    printf("\n  概念链:\n");
    for (int h = 0; h < n_hits; h++) {
        int cur = hits[h];
        printf("  %s", bpe_token_map[cur].utf8);
        get_concept_embedding(m, bpe_token_map[cur].utf8, q_emb, n_embd);

        int chain[16]; int chain_len = 0;
        int frontier = cur;
        for (int d = 0; d < depth; d++) {
            /* 在概念表里找与 frontier 余弦最高的 top_k, 排除已访问 */
            float best_sim[16]; int best_idx[16];
            for (int k = 0; k < top_k; k++) { best_sim[k] = -2.0f; best_idx[k] = -1; }
            get_concept_embedding(m, bpe_token_map[frontier].utf8, c_emb, n_embd);
            for (int j = 0; j < (int)N_BPE_MAP; j++) {
                if (visited[j]) continue;
                float tmp[4096];
                get_concept_embedding(m, bpe_token_map[j].utf8, tmp, n_embd);
                float sim = cosine_sim(c_emb, tmp, n_embd);
                /* 插入 top_k */
                if (sim > best_sim[top_k - 1]) {
                    int p = top_k - 1;
                    while (p > 0 && best_sim[p - 1] < sim) {
                        best_sim[p] = best_sim[p - 1]; best_idx[p] = best_idx[p - 1]; p--;
                    }
                    best_sim[p] = sim; best_idx[p] = j;
                }
            }
            if (best_idx[0] < 0) break;
            /* 输出本层 top 关联 */
            printf(" → ");
            for (int k = 0; k < top_k && best_idx[k] >= 0; k++) {
                int j = best_idx[k];
                visited[j] = 1;
                if (k > 0) printf("/");
                printf("%s(%.2f)", bpe_token_map[j].utf8, best_sim[k]);
                if (k == 0) frontier = j;  /* 沿最强链继续展开 */
                if (d == 0) chain[chain_len++] = j;
            }
        }
        printf("\n");
    }
    printf("\n  (链中数字为 wte 余弦相似度; 无需前向, 纯嵌入空间检索)\n");
    free(q_emb); free(c_emb);
}

/* === 生成文本 (用 stateful sliding window inference) === */
static void generate_text(Model *m, const char *prompt, const int *prompt_ids,
                          int n_prompt_ids, int max_gen, float temp, int top_k) {
    printf("\n=== Generation (temp=%.2f, top_k=%d, max=%d tokens) ===\n",
           temp, top_k, max_gen);
    g_use_adam = 0;
    g_use_pure_float = 1;  /* 生成用 pure_float(完整浮点),避免 bin_forward 二值化丢失输入区分度 */
    g_use_real_attention = 1;
    g_skip_wv = 1;  /* v13l: match training — skip W_v projection */

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
    /* 重复惩罚:对已生成的 token 降权 (每个 token 只惩罚一次,不重复)
     * BUG #39 FIX: 用 char 数组标记是否已惩罚,避免 float 的精度问题 */
    static char penalty[40000];
    memset(penalty, 0, vocab * sizeof(char));

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

        /* 重复惩罚:对已生成的 token logit 除以 1.5
         * BUG #39 FIX: 旧代码对同一个 token 多次出现会重复惩罚
         * (如 token 5 出现 3 次 → logit /= 1.5 三次 → logit / 3.375).
         * 标准 HuggingFace 重复惩罚只对每个 token 惩罚一次,不管出现几次。
         * 修复:用 penalty[] 数组标记已生成的 token,只惩罚一次。 */
        for (int j = 0; j < out_len; j++) {
            int tok = out_tokens[j];
            if (tok >= 0 && tok < vocab && !penalty[tok]) {
                if (adjusted_logits[tok] > 0)
                    adjusted_logits[tok] /= 1.5f;
                else
                    adjusted_logits[tok] *= 1.5f;
                penalty[tok] = 1;  /* 标记已惩罚,后续不再重复 */
            }
        }

        /* No-repeat-trigram: 如果当前 token 会和前 2 个 token 形成一个已经出现过的
         * 3-gram,就屏蔽它。这防止模型陷入 "准和准和准和..." 这样的循环。
         * 标准 GPT-2/LLaMA 生成都用这个技巧。
         * 只在 out_len >= 2 时检查 (需要至少 2 个前缀 token)。 */
        if (out_len >= 2) {
            int prev1 = out_tokens[out_len - 1];
            int prev2 = out_tokens[out_len - 2];
            /* 搜索所有历史位置,看 (prev2, prev1, X) 是否出现过 */
            for (int j = 0; j + 2 < out_len; j++) {
                if (out_tokens[j] == prev2 && out_tokens[j+1] == prev1) {
                    int banned = out_tokens[j+2];
                    if (banned >= 0 && banned < vocab)
                        adjusted_logits[banned] = -1e30f;
                }
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
    /* ========================================================================
     * 默认参数已固化 = 对话训练唯一正确的路 (v5 验证: 4000步 avg_loss=1.58,
     * 句子层出"你好。"完整短句, 概念链层 火→光(0.93)/鸟→动物(0.89)).
     * 克隆仓库的人直接 `./ste_train` 或 `make dialogue-train` 即走此路,
     * 无需手动拼参数, 避免走岔路浪费时间.
     * ===================================================================== */
    int n_steps = 4000;      /* 对话训练标准步数 (v5); 改小会句子层不出可读短句 */
    int batch_size = 4;
    float base_lr = 0.001f;  /* CE 学习率 */
    float logic_lr = 0.005f; /* 逻辑/关系正则倍率: 拉概念关系(火→热/鸟→动物) */
    int phase_idx = 0;       /* 8L/448d ~22M, 对话训练稳定档 */
    int vocab_size = 32768;  /* BPE 中文 tokenizer, 必须 32768 (非 256 byte 模式) */
    int start_step = 0;    /* --start-step: 续训时跳过的步数,用于正确恢复 LR schedule */
    int total_schedule_steps = 0; /* --total-steps: LR schedule 总步数 (0=用 n_steps) */
    const char *data_path = "data/dialogue_bpe.bin";  /* 对话+认知合并数据(唯一正确数据) */
    const char *weights_path = "/tmp/lal_ste_model.bin";
    const char *save_path = "model_dialogue.ste";  /* 默认落盘名, 避免忘了 --save */
    const char *resume_path = NULL; /* --resume: 加载已训练权重续训 */
    const char *prompt = "\xe7\x81\xab\xe6\x98\xaf\xe4\xbb\x80\xe4\xb9\x88";  /* 火是什么 */
    int prompt_ids[64];      /* --prompt-ids 直接传 BPE token id */
    int n_prompt_ids = 0;    /* >0 表示用 --prompt-ids 而非 --prompt */
    int do_generate = 1;
    int max_gen = 40;
    float temp = 0.4f;
    int top_k = 8;
    int logic_only = 0;  /* v13g: --logic-only, 课程学习 stage 1 只训概念对不训 CE */
    /* v15: 原生推理概念链默认开启 —— LAL 的核心形态(纯 wte 余弦, 不走 transformer 前向).
     * diagnose 模式(n_steps==0)下自动跑, 无需手动加 --native-chain. */
    int native_chain = 1; /* 默认开启 LAL 原生推理概念链 */
    int native_topk = 3;  /* --native-topk: 每层检索概念数 */
    int native_depth = 2; /* --native-depth: 链展开层数 */
    const char *tokenizer_path = "tokenizer/chinese_bpe.vocab";  /* --tokenizer */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-size") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) base_lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--logic-lr") && i+1 < argc) logic_lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--phase") && i+1 < argc) phase_idx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--start-step") && i+1 < argc) start_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--total-steps") && i+1 < argc) total_schedule_steps = atoi(argv[++i]);
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
        else if (!strcmp(argv[i], "--logic-only")) { logic_only = 1; }  /* v13g */
        else if (!strcmp(argv[i], "--tokenizer") && i+1 < argc) tokenizer_path = argv[++i];
        else if (!strcmp(argv[i], "--native-chain")) { native_chain = 1; }
        else if (!strcmp(argv[i], "--native-topk") && i+1 < argc) native_topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--native-depth") && i+1 < argc) native_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: ste_train [options]\n"
                   "  --steps N         Training steps (default 4000) [对话训练推荐值]\n"
                   "  --batch-size N    Batch size (default 4)\n"
                   "  --lr F            Base learning rate (default 0.001)\n"
                   "  --logic-lr F      Logic guidance multiplier (default 0.005 = 拉概念关系)\n"
                   "  --start-step N    Skip N steps for LR schedule (used with --resume for chunked training)\n"
                   "  --total-steps N   Total steps for LR schedule (default: --steps; set larger for chunked training)\n"
                   "  --data PATH       Data .bin path (default: data/dialogue_bpe.bin)\n"
                   "  --tokenizer PATH  Tokenizer .vocab path (default: tokenizer/chinese_bpe.vocab)\n"
                   "  --save PATH       Save trained weights to .ste file (default: model_dialogue.ste)\n"
                   "  --resume PATH     Load .ste weights and continue training\n"
                   "  --diagnose-only   Skip training, just load + diagnose + generate\n"
                   "  --prompt TEXT     Generation prompt\n"
                   "  --max-gen N       Max generation tokens (default 40)\n"
                   "  --temp F          Sampling temperature (default 0.4)\n"
                   "  --no-generate     Skip generation\n"
                   "  --native-chain    LAL native reasoning: print concept chain via wte cosine (no forward) [默认已开启]\n"
                   "  --native-topk N   Concepts per chain layer (default 3)\n"
                   "  --native-depth N  Chain expansion depth (default 2)\n"
                   "\n"
                   "=== 唯一正确的路 (勿手动改参数, 避免走岔路) ===\n"
                   "  训练:   make dialogue          (生成数据→训练4000步→测双形态)\n"
                   "    或:   ./ste_train --steps 4000 --batch-size 4 --lr 0.001 --logic-lr 0.005 \\\n"
                   "              --phase 0 --vocab 32768 --data data/dialogue_bpe.bin --no-generate --save model_dialogue.ste\n"
                   "  诊断:   ./ste_train --diagnose-only --phase 0 --vocab 32768 --resume model_dialogue.ste \\\n"
                   "              --prompt '鸟为什么天上飞？'   (原生推理概念链默认开启)\n"
                   "  数据:   对话数据=data/dialogue_bpe.bin (data/gen_dialogue_data.py 生成, 与认知数据合并)\n"
                   "  注意:   vocab 必须 32768(BPE), 不要用 256(byte 模式); 数据必须用 dialogue_bpe.bin\n");
            return 0;
        }
    }

    srand(time(NULL));
    setvbuf(stdout, NULL, _IONBF, 0);

    /* BPE 模式加载 vocab */
    if (vocab_size > 256) {
        load_vocab(tokenizer_path);
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
    g_skip_wv = 1;  /* v13l: Skip W_v projection, use LN1 output as attention output */
    g_use_logic_binarization = 1;
    g_use_cuda = 1;  /* v13i: enable CUDA for attn_q (no logic_mask layers) */
    /* Ablation: MLP-only test (set attn scale to 0 via env var) */
    {
        const char *attn_scale = getenv("LAL_ATTN_SCALE");
        if (attn_scale) {
            g_attn_residual_scale = atof(attn_scale);
            printf("[*] Ablation: g_attn_residual_scale = %.2f\n", g_attn_residual_scale);
        }
    }

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
        int total_steps = total_schedule_steps > 0 ? total_schedule_steps : n_steps;
        ste_train(&model, &dl, n_steps, base_lr, 50, batch_size, 256, 10, logic_lr, start_step, total_steps, logic_only);
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

    /* v14: 关系探针 — 监控概念间关系(火→热,猫→动物),不只看概念区分 */
    relation_probe(&model);

    /* === 逐层 hidden state 诊断: 定位 logits 坍缩根因 ===
     * 比较两个不同 prompt ('火' vs '水') 在每层的 hidden state cosine similarity.
     * 如果某层后 cosine→1.0, 说明该层抹掉了 prompt 信号. */
    {
        int n = model.cfg.n_embd;
        int nL = model.cfg.n_layer;
        int tok_fire = 1164;   /* 火 (v2 tokenizer: ▁火 = 259+1164, use 1164 directly) */
        int tok_water = 962;   /* 水 (v2 tokenizer: ▁水 = 259+962, use 962 directly) */

        static float xa[4096], xb[4096];
        static float la[16][4096], lb[16][4096];

        /* prompt A: 火 */
        for (int i = 0; i < n; i++) {
            xa[i] = model.wte[tok_fire * n + i];
            if (model.wpe) xa[i] += model.wpe[0 * n + i];
        }
        memcpy(la[0], xa, n * sizeof(float));
        for (int l = 0; l < nL; l++) {
            trans_layer_forward(xa, &model.layers[l], &model.acts[l], &model.cfg, 0);
            memcpy(la[l+1], xa, n * sizeof(float));
        }

        /* prompt B: 水 */
        for (int i = 0; i < n; i++) {
            xb[i] = model.wte[tok_water * n + i];
            if (model.wpe) xb[i] += model.wpe[0 * n + i];
        }
        memcpy(lb[0], xb, n * sizeof(float));
        for (int l = 0; l < nL; l++) {
            trans_layer_forward(xb, &model.layers[l], &model.acts[l], &model.cfg, 0);
            memcpy(lb[l+1], xb, n * sizeof(float));
        }

        printf("\n========================================\n");
        printf("  LAYER-BY-LAYER COLLAPSE DIAGNOSIS\n");
        printf("  (火 vs 水 — cosine similarity per layer)\n");
        printf("========================================\n\n");
        printf("%-12s  %-12s  %-12s  %-12s\n", "Layer", "cosine_sim", "l2_dist", "||a||");
        printf("%-12s  %-12s  %-12s  %-12s\n", "-----", "----------", "-------", "------");
        for (int l = 0; l <= nL; l++) {
            float dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) {
                dot += la[l][i] * lb[l][i];
                na += la[l][i] * la[l][i];
                nb += lb[l][i] * lb[l][i];
            }
            float cs = dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
            float l2 = 0;
            for (int i = 0; i < n; i++) { float d = la[l][i]-lb[l][i]; l2 += d*d; }
            l2 = sqrtf(l2);
            char lname[16];
            if (l == 0) snprintf(lname, sizeof lname, "emb");
            else snprintf(lname, sizeof lname, "L%d", l-1);
            printf("%-12s  %-12.6f  %-12.6f  %-12.6f\n", lname, cs, l2, sqrtf(na));
        }

        /* final LayerNorm 后 */
        float fa[4096], fb[4096];
        memcpy(fa, la[nL], n * sizeof(float));
        memcpy(fb, lb[nL], n * sizeof(float));
        norm_forward(fa, fa, model.ln_f_w, model.ln_f_b, model.cfg.norm_type, n);
        norm_forward(fb, fb, model.ln_f_w, model.ln_f_b, model.cfg.norm_type, n);
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += fa[i] * fb[i]; na += fa[i]*fa[i]; nb += fb[i]*fb[i];
        }
        printf("%-12s  %-12.6f  (final_ln output)\n", "final_ln",
               dot / (sqrtf(na) * sqrtf(nb) + 1e-12f));
        printf("\n  (1.0=identical, 0.0=orthogonal)\n\n");

        /* 检查 L0 的关键权重 */
        printf("--- L0 Weight Check ---\n");
        TransLayer *tl0 = &model.layers[0];
        printf("  L0 norm1_w[0:5]: ");
        for (int i = 0; i < 5 && i < n; i++) printf("%.4f ", tl0->norm1_w[i]);
        printf("\n");
        printf("  L0 norm1_w mean: ");
        float mw = 0;
        for (int i = 0; i < n; i++) mw += tl0->norm1_w[i];
        printf("%.4f\n", mw / n);
        printf("  L0 norm1_b[0:5]: ");
        for (int i = 0; i < 5 && i < n; i++) printf("%.4f ", tl0->norm1_b[i]);
        printf("\n");

        /* 深度诊断: norm1_out 和 V 的对比 */
        printf("\n--- L0 Deep Diagnosis (norm1_out + V) ---\n");
        /* 重新前向 L0, 捕获中间结果 */
        static float xa2[4096], xb2[4096];
        static float norm1_a[4096], norm1_b[4096];
        static float q_a[4096], q_b[4096];  /* QKV merged: [Q|K|V] */
        static float v_a[4096], v_b[4096];
        static float attn_out_a[4096], attn_out_b[4096];
        static float proj_out_a[4096], proj_out_b[4096];

        /* prompt A: 火 */
        for (int i = 0; i < n; i++) {
            xa2[i] = model.wte[tok_fire * n + i];
            if (model.wpe) xa2[i] += model.wpe[0 * n + i];
        }
        norm_forward(norm1_a, xa2, tl0->norm1_w, tl0->norm1_b, model.cfg.norm_type, n);
        bin_forward_pure_float(q_a, norm1_a, &tl0->attn_q);  /* QKV merged */
        /* V is at offset 2*n */
        memcpy(v_a, q_a + 2*n, n * sizeof(float));

        /* prompt B: 水 */
        for (int i = 0; i < n; i++) {
            xb2[i] = model.wte[tok_water * n + i];
            if (model.wpe) xb2[i] += model.wpe[0 * n + i];
        }
        norm_forward(norm1_b, xb2, tl0->norm1_w, tl0->norm1_b, model.cfg.norm_type, n);
        bin_forward_pure_float(q_b, norm1_b, &tl0->attn_q);
        memcpy(v_b, q_b + 2*n, n * sizeof(float));

        /* cosine sim at each stage */
        float cs_norm1 = 0, cs_v = 0, cs_emb = 0;
        float na1=0, nb1=0, na2_=0, nb2_=0, na3=0, nb3=0;
        for (int i = 0; i < n; i++) {
            cs_emb += xa2[i]*xb2[i]; na3 += xa2[i]*xa2[i]; nb3 += xb2[i]*xb2[i];
            cs_norm1 += norm1_a[i]*norm1_b[i]; na1 += norm1_a[i]*norm1_a[i]; nb1 += norm1_b[i]*norm1_b[i];
            cs_v += v_a[i]*v_b[i]; na2_ += v_a[i]*v_a[i]; nb2_ += v_b[i]*v_b[i];
        }
        cs_emb /= (sqrtf(na3)*sqrtf(nb3)+1e-12f);
        cs_norm1 /= (sqrtf(na1)*sqrtf(nb1)+1e-12f);
        cs_v /= (sqrtf(na2_)*sqrtf(nb2_)+1e-12f);

        printf("  embedding (火vs水) cosine: %.6f  ||火||=%.4f ||水||=%.4f\n", cs_emb, sqrtf(na3), sqrtf(nb3));
        printf("  norm1_out   (火vs水) cosine: %.6f  ||火||=%.4f ||水||=%.4f\n", cs_norm1, sqrtf(na1), sqrtf(nb1));
        printf("  V=attn_v    (火vs水) cosine: %.6f  ||火||=%.4f ||水||=%.4f\n", cs_v, sqrtf(na2_), sqrtf(nb2_));

        /* 打印前5维对比 */
        printf("\n  norm1_out 火 [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", norm1_a[i]);
        printf("\n  norm1_out 水 [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", norm1_b[i]);
        printf("\n  V 火        [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", v_a[i]);
        printf("\n  V 水        [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", v_b[i]);
        printf("\n");

        /* attn_v 权重统计 */
        float wv_mean = 0, wv_max = 0, wv_min = 1e10;
        int wv_n = tl0->attn_q.out_dim * tl0->attn_q.in_dim;
        for (int i = 0; i < wv_n; i++) {
            float w = tl0->attn_q.w_float[i];
            wv_mean += w;
            if (w > wv_max) wv_max = w;
            if (w < wv_min) wv_min = w;
        }
        printf("\n  attn_q (QKV merged) weights: mean=%.6f min=%.6f max=%.6f n=%d\n",
               wv_mean/wv_n, wv_min, wv_max, wv_n);
        printf("  (如果 V 火vs水 cosine 高, 但 norm1_out cosine 低, 说明 W_v 把不同输入映射到相似输出)\n\n");
    }

    /* LAL 原生推理: 概念链(不走 transformer 前向, 纯 wte 余弦检索).
     * 默认开启: diagnose 模式(n_steps==0)下自动跑, 这就是 LAL 的核心形态,
     * 别人诊断模型时自动看到概念链, 不会再误走 transformer 自回归岔路. */
    if (n_steps == 0) {
        lal_native_chain(&model, prompt, native_topk, native_depth);
    }

    /* 生成(句子层, transformer 自回归). 默认也跑, 但原生推理是主形态. */
    if (do_generate) {
        generate_text(&model, prompt, prompt_ids, n_prompt_ids, max_gen, temp, top_k);
    }

    printf("\n[*] Done\n");
    model_free(&model);
    dataloader_free(&dl);
    return 0;
}
