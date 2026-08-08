/* tests/test_concept_attn.c — Concept-Aware Attention 单元测试
 *
 * 验证四层概念感知注意力优化的正确性：
 *   1. 信使缓存管理（alloc/free/reset）
 *   2. 信使生成（均值池化正确性）
 *   3. 概念边界门控（软门控 + 回退通路）
 *   4. 异构多头访问域分配
 *   5. 概念感知注意力前向（与标准注意力在 enable=0 时一致）
 *   6. 概念感知注意力反向（梯度形状正确）
 *
 * Build: make test-concept
 * Run:   ./test_concept
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../runtime/lal_runtime.h"
#include "../runtime/lal_concept_attn.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { g_tests_passed++; printf("  [PASS] %s\n", msg); } \
    else { g_tests_failed++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define TEST_ASSERT_FLOAT_NEAR(a, b, eps, msg) do { \
    float _diff = fabsf((a) - (b)); \
    if (_diff < (eps)) { g_tests_passed++; printf("  [PASS] %s (diff=%.6f)\n", msg, _diff); } \
    else { g_tests_failed++; printf("  [FAIL] %s (a=%.6f, b=%.6f, diff=%.6f)\n", msg, (a), (b), _diff); } \
} while(0)

/* ─── Test 1: Messenger Cache Management ───────────────────────── */
static void test_messenger_cache_lifecycle(void) {
    printf("\n[Test 1] Messenger Cache Lifecycle\n");
    MessengerCache mc = {0};
    messenger_cache_alloc(&mc, 8, 4, 64);
    TEST_ASSERT(mc.segment_capacity == 8, "alloc: segment_capacity == 8");
    TEST_ASSERT(mc.num_messengers == 4, "alloc: num_messengers == 4");
    TEST_ASSERT(mc.n_embd == 64, "alloc: n_embd == 64");
    TEST_ASSERT(mc.messenger_k != NULL, "alloc: messenger_k != NULL");
    TEST_ASSERT(mc.messenger_v != NULL, "alloc: messenger_v != NULL");
    TEST_ASSERT(mc.segment_filled != NULL, "alloc: segment_filled != NULL");

    /* 写入一些数据，然后 reset 验证清零 */
    memset(mc.messenger_k, 0xAA, 8 * 4 * 64 * sizeof(float));
    mc.segment_filled[0] = 1;
    mc.n_filled = 1;
    messenger_cache_reset(&mc);
    TEST_ASSERT(mc.n_filled == 0, "reset: n_filled == 0");
    TEST_ASSERT(mc.segment_filled[0] == 0, "reset: segment_filled[0] == 0");
    int all_zero = 1;
    for (int i = 0; i < 8 * 4 * 64; i++) {
        if (mc.messenger_k[i] != 0.0f) { all_zero = 0; break; }
    }
    TEST_ASSERT(all_zero, "reset: messenger_k all zero");

    messenger_cache_free(&mc);
    TEST_ASSERT(mc.messenger_k == NULL, "free: messenger_k == NULL");
    TEST_ASSERT(mc.messenger_v == NULL, "free: messenger_v == NULL");
    TEST_ASSERT(mc.segment_filled == NULL, "free: segment_filled == NULL");
    TEST_ASSERT(mc.segment_capacity == 0, "free: segment_capacity == 0");
}

/* ─── Test 2: Segment Messenger Generation ─────────────────────── */
static void test_messenger_generation(void) {
    printf("\n[Test 2] Segment Messenger Generation (mean pooling)\n");
    int seg_len = 8;
    int n_embd = 4;
    int num_messengers = 2;
    float v_seg[8 * 4];
    /* 构造已知 V：每个 token 的 V = [t, t, t, t] */
    for (int t = 0; t < seg_len; t++)
        for (int d = 0; d < n_embd; d++)
            v_seg[t * n_embd + d] = (float)t;

    float mk[2 * 4], mv[2 * 4];
    generate_segment_messengers(v_seg, seg_len, n_embd, num_messengers, mk, mv);

    /* 桶 0: tokens 0..3, 均值 = (0+1+2+3)/4 = 1.5 */
    /* 桶 1: tokens 4..7, 均值 = (4+5+6+7)/4 = 5.5 */
    TEST_ASSERT_FLOAT_NEAR(mv[0], 1.5f, 1e-5f, "messenger 0 V[0] == 1.5 (mean of 0..3)");
    TEST_ASSERT_FLOAT_NEAR(mv[4], 5.5f, 1e-5f, "messenger 1 V[0] == 5.5 (mean of 4..7)");
    /* 信使 K = V */
    TEST_ASSERT_FLOAT_NEAR(mk[0], 1.5f, 1e-5f, "messenger 0 K[0] == V[0] (self-association)");
    TEST_ASSERT_FLOAT_NEAR(mk[4], 5.5f, 1e-5f, "messenger 1 K[0] == V[0] (self-association)");
}

/* ─── Test 3: Concept Boundary Gate ────────────────────────────── */
static void test_concept_boundary_gate(void) {
    printf("\n[Test 3] Concept Boundary Gate (soft gating + fallback)\n");
    ConceptAttnConfig cfg = {0};
    cfg.gate_enable = 1;
    cfg.gate_window = 64;
    cfg.gate_threshold = 0.3f;
    cfg.gate_fallback_prob = 0.01f;  /* 极小回退概率 */
    cfg.gate_distance_prior = 1;
    cfg.segment_len = 32;

    int head_dim = 32;
    float q[32], k[32];
    /* 高相似度：Q == K，应该保留 */
    for (int i = 0; i < head_dim; i++) { q[i] = 1.0f; k[i] = 1.0f; }
    int keep = concept_boundary_gate(q, k, head_dim, 1, &cfg);
    TEST_ASSERT(keep == 1, "high similarity (Q==K, dist=1): keep");

    /* 低相似度 + 远距离：应该大概率屏蔽 */
    for (int i = 0; i < head_dim; i++) { q[i] = 1.0f; k[i] = -1.0f; }
    int masked_count = 0;
    for (int trial = 0; trial < 100; trial++) {
        /* 用不同 distance 触发不同 hash */
        if (!concept_boundary_gate(q, k, head_dim, 100 + trial, &cfg))
            masked_count++;
    }
    /* 绝大多数应该被屏蔽（允许极少数回退） */
    TEST_ASSERT(masked_count >= 90, "low sim + far distance: >=90% masked (100 trials)");

    /* 门控禁用：全部保留 */
    cfg.gate_enable = 0;
    keep = concept_boundary_gate(q, k, head_dim, 1000, &cfg);
    TEST_ASSERT(keep == 1, "gate disabled: always keep");
}

/* ─── Test 4: Heterogeneous Head Access ────────────────────────── */
static void test_hetero_heads(void) {
    printf("\n[Test 4] Heterogeneous Head Access Domains\n");
    ConceptAttnConfig cfg = {0};
    cfg.hetero_enable = 1;
    cfg.n_local_heads = 4;       /* 头 0..3: LOCAL */
    cfg.n_messenger_heads = 2;  /* 头 4..5: MESSENGER */
    /* 头 6..7: GLOBAL */

    int n_head = 8;
    TEST_ASSERT(get_head_access_type(0, n_head, &cfg) == HEAD_LOCAL, "head 0: LOCAL");
    TEST_ASSERT(get_head_access_type(3, n_head, &cfg) == HEAD_LOCAL, "head 3: LOCAL");
    TEST_ASSERT(get_head_access_type(4, n_head, &cfg) == HEAD_MESSENGER, "head 4: MESSENGER");
    TEST_ASSERT(get_head_access_type(5, n_head, &cfg) == HEAD_MESSENGER, "head 5: MESSENGER");
    TEST_ASSERT(get_head_access_type(6, n_head, &cfg) == HEAD_GLOBAL, "head 6: GLOBAL");
    TEST_ASSERT(get_head_access_type(7, n_head, &cfg) == HEAD_GLOBAL, "head 7: GLOBAL");

    /* 窗口分配：LOCAL < MESSENGER < GLOBAL */
    int base_window = 64;
    int w_local = get_head_window(0, n_head, base_window, &cfg);
    int w_msg = get_head_window(4, n_head, base_window, &cfg);
    int w_global = get_head_window(6, n_head, base_window, &cfg);
    TEST_ASSERT(w_local == 32, "LOCAL head window = base/2 = 32");
    TEST_ASSERT(w_msg == 64, "MESSENGER head window = base = 64");
    TEST_ASSERT(w_global == 128, "GLOBAL head window = base*2 = 128");

    /* 信使访问权限 */
    TEST_ASSERT(head_can_access_messenger(0, n_head, &cfg) == 0, "LOCAL head: no messenger access");
    TEST_ASSERT(head_can_access_messenger(4, n_head, &cfg) == 1, "MESSENGER head: has messenger access");
    TEST_ASSERT(head_can_access_messenger(6, n_head, &cfg) == 1, "GLOBAL head: has messenger access");

    /* 异构禁用：全部 GLOBAL，全部可访问信使 */
    cfg.hetero_enable = 0;
    TEST_ASSERT(get_head_access_type(0, n_head, &cfg) == HEAD_GLOBAL, "hetero disabled: all GLOBAL");
    TEST_ASSERT(head_can_access_messenger(0, n_head, &cfg) == 1, "hetero disabled: all can access messenger");
}

/* ─── Test 5: Concept Attention Forward (disabled == standard) ─── */
static void test_concept_attn_disabled_matches_standard(void) {
    printf("\n[Test 5] Concept Attention (disabled) matches standard attention\n");
    int n_embd = 64;
    int n_head = 4;
    int n_ctx = 16;
    int seq_pos = 8;

    float qkv[3 * n_embd];
    float k_cache[n_ctx * n_embd];
    float v_cache[n_ctx * n_embd];
    float out_std[n_embd], out_concept[n_embd];

    /* 随机初始化（确定性种子） */
    srand(42);
    for (int i = 0; i < 3 * n_embd; i++) qkv[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < n_ctx * n_embd; i++) {
        k_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* 标准注意力 */
    float kc_std[n_ctx * n_embd], vc_std[n_ctx * n_embd];
    memcpy(kc_std, k_cache, sizeof(kc_std));
    memcpy(vc_std, v_cache, sizeof(vc_std));
    attention_forward(out_std, qkv, n_embd, n_head, seq_pos, kc_std, vc_std);

    /* 概念感知注意力（disabled） */
    ConceptAttnConfig cfg = {0};  /* enable = 0 */
    attention_forward_concept(out_concept, qkv, n_embd, n_head, seq_pos,
                               k_cache, v_cache, n_ctx, &cfg, NULL);

    /* 验证输出一致（允许浮点误差） */
    int match = 1;
    float max_diff = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        float diff = fabsf(out_std[i] - out_concept[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) match = 0;
    }
    TEST_ASSERT(match, "disabled concept attn matches standard attn");
    printf("    max_diff = %.6f\n", max_diff);
}

/* ─── Test 6: Concept Attention Forward (enabled, basic run) ────── */
static void test_concept_attn_enabled_runs(void) {
    printf("\n[Test 6] Concept Attention (enabled) runs without crash\n");
    int n_embd = 64;
    int n_head = 4;
    int n_ctx = 64;
    int seq_pos = 32;

    float qkv[3 * n_embd];
    float k_cache[n_ctx * n_embd];
    float v_cache[n_ctx * n_embd];
    float out[n_embd];

    srand(123);
    for (int i = 0; i < 3 * n_embd; i++) qkv[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < n_ctx * n_embd; i++) {
        k_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* 配置：启用全部四层 */
    ConceptAttnConfig cfg = {0};
    cfg.enable = 1;
    cfg.segment_len = 16;
    cfg.num_messengers = 4;
    cfg.messenger_neighbors = 2;
    cfg.gate_enable = 1;
    cfg.gate_window = 32;
    cfg.gate_threshold = 0.1f;
    cfg.gate_fallback_prob = 0.05f;
    cfg.gate_distance_prior = 1;
    cfg.hetero_enable = 1;
    cfg.n_local_heads = 2;
    cfg.n_messenger_heads = 1;

    MessengerCache mc = {0};
    messenger_cache_alloc(&mc, 8, cfg.num_messengers, n_embd);

    /* 模拟前 32 个 token 已经处理（填充 KV cache + 生成前两个片段的信使） */
    /* 片段 0: tokens 0..15, 片段 1: tokens 16..31 */
    for (int seg = 0; seg < 2; seg++) {
        int seg_start = seg * cfg.segment_len;
        float *v_seg = v_cache + (size_t)seg_start * n_embd;
        float *mk = mc.messenger_k + (size_t)seg * cfg.num_messengers * n_embd;
        float *mv = mc.messenger_v + (size_t)seg * cfg.num_messengers * n_embd;
        generate_segment_messengers(v_seg, cfg.segment_len, n_embd,
                                    cfg.num_messengers, mk, mv);
        mc.segment_filled[seg] = 1;
    }
    mc.n_filled = 2;

    /* 运行概念感知注意力（不应崩溃） */
    attention_forward_concept(out, qkv, n_embd, n_head, seq_pos,
                              k_cache, v_cache, n_ctx, &cfg, &mc);

    /* 验证输出不是全零（有信息聚合） */
    float out_norm = 0.0f;
    for (int i = 0; i < n_embd; i++) out_norm += out[i] * out[i];
    out_norm = sqrtf(out_norm);
    TEST_ASSERT(out_norm > 1e-6f, "enabled concept attn produces non-zero output");
    printf("    output L2 norm = %.6f\n", out_norm);

    messenger_cache_free(&mc);
}

/* ─── Test 7: Concept Attention Backward (gradient shape) ──────── */
static void test_concept_attn_backward(void) {
    printf("\n[Test 7] Concept Attention Backward (gradient shape)\n");
    int n_embd = 32;
    int n_head = 2;
    int n_ctx = 16;
    int seq_pos = 4;

    float qkv[3 * n_embd];
    float grad_attn_out[n_embd];
    float grad_qkv[3 * n_embd];
    float k_cache[n_ctx * n_embd];
    float v_cache[n_ctx * n_embd];

    srand(77);
    for (int i = 0; i < 3 * n_embd; i++) qkv[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < n_embd; i++) grad_attn_out[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < n_ctx * n_embd; i++) {
        k_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        v_cache[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    ConceptAttnConfig cfg = {0};
    cfg.enable = 1;
    cfg.segment_len = 8;
    cfg.num_messengers = 2;
    cfg.messenger_neighbors = 1;
    cfg.gate_enable = 0;  /* 禁用门控简化测试 */
    cfg.hetero_enable = 0;

    MessengerCache mc = {0};
    messenger_cache_alloc(&mc, 4, cfg.num_messengers, n_embd);

    attention_backward_concept(grad_qkv, grad_attn_out, qkv, n_embd, n_head,
                                seq_pos, k_cache, v_cache, n_ctx, &cfg, &mc);

    /* 验证梯度不全为零（Q/K/V 都应有梯度） */
    float gQ_norm = 0, gK_norm = 0, gV_norm = 0;
    for (int i = 0; i < n_embd; i++) {
        gQ_norm += fabsf(grad_qkv[i]);
        gK_norm += fabsf(grad_qkv[n_embd + i]);
        gV_norm += fabsf(grad_qkv[2 * n_embd + i]);
    }
    TEST_ASSERT(gQ_norm > 1e-6f, "grad_Q non-zero");
    TEST_ASSERT(gK_norm > 1e-6f, "grad_K non-zero");
    TEST_ASSERT(gV_norm > 1e-6f, "grad_V non-zero");
    printf("    |grad_Q| = %.6f, |grad_K| = %.6f, |grad_V| = %.6f\n",
           gQ_norm, gK_norm, gV_norm);

    messenger_cache_free(&mc);
}

/* ─── Test 8: Numerical Gradient Check ─────────────────────────── */
static void test_numerical_gradient(void) {
    printf("\n[Test 8] Numerical Gradient Check (Q gradient)\n");
    int n_embd = 16;
    int n_head = 2;
    int n_ctx = 8;
    int seq_pos = 3;

    float qkv[3 * n_embd];
    float k_cache[n_ctx * n_embd];
    float v_cache[n_ctx * n_embd];
    float out[n_embd];
    float grad_out[n_embd];
    float grad_qkv[3 * n_embd];

    srand(99);
    for (int i = 0; i < 3 * n_embd; i++) qkv[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < n_ctx * n_embd; i++) {
        k_cache[i] = (float)(rand() % 100) / 100.0f;
        v_cache[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < n_embd; i++) grad_out[i] = 1.0f;  /* 简单 loss = sum(out) */

    ConceptAttnConfig cfg = {0};
    cfg.enable = 1;
    cfg.segment_len = 4;
    cfg.num_messengers = 1;
    cfg.gate_enable = 0;
    cfg.hetero_enable = 0;

    MessengerCache mc = {0};
    messenger_cache_alloc(&mc, 4, 1, n_embd);

    /* 解析梯度 */
    attention_backward_concept(grad_qkv, grad_out, qkv, n_embd, n_head,
                                seq_pos, k_cache, v_cache, n_ctx, &cfg, &mc);

    /* 数值梯度：对 Q[0] 做有限差分 */
    float eps = 1e-4f;
    float orig = qkv[0];
    qkv[0] = orig + eps;
    attention_forward_concept(out, qkv, n_embd, n_head, seq_pos,
                              k_cache, v_cache, n_ctx, &cfg, &mc);
    float loss_plus = 0.0f;
    for (int i = 0; i < n_embd; i++) loss_plus += out[i];

    qkv[0] = orig - eps;
    attention_forward_concept(out, qkv, n_embd, n_head, seq_pos,
                              k_cache, v_cache, n_ctx, &cfg, &mc);
    float loss_minus = 0.0f;
    for (int i = 0; i < n_embd; i++) loss_minus += out[i];
    qkv[0] = orig;

    float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
    float ana_grad = grad_qkv[0];
    float rel_err = fabsf(num_grad - ana_grad) / (fabsf(num_grad) + fabsf(ana_grad) + 1e-8f);
    printf("    Q[0]: analytical=%.6f, numerical=%.6f, rel_err=%.6f\n",
           ana_grad, num_grad, rel_err);
    TEST_ASSERT_FLOAT_NEAR(num_grad, ana_grad, 0.05f, "Q[0] gradient matches numerical");

    messenger_cache_free(&mc);
}

/* ─── Main ──────────────────────────────────────────────────────── */
int main(void) {
    printf("============================================================\n");
    printf(" Concept-Aware Attention Test Suite\n");
    printf(" 基于「理解(概念-边界) + 推理(关系演化)」框架\n");
    printf("============================================================\n");

    test_messenger_cache_lifecycle();
    test_messenger_generation();
    test_concept_boundary_gate();
    test_hetero_heads();
    test_concept_attn_disabled_matches_standard();
    test_concept_attn_enabled_runs();
    test_concept_attn_backward();
    test_numerical_gradient();

    printf("\n============================================================\n");
    printf(" Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("============================================================\n");
    return g_tests_failed == 0 ? 0 : 1;
}
