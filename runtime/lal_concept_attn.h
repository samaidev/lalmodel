/* runtime/lal_concept_attn.h — Concept-Aware Attention
 *
 * 基于「理解(概念-边界) + 推理(关系演化)」框架优化的注意力机制。
 *
 * 设计原点（从框架公理推导，非抄工程现成方案）：
 *   - Attention 的本职：发现概念之间真实的关系，给关系分配权重，聚合V信息。
 *   - 算力浪费根源：标准自注意力强制对全部两两概念对执行 Q-K 匹配，
 *     大量 token-pair 概念边界本身互相隔离，本来就几乎不会产生有效关系，
 *     却依然完整计算 QK^T。
 *   - 优化目标：不破坏"识别关系强弱、绑定实体"这个理解能力，
 *     砍掉大量无效概念对的匹配计算。
 *
 * 四层设计：
 *   Layer 1: 基于概念边界，区分「需要精细匹配」和「可以间接通信」的概念集
 *            - 片段内部：完整多头注意力
 *            - 片段之间：通过 segment-messenger 间接通信
 *   Layer 2: 关系强度门控，过滤本就边界隔离的概念对
 *            - 距离先验 + 粗粒度相似度快速预判
 *            - 软门控（保留极小概率回退通路，避免切断长距离指代）
 *   Layer 3: 多头的差异化算力分配（异构多头）
 *            - 局部语法头：只在局部窗口做匹配
 *            - 指代/因果头：可以访问窗口 + 信使
 *   Layer 4: 推理侧约束：区分训练阶段和推理 KV-Cache 的概念复用
 *            - 历史概念的 K/V 已编码完成，直接复用
 *            - 信使 token 也进入 KV-cache，保证间接长距离通路复用
 *
 * 与现有方案的本质区别：
 *   - Mistral 滑动窗口：单纯位置窗口，没有信使；长距离依赖能力弱。
 *   - Longformer global-token：全局固定几个 token 读取全部位置；
 *     不是每个片段动态聚合本片段语义状态的信使。
 *   - Performer 线性注意力：纯数学核近似，不关心"概念边界、关系强弱"。
 *
 * 本优化只改造理解阶段(Attention)的信息交互通路，不改动 FFN 推理演化逻辑。
 * 只要真实关系的匹配通路保留，上层推理能力就不会被破坏。
 *
 * Build: 与 lal_runtime.c 一起编译（#include "lal_concept_attn.h"）
 */
#ifndef LAL_CONCEPT_ATTN_H
#define LAL_CONCEPT_ATTN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Concept-Aware Attention Configuration
 * ======================================================================== */

/* 头的访问域类型（Layer 3: 异构多头） */
typedef enum {
    HEAD_LOCAL     = 0,  /* 局部语法关系（主谓宾、修饰）：强局部性，小窗口 */
    HEAD_MESSENGER = 1,  /* 指代、实体绑定、因果时序：窗口 + 信使 */
    HEAD_GLOBAL    = 2,  /* 全局兜底（少数头，访问全部历史 + 信使） */
} HeadAccessType;

/* 概念感知注意力配置 */
typedef struct {
    /* === Layer 1: 片段 + 信使 === */
    int   segment_len;          /* 语义片段长度 L（token 数）。0 = 禁用片段化 */
    int   num_messengers;       /* 每个片段的信使数目 S (S << L)。默认 4 */
    int   messenger_neighbors;  /* 邻近片段信使数（普通 token 可见的邻居片段数）。默认 2 */

    /* === Layer 2: 关系强度门控 === */
    int   gate_enable;          /* 1 = 启用关系预筛选门控，0 = 禁用 */
    int   gate_window;           /* 门控生效的局部窗口大小（窗口内才做门控） */
    float gate_threshold;        /* 概念边界相似度阈值，低于此值判定为边界隔离 */
    float gate_fallback_prob;    /* 回退概率（极小，避免硬切断长距离指代）。默认 0.01 */
    int   gate_distance_prior;   /* 距离先验权重（>0 启用）。距离越远，门控越严 */

    /* === Layer 3: 异构多头 === */
    int   hetero_enable;         /* 1 = 启用异构多头，0 = 所有头同等访问 */
    int   n_local_heads;         /* 局部语法头数量 */
    int   n_messenger_heads;     /* 指代/因果头数量（访问窗口+信使） */
    /* 剩余头 = HEAD_GLOBAL */

    /* === Layer 4: 推理 KV-Cache === */
    int   cache_messenger;       /* 1 = 信使也进入 KV-cache（保证间接长距离通路复用） */

    /* === 主开关 === */
    int   enable;                /* 1 = 启用概念感知注意力，0 = 回退到标准 attention_forward */
} ConceptAttnConfig;

/* 默认配置（唯一路线底座：CORE/BINARY/PRUNE + 浮点 + 概念注意力）
 *
 * 这是项目唯一保留的注意力路线——不再是"可选开关"。
 * 四层优化（片段信使 / 关系门控 / 异构多头 / KV 复用）默认全开，
 * 并按审查意见固化：去中心化门控、去中心化信使、窗口对称(128)、自适应阈值。
 * 环境变量 LAL_CONCEPT_ATTN=0 仅作为调试逃生口（强制关，回退标准 attention）。 */
static inline ConceptAttnConfig concept_attn_default_config(void) {
    ConceptAttnConfig c;
    c.segment_len         = 64;    /* 64 token 一个片段 */
    c.num_messengers      = 4;     /* 每片段 4 个信使 */
    c.messenger_neighbors = 2;     /* 看 2 个邻近片段的信使 */
    c.gate_enable        = 1;
    c.gate_window         = 128;   /* 窗口对称：概念路径与标准路径一致 (审查 Bug Fix 3: 32→128) */
    /* 阈值语义：固定 0.1 在塌缩表征下失效（审查 Bug 2）。
     * 运行时用门控分数分布的 P25 分位数自适应覆盖（概念注意力前向内维护 g_ca_quantile）。
     * 此处的 0.1 仅作初始/兜底值，不再作为稳定工作点。 */
    c.gate_threshold      = 0.1f;
    c.gate_fallback_prob  = 0.01f; /* 1% 回退概率，避免硬切断长距离指代 */
    c.gate_distance_prior = 1;    /* 启用距离先验（去中心化后作为相对偏差项） */
    c.hetero_enable      = 1;     /* 异构多头：局部/信使/全局分层算力 */
    c.n_local_heads       = -1;    /* -1 = 自动：n_head 的一半 */
    c.n_messenger_heads   = -1;    /* -1 = 自动：n_head 的 1/4 */
    c.cache_messenger     = 1;     /* 信使进 KV-cache，保证间接长距离通路复用 */
    c.enable              = 1;    /* 默认开启——概念注意力是基座路线，非可选 */
    return c;
}

/* ========================================================================
 * Messenger Cache (Layer 1 + Layer 4)
 * ========================================================================
 * 每层一个 MessengerCache，存储已聚合的片段信使 K/V。
 * 信使是本片段全部概念与关系状态的压缩载体。
 *
 * 内存布局：
 *   messenger_k: [segment_capacity * num_messengers * n_embd] floats
 *   messenger_v: 同上
 *   segment_filled[segment_capacity]: 该片段的信使是否已生成
 */
typedef struct {
    float     *messenger_k;       /* [segment_capacity * num_messengers * n_embd] */
    float     *messenger_v;       /* 同上 */
    uint8_t   *segment_filled;    /* [segment_capacity] 该片段信使是否已生成 */
    int        segment_capacity;  /* 最大片段数（= n_ctx / segment_len + 1） */
    int        num_messengers;    /* 每片段信使数 */
    int        n_embd;            /* 嵌入维度 */
    int        n_filled;          /* 已生成的片段数 */
} MessengerCache;

/* 分配信使缓存。在 model_load 后调用。 */
void messenger_cache_alloc(MessengerCache *mc, int segment_capacity,
                           int num_messengers, int n_embd);

/* 释放信使缓存 */
void messenger_cache_free(MessengerCache *mc);

/* 重置信使缓存（推理新会话开始时调用） */
void messenger_cache_reset(MessengerCache *mc);

/* ========================================================================
 * Layer 1: Segment Messenger Generation
 * ========================================================================
 * 在每个 segment 内部，基于本片段全部 V，聚合生成少量信使向量。
 * 信使是本片段全部概念与关系状态的压缩载体。
 *
 * 聚合策略：对片段内 V 做均匀分桶 + 均值池化
 *   - 将片段内 V[0..seg_len-1] 均匀分成 num_messengers 个桶
 *   - 每个桶内做均值池化，得到一个信使向量
 *   - 信使的 K = 信使的 V（自关联，简化）
 *
 * 数学复杂度：O(seg_len * n_embd)，远小于注意力本身
 *
 * 语义意义：远方片段的整体语义，由信使代为表达。
 *          普通token通过信使间接获得远方概念集合的状态，
 *          而不是挨个访问每一个远方概念。
 *
 * 参数：
 *   v_seg:     [seg_len * n_embd] — 本片段的 V 缓存
 *   seg_len:   片段长度
 *   out_k:     [num_messengers * n_embd] — 输出信使 K
 *   out_v:     [num_messengers * n_embd] — 输出信使 V
 */
void generate_segment_messengers(const float *v_seg, int seg_len, int n_embd,
                                 int num_messengers,
                                 float *out_k, float *out_v);

/* ========================================================================
 * Layer 2: Concept Boundary Gate (关系强度门控)
 * ========================================================================
 * 给定 token-i（Q侧）、token-j（K侧），利用距离先验 + 粗粒度相似度
 * 快速预判：如果预判两个概念边界隔离，潜在关系极弱，
 * 直接把该位置置 -inf，不参与完整内积计算。
 *
 * 注意：不是简单按位置距离硬截断；是"概念边界是否可能产生关系"的软判断。
 * 避免错误切断长距离指代这种真实强关系。
 *
 * 门控公式（软门控，保留回退通路）：
 *   sim_coarse = <Q_i, K_j> / (||Q_i|| * ||K_j|| + eps)   // 粗粒度余弦相似度
 *   dist_prior = exp(-distance / tau)                     // 距离先验
 *   gate_score = sim_coarse * (1 + gate_distance_prior * dist_prior)
 *   if gate_score < gate_threshold:
 *       以 (1 - gate_fallback_prob) 概率置 -inf
 *       以 gate_fallback_prob 概率保留（回退通路）
 *
 * 参数：
 *   q_i:       [head_dim] — 当前 token 的 Q（某头）
 *   k_j:       [head_dim] — 候选 token 的 K（某头）
 *   distance:  i 和 j 的位置距离
 *   cfg:       概念感知注意力配置
 * 返回：1 = 保留（参与完整 QK 计算），0 = 屏蔽（置 -inf）
 */
int concept_boundary_gate(const float *q_i, const float *k_j,
                          int head_dim, int distance,
                          const ConceptAttnConfig *cfg);

/* ========================================================================
 * Layer 3: Heterogeneous Head Access Configuration
 * ========================================================================
 * 不同类型关系本身就有不同的"概念交互范围"，不需要统一全序列扫描。
 *   - 头A：局部语法关系（主谓宾、修饰）：强局部性，适合小窗口。
 *   - 头B：指代、实体绑定：偶尔需要长距离跳跃。
 *   - 头C：因果、时序关系：中等范围依赖。
 *
 * 配置：根据 n_head 自动分配头的访问域
 *   - 前 n_local_heads 个头 = HEAD_LOCAL
 *   - 接下来 n_messenger_heads 个头 = HEAD_MESSENGER
 *   - 剩余 = HEAD_GLOBAL
 */
HeadAccessType get_head_access_type(int head_idx, int n_head,
                                   const ConceptAttnConfig *cfg);

/* 获取某头的实际访问窗口大小 */
int get_head_window(int head_idx, int n_head, int base_window,
                    const ConceptAttnConfig *cfg);

/* 某头是否可以访问信使 */
int head_can_access_messenger(int head_idx, int n_head,
                              const ConceptAttnConfig *cfg);

/* ========================================================================
 * Layer 4: Concept-Aware Attention Forward (主入口)
 * ========================================================================
 * 概念感知注意力前向传播。整合四层优化：
 *
 *   1. 切成语义片段（segment_len）
 *   2. 片段内部：完整 QKV，充分做片段内概念理解
 *   3. 生成本片段信使：聚合本片段全部概念-关系状态
 *   4. 本片段普通 token：只和【局部窗口 + 本片段信使 + 邻近片段信使】做匹配
 *   5. 关系门控：过滤边界隔离的概念对（Layer 2）
 *   6. 异构多头：不同头不同访问域（Layer 3）
 *   7. KV-Cache：历史 K/V 直接复用，信使也进 cache（Layer 4）
 *
 * 参数：
 *   attn_out:   [n_embd]      — 输出
 *   qkv:        [3 * n_embd]  — Q | K | V 拼接，单 token
 *   n_embd, n_head:           — 维度与头数
 *   seq_pos:                   — 当前序列位置
 *   k_cache, v_cache:         — [n_ctx * n_embd] KV 缓存
 *   n_ctx:                     — 上下文长度
 *   cfg:                       — 概念感知注意力配置
 *   mc:                        — 信使缓存（可为 NULL，则不使用信使）
 *
 * 因果性：seq_pos 只关注 0..seq_pos（含）。
 * 多头：head_dim = n_embd / n_head（必须整除）。
 */
void attention_forward_concept(float *attn_out, const float *qkv,
                               int n_embd, int n_head, int seq_pos,
                               float *k_cache, float *v_cache,
                               int n_ctx,
                               const ConceptAttnConfig *cfg,
                               MessengerCache *mc);

/* ========================================================================
 * Layer 4: Concept-Aware Attention Backward
 * ========================================================================
 * 概念感知注意力反向传播。计算当前 token 的 Q/K/V 梯度。
 * 缓存的 K/V（位置 0..seq_pos-1）视为常量（与 attention_backward 一致）。
 *
 * 参数：
 *   grad_qkv:       [3 * n_embd] — 输出梯度（Q|K|V）
 *   grad_attn_out:   [n_embd]    — 来自上层的 attn_out 梯度
 *   qkv:            [3 * n_embd] — 前向时用的 Q|K|V
 *   n_embd, n_head, seq_pos
 *   k_cache, v_cache, n_ctx
 *   cfg, mc
 */
void attention_backward_concept(float *grad_qkv, const float *grad_attn_out,
                                const float *qkv, int n_embd, int n_head,
                                int seq_pos,
                                const float *k_cache, const float *v_cache,
                                int n_ctx,
                                const ConceptAttnConfig *cfg,
                                MessengerCache *mc);

/* ========================================================================
 * Transformer Layer Forward Integration
 * ========================================================================
 * trans_layer_forward 的实现包含比例缩放、残差归一化等复杂逻辑，
 * 完整复制易引入 bug。推荐集成方式：
 *
 *   在现有 trans_layer_forward() 中，将 attention_forward 调用替换为：
 *     if (g_concept_attn_cfg.enable && g_messenger_caches) {
 *         attention_forward_concept(act->attn_out, qkv_ptr,
 *                                    n, cfg->n_head, abs_pos,
 *                                    tl->kv_k, tl->kv_v, cfg->n_ctx,
 *                                    &g_concept_attn_cfg,
 *                                    &g_messenger_caches[layer_idx]);
 *     } else {
 *         attention_forward(act->attn_out, qkv_ptr, n, cfg->n_head,
 *                            abs_pos, tl->kv_k, tl->kv_v);
 *     }
 *
 *   反向传播同理：将 attention_backward 替换为 attention_backward_concept。
 *   通过 model_set_concept_attn() 在运行时配置，无需修改模型结构。
 */

/* ========================================================================
 * Global Config (便于全局开关，无需改 ModelConfig)
 * ========================================================================
 * 全局概念感知注意力配置。设为 enable=1 即启用。
 * 优先级：ModelConfig 中的 concept_cfg > 全局 g_concept_attn_cfg。
 */
extern ConceptAttnConfig g_concept_attn_cfg;

/* 全局信使缓存（每层一个，按 layer_idx 索引）。
 * 在 model_load 时分配，model_free 时释放。 */
extern MessengerCache *g_messenger_caches;  /* [n_layer] */

/* Model-dependent functions (declared in lal_runtime.h after Model is defined):
 *   void model_messenger_caches_alloc(Model *m, const ConceptAttnConfig *cfg);
 *   void model_messenger_caches_free(void);
 *   void model_messenger_caches_reset(void);
 *   void model_set_concept_attn(Model *m, const ConceptAttnConfig *cfg);
 */

#ifdef __cplusplus
}
#endif

#endif /* LAL_CONCEPT_ATTN_H */
