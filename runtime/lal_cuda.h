#include <cuda_runtime.h>
#ifndef LAL_CUDA_H
#define LAL_CUDA_H
#ifdef __cplusplus
extern "C" {
#endif

#include "lal_runtime.h"

/* Global GPU switch. When set, bin_forward() / bin_backward_ste() in
 * lal_runtime.c dispatch to the CUDA backend instead of the scalar CPU path.
 * Enable with: lal_cuda_set_enabled(1);  (or `--cuda` in models/gpt2.c) */
void lal_cuda_set_enabled(int on);
int  lal_cuda_enabled(void);

/* === Binary (BWN) forward on GPU ===
 *   y[j] = alpha[j] * K * sum_i sign(w_float[j*in+i]) * x[i] + bias[j]
 *   K = mean(|x|)  (XNOR-Net K-norm, computed on host)
 * Computed directly from w_float (sign), so it does NOT depend on the packed
 * wbits — this keeps the GPU training path fully consistent with STE. */
void lal_cuda_bin_forward(float *y, const float *x, const BinLayer *bl);

/* === STE backward on GPU ===
 *   grad_x[i] = sum_j grad_y[j] * alpha[j] * sign(w_float[j*in+i])
 *   and updates w_float / bias in place:
 *     SGD : w_float[j*in+i] -= lr * grad_y[j] * x[i]
 *     Adam: per-param adaptive update using bl->m_adam / bl->v_adam (g_opt_step)
 * After the update, alpha[j] is recomputed from updated |w_float| (host side).
 * wbits are intentionally NOT repacked here (forward uses sign(w_float)); call
 * lal_cuda_repack_wbits() before exporting/running binary inference. */
void lal_cuda_bin_backward_ste(float *grad_x, const float *grad_y,
                               const float *x, BinLayer *bl, float lr);

/* Recompute wbits (sign pack) and alpha from w_float on the host.
 * Call after GPU STE training if you want to export or run binary inference. */
void lal_cuda_repack_wbits(BinLayer *bl);

/* v13j: Logic-guided pure-float forward/backward on GPU. */
void lal_cuda_bin_forward_pure_float_logic(float *y, const float *x, const BinLayer *bl);
void lal_cuda_bin_backward_pure_float_logic(float *grad_x, const float *grad_y, const float *x, BinLayer *bl, float lr);

/* === Pure-float (fp32) matmul on GPU ===
 * Used by the standalone GPU demo (tools/gpu_train.cu) and can back the
 * pure-float (teacher) path. W is row-major [out, in]. */
void lal_cuda_matmul_f32(float *y, const float *x, const float *W,
                         const float *b, int in, int out);
/* grad_W is returned already scaled by lr (caller does W -= grad_W);
 * grad_x = W^T @ grad_y. */
void lal_cuda_matmul_f32_backward(float *grad_W, float *grad_x,
                                  const float *grad_y, const float *x,
                                  const float *W, int in, int out, float lr);

/* v13k: Resident weight API — upload weights once, reuse across forward calls.
 * Eliminates per-call cudaMalloc/Memcpy/Free overhead (was 13x kernel time). */
typedef struct {
    float   *d_w;       /* [out * in] weights, resident */
    float   *d_bias;    /* [out] bias, resident */
    uint8_t *d_mask;    /* [out] logic_mask, resident (NULL if no mask) */
    int      uploaded;  /* 1 after upload, 0 after free */
} LayerGPU;

int  lal_cuda_upload_layer(BinLayer *bl);   /* alloc + copy weights to GPU */
void lal_cuda_sync_layer(BinLayer *bl);     /* push updated w_float to GPU */
void lal_cuda_free_layer(BinLayer *bl);     /* free device buffers */

/* Forward/backward using resident weights + cuBLAS. Only x/grad_y in, y/grad_x out. */
void lal_cuda_fwd_resident(float *y, const float *x, const BinLayer *bl);
void lal_cuda_bwd_resident(float *grad_x, const float *grad_y, const BinLayer *bl);

/* v13l: Batch forward — multiple inputs in one cuBLAS call.
 * Replaces 14× cublasSgemv with 1× cublasSgemm for logic_reg.
 * y[batch, out] = X[batch, in] @ W[out, in]^T + bias[out] */
void lal_cuda_fwd_batch(float *y, const float *x, const BinLayer *bl, int batch);

/* v13m: Fused batch layer forward — ALL intermediates on GPU, CUDA stream async.
 * Does full transformer layer (norm+attn+residual+norm+mlp+residual) for [batch]
 * inputs. Only transfers: norm2 D2H (for gate_input cache). Everything else
 * stays on GPU. cuBLAS calls queued on stream (no host sync between calls).
 * d_x is modified in-place (residual stream). */
/* ModelConfig and TransLayer are already defined in lal_runtime.h (included above) */
void lal_cuda_layer_forward_batch(
    float *d_x,           /* [batch, n_embd] device — residual stream */
    float *h_norm2_out,   /* [batch, n_embd] host — norm2 for gate_input */
    TransLayer *tl, ModelConfig *cfg,
    int batch, void *stream);

/* v13m: High-level — compute gate_inputs for all layers, all batch, fully on GPU.
 * Uploads embeddings, runs fused layer forward, returns norm2 per layer.
 * This is the main entry point for logic_reg batch forward. */
void lal_cuda_compute_gate_inputs_batch(
    Model *m, const float *embs, int batch,
    float *out_gate_inputs, int n_embd);

/* v13p/q: Full GPU forward+backward */
float lal_cuda_logic_reg(Model *m, const float *gate_a, const float *gate_b, float lr);
float lal_cuda_full_forward(Model *m, const int *tokens, int seq_len,
                            int target, float *grad_hidden, int *predicted);
void lal_cuda_full_backward(Model *m, const int *tokens, int seq_len, int target);

#ifdef __cplusplus
}
#endif
#endif /* LAL_CUDA_H */
