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

#ifdef __cplusplus
}
#endif
#endif /* LAL_CUDA_H */
