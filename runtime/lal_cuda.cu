/* lal_cuda.cu — CUDA backend for LAL GPU training.
 *
 * Provides the GPU implementations of the binary (BWN) forward / STE backward
 * matmul used by lal_runtime.c, plus generic fp32 matmul operators used by the
 * standalone demo (tools/gpu_train.cu).
 *
 * Build: nvcc -O3 -DLAL_CUDA -I. -c runtime/lal_cuda.cu -o lal_cuda.o
 * Links against -lcudart. Requires an NVIDIA GPU + CUDA driver.
 *
 * NOTE: for correctness/simplicity each call does its own cudaMalloc/copy.
 * A production build should keep w_float/alpha/bias resident on the device and
 * only sync at checkpoint/export time. The scalar loops here are intentionally
 * simple (one row or one column per thread) — they are correct and portable;
 * vectorized/tiled kernels are a future optimization.
 */
#include "lal_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

/* Mirror training globals defined in lal_runtime.c (C linkage). */
extern "C" {
extern int   g_opt_step;
extern float g_adam_beta1;
extern float g_adam_beta2;
extern float g_adam_eps;
extern int   g_use_adam;
}

static int g_cuda_enabled = 0;
void lal_cuda_set_enabled(int on) { g_cuda_enabled = on; }
int  lal_cuda_enabled(void)       { return g_cuda_enabled; }

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess)                                             \
            fprintf(stderr, "[CUDA] %s:%d %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
    } while (0)

/* ===================== Binary (BWN) forward ===================== */
__global__ void k_bin_fwd(const float *x, const float *wf, const float *alpha,
                          const float *bias, float K, int in, int out, float *y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float s = 0.0f;
    const float *wr = wf + (size_t)j * in;
    for (int i = 0; i < in; i++)
        s += (wr[i] > 0.0f ? 1.0f : -1.0f) * x[i];
    y[j] = s * alpha[j] * K + bias[j];
}

void lal_cuda_bin_forward(float *y, const float *x, const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim;
    float K = 0.0f;
    for (int i = 0; i < in; i++) K += fabsf(x[i]);
    K /= (in > 0 ? in : 1);

    float *d_x, *d_wf, *d_a, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x,  in        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wf, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a,  out       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,  out       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  out       * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wf, bl->w_float, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, bl->alpha, out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bl->bias,  out * sizeof(float), cudaMemcpyHostToDevice));

    int thr = 256, blk = (out + thr - 1) / thr;
    k_bin_fwd<<<blk, thr>>>(d_x, d_wf, d_a, d_b, K, in, out, d_y);
    CUDA_CHECK(cudaMemcpy(y, d_y, out * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x); cudaFree(d_wf); cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

/* ===================== STE backward ===================== */
/* grad_x[i] = sum_j grad_y[j] * alpha[j] * sign(w_float[j*in+i]) */
__global__ void k_ste_gradx(const float *wf, const float *gy, const float *alpha,
                            int in, int out, float *grad_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in) return;
    float s = 0.0f;
    for (int j = 0; j < out; j++) {
        float w = wf[(size_t)j * in + i];
        s += gy[j] * alpha[j] * (w > 0.0f ? 1.0f : -1.0f);
    }
    grad_x[i] = s;
}

/* SGD weight update: wf[j*in+i] -= lr * gy[j] * x[i] */
__global__ void k_ste_update_sgd(float *wf, const float *gy, const float *x,
                                  int in, int out, float lr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float g = gy[j];
    if (fabsf(g) < 1e-8f) return;
    float scale = lr * g;
    float *wr = wf + (size_t)j * in;
    for (int i = 0; i < in; i++) wr[i] -= scale * x[i];
}

/* Adam weight update (mirrors bin_backward_ste in lal_runtime.c). */
__global__ void k_ste_update_adam(float *wf, float *m, float *v, const float *gy,
                                   const float *x, int in, int out, float lr,
                                   float b1, float b2, float eps, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float g = gy[j];
    if (fabsf(g) < 1e-8f) return;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    float *wr = wf + (size_t)j * in;
    float *mr = m  + (size_t)j * in;
    float *vr = v  + (size_t)j * in;
    for (int i = 0; i < in; i++) {
        float gi = g * x[i];
        mr[i] = b1 * mr[i] + (1.0f - b1) * gi;
        vr[i] = b2 * vr[i] + (1.0f - b2) * gi * gi;
        wr[i] -= lr * (mr[i] / bc1) / (sqrtf(vr[i] / bc2) + eps);
    }
}

void lal_cuda_bin_backward_ste(float *grad_x, const float *grad_y,
                               const float *x, BinLayer *bl, float lr) {
    int in = bl->in_dim, out = bl->out_dim;
    float *d_x, *d_wf, *d_gy, *d_gx, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_x,  in        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wf, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, out       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, in        * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x,  x,  in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wf, bl->w_float, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, grad_y, out * sizeof(float), cudaMemcpyHostToDevice));

    int thr = 256;
    int blk_gx = (in  + thr - 1) / thr;
    int blk_up = (out + thr - 1) / thr;
    k_ste_gradx<<<blk_gx, thr>>>(d_wf, d_gy, bl->alpha, in, out, d_gx);
    CUDA_CHECK(cudaMemcpy(grad_x, d_gx, in * sizeof(float), cudaMemcpyDeviceToHost));

    if (g_use_adam && bl->m_adam) {
        CUDA_CHECK(cudaMalloc(&d_m, (size_t)in * out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, (size_t)in * out * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_m, bl->m_adam, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, bl->v_adam, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
        int t = g_opt_step + 1;
        k_ste_update_adam<<<blk_up, thr>>>(d_wf, d_m, d_v, d_gy, d_x, in, out, lr,
                                           g_adam_beta1, g_adam_beta2, g_adam_eps, t);
        CUDA_CHECK(cudaMemcpy(bl->m_adam, d_m, (size_t)in * out * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bl->v_adam, d_v, (size_t)in * out * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_m); cudaFree(d_v);
    } else {
        k_ste_update_sgd<<<blk_up, thr>>>(d_wf, d_gy, d_x, in, out, lr);
    }

    /* bias update (SGD always — marginal Adam benefit for scalars). */
    for (int j = 0; j < out; j++) bl->bias[j] -= lr * grad_y[j];

    CUDA_CHECK(cudaMemcpy(bl->w_float, d_wf, (size_t)in * out * sizeof(float), cudaMemcpyDeviceToHost));

    /* Recompute alpha[j] = mean(|w_float[j]|) on host (forward uses sign(w_float)). */
    for (int j = 0; j < out; j++) {
        float s = 0.0f;
        const float *wr = bl->w_float + (size_t)j * in;
        for (int i = 0; i < in; i++) s += fabsf(wr[i]);
        bl->alpha[j] = s / (in > 0 ? in : 1);
    }
    cudaFree(d_x); cudaFree(d_wf); cudaFree(d_gy); cudaFree(d_gx);
}

/* Recompute wbits (sign pack) + alpha from w_float on the host. */
void lal_cuda_repack_wbits(BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim, nw = bl->n_words;
    for (int j = 0; j < out; j++) {
        const float *wr = bl->w_float + (size_t)j * in;
        float s = 0.0f;
        for (int i = 0; i < in; i++) s += fabsf(wr[i]);
        bl->alpha[j] = s / (in > 0 ? in : 1);
        for (int wi = 0; wi < nw; wi++) {
            uint64_t word = 0;
            for (int bi = 0; bi < 64; bi++) {
                int idx = wi * 64 + bi;
                if (idx < in && wr[idx] > 0.0f) word |= (1ULL << bi);
            }
            bl->wbits[j * nw + wi] = word;
        }
    }
}

/* ===================== Pure-float (fp32) matmul ===================== */
__global__ void k_mm_fwd(const float *x, const float *W, const float *b,
                         int in, int out, float *y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float s = (b ? b[j] : 0.0f);
    const float *wr = W + (size_t)j * in;
    for (int i = 0; i < in; i++) s += wr[i] * x[i];
    y[j] = s;
}
__global__ void k_mm_bwd_w(const float *x, const float *gy, int in, int out, float *dW) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float g = gy[j];
    float *wr = dW + (size_t)j * in;
    for (int i = 0; i < in; i++) wr[i] = g * x[i];   /* grad_W[j,i] = gy[j]*x[i] */
}
__global__ void k_mm_bwd_x(const float *W, const float *gy, int in, int out, float *gx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in) return;
    float s = 0.0f;
    for (int j = 0; j < out; j++) s += gy[j] * W[(size_t)j * in + i];
    gx[i] = s;
}

void lal_cuda_matmul_f32(float *y, const float *x, const float *W, const float *b,
                         int in, int out) {
    float *d_x, *d_W, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, out * sizeof(float), cudaMemcpyHostToDevice));
    int thr = 256, blk = (out + thr - 1) / thr;
    k_mm_fwd<<<blk, thr>>>(d_x, d_W, d_b, in, out, d_y);
    CUDA_CHECK(cudaMemcpy(y, d_y, out * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_y);
}

void lal_cuda_matmul_f32_backward(float *grad_W, float *grad_x, const float *grad_y,
                                  const float *x, const float *W, int in, int out, float lr) {
    float *d_x, *d_W, *d_gy, *d_gW, *d_gx;
    CUDA_CHECK(cudaMalloc(&d_x, in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gW, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, in * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, grad_y, out * sizeof(float), cudaMemcpyHostToDevice));
    int thr = 256;
    k_mm_bwd_w<<<(out + thr - 1) / thr, thr>>>(d_x, d_gy, in, out, d_gW);
    k_mm_bwd_x<<<(in  + thr - 1) / thr, thr>>>(d_W, d_gy, in, out, d_gx);
    CUDA_CHECK(cudaMemcpy(grad_W, d_gW, (size_t)in * out * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_x, d_gx, in * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < in * out; k++) grad_W[k] *= lr;   /* caller does W -= grad_W */
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_gy); cudaFree(d_gW); cudaFree(d_gx);
}

/* ===================== Logic-guided pure-float forward =====================
 * v13j: CUDA acceleration for bin_forward_pure_float with logic_mask.
 *
 * In pure_float mode, CORE and BINARY both use w_float for matmul
 * (w_core is a synced copy of w_float for CORE rows).
 * PRUNE rows output 0.
 *
 * Replaces the per-output CPU loop that was the main bottleneck
 * (mlp_gate: 1792 outputs x 512 inputs = 917K FMAs per layer).
 */
__global__ void k_logic_pure_fwd(const float *x, const float *wf,
                                  const float *bias, const uint8_t *mask,
                                  int in, int out, float *y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    if (mask && mask[j] == 2) {  /* PRUNE */
        y[j] = 0.0f;
        return;
    }
    const float *wr = wf + (size_t)j * in;
    float s = bias ? bias[j] : 0.0f;
    for (int i = 0; i < in; i++)
        s += wr[i] * x[i];
    y[j] = s;
}

extern "C"
void lal_cuda_bin_forward_pure_float_logic(float *y, const float *x,
                                            const BinLayer *bl) {
    int in = bl->in_dim, out = bl->out_dim;
    float *d_x, *d_wf, *d_b, *d_y;
    uint8_t *d_mask;
    CUDA_CHECK(cudaMalloc(&d_x,  in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wf, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,  out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, out * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wf, bl->w_float, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    if (bl->bias)
        CUDA_CHECK(cudaMemcpy(d_b, bl->bias, out * sizeof(float), cudaMemcpyHostToDevice));
    else
        cudaMemset(d_b, 0, out * sizeof(float));
    if (bl->logic_mask)
        CUDA_CHECK(cudaMemcpy(d_mask, bl->logic_mask, out * sizeof(uint8_t), cudaMemcpyHostToDevice));
    else
        cudaMemset(d_mask, 0, out * sizeof(uint8_t));

    int thr = 256, blk = (out + thr - 1) / thr;
    k_logic_pure_fwd<<<blk, thr>>>(d_x, d_wf, d_b, d_mask, in, out, d_y);
    CUDA_CHECK(cudaMemcpy(y, d_y, out * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x); cudaFree(d_wf); cudaFree(d_b); cudaFree(d_y); cudaFree(d_mask);
}

/* ===================== Logic-guided grad_x (W^T @ gy) ==================== */
__global__ void k_logic_grad_x(const float *wf, const float *gy,
                                const uint8_t *mask, int in, int out, float *gx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in) return;
    float s = 0.0f;
    for (int j = 0; j < out; j++) {
        if (mask && mask[j] == 2) continue;
        s += gy[j] * wf[(size_t)j * in + i];
    }
    gx[i] = s;
}

extern "C"
void lal_cuda_bin_backward_pure_float_logic(float *grad_x, const float *grad_y,
                                             const float *x, BinLayer *bl, float lr) {
    int in = bl->in_dim, out = bl->out_dim;
    float *d_wf, *d_gy, *d_gx;
    uint8_t *d_mask;
    CUDA_CHECK(cudaMalloc(&d_wf, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, out * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_wf, bl->w_float, (size_t)in * out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, grad_y, out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, bl->logic_mask, out * sizeof(uint8_t), cudaMemcpyHostToDevice));

    int thr = 256, blk = (in + thr - 1) / thr;
    k_logic_grad_x<<<blk, thr>>>(d_wf, d_gy, d_mask, in, out, d_gx);
    CUDA_CHECK(cudaMemcpy(grad_x, d_gx, in * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_wf); cudaFree(d_gy); cudaFree(d_gx); cudaFree(d_mask);
}
