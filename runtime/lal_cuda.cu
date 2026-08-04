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

/* ===================== v13k: Resident weight API + cuBLAS =====================
 *
 * Old v13j approach: cudaMalloc + cudaMemcpy + kernel + cudaFree per call.
 *   For 512x1792 matrix: 13 GPU calls × ~50us = 650us overhead, but kernel
 *   only ~50us. Overhead dominates 13x → GPU slower than CPU.
 *
 * v13k: Upload w_float/bias/mask once (upload_layer). Forward only copies
 *   x in (2KB) and y out (7KB). With cuBLAS sgemm (T4 Tensor Core), this
 *   should give 5-10x speedup over CPU.
 *
 * Weight sync: w_float updated on CPU during Adam. Call sync_layer() after
 * model_batch_apply to push updated weights to GPU (one cudaMemcpy).
 */
#include <cublas_v2.h>

static cublasHandle_t g_cublas = NULL;
static int g_cublas_init = 0;

static cublasHandle_t get_cublas() {
    if (!g_cublas_init) {
        cublasCreate(&g_cublas);
        g_cublas_init = 1;
    }
    return g_cublas;
}

/* Upload w_float, bias, logic_mask to GPU. Called once after model_load. */
extern "C"
int lal_cuda_upload_layer(BinLayer *bl) {
    if (!bl->w_float) return -1;
    int in = bl->in_dim, out = bl->out_dim;
    LayerGPU *g = (LayerGPU*)calloc(1, sizeof(LayerGPU));

    CUDA_CHECK(cudaMalloc(&g->d_w, (size_t)in * out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g->d_w, bl->w_float, (size_t)in * out * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (bl->bias) {
        CUDA_CHECK(cudaMalloc(&g->d_bias, out * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(g->d_bias, bl->bias, out * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (bl->logic_mask) {
        CUDA_CHECK(cudaMalloc(&g->d_mask, out * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(g->d_mask, bl->logic_mask, out * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    }
    g->uploaded = 1;
    bl->_gpu = g;
    /* v13s: allocate device grad_accum */
    cudaMalloc(&bl->d_grad_accum, (size_t)in * out * sizeof(float));
    cudaMalloc(&bl->d_bias_grad_accum, out * sizeof(float));
    cudaMemset(bl->d_grad_accum, 0, (size_t)in * out * sizeof(float));
    cudaMemset(bl->d_bias_grad_accum, 0, out * sizeof(float));
    return 0;
}

/* Sync updated w_float from host to device (after Adam update). */
extern "C"
void lal_cuda_sync_layer(BinLayer *bl) {
    LayerGPU *g = (LayerGPU*)bl->_gpu;
    if (!g || !g->uploaded) return;
    int in = bl->in_dim, out = bl->out_dim;
    CUDA_CHECK(cudaMemcpy(g->d_w, bl->w_float,
                          (size_t)in * out * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (bl->bias && g->d_bias)
        CUDA_CHECK(cudaMemcpy(g->d_bias, bl->bias, out * sizeof(float),
                              cudaMemcpyHostToDevice));
}

/* Free device buffers. Called from bin_layer_free. */
extern "C"
void lal_cuda_free_layer(BinLayer *bl) {
    LayerGPU *g = (LayerGPU*)bl->_gpu;
    if (!g) return;
    if (g->d_w) cudaFree(g->d_w);
    if (g->d_bias) cudaFree(g->d_bias);
    if (g->d_mask) cudaFree(g->d_mask);
    free(g);
    bl->_gpu = NULL;
    /* v13s: free device grad_accum */
    if (bl->d_grad_accum) { cudaFree(bl->d_grad_accum); bl->d_grad_accum = NULL; }
    if (bl->d_bias_grad_accum) { cudaFree(bl->d_bias_grad_accum); bl->d_bias_grad_accum = NULL; }
}

/* v13k: Forward using resident weights + cuBLAS.
 *   y = W @ x + bias  (W is [out, in] row-major)
 *   cuBLAS uses column-major, so W[out,in] row-major == W^T[in,out] col-major
 *   y = W^T @ x in col-major terms → cublasSgemm with op=T for W
 *
 *   For logic_mask layers: after matmul, zero out PRUNE rows via kernel.
 *   This is cheaper than masking inside matmul. */
__global__ void k_zero_prune(float *y, const uint8_t *mask, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out && mask[j] == 2) y[j] = 0.0f;
}

extern "C"
void lal_cuda_fwd_resident(float *y, const float *x, const BinLayer *bl) {
    LayerGPU *g = (LayerGPU*)bl->_gpu;
    if (!g || !g->uploaded) {
        /* Fallback to v13j per-call path */
        lal_cuda_bin_forward_pure_float_logic(y, x, bl);
        return;
    }
    int in = bl->in_dim, out = bl->out_dim;
    cublasHandle_t h = get_cublas();

    /* v13o: persistent x/y buffers. Use 2x cap for safety (cuBLAS internal
     * sgemm optimization may access slightly beyond out elements). */
    static float *d_x = NULL, *d_y = NULL;
    static int d_cap = 0;
    int need = (in > out ? in : out) * 4;  /* v13o: 4x cap for cuBLAS gemvx prefetch */
    if (need > d_cap) {
        if (d_x) { cudaFree(d_x); cudaFree(d_y); }
        cudaMalloc(&d_x, need * sizeof(float));
        cudaMalloc(&d_y, need * sizeof(float));
        d_cap = need;
    }
    /* Clear d_y to 0 first (in case bias is NULL, beta=0 still works) */
    cudaMemset(d_y, 0, out * sizeof(float));

    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));

    /* v13o: Use sgemm instead of sgemv for correctness with large out_dim.
     * sgemv with OP_T internally dispatches to sgemm kernel that had
     * out-of-bounds access for QKV merged layers (out=3*n=1536).
     * sgemm with n=1 (batch=1) is equivalent but uses different code path.
     *
     * y[out] = W[out,in] @ x[in]  (W row-major == W col-major [in,out])
     * cuBLAS col-major: Y[out,1] = W_cm[in,out] @ X[in,1]
     *   cublasSgemm(OP_N, OP_N, m=out, n=1, k=in,
     *               alpha, d_w, lda=in, d_x, ldb=in,
     *               beta, d_y, ldc=out)
     * With bias: pre-fill d_y with bias, beta=1. */
    if (g->d_bias) {
        CUDA_CHECK(cudaMemcpy(d_y, g->d_bias, out * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    float alpha = 1.0f;
    float beta = g->d_bias ? 1.0f : 0.0f;
    /* v13o: use sgemv (correct for vector case). A=[lda=in,n=out] col-major.
     * y = alpha * A^T * x + beta * y  (OP_T transposes A) */
    cublasSgemv(h, CUBLAS_OP_T, in, out, &alpha, g->d_w, in, d_x, 1,
                &beta, d_y, 1);

    /* Zero out PRUNE rows if logic_mask present */
    if (g->d_mask) {
        int thr = 256, blk = (out + thr - 1) / thr;
        k_zero_prune<<<blk, thr>>>(d_y, g->d_mask, out);
    }

    CUDA_CHECK(cudaMemcpy(y, d_y, out * sizeof(float), cudaMemcpyDeviceToHost));
}

/* v13k: Backward grad_x using resident weights + cuBLAS.
 *   grad_x = W^T @ grad_y  (W is [out, in], so W^T is [in, out])
 *   In col-major: grad_x[in] = W[out,in] @ grad_y[out]
 *   → cublasSgemv(handle, CUBLAS_OP_N, in, out, &alpha, d_w, in, d_gy, 1, &beta, d_gx, 1)
 *   Wait: W row-major [out,in] == W col-major [in,out] transposed... let me think.
 *
 *   W stored row-major: W[j][i] = w_float[j*in + i], j=0..out-1, i=0..in-1
 *   Col-major view: same memory, W_cm[i][j] = w_float[j*in + i] ... no.
 *   Col-major [in,out]: element (i,j) at i + j*in. Same as row-major [out,in]!
 *   So W row-major [out,in] == W_cm [in,out] (not transposed).
 *
 *   grad_x[i] = sum_j W[j][i] * grad_y[j] = sum_j W_cm[i][j] * grad_y[j]
 *   = W_cm @ grad_y  where W_cm is [in,out], grad_y is [out]
 *   → cublasSgemv(handle, CUBLAS_OP_N, in, out, &alpha, d_w, in, d_gy, 1, &beta, d_gx, 1) */
extern "C"
void lal_cuda_bwd_resident(float *grad_x, const float *grad_y, const BinLayer *bl) {
    LayerGPU *g = (LayerGPU*)bl->_gpu;
    if (!g || !g->uploaded) return;
    int in = bl->in_dim, out = bl->out_dim;
    cublasHandle_t h = get_cublas();

    /* v13o: 2x buffer cap + sgemm instead of sgemv (same fix as fwd) */
    static float *d_gy = NULL, *d_gx = NULL;
    static int d_cap_bwd = 0;
    int need = (in > out ? in : out) * 4;
    if (need > d_cap_bwd) {
        if (d_gy) { cudaFree(d_gy); cudaFree(d_gx); }
        cudaMalloc(&d_gy, need * sizeof(float));
        cudaMalloc(&d_gx, need * sizeof(float));
        d_cap_bwd = need;
    }

    CUDA_CHECK(cudaMemcpy(d_gy, grad_y, out * sizeof(float), cudaMemcpyHostToDevice));

    /* For PRUNE rows, zero their grad_y before matmul. */
    if (g->d_mask) {
        int thr = 256, blk = (out + thr - 1) / thr;
        k_zero_prune<<<blk, thr>>>(d_gy, g->d_mask, out);
    }

    /* v13o: Use sgemm instead of sgemv.
     * grad_x[in] = W_cm[in,out] @ grad_y[out]
     * cuBLAS: grad_x[m=in,1] = A[lda=in,n=out] @ x[ldb=out,1]
     *   cublasSgemm(OP_N, OP_N, m=in, n=1, k=out,
     *               alpha, d_w, lda=in, d_gy, ldb=out,
     *               beta, d_gx, ldc=in) */
    float alpha = 1.0f, beta = 0.0f;
    /* grad_x[in] = W_cm[in,out] @ grad_y[out], OP_N (no transpose) */
    cublasSgemv(h, CUBLAS_OP_N, in, out, &alpha, g->d_w, in, d_gy, 1,
                &beta, d_gx, 1);

    CUDA_CHECK(cudaMemcpy(grad_x, d_gx, in * sizeof(float), cudaMemcpyDeviceToHost));
}

/* Helper kernel: fill d_y[b*out..b*out+out] = bias[0..out] for each b */
__global__ void k_fill_bias(float *y, const float *bias, int batch, int out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out) return;
    int j = idx % out;
    y[idx] = bias[j];
}

/* Helper kernel: zero PRUNE rows across all batches */
__global__ void k_zero_prune_batch(float *y, const uint8_t *mask, int batch, int out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out) return;
    int j = idx % out;
    if (mask[j] == 2) y[idx] = 0.0f;
}

/* ===================== v13l: Batch forward (multiple inputs at once) =====================
 *
 * v13k problem: logic_reg does 1120 cuBLAS calls (7 pairs × 2 concepts ×
 * 10 layers × 8 matmuls). Each call has ~12ms launch overhead = 13.4s total.
 *
 * v13l solution: batch all 14 concepts into one [batch, n_embd] matrix,
 * do one cublasSgemm per layer instead of 14 cublasSgemv. This reduces
 * 1120 calls to 80 (10 layers × 8 matmuls), each slightly larger but
 * total launch overhead drops 14x.
 *
 * Y[batch, out] = X[batch, in] @ W[out, in]^T + bias[out]
 *   W is row-major [out, in], same as before.
 *   cuBLAS col-major: Y_cm[out, batch] = W_cm[in, out]^T @ X_cm[in, batch]
 *     = W_row[out, in] @ X_cm[in, batch]  (since W_row == W_cm not transposed)
 *   → cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
 *                 m=out, n=batch, k=in,
 *                 alpha, d_w (ld=in), d_x (ld=in),
 *                 beta, d_y (ld=out))
 *   With bias: pre-fill d_y with bias broadcasted, beta=1.
 */
extern "C"
void lal_cuda_fwd_batch(float *y,          /* [batch * out] output */
                        const float *x,    /* [batch * in] input, row-major */
                        const BinLayer *bl,
                        int batch) {
    LayerGPU *g = (LayerGPU*)bl->_gpu;
    if (!g || !g->uploaded) {
        /* Fallback: loop over batch using resident forward */
        int in = bl->in_dim, out = bl->out_dim;
        for (int b = 0; b < batch; b++) {
            lal_cuda_fwd_resident(y + b * out, x + b * in, bl);
        }
        return;
    }
    int in = bl->in_dim, out = bl->out_dim;
    cublasHandle_t h = get_cublas();

    /* Persistent batch buffers (grow as needed) */
    static float *d_xb = NULL, *d_yb = NULL;
    static int d_cap_b = 0;
    int need = batch * (in > out ? in : out);
    if (need > d_cap_b) {
        if (d_xb) { cudaFree(d_xb); cudaFree(d_yb); }
        cudaMalloc(&d_xb, need * sizeof(float));
        cudaMalloc(&d_yb, need * sizeof(float));
        d_cap_b = need;
    }

    CUDA_CHECK(cudaMemcpy(d_xb, x, (size_t)batch * in * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Pre-fill d_yb with bias broadcasted (each row = bias) */
    if (g->d_bias) {
        /* d_yb[b*out + j] = bias[j] for all b.
         * cuBLAS doesn't broadcast, so use a kernel. */
        int total = batch * out;
        int thr = 256, blk = (total + thr - 1) / thr;
        k_fill_bias<<<blk, thr>>>(d_yb, g->d_bias, batch, out);
        float alpha = 1.0f, beta = 1.0f;
        /* Y[out, batch] = W[in, out]^T_noop @ X[in, batch]
         * Actually: Y[b][j] = sum_i W[j][i] * X[b][i] + bias[j]
         * cuBLAS col-major: Y_cm[j, b] = sum_i W_cm[i, j] * X_cm[i, b]
         * W row-major [out, in]: W[j][i] at j*in+i. Col-major [in, out]: W_cm[i,j] at i+j*in = j*in+i. Same!
         * So W_cm == W_row (not transposed). Y_cm = W_cm @ X_cm.
         * cublasSgemm(m=out, n=batch, k=in, op=N, op=N, d_w, ld=in, d_xb, ld=in, d_yb, ld=out) */
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    out, batch, in,
                    &alpha, g->d_w, in, d_xb, in,
                    &beta, d_yb, out);
    } else {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    out, batch, in,
                    &alpha, g->d_w, in, d_xb, in,
                    &beta, d_yb, out);
    }

    /* Zero PRUNE rows (all batches) */
    if (g->d_mask) {
        int total = batch * out;
        int thr = 256, blk = (total + thr - 1) / thr;
        k_zero_prune_batch<<<blk, thr>>>(d_yb, g->d_mask, batch, out);
    }

    CUDA_CHECK(cudaMemcpy(y, d_yb, (size_t)batch * out * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

/* ===================== v13m: GPU-resident intermediates + fused layer =====================
 *
 * v13l problem: each cuBLAS call did H2D(input) + sgemm + D2H(output).
 * 6 transfers per layer × 10 layers × 2 (A+B) = 120 transfers, each ~50us
 * sync overhead = 6ms... but actual 13s due to full pipeline stalls.
 *
 * v13m solution: keep ALL intermediates on GPU. Only transfer:
 *   - Once: embeddings H2D (batch * n_embd)
 *   - Per layer: norm2 D2H (batch * n_embd) for gate_input cache
 *   - Once: final x D2H
 *
 * GPU kernels replace CPU operations:
 *   k_batch_layernorm, k_residual_clip, k_gelu, k_extract_v, k_silu_swiglu
 *
 * cuBLAS calls use device pointers directly → no sync between calls.
 * Use CUDA stream to queue cuBLAS + kernels asynchronously.
 */

/* GPU kernel: batched LayerNorm
 *   y[b][i] = (x[b][i] - mean_b) / sqrt(var_b + eps) * w[i] + bias[i]
 *   One block per batch row, 256 threads do reduction for mean/var. */
__global__ void k_batch_layernorm(float *y, const float *x,
                                    const float *w, const float *bias,
                                    int batch, int n) {
    int b = blockIdx.x;
    if (b >= batch) return;
    const float *xb = x + (size_t)b * n;
    float *yb = y + (size_t)b * n;

    /* Thread-local sum, then warp reduce */
    __shared__ float s_mean, s_var;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local_sum += xb[i];
    /* Block reduce */
    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        s_mean = sdata[0] / n;
        sdata[0] = 0.0f;  /* reuse for var */
    }
    __syncthreads();

    /* Variance */
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = xb[i] - s_mean;
        local_var += d * d;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        s_var = sdata[0] / n;
    __syncthreads();

    float inv_std = rsqrtf(s_var + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yb[i] = (xb[i] - s_mean) * inv_std * w[i] + (bias ? bias[i] : 0.0f);
}

/* GPU kernel: x += rs * delta, then clip to [-clip, clip] */
__global__ void k_residual_clip(float *x, const float *delta,
                                 float rs, float clip, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = x[idx] + rs * delta[idx];
    if (v > clip) v = clip;
    if (v < -clip) v = -clip;
    x[idx] = v;
}

/* GPU kernel: GELU activation (GPT-2 style: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))) */
__global__ void k_gelu(float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = x[idx];
    float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    x[idx] = 0.5f * v * (1.0f + tanhf(c));
}

/* GPU kernel: extract V from QKV merged buffer
 *   v[b][i] = qkv[b][2*n + i]  (V is 3rd third of [Q|K|V]) */
__global__ void k_extract_v(float *v, const float *qkv, int batch, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n;
    if (idx >= total) return;
    int b = idx / n;
    int i = idx % n;
    v[idx] = qkv[(size_t)b * 3 * n + 2 * n + i];
}

/* GPU kernel: copy (for V-copy attention, no QK mixing) */
__global__ void k_batch_copy(float *dst, const float *src, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = src[idx];
}

/* v13m: Fused batch layer forward — ALL intermediates stay on GPU.
 *
 * Does one full transformer layer forward for [batch] inputs:
 *   norm1 = LN(x)
 *   qkv = W_q @ norm1
 *   v = qkv[2n:]  (V-copy, no real attention)
 *   proj = W_o @ v
 *   x += rs * proj, clip
 *   norm2 = LN(x)
 *   hidden = gelu(W_gate @ norm2)
 *   mlp_out = W_down @ hidden
 *   x += rs * mlp_out, clip
 *
 * Only d_x (residual) is modified in-place. norm2 copied to CPU for gate_input.
 * All cuBLAS calls use device pointers → queued on stream, no host sync. */
extern "C"
void lal_cuda_layer_forward_batch(
    float *d_x,           /* [batch, n] — residual stream (device, modified in-place) */
    float *h_norm2_out,   /* [batch, n] — CPU output: norm2 for gate_input cache */
    TransLayer *tl,       /* layer weights (all resident on GPU via _gpu) */
    ModelConfig *cfg,
    int batch,
    void *stream_void
) {
    cudaStream_t stream = (cudaStream_t)stream_void;
    int n = cfg->n_embd;
    int mlp_dim = cfg->mlp_dim;
    int m = mlp_dim;
    float rs = cfg->residual_scale;

    /* Persistent device scratch buffers (grown on demand) */
    static float *d_n1 = NULL, *d_n2 = NULL, *d_qkv = NULL;
    static float *d_ao = NULL, *d_proj = NULL, *d_hid = NULL, *d_mlp = NULL;
    /* v13m: norm weights on GPU (small, uploaded per layer via async memcpy) */
    static float *d_nw1 = NULL, *d_nb1 = NULL, *d_nw2 = NULL, *d_nb2 = NULL;
    static int d_cap_m = 0;
    int max_dim = n > m ? n : m;
    int need = batch * (3 * n > max_dim ? 3 * n : max_dim);
    if (need > d_cap_m) {
        if (d_n1) { cudaFree(d_n1); cudaFree(d_n2); cudaFree(d_qkv);
                    cudaFree(d_ao); cudaFree(d_proj); cudaFree(d_hid); cudaFree(d_mlp);
                    cudaFree(d_nw1); cudaFree(d_nb1); cudaFree(d_nw2); cudaFree(d_nb2); }
        cudaMalloc(&d_n1, batch * n * sizeof(float));
        cudaMalloc(&d_n2, batch * n * sizeof(float));
        cudaMalloc(&d_qkv, batch * 3 * n * sizeof(float));
        cudaMalloc(&d_ao, batch * n * sizeof(float));
        cudaMalloc(&d_proj, batch * n * sizeof(float));
        cudaMalloc(&d_hid, batch * m * sizeof(float));
        cudaMalloc(&d_mlp, batch * n * sizeof(float));
        cudaMalloc(&d_nw1, n * sizeof(float));
        cudaMalloc(&d_nb1, n * sizeof(float));
        cudaMalloc(&d_nw2, n * sizeof(float));
        cudaMalloc(&d_nb2, n * sizeof(float));
        d_cap_m = need;
    }

    /* Upload norm weights to GPU (async, small: 4 × 2KB = 8KB) */
    if (tl->norm1_w) cudaMemcpyAsync(d_nw1, tl->norm1_w, n * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (tl->norm1_b) cudaMemcpyAsync(d_nb1, tl->norm1_b, n * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (tl->norm2_w) cudaMemcpyAsync(d_nw2, tl->norm2_w, n * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (tl->norm2_b) cudaMemcpyAsync(d_nb2, tl->norm2_b, n * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);

    cublasHandle_t h = get_cublas();
    cublasSetStream(h, stream);

    int thr = 256;
    float alpha = 1.0f, beta = 0.0f;

    /* 1. norm1 = LayerNorm(x) — GPU kernel, uses device-resident norm weights */
    k_batch_layernorm<<<batch, thr, 0, stream>>>(d_n1, d_x, d_nw1,
                                                    d_nb1, batch, n);

    /* 2. qkv = W_q @ norm1 (QKV merged, out=3n) — cuBLAS on device pointers */
    LayerGPU *gq = (LayerGPU*)tl->attn_q._gpu;
    if (gq && gq->uploaded) {
        if (gq->d_bias) {
            /* Pre-fill d_qkv with bias broadcast, beta=1 */
            k_fill_bias<<<(batch * 3 * n + thr - 1) / thr, thr, 0, stream>>>(
                d_qkv, gq->d_bias, batch, 3 * n);
            beta = 1.0f;
        }
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    3 * n, batch, n, &alpha, gq->d_w, n, d_n1, n,
                    &beta, d_qkv, 3 * n);
        beta = 0.0f;
    }

    /* 3. v = qkv[2n:] — GPU kernel */
    k_extract_v<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
        d_ao, d_qkv, batch, n);

    /* Zero PRUNE rows for attn_o */
    LayerGPU *go = (LayerGPU*)tl->attn_o._gpu;
    if (go && go->uploaded) {
        if (go->d_bias) {
            k_fill_bias<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
                d_proj, go->d_bias, batch, n);
            beta = 1.0f;
        }
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, batch, n, &alpha, go->d_w, n, d_ao, n,
                    &beta, d_proj, n);
        if (go->d_mask)
            k_zero_prune_batch<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
                d_proj, go->d_mask, batch, n);
        beta = 0.0f;
    }

    /* 4. x += rs * proj, clip — GPU kernel */
    k_residual_clip<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
        d_x, d_proj, rs, 10.0f, batch * n);

    /* 5. norm2 = LayerNorm(x) — GPU kernel */
    k_batch_layernorm<<<batch, thr, 0, stream>>>(d_n2, d_x, d_nw2,
                                                    d_nb2, batch, n);

    /* Copy norm2 to CPU (gate_input for gradient computation) */
    cudaMemcpyAsync(h_norm2_out, d_n2, batch * n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    /* 6. hidden = gelu(W_gate @ norm2) */
    LayerGPU *gg = (LayerGPU*)tl->mlp_gate._gpu;
    if (gg && gg->uploaded) {
        if (gg->d_bias) {
            k_fill_bias<<<(batch * m + thr - 1) / thr, thr, 0, stream>>>(
                d_hid, gg->d_bias, batch, m);
            beta = 1.0f;
        }
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, batch, n, &alpha, gg->d_w, n, d_n2, n,
                    &beta, d_hid, m);
        if (gg->d_mask)
            k_zero_prune_batch<<<(batch * m + thr - 1) / thr, thr, 0, stream>>>(
                d_hid, gg->d_mask, batch, m);
        beta = 0.0f;
    }
    k_gelu<<<(batch * m + thr - 1) / thr, thr, 0, stream>>>(d_hid, batch * m);

    /* 7. mlp_out = W_down @ hidden */
    LayerGPU *gd = (LayerGPU*)tl->mlp_down._gpu;
    if (gd && gd->uploaded) {
        if (gd->d_bias) {
            k_fill_bias<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
                d_mlp, gd->d_bias, batch, n);
            beta = 1.0f;
        }
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, batch, m, &alpha, gd->d_w, m, d_hid, m,
                    &beta, d_mlp, n);
        /* Note: mlp_down typically doesn't have logic_mask, skip PRUNE zero */
        beta = 0.0f;
    }

    /* 8. x += rs * mlp_out, clip — GPU kernel */
    k_residual_clip<<<(batch * n + thr - 1) / thr, thr, 0, stream>>>(
        d_x, d_mlp, rs, 10.0f, batch * n);
}

/* v13m: High-level batch gate_inputs computation.
 * Uploads embeddings, runs all layers fused on GPU, returns gate_inputs.
 * Manages stream + device buffer internally. */

/* Forward declaration — defined below */
__global__ void k_add_wpe(float *x, const float *wpe, int batch, int n);

extern "C"
void lal_cuda_compute_gate_inputs_batch(
    Model *m,              /* model with layers loaded */
    const float *embs,     /* [batch * n_embd] host embeddings */
    int batch,
    float *out_gate_inputs, /* [batch * n_layer * n_embd] host output */
    int n_embd
) {
    int n_layer = m->cfg.n_layer;
    int n = n_embd;

    /* Persistent stream + residual buffer */
    static cudaStream_t s_stream = NULL;
    static float *d_x = NULL;
    static int d_x_cap = 0;
    if (!s_stream) cudaStreamCreate(&s_stream);
    int need = batch * n;
    if (need > d_x_cap) {
        if (d_x) cudaFree(d_x);
        cudaMalloc(&d_x, need * sizeof(float));
        d_x_cap = need;
    }

    /* Upload embeddings to GPU */
    cudaMemcpyAsync(d_x, embs, (size_t)batch * n * sizeof(float),
                    cudaMemcpyHostToDevice, s_stream);
    if (m->wpe) {
        /* Upload wpe[0..n-1] to device, then add to each row */
        static float *d_wpe = NULL;
        static int d_wpe_n = 0;
        if (n > d_wpe_n) {
            if (d_wpe) cudaFree(d_wpe);
            cudaMalloc(&d_wpe, n * sizeof(float));
            d_wpe_n = n;
        }
        cudaMemcpyAsync(d_wpe, m->wpe, n * sizeof(float),
                        cudaMemcpyHostToDevice, s_stream);
        k_add_wpe<<<(need + 255) / 256, 256, 0, s_stream>>>(d_x, d_wpe, batch, n);
    }

    /* Run all layers fused on GPU */
    for (int l = 0; l < n_layer; l++) {
        float *norm2_out = &out_gate_inputs[(size_t)l * batch * n];
        lal_cuda_layer_forward_batch(d_x, norm2_out, &m->layers[l], &m->cfg,
                                      batch, (void*)s_stream);
    }

    /* Sync — wait for all layers to complete */
    cudaStreamSynchronize(s_stream);
}

/* Helper: add wpe[0..n-1] to each row of x[batch, n] */
__global__ void k_add_wpe(float *x, const float *wpe, int batch, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * n) return;
    int i = idx % n;
    x[idx] += wpe[i];
}

/* ===================== v13p: Full GPU forward (llama.cpp style) =====================
 *
 * Like llama.cpp: entire model + all intermediates stay on GPU. Only transfers:
 *   H2D: tokens (4 bytes) + wte lookup (done on GPU)
 *   D2H: logits (vocab * 4 bytes)
 *
 * Zero intermediate transfers. All norm/clip/gelu/residual are GPU kernels.
 * cuBLAS calls use device pointers only, queued on single stream (no sync).
 *
 * This replaces trans_layer_forward + model_forward for the CE training path.
 * logic_reg still uses lal_cuda_compute_gate_inputs_batch (already GPU).
 */

/* GPU kernel: embedding lookup — x = wte[token] + wpe[pos] */
__global__ void k_embedding_lookup(float *x, const float *wte, const float *wpe,
                                     int token, int pos, int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    x[i] = wte[(size_t)token * n_embd + i] + (wpe ? wpe[(size_t)pos * n_embd + i] : 0.0f);
}

/* GPU kernel: final layernorm + logits = wte @ x  (partial, only target token)
 * Actually for CE loss we need full logits = wte @ final_hidden.
 * wte is [vocab, n_embd], final_hidden is [n_embd].
 * logits[v] = sum_i wte[v*n_embd+i] * final_hidden[i]
 * This is a matrix-vector product: cublasSgemv.
 * But vocab=32768 is large, so we compute it on GPU. */

/* GPU kernel: MLP cap + residual add
 *   if ||mlp_out|| > cap: scale = cap / ||mlp_out||, else scale = 1
 *   x[i] += rs * scale * mlp_out[i]
 *   Then normalize_residual: if ||x|| > target, scale x to target */
__global__ void k_mlp_cap_residual(float *x, const float *mlp_out,
                                     float rs, float cap, float target_norm,
                                     int n) {
    /* This kernel needs ||mlp_out|| which requires reduction.
     * Use shared memory reduction like layernorm. */
    extern __shared__ float sdata[];
    
    /* Phase 1: compute ||mlp_out|| */
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local += mlp_out[i] * mlp_out[i];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mlp_norm_sq = sdata[0];
    float mlp_norm = sqrtf(mlp_norm_sq) + 1e-8f;
    float mlp_scale = (mlp_norm > cap) ? (cap / mlp_norm) : 1.0f;
    __syncthreads();
    
    /* Phase 2: x += rs * mlp_scale * mlp_out */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] += rs * mlp_scale * mlp_out[i];
    __syncthreads();
    
    /* Phase 3: compute ||x|| for normalize_residual */
    local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local += x[i] * x[i];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float x_norm = sqrtf(sdata[0]) + 1e-8f;
    float x_scale = (x_norm > target_norm) ? (target_norm / x_norm) : 1.0f;
    __syncthreads();
    
    /* Phase 4: scale x */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] *= x_scale;
}

/* ===================== v13q: Full GPU backward (cached activations) =====================
 *
 * Forward caches all intermediate activations on GPU (d_act array).
 * Backward replays in reverse, computing gradients for w_float using
 * k_matvec_transpose (W^T @ grad) and accumulating into grad_accum.
 *
 * Architecture (per layer l, reversed):
 *   Forward:  x → norm1 → qkv → v_copy → proj → x+=proj → norm2 → gate → gelu → down → x+=mlp
 *   Backward: grad_x ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
 *
 * Key kernels needed for backward:
 *   k_matvec_T: grad_x[i] = sum_j W[j*in+i] * grad_y[j]  (W^T @ grad_y)
 *   k_grad_W:   grad_W[j*in+i] += grad_y[j] * x[i]       (outer product accumulation)
 *   k_gelu_grad: grad = grad * gelu_grad(x)
 *   k_layernorm_grad: backprop through layernorm
 */

/* Persistent activation cache for backward */
typedef struct {
    float *d_x_pre;      /* [n] x before each layer (for residual skip) */
    float *d_n1;         /* [n] norm1 output */
    float *d_n2;         /* [n] norm2 output */
    float *d_ao;         /* [n] attn_out (V copy) */
    float *d_hid;        /* [mlp_dim] hidden (post-gelu) */
    float *d_proj;       /* [n] proj_out */
    float *d_mlp;        /* [n] mlp_out */
    float *d_mlp_scale;  /* [1] mlp cap scale */
} LayerAct;

static LayerAct *g_acts = NULL;
static int g_acts_nlayer = 0;
static int g_acts_n = 0;
static int g_acts_mlp = 0;
static float *d_grad_x = NULL;      /* [n] gradient w.r.t. x (working buffer) */
static float *d_grad_logits = NULL; /* [vocab] softmax - onehot */
static float *d_softmax = NULL;     /* [vocab] softmax probabilities */

/* Allocate activation cache for all layers */
static void alloc_acts(int n_layer, int n, int mlp_dim, int vocab) {
    if (g_acts && g_acts_nlayer == n_layer && g_acts_n == n) return;
    if (g_acts) {
        for (int l = 0; l < g_acts_nlayer; l++) {
            cudaFree(g_acts[l].d_x_pre); cudaFree(g_acts[l].d_n1);
            cudaFree(g_acts[l].d_n2); cudaFree(g_acts[l].d_ao);
            cudaFree(g_acts[l].d_hid); cudaFree(g_acts[l].d_proj);
            cudaFree(g_acts[l].d_mlp);
        }
        free(g_acts);
    }
    g_acts = (LayerAct*)calloc(n_layer, sizeof(LayerAct));
    for (int l = 0; l < n_layer; l++) {
        cudaMalloc(&g_acts[l].d_x_pre, n * sizeof(float));
        cudaMalloc(&g_acts[l].d_n1, n * sizeof(float));
        cudaMalloc(&g_acts[l].d_n2, n * sizeof(float));
        cudaMalloc(&g_acts[l].d_ao, n * sizeof(float));
        cudaMalloc(&g_acts[l].d_hid, mlp_dim * sizeof(float));
        cudaMalloc(&g_acts[l].d_proj, n * sizeof(float));
        cudaMalloc(&g_acts[l].d_mlp, n * sizeof(float));
    }
    if (!d_grad_x) cudaMalloc(&d_grad_x, n * sizeof(float));
    if (!d_grad_logits) cudaMalloc(&d_grad_logits, vocab * sizeof(float));
    if (!d_softmax) cudaMalloc(&d_softmax, vocab * sizeof(float));
    g_acts_nlayer = n_layer; g_acts_n = n; g_acts_mlp = mlp_dim;
}

/* GPU kernel: transpose matvec — grad_x = W^T @ grad_y
 *   W is [out, in] row-major: W[j*in + i]
 *   grad_x[i] = sum_j W[j*in+i] * grad_y[j] */
__global__ void k_matvec_T(float *grad_x, const float *W,
                            const float *grad_y, int in_dim, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_dim) return;
    float s = 0.0f;
    for (int j = 0; j < out_dim; j++)
        s += W[(size_t)j * in_dim + i] * grad_y[j];
    grad_x[i] = s;
}

/* GPU kernel: accumulate weight gradient — grad_W[j*in+i] += grad_y[j] * x[i]
 * Also accumulate bias gradient: grad_bias[j] += grad_y[j] */
__global__ void k_grad_W_bias(float *grad_W, float *grad_bias,
                                const float *grad_y, const float *x,
                                int in_dim, int out_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    float gy = grad_y[j];
    float *gw = grad_W + (size_t)j * in_dim;
    for (int i = 0; i < in_dim; i++)
        gw[i] += gy * x[i];
    if (grad_bias)
        grad_bias[j] += gy;
}

/* GPU kernel: GELU gradient — grad *= 0.5 * (1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
 *   + constant from gelu'(x) */

/* GPU kernel: LayerNorm backward
 * Given: grad_y (gradient w.r.t. LN output), x (pre-LN input), w (LN weight)
 * Compute: grad_x (gradient w.r.t. x)
 * LN forward: y = (x - mean) * inv_std * w + b
 * grad_x[i] = inv_std * w[i] * (grad_y[i] - mean(grad_y*w) - x_norm[i] * mean(grad_y*w*x_norm))
 * where x_norm[i] = (x[i] - mean) * inv_std
 *
 * Two-pass: first compute mean and dot products, then apply.
 * Uses shared memory for reduction. */
__global__ void k_layernorm_backward(float *grad_x, const float *grad_y,
                                       const float *x, const float *w,
                                       int n) {
    extern __shared__ float smem[];
    
    /* Pass 1: compute mean(x), inv_std, mean(grad_y*w), mean(grad_y*w*x_norm) */
    float local_sum = 0.0f, local_gw = 0.0f, local_gwx = 0.0f;
    float local_mean = 0.0f; /* will compute via reduction */
    
    /* First: compute mean of x */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local_sum += x[i];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / n;
    __syncthreads();
    
    /* Compute var and inv_std */
    local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = x[i] - mean;
        local_sum += d * d;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / n + 1e-5f);
    __syncthreads();
    
    /* Compute x_norm[i] = (x[i]-mean)*inv_std, then mean(grad_y*w*x_norm) */
    local_gw = 0.0f; local_gwx = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xn = (x[i] - mean) * inv_std;
        float gyw = grad_y[i] * w[i];
        local_gw += gyw;
        local_gwx += gyw * xn;
    }
    smem[threadIdx.x] = local_gw;
    smem[blockDim.x + threadIdx.x] = local_gwx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
            smem[blockDim.x + threadIdx.x] += smem[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_gw = smem[0] / n;
    float mean_gwx = smem[blockDim.x] / n;
    __syncthreads();
    
    /* Pass 2: grad_x[i] = inv_std * w[i] * (grad_y[i] - mean_gw - x_norm[i]*mean_gwx) */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xn = (x[i] - mean) * inv_std;
        grad_x[i] = inv_std * w[i] * (grad_y[i] - mean_gw - xn * mean_gwx);
    }
}

__global__ void k_gelu_grad(float *grad, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    float t = tanhf(c);
    /* gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t*t) * 0.7978845608 * (1 + 0.134145 * x*x) */
    float dv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t*t) * 0.7978845608f * (1.0f + 0.134145f * v * v);
    grad[i] *= dv;
}

/* GPU kernel: softmax - onehot (CE gradient w.r.t. logits) */
__global__ void k_ce_grad(float *grad_logits, float *softmax,
                           const float *logits, int target, float max_l, int vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    float p = expf(logits[i] - max_l);  /* unnormalized */
    softmax[i] = p;  /* will normalize on host or via reduction */
    grad_logits[i] = p;  /* temp, normalize after */
}

/* Normalize softmax and compute grad_logits = softmax - onehot(target) */
__global__ void k_normalize_ce_grad(float *grad_logits, float *softmax,
                                      float inv_sum, int target, int vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    float p = softmax[i] * inv_sum;
    softmax[i] = p;
    grad_logits[i] = p - (i == target ? 1.0f : 0.0f);
}

/* v13q: Full GPU backward.
 * Given softmax from forward, compute gradients for all weights.
 * Accumulates into bl->grad_accum (host-side, downloaded at end). */

static float *d_logits = NULL;  /* [vocab] shared forward/backward */
static float *d_wte = NULL;     /* [vocab*n] shared */

/* v13p: Full GPU forward for CE training.
 * Input: token IDs [seq_len], target token
 * Output: CE loss (returned), grad_hidden [n_embd] (for backward)
 *
 * Forward:
 *   x = wte[token] + wpe[pos]
 *   for each layer: norm1 → qkv → v_copy → proj → residual → norm2 → gate → gelu → down → residual
 *   final_norm → logits = wte @ x → CE loss
 *
 * All on GPU, single stream, zero intermediate H2D/D2H. */

/* v13p: custom matvec — no cuBLAS tiling OOB */
__global__ void k_matvec(float *y, const float *W, const float *x,
                          const float *bias, int in_dim, int out_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    float s = bias ? bias[j] : 0.0f;
    const float *wr = W + (size_t)j * in_dim;
    for (int i = 0; i < in_dim; i++)
        s += wr[i] * x[i];
    y[j] = s;
}

extern "C"
float lal_cuda_full_forward(
    Model *m,
    const int *tokens,     /* host, [seq_len] */
    int seq_len,
    int target,            /* target token for CE loss */
    float *grad_hidden,    /* host, [n_embd] — gradient w.r.t. final hidden (for backward) */
    int *predicted         /* host, [1] — argmax token (for accuracy) */
) {
    int n = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int vocab = m->cfg.vocab_size;
    int mlp_dim = m->cfg.mlp_dim;
    float rs = m->cfg.residual_scale;
    
    static cudaStream_t s = NULL;
    if (!s) cudaStreamCreate(&s);
    cublasHandle_t h = get_cublas();
    cublasSetStream(h, s);
    
    /* Persistent device buffers */
    static float *d_x = NULL;       /* [n] residual stream */
    static float *d_n1 = NULL, *d_n2 = NULL;  /* [n] norm outputs */
    static float *d_qkv = NULL;     /* [3n] QKV merged */
    static float *d_ao = NULL;      /* [n] attn out (V copy) */
    static float *d_proj = NULL;    /* [n] proj out */
    static float *d_hid = NULL;     /* [mlp_dim] hidden */
    static float *d_mlp = NULL;     /* [n] mlp out */
    /* d_logits and d_wte are file-scope static (shared with backward) */
    static float *d_wpe = NULL;     /* [n_ctx * n] wpe on device */
    static float *d_nw1=NULL,*d_nb1=NULL,*d_nw2=NULL,*d_nb2=NULL,*d_lnfw=NULL,*d_lnfb=NULL;
    static int d_cap = 0;
    
    int need = n > mlp_dim ? n : mlp_dim;
    if (need > d_cap || !d_x) {
        if (d_x) { cudaFree(d_x); cudaFree(d_n1); cudaFree(d_n2); cudaFree(d_qkv);
                   cudaFree(d_ao); cudaFree(d_proj); cudaFree(d_hid); cudaFree(d_mlp);
                   cudaFree(d_logits); }
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_n1, n * sizeof(float));
        cudaMalloc(&d_n2, n * sizeof(float));
        cudaMalloc(&d_qkv, 3 * n * sizeof(float));
        cudaMalloc(&d_ao, n * sizeof(float));
        cudaMalloc(&d_proj, n * sizeof(float));
        cudaMalloc(&d_hid, mlp_dim * sizeof(float));
        cudaMalloc(&d_mlp, n * sizeof(float));
        cudaMalloc(&d_logits, vocab * sizeof(float));
        cudaMalloc(&d_nw1, n*4); cudaMalloc(&d_nb1, n*4);
        cudaMalloc(&d_nw2, n*4); cudaMalloc(&d_nb2, n*4);
        cudaMalloc(&d_lnfw, n*4); cudaMalloc(&d_lnfb, n*4);
        /* Upload wte + wpe once */
        cudaMalloc(&d_wte, (size_t)vocab * n * sizeof(float));
        cudaMemcpy(d_wte, m->wte, (size_t)vocab * n * sizeof(float), cudaMemcpyHostToDevice);
        if (m->wpe) {
            cudaMalloc(&d_wpe, (size_t)m->cfg.n_ctx * n * sizeof(float));
            cudaMemcpy(d_wpe, m->wpe, (size_t)m->cfg.n_ctx * n * sizeof(float), cudaMemcpyHostToDevice);
        }
        d_cap = need;
    }
    
    int pos = seq_len - 1;  /* predict next token at last position */
    int token = tokens[pos];
    
    /* 1. Embedding lookup on GPU */
    k_embedding_lookup<<<(n+255)/256, 256, 0, s>>>(d_x, d_wte, d_wpe, token, pos, n);
    
    float alpha = 1.0f, beta = 0.0f;
    int thr = 256;
    
    /* v13q: allocate activation cache for backward */
    alloc_acts(n_layer, n, mlp_dim, vocab);
    
    /* 2. Run all layers on GPU */
    for (int l = 0; l < n_layer; l++) {
        TransLayer *tl = &m->layers[l];
        
        /* v13q: cache x before layer for backward */
        cudaMemcpyAsync(g_acts[l].d_x_pre, d_x, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* Upload norm weights (small, async) */
        cudaMemcpyAsync(d_nw1, tl->norm1_w, n*4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_nb1, tl->norm1_b, n*4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_nw2, tl->norm2_w, n*4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_nb2, tl->norm2_b, n*4, cudaMemcpyHostToDevice, s);
        
        /* norm1 = LN(x) */
        k_batch_layernorm<<<1, thr, 0, s>>>(d_n1, d_x, d_nw1, d_nb1, 1, n);
        cudaMemcpyAsync(g_acts[l].d_n1, d_n1, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* qkv = W_q @ norm1 (QKV merged) */
        LayerGPU *gq = (LayerGPU*)tl->attn_q._gpu;
        if (gq && gq->uploaded) {
            k_matvec<<<(3*n+255)/256, 256, 0, s>>>(d_qkv, gq->d_w, d_n1, gq->d_bias, n, 3*n);
        }
        
        /* V copy: attn_out = qkv[2n:3n] */
        cudaMemcpyAsync(g_acts[l].d_ao, d_qkv + 2*n, n*4, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(d_ao, d_qkv + 2*n, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* proj = W_o @ attn_out */
        LayerGPU *go = (LayerGPU*)tl->attn_o._gpu;
        if (go && go->uploaded) {
            k_matvec<<<(n+255)/256, 256, 0, s>>>(d_proj, go->d_w, d_ao, go->d_bias, n, n);
            if (go->d_mask)
                k_zero_prune<<<(n+255)/256, 256, 0, s>>>(d_proj, go->d_mask, n);
        }
        
        cudaMemcpyAsync(g_acts[l].d_proj, d_proj, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* x += rs * proj */
        k_residual_clip<<<(n+255)/256, 256, 0, s>>>(d_x, d_proj, rs, 10.0f, n);
        
        /* norm2 = LN(x) */
        k_batch_layernorm<<<1, thr, 0, s>>>(d_n2, d_x, d_nw2, d_nb2, 1, n);
        cudaMemcpyAsync(g_acts[l].d_n2, d_n2, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* hidden = gelu(W_gate @ norm2) */
        LayerGPU *gg = (LayerGPU*)tl->mlp_gate._gpu;
        if (gg && gg->uploaded) {
            k_matvec<<<(mlp_dim+255)/256, 256, 0, s>>>(d_hid, gg->d_w, d_n2, gg->d_bias, n, mlp_dim);
            if (gg->d_mask)
                k_zero_prune<<<(mlp_dim+255)/256, 256, 0, s>>>(d_hid, gg->d_mask, mlp_dim);
        }
        /* v13t: cache pre-GELU for backward */
        cudaMemcpyAsync(g_acts[l].d_hid, d_hid, mlp_dim*4, cudaMemcpyDeviceToDevice, s);
        k_gelu<<<(mlp_dim+255)/256, 256, 0, s>>>(d_hid, mlp_dim);
        
        /* mlp_out = W_down @ hidden */
        LayerGPU *gd = (LayerGPU*)tl->mlp_down._gpu;
        if (gd && gd->uploaded) {
            k_matvec<<<(n+255)/256, 256, 0, s>>>(d_mlp, gd->d_w, d_hid, gd->d_bias, mlp_dim, n);
        }
        
        cudaMemcpyAsync(g_acts[l].d_mlp, d_mlp, n*4, cudaMemcpyDeviceToDevice, s);
        
        /* x += rs * mlp_out (with cap) + normalize_residual */
        k_mlp_cap_residual<<<1, thr, thr*sizeof(float), s>>>(d_x, d_mlp, rs, 0.5f, 3.0f, n);
    }
    
    /* 3. Final layernorm */
    cudaMemcpyAsync(d_lnfw, m->ln_f_w, n*4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_lnfb, m->ln_f_b, n*4, cudaMemcpyHostToDevice, s);
    k_batch_layernorm<<<1, thr, 0, s>>>(d_x, d_x, d_lnfw, d_lnfb, 1, n);
    
    /* 4. logits = wte @ x  (full vocab) */
    k_matvec<<<(vocab+255)/256, 256, 0, s>>>(d_logits, d_wte, d_x, NULL, n, vocab);
    
    /* 5. CE loss + argmax on GPU (download only loss + argmax + grad) */
    /* For now: download logits, compute CE on CPU (simpler, vocab=32768 is small) */
    cudaStreamSynchronize(s);
    
    static float *h_logits = NULL;
    if (!h_logits) h_logits = (float*)malloc(vocab * sizeof(float));
    cudaMemcpy(h_logits, d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    
    /* CE loss on CPU */
    float max_l = h_logits[0];
    int argmax = 0;
    for (int i = 1; i < vocab; i++) {
        if (h_logits[i] > max_l) { max_l = h_logits[i]; argmax = i; }
    }
    float exp_sum = 0;
    for (int i = 0; i < vocab; i++) exp_sum += expf(h_logits[i] - max_l);
    float loss = -logf(expf(h_logits[target] - max_l) / exp_sum);
    *predicted = argmax;
    
    /* Compute grad_hidden = wte[target] - softmax @ wte  (for backward) */
    /* grad_hidden[i] = wte[target][i] - sum_v softmax[v] * wte[v][i] */
    /* Download x (final hidden) and compute on CPU for now */
    cudaMemcpy(grad_hidden, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    return loss;
}

extern "C"
void lal_cuda_full_backward(
    Model *m,
    const int *tokens,
    int seq_len,
    int target
) {
    int n = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int vocab = m->cfg.vocab_size;
    int mlp_dim = m->cfg.mlp_dim;
    float rs = m->cfg.residual_scale;
    
    static cudaStream_t s = NULL;
    if (!s) cudaStreamCreate(&s);
    
    alloc_acts(n_layer, n, mlp_dim, vocab);
    
    int thr = 256;
    
    /* 1. Compute softmax and CE gradient on GPU
     * grad_logits = softmax(logits) - onehot(target) */
    /* d_logits still has logits from forward. d_softmax = exp(logits - max) */
    /* Need max first — do on CPU side (already have h_logits from forward) */
    /* For simplicity: recompute softmax on GPU via two-pass */
    
    /* Pass 1: compute exp(logits - max) into d_softmax */
    /* We need max_l — use a reduction kernel or just pass from forward */
    /* For now: download logits, compute softmax, upload grad_logits */
    static float *h_logits = NULL;
    if (!h_logits) h_logits = (float*)malloc(vocab * sizeof(float));
    cudaMemcpy(h_logits, d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_l = h_logits[0];
    for (int i = 1; i < vocab; i++)
        if (h_logits[i] > max_l) max_l = h_logits[i];
    float exp_sum = 0;
    for (int i = 0; i < vocab; i++) {
        h_logits[i] = expf(h_logits[i] - max_l);
        exp_sum += h_logits[i];
    }
    float inv_sum = 1.0f / (exp_sum + 1e-12f);
    for (int i = 0; i < vocab; i++) {
        float p = h_logits[i] * inv_sum;
        h_logits[i] = p - (i == target ? 1.0f : 0.0f);  /* grad_logits */
    }
    cudaMemcpy(d_grad_logits, h_logits, vocab * sizeof(float), cudaMemcpyHostToDevice);
    
    /* 2. grad_x_final = wte^T @ grad_logits  (backprop through logits) */
    k_matvec_T<<<(n+thr-1)/thr, thr, 0, s>>>(d_grad_x, d_wte, d_grad_logits, n, vocab);
    
    /* Also accumulate grad_wte: grad_wte[v*n+i] += grad_logits[v] * final_hidden[i]
     * But final_hidden (d_x) was overwritten by final layernorm. We need pre-LN x.
     * For now skip wte gradient (it's updated by logic_reg separately). */
    
    /* 3. Backprop through final layernorm (simplified: just pass grad through)
     * TODO: proper layernorm backward. For now, scale by ln_f_w. */
    k_residual_clip<<<(n+thr-1)/thr, thr, 0, s>>>(d_grad_x, d_grad_x, 1.0f, 1e10f, n);
    
    /* 4. Backprop through layers in reverse */
    for (int l = n_layer - 1; l >= 0; l--) {
        TransLayer *tl = &m->layers[l];
        LayerAct *a = &g_acts[l];
        
        /* --- MLP backward ---
         * Forward: mlp_out = W_down @ hidden; x += rs * mlp_scale * mlp_out
         * Backward: grad_mlp = grad_x (scaled by mlp_scale)
         *           grad_hidden = W_down^T @ grad_mlp
         *           grad_W_down += grad_mlp @ hidden^T  (outer product)
         *           grad_x = grad_x (residual, already has grad)
         *           Then: grad_x -= rs * mlp_scale * grad_mlp (undo residual)
         *                 Actually: grad_x_residual = grad_x (from next layer)
         *                           grad_mlp_out = grad_x * rs * mlp_scale
         *                           grad_x = grad_x_residual + grad_through_mlp */
        
        /* grad_mlp = grad_x * rs (residual contribution) */
        /* For simplicity: treat mlp_scale as 1.0 (cap is small, gradient is approximate) */
        /* grad_mlp_out = d_grad_x * rs */
        /* Use k_residual_clip to copy with scale */
        /* Actually we need: grad_mlp = grad_x, then grad_W_down += grad_mlp outer hidden,
         * grad_hidden = W_down^T @ grad_mlp, grad_x += rs * grad_hidden (undo residual is complex) */
        
        /* Simplified backward (skip mlp_cap and normalize_residual gradients): */
        
        /* grad_mlp_out = grad_x (from residual path) */
        /* grad_W_down[j*mlp+i] += grad_mlp[j] * hidden[i] */
        LayerGPU *gd = (LayerGPU*)tl->mlp_down._gpu;
        if (gd && gd->uploaded && tl->mlp_down.grad_accum) {
            /* v13s: GPU accumulation to device grad_accum (no download!) */
            k_grad_W_bias<<<(n+thr-1)/thr, thr, 0, s>>>(
                tl->mlp_down.d_grad_accum, tl->mlp_down.d_bias_grad_accum,
                d_grad_x, a->d_hid, mlp_dim, n);
            
            /* grad_hidden = W_down^T @ grad_mlp */
            k_matvec_T<<<(mlp_dim+thr-1)/thr, thr, 0, s>>>(a->d_hid, gd->d_w, d_grad_x, mlp_dim, n);
        }
        
        /* GELU backward: grad_hidden *= gelu'(pre_gelu) */
        /* But d_hid was overwritten by gelu forward. We need pre-gelu activation.
         * For now: approximate gelu_grad using post-gelu value (less accurate) */
        k_gelu_grad<<<(mlp_dim+thr-1)/thr, thr, 0, s>>>(a->d_hid, a->d_hid, mlp_dim);
        
        /* grad_W_gate += grad_hidden outer norm2 */
        LayerGPU *gg = (LayerGPU*)tl->mlp_gate._gpu;
        if (gg && gg->uploaded && tl->mlp_gate.grad_accum) {
            /* v13s: GPU accumulation */
            k_grad_W_bias<<<(mlp_dim+thr-1)/thr, thr, 0, s>>>(
                tl->mlp_gate.d_grad_accum, tl->mlp_gate.d_bias_grad_accum,
                a->d_hid, a->d_n2, n, mlp_dim);
            
            /* grad_norm2 = W_gate^T @ grad_hidden */
            k_matvec_T<<<(n+thr-1)/thr, thr, 0, s>>>(a->d_n2, gg->d_w, a->d_hid, n, mlp_dim);
        }
        
        /* v13t: Proper LayerNorm2 backward */
        /* Upload norm2 weights */
        static float *d_nw2_bwd = NULL, *d_nb2_bwd = NULL;
        if (!d_nw2_bwd) { cudaMalloc(&d_nw2_bwd, n*4); cudaMalloc(&d_nb2_bwd, n*4); }
        cudaMemcpyAsync(d_nw2_bwd, tl->norm2_w, n*4, cudaMemcpyHostToDevice, s);
        k_layernorm_backward<<<1, thr, 2*thr*sizeof(float), s>>>(
            a->d_n2, a->d_n2, a->d_x_pre, d_nw2_bwd, n);
        /* grad_x += grad_norm2 (residual path) */
        k_residual_clip<<<(n+thr-1)/thr, thr, 0, s>>>(d_grad_x, a->d_n2, 1.0f, 1e10f, n);
        
        /* --- Attention backward ---
         * Forward: proj = W_o @ attn_out; x += rs * proj
         * Backward: grad_proj = grad_x * rs
         *           grad_W_o += grad_proj outer attn_out
         *           grad_attn_out = W_o^T @ grad_proj
         *           grad_x -= rs * grad_proj (undo residual, then add grad_attn_out path) */
        
        LayerGPU *go = (LayerGPU*)tl->attn_o._gpu;
        if (go && go->uploaded && tl->attn_o.grad_accum) {
            /* v13s: GPU accumulation */
            k_grad_W_bias<<<(n+thr-1)/thr, thr, 0, s>>>(
                tl->attn_o.d_grad_accum, tl->attn_o.d_bias_grad_accum,
                d_grad_x, a->d_ao, n, n);
            
            /* grad_attn_out = W_o^T @ grad_proj */
            k_matvec_T<<<(n+thr-1)/thr, thr, 0, s>>>(a->d_ao, go->d_w, d_grad_x, n, n);
        }
        
        /* V-copy backward: grad_v = grad_attn_out (V copy, no learnable params) */
        /* grad_qkv[2n:3n] = grad_attn_out */
        /* For QKV merged: grad_W_qkv += grad_qkv outer norm1 */
        LayerGPU *gq = (LayerGPU*)tl->attn_q._gpu;
        if (gq && gq->uploaded && tl->attn_q.grad_accum) {
            /* v13s: GPU accumulation — only V part (2n:3n) has gradient */
            /* Upload grad_qkv to device (only V part nonzero) */
            static float *d_gqkv = NULL;
            if (!d_gqkv) cudaMalloc(&d_gqkv, 3*n*sizeof(float));
            cudaMemsetAsync(d_gqkv, 0, 3*n*sizeof(float), s);
            cudaMemcpyAsync(d_gqkv + 2*n, a->d_ao, n*sizeof(float), cudaMemcpyDeviceToDevice, s);
            k_grad_W_bias<<<(3*n+thr-1)/thr, thr, 0, s>>>(
                tl->attn_q.d_grad_accum, tl->attn_q.d_bias_grad_accum,
                d_gqkv, a->d_n1, n, 3*n);
            
            /* grad_norm1 = W_q^T @ grad_qkv (reuse d_gqkv already on device) */
            k_matvec_T<<<(n+thr-1)/thr, thr, 0, s>>>(a->d_n1, gq->d_w, d_gqkv, n, 3*n);
        }
        
        /* v13t: Proper LayerNorm1 backward */
        static float *d_nw1_bwd = NULL;
        if (!d_nw1_bwd) cudaMalloc(&d_nw1_bwd, n*4);
        cudaMemcpyAsync(d_nw1_bwd, tl->norm1_w, n*4, cudaMemcpyHostToDevice, s);
        k_layernorm_backward<<<1, thr, 2*thr*sizeof(float), s>>>(
            a->d_n1, a->d_n1, a->d_x_pre, d_nw1_bwd, n);
        k_residual_clip<<<(n+thr-1)/thr, thr, 0, s>>>(d_grad_x, a->d_n1, 1.0f, 1e10f, n);
    }
    
    cudaStreamSynchronize(s);
}

/* ===================== v13r: GPU logic_reg =====================
 *
 * Replaces CPU logic_guided_regularization inner loop.
 * For each layer, for each concept pair (a,b):
 *   1. act_a = simulate_activation(gate_a)  → k_matvec on GPU
 *   2. act_b = simulate_activation(gate_b)  → k_matvec on GPU
 *   3. For each output j: compute diff, loss, gradient
 *   4. Accumulate gradient into grad_accum
 *
 * All concept pairs batched: act_a/b is [N_PAIRS, mlp_dim].
 */

/* GPU kernel: simulate_activation for all pairs at once.
 * act[pair, j] = bias[j] + sum_i W[j*in+i] * gate[pair, i]
 * W is [out, in] row-major, gate is [batch, in], act is [batch, out].
 * This is a batched matvec: act = gate @ W^T
 * Each thread computes one (pair, j) element. */
__global__ void k_sim_act_batch(float *act, const float *W, const float *bias,
                                  const float *gate, int in_dim, int out_dim,
                                  int batch, const uint8_t *mask) {
    int pair = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim || pair >= batch) return;
    if (mask && mask[j] == 2) {  /* PRUNE */
        act[pair * out_dim + j] = 0.0f;
        return;
    }
    float s = bias ? bias[j] : 0.0f;
    const float *wr = W + (size_t)j * in_dim;
    const float *g = gate + (size_t)pair * in_dim;
    for (int i = 0; i < in_dim; i++)
        s += wr[i] * g[i];
    act[pair * out_dim + j] = s;
}

/* GPU kernel: compute logic_reg loss + gradient accumulation.
 * For each (pair, j):
 *   CORE: loss -= alpha * tanh(|diff|*0.5) + gamma*(|act_a|+|act_b|)
 *         grad: -alpha*0.5*sign(diff)*(1-tanh²) + gamma*sign(act)
 *   BINARY: loss += beta * diff²
 *           grad: beta*2*diff
 * Accumulate grad into grad_accum[j*in + i] += grad_scale * (gate_a[i] - gate_b[i])
 *
 * This is the expensive part: for each j, loop over in_dim.
 * Thread per j, each thread loops over in_dim (512). */
__global__ void k_logic_grad_accum(
    float *grad_accum,       /* [out * in] — weight gradient */
    float *bias_grad_accum,  /* [out] — bias gradient */
    const float *act_a,      /* [batch * out] — activations for concept A */
    const float *act_b,      /* [batch * out] — activations for concept B */
    const float *gate_a,     /* [batch * in] — gate inputs for A */
    const float *gate_b,     /* [batch * in] — gate inputs for B */
    const uint8_t *mask,     /* [out] — logic mask */
    int in_dim, int out_dim, int batch,
    float alpha, float beta, float gamma,
    float lr, float inv_nc, float inv_nb, float layer_lr_scale
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    if (mask && mask[j] == 2) return;  /* PRUNE: skip */

    /* Aggregate loss + gradient across all pairs */
    float total_grad_a = 0.0f;  /* accumulated grad for gate_a direction */
    float total_grad_b = 0.0f;  /* accumulated grad for gate_b direction */

    for (int p = 0; p < batch; p++) {
        float aa = act_a[p * out_dim + j];
        float ab = act_b[p * out_dim + j];
        float diff = aa - ab;
        float adiff = fabsf(diff);

        if (mask[j] == 0) {  /* CORE */
            float s = diff > 0 ? 1.0f : -1.0f;
            float tanh_adiff = tanhf(adiff * 0.5f);
            float diff_grad = -alpha * 0.5f * s * (1.0f - tanh_adiff * tanh_adiff);
            float mag_grad_a = gamma * (aa > 0 ? 1.0f : -1.0f);
            float mag_grad_b = gamma * (ab > 0 ? 1.0f : -1.0f);
            total_grad_a += (diff_grad + mag_grad_a) * inv_nc * layer_lr_scale * lr;
            total_grad_b += mag_grad_b * inv_nc * layer_lr_scale * lr;
        } else {  /* BINARY */
            float grad_scale = beta * 2.0f * diff * inv_nb * layer_lr_scale * lr;
            total_grad_a += grad_scale;
            total_grad_b -= grad_scale;
        }
    }

    /* Accumulate: grad_accum[j*in + i] += total_grad_a * gate_a[p*in+i]
     *                                + total_grad_b * gate_b[p*in+i]
     * But gate_a/b differ per pair. Need to loop over pairs again. */
    float *ga = grad_accum + (size_t)j * in_dim;
    for (int p = 0; p < batch; p++) {
        const float *g_a = gate_a + (size_t)p * in_dim;
        const float *g_b = gate_b + (size_t)p * in_dim;
        for (int i = 0; i < in_dim; i++)
            ga[i] += total_grad_a * g_a[i] + total_grad_b * g_b[i];
    }

    if (bias_grad_accum)
        bias_grad_accum[j] += total_grad_a + total_grad_b;
}

/* v13r: GPU logic_guided_regularization.
 * Computes simulate_activation + grad_accum for all layers, all pairs, on GPU.
 * gate_a/b are [batch, n_embd] per layer (from CPU compute_all_gate_inputs_batch).
 * Uploads them to GPU, runs k_sim_act_batch + k_logic_grad_accum per layer. */
extern "C"
float lal_cuda_logic_reg(
    Model *m,
    const float *gate_a_host,  /* [n_layer * batch * n_embd] */
    const float *gate_b_host,  /* [n_layer * batch * n_embd] */
    float lr
) {
    int n = m->cfg.n_embd;
    int n_layer = m->cfg.n_layer;
    int mlp_dim = m->cfg.mlp_dim;
    int batch = 7;  /* N_PROBE_PAIRS */
    float alpha = 4.0f, beta = 0.2f, gamma = 0.3f;
    float total_loss = 0.0f;

    static cudaStream_t s = NULL;
    if (!s) cudaStreamCreate(&s);

    /* Persistent device buffers */
    static float *d_gate_a = NULL, *d_gate_b = NULL;
    static float *d_act_a = NULL, *d_act_b = NULL;
    static int d_cap = 0;
    int need = batch * (n > mlp_dim ? n : mlp_dim);
    if (need > d_cap) {
        if (d_gate_a) { cudaFree(d_gate_a); cudaFree(d_gate_b);
                        cudaFree(d_act_a); cudaFree(d_act_b); }
        cudaMalloc(&d_gate_a, batch * n * sizeof(float));
        cudaMalloc(&d_gate_b, batch * n * sizeof(float));
        cudaMalloc(&d_act_a, batch * mlp_dim * sizeof(float));
        cudaMalloc(&d_act_b, batch * mlp_dim * sizeof(float));
        d_cap = need;
    }

    int thr = 256;

    for (int l = 0; l < n_layer; l++) {
        BinLayer *fc = &m->layers[l].mlp_gate;
        if (!fc->logic_mask || !fc->_gpu) continue;

        LayerGPU *g = (LayerGPU*)fc->_gpu;
        int in_dim = fc->in_dim;
        int out_dim = fc->out_dim;

        /* Upload gate inputs for this layer */
        const float *layer_gate_a = gate_a_host + (size_t)l * batch * n;
        const float *layer_gate_b = gate_b_host + (size_t)l * batch * n;
        cudaMemcpyAsync(d_gate_a, layer_gate_a, batch * n * sizeof(float),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_gate_b, layer_gate_b, batch * n * sizeof(float),
                        cudaMemcpyHostToDevice, s);

        /* Upload logic_mask */
        static float *d_mask = NULL;
        if (!d_mask) cudaMalloc(&d_mask, out_dim * sizeof(uint8_t));
        cudaMemcpyAsync(d_mask, fc->logic_mask, out_dim * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, s);

        /* Compute activations: act = gate @ W^T (+ bias, with PRUNE zeroing) */
        dim3 grid_a((out_dim + thr - 1) / thr, batch);
        k_sim_act_batch<<<grid_a, thr, 0, s>>>(d_act_a, g->d_w, g->d_bias,
                                                d_gate_a, in_dim, out_dim, batch,
                                                (const uint8_t*)d_mask);
        k_sim_act_batch<<<grid_a, thr, 0, s>>>(d_act_b, g->d_w, g->d_bias,
                                                d_gate_b, in_dim, out_dim, batch,
                                                (const uint8_t*)d_mask);

        /* Download activations for loss computation (CPU side) */
        /* Actually: compute loss on GPU too. But loss needs reduction.
         * For simplicity: download act_a, act_b, compute loss on CPU. */
        static float *h_act_a = NULL, *h_act_b = NULL;
        if (!h_act_a) { h_act_a = (float*)malloc(batch * mlp_dim * sizeof(float));
                        h_act_b = (float*)malloc(batch * mlp_dim * sizeof(float)); }
        cudaMemcpyAsync(h_act_a, d_act_a, batch * mlp_dim * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(h_act_b, d_act_b, batch * mlp_dim * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        /* Compute loss + counts on CPU */
        int nc = 0, nb = 0;
        for (int p = 0; p < batch; p++) {
            for (int j = 0; j < out_dim; j++) {
                if (fc->logic_mask[j] == 2) continue;
                float diff = h_act_a[p*out_dim+j] - h_act_b[p*out_dim+j];
                float adiff = fabsf(diff);
                if (fc->logic_mask[j] == 0) {
                    total_loss -= alpha * tanhf(adiff * 0.5f);
                    total_loss += gamma * (fabsf(h_act_a[p*out_dim+j]) + fabsf(h_act_b[p*out_dim+j]));
                    nc++;
                } else {
                    total_loss += beta * diff * diff;
                    nb++;
                }
            }
        }

        float inv_nc = nc > 0 ? 1.0f / sqrtf((float)nc / batch) : 0;
        float inv_nb = nb > 0 ? 1.0f / sqrtf((float)nb / batch) : 0;
        float layer_lr_scale = (l == 0) ? 1.0f : 0.5f;

        /* v13s: Accumulate directly to device grad_accum (no H2D/D2H!) */
        k_logic_grad_accum<<<(out_dim + thr - 1) / thr, thr, 0, s>>>(
            fc->d_grad_accum, fc->d_bias_grad_accum,
            d_act_a, d_act_b,
            d_gate_a, d_gate_b,
            (const uint8_t*)d_mask,
            in_dim, out_dim, batch,
            alpha, beta, gamma,
            lr, inv_nc, inv_nb, layer_lr_scale
        );
    }

    return total_loss / (batch * n_layer);
}

