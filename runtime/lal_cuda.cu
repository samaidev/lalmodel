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

    /* v13k: persistent x/y buffers to avoid per-call cudaMalloc/Free.
     * Max dim is n_embd=512 or mlp_dim=1792, so allocate once for max size. */
    static float *d_x = NULL, *d_y = NULL;
    static int d_cap = 0;
    int need = in > out ? in : out;
    if (need > d_cap) {
        if (d_x) { cudaFree(d_x); cudaFree(d_y); }
        cudaMalloc(&d_x, need * sizeof(float));
        cudaMalloc(&d_y, need * sizeof(float));
        d_cap = need;
    }

    CUDA_CHECK(cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice));

    /* y = W @ x + bias, beta=1 adds bias pre-filled */
    if (g->d_bias) {
        CUDA_CHECK(cudaMemcpy(d_y, g->d_bias, out * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        float alpha = 1.0f, beta = 1.0f;
        cublasSgemv(h, CUBLAS_OP_T, in, out, &alpha, g->d_w, in, d_x, 1,
                    &beta, d_y, 1);
    } else {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(h, CUBLAS_OP_T, in, out, &alpha, g->d_w, in, d_x, 1,
                    &beta, d_y, 1);
    }

    /* Zero out PRUNE rows if logic_mask present */
    if (g->d_mask) {
        int thr = 256, blk = (out + thr - 1) / thr;
        k_zero_prune<<<blk, thr>>>(d_y, g->d_mask, out);
    }

    CUDA_CHECK(cudaMemcpy(y, d_y, out * sizeof(float), cudaMemcpyDeviceToHost));
    /* v13k: don't free d_x/d_y — they're persistent static buffers */
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

    /* v13k: reuse forward's persistent buffers (d_x as d_gy, d_y as d_gx) */
    static float *d_gy = NULL, *d_gx = NULL;
    static int d_cap_bwd = 0;
    int need = in > out ? in : out;
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

    float alpha = 1.0f, beta = 0.0f;
    /* grad_x = W_cm @ grad_y, W_cm is [in, out] col-major == W row-major [out, in] */
    cublasSgemv(h, CUBLAS_OP_N, in, out, &alpha, g->d_w, in, d_gy, 1,
                &beta, d_gx, 1);

    CUDA_CHECK(cudaMemcpy(grad_x, d_gx, in * sizeof(float), cudaMemcpyDeviceToHost));
    /* v13k: don't free — persistent buffers */
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

    /* Upload embeddings + wpe to GPU */
    cudaMemcpyAsync(d_x, embs, (size_t)batch * n * sizeof(float),
                    cudaMemcpyHostToDevice, s_stream);
    if (m->wpe) {
        /* Add wpe[0] to each row — use a kernel */
        k_add_wpe<<<(need + 255) / 256, 256, 0, s_stream>>>(d_x, m->wpe, batch, n);
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
