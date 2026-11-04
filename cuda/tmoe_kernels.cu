// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Ternary MoE kernels: int8 act quant, dp4a GEMV, tiled GEMM, grouped expert GEMM.
#include "tmoe_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#define TMOE_BLOCK 128

static char g_err[512] = "";

static int set_err(const char* what, cudaError_t e) {
  snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
  return (int)e ? (int)e : 1;
}
static int set_msg(const char* what) {
  snprintf(g_err, sizeof(g_err), "%s", what);
  return 1;
}
#define CK(call)                                        \
  do {                                                  \
    cudaError_t e_ = (call);                            \
    if (e_ != cudaSuccess) return set_err(#call, e_);   \
  } while (0)
#define CKL(name)                                       \
  do {                                                  \
    cudaError_t e_ = cudaGetLastError();                \
    if (e_ != cudaSuccess) return set_err(name, e_);    \
  } while (0)

// byte of four 2-bit codes -> int8x4 (-1/0/+1) packed little-endian in an int.
__device__ __forceinline__ int lut_entry(int c) {
  int r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int code = (c >> (2 * i)) & 3;
    int v = code == 1 ? 1 : (code == 2 ? -1 : 0);
    r |= (v & 0xFF) << (8 * i);
  }
  return r;
}

__device__ __forceinline__ void build_lut(int* lut) {
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = lut_entry(i);
}

__device__ __forceinline__ int dot4(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const signed char* pa = (const signed char*)&a;
  const signed char* pb = (const signed char*)&b;
  return c + pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
#endif
}

// ---------------------------------------------------------------- act quant
__global__ void __launch_bounds__(256) act_quant_kernel(const float* __restrict__ x, int cols,
                                                        const int* __restrict__ mask,
                                                        int8_t* __restrict__ q,
                                                        float* __restrict__ scales) {
  const int row = blockIdx.x;
  if (mask && mask[row] < 0) return;
  const float* xr = x + (size_t)row * cols;
  float amax = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) amax = fmaxf(amax, fabsf(xr[c]));
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  amax = red[0];
  float scale, inv;
  if (amax > 0.f) {
    scale = __fdiv_rn(amax, 127.f);
    inv = __fdiv_rn(127.f, amax);
  } else {
    scale = 0.f;
    inv = 0.f;
  }
  int8_t* qr = q + (size_t)row * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = rintf(__fmul_rn(xr[c], inv));
    v = fminf(fmaxf(v, -127.f), 127.f);
    qr[c] = (int8_t)(int)v;
  }
  if (threadIdx.x == 0) scales[row] = scale;
}

// ---------------------------------------------------------------- GEMV
// One warp per output row. Each lane takes 16 columns (one u32 of codes, one
// int4 of activations) per step; the int sum of each 128 block is scaled in fp32.
#define GEMV_WARPS 4
__global__ void __launch_bounds__(GEMV_WARPS * 32) ternary_gemv_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ ws, int n, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y) {
  __shared__ int lut[256];
  build_lut(lut);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
  if (row >= n) return;
  const int words = k >> 4;  // 16 columns per u32
  const int nb = k / TMOE_BLOCK;
  const uint32_t* wr = (const uint32_t*)(w + (size_t)row * (k >> 2));
  const int4* xv = (const int4*)xq;
  const float* sr = ws + (size_t)row * nb;
  float acc = 0.f;
  // 8 words per block, so lanes 8b..8b+7 of a 32-word sweep share block b.
  for (int base = 0; base < words; base += 32) {
    int wi = base + lane;
    int s = 0;
    if (wi < words) {
      uint32_t p = __ldg(wr + wi);
      int4 a = __ldg(xv + wi);
      s = dot4(lut[p & 0xFF], a.x, s);
      s = dot4(lut[(p >> 8) & 0xFF], a.y, s);
      s = dot4(lut[(p >> 16) & 0xFF], a.z, s);
      s = dot4(lut[p >> 24], a.w, s);
    }
    // reduce the 8 lanes of each block in integers, then scale once
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    if ((lane & 7) == 0 && wi < words) acc += (float)s * __ldg(sr + (wi >> 3));
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 8);
  acc += __shfl_xor_sync(0xffffffffu, acc, 16);
  if (lane == 0) y[row] = acc * xs[0];
}

// ---------------------------------------------------------------- tiled GEMM
// Tile: BM activation rows x BN weight rows, BK = 128 = one scale block, so
// the integer tile sum is scaled once per k-tile. 256 threads, 2x4 outputs each.
#define BM 32
#define BN 64
#define BKW (TMOE_BLOCK / 4)  // int8x4 words per k-tile row
#define PADW (BKW + 1)

// Grouped: blockIdx.z = expert; tile rows map through perm.
template <bool GROUPED>
__global__ void __launch_bounds__(256) ternary_gemm_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ ws, int n, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xs, int m, float* __restrict__ y,
    const int* __restrict__ counts, const int* __restrict__ offsets,
    const int* __restrict__ perm, int in_by_slot, int topk) {
  __shared__ int lut[256];
  __shared__ int sx[BM][PADW];
  __shared__ int sw[BN][PADW];
  __shared__ int in_row[BM];
  __shared__ int out_row[BM];

  const int tid = threadIdx.x;
  const int nb = k / TMOE_BLOCK;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  int rows_here;
  if (GROUPED) {
    const int e = blockIdx.z;
    const int cnt = counts[e];
    if (m0 >= cnt) return;
    rows_here = min(BM, cnt - m0);
    w += (size_t)e * n * (k >> 2);
    ws += (size_t)e * n * nb;
    if (tid < BM) {
      int s = tid < rows_here ? perm[offsets[e] + m0 + tid] : -1;
      out_row[tid] = s;
      in_row[tid] = s < 0 ? -1 : (in_by_slot ? s : s / topk);
    }
  } else {
    if (m0 >= m) return;
    rows_here = min(BM, m - m0);
    if (tid < BM) {
      int r = tid < rows_here ? m0 + tid : -1;
      out_row[tid] = r;
      in_row[tid] = r;
    }
  }
  build_lut(lut);
  __syncthreads();

  const int tx = tid & 15;  // n: tx, tx+16, tx+32, tx+48
  const int ty = tid >> 4;  // m: ty, ty+16
  float acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int kb = 0; kb < nb; ++kb) {
    // activations: BM x 128 bytes = 256 int4, one per thread
    {
      const int r = tid >> 3, c4 = tid & 7;  // c4: 16-byte chunk
      int4 v = make_int4(0, 0, 0, 0);
      const int ir = in_row[r];
      if (ir >= 0) v = __ldg((const int4*)(xq + (size_t)ir * k + kb * TMOE_BLOCK) + c4);
      sx[r][c4 * 4 + 0] = v.x;
      sx[r][c4 * 4 + 1] = v.y;
      sx[r][c4 * 4 + 2] = v.z;
      sx[r][c4 * 4 + 3] = v.w;
    }
    // weights: BN rows x 32 packed bytes = 512 u32, two per thread, 4 words out each
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int idx = tid + h * 256;
      const int r = idx >> 3, c = idx & 7;
      uint32_t p = 0;
      if (n0 + r < n)
        p = __ldg((const uint32_t*)(w + (size_t)(n0 + r) * (k >> 2) + kb * (TMOE_BLOCK / 4)) + c);
      sw[r][c * 4 + 0] = lut[p & 0xFF];
      sw[r][c * 4 + 1] = lut[(p >> 8) & 0xFF];
      sw[r][c * 4 + 2] = lut[(p >> 16) & 0xFF];
      sw[r][c * 4 + 3] = lut[p >> 24];
    }
    __syncthreads();
    int is[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) is[i][j] = 0;
#pragma unroll 8
    for (int kk = 0; kk < BKW; ++kk) {
      const int a0 = sx[ty][kk], a1 = sx[ty + 16][kk];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int b = sw[tx + 16 * j][kk];
        is[0][j] = dot4(a0, b, is[0][j]);
        is[1][j] = dot4(a1, b, is[1][j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + tx + 16 * j;
      const float sc = col < n ? __ldg(ws + (size_t)col * nb + kb) : 0.f;
      acc[0][j] += (float)is[0][j] * sc;
      acc[1][j] += (float)is[1][j] * sc;
    }
    __syncthreads();
  }
  // epilogue: per-token activation scale
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = ty + 16 * i;
    const int orow = out_row[r];
    if (orow < 0) continue;
    const float xsc = xs[in_row[r]];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + tx + 16 * j;
      if (col < n) y[(size_t)orow * n + col] = acc[i][j] * xsc;
    }
  }
}

// ---------------------------------------------------------------- dispatch
__global__ void hist_kernel(const int* __restrict__ se, int nslots, int* counts) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nslots && se[s] >= 0) atomicAdd(counts + se[s], 1);
}
// single block, experts <= 1024
__global__ void scan_kernel(const int* counts, int experts, int* offsets, int* cursor) {
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < experts; ++e) {
      offsets[e] = acc;
      cursor[e] = acc;
      acc += counts[e];
    }
    offsets[experts] = acc;
  }
}
__global__ void scatter_kernel(const int* __restrict__ se, int nslots, int* cursor, int* perm) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nslots && se[s] >= 0) perm[atomicAdd(cursor + se[s], 1)] = s;
}

// ---------------------------------------------------------------- SwiGLU + combine
__global__ void swiglu_kernel(float* __restrict__ g, const float* __restrict__ u,
                              const int* __restrict__ se, int nslots, int hidden) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)nslots * hidden) return;
  if (se[i / hidden] < 0) return;
  float a = g[i];
  float s = a < -80.f ? 0.f : a / (1.f + expf(-a));
  g[i] = s * u[i];
}
// fixed slot order, so the sum is deterministic
__global__ void combine_kernel(const float* __restrict__ ys, const int* __restrict__ se,
                               const float* __restrict__ gate, int tokens, int topk, int dim,
                               float* __restrict__ y) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)tokens * dim) return;
  const int t = (int)(i / dim), d = (int)(i % dim);
  float acc = 0.f;
  for (int j = 0; j < topk; ++j) {
    const int s = t * topk + j;
    if (se[s] >= 0) acc += gate[s] * ys[(size_t)s * dim + d];
  }
  y[i] += acc;
}

// ================================================================ C ABI
extern "C" {

int tmoe_cuda_device_count(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

int tmoe_cuda_device_name(int dev, char* buf, int len) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  snprintf(buf, len, "%s (sm_%d%d)", p.name, p.major, p.minor);
  return 0;
}

const char* tmoe_cuda_last_error(void) { return g_err; }
int tmoe_cuda_malloc(void** ptr, size_t bytes) { CK(cudaMalloc(ptr, bytes)); return 0; }
int tmoe_cuda_free(void* ptr) { CK(cudaFree(ptr)); return 0; }
int tmoe_cuda_h2d(void* d, const void* s, size_t b) { CK(cudaMemcpy(d, s, b, cudaMemcpyHostToDevice)); return 0; }
int tmoe_cuda_d2h(void* d, const void* s, size_t b) { CK(cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost)); return 0; }
int tmoe_cuda_sync(void) { CK(cudaDeviceSynchronize()); return 0; }

int tmoe_act_quant_int8(const float* x, int rows, int cols, const int* mask, int8_t* q, float* scales) {
  if (rows <= 0) return 0;
  act_quant_kernel<<<rows, 256>>>(x, cols, mask, q, scales);
  CKL("act_quant_kernel");
  return 0;
}

int tmoe_ternary_gemv(const uint8_t* w, const float* ws, int n, int k, const int8_t* xq,
                      const float* xs, float* y) {
  if (k % TMOE_BLOCK) return set_msg("k must be a multiple of 128");
  ternary_gemv_kernel<<<(n + GEMV_WARPS - 1) / GEMV_WARPS, GEMV_WARPS * 32>>>(w, ws, n, k, xq, xs, y);
  CKL("ternary_gemv_kernel");
  return 0;
}

int tmoe_ternary_gemm(const uint8_t* w, const float* ws, int n, int k, const int8_t* xq,
                      const float* xs, int m, float* y) {
  if (k % TMOE_BLOCK) return set_msg("k must be a multiple of 128");
  if (m <= 0) return 0;
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, 1);
  ternary_gemm_kernel<false><<<grid, 256>>>(w, ws, n, k, xq, xs, m, y, nullptr, nullptr, nullptr, 0, 1);
  CKL("ternary_gemm_kernel");
  return 0;
}

int tmoe_moe_dispatch(const int* se, int nslots, int experts, int* counts, int* offsets, int* perm) {
  if (experts > 1024) return set_msg("too many experts");
  // cursor lives after offsets[experts]
  int* cursor = offsets + experts + 1;
  CK(cudaMemset(counts, 0, sizeof(int) * experts));
  const int th = 256, bl = (nslots + th - 1) / th;
  if (bl > 0) hist_kernel<<<bl, th>>>(se, nslots, counts);
  scan_kernel<<<1, 32>>>(counts, experts, offsets, cursor);
  if (bl > 0) scatter_kernel<<<bl, th>>>(se, nslots, cursor, perm);
  CKL("moe dispatch");
  return 0;
}

int tmoe_moe_grouped_gemm(const uint8_t* w, const float* ws, int experts, int n, int k,
                          const int8_t* xq, const float* xs, int in_by_slot, int topk,
                          const int* counts, const int* offsets, const int* perm, int nslots,
                          float* y_slot) {
  if (k % TMOE_BLOCK) return set_msg("k must be a multiple of 128");
  if (nslots <= 0) return 0;
  // worst case all slots on one expert; empty tiles return at once
  dim3 grid((n + BN - 1) / BN, (nslots + BM - 1) / BM, experts);
  ternary_gemm_kernel<true><<<grid, 256>>>(w, ws, n, k, xq, xs, 0, y_slot, counts, offsets, perm,
                                           in_by_slot, topk);
  CKL("grouped ternary_gemm_kernel");
  return 0;
}

static size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

struct MoeWs {
  int8_t* xq; float* xs; int* counts; int* offsets; int* perm;
  float* g; float* u; int8_t* hq; float* hs; float* ys;
};

static size_t moe_layout(int T, int D, int H, int E, int K, char* base, MoeWs* o) {
  const size_t S = (size_t)T * K;
  size_t off = 0;
  auto take = [&](size_t bytes) { char* p = base ? base + off : nullptr; off += align256(bytes); return p; };
  MoeWs w;
  w.xq = (int8_t*)take((size_t)T * D);
  w.xs = (float*)take(sizeof(float) * T);
  w.counts = (int*)take(sizeof(int) * E);
  w.offsets = (int*)take(sizeof(int) * (2 * E + 1));
  w.perm = (int*)take(sizeof(int) * S);
  w.g = (float*)take(sizeof(float) * S * H);
  w.u = (float*)take(sizeof(float) * S * H);
  w.hq = (int8_t*)take(S * H);
  w.hs = (float*)take(sizeof(float) * S);
  w.ys = (float*)take(sizeof(float) * S * D);
  if (o) *o = w;
  return off;
}

size_t tmoe_moe_workspace_bytes(int tokens, int dim, int hidden, int experts, int topk) {
  return moe_layout(tokens, dim, hidden, experts, topk, nullptr, nullptr);
}

int tmoe_moe_routed_forward(const float* x, int T, int D, int H, int E, int K,
                            const int* se, const float* gate,
                            const uint8_t* wg, const float* sg, const uint8_t* wu, const float* su,
                            const uint8_t* wd, const float* sd, float* y, void* workspace) {
  if (T <= 0) return 0;
  MoeWs w;
  moe_layout(T, D, H, E, K, (char*)workspace, &w);
  const int S = T * K;
  int r;
  if ((r = tmoe_act_quant_int8(x, T, D, nullptr, w.xq, w.xs))) return r;
  if ((r = tmoe_moe_dispatch(se, S, E, w.counts, w.offsets, w.perm))) return r;
  if ((r = tmoe_moe_grouped_gemm(wg, sg, E, H, D, w.xq, w.xs, 0, K, w.counts, w.offsets, w.perm, S, w.g))) return r;
  if ((r = tmoe_moe_grouped_gemm(wu, su, E, H, D, w.xq, w.xs, 0, K, w.counts, w.offsets, w.perm, S, w.u))) return r;
  {
    size_t n = (size_t)S * H;
    swiglu_kernel<<<(unsigned)((n + 255) / 256), 256>>>(w.g, w.u, se, S, H);
    CKL("swiglu_kernel");
  }
  if ((r = tmoe_act_quant_int8(w.g, S, H, se, w.hq, w.hs))) return r;
  if ((r = tmoe_moe_grouped_gemm(wd, sd, E, D, H, w.hq, w.hs, 1, K, w.counts, w.offsets, w.perm, S, w.ys))) return r;
  {
    size_t n = (size_t)T * D;
    combine_kernel<<<(unsigned)((n + 255) / 256), 256>>>(w.ys, se, gate, T, K, D, y);
    CKL("combine_kernel");
  }
  return 0;
}

}  // extern "C"

// ---------------------------------------------------------------- host wrappers
struct DevBuf {
  std::vector<void*> ptrs;
  ~DevBuf() { for (void* p : ptrs) cudaFree(p); }
  template <class T> T* up(const T* h, size_t count) {
    void* d = nullptr;
    if (cudaMalloc(&d, count * sizeof(T) + 16) != cudaSuccess) return nullptr;
    ptrs.push_back(d);
    if (h && cudaMemcpy(d, h, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return (T*)d;
  }
  template <class T> T* zeros(size_t count) {
    T* d = up<T>(nullptr, count);
    if (d && cudaMemset(d, 0, count * sizeof(T) + 16) != cudaSuccess) return nullptr;
    return d;
  }
};
#define NN(p) do { if (!(p)) return set_err("device alloc/copy", cudaGetLastError()); } while (0)

extern "C" {

int tmoe_h_act_quant_int8(const float* x, int rows, int cols, int8_t* q, float* scales) {
  DevBuf b;
  const float* dx = b.up(x, (size_t)rows * cols); NN(dx);
  int8_t* dq = b.zeros<int8_t>((size_t)rows * cols); NN(dq);
  float* ds = b.zeros<float>(rows); NN(ds);
  int r = tmoe_act_quant_int8(dx, rows, cols, nullptr, dq, ds);
  if (r) return r;
  CK(cudaMemcpy(q, dq, (size_t)rows * cols, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(scales, ds, sizeof(float) * rows, cudaMemcpyDeviceToHost));
  return 0;
}

int tmoe_h_ternary_gemv(const uint8_t* w, const float* ws, int n, int k, const int8_t* xq,
                        float xs, float* y) {
  DevBuf b;
  const uint8_t* dw = b.up(w, (size_t)n * k / 4); NN(dw);
  const float* dws = b.up(ws, (size_t)n * (k / TMOE_BLOCK)); NN(dws);
  const int8_t* dx = b.up(xq, (size_t)k); NN(dx);
  const float* dxs = b.up(&xs, 1); NN(dxs);
  float* dy = b.zeros<float>(n); NN(dy);
  int r = tmoe_ternary_gemv(dw, dws, n, k, dx, dxs, dy);
  if (r) return r;
  CK(cudaMemcpy(y, dy, sizeof(float) * n, cudaMemcpyDeviceToHost));
  return 0;
}

int tmoe_h_ternary_gemm(const uint8_t* w, const float* ws, int n, int k, const int8_t* xq,
                        const float* xs, int m, float* y) {
  DevBuf b;
  const uint8_t* dw = b.up(w, (size_t)n * k / 4); NN(dw);
  const float* dws = b.up(ws, (size_t)n * (k / TMOE_BLOCK)); NN(dws);
  const int8_t* dx = b.up(xq, (size_t)m * k); NN(dx);
  const float* dxs = b.up(xs, m); NN(dxs);
  float* dy = b.zeros<float>((size_t)m * n); NN(dy);
  int r = tmoe_ternary_gemm(dw, dws, n, k, dx, dxs, m, dy);
  if (r) return r;
  CK(cudaMemcpy(y, dy, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
  return 0;
}

int tmoe_h_moe_routed_forward(const float* x, int T, int D, int H, int E, int K,
                              const int* se, const float* gate,
                              const uint8_t* wg, const float* sg, const uint8_t* wu, const float* su,
                              const uint8_t* wd, const float* sd, float* y) {
  DevBuf b;
  const size_t S = (size_t)T * K;
  const float* dx = b.up(x, (size_t)T * D); NN(dx);
  const int* dse = b.up(se, S); NN(dse);
  const float* dg = b.up(gate, S); NN(dg);
  const uint8_t* dwg = b.up(wg, (size_t)E * H * D / 4); NN(dwg);
  const float* dsg = b.up(sg, (size_t)E * H * (D / TMOE_BLOCK)); NN(dsg);
  const uint8_t* dwu = b.up(wu, (size_t)E * H * D / 4); NN(dwu);
  const float* dsu = b.up(su, (size_t)E * H * (D / TMOE_BLOCK)); NN(dsu);
  const uint8_t* dwd = b.up(wd, (size_t)E * D * H / 4); NN(dwd);
  const float* dsd = b.up(sd, (size_t)E * D * (H / TMOE_BLOCK)); NN(dsd);
  float* dy = b.up(y, (size_t)T * D); NN(dy);
  char* ws = b.zeros<char>(tmoe_moe_workspace_bytes(T, D, H, E, K)); NN(ws);
  int r = tmoe_moe_routed_forward(dx, T, D, H, E, K, dse, dg, dwg, dsg, dwu, dsu, dwd, dsd, dy, ws);
  if (r) return r;
  CK(cudaMemcpy(y, dy, sizeof(float) * T * D, cudaMemcpyDeviceToHost));
  return 0;
}

int tmoe_h_bench_gemv(int n, int k, int iters, double* us) {
  if (k % TMOE_BLOCK) return set_msg("k must be a multiple of 128");
  DevBuf b;
  std::vector<uint8_t> w((size_t)n * k / 4);
  uint32_t s = 12345u;
  for (auto& v : w) { s = s * 1664525u + 1013904223u; v = (uint8_t)(s >> 24) & 0x55; }
  std::vector<float> ws((size_t)n * (k / TMOE_BLOCK), 0.01f);
  std::vector<int8_t> x(k);
  for (auto& v : x) { s = s * 1664525u + 1013904223u; v = (int8_t)((int)(s >> 25) - 64); }
  float xs = 0.02f;
  const uint8_t* dw = b.up(w.data(), w.size()); NN(dw);
  const float* dws = b.up(ws.data(), ws.size()); NN(dws);
  const int8_t* dx = b.up(x.data(), x.size()); NN(dx);
  const float* dxs = b.up(&xs, 1); NN(dxs);
  float* dy = b.zeros<float>(n); NN(dy);
  int r = tmoe_ternary_gemv(dw, dws, n, k, dx, dxs, dy);
  if (r) return r;
  cudaEvent_t a, z;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&z));
  CK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) ternary_gemv_kernel<<<(n + GEMV_WARPS - 1) / GEMV_WARPS, GEMV_WARPS * 32>>>(dw, dws, n, k, dx, dxs, dy);
  CK(cudaEventRecord(z));
  CK(cudaEventSynchronize(z));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, z));
  cudaEventDestroy(a);
  cudaEventDestroy(z);
  *us = 1000.0 * ms / (iters > 0 ? iters : 1);
  return 0;
}

}  // extern "C"
