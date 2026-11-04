// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// C ABI of libtmoe_cuda. Bound from Pascal in src/tmoe_cuda.pas.
//
// Weight layout (same as src/tmoe_ternary.pas): row-major, K/4 bytes per row,
// 2-bit codes 00=0 01=+1 10=-1, column 4j+i in bits 2i..2i+1 of byte j,
// one float scale per 128 columns. K must be a multiple of 128.
// Every function returns 0 on success; tmoe_cuda_last_error() explains a
// non-zero result. Device-pointer functions run on the default stream.
#ifndef TMOE_CUDA_H
#define TMOE_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tmoe_cuda_device_count(void);
int tmoe_cuda_device_name(int dev, char* buf, int len);
const char* tmoe_cuda_last_error(void);
int tmoe_cuda_malloc(void** ptr, size_t bytes);
int tmoe_cuda_free(void* ptr);
int tmoe_cuda_h2d(void* dst, const void* src, size_t bytes);
int tmoe_cuda_d2h(void* dst, const void* src, size_t bytes);
int tmoe_cuda_sync(void);

// ---- device pointers ----
// q[r,c] = rint(x[r,c] * 127/amax_r), scales[r] = amax_r/127. mask may be NULL;
// rows with mask[r] < 0 are skipped.
int tmoe_act_quant_int8(const float* x, int rows, int cols, const int* mask,
                        int8_t* q, float* scales);
// Decode GEMV, batch 1: y[n] = xs[0] * sum_b ws[n,b] * dot4a(w[n,b], xq[b]).
int tmoe_ternary_gemv(const uint8_t* w, const float* ws, int n, int k,
                      const int8_t* xq, const float* xs, float* y);
// Prefill GEMM: y[m,n] = xs[m] * sum_b ws[n,b] * dot(w[n,b], xq[m,b]); y is M x N.
int tmoe_ternary_gemm(const uint8_t* w, const float* ws, int n, int k,
                      const int8_t* xq, const float* xs, int m, float* y);
// Router dispatch: slot_expert has nslots entries (-1 = dropped). Produces
// per-expert counts and offsets (E+1) and perm (slots grouped by expert).
int tmoe_moe_dispatch(const int* slot_expert, int nslots, int experts,
                      int* counts, int* offsets, int* perm);
// Expert-batched GEMM over E stacked matrices (E x N x K). Input row of slot s
// is s (in_by_slot=1) or s/topk (=token). y_slot is nslots x N.
int tmoe_moe_grouped_gemm(const uint8_t* w, const float* ws, int experts, int n, int k,
                          const int8_t* xq, const float* xs, int in_by_slot, int topk,
                          const int* counts, const int* offsets, const int* perm,
                          int nslots, float* y_slot);
// Whole routed-expert path of one MoE layer (SwiGLU, int8 acts), adding
// sum_j gate * expert(x) into y (T x D). Workspace size from the query below.
size_t tmoe_moe_workspace_bytes(int tokens, int dim, int hidden, int experts, int topk);
int tmoe_moe_routed_forward(const float* x, int tokens, int dim, int hidden, int experts, int topk,
                            const int* slot_expert, const float* slot_gate,
                            const uint8_t* wg, const float* sg,
                            const uint8_t* wu, const float* su,
                            const uint8_t* wd, const float* sd,
                            float* y, void* workspace);

// ---- host pointers (allocate, copy, run, copy back; for tests and tools) ----
int tmoe_h_act_quant_int8(const float* x, int rows, int cols, int8_t* q, float* scales);
int tmoe_h_ternary_gemv(const uint8_t* w, const float* ws, int n, int k,
                        const int8_t* xq, float xs, float* y);
int tmoe_h_ternary_gemm(const uint8_t* w, const float* ws, int n, int k,
                        const int8_t* xq, const float* xs, int m, float* y);
int tmoe_h_moe_routed_forward(const float* x, int tokens, int dim, int hidden, int experts, int topk,
                              const int* slot_expert, const float* slot_gate,
                              const uint8_t* wg, const float* sg,
                              const uint8_t* wu, const float* su,
                              const uint8_t* wd, const float* sd, float* y);
// Times `iters` decode GEMVs on resident weights; returns microseconds per call.
int tmoe_h_bench_gemv(int n, int k, int iters, double* us_per_call);

#ifdef __cplusplus
}
#endif
#endif
