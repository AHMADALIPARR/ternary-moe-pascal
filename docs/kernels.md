# CUDA kernels

All kernels are in `cuda/tmoe_kernels.cu` and exported through the C header
`cuda/tmoe_cuda.h`. The library links the CUDA runtime statically, so
`libtmoe_cuda.so` needs only the driver at run time. Pascal loads it with
`dlopen` (`src/tmoe_cuda.pas`); if the file or a device is missing, the
CPU path is used and `test_gpu` skips.

Search order for the library: `$TMOE_CUDA_LIB`, then `libtmoe_cuda.so` next to
the executable, `../libtmoe_cuda.so`, `../build/libtmoe_cuda.so`, then the
normal loader path.

## Byte to int8x4 table

Every kernel that reads packed weights first builds a 256-entry table in shared
memory: entry `b` is the four codes of byte `b` as signed bytes packed in one
`int`. A packed weight byte becomes a `dp4a` operand with one shared-memory read.

## `ternary_gemv_kernel` (decode)

One warp per output row, four warps per block. Each lane reads one `u32` of
packed codes (16 weights) and the matching `int4` of activations (16 int8), turns
the four code bytes into four int8x4 words and does four `dp4a`. Eight lanes
cover one 128-column block; an 8-lane shuffle reduction gives the exact int32
block sum, which is multiplied by the block scale and accumulated. The
activation scale is applied once at the end. Memory traffic per row is the
packed weights (`k/4` bytes) plus scales (`k/32` bytes).

## `ternary_gemm_kernel` (prefill and experts)

Output tile 32 tokens x 64 rows, k-tile 128 (exactly one scale block), 256
threads, each thread computing 2 x 4 outputs. Per k-tile the block loads
32 x 128 activation bytes as 16-byte `int4`s (one per thread) and 64 x 32
packed weight bytes as 16-byte `uint4`s (two per row, threads 0..127). Each
row's 32 packed bytes fill one 32-byte sector. Rows are `K/4` bytes apart, so
a warp does not merge them into one transaction unless `K = 128`. The weights
are expanded through the LUT into a padded `[64][33]` shared array (the extra
column avoids bank conflicts on the `dp4a` reads), the block runs 32 `dp4a`
steps per output and folds the int32 result into fp32 with the block scale.
The token scale is applied in the epilogue. Tail rows and columns are
predicated: invalid weight rows are not loaded.

`ternary_gemm_naive_kernel` is the single-launch baseline: one warp per output
row per token, same int32 block sum, no cross-token shared-memory reuse.
`tmoe_h_bench_gemm` times both with CUDA events. Neither kernel calls cuBLAS.

Tile sizes, bank map, issued bytes and the Orin parameter table are in
[orin-gemm.md](orin-gemm.md). That document does not report measured cycles.

The same kernel, compiled with `GROUPED = true`, is the grouped expert GEMM:
`blockIdx.z` selects the expert, tile rows are read through the permutation
produced by the dispatch step, and results are scattered back to slot order.

## Routed MoE path

`tmoe_moe_routed_forward` runs a whole routed layer from a per-slot expert list
(`-1` for a dropped slot) and gates:

1. `hist_kernel` counts slots per expert, `scan_kernel` makes offsets,
   `scatter_kernel` writes the slot permutation (`tmoe_moe_dispatch`).
2. `act_quant_kernel` quantises the token rows to INT8.
3. Grouped GEMM for gate and up (input row = slot / top-k).
4. `swiglu_kernel`, then `act_quant_kernel` on the hidden rows (masked to kept slots).
5. Grouped GEMM for down (input row = slot).
6. `combine_kernel` adds `gate * expert_out` per token in slot order, so the
   result does not depend on the scatter order.

The workspace size comes from `tmoe_moe_workspace_bytes`.

`MoeForward` keeps that math on the CPU unless `MoeSetDevice(True)` was
called. `tmoe_generate` and `tmoe_bench` do so unless `--cpu`. The device
path uploads the layer's packed experts once (`tmoe_resident_create`) and
adds the routed sum into `y`, which already holds the shared expert. A
missing library or a failed upload or launch falls back to the CPU loop.
`ActQuant` off stays on the CPU: the GPU path always quantises to int8.

## INT8 activation quantisation

`act_quant_kernel`: one block per row, block-wide max reduction, then
`rintf(__fmul_rn(x, __fdiv_rn(127, amax)))` clamped to ±127, matching the CPU
rounding.

## Host wrappers

`tmoe_h_*` functions take host pointers, allocate, copy, run and copy back.
They exist for tests and simple tools; `tmoe_h_bench_gemv` times the GEMV with
CUDA events on random packed weights.

## Build

```sh
make lib                                   # sm_80 sm_86 sm_87 sm_89 sm_90 + compute_90 PTX
make lib CUDA_ARCHS=87                     # Jetson Orin
make lib CUDA_ARCHS="86 89" NVCC=/usr/local/cuda/bin/nvcc
```

`ptxas -v` output is written to `build/ptxas.log`; `make lib` prints the
spill/stack summary.

## ptxas report

The table below is the nvcc 12.9 report for the kernels as they stood before
the 16-byte weight load. It was not regenerated for the current source: this
tree was built without nvcc. Run `make lib` and read `build/ptxas.log` before
treating register counts as current. Do not use them for occupancy until then.

Historical nvcc 12.9 note: every kernel on every architecture in that log
reported `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.

| Kernel | sm_80 | sm_86 | sm_87 | sm_89 | sm_90 | shared mem |
|---|---|---|---|---|---|---|
| ternary_gemm (grouped) | 63 | 63 | 63 | 63 | 64 | 13,952 B |
| ternary_gemm (dense) | 40 | 40 | 40 | 40 | 40 | 13,952 B |
| ternary_gemv | 29 | 35 | 35 | 35 | 29 | 1,024 B |
| act_quant | 17 | 19 | 19 | 19 | 19 | 128 B |
| combine | 28 | 28 | 28 | 28 | 32 | |
| swiglu | 19 | 18 | 18 | 18 | 18 | |
| scan | 22 | 22 | 22 | 22 | 22 | |
| scatter | 12 | 12 | 12 | 12 | 12 | |
| hist | 8 | 8 | 8 | 8 | 10 | |

Values are registers per thread.
