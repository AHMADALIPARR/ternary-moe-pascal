# Packed ternary GEMM on Jetson Orin (sm_87)

This is a static account of the kernel in `cuda/tmoe_kernels.cu`. It is not a
profile. No Orin, no other CUDA device, and no nvcc were available when it
was written, so there are no measured cycles, no achieved bandwidth, and no
speedup.

"Pascal" in the repository name is Free Pascal. The CUDA gencode list is
`sm_80 sm_86 sm_87 sm_89 sm_90` (`Makefile`). NVIDIA Pascal (sm_61) is not a
target. `__dp4a` has a scalar fallback for `__CUDA_ARCH__ < 610` so an
experimental older arch still compiles; that path is not shipped.

## What sm_87 actually is

Jetson AGX Orin, Orin NX and Orin Nano are all compute capability 8.7
(Ampere). They are not one GPU. SM count, DRAM width, clocks and power are
SKU- and nvpmodel-dependent. Do not pick "AGX" when the board is NX or Nano.

Cited from NVIDIA, not measured here:

* **AGX Orin technical brief** (GPU section): up to 2 GPCs, up to 8 TPCs, up
  to 16 SMs, 128 CUDA cores per SM, four 3rd-generation Tensor Cores per SM,
  192 KB L1 per SM, 4 MB L2. AGX 64GB is 2048 CUDA cores and 64 Tensor Cores
  (16 SMs). AGX 32GB in that brief is 1792 CUDA cores and 56 Tensor Cores
  (14 SMs). The brief's "up to 5.3 FP32 TFLOPs" for the 64GB module matches
  `2048 cores * 2 * 1.3 GHz`. That is an FP32 FMA peak at the quoted clock,
  not a dp4a rate and not a guaranteed clock.
* **CUDA Programming Guide**, compute-capability tables (the 8.7 column):
  64K 32-bit registers per SM, 255 per thread, 32 shared-memory banks,
  164 KB shared memory per SM, 163 KB per block, unified data cache 192 KB,
  carveouts `0, 8, 16, 32, 64, 100, 132, 164` KB. Static shared memory above
  48 KB needs dynamic allocation and an opt-in; this kernel stays under that
  (13952 bytes). Tensor-core input types on 8.7 include FP16, BF16, TF32,
  INT8 and INT4. There is no ternary type. The rendered table leaves the
  8.7 cells for resident warps, resident threads and resident blocks blank
  (they may be a colspan with 8.6, which would be 48 warps / 1536 threads /
  16 blocks, but the extract does not prove the colspan). Read
  `maxThreadsPerMultiProcessor` from the device. `tmoe_cuda_device_info`
  prints that, plus the register file, shared memory, L2, bus width and the
  **current** clocks.
* **Jetson Linux Developer Guide r36.4**, nvpmodel tables. "GPU TPC" changes
  with the power mode. The brief's 8 TPC / 16 SM relationship is 2 SMs per
  TPC, so SM count is `2 * GPU TPC` for these modules. Examples from that
  table, AGX Orin 64GB: MAXN is 8 TPC and 1301 MHz GPU; 15W is 3 TPC and
  408 MHz; the default 30W mode is 4 TPC and 612 MHz; 50W is 8 TPC and
  816 MHz. Memory max frequency is 3200 MHz in 30W/50W/MAXN and 2133 MHz in
  15W. MAXN is described as unconstrained and not the best-performance mode,
  because the module can throttle. NX and Nano rows are different again
  (NX 16GB MAXN is 4 TPC / 918 MHz; its 15W default is 2 TPC / 612 MHz).
* **Product-page peaks**, full DRAM rate, not a power-mode guarantee: AGX
  256-bit LPDDR5 204.8 GB/s; Orin NX 128-bit 102.4 GB/s. Nano figures on the
  product page (about 102 GB/s and 51 GB/s) match a higher memory clock than
  the r36.4 15W/10W rows. `tmoe_cost` prints both the nvpmodel inputs and an
  EMC estimate.

Estimated ceiling at a mode's max EMC, not a measured bandwidth:

```
GB/s ≈ (bus_bits / 8) * 2 * mem_MHz / 1000
```

At 256-bit and 3200 MHz that is 204.8 GB/s, matching the AGX brief. At
2133 MHz it is about 136.5 GB/s. DVFS can sit below the cap. `tegrastats`
and the EMC debugfs node are the measurement.

## Why dp4a, not Tensor Cores

A weight is a 2-bit code, `00 = 0`, `01 = +1`, `10 = -1`, `11` reads as 0,
with one fp32 scale per 128 columns applied **after** the int32 sum of that
block. The CPU reference checks that sum exactly.

sm_87 Tensor Cores can do INT8 and INT4 MMA (programming-guide table). Both
are wider than 2 bits:

* INT8 IMMA wants int8 operands. Expanding the packed matrix to int8 before
  the load moves 4× the weight bytes. Expanding after a packed load is
  possible, but the MMA fragment (`ldmatrix` / `mma.sync`) is a different
  layout from the byte we already decode, and the scale still has to be
  applied on the 128-column boundary, outside the MMA.
* Signed INT4 can hold −1, 0 and +1, but the MMA register holds 4 bits per
  weight, twice the packed payload. Same story: it does not reduce the
  packed global load, and it does not match the CPU's "one byte → four
  dp4a inputs" decode.
* 1-bit BMMA has no distinct zero. A ternary zero must contribute nothing.
* FP16 / BF16 / TF32 would dequantize first, expand the traffic further, and
  drop the exact int32 block sum.

`__dp4a` is a CUDA-core instruction (sm_61 and later). One packed byte is
four codes and one `dp4a` consumes four int8s, which is what the 256-entry
LUT produces. The kernel emits no `mma.sync`. The Orin brief's FP32 TFLOP
figure is not used as a dp4a throughput; that throughput is not stated there,
so this account does not divide an instruction count by a made-up rate and
call the result cycles.

## Global layout and coalescing

Unchanged from `src/tmoe_ternary.pas` and `cuda/tmoe_cuda.h`:

* Row-major, `K/4` bytes per row. `K` is a multiple of 128, so a row is a
  multiple of 32 bytes.
* Column `4j+i` is in bits `2i .. 2i+1` of byte `j`.
* Scales are a separate array, one fp32 per row per 128 columns.
* `cudaMalloc` is 256-byte aligned. A k-tile starts at byte `kb * 32` in the
  row, so the 16-byte loads are aligned.

A warp wants 128 consecutive bytes for one fully coalesced transaction.
Along a row, one 128-column tile is only 32 packed bytes: one 32-byte
sector, fully used, not a 128-byte line. The next 32 bytes of the same row
are the next k-tile (`K` columns are contiguous), but the scale changes
every 128 columns, so the kernel keeps the k-tile at 128 and applies the
scale before mixing tiles. Consecutive rows are `K/4` bytes apart. The
weight load uses 128 threads, two 16-byte loads per row. A warp therefore
covers 16 rows × 32 bytes, as 16 separate 32-byte segments whenever
`K > 128`. That is the coalescing limit of this layout. Changing the
on-disk order would break the CPU packer and the checkpoints; it was not
changed.

Activations are int8, row-major, `K` a multiple of 128. The dense GEMM loads
each token row as eight 16-byte chunks, 128 contiguous bytes, 128-byte
aligned if the base is. Grouped expert GEMM gathers token rows through the
dispatch permutation, so different rows are not adjacent. Each row is still
a contiguous 128-byte run. Dropped slots are not in the permutation and are
not loaded.

Invalid tile rows (`n0 + r >= N`, or a grouped row past the expert's count)
store zeros and do not issue the global load.

## Shared memory, banks, registers

```
lut[256]           1024 B
sx[32][33]         4224 B     activations as int8x4 words, pad 33
sw[64][33]         8448 B     decoded weights, pad 33
in_row[32], out_row[32]  256 B
total              13952 B
```

32 banks, 4-byte words (programming guide). Index
`(row * 33 + kk) mod 32 = (row + kk) mod 32` because 33 ≡ 1 (mod 32).

Warp mapping, 256 threads, `tid` in `0..255`:

* `tx = tid & 15`, `ty = tid >> 4` (`ty` is 0..15).
* The thread owns output rows `ty` and `ty+16`, and columns
  `n0 + tx + 16*j` for `j = 0..3`. That is 2×4 register outputs, 8 floats.
* A warp is 32 consecutive tids, so two groups of 16 threads with `tx = 0..15`
  and neighbouring `ty`. Both groups read the same `sw[tx + 16*j][kk]`
  (broadcast) and the same `sx[ty][kk]` inside the group of 16 (broadcast).
  Sixteen consecutive rows at a fixed `kk` hit sixteen consecutive banks.
  No extra serialization is implied by that map. A conflict would show up in
  a shared-memory bank-conflict counter; none was collected here.

The int32 tile sum lives in eight registers per thread and is scaled into
eight float accumulators once per k-tile. Occupancy is

```
blocks_by_regs  = floor(65536 / (regs_per_thread * 256))
blocks_by_smem  = floor(smem_carveout / 13952)
blocks_by_warps = floor(max_warps_per_sm / 8)
```

`regs_per_thread` is whatever ptxas assigns. It is not known until `make lib`.
The historical ptxas table in `kernels.md` is for an older build. The
carveout is chosen at runtime; the maximum on sm_87 is 164 KB, and 13952
fits the 48 KB static limit without an opt-in. `max_warps_per_sm` is the
blank cell above. This is a formula, not an occupancy that was measured.

## Instruction counts, latency, and cycles

These are issue counts. They are not latencies and they are not added
together into a cycle total. A barrier does not cost a fixed number of
cycles on top of the arithmetic; it waits for the slowest warp. A global
load's latency is overlapped with other warps when any are eligible, which
is why a byte roof and an instruction count are not a serial sum.

Per k-tile, per full 32×64 tile, per block:

| what | count | note |
|---|---|---|
| packed weight bytes loaded | 64 × 32 = 2048 | 128 × 16-byte loads |
| activation bytes loaded | 32 × 128 = 4096 | 256 × 16-byte loads |
| scale bytes | 64 × 4 = 256 | one float per output column |
| LUT lookups while decoding | 2048 | one per packed byte, not per dp4a |
| `dp4a` | 256 threads × 8 outputs × 32 steps = 65536 | 4 MACs each |
| `__syncthreads` | 2 | after the fill, after the math |

Plus one barrier after the LUT is built, once per block. For `K/128`
k-tiles the barrier count is `1 + 2*(K/128)` per block. Tail tiles still
issue the full 65536 `dp4a`s; stores are masked. Useful MACs are `M*N*K`.
Issued `dp4a * 4` equals `ceil(M/32)*32 * ceil(N/64)*64 * K`.

`tmoe_account.AccountGemm` is this arithmetic. `tmoe_cost --check` fails if
the 32×64×128 case is not 2048 + 4096 + 256 bytes and 65536 `dp4a`.

## Bytes, reuse, intensity

Issued global bytes (the loads the kernel performs, before L1/L2):

```
tiled weight = ceil(M/32) * N * (K/4)
tiled act    = ceil(N/64) * M * K
tiled scale  = ceil(M/32) * N * (K/128) * 4

naive weight = M * N * (K/4)          one warp per output, no smem reuse
naive act    = M * N * K
```

When `M` is a multiple of 32, the tiled kernel issues `32×` fewer weight
loads than the naive kernel, because the k-tile is reused across the 32
token rows in registers/shared memory. When `N` is a multiple of 64, it
issues `64×` fewer activation loads, because the activation tile is reused
across 64 output columns. That is the reduction in **issued** traffic. It
is not a reduction in compulsory DRAM traffic below one read of the packed
matrix. If the working set fits in L2, repeats may never leave the chip.
AGX Orin's L2 is cited as 4 MB. NX/Nano L2 is not stated in the sources
above; do not copy the 4 MB figure onto those modules.

Packing versus fp32 is a separate cut in the weight term: `N*K/4` bytes
instead of `N*K*4`, 16×, plus the scale vector (`4` bytes per 128 weights,
which is 1/8 of the packed row). A decode GEMV (`M = 1`) still has to stream
the packed matrix. Packing makes that stream smaller. It does not remove it.
For `M = 1` the tile's `BM = 32` does not reduce weight loads below one read
per row (there is only one token tile). The GEMV kernel is the decode path;
the GEMM tile is the prefill / multi-token path.

Arithmetic intensity here means useful MACs per issued byte. It is an issue
ratio, not a achieved FLOP/byte. Zeros still cost a `dp4a`. No roofline
crossing is claimed, because the dp4a throughput of sm_87 is not in the
Orin brief and was not measured.

## Worked example (assumptions labeled, not a prediction)

Shape: one small-config gate, 32 tokens on one expert. `M=32`, `N=1024`,
`K=512`.

```
weight bytes issued = 1 * 1024 * 128 = 131072
act bytes issued    = 16 * 32 * 512  = 262144
scale bytes         = 1 * 1024 * 4 * 4 = 16384
total issued        = 409600 bytes
dp4a issued         = 16 blocks * 4 k-tiles * 65536
```

(`ceil(1024/64) = 16` n-tiles, `512/128 = 4` k-tiles, one m-tile.)

Assumptions, all cited rather than queried: AGX Orin 64GB, r36.4 nvpmodel,
2 SMs/TPC, EMC scaled linearly from 204.8 GB/s at 3200 MHz.

* 30W default: 4 TPC → 8 SMs, GPU cap 612 MHz, EMC cap 3200 MHz, estimate
  204.8 GB/s.
* MAXN: 8 TPC → 16 SMs, GPU cap 1301 MHz, same EMC estimate. More SMs and a
  higher clock cap, not a higher DRAM estimate.
* 15W: 3 TPC → 6 SMs, GPU cap 408 MHz, EMC cap 2133 MHz, estimate ~136.5 GB/s.

A DRAM roof **if every issued byte missed and the kernel did nothing else**
is `409600 / 204.8e9 ≈ 2.0 µs` at the 204.8 GB/s estimate. That is not a
predicted runtime. The issued set is about 400 KiB, under the AGX 4 MB L2,
so a warm cache can make DRAM traffic smaller than the issued count. The
roof assumes the opposite. Compute is not added on top: if the loads and
the `dp4a`s overlap, the wall time is not the sum. `tmoe_cost` prints this
example with the same labels.

The 4096×4096 GEMV case is the one that is large next to that L2: packed
weights are 4 MiB plus a 512 KiB scale vector, against 64 MiB of fp32
weights. The compulsory weight term shrinks by about 16×. The kernel still
reads the packed matrix. No bandwidth number from a run is attached to that
sentence.

## How to reproduce on a device

CPU checks, no GPU:

```sh
make test          # Pascal suites, tmoe_cost --check, tests/ref_ternary.py
make cost          # print the SKU table and the worked example
```

CUDA build (nvcc required; the default `NVCC` is `/workspace/.tools/cuda/bin/nvcc`):

```sh
make lib NVCC=/usr/local/cuda/bin/nvcc
make lib NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=87    # Orin only
```

`make lib` writes `build/ptxas.log`. Register and spill counts belong to
that log, not to this file, until you have run it.

Timings, only after `libtmoe_cuda.so` loads and a device is present.
`tmoe_bench` uses CUDA events. Warmup launches are outside the timed
region for the GEMM (`--warmup`, default 10; `--iters`, default 50). The
GEMV wrapper does one untimed call then 200 timed calls. The printed
microseconds are per launch. They are not bandwidth.

```sh
sudo nvpmodel -q
tegrastats --interval 1000
# current clocks are whatever the driver reports; power mode matters
./build/bin/tmoe_bench --config small --warmup 10 --iters 50
./build/bin/tmoe_cost --m 32 --n 1024 --k 512
```

`--cpu` on `tmoe_bench` / `tmoe_generate` keeps the routed experts on the
CPU. Without it, generate and bench call `MoeSetDevice(True)` and fall
back to the CPU experts if the library or the upload fails.

Nsight, if the JetPack install has it. Metric names move between releases;
drop any name `ncu` rejects. These are counters to collect, not numbers
from this repository:

```sh
nsys profile --stats=true ./build/bin/tmoe_bench --config tiny
ncu --metrics \
  dram__bytes_read.sum,\
  lts__t_bytes.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  ./build/bin/tmoe_bench --config tiny
```

`dram__bytes_read` is the DRAM figure. The issued-byte formulas above will
not match it when L2 absorbs reloads. That gap is the point of separating
the two.

## What is not claimed

* No measured cycle count, occupancy, or SOL%.
* No statement that the memory bottleneck is gone. Issued weight traffic
  drops by the tile reuse factor relative to the naive per-output reload,
  and the stored weight is 2 bits rather than fp32. Both leaves a real
  load: the packed matrix, the int8 activations (re-read every 64 output
  columns), and the scales.
* No Tensor Core kernel, for the semantic and traffic reasons above.
* GPU correctness (`test_gpu`), compute-sanitizer, and the event bench were
  not run in the environment that added this account. `test_gpu` skips
  cleanly when `libtmoe_cuda.so` or a device is missing.
