# ternary-moe-pascal

![License](https://img.shields.io/badge/license-AGPL--3.0--only%20%7C%20SnapKitty--Opaque--1.0-blue)
![Free Pascal](https://img.shields.io/badge/Free%20Pascal-3.2%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-sm__80%20%7C%2086%20%7C%2087%20%7C%2089%20%7C%2090-76b900)
![Tests](https://img.shields.io/badge/make%20test-8%20suites-brightgreen)

A small decoder-only transformer whose feed-forward layers are a mixture of
ternary experts, in the spirit of MoTE. Model, runtime and trainer are written in
Free Pascal; the GPU kernels are CUDA C++ behind a plain C ABI. Everything runs on
the CPU too, so you can train, sample and test without a GPU.

## Architecture

Each block is pre-norm: RMSNorm, grouped-query attention with NEOX RoPE and a
KV cache, a residual, then RMSNorm and the MoE layer with a second residual.

The MoE layer is one shared dense SwiGLU expert that every token goes through,
plus E routed SwiGLU experts. A linear router takes a softmax over experts and
keeps the top-k (top-1 by default, with optional renormalisation and a capacity
limit). Training adds the Switch load-balance loss `coef * E * sum(f_e * P_e)`.

Routed expert weights are ternary {-1, 0, +1} with one fp32 scale per 128
columns, stored at 2 bits per weight. Activations going into them are quantised
to INT8 per token, so an expert matvec is an integer dot product per 128-wide
block followed by one multiply by `w_scale * x_scale`. The GPU path does this
with `dp4a` and keeps the weights packed in device memory.

More detail: [docs/architecture.md](docs/architecture.md),
[docs/kernels.md](docs/kernels.md), [docs/training.md](docs/training.md).

## Build and run

Needs Free Pascal 3.2+ and make. The CUDA library additionally needs nvcc.

```sh
make                 # tests and tools (CPU)
make test            # all test suites; the GPU suite skips without a device
make train           # 400 steps on data/tiny.txt, then sample from it
make bench           # CPU throughput for the tiny and small configs

make lib NVCC=/usr/local/cuda/bin/nvcc                    # sm_80..sm_90 + PTX
make lib NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=87      # Jetson Orin only
```

The tools pick up `build/libtmoe_cuda.so` on their own; `TMOE_CUDA_LIB` points
them somewhere else.

```sh
build/bin/tmoe_train --config tiny --steps 400 --out build/tiny.ckpt
build/bin/tmoe_generate --ckpt build/tiny.ckpt --prompt "The keeper" --n 200 --temp 0.8
build/bin/tmoe_bench --config small
```

## Benchmarks

Measured with `tmoe_bench` (single thread, best of 5, 64-token prompt, 64
decoded tokens) on an 8-vCPU Intel Xeon box. On a machine with a CUDA device
the bench also times the 4096x4096 GEMV kernel, which goes in the GPU columns.

| Config | Params | Resident weights | Weight bytes / token | CPU prefill tok/s | CPU decode tok/s |
|---|---|---|---|---|---|
| tiny (d=128, 2 layers, 4 experts, top-1) | 1.15 M | 1.59 MiB | 1.31 MiB | 4020.6 | 3316.1 |
| small (d=512, 8 layers, 8 experts, top-1) | 118.8 M | 96.16 MiB | 72.04 MiB | 96.0 | 90.5 |

Ternary GEMV with INT8 activations, one call:

| Shape | CPU (1 thread) | Jetson Orin (sm_87) | Desktop GPU |
|---|---|---|---|
| 256x128 | 11.34 us | | |
| 1024x512 | 166.39 us | | |
| 4096x4096 | 4598.61 us | | |

A routed expert in the small config is 432 KiB packed instead of 6 MiB in fp32.

## Training

`tmoe_train` trains with straight-through fake quantisation of the routed
experts and their inputs, AdamW, gradient clipping and a warmup plus cosine
schedule. On the bundled 4.3 KB text, 400 steps take the eval cross-entropy
from 5.81 to 2.44 nats per byte. See [docs/training.md](docs/training.md).

## License

Dual licensed under AGPL-3.0-only or LicenseRef-SnapKitty-Opaque-1.0, at your
option. See [LICENSE](LICENSE).

Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
