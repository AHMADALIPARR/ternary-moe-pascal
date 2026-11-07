# Architecture

## Model

Byte-level vocabulary (256). For each of `Layers` blocks:

```
h = x + Attn(RMSNorm(x))
y = h + MoE(RMSNorm(h))
```

followed by a final RMSNorm and an fp32 LM head.

**Attention.** `Heads` query heads, `KVHeads` key/value heads (GQA, query head
`h` reads KV head `h div (Heads/KVHeads)`), head size `HeadDim`, causal
softmax with `1/sqrt(HeadDim)` scaling. RoPE uses the NEOX layout: dimension
`i` rotates with `i + HeadDim/2`, frequency `theta^(-2i/HeadDim)`. Keys and
values are kept per layer in a `MaxSeq`-long KV cache; prefill and
token-by-token decode give identical logits (checked in `test_moe`).

**MoE layer.**

```
MoE(x) = Shared(x) + sum over kept slots s of gate_s * Expert_{e_s}(x)
Shared, Expert = SwiGLU: down( silu(gate x) * (up x) )
```

The shared expert is dense fp32 with hidden size `SharedHidden`. The `Experts`
routed experts have hidden size `ExpertHidden`, ternary weights and INT8 inputs
(both the input of gate/up and the hidden vector going into down are quantised
per token).

**Router.** `p = softmax(W_r x)`, then the `TopK` largest probabilities (ties go
to the lower expert index). Gates are the selected probabilities, or with
`RenormTopK` the selected probabilities divided by their sum.

Capacity: with `CapacityFactor > 0` each expert accepts at most
`ceil(CF * T * K / E)` slots per batch. All first choices are placed before any
second choice, and within a choice rank tokens go in order. A dropped slot
contributes nothing; the shared expert still runs. `CapacityFactor <= 0` turns
the limit off.

Load-balance loss: `aux = AuxCoef * E * sum_e f_e * P_e`. For top-1, `f_e` is
the fraction of tokens whose argmax is `e` **before** capacity drops, and
`P_e` is the mean router probability. That is the Switch Transformer formula
(Fedus, Zoph, Shazeer, arXiv:2101.03961). `f` is a stop-gradient. Perfect
balance gives `AuxCoef`; a collapsed argmax and a collapsed probability both
give `AuxCoef * E`. Top-k > 1 uses the same algebra on the fraction of
selected slots, which is not the published top-1 formula. The post-drop kept
fraction is recorded and is not what the loss uses. See
[routing.md](routing.md).

## Configurations

| | tiny | small |
|---|---|---|
| Dim | 128 | 512 |
| Layers | 2 | 8 |
| Heads / KV heads / head size | 4 / 2 / 32 | 8 / 2 / 64 |
| Shared hidden | 256 | 1024 |
| Expert hidden | 256 | 1024 |
| Experts / top-k | 4 / 1 | 8 / 1 |
| Max sequence | 128 | 512 |
| Parameters | 1,148,544 | 118,792,704 |

`--experts`, `--topk`, `--capacity` and `--renorm` override these on the
command line of every tool.

## Ternary format

A matrix with `Rows x Cols` weights (`Cols` a multiple of 128) is stored as

* `bits`: `Rows * Cols / 4` bytes. Column `4j + i` of a row lives in bits
  `2i .. 2i+1` of byte `j` of that row. Codes: `00` = 0, `01` = +1, `10` = -1
  (`11` is never written and reads as 0).
* `scales`: one fp32 per row per 128 columns, the mean absolute value of the
  latent weights in that block.

Quantisation is `code = clamp(round(w / s), -1, 1)` with `s = mean|w|` over the
block. A routed SwiGLU expert of the small config takes 3 x 1024 x 512 / 4 bytes
plus 3 x 1024 x 4 x 4 bytes of scales = 442,368 bytes.

## INT8 activations

Per token: `amax = max|x|`, `scale = amax / 127`, `q = clamp(rne(x * 127 / amax), -127, 127)`.
A zero row gets scale 0 and all-zero codes. The CPU and CUDA versions use the
same rounding, so codes agree.

## Matvec

For row `r` and 128-block `b`:

```
acc_b = sum over the block of code(r, c) * q(c)        (exact int32)
y_r   = x_scale * sum_b w_scale(r, b) * acc_b
```

The scales are applied once per block in the epilogue; the inner loop is pure
integer work. `test_gemv` checks the block sums are exact against a naive
version and the float result against a double-precision reference.

## Units

| Unit | Contents |
|---|---|
| `tmoe_config` | `TMoeConfig`, tiny/small presets, validation |
| `tmoe_tensor` | arrays, RNG, RMSNorm, softmax, SiLU, fp32 matvec |
| `tmoe_ternary` | code encoding, packing, absmean quantiser, fake-quant |
| `tmoe_quant` | INT8 per-row quantisation |
| `tmoe_gemv` | ternary GEMV/GEMM (INT8 and fp32 inputs) |
| `tmoe_router` | softmax top-k, capacity, aux loss and their gradients |
| `tmoe_params` | parameter layout and initialisation |
| `tmoe_kvcache`, `tmoe_attention` | KV cache, GQA + RoPE |
| `tmoe_moe` | MoE layer forward on packed experts |
| `tmoe_model` | full forward, byte accounting |
| `tmoe_sampler` | temperature, top-k, top-p sampling |
| `tmoe_train` | training forward/backward, AdamW |
| `tmoe_data` | text loading, batches, eval loss |
| `tmoe_checkpoint` | binary checkpoints |
| `tmoe_cuda` | dynamic binding of `libtmoe_cuda.so` |
