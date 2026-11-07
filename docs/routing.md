# Routing, SwiGLU and the load-balance loss

CPU reference. The packed GPU expert path uses the same gate values and the
same SiLU cutoff; it is not what these tests execute. `test_gpu` compares the
two when a device is present and otherwise skips.

## Top-k

`Route` softmaxes each token (max-subtracted, so large logits stay finite) and
keeps the `K` largest probabilities. Ties go to the lower expert index, so
all-equal logits select experts `0 .. K-1` in that order.

Gates are those probabilities. With `RenormTopK`, each gate is divided by the
sum of the `K` selected probabilities. That renorm happens **before** capacity.
A slot that capacity later drops still has a gate; combine multiplies only
slots with `Kept` set, and does not redistribute the dropped gate onto the
survivor.

## Capacity

`ExpertCapacity = ceil(CapacityFactor * T * K / E)` when the factor is
positive, and at least 1. `CapacityFactor <= 0` disables the limit (the cap
is `T`, which is enough because top-k never repeats an expert inside one
token).

Placement walks choice rank 0 (the best expert) for every token, then choice
rank 1, and so on. Inside a rank, tokens run in order. A slot is kept while
the expert's kept count is below the cap. Dropped slots add nothing. The
shared expert still runs. `T = 0` builds empty arrays and the aux loss is 0.

`Counts[e]` is the number of slots assigned to `e` before this pass.
`Load[e]` is the number kept after it.

## Switch loss, top-1

Fedus, Zoph and Shazeer, *Switch Transformers* (arXiv:2101.03961), define

```
f_i = (1/T) * |{ x in the batch : argmax p(x) = i }|
P_i = (1/T) * sum_x p_i(x)
L   = alpha * N * sum_i f_i * P_i
```

`f_i` is the router's discrete choice. It is not differentiated. `P_i` is.
With `K = 1`, `AssignFraction(e)` is that `f_i`, using `Counts` (pre-drop),
and `MeanProb(e)` is `P_i`. `AuxLoss` is `L` with `alpha = AuxCoef` and
`N = E`.

If `f` is uniform, `L = alpha` even when `P` is not, because `sum_i P_i = 1`.
If every token's argmax is expert 0 and that expert also holds all of the
probability, `L = alpha * E`. The training doc's "balanced value of 0.02" is
`AuxCoef * layers` at balance (`0.01 * 2`), not `0.02` per layer.

`KeptFraction` is `Load[e] / (T*K)`. It is **not** substituted into `L`.
After a collapse that drops most tokens, the pre-drop `f_0` stays 1, so the
penalty stays near `alpha * E`. The post-drop fraction would shrink that
penalty and reward the overflow.

`AuxLossBackward` adds `alpha * E * f_e / T` to every token's `dP[e]`. An
expert with `f_e = 0` gets no direct aux gradient. Softmax still moves mass
off an overloaded expert because raising its logit is what the positive
gradient on its probability opposes.

## Top-k is a different formula

For `K > 1` the code keeps the same algebra with

```
f_i = (slots assigned to i, before capacity) / (T * K)
P_i = (1/T) * sum_tokens p_i
```

Each token contributes `K` distinct slots, so `f` sums to 1, and a token
whose top-2 are experts 0 and 1 has `f_0 = 1/K`, not the top-1 value 1.
`test_switch` and `tests/ref_ternary.py` both check that this number is not
the argmax-only Switch loss. Do not quote the paper's "perfect balance =
alpha, collapse = alpha * N" as a theorem for `K > 1`: those endpoints assume
a single argmax.

## SwiGLU

Shazeer, *GLU Variants Improve Transformer* (arXiv:2002.05202). SiLU is the
swish activation of Elfwing et al. and of Hendrycks & Gimpel (arXiv:1606.08415):

```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
```

`x < -80` is defined as 0 in both the value and the derivative (`Silu`,
`SiluDx` in `tmoe_tensor.pas`). The trainer's `SiluGrad` calls `SiluDx`.
The CUDA `swiglu_kernel` uses the same cutoff and the same product
`silu(gate) * up`.

Shapes, one routed expert:

| projection | rows | cols | input |
|---|---|---|---|
| gate, up | ExpertHidden | Dim | token (int8 if `ActQuant`) |
| down | Dim | ExpertHidden | hidden (int8 if `ActQuant`) |

The shared expert is the same product with dense fp32 weights and hidden size
`SharedHidden`. `test_swiglu` checks the shapes and that the packed float
forward matches `down(silu(gate x) * (up x))` against dequantised matvecs.

Straight-through training copies the hidden gradient through both the SiLU
product and the int8 fake-quant. Capacity-dropped slots do not contribute a
gate gradient; the aux term still sees their pre-drop assignment.
