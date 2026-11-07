# Training

`tmoe_train` (source `tools/tmoe_trainer.pas`, trainer in `src/tmoe_train.pas`)
trains byte-level next-token prediction on a text file.

## Method

* **Latent weights.** Every parameter is an fp32 latent. Before each step the
  routed expert matrices are fake-quantised (absmean ternary per 128-block);
  forward uses the quantised copy, backward passes gradients straight through
  to the latents.
* **Activations.** With `ActQuant` on (the default), the inputs to the routed
  experts and their hidden vectors are fake-quantised to INT8 per token, again
  with a straight-through gradient. This keeps the training forward identical
  to the packed inference path (`test_moe` checks they agree to 9.5e-7).
* **Loss.** Mean cross-entropy over all positions plus the load-balance loss of
  every MoE layer. Router gradients include the gate path and the aux term.
  The aux term differentiates the mean router probability only; the discrete
  expert counts are constant. See [routing.md](routing.md). The SwiGLU
  derivative is `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`, with the same
  `x < -80` cutoff as the forward SiLU.
* **Optimiser.** AdamW (beta1 0.9, beta2 0.95, eps 1e-8, weight decay 0.01 on
  the attention, shared-expert, routed-expert and output matrices; none on
  the embedding, router or norms), global gradient-norm clipping at 1.0, linear warmup for 10
  steps then cosine decay to `lr / 10`.
* **Batches.** Random windows of `seq + 1` bytes from the first 90% of the file;
  the last 10% is held out for eval loss.

The backward pass is written by hand. `test_gradcheck` compares it against
central finite differences for one element of every tensor kind.

## Run

```sh
make build/bin/tmoe_train build/bin/tmoe_generate
build/bin/tmoe_train --data data/tiny.txt --config tiny --steps 400 --batch 4 --seq 64 \
    --lr 3e-3 --out build/tiny.ckpt
build/bin/tmoe_generate --ckpt build/tiny.ckpt --prompt "The keeper" --n 200 --temp 0.8
```

Options: `--config tiny|small`, `--experts N`, `--topk K`, `--capacity F`,
`--renorm`, `--seed S`.

## Reference run

Tiny config, `data/tiny.txt` (3,924 train / 436 eval bytes), 400 steps of
4 x 64 tokens, one CPU thread, 113.3 s (904 train tokens/s).

| step | loss | CE | aux |
|---|---|---|---|
| 1 | 5.9419 | 5.9125 | 0.02944 |
| 100 | 1.9006 | 1.8787 | 0.02184 |
| 200 | 1.5569 | 1.5362 | 0.02069 |
| 300 | 1.1213 | 1.1004 | 0.02084 |
| 400 | 0.9676 | 0.9468 | 0.02075 |

Eval CE went from 5.8136 to 2.4400. The aux loss sits near its balanced value
of 0.02 (coef 0.01 x 2 layers). The text is tiny, so the model memorises it;
the run is a check that the whole pipeline learns, not a language model.

Sample after that run (`--temp 0.7`, 120 bytes):

```
The keeper the stetle of ther wimm trepe. The keeper befo the stelver fa teain the lame parred bens hile batt tan mow honkean cl
```

## Checkpoint format

Little-endian, written by `src/tmoe_checkpoint.pas`:

```
"TMOECKP1"  u32 version  config  u32 flags  i32 step
u32 ntensors { u16 len, name, u8 kind, i32 count, i32 rows, i32 cols, i64 offset }
i64 nfloats, f32 params [, f32 adam_m, f32 adam_v  when flags & 1]
u32 nmatrices { i32 rows, i32 cols, u8 bits[rows*cols/4], f32 scales[rows*cols/128] }
u64 FNV-1a of every byte above
```

The packed section stores every routed expert in its 2-bit resident form,
ordered by layer, then gate/up/down, then expert, so inference can load it
without re-quantising. Loading checks the tensor table against the layout built
from the stored config and rejects a file whose checksum does not match. The
tiny checkpoint without optimiser state is 4,816,777 bytes.
