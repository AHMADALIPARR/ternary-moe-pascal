#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
"""Independent reference for 2-bit ternary packing, block-scaled GEMM and the
Switch auxiliary loss. CPU only. Does not claim anything about a GPU.
"""

from __future__ import annotations

import math


def encode(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return 2
    return 0


def decode(bits: int) -> int:
    bits &= 3
    if bits == 1:
        return 1
    if bits == 2:
        return -1
    return 0


def pack(codes: list[int]) -> bytes:
    out = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        out[i >> 2] |= encode(c) << (2 * (i & 3))
    return bytes(out)


def unpack(buf: bytes, n: int) -> list[int]:
    return [decode(buf[i >> 2] >> (2 * (i & 3))) for i in range(n)]


def gemv(bits: bytes, scales: list[float], n: int, k: int, xq: list[int], xs: float) -> list[float]:
    nb = k // 128
    y = []
    for r in range(n):
        acc = 0.0
        for b in range(nb):
            s = 0
            base = r * (k // 4) + b * 32
            for j in range(128):
                col = b * 128 + j
                code = decode(bits[base + j // 4] >> (2 * (j & 3)))
                s += code * xq[col]
            acc += scales[r * nb + b] * s
        y.append(acc * xs)
    return y


def switch_loss(assign_counts, probs, alpha: float) -> tuple[float, str]:
    """Top-1 Switch loss. assign_counts is the pre-drop argmax histogram.

    Fedus, Zoph, Shazeer, arXiv:2101.03961:
      f_i = count_i / T
      P_i = mean_t p_i(t)
      L = alpha * N * sum f_i P_i
    counts are not differentiated.
    """
    t = len(probs)
    if t == 0:
        return 0.0, "top-1"
    n = len(probs[0])
    f = [assign_counts[i] / t for i in range(n)]
    p = [sum(row[i] for row in probs) / t for i in range(n)]
    loss = alpha * n * sum(fi * pi for fi, pi in zip(f, p))
    return loss, "top-1"


def topk_generalization(slot_counts, probs, k: int, alpha: float) -> float:
    """NOT the published Switch formula. f_i = slots_assigned_to_i / (T*K)."""
    t = len(probs)
    n = len(probs[0])
    f = [slot_counts[i] / (t * k) for i in range(n)]
    p = [sum(row[i] for row in probs) / t for i in range(n)]
    return alpha * n * sum(fi * pi for fi, pi in zip(f, p))


def silu(x: float) -> float:
    if x < -80:
        return 0.0
    return x / (1.0 + math.exp(-x))


def silu_dx(x: float) -> float:
    if x < -80:
        return 0.0
    s = 1.0 / (1.0 + math.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def main() -> None:
    bad = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal bad
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            bad += 1

    print("[pack]")
    for lane in range(4):
        codes = [0, 0, 0, 0]
        codes[lane] = 1
        b = pack(codes)[0]
        check(b == (1 << (2 * lane)), f"+1 in lane {lane} -> byte {b:#x}")
        codes[lane] = -1
        b = pack(codes)[0]
        check(b == (2 << (2 * lane)), f"-1 in lane {lane} -> byte {b:#x}")
    for byte in range(256):
        got = unpack(bytes([byte]), 4)
        for lane in range(4):
            expect = decode(byte >> (2 * lane))
            if got[lane] != expect:
                bad += 1
    check(decode(3) == 0, "code 11 decodes as 0")
    for n in (1, 2, 3, 5, 127, 130):
        codes = [(i % 3) - 1 for i in range(n)]
        check(unpack(pack(codes), n) == codes, f"roundtrip n={n}")

    print("[gemm tails]")
    k = 128
    codes = [(i % 3) - 1 for i in range(65 * k)]
    bits = pack(codes)
    scales = [0.01 * ((i % 5) - 2) for i in range(65)]
    xq = [((i * 17) % 255) - 127 for i in range(k)]
    # code 11 planted in row 0, columns 0..3
    mutable = bytearray(bits)
    mutable[0] = 0xFF
    bits = bytes(mutable)
    y = gemv(bits, scales, 65, k, xq, 0.5)
    # independent accumulation in a different loop order (per column, not per byte)
    ref = []
    for r in range(65):
        acc = 0.0
        for c in range(k):
            acc += scales[r] * decode(bits[(r * k + c) // 4] >> (2 * ((r * k + c) & 3))) * xq[c]
        ref.append(acc * 0.5)
    err = max(abs(a - b) for a, b in zip(y, ref))
    check(err < 1e-9, f"65x128 gemv vs column loop, max diff {err:.3e}")
    check(all(v == 0.0 for v in gemv(bits, [0.0] * 65, 65, k, xq, 1.0)), "zero scales")

    print("[switch]")
    # balanced top-1, uniform probs: L = alpha
    probs = [[0.25, 0.25, 0.25, 0.25] for _ in range(8)]
    counts = [2, 2, 2, 2]
    loss, kind = switch_loss(counts, probs, 0.01)
    check(kind == "top-1" and abs(loss - 0.01) < 1e-12, f"balanced {loss}")
    # collapsed, pre-drop f_0 = 1 even if capacity would keep only 4/8
    loss, _ = switch_loss([8, 0, 0, 0], probs, 0.01)
    # uniform probs and a collapsed assignment: P_0 = 0.25, f_0 = 1
    # L = 0.01 * 4 * 1 * 0.25 = 0.01, which is NOT alpha*N.
    # alpha*N needs both f and P collapsed. Keep that explicit.
    check(abs(loss - 0.01) < 1e-12, f"collapsed assign + uniform P is alpha, not alpha*N ({loss})")
    one = [[1.0, 0.0, 0.0, 0.0] for _ in range(8)]
    loss, _ = switch_loss([8, 0, 0, 0], one, 0.01)
    check(abs(loss - 0.04) < 1e-12, f"collapsed f and P is alpha*N ({loss})")
    # post-drop would use counts [4,0,0,0] on T=8 -> f=0.5, L = 0.01*4*0.5*1 = 0.02
    post, _ = switch_loss([4, 0, 0, 0], one, 0.01)
    check(abs(post - 0.02) < 1e-12 and abs(post - loss) > 1e-6, "post-drop fraction is a different loss")
    loss0, _ = switch_loss([], [], 0.01)
    check(loss0 == 0.0, "empty batch loss is 0")
    # top-k generalization: K=2, every token assigns one slot to expert 0 and one to 1
    slot = [4, 4, 0, 0]
    p2 = [[0.5, 0.5, 0.0, 0.0] for _ in range(4)]
    g = topk_generalization(slot, p2, 2, 0.02)
    # f = [0.5, 0.5, 0, 0], P = [0.5, 0.5, 0, 0], sum fP = 0.5, *N = 2, *alpha = 0.04
    check(abs(g - 0.04) < 1e-12, f"top-2 generalization {g}")
    # Skewed probabilities: slot-fraction f and argmax f are different losses.
    skew = [[0.7, 0.1, 0.1, 0.1] for _ in range(4)]
    g2 = topk_generalization([4, 4, 0, 0], skew, 2, 0.02)
    t1 = switch_loss([4, 0, 0, 0], skew, 0.02)[0]
    check(abs(g2 - 0.032) < 1e-12 and abs(t1 - 0.056) < 1e-12,
          f"top-2 loss {g2} vs top-1 argmax loss {t1}")

    print("[silu]")
    for x in (-40.0, -1.0, 0.0, 0.5, 3.0):
        h = 1e-4
        num = (silu(x + h) - silu(x - h)) / (2 * h)
        check(abs(num - silu_dx(x)) < 1e-4, f"silu'({x}) err {abs(num - silu_dx(x)):.3e}")
    check(silu(-100) == 0.0 and silu_dx(-100) == 0.0, "cutoff")

    print(f"ref_ternary: {bad} failed")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
