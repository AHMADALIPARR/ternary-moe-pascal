# TERNARY-Ω — Self-Hosting Compiler Stress Test

**Execution status:** Partially realizable. Full specification is not satisfiable as written.

This directory contains an independent C reference / stress harness for balanced-ternary arithmetic, message routing, and S₃ permutation composition. It is *not* part of the Free Pascal MoE runtime; it exists to exercise the same mathematical substrate (trits in {-1, 0, +1}, packing, carry propagation) under the constraints of a self-hosting compiler thought experiment.

## Fundamental conflicts in the original directive

1. **Arbitrary precision versus finite bare-metal memory.**  
   Unbounded integers require unbounded storage. Any physical implementation must detect allocation exhaustion (here: `ERR_MEMORY` when the 65 536-byte arena is exhausted) rather than silently truncate.

2. **Immutable WORM state versus mutable vtables.**  
   Historical state can be immutable; the execution state that holds a dispatch pointer cannot be immutable at the same address. The two must be separated.

3. **Zero abstraction versus self-hosting compilation.**  
   A compiler necessarily implements representations and transformations (syntax → IR → machine). Eliminating all abstractions eliminates the mechanisms required to compile.

## Symbol table (this implementation)

| Symbol              | Role                      | Storage / constraint          |
|---------------------|---------------------------|-------------------------------|
| `arena`             | Trit memory               | 65 536 bytes, static          |
| `cursor`            | Arena allocation position | 32-bit unsigned               |
| `Ternary.digits`    | Signed ternary digits     | −1, 0, +1                     |
| `Ternary.length`    | Significant trit count    | 32-bit unsigned               |
| `Ring.entries`      | Message queue             | 64 fixed records              |
| `Ring.read/write`   | Consumer / producer index | Modulo 64                     |
| `Permutation.p`     | Three-element permutation | Elements 0–2                  |
| `pack_trits`        | Compact serializer        | Five trits per byte (3⁵=243)  |

## Build & run

```sh
cc -std=c11 -Wall -Wextra -o ternary_omega ternary_omega.c
./ternary_omega          # returns 0 on success
```

The `main` currently verifies:

- conversion of 42 and −17,
- addition and multiplication (results not yet decoded by an oracle),
- packing of the sum,
- a single push/pop on the ring buffer,
- that two elements of S₃ do not commute.

## Formal verification obligations (for a Lean 4 development)

A correct Lean 4 development must establish at least:

\[
\operatorname{value}(a \oplus b) = \operatorname{value}(a) + \operatorname{value}(b)
\]
\[
\operatorname{value}(a \otimes b) = \operatorname{value}(a) \cdot \operatorname{value}(b)
\]
\[
\operatorname{decode}(\operatorname{encode}(a)) = a
\]
\[
\exists\, p,q \in S_3 :\; p \circ q \ne q \circ p
\]

For each target architecture \(T\):

\[
\forall P,s,\quad
\operatorname{run}_T(\operatorname{compile}_T(P), s)
\sim
\operatorname{run}_{VM}(P, s)
\]

## License

Dual licensed under AGPL-3.0-only or LicenseRef-SnapKitty-Opaque-1.0, at your option.  
Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
