// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Static byte and instruction accounting for the tiled ternary GEMM.
// These are counts of loads and dp4a ops the kernel issues. They are not
// measured cycles, not DRAM traffic (L1/L2 can filter repeats), and not a
// runtime. See docs/orin-gemm.md for what is cited versus estimated.
unit tmoe_account;

{$mode objfpc}{$H+}

interface

const
  // Must match cuda/tmoe_kernels.cu (BM, BN, BKW, PADW, 256 threads).
  GEMM_BM = 32;
  GEMM_BN = 64;
  GEMM_BK = 128;
  GEMM_THREADS = 256;
  GEMM_PADW = 33;          // BKW+1, BKW = 128/4
  // lut[256] + sx[BM][PADW] + sw[BN][PADW] + in_row[BM] + out_row[BM], ints
  GEMM_SMEM = (256 + GEMM_BM * GEMM_PADW + GEMM_BN * GEMM_PADW + GEMM_BM + GEMM_BM) * 4;

type
  TGemmAccount = record
    Valid: Boolean;
    M, N, K: Integer;
    TilesM, TilesN, TilesK: Integer;
    // Bytes requested by global loads, before any cache.
    Weight, Act, Scale, Total: Int64;
    NaiveWeight, NaiveAct, NaiveScale, NaiveTotal: Int64;
    // dp4a instructions issued, including full tiles that cover a tail.
    Dp4aIssued: Int64;
    MacUseful, MacPadded: Int64;
    Blocks: Int64;
    BarriersPerBlock: Integer;
    SmemBytes: Integer;
  end;

function AccountGemm(M, N, K: Integer): TGemmAccount;
// True when the identities in docs/orin-gemm.md hold for the built-in cases.
function AccountsConsistent(out Why: string): Boolean;

implementation

uses SysUtils;

function CeilDiv(A, B: Integer): Integer; inline;
begin
  Result := (A + B - 1) div B;
end;

function AccountGemm(M, N, K: Integer): TGemmAccount;
var
  A: TGemmAccount;
  Ntm, Ntn, Ntk: Integer;
begin
  FillChar(A, SizeOf(A), 0);
  A.M := M;
  A.N := N;
  A.K := K;
  A.SmemBytes := GEMM_SMEM;
  if (M <= 0) or (N <= 0) or (K <= 0) or (K mod GEMM_BK <> 0) then
  begin
    Result := A;
    Exit;
  end;
  Ntm := CeilDiv(M, GEMM_BM);
  Ntn := CeilDiv(N, GEMM_BN);
  Ntk := K div GEMM_BK;
  A.Valid := True;
  A.TilesM := Ntm;
  A.TilesN := Ntn;
  A.TilesK := Ntk;
  // Each m-tile reloads every valid weight row. Tail rows are not loaded.
  A.Weight := Int64(Ntm) * N * (K div 4);
  // Each n-tile reloads every valid activation row (K int8 bytes).
  A.Act := Int64(Ntn) * M * K;
  A.Scale := Int64(Ntm) * N * (K div GEMM_BK) * 4;
  A.Total := A.Weight + A.Act + A.Scale;
  // Naive kernel: one warp per output row per token, no smem reuse across
  // tokens or across output rows of the same token.
  A.NaiveWeight := Int64(M) * N * (K div 4);
  A.NaiveAct := Int64(M) * N * K;
  A.NaiveScale := Int64(M) * N * (K div GEMM_BK) * 4;
  A.NaiveTotal := A.NaiveWeight + A.NaiveAct + A.NaiveScale;
  // Per full tile per k-step the block issues BM*BN*BK/4 dp4a (8 per thread
  // times 256 threads times 32 int8x4 steps). Tail tiles still do that work.
  A.Dp4aIssued := Int64(Ntm) * Ntn * Ntk * GEMM_THREADS * (8 * (GEMM_BK div 4));
  A.MacUseful := Int64(M) * N * K;
  A.MacPadded := Int64(Ntm) * GEMM_BM * Int64(Ntn) * GEMM_BN * K;
  A.Blocks := Int64(Ntm) * Ntn;
  // One barrier after the LUT, then one after the load and one after the
  // compute of every k-tile. A count of sync events, not a cycle count.
  A.BarriersPerBlock := 1 + 2 * Ntk;
  Result := A;
end;

function AccountsConsistent(out Why: string): Boolean;
var
  A: TGemmAccount;
  Smem: Integer;
begin
  Why := '';
  Smem := (256 + GEMM_BM * GEMM_PADW + GEMM_BN * GEMM_PADW + GEMM_BM + GEMM_BM) * 4;
  if Smem <> 13952 then
  begin
    Why := Format('shared-memory formula %d, kernel layout is 13952 bytes', [Smem]);
    Exit(False);
  end;
  A := AccountGemm(32, 64, 128);
  if (not A.Valid) or (A.Weight <> 2048) or (A.Act <> 4096) or (A.Scale <> 256) or
     (A.Total <> 6400) or (A.NaiveWeight <> 65536) or (A.Dp4aIssued <> 65536) or
     (A.MacUseful <> 262144) or (A.MacPadded <> A.Dp4aIssued * 4) or
     (A.Blocks <> 1) or (A.BarriersPerBlock <> 3) then
  begin
    Why := Format('32x64x128 account mismatch w=%d a=%d s=%d dp4a=%d',
      [A.Weight, A.Act, A.Scale, A.Dp4aIssued]);
    Exit(False);
  end;
  // M multiple of BM: tiled weight loads are naive/BM (each weight reused
  // across BM tokens instead of reissued per token).
  A := AccountGemm(128, 256, 512);
  if A.NaiveWeight <> A.Weight * GEMM_BM then
  begin
    Why := 'weight reuse identity failed';
    Exit(False);
  end;
  if A.NaiveAct <> A.Act * GEMM_BN then
  begin
    Why := 'activation reuse identity failed';
    Exit(False);
  end;
  if A.MacPadded <> A.Dp4aIssued * 4 then
  begin
    Why := 'padded MAC vs dp4a failed';
    Exit(False);
  end;
  // Tail: M=33, N=65 still valid, padded MACs exceed useful MACs.
  A := AccountGemm(33, 65, 128);
  if (not A.Valid) or (A.TilesM <> 2) or (A.TilesN <> 2) or (A.MacPadded <= A.MacUseful) then
  begin
    Why := 'tail tile accounting failed';
    Exit(False);
  end;
  A := AccountGemm(4, 4, 100);
  if A.Valid then
  begin
    Why := 'K not multiple of 128 should be invalid';
    Exit(False);
  end;
  Result := True;
end;

end.
