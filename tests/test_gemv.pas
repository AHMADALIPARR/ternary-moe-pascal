// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CPU ternary GEMV/GEMM against naive references.
program test_gemv;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_tensor, tmoe_ternary, tmoe_quant, tmoe_gemv, testutil, testmats;

const
  Shapes: array[0..3, 0..1] of Integer = ((1, 128), (17, 256), (64, 512), (96, 1024));

var
  Rng: TRng;
  M: TTernaryMatrix;
  Xq: TShortArray;
  Xf, Xs, Y, Y2, Dq: TSingleArray;
  Yd: array of Double;
  C: Integer;
  Sums: TIntArray;
  S, R, B, I, T, Bad, NB: Integer;
  Err, Rel: Double;
begin
  RngSeed(Rng, 7);
  for S := 0 to High(Shapes) do
  begin
    Section(Format('%dx%d', [Shapes[S, 0], Shapes[S, 1]]));
    // integer block sums exact
    RandomTernary(Rng, Shapes[S, 0], Shapes[S, 1], False, M);
    SetLength(Xq, M.Cols);
    RandomInt8(Rng, M.Cols, @Xq[0]);
    NB := M.Cols div 128;
    SetLength(Sums, NB);
    Bad := 0;
    for R := 0 to M.Rows - 1 do
    begin
      TernaryRowBlockSums(M, R, @Xq[0], @Sums[0]);
      for B := 0 to NB - 1 do
        if Sums[B] <> NaiveBlockSum(M, R, B, @Xq[0]) then Inc(Bad);
    end;
    Check(Bad = 0, Format('int32 block sums exact (%d blocks, %d mismatches)', [M.Rows * NB, Bad]));

    // unit scales: float result is an exact integer
    for I := 0 to High(M.Scales) do M.Scales[I] := 1;
    SetLength(Y, M.Rows); SetLength(Yd, M.Rows);
    TernaryGemvInt8(M, @Xq[0], 1, @Y[0]);
    NaiveGemv(M, @Xq[0], 1, @Yd[0]);
    Bad := 0;
    for R := 0 to M.Rows - 1 do if Y[R] <> Yd[R] then Inc(Bad);
    Check(Bad = 0, Format('unit scales: bit-exact vs naive (%d mismatches)', [Bad]));

    // random scales: tight tolerance vs double reference
    RandomTernary(Rng, Shapes[S, 0], Shapes[S, 1], False, M);
    TernaryGemvInt8(M, @Xq[0], 0.0123, @Y[0]);
    NaiveGemv(M, @Xq[0], 0.0123, @Yd[0]);
    // error against the fp32 bound: (sum of |block terms|) x NB x eps
    Rel := 0;
    for R := 0 to M.Rows - 1 do
    begin
      Err := 0;
      for B := 0 to NB - 1 do
        Err := Err + Abs(M.Scales[R * NB + B] * NaiveBlockSum(M, R, B, @Xq[0])) * 0.0123;
      Rel := Max(Rel, Abs(Y[R] - Yd[R]) / Max(1e-30, Err));
    end;
    Check(Rel < (NB + 2) * 1.2e-7, Format('random scales: max err %.2e of row L1 (bound %.1e)', [Rel, (NB + 2) * 1.2e-7]));

    // GEMM over T tokens equals T GEMVs bit for bit
    T := 5;
    SetLength(Xq, T * M.Cols); SetLength(Xs, T);
    RandomInt8(Rng, T * M.Cols, @Xq[0]);
    for I := 0 to T - 1 do Xs[I] := 0.001 * (I + 1);
    SetLength(Y, T * M.Rows); SetLength(Y2, M.Rows);
    TernaryGemmInt8(M, T, @Xq[0], @Xs[0], @Y[0]);
    Bad := 0;
    for I := 0 to T - 1 do
    begin
      TernaryGemvInt8(M, @Xq[I * M.Cols], Xs[I], @Y2[0]);
      for R := 0 to M.Rows - 1 do if Y[I * M.Rows + R] <> Y2[R] then Inc(Bad);
    end;
    Check(Bad = 0, Format('GEMM (T=%d) == GEMV per token (%d mismatches)', [T, Bad]));

    // float GEMV vs dequantised matvec
    SetLength(Xf, M.Cols); SetLength(Dq, PtrInt(M.Rows) * M.Cols); SetLength(Y, M.Rows); SetLength(Y2, M.Rows);
    for I := 0 to M.Cols - 1 do Xf[I] := RngNormal(Rng);
    TernaryGemvF32(M, @Xf[0], @Y[0]);
    DequantTernary(M, @Dq[0]);
    MatVec(@Dq[0], M.Rows, M.Cols, @Xf[0], @Y2[0]);
    Err := MaxAbsDiff(@Y[0], @Y2[0], M.Rows);
      Check(Err < 1e-4, Format('f32 GEMV vs dequant matvec, max diff %.2e', [Err]));
  end;

  Section('tails, code 11, zero scale');
  // Tile is 32 x 64 in the CUDA kernel. These shapes cross that boundary
  // on the CPU reference, which the GPU test compares against.
  for S := 0 to 4 do
  begin
    case S of
      0: begin R := 1; C := 128; end;
      1: begin R := 33; C := 128; end;
      2: begin R := 65; C := 256; end;
      3: begin R := 31; C := 384; end;
      4: begin R := 64; C := 128; end;
    else
      begin R := 1; C := 128; end;
    end;
    RandomTernary(Rng, R, C, False, M);
    T := 33;
    SetLength(Xq, T * M.Cols); SetLength(Xs, T);
    RandomInt8(Rng, T * M.Cols, @Xq[0]);
    for I := 0 to T - 1 do Xs[I] := 0.02 * (I - 16);
    SetLength(Y, T * M.Rows); SetLength(Yd, T * M.Rows);
    TernaryGemmInt8(M, T, @Xq[0], @Xs[0], @Y[0]);
    Bad := 0; Err := 0;
    for I := 0 to T - 1 do
    begin
      NaiveGemv(M, @Xq[I * M.Cols], Xs[I], @Yd[0]);
      for B := 0 to M.Rows - 1 do
        Err := Max(Err, Abs(Y[I * M.Rows + B] - Yd[B]));
    end;
    Check(Err < 1e-4, Format('GEMM %dx%d T=%d vs naive double, max diff %.2e',
      [R, C, T, Err]));
  end;
  RandomTernary(Rng, 8, 128, True, M);
  M.Bits[0] := $FF;
  SetLength(Xq, 128); SetLength(Y, 8); SetLength(Yd, 8);
  for I := 0 to 127 do Xq[I] := 7;
  TernaryGemvInt8(M, @Xq[0], 1, @Y[0]);
  NaiveGemv(M, @Xq[0], 1, @Yd[0]);
  Err := 0;
  for I := 0 to 7 do Err := Max(Err, Abs(Y[I] - Yd[I]));
  Check((Err < 1e-6) and (Y[0] = Yd[0]), Format('code 11 GEMV matches naive (diff %.2e)', [Err]));
  for I := 0 to High(M.Scales) do M.Scales[I] := 0;
  TernaryGemvInt8(M, @Xq[0], 3, @Y[0]);
  Bad := 0;
  for I := 0 to 7 do if Y[I] <> 0 then Inc(Bad);
  Check(Bad = 0, 'zero scales produce a zero row');
  Xs[0] := 0;
  for I := 0 to High(M.Scales) do M.Scales[I] := 1;
  TernaryGemvInt8(M, @Xq[0], 0, @Y[0]);
  Bad := 0;
  for I := 0 to 7 do if Y[I] <> 0 then Inc(Bad);
  Check(Bad = 0, 'zero activation scale produces a zero row');

  Section('int8 quant');
  SetLength(Xf, 256); SetLength(Xq, 256);
  for I := 0 to 255 do Xf[I] := RngNormal(Rng) * 3;
  QuantizeInt8Row(@Xf[0], 256, @Xq[0], Xs[0]);
  Bad := 0; Err := 0;
  for I := 0 to 255 do
  begin
    if Abs(Xq[I]) > 127 then Inc(Bad);
    Err := Max(Err, Abs(Xq[I] * Xs[0] - Xf[I]));
  end;
  Check(Bad = 0, 'codes within [-127,127]');
  Check(Err <= 0.5 * Xs[0] * 1.0001, Format('dequant error %.4f <= scale/2 = %.4f', [Err, 0.5 * Xs[0]]));
  Finish('test_gemv');
end.
