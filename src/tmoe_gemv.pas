// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CPU reference ternary GEMV / GEMM. Same math as the CUDA kernels:
// integer dot products of 2-bit codes against int8 activations inside each
// 128-column block, the block scale applied to the int32 block sum, and the
// per-token activation scale applied once in the epilogue.
unit tmoe_gemv;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor, tmoe_ternary;

// Integer block sums of one row: Sums[b] = sum_{c in block b} code * xq.
procedure TernaryRowBlockSums(const M: TTernaryMatrix; R: Integer; Xq: PShortInt; Sums: PInteger);
// y[r] = xs * sum_b scale[r,b] * Sums[b]
procedure TernaryGemvInt8(const M: TTernaryMatrix; Xq: PShortInt; Xs: Single; Y: PSingle);
// Y[t, r] for T tokens; Xq is T x Cols, Xs has T entries.
procedure TernaryGemmInt8(const M: TTernaryMatrix; T: Integer; Xq: PShortInt; Xs: PSingle; Y: PSingle);
// Float activations against packed weights (ActQuant off).
procedure TernaryGemvF32(const M: TTernaryMatrix; X: PSingle; Y: PSingle);

implementation

procedure TernaryRowBlockSums(const M: TTernaryMatrix; R: Integer; Xq: PShortInt; Sums: PInteger);
var
  B, J, NB, S: Integer;
  P: PByte;
  X: PShortInt;
begin
  NB := M.Cols div 128;
  P := @M.Bits[PtrInt(R) * (M.Cols div 4)];
  X := Xq;
  for B := 0 to NB - 1 do
  begin
    S := 0;
    for J := 0 to 31 do
    begin
      S := S + TernaryLUT[P^, 0] * X[0] + TernaryLUT[P^, 1] * X[1]
             + TernaryLUT[P^, 2] * X[2] + TernaryLUT[P^, 3] * X[3];
      Inc(P);
      Inc(X, 4);
    end;
    Sums[B] := S;
  end;
end;

function RowInt8(const M: TTernaryMatrix; R: Integer; Xq: PShortInt): Single; inline;
var
  Sums: array[0..1023] of Integer;
  B, NB: Integer;
  Acc: Single;
begin
  NB := M.Cols div 128;
  TernaryRowBlockSums(M, R, Xq, @Sums[0]);
  Acc := 0;
  for B := 0 to NB - 1 do
    Acc := Acc + M.Scales[R * NB + B] * Sums[B];
  Result := Acc;
end;

procedure TernaryGemvInt8(const M: TTernaryMatrix; Xq: PShortInt; Xs: Single; Y: PSingle);
var R: Integer;
begin
  for R := 0 to M.Rows - 1 do
    Y[R] := RowInt8(M, R, Xq) * Xs;
end;

procedure TernaryGemmInt8(const M: TTernaryMatrix; T: Integer; Xq: PShortInt; Xs: PSingle; Y: PSingle);
var R, I: Integer;
begin
  for R := 0 to M.Rows - 1 do
    for I := 0 to T - 1 do
      Y[PtrInt(I) * M.Rows + R] := RowInt8(M, R, @Xq[PtrInt(I) * M.Cols]) * Xs[I];
end;

procedure TernaryGemvF32(const M: TTernaryMatrix; X: PSingle; Y: PSingle);
var
  R, B, J, NB: Integer;
  P: PByte;
  XP: PSingle;
  S, Acc: Single;
begin
  NB := M.Cols div 128;
  for R := 0 to M.Rows - 1 do
  begin
    P := @M.Bits[PtrInt(R) * (M.Cols div 4)];
    XP := X;
    Acc := 0;
    for B := 0 to NB - 1 do
    begin
      S := 0;
      for J := 0 to 31 do
      begin
        S := S + TernaryLUT[P^, 0] * XP[0] + TernaryLUT[P^, 1] * XP[1]
               + TernaryLUT[P^, 2] * XP[2] + TernaryLUT[P^, 3] * XP[3];
        Inc(P);
        Inc(XP, 4);
      end;
      Acc := Acc + M.Scales[R * NB + B] * S;
    end;
    Y[R] := Acc;
  end;
end;

end.
