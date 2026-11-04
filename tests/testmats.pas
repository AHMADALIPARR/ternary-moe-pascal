// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Random test matrices and naive references.
unit testmats;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor, tmoe_ternary;

procedure RandomTernary(var Rng: TRng; Rows, Cols: Integer; UnitScales: Boolean; out M: TTernaryMatrix);
procedure RandomInt8(var Rng: TRng; N: Integer; Q: PShortInt);
// y[r] = xs * sum_c s[r,c/128] * code[r,c] * xq[c], in doubles with codes read one by one.
procedure NaiveGemv(const M: TTernaryMatrix; Xq: PShortInt; Xs: Double; Y: PDouble);
// integer dot of one row and block
function NaiveBlockSum(const M: TTernaryMatrix; R, B: Integer; Xq: PShortInt): Integer;

implementation

procedure RandomTernary(var Rng: TRng; Rows, Cols: Integer; UnitScales: Boolean; out M: TTernaryMatrix);
var
  Codes: TShortArray;
  I: Integer;
begin
  InitTernary(M, Rows, Cols);
  SetLength(Codes, PtrInt(Rows) * Cols);
  for I := 0 to High(Codes) do Codes[I] := RngInt(Rng, 3) - 1;
  PackCodes(@Codes[0], Length(Codes), @M.Bits[0]);
  for I := 0 to High(M.Scales) do
    if UnitScales then M.Scales[I] := 1 else M.Scales[I] := 0.005 + 0.05 * RngUniform(Rng);
end;

procedure RandomInt8(var Rng: TRng; N: Integer; Q: PShortInt);
var I: Integer;
begin
  for I := 0 to N - 1 do Q[I] := RngInt(Rng, 255) - 127;
end;

function NaiveBlockSum(const M: TTernaryMatrix; R, B: Integer; Xq: PShortInt): Integer;
var C: Integer;
begin
  Result := 0;
  for C := B * 128 to B * 128 + 127 do
    Result := Result + TernaryCode(M, R, C) * Xq[C];
end;

procedure NaiveGemv(const M: TTernaryMatrix; Xq: PShortInt; Xs: Double; Y: PDouble);
var
  R, C, NB: Integer;
  Acc: Double;
begin
  NB := M.Cols div 128;
  for R := 0 to M.Rows - 1 do
  begin
    Acc := 0;
    for C := 0 to M.Cols - 1 do
      Acc := Acc + Double(M.Scales[R * NB + C div 128]) * TernaryCode(M, R, C) * Xq[C];
    Y[R] := Acc * Xs;
  end;
end;

end.
