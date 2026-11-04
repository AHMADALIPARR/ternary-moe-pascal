// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// INT8 per-token (per-row) symmetric activation quantization.
//   amax = max|x|, scale = amax / 127, q = round_half_even(x * (127 / amax))
// The CUDA kernel computes the same three float ops, so codes match exactly.
unit tmoe_quant;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor;

procedure QuantizeInt8Row(X: PSingle; N: Integer; Q: PShortInt; out Scale: Single);
procedure QuantizeInt8Rows(X: PSingle; Rows, N: Integer; Q: PShortInt; Scales: PSingle);
// q * scale written back as float (straight-through fake quant for training)
procedure FakeQuantInt8Row(X: PSingle; N: Integer; Y: PSingle);

implementation

procedure QuantizeInt8Row(X: PSingle; N: Integer; Q: PShortInt; out Scale: Single);
var
  I: Integer;
  AMax, Inv: Single;
  R: Int64;
begin
  AMax := 0;
  for I := 0 to N - 1 do
    if Abs(X[I]) > AMax then AMax := Abs(X[I]);
  if AMax = 0 then
  begin
    Scale := 0;
    FillChar(Q^, N, 0);
    Exit;
  end;
  Scale := AMax / 127.0;
  Inv := 127.0 / AMax;
  for I := 0 to N - 1 do
  begin
    R := Round(X[I] * Inv);
    if R > 127 then R := 127;
    if R < -127 then R := -127;
    Q[I] := R;
  end;
end;

procedure QuantizeInt8Rows(X: PSingle; Rows, N: Integer; Q: PShortInt; Scales: PSingle);
var R: Integer;
begin
  for R := 0 to Rows - 1 do
    QuantizeInt8Row(@X[PtrInt(R) * N], N, @Q[PtrInt(R) * N], Scales[R]);
end;

procedure FakeQuantInt8Row(X: PSingle; N: Integer; Y: PSingle);
var
  Q: array of ShortInt;
  S: Single;
  I: Integer;
begin
  SetLength(Q, N);
  QuantizeInt8Row(X, N, @Q[0], S);
  for I := 0 to N - 1 do Y[I] := Q[I] * S;
end;

end.
