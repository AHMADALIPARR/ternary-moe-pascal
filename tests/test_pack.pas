// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// 2-bit ternary packing: layout, roundtrip, and absmean quantizer.
program test_pack;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_tensor, tmoe_ternary, testutil;

var
  Rng: TRng;
  Codes, Back: TShortArray;
  Buf: TByteArray;
  W, WQ, WD: TSingleArray;
  M: TTernaryMatrix;
  I, N, R, C, Bad: Integer;
  S, Mean, Err: Double;
begin
  RngSeed(Rng, 1);
  Section('codes');
  Check((EncodeCode(0) = 0) and (EncodeCode(1) = 1) and (EncodeCode(-1) = 2), 'encode 0/+1/-1 -> 00/01/10');
  Check((DecodeCode(0) = 0) and (DecodeCode(1) = 1) and (DecodeCode(2) = -1) and (DecodeCode(3) = 0),
    'decode, 11 reads as 0');

  Section('layout');
  SetLength(Codes, 4);
  Codes[0] := 1; Codes[1] := -1; Codes[2] := 0; Codes[3] := 1;
  SetLength(Buf, 1);
  PackCodes(@Codes[0], 4, @Buf[0]);
  // col0 bits0-1=01, col1 bits2-3=10, col2=00, col3 bits6-7=01 -> 0b01_00_10_01
  Check(Buf[0] = $49, Format('byte = $%.2x (want $49)', [Buf[0]]));
  Check((TernaryLUT[$49, 0] = 1) and (TernaryLUT[$49, 1] = -1) and (TernaryLUT[$49, 2] = 0)
    and (TernaryLUT[$49, 3] = 1), 'LUT expands $49');

  Section('roundtrip');
  N := 128 * 37;
  SetLength(Codes, N); SetLength(Back, N); SetLength(Buf, N div 4);
  for I := 0 to N - 1 do Codes[I] := RngInt(Rng, 3) - 1;
  PackCodes(@Codes[0], N, @Buf[0]);
  UnpackCodes(@Buf[0], N, @Back[0]);
  Bad := 0;
  for I := 0 to N - 1 do if Codes[I] <> Back[I] then Inc(Bad);
  Check(Bad = 0, Format('%d codes pack->unpack, %d mismatches', [N, Bad]));

  Section('quantizer');
  R := 48; C := 384;
  SetLength(W, R * C); SetLength(WQ, R * C); SetLength(WD, R * C);
  for I := 0 to High(W) do W[I] := RngNormal(Rng) * 0.05;
  QuantizeAbsmean(@W[0], R, C, M);
  Check((Length(M.Bits) = R * C div 4) and (Length(M.Scales) = R * C div 128),
    Format('sizes: %d bytes, %d scales', [Length(M.Bits), Length(M.Scales)]));
  Check(TernaryBytes(M) = Int64(R) * C div 4 + Int64(R) * (C div 128) * 4, 'TernaryBytes = bits + scales');
  // scale of block 0 = mean |w|
  S := 0;
  for I := 0 to 127 do S := S + Abs(W[I]);
  Mean := S / 128;
  Check(Abs(M.Scales[0] - Mean) < 1e-6, Format('scale[0] = %.6f, mean|w| = %.6f', [M.Scales[0], Mean]));
  Bad := 0;
  for I := 0 to C - 1 do
    if TernaryCode(M, 0, I) <> Round(Max(-1.0, Min(1.0, W[I] / M.Scales[I div 128]))) then Inc(Bad);
  Check(Bad = 0, 'codes = clamp(round(w/s))');
  DequantTernary(M, @WD[0]);
  FakeQuantAbsmean(@W[0], R, C, @WQ[0]);
  Err := MaxAbsDiff(@WD[0], @WQ[0], R * C);
  Check(Err = 0, Format('dequant(quantize(w)) == fakequant(w), max diff %g', [Err]));
  Finish('test_pack');
end.
