// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Ternary weights: 2-bit packing, per-128-column block scales and the
// absmean quantizer.
//
// Code per weight: 00 = 0, 01 = +1, 10 = -1 (11 is never written).
// Byte j of a row holds columns 4j..4j+3, column 4j+i in bits 2i..2i+1.
// A row of K columns is K/4 bytes followed (in a separate array) by K/128
// float scales. The CUDA kernels use the exact same layout.
unit tmoe_ternary;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor;

type
  TTernaryMatrix = record
    Rows, Cols: Integer;        // Cols is a multiple of 128
    Bits: TByteArray;           // Rows * Cols / 4
    Scales: TSingleArray;       // Rows * Cols / 128
  end;

var
  // Byte -> four signed codes, used by the CPU kernels.
  TernaryLUT: array[0..255, 0..3] of ShortInt;

function EncodeCode(V: Integer): Byte; inline;
function DecodeCode(B: Byte): Integer; inline;
procedure PackCodes(Codes: PShortInt; N: Integer; Dest: PByte);
procedure UnpackCodes(Src: PByte; N: Integer; Codes: PShortInt);

procedure InitTernary(out M: TTernaryMatrix; Rows, Cols: Integer);
// Absmean per block: s = mean|w|, code = clamp(round(w/s), -1, 1).
procedure QuantizeAbsmean(W: PSingle; Rows, Cols: Integer; out M: TTernaryMatrix);
// Same rule, written back as dequantized floats (code * s). Used by the
// straight-through trainer so training sees exactly the packed weights.
procedure FakeQuantAbsmean(W: PSingle; Rows, Cols: Integer; WQ: PSingle);
procedure DequantTernary(const M: TTernaryMatrix; Dest: PSingle);
function TernaryCode(const M: TTernaryMatrix; R, C: Integer): Integer;
function TernaryBytes(const M: TTernaryMatrix): Int64;

implementation

function EncodeCode(V: Integer): Byte; inline;
begin
  if V > 0 then Result := 1
  else if V < 0 then Result := 2
  else Result := 0;
end;

function DecodeCode(B: Byte): Integer; inline;
begin
  case B and 3 of
    1: Result := 1;
    2: Result := -1;
  else
    Result := 0;
  end;
end;

procedure PackCodes(Codes: PShortInt; N: Integer; Dest: PByte);
var
  I: Integer;
begin
  FillChar(Dest^, (N + 3) div 4, 0);
  for I := 0 to N - 1 do
    Dest[I shr 2] := Dest[I shr 2] or (EncodeCode(Codes[I]) shl ((I and 3) * 2));
end;

procedure UnpackCodes(Src: PByte; N: Integer; Codes: PShortInt);
var I: Integer;
begin
  for I := 0 to N - 1 do
    Codes[I] := DecodeCode(Src[I shr 2] shr ((I and 3) * 2));
end;

procedure InitTernary(out M: TTernaryMatrix; Rows, Cols: Integer);
begin
  M.Rows := Rows;
  M.Cols := Cols;
  M.Bits := nil;
  M.Scales := nil;
  SetLength(M.Bits, PtrInt(Rows) * Cols div 4);
  SetLength(M.Scales, PtrInt(Rows) * Cols div 128);
end;

function BlockScale(W: PSingle): Single;
var
  I: Integer;
  S: Single;
begin
  S := 0;
  for I := 0 to 127 do S := S + Abs(W[I]);
  Result := S / 128;
  if Result < 1e-8 then Result := 1e-8;
end;

function QCode(W, Inv: Single): Integer; inline;
var R: Int64;
begin
  R := Round(W * Inv);
  if R > 1 then R := 1;
  if R < -1 then R := -1;
  Result := R;
end;

procedure QuantizeAbsmean(W: PSingle; Rows, Cols: Integer; out M: TTernaryMatrix);
var
  R, B, I, NB: Integer;
  S, Inv: Single;
  Src: PSingle;
  Codes: array[0..127] of ShortInt;
begin
  InitTernary(M, Rows, Cols);
  NB := Cols div 128;
  for R := 0 to Rows - 1 do
    for B := 0 to NB - 1 do
    begin
      Src := @W[PtrInt(R) * Cols + B * 128];
      S := BlockScale(Src);
      Inv := 1.0 / S;
      for I := 0 to 127 do Codes[I] := QCode(Src[I], Inv);
      M.Scales[R * NB + B] := S;
      PackCodes(@Codes[0], 128, @M.Bits[(PtrInt(R) * Cols + B * 128) div 4]);
    end;
end;

procedure FakeQuantAbsmean(W: PSingle; Rows, Cols: Integer; WQ: PSingle);
var
  R, B, I, NB: Integer;
  S, Inv: Single;
  Src, Dst: PSingle;
begin
  NB := Cols div 128;
  for R := 0 to Rows - 1 do
    for B := 0 to NB - 1 do
    begin
      Src := @W[PtrInt(R) * Cols + B * 128];
      Dst := @WQ[PtrInt(R) * Cols + B * 128];
      S := BlockScale(Src);
      Inv := 1.0 / S;
      for I := 0 to 127 do Dst[I] := QCode(Src[I], Inv) * S;
    end;
end;

function TernaryCode(const M: TTernaryMatrix; R, C: Integer): Integer;
var Idx: PtrInt;
begin
  Idx := PtrInt(R) * M.Cols + C;
  Result := DecodeCode(M.Bits[Idx shr 2] shr ((Idx and 3) * 2));
end;

procedure DequantTernary(const M: TTernaryMatrix; Dest: PSingle);
var R, C, NB: Integer;
begin
  NB := M.Cols div 128;
  for R := 0 to M.Rows - 1 do
    for C := 0 to M.Cols - 1 do
      Dest[PtrInt(R) * M.Cols + C] := TernaryCode(M, R, C) * M.Scales[R * NB + C div 128];
end;

function TernaryBytes(const M: TTernaryMatrix): Int64;
begin
  Result := Length(M.Bits) + Int64(Length(M.Scales)) * SizeOf(Single);
end;

procedure BuildLUT;
var B, I: Integer;
begin
  for B := 0 to 255 do
    for I := 0 to 3 do
      TernaryLUT[B, I] := DecodeCode(B shr (I * 2));
end;

initialization
  BuildLUT;
end.
