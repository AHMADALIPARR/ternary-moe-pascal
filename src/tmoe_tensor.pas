// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Float vectors, a deterministic RNG and the dense float kernels shared by
// the inference path and the trainer. All matrices are row-major.
unit tmoe_tensor;

{$mode objfpc}{$H+}

interface

type
  TSingleArray = array of Single;
  TIntArray = array of Integer;
  TByteArray = array of Byte;
  TShortArray = array of ShortInt;

  TRng = record
    State: QWord;
  end;

procedure RngSeed(var R: TRng; Seed: QWord);
function RngNext(var R: TRng): QWord;
function RngUniform(var R: TRng): Single;            // [0,1)
function RngNormal(var R: TRng): Single;
function RngInt(var R: TRng; N: Integer): Integer;   // [0,N)

function Dot(A, B: PSingle; N: Integer): Single;
// y[n] = sum_k W[n,k] * x[k]
procedure MatVec(W: PSingle; N, K: Integer; X, Y: PSingle);
// C[t,n] = sum_k A[t,k] * W[n,k]     (C = A W^T)
procedure MatMulNT(A: PSingle; T, K: Integer; W: PSingle; N: Integer; C: PSingle);
// dA[t,k] += sum_n dC[t,n] * W[n,k]
procedure MatMulBackX(dC: PSingle; T, N: Integer; W: PSingle; K: Integer; dA: PSingle);
// dW[n,k] += sum_t dC[t,n] * A[t,k]
procedure MatMulBackW(dC: PSingle; T, N: Integer; A: PSingle; K: Integer; dW: PSingle);

procedure RMSNorm(X, G: PSingle; N: Integer; Eps: Single; Y: PSingle);
function RMSInv(X: PSingle; N: Integer; Eps: Single): Single;
procedure SoftmaxInPlace(X: PSingle; N: Integer);
function Silu(X: Single): Single; inline;
function Sigmoid(X: Single): Single; inline;
procedure RopeNeox(V: PSingle; HeadDim, Pos: Integer; Theta: Single; Inverse: Boolean);

procedure FillZero(var A: TSingleArray);
function MaxAbsDiff(A, B: PSingle; N: Integer): Single;

implementation

uses Math;

procedure RngSeed(var R: TRng; Seed: QWord);
begin
  R.State := Seed xor QWord($9E3779B97F4A7C15);
  if R.State = 0 then R.State := 1;
end;

function RngNext(var R: TRng): QWord;
var X: QWord;
begin
  X := R.State;
  X := X xor (X shr 12);
  X := X xor (X shl 25);
  X := X xor (X shr 27);
  R.State := X;
  Result := X * QWord($2545F4914F6CDD1D);
end;

function RngUniform(var R: TRng): Single;
begin
  Result := (RngNext(R) shr 40) * (1.0 / 16777216.0);
end;

function RngNormal(var R: TRng): Single;
var U1, U2: Double;
begin
  repeat
    U1 := (RngNext(R) shr 11) * (1.0 / 9007199254740992.0);
  until U1 > 1e-12;
  U2 := (RngNext(R) shr 11) * (1.0 / 9007199254740992.0);
  Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * Pi * U2);
end;

function RngInt(var R: TRng; N: Integer): Integer;
begin
  Result := Integer(RngNext(R) mod QWord(N));
end;

function Dot(A, B: PSingle; N: Integer): Single;
var
  I: Integer;
  S0, S1, S2, S3: Single;
begin
  S0 := 0; S1 := 0; S2 := 0; S3 := 0;
  I := 0;
  while I + 3 < N do
  begin
    S0 := S0 + A[I] * B[I];
    S1 := S1 + A[I + 1] * B[I + 1];
    S2 := S2 + A[I + 2] * B[I + 2];
    S3 := S3 + A[I + 3] * B[I + 3];
    Inc(I, 4);
  end;
  while I < N do
  begin
    S0 := S0 + A[I] * B[I];
    Inc(I);
  end;
  Result := (S0 + S1) + (S2 + S3);
end;

procedure MatVec(W: PSingle; N, K: Integer; X, Y: PSingle);
var I: Integer;
begin
  for I := 0 to N - 1 do
    Y[I] := Dot(@W[PtrInt(I) * K], X, K);
end;

procedure MatMulNT(A: PSingle; T, K: Integer; W: PSingle; N: Integer; C: PSingle);
var I, J: Integer;
begin
  for I := 0 to T - 1 do
    for J := 0 to N - 1 do
      C[PtrInt(I) * N + J] := Dot(@A[PtrInt(I) * K], @W[PtrInt(J) * K], K);
end;

procedure MatMulBackX(dC: PSingle; T, N: Integer; W: PSingle; K: Integer; dA: PSingle);
var
  I, J, L: Integer;
  G: Single;
  Row, Out: PSingle;
begin
  for I := 0 to T - 1 do
  begin
    Out := @dA[PtrInt(I) * K];
    for J := 0 to N - 1 do
    begin
      G := dC[PtrInt(I) * N + J];
      if G = 0 then Continue;
      Row := @W[PtrInt(J) * K];
      for L := 0 to K - 1 do
        Out[L] := Out[L] + G * Row[L];
    end;
  end;
end;

procedure MatMulBackW(dC: PSingle; T, N: Integer; A: PSingle; K: Integer; dW: PSingle);
var
  I, J, L: Integer;
  G: Single;
  Row, Out: PSingle;
begin
  for I := 0 to T - 1 do
  begin
    Row := @A[PtrInt(I) * K];
    for J := 0 to N - 1 do
    begin
      G := dC[PtrInt(I) * N + J];
      if G = 0 then Continue;
      Out := @dW[PtrInt(J) * K];
      for L := 0 to K - 1 do
        Out[L] := Out[L] + G * Row[L];
    end;
  end;
end;

function RMSInv(X: PSingle; N: Integer; Eps: Single): Single;
var
  I: Integer;
  S: Single;
begin
  S := 0;
  for I := 0 to N - 1 do S := S + X[I] * X[I];
  Result := 1.0 / Sqrt(S / N + Eps);
end;

procedure RMSNorm(X, G: PSingle; N: Integer; Eps: Single; Y: PSingle);
var
  I: Integer;
  R: Single;
begin
  R := RMSInv(X, N, Eps);
  for I := 0 to N - 1 do Y[I] := X[I] * R * G[I];
end;

procedure SoftmaxInPlace(X: PSingle; N: Integer);
var
  I: Integer;
  M, S: Single;
begin
  M := X[0];
  for I := 1 to N - 1 do if X[I] > M then M := X[I];
  S := 0;
  for I := 0 to N - 1 do
  begin
    X[I] := Exp(X[I] - M);
    S := S + X[I];
  end;
  S := 1.0 / S;
  for I := 0 to N - 1 do X[I] := X[I] * S;
end;

function Sigmoid(X: Single): Single; inline;
begin
  if X < -80 then Exit(0);
  Result := 1.0 / (1.0 + Exp(-X));
end;

function Silu(X: Single): Single; inline;
begin
  if X < -80 then Exit(0);
  Result := X / (1.0 + Exp(-X));
end;

// NEOX-style rotary embedding: element i pairs with element i + HeadDim/2.
procedure RopeNeox(V: PSingle; HeadDim, Pos: Integer; Theta: Single; Inverse: Boolean);
var
  I, Half: Integer;
  Freq, Ang, C, S, A, B: Double;
begin
  Half := HeadDim div 2;
  for I := 0 to Half - 1 do
  begin
    Freq := Power(Theta, -2.0 * I / HeadDim);
    Ang := Pos * Freq;
    C := Cos(Ang);
    S := Sin(Ang);
    if Inverse then S := -S;
    A := V[I];
    B := V[I + Half];
    V[I] := A * C - B * S;
    V[I + Half] := A * S + B * C;
  end;
end;

procedure FillZero(var A: TSingleArray);
begin
  if Length(A) > 0 then FillChar(A[0], Length(A) * SizeOf(Single), 0);
end;

function MaxAbsDiff(A, B: PSingle; N: Integer): Single;
var I: Integer;
begin
  Result := 0;
  for I := 0 to N - 1 do
    if Abs(A[I] - B[I]) > Result then Result := Abs(A[I] - B[I]);
end;

initialization
  // IEEE behaviour (inf/nan instead of exceptions), same as the CUDA side.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow, exUnderflow, exPrecision]);
end.
