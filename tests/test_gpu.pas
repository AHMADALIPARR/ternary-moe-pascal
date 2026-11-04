// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CUDA kernels against the CPU reference. Skips when there is no library or GPU.
program test_gpu;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_config, tmoe_tensor, tmoe_ternary, tmoe_quant, tmoe_gemv,
  tmoe_params, tmoe_model, tmoe_moe, tmoe_router, tmoe_cuda, testutil, testmats;

function RelErr(A, B: PSingle; N: Integer): Double;
var I: Integer; Mx, D: Double;
begin
  Mx := 0; D := 0;
  for I := 0 to N - 1 do
  begin
    Mx := Max(Mx, Abs(B[I]));
    D := Max(D, Abs(A[I] - B[I]));
  end;
  Result := D / Max(Mx, 1e-30);
end;

procedure Stack(const Ms: array of TTernaryMatrix; out Bits: TByteArray; out Sc: TSingleArray);
var E, NB, NS: Integer;
begin
  NB := Length(Ms[0].Bits); NS := Length(Ms[0].Scales);
  Bits := nil; Sc := nil;
  SetLength(Bits, NB * Length(Ms)); SetLength(Sc, NS * Length(Ms));
  for E := 0 to High(Ms) do
  begin
    Move(Ms[E].Bits[0], Bits[E * NB], NB);
    Move(Ms[E].Scales[0], Sc[E * NS], NS * 4);
  end;
end;

var
  Why: string;
  Rng: TRng;
  M: TTernaryMatrix;
  X, Xs, Gs, Y, Yc, Ref, Tmp: TSingleArray;
  Q, Qg: TShortArray;
  Rows, Cols, I, J, E, T, Bad, MaxD, Rc: Integer;
  Sx, Sg: Single;
  C: TMoeConfig;
  S: TParamStore;
  Md: TModel;
  R: TRouting;
  SE: TIntArray;
  BG, BU, BD: TByteArray;
  SG_, SU, SD, Gate: TSingleArray;
  Shapes: array[0..2, 0..1] of Integer = ((64, 128), (300, 512), (1024, 2048));
  K: Integer;
begin
  if not CudaAvailable(Why) then Skip('test_gpu', Why);
  WriteLn('device: ', CudaDeviceName, '  lib: ', Cuda.Path);
  RngSeed(Rng, 77);

  Section('int8 act quant');
  Rows := 37; Cols := 1024;
  SetLength(X, Rows * Cols); SetLength(Q, Rows * Cols); SetLength(Qg, Rows * Cols);
  SetLength(Xs, Rows); SetLength(Gs, Rows);
  for I := 0 to High(X) do X[I] := RngNormal(Rng) * (1 + I mod 7);
  QuantizeInt8Rows(@X[0], Rows, Cols, @Q[0], @Xs[0]);
  Rc := Cuda.ActQuant(@X[0], Rows, Cols, @Qg[0], @Gs[0]);
  Check(Rc = 0, 'kernel ran ' + CudaError);
  Bad := 0; MaxD := 0;
  for I := 0 to High(Q) do
    if Q[I] <> Qg[I] then begin Inc(Bad); MaxD := Max(MaxD, Abs(Q[I] - Qg[I])); end;
  Check(MaxD <= 1, Format('codes: %d of %d differ (max |diff| %d)', [Bad, Rows * Cols, MaxD]));
  Check(RelErr(@Gs[0], @Xs[0], Rows) < 1e-6, 'row scales');

  for K := 0 to High(Shapes) do
  begin
    Rows := Shapes[K, 0]; Cols := Shapes[K, 1];
    Section(Format('GEMV/GEMM %dx%d', [Rows, Cols]));
    RandomTernary(Rng, Rows, Cols, False, M);
    T := 45;
    SetLength(Q, T * Cols); SetLength(Xs, T);
    RandomInt8(Rng, T * Cols, @Q[0]);
    for I := 0 to T - 1 do Xs[I] := 0.001 + 0.01 * RngUniform(Rng);
    SetLength(Y, Rows); SetLength(Yc, Rows);
    TernaryGemvInt8(M, @Q[0], Xs[0], @Yc[0]);
    Rc := Cuda.Gemv(@M.Bits[0], @M.Scales[0], Rows, Cols, @Q[0], Xs[0], @Y[0]);
    Check((Rc = 0) and (RelErr(@Y[0], @Yc[0], Rows) < 1e-5),
      Format('dp4a GEMV vs CPU, rel err %.2e %s', [RelErr(@Y[0], @Yc[0], Rows), CudaError]));
    SetLength(Y, T * Rows); SetLength(Yc, T * Rows);
    TernaryGemmInt8(M, T, @Q[0], @Xs[0], @Yc[0]);
    Rc := Cuda.Gemm(@M.Bits[0], @M.Scales[0], Rows, Cols, @Q[0], @Xs[0], T, @Y[0]);
    Check((Rc = 0) and (RelErr(@Y[0], @Yc[0], T * Rows) < 1e-5),
      Format('tiled GEMM (M=%d) vs CPU, rel err %.2e', [T, RelErr(@Y[0], @Yc[0], T * Rows)]));
  end;

  Section('grouped expert GEMM (routed MoE path)');
  C := TinyConfig;
  C.TopK := 2; C.CapacityFactor := 1.0;
  BuildLayout(S, C);
  InitParams(S, 3);
  BuildModel(Md, S);
  T := 50;
  SetLength(X, T * C.Dim); SetLength(Y, T * C.Dim); SetLength(Ref, T * C.Dim); SetLength(Tmp, C.Dim);
  for I := 0 to High(X) do X[I] := RngNormal(Rng);
  MoeForward(C, Md.Layers[0].Moe, @X[0], T, @Y[0], R);
  FillZero(Ref);
  SetLength(SE, T * C.TopK); SetLength(Gate, T * C.TopK);
  for I := 0 to T * C.TopK - 1 do
  begin
    if R.Kept[I] then SE[I] := R.Idx[I] else SE[I] := -1;
    Gate[I] := R.Gate[I];
    if SE[I] >= 0 then
    begin
      RoutedExpertForward(C, Md.Layers[0].Moe, SE[I], @X[(I div C.TopK) * C.Dim], @Tmp[0]);
      for J := 0 to C.Dim - 1 do Ref[(I div C.TopK) * C.Dim + J] += Gate[I] * Tmp[J];
    end;
  end;
  Stack(Md.Layers[0].Moe.ExGate, BG, SG_);
  Stack(Md.Layers[0].Moe.ExUp, BU, SU);
  Stack(Md.Layers[0].Moe.ExDown, BD, SD);
  FillZero(Y);
  Rc := Cuda.MoeRouted(@X[0], T, C.Dim, C.ExpertHidden, C.Experts, C.TopK, @SE[0], @Gate[0],
    @BG[0], @SG_[0], @BU[0], @SU[0], @BD[0], @SD[0], @Y[0]);
  Check(Rc = 0, 'kernels ran ' + CudaError);
  // int8 re-quantisation of the hidden layer can flip a code by one; the
  // result still has to agree to well under a percent
  Check(RelErr(@Y[0], @Ref[0], T * C.Dim) < 1e-2,
    Format('GPU routed experts vs CPU, rel err %.2e', [RelErr(@Y[0], @Ref[0], T * C.Dim)]));
  Finish('test_gpu');
end.
