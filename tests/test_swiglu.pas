// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// SiLU derivative and SwiGLU expert shapes. CPU reference. The packed GPU
// SwiGLU uses the same cutoff (x < -80 -> 0) in swiglu_kernel; this file
// does not launch it.
program test_swiglu;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_config, tmoe_tensor, tmoe_ternary, tmoe_params,
  tmoe_model, tmoe_moe, testutil;

function DxNum(X, H: Single): Double;
begin
  Result := (Double(Silu(X + H)) - Double(Silu(X - H))) / (2 * H);
end;

var
  Xs: array[0..6] of Single = (-40, -2, -0.5, 0, 0.5, 2, 20);
  I: Integer;
  Err: Double;
  C: TMoeConfig;
  S: TParamStore;
  M: TModel;
  X, Y, G, U, Hdn, Dq, Manual: TSingleArray;
  Rng: TRng;
begin
  Section('silu derivative');
  Err := 0;
  for I := 0 to High(Xs) do
    Err := Max(Err, Abs(SiluDx(Xs[I]) - DxNum(Xs[I], 1e-3)));
  Check(Err < 1e-3, Format('SiluDx vs central difference, max err %.2e', [Err]));
  Check(Silu(-100) = 0, 'silu cutoff at x < -80 is 0');
  Check(SiluDx(-100) = 0, 'silu derivative uses the same cutoff');
  Check(Abs(SiluDx(0) - 0.5) < 1e-6, 'd(silu)/dx at 0 is 1/2');

  Section('projection shapes');
  C := TinyConfig;
  BuildLayout(S, C);
  InitParams(S, 4);
  BuildModel(M, S);
  Check((M.Layers[0].Moe.ExGate[0].Rows = C.ExpertHidden) and
        (M.Layers[0].Moe.ExGate[0].Cols = C.Dim), 'gate is ExpertHidden x Dim');
  Check((M.Layers[0].Moe.ExUp[0].Rows = C.ExpertHidden) and
        (M.Layers[0].Moe.ExUp[0].Cols = C.Dim), 'up is ExpertHidden x Dim');
  Check((M.Layers[0].Moe.ExDown[0].Rows = C.Dim) and
        (M.Layers[0].Moe.ExDown[0].Cols = C.ExpertHidden), 'down is Dim x ExpertHidden');

  Section('swiglu forward, float activations');
  C.ActQuant := False;
  RngSeed(Rng, 9);
  SetLength(X, C.Dim); SetLength(Y, C.Dim);
  SetLength(G, C.ExpertHidden); SetLength(U, C.ExpertHidden); SetLength(Hdn, C.ExpertHidden);
  SetLength(Dq, C.ExpertHidden * C.Dim); SetLength(Manual, C.Dim);
  for I := 0 to C.Dim - 1 do X[I] := RngNormal(Rng);
  RoutedExpertForward(C, M.Layers[0].Moe, 0, @X[0], @Y[0]);
  DequantTernary(M.Layers[0].Moe.ExGate[0], @Dq[0]);
  MatVec(@Dq[0], C.ExpertHidden, C.Dim, @X[0], @G[0]);
  DequantTernary(M.Layers[0].Moe.ExUp[0], @Dq[0]);
  MatVec(@Dq[0], C.ExpertHidden, C.Dim, @X[0], @U[0]);
  for I := 0 to C.ExpertHidden - 1 do Hdn[I] := Silu(G[I]) * U[I];
  SetLength(Dq, C.Dim * C.ExpertHidden);
  DequantTernary(M.Layers[0].Moe.ExDown[0], @Dq[0]);
  MatVec(@Dq[0], C.Dim, C.ExpertHidden, @Hdn[0], @Manual[0]);
  Err := MaxAbsDiff(@Y[0], @Manual[0], C.Dim);
  Check(Err < 1e-4, Format('expert = down(silu(gate x) * up x), max diff %.2e', [Err]));
  Finish('test_swiglu');
end.
