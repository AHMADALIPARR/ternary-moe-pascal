// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Backward pass against central finite differences along a random direction,
// one check per tensor. Experts and activations stay in float here so the
// loss is smooth; the STE path reuses the same backward.
program test_gradcheck;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_config, tmoe_tensor, tmoe_params, tmoe_train, testutil;

var
  C: TMoeConfig;
  S: TParamStore;
  Tr: TTrainer;
  Rng: TRng;
  Tok: TIntArray;
  Dir: array of Double;
  T, I, P, N: Integer;
  Base: PtrInt;
  Nrm, Ana, Num, Lp, Lm, H, Rel, Worst: Double;

function Loss: Double;
begin
  PrepareWeights(Tr);
  Result := ForwardSeq(Tr, @Tok[0], @Tok[1], T);
end;

begin
  C := TinyConfig;
  C.TernaryExperts := False;
  C.ActQuant := False;
  C.TopK := 2;
  C.RenormTopK := True;
  C.AuxCoef := 0.05;
  BuildLayout(S, C);
  InitParams(S, 99);
  InitTrainer(Tr, S, DefaultTrainOpts);
  RngSeed(Rng, 1234);
  T := 12;
  SetLength(Tok, T + 1);
  for I := 0 to T do Tok[I] := RngInt(Rng, C.Vocab);
  ZeroGrad(Tr);
  Loss;
  BackwardSeq(Tr, @Tok[0], @Tok[1], T, 1.0);
  WriteLn('loss ', Loss:0:6);
  Worst := 0;
  for P := 0 to High(S.Info) do
  begin
    N := S.Info[P].Count * S.Info[P].Rows * S.Info[P].Cols;
    Base := S.Info[P].Offset;
    SetLength(Dir, N);
    Nrm := 0;
    for I := 0 to N - 1 do begin Dir[I] := RngNormal(Rng); Nrm := Nrm + Sqr(Dir[I]); end;
    Nrm := Sqrt(Nrm);
    Ana := 0;
    for I := 0 to N - 1 do
    begin
      Dir[I] := Dir[I] / Nrm;
      Ana := Ana + Dir[I] * S.Grad[Base + I];
    end;
    H := 1e-2;
    for I := 0 to N - 1 do S.Data[Base + I] := S.Data[Base + I] + H * Dir[I];
    Lp := Loss;
    for I := 0 to N - 1 do S.Data[Base + I] := S.Data[Base + I] - 2 * H * Dir[I];
    Lm := Loss;
    for I := 0 to N - 1 do S.Data[Base + I] := S.Data[Base + I] + H * Dir[I];
    Num := (Lp - Lm) / (2 * H);
    Rel := Abs(Ana - Num) / Max(Max(Abs(Ana), Abs(Num)), 1e-3);
    Worst := Max(Worst, Rel);
    Check(Rel < 5e-2, Format('%-22s analytic %10.6f  numeric %10.6f  rel %.1e', [S.Info[P].Name, Ana, Num, Rel]));
  end;
  WriteLn('worst relative error ', Worst:0:5);
  Finish('test_gradcheck');
end.
