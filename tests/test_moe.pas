// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// MoE layer equals the manual sum of shared + gated routed experts; prefill
// equals step-by-step decoding; the packed model matches the training forward.
program test_moe;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_config, tmoe_tensor, tmoe_params, tmoe_router, tmoe_moe,
  tmoe_model, tmoe_kvcache, tmoe_train, testutil;

procedure CheckManualSum(const Name: string; C: TMoeConfig);
var
  S: TParamStore;
  M: TModel;
  Rng: TRng;
  X, Y, Ref, Tmp: TSingleArray;
  R: TRouting;
  T, I, J, E, Sl, Dropped: Integer;
  Err: Single;
begin
  BuildLayout(S, C);
  InitParams(S, 11);
  BuildModel(M, S);
  RngSeed(Rng, 5);
  T := 23;
  SetLength(X, T * C.Dim); SetLength(Y, T * C.Dim); SetLength(Ref, T * C.Dim); SetLength(Tmp, C.Dim);
  for I := 0 to High(X) do X[I] := RngNormal(Rng);
  MoeForward(C, M.Layers[0].Moe, @X[0], T, @Y[0], R);
  Dropped := 0;
  for I := 0 to T - 1 do
  begin
    SharedExpertForward(C, M.Layers[0].Moe, @X[I * C.Dim], @Ref[I * C.Dim]);
    // same order as the layer: experts ascending
    for E := 0 to C.Experts - 1 do
      for J := 0 to C.TopK - 1 do
      begin
        Sl := I * C.TopK + J;
        if R.Idx[Sl] <> E then Continue;
        if not R.Kept[Sl] then begin Inc(Dropped); Continue; end;
        RoutedExpertForward(C, M.Layers[0].Moe, E, @X[I * C.Dim], @Tmp[0]);
        for Sl := 0 to C.Dim - 1 do
          Ref[I * C.Dim + Sl] := Ref[I * C.Dim + Sl] + R.Gate[I * C.TopK + J] * Tmp[Sl];
      end;
  end;
  Err := MaxAbsDiff(@Y[0], @Ref[0], T * C.Dim);
  Check(Err = 0, Format('%s: layer == shared + sum gate*expert, max diff %g (%d slots dropped)', [Name, Err, Dropped]));
end;

var
  C: TMoeConfig;
  S: TParamStore;
  M: TModel;
  Cache: TKVCache;
  Tr: TTrainer;
  Tok: TIntArray;
  LA, LB: TSingleArray;
  T, I: Integer;
  Err, Mx: Single;
  Loss: Single;
begin
  Section('manual sum');
  C := TinyConfig;
  CheckManualSum('top1 int8', C);
  C.TopK := 2; C.CapacityFactor := 1.0;
  CheckManualSum('top2 cap1.0 int8', C);
  C.RenormTopK := True;
  CheckManualSum('top2 renorm', C);
  C.ActQuant := False;
  CheckManualSum('top2 f32 acts', C);

  Section('prefill vs decode');
  C := TinyConfig;
  BuildLayout(S, C);
  InitParams(S, 21);
  BuildModel(M, S);
  T := 17;
  SetLength(Tok, T + 1);
  for I := 0 to T do Tok[I] := (I * 37 + 11) mod C.Vocab;
  SetLength(LA, T * C.Vocab); SetLength(LB, T * C.Vocab);
  InitKVCache(Cache, C);
  ModelForward(M, Cache, @Tok[0], T, @LA[0], False);
  Check(Cache.Len = T, Format('cache length %d after prefill', [Cache.Len]));
  ResetKVCache(Cache);
  for I := 0 to T - 1 do
    ModelForward(M, Cache, @Tok[I], 1, @LB[I * C.Vocab], True);
  Err := MaxAbsDiff(@LA[0], @LB[0], T * C.Vocab);
  Check(Err < 1e-4, Format('prefill logits == token-by-token decode, max diff %.2e', [Err]));

  Section('inference vs training forward');
  InitTrainer(Tr, S, DefaultTrainOpts);
  PrepareWeights(Tr);
  Loss := ForwardSeq(Tr, @Tok[0], @Tok[1], T);
  Mx := 0;
  for I := 0 to T * C.Vocab - 1 do Mx := Max(Mx, Abs(LA[I]));
  Err := MaxAbsDiff(@LA[0], @Tr.Logits[0], T * C.Vocab);
  // the trainer fake-quantises in fp32 while inference packs to 2 bits; they
  // agree up to fp32 rounding of the same ternary weights and int8 codes
  Check(Err < 1e-3 * Max(1, Mx), Format('packed model vs trainer logits: max diff %.2e (max |logit| %.2f)', [Err, Mx]));
  Check(not IsNan(Loss) and (Loss > 0), Format('training forward loss %.4f', [Loss]));
  Finish('test_moe');
end.
