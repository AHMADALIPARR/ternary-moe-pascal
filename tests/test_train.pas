// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Tiny training run on data/tiny.txt: loss must go down.
program test_train;

{$mode objfpc}{$H+}

uses SysUtils, DateUtils, tmoe_config, tmoe_tensor, tmoe_params, tmoe_train, tmoe_data, testutil;

const
  Steps = 80;
  NSeq = 4;
  SeqLen = 64;

var
  C: TMoeConfig;
  S: TParamStore;
  Tr: TTrainer;
  D: TTextData;
  Rng: TRng;
  Batch: TIntArray;
  Opt: TTrainOpts;
  I: Integer;
  L, First5, Last5, Ev0, Ev1: Double;
  T0: TDateTime;
  Path: string;
begin
  Path := 'data/tiny.txt';
  if ParamCount > 0 then Path := ParamStr(1);
  LoadText(Path, 0.1, D);
  WriteLn('data: ', Length(D.Train), ' train bytes, ', Length(D.Eval), ' eval bytes');
  C := TinyConfig;
  WriteLn(ConfigSummary(C));
  BuildLayout(S, C);
  InitParams(S, 42);
  Opt := DefaultTrainOpts;
  Opt.TotalSteps := Steps;
  InitTrainer(Tr, S, Opt);
  RngSeed(Rng, 2026);
  SetLength(Batch, NSeq * (SeqLen + 1));
  Ev0 := EvalLoss(Tr, D.Eval, SeqLen, 8);
  WriteLn(Format('eval CE before: %.4f', [Ev0]));
  First5 := 0; Last5 := 0;
  T0 := Now;
  for I := 1 to Steps do
  begin
    SampleBatch(D.Train, Rng, NSeq, SeqLen, @Batch[0]);
    L := TrainStep(Tr, @Batch[0], NSeq, SeqLen);
    if I <= 5 then First5 := First5 + Tr.LastCE / 5;
    if I > Steps - 5 then Last5 := Last5 + Tr.LastCE / 5;
    if (I = 1) or (I mod 10 = 0) then
      WriteLn(Format('step %3d  loss %.4f  ce %.4f  aux %.5f  |g| %.3f  lr %.2e',
        [I, L, Tr.LastCE, Tr.LastAux, Tr.LastGradNorm, LRAt(Opt, I)]));
  end;
  WriteLn(Format('%d steps x %d x %d tokens in %.1f s', [Steps, NSeq, SeqLen, MilliSecondsBetween(Now, T0) / 1000]));
  Ev1 := EvalLoss(Tr, D.Eval, SeqLen, 8);
  WriteLn(Format('eval CE after: %.4f', [Ev1]));
  Check(Last5 < First5 - 0.5, Format('train CE mean of first 5 steps %.4f -> last 5 steps %.4f', [First5, Last5]));
  Check(Ev1 < Ev0 - 0.5, Format('eval CE %.4f -> %.4f', [Ev0, Ev1]));
  Finish('test_train');
end.
