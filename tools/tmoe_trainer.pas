// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Train a ternary MoE on a byte-level text file and write a checkpoint.
program tmoe_trainer;

{$mode objfpc}{$H+}

uses SysUtils, Math, DateUtils, tmoe_config, tmoe_tensor, tmoe_params, tmoe_train, tmoe_data,
  tmoe_checkpoint, tmoe_args;

var
  C: TMoeConfig;
  S: TParamStore;
  Tr: TTrainer;
  D: TTextData;
  Opt: TTrainOpts;
  Rng: TRng;
  Batch: TIntArray;
  Steps, NSeq, SeqLen, I, Every: Integer;
  L: Single;
  T0: TDateTime;
  Secs: Double;
  Data, OutF: string;
begin
  if HasFlag('help') then
  begin
    WriteLn('tmoe_train [--data data/tiny.txt] [--config tiny|small] [--steps 200] [--batch 4]');
    WriteLn('           [--seq 64] [--lr 3e-3] [--seed 1] [--out build/tiny.ckpt]');
    WriteLn('           [--experts N] [--topk K] [--capacity F] [--renorm]');
    Halt(0);
  end;
  C := ArgConfig;
  Data := ArgStr('data', 'data/tiny.txt');
  OutF := ArgStr('out', 'build/tiny.ckpt');
  Steps := ArgInt('steps', 200);
  NSeq := ArgInt('batch', 4);
  SeqLen := ArgInt('seq', 64);
  if SeqLen > C.MaxSeq then SeqLen := C.MaxSeq;
  Opt := DefaultTrainOpts;
  Opt.LR := ArgFloat('lr', Opt.LR);
  Opt.MinLR := Opt.LR / 10;
  Opt.TotalSteps := Steps;
  LoadText(Data, 0.1, D);
  WriteLn(ConfigSummary(C));
  WriteLn(Format('data %s: %d train / %d eval bytes', [Data, Length(D.Train), Length(D.Eval)]));
  BuildLayout(S, C);
  InitParams(S, ArgInt('seed', 1));
  WriteLn(Format('parameters: %d', [S.Total]));
  InitTrainer(Tr, S, Opt);
  RngSeed(Rng, ArgInt('seed', 1) + 1000);
  SetLength(Batch, NSeq * (SeqLen + 1));
  WriteLn(Format('eval ce %.4f', [EvalLoss(Tr, D.Eval, SeqLen, 16)]));
  Every := Steps div 20;
  if Every < 1 then Every := 1;
  T0 := Now;
  for I := 1 to Steps do
  begin
    SampleBatch(D.Train, Rng, NSeq, SeqLen, @Batch[0]);
    L := TrainStep(Tr, @Batch[0], NSeq, SeqLen);
    if (I = 1) or (I mod Every = 0) or (I = Steps) then
      WriteLn(Format('step %5d  loss %.4f  ce %.4f  aux %.5f  |g| %.3f  lr %.2e',
        [I, L, Tr.LastCE, Tr.LastAux, Tr.LastGradNorm, LRAt(Opt, I)]));
  end;
  Secs := MilliSecondsBetween(Now, T0) / 1000;
  WriteLn(Format('%d steps in %.1f s (%.0f train tok/s)', [Steps, Secs, Steps * NSeq * SeqLen / Max(Secs, 1e-3)]));
  WriteLn(Format('eval ce %.4f', [EvalLoss(Tr, D.Eval, SeqLen, 16)]));
  SaveCheckpoint(OutF, S, False);
  WriteLn('wrote ', OutF);
end.
