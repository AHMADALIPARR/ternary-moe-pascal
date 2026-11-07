// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Sample text from a checkpoint with the packed 2-bit experts.
program tmoe_generate;

{$mode objfpc}{$H+}

uses SysUtils, tmoe_config, tmoe_tensor, tmoe_params, tmoe_ternary, tmoe_model,
  tmoe_kvcache, tmoe_sampler, tmoe_checkpoint, tmoe_moe, tmoe_args;

var
  S: TParamStore;
  P: TTernaryArray;
  M: TModel;
  Cache: TKVCache;
  Smp: TSampler;
  Logits: TSingleArray;
  Tok: TIntArray;
  Prompt: string;
  N, I, Next: Integer;
begin
  if HasFlag('help') then
  begin
    WriteLn('tmoe_generate [--ckpt build/tiny.ckpt] [--prompt "The keeper"] [--n 200]');
    WriteLn('              [--temp 0.8] [--topk 40] [--topp 0.95] [--seed 1] [--cpu]');
    WriteLn('routed experts use the CUDA library when it loads; --cpu keeps the CPU path');
    Halt(0);
  end;
  LoadCheckpoint(ArgStr('ckpt', 'build/tiny.ckpt'), S, P);
  if not HasFlag('cpu') then MoeSetDevice(True);
  BuildModel(M, S);
  UsePackedExperts(M, P);
  InitKVCache(Cache, M.Cfg);
  InitSampler(Smp, ArgFloat('temp', 0.8), ArgInt('topk', 40), ArgFloat('topp', 0.95), ArgInt('seed', 1));
  Prompt := ArgStr('prompt', 'The keeper');
  N := ArgInt('n', 200);
  SetLength(Tok, Length(Prompt));
  for I := 1 to Length(Prompt) do Tok[I - 1] := Ord(Prompt[I]);
  SetLength(Logits, M.Cfg.Vocab);
  WriteLn(StdErr, 'routed experts: ', MoeDeviceStatus);
  Write(Prompt);
  ModelForward(M, Cache, @Tok[0], Length(Tok), @Logits[0], True);
  for I := 1 to N do
  begin
    if Cache.Len >= M.Cfg.MaxSeq then Break;
    Next := SampleToken(Smp, @Logits[0], M.Cfg.Vocab);
    Write(Chr(Next));
    ModelForward(M, Cache, @Next, 1, @Logits[0], True);
  end;
  WriteLn;
end.
