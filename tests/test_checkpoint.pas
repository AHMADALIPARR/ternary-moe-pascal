// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Checkpoint save/load roundtrip, packed experts, and corruption detection.
program test_checkpoint;

{$mode objfpc}{$H+}

uses SysUtils, Classes, tmoe_config, tmoe_tensor, tmoe_params, tmoe_ternary, tmoe_model,
  tmoe_kvcache, tmoe_train, tmoe_checkpoint, testutil;

function FileSizeOf(const N: string): Int64;
var Fs: TFileStream;
begin
  Fs := TFileStream.Create(N, fmOpenRead);
  try Result := Fs.Size; finally Fs.Free; end;
end;

function SameFloats(const A, B: TSingleArray): Boolean;
begin
  Result := (Length(A) = Length(B)) and ((Length(A) = 0) or CompareMem(@A[0], @B[0], Length(A) * SizeOf(Single)));
end;

var
  C: TMoeConfig;
  S, S2: TParamStore;
  Tr: TTrainer;
  M, M2: TModel;
  P: TTernaryArray;
  Cache: TKVCache;
  Batch, Tok: TIntArray;
  LA, LB: TSingleArray;
  I, L, E, Bad: Integer;
  FN: string;
  F: TFileStream;
  B: Byte;
  Raised: Boolean;
  Rng: TRng;
begin
  FN := GetTempDir + 'tmoe_test_' + IntToStr(GetProcessID) + '.ckpt';
  C := TinyConfig;
  C.TopK := 2;
  BuildLayout(S, C);
  InitParams(S, 8);
  InitTrainer(Tr, S, DefaultTrainOpts);
  RngSeed(Rng, 1);
  SetLength(Batch, 2 * 33);
  for I := 0 to High(Batch) do Batch[I] := RngInt(Rng, C.Vocab);
  for I := 1 to 3 do TrainStep(Tr, @Batch[0], 2, 32);

  Section('roundtrip');
  SaveCheckpoint(FN, S, True);
  WriteLn('  file ', FN, ' (', FileSizeOf(FN), ' bytes)');
  LoadCheckpoint(FN, S2, P);
  Check((ConfigSummary(S.Cfg) = ConfigSummary(S2.Cfg)) and (S.Cfg.AuxCoef = S2.Cfg.AuxCoef) and
    (S.Cfg.CapacityFactor = S2.Cfg.CapacityFactor) and (S.Cfg.RopeTheta = S2.Cfg.RopeTheta) and
    (S.Cfg.NormEps = S2.Cfg.NormEps) and (S.Cfg.RenormTopK = S2.Cfg.RenormTopK) and (S.Cfg.MaxSeq = S2.Cfg.MaxSeq), 'config');
  Check((S2.Step = S.Step) and (S.Step = 3), Format('optimizer step %d', [S2.Step]));
  Check(SameFloats(S.Data, S2.Data), Format('%d parameters bit-identical', [S.Total]));
  Check(SameFloats(S.M, S2.M) and SameFloats(S.V, S2.V), 'AdamW moments bit-identical');
  Check(Length(P) = C.Layers * 3 * C.Experts, Format('%d packed expert matrices', [Length(P)]));

  BuildModel(M, S);
  BuildModel(M2, S2);
  Bad := 0;
  for L := 0 to C.Layers - 1 do
    for E := 0 to C.Experts - 1 do
    begin
      if not CompareMem(@M.Layers[L].Moe.ExGate[E].Bits[0], @P[L * 3 * C.Experts + E].Bits[0], Length(P[L * 3 * C.Experts + E].Bits)) then Inc(Bad);
      if not CompareMem(@M.Layers[L].Moe.ExDown[E].Scales[0], @P[L * 3 * C.Experts + 2 * C.Experts + E].Scales[0], Length(P[L * 3 * C.Experts + 2 * C.Experts + E].Scales) * 4) then Inc(Bad);
    end;
  Check(Bad = 0, 'stored packed experts == packing the loaded latents');
  UsePackedExperts(M2, P);
  SetLength(Tok, 9);
  for I := 0 to 8 do Tok[I] := Batch[I];
  SetLength(LA, 9 * C.Vocab); SetLength(LB, 9 * C.Vocab);
  InitKVCache(Cache, C);
  ModelForward(M, Cache, @Tok[0], 9, @LA[0], False);
  ResetKVCache(Cache);
  ModelForward(M2, Cache, @Tok[0], 9, @LB[0], False);
  Check(SameFloats(LA, LB), 'logits of reloaded model bit-identical');

  Section('corruption');
  F := TFileStream.Create(FN, fmOpenReadWrite);
  try
    F.Position := F.Size div 2;
    F.ReadBuffer(B, 1);
    B := B xor $10;
    F.Position := F.Size div 2;
    F.WriteBuffer(B, 1);
  finally
    F.Free;
  end;
  Raised := False;
  try
    LoadCheckpoint(FN, S2, P);
  except
    on Ex: Exception do
    begin
      Raised := True;
      WriteLn('  load error: ', Ex.Message);
    end;
  end;
  Check(Raised, 'flipped bit is rejected by the checksum');
  DeleteFile(FN);
  Finish('test_checkpoint');
end.
