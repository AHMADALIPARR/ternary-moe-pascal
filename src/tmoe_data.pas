// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Byte-level text data: load a file, split train/eval, draw random windows.
unit tmoe_data;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor, tmoe_train;

type
  TTextData = record
    Train, Eval: TIntArray;
  end;

procedure LoadText(const FileName: string; EvalFraction: Single; out D: TTextData);
// NSeq windows of SeqLen+1 tokens, back to back, as TrainStep expects.
procedure SampleBatch(const Src: TIntArray; var Rng: TRng; NSeq, SeqLen: Integer; Dest: PInteger);
// Mean CE over consecutive non-overlapping windows of the eval split.
function EvalLoss(var Tr: TTrainer; const Src: TIntArray; SeqLen, MaxWindows: Integer): Single;

implementation

uses SysUtils, Classes;

procedure LoadText(const FileName: string; EvalFraction: Single; out D: TTextData);
var
  F: TFileStream;
  B: TBytes;
  N, NE, I: Integer;
begin
  F := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    N := F.Size;
    SetLength(B, N);
    if N > 0 then F.ReadBuffer(B[0], N);
  finally
    F.Free;
  end;
  NE := Round(N * EvalFraction);
  D.Train := nil; D.Eval := nil;
  SetLength(D.Train, N - NE);
  SetLength(D.Eval, NE);
  for I := 0 to N - NE - 1 do D.Train[I] := B[I];
  for I := 0 to NE - 1 do D.Eval[I] := B[N - NE + I];
end;

procedure SampleBatch(const Src: TIntArray; var Rng: TRng; NSeq, SeqLen: Integer; Dest: PInteger);
var I, J, P: Integer;
begin
  for I := 0 to NSeq - 1 do
  begin
    P := RngInt(Rng, Length(Src) - SeqLen - 1);
    for J := 0 to SeqLen do Dest[I * (SeqLen + 1) + J] := Src[P + J];
  end;
end;

function EvalLoss(var Tr: TTrainer; const Src: TIntArray; SeqLen, MaxWindows: Integer): Single;
var
  W, N: Integer;
  S: Double;
begin
  PrepareWeights(Tr);
  S := 0; N := 0;
  W := 0;
  while (W + SeqLen + 1 <= Length(Src)) and (N < MaxWindows) do
  begin
    ForwardSeq(Tr, @Src[W], @Src[W + 1], SeqLen);
    S := S + Tr.LastCE;
    Inc(N);
    Inc(W, SeqLen);
  end;
  if N = 0 then Exit(0);
  Result := S / N;
end;

end.
