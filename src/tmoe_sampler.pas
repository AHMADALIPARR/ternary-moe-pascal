// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Token sampler: greedy, temperature, top-k and top-p (nucleus).
unit tmoe_sampler;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor;

type
  TSampler = record
    Temperature: Single;  // <= 0 means greedy
    TopK: Integer;        // <= 0 means off
    TopP: Single;         // >= 1 means off
    Rng: TRng;
  end;

procedure InitSampler(out S: TSampler; Temperature: Single; TopK: Integer; TopP: Single; Seed: QWord);
function Argmax(Logits: PSingle; N: Integer): Integer;
function SampleToken(var S: TSampler; Logits: PSingle; N: Integer): Integer;

implementation

procedure InitSampler(out S: TSampler; Temperature: Single; TopK: Integer; TopP: Single; Seed: QWord);
begin
  S.Temperature := Temperature;
  S.TopK := TopK;
  S.TopP := TopP;
  RngSeed(S.Rng, Seed);
end;

function Argmax(Logits: PSingle; N: Integer): Integer;
var I: Integer;
begin
  Result := 0;
  for I := 1 to N - 1 do
    if Logits[I] > Logits[Result] then Result := I;
end;

function SampleToken(var S: TSampler; Logits: PSingle; N: Integer): Integer;
var
  Idx: TIntArray;
  P: TSingleArray;
  I, J, Keep, Tmp: Integer;
  Sum, Acc, R: Single;
begin
  if S.Temperature <= 0 then Exit(Argmax(Logits, N));
  SetLength(P, N);
  SetLength(Idx, N);
  for I := 0 to N - 1 do
  begin
    P[I] := Logits[I] / S.Temperature;
    Idx[I] := I;
  end;
  SoftmaxInPlace(@P[0], N);
  // sort indices by probability, descending (insertion sort, N is small)
  for I := 1 to N - 1 do
  begin
    Tmp := Idx[I];
    J := I - 1;
    while (J >= 0) and (P[Idx[J]] < P[Tmp]) do
    begin
      Idx[J + 1] := Idx[J];
      Dec(J);
    end;
    Idx[J + 1] := Tmp;
  end;
  Keep := N;
  if (S.TopK > 0) and (S.TopK < Keep) then Keep := S.TopK;
  if S.TopP < 1 then
  begin
    Acc := 0;
    for I := 0 to Keep - 1 do
    begin
      Acc := Acc + P[Idx[I]];
      if Acc >= S.TopP then
      begin
        Keep := I + 1;
        Break;
      end;
    end;
  end;
  Sum := 0;
  for I := 0 to Keep - 1 do Sum := Sum + P[Idx[I]];
  R := RngUniform(S.Rng) * Sum;
  Acc := 0;
  for I := 0 to Keep - 1 do
  begin
    Acc := Acc + P[Idx[I]];
    if R < Acc then Exit(Idx[I]);
  end;
  Result := Idx[Keep - 1];
end;

end.
