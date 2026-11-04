// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Router: softmax over expert logits, top-k selection, per-expert capacity
// and the Switch-style load-balance loss  L = a * E * sum_i f_i * P_i
// (f_i = share of routed slots that picked expert i, P_i = mean router
// probability of expert i). Backward helpers for the trainer live here too.
unit tmoe_router;

{$mode objfpc}{$H+}

interface

uses tmoe_tensor;

type
  TRouting = record
    T, E, K: Integer;
    Probs: TSingleArray;   // T x E
    Idx: TIntArray;        // T x K, expert ids, best first
    Gate: TSingleArray;    // T x K, gate weights
    Kept: array of Boolean; // T x K, False when capacity dropped the slot
    Counts: TIntArray;     // E, slots chosen per expert (before capacity)
    Load: TIntArray;       // E, slots kept per expert (after capacity)
  end;

// Indices of the K largest values, descending, ties to the lower index.
procedure TopK(P: PSingle; N, K: Integer; Idx: PInteger);
function ExpertCapacity(T, E, K: Integer; CapacityFactor: Single): Integer;
// Logits (T x E) -> routing. Logits are not modified.
procedure Route(Logits: PSingle; T, E, K: Integer; CapacityFactor: Single;
  Renorm: Boolean; out R: TRouting);
function AuxLoss(const R: TRouting; Coef: Single): Single;
// dProbs (T x E) += d(aux)/d(probs)
procedure AuxLossBackward(const R: TRouting; Coef: Single; dProbs: PSingle);
// dProbs row += d(gates)/d(probs) applied to dGate row
procedure GateBackward(const R: TRouting; Tok: Integer; Renorm: Boolean; dGate, dProbs: PSingle);
// dLogits = p * (dp - <dp, p>)
procedure SoftmaxBackward(P, dP: PSingle; N: Integer; dLogits: PSingle);

implementation

uses Math;

procedure TopK(P: PSingle; N, K: Integer; Idx: PInteger);
var
  I, J, Best: Integer;
  Used: array of Boolean;
begin
  SetLength(Used, N);
  for J := 0 to K - 1 do
  begin
    Best := -1;
    for I := 0 to N - 1 do
      if (not Used[I]) and ((Best < 0) or (P[I] > P[Best])) then Best := I;
    Used[Best] := True;
    Idx[J] := Best;
  end;
end;

function ExpertCapacity(T, E, K: Integer; CapacityFactor: Single): Integer;
begin
  if CapacityFactor <= 0 then Exit(T);
  Result := Ceil(CapacityFactor * T * K / E);
  if Result < 1 then Result := 1;
end;

procedure Route(Logits: PSingle; T, E, K: Integer; CapacityFactor: Single;
  Renorm: Boolean; out R: TRouting);
var
  I, J, Ex, Cap: Integer;
  S: Single;
begin
  R.T := T; R.E := E; R.K := K;
  R.Probs := nil; R.Idx := nil; R.Gate := nil; R.Kept := nil; R.Counts := nil; R.Load := nil;
  SetLength(R.Probs, T * E);
  SetLength(R.Idx, T * K);
  SetLength(R.Gate, T * K);
  SetLength(R.Kept, T * K);
  SetLength(R.Counts, E);
  SetLength(R.Load, E);
  Move(Logits^, R.Probs[0], T * E * SizeOf(Single));
  for I := 0 to T - 1 do
  begin
    SoftmaxInPlace(@R.Probs[I * E], E);
    TopK(@R.Probs[I * E], E, K, @R.Idx[I * K]);
    S := 0;
    for J := 0 to K - 1 do S := S + R.Probs[I * E + R.Idx[I * K + J]];
    for J := 0 to K - 1 do
    begin
      R.Gate[I * K + J] := R.Probs[I * E + R.Idx[I * K + J]];
      if Renorm then R.Gate[I * K + J] := R.Gate[I * K + J] / S;
      Inc(R.Counts[R.Idx[I * K + J]]);
    end;
  end;
  // capacity: first choices of every token get priority over second choices
  Cap := ExpertCapacity(T, E, K, CapacityFactor);
  for J := 0 to K - 1 do
    for I := 0 to T - 1 do
    begin
      Ex := R.Idx[I * K + J];
      R.Kept[I * K + J] := R.Load[Ex] < Cap;
      if R.Kept[I * K + J] then Inc(R.Load[Ex]);
    end;
end;

function AuxLoss(const R: TRouting; Coef: Single): Single;
var
  I, Ex: Integer;
  F, P: Double;
  S: Double;
begin
  S := 0;
  for Ex := 0 to R.E - 1 do
  begin
    F := R.Counts[Ex] / (R.T * R.K);
    P := 0;
    for I := 0 to R.T - 1 do P := P + R.Probs[I * R.E + Ex];
    P := P / R.T;
    S := S + F * P;
  end;
  Result := Coef * R.E * S;
end;

procedure AuxLossBackward(const R: TRouting; Coef: Single; dProbs: PSingle);
var
  I, Ex: Integer;
  G: Single;
begin
  for Ex := 0 to R.E - 1 do
  begin
    G := Coef * R.E * (R.Counts[Ex] / (R.T * R.K)) / R.T;
    for I := 0 to R.T - 1 do
      dProbs[I * R.E + Ex] := dProbs[I * R.E + Ex] + G;
  end;
end;

procedure GateBackward(const R: TRouting; Tok: Integer; Renorm: Boolean; dGate, dProbs: PSingle);
var
  J, Ex: Integer;
  S, Cross: Single;
begin
  if not Renorm then
  begin
    for J := 0 to R.K - 1 do
    begin
      Ex := R.Idx[Tok * R.K + J];
      dProbs[Ex] := dProbs[Ex] + dGate[J];
    end;
    Exit;
  end;
  // g_j = p_j / S,  S = sum_sel p
  S := 0; Cross := 0;
  for J := 0 to R.K - 1 do S := S + R.Probs[Tok * R.E + R.Idx[Tok * R.K + J]];
  for J := 0 to R.K - 1 do
    Cross := Cross + dGate[J] * R.Probs[Tok * R.E + R.Idx[Tok * R.K + J]];
  for J := 0 to R.K - 1 do
  begin
    Ex := R.Idx[Tok * R.K + J];
    dProbs[Ex] := dProbs[Ex] + dGate[J] / S - Cross / (S * S);
  end;
end;

procedure SoftmaxBackward(P, dP: PSingle; N: Integer; dLogits: PSingle);
var
  I: Integer;
  S: Single;
begin
  S := 0;
  for I := 0 to N - 1 do S := S + P[I] * dP[I];
  for I := 0 to N - 1 do dLogits[I] := P[I] * (dP[I] - S);
end;

end.
