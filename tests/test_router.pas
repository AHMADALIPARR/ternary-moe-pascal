// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Router: top-k selection, gates, capacity, load-balance loss and gradients.
program test_router;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_tensor, tmoe_router, testutil;

var
  Rng: TRng;
  L, P, dP, dL, dL2, G: TSingleArray;
  R: TRouting;
  Idx: TIntArray;
  T, E, K, I, J, Bad, Cap, Kept: Integer;
  A, Want, Err, H, Lp, Lm, Num: Double;
  Sorted: Boolean;

function GateObjective(const PP: TSingleArray; const Sel: TIntArray; KK: Integer; const W: TSingleArray): Double;
var S: Double; Q: Integer;
begin
  S := 0;
  for Q := 0 to KK - 1 do S := S + PP[Sel[Q]];
  Result := 0;
  for Q := 0 to KK - 1 do Result := Result + W[Q] * PP[Sel[Q]] / S;
end;

begin
  RngSeed(Rng, 3);
  Section('top-k');
  SetLength(P, 8); SetLength(Idx, 3);
  P[0] := 0.1; P[1] := 0.3; P[2] := 0.05; P[3] := 0.3; P[4] := 0.2; P[5] := 0.0; P[6] := 0.04; P[7] := 0.01;
  TopK(@P[0], 8, 3, @Idx[0]);
  Check((Idx[0] = 1) and (Idx[1] = 3) and (Idx[2] = 4), Format('top3 = %d,%d,%d (want 1,3,4; tie -> lower index)', [Idx[0], Idx[1], Idx[2]]));
  Bad := 0;
  SetLength(P, 16); SetLength(Idx, 4);
  for J := 1 to 200 do
  begin
    for I := 0 to 15 do P[I] := RngUniform(Rng);
    TopK(@P[0], 16, 4, @Idx[0]);
    Sorted := True;
    for I := 1 to 3 do if P[Idx[I]] > P[Idx[I - 1]] then Sorted := False;
    for I := 0 to 15 do
      if (I <> Idx[0]) and (I <> Idx[1]) and (I <> Idx[2]) and (I <> Idx[3]) and (P[I] > P[Idx[3]]) then Sorted := False;
    if not Sorted then Inc(Bad);
  end;
  Check(Bad = 0, Format('200 random top-4 of 16: %d wrong', [Bad]));

  Section('gates');
  T := 6; E := 4; K := 2;
  SetLength(L, T * E);
  for I := 0 to High(L) do L[I] := RngNormal(Rng);
  Route(@L[0], T, E, K, 0, False, R);
  Err := 0;
  for I := 0 to T - 1 do
  begin
    A := 0;
    for J := 0 to E - 1 do A := A + R.Probs[I * E + J];
    Err := Max(Err, Abs(A - 1));
  end;
  Check(Err < 1e-6, Format('softmax rows sum to 1 (max err %.1e)', [Err]));
  Bad := 0;
  for I := 0 to T * K - 1 do
    if R.Gate[I] <> R.Probs[(I div K) * E + R.Idx[I]] then Inc(Bad);
  Check(Bad = 0, 'plain gates = selected probabilities');
  Route(@L[0], T, E, K, 0, True, R);
  Err := 0;
  for I := 0 to T - 1 do Err := Max(Err, Abs(R.Gate[I * K] + R.Gate[I * K + 1] - 1));
  Check(Err < 1e-6, Format('renormalised gates sum to 1 (max err %.1e)', [Err]));

  Section('capacity');
  // every token prefers expert 0
  T := 10; E := 4; K := 1;
  SetLength(L, T * E);
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = 0, 5.0, 0.0);
  Cap := ExpertCapacity(T, E, K, 1.25);
  Check(Cap = 4, Format('capacity ceil(1.25*10*1/4) = %d', [Cap]));
  Route(@L[0], T, E, K, 1.25, False, R);
  Kept := 0;
  for I := 0 to T - 1 do if R.Kept[I] then Inc(Kept);
  Check((Kept = 4) and R.Kept[0] and R.Kept[3] and not R.Kept[4], Format('kept %d slots, first come first served', [Kept]));
  Check((R.Counts[0] = 10) and (R.Load[0] = 4), Format('counts[0]=%d load[0]=%d', [R.Counts[0], R.Load[0]]));
  Route(@L[0], T, E, K, 0, False, R);
  Kept := 0;
  for I := 0 to T - 1 do if R.Kept[I] then Inc(Kept);
  Check(Kept = 10, 'capacity factor 0 keeps everything');
  // second choices yield to first choices
  T := 4; E := 2; K := 2;
  SetLength(L, T * E);
  for I := 0 to T - 1 do begin L[I * E] := IfThen(I < 2, 1.0, 0.0); L[I * E + 1] := IfThen(I < 2, 0.0, 1.0); end;
  Route(@L[0], T, E, K, 0.5, False, R);   // cap = ceil(0.5*4*2/2) = 2
  Bad := 0;
  for I := 0 to T - 1 do
  begin
    if not R.Kept[I * K] then Inc(Bad);
    if R.Kept[I * K + 1] then Inc(Bad);
  end;
  Check(Bad = 0, 'first choices fill capacity before second choices');

  Section('aux loss');
  T := 8; E := 4; K := 1;
  SetLength(L, T * E);
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = I mod E, 1e-3, 0.0);
  Route(@L[0], T, E, K, 0, False, R);
  A := AuxLoss(R, 0.01);
  Check(Abs(A - 0.01) < 1e-6, Format('balanced routing: aux = %.6f (want coef = 0.01)', [A]));
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = 0, 20.0, 0.0);
  Route(@L[0], T, E, K, 0, False, R);
  A := AuxLoss(R, 0.01);
  Check(Abs(A - 0.04) < 1e-6, Format('collapsed routing: aux = %.6f (want coef*E = 0.04)', [A]));
  // gradient: d aux / d p[t,e] = coef * E * f_e / T, f held constant
  T := 5; E := 4; K := 2;
  SetLength(L, T * E);
  for I := 0 to High(L) do L[I] := RngNormal(Rng);
  Route(@L[0], T, E, K, 0, False, R);
  SetLength(dP, T * E);
  for I := 0 to High(dP) do dP[I] := 0;
  AuxLossBackward(R, 0.01, @dP[0]);
  Err := 0;
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do
    begin
      H := 1e-3;
      R.Probs[I * E + J] := R.Probs[I * E + J] + H; Lp := AuxLoss(R, 0.01);
      R.Probs[I * E + J] := R.Probs[I * E + J] - 2 * H; Lm := AuxLoss(R, 0.01);
      R.Probs[I * E + J] := R.Probs[I * E + J] + H;
      Err := Max(Err, Abs((Lp - Lm) / (2 * H) - dP[I * E + J]));
    end;
  Check(Err < 1e-5, Format('aux gradient vs finite difference, max err %.1e', [Err]));

  Section('backward');
  // softmax backward
  SetLength(L, 6); SetLength(P, 6); SetLength(dP, 6); SetLength(dL, 6); SetLength(G, 6);
  for I := 0 to 5 do begin L[I] := RngNormal(Rng); G[I] := RngNormal(Rng); end;
  Move(L[0], P[0], 6 * 4); SoftmaxInPlace(@P[0], 6);
  SoftmaxBackward(@P[0], @G[0], 6, @dL[0]);
  Err := 0;
  SetLength(dL2, 6);
  for I := 0 to 5 do
  begin
    H := 1e-3;
    L[I] := L[I] + H; Move(L[0], dL2[0], 24); SoftmaxInPlace(@dL2[0], 6);
    Lp := 0; for J := 0 to 5 do Lp := Lp + G[J] * dL2[J];
    L[I] := L[I] - 2 * H; Move(L[0], dL2[0], 24); SoftmaxInPlace(@dL2[0], 6);
    Lm := 0; for J := 0 to 5 do Lm := Lm + G[J] * dL2[J];
    L[I] := L[I] + H;
    Err := Max(Err, Abs((Lp - Lm) / (2 * H) - dL[I]));
  end;
  Check(Err < 1e-4, Format('softmax backward vs finite difference, max err %.1e', [Err]));
  // renormalised gate backward
  T := 1; E := 6; K := 3;
  Route(@L[0], T, E, K, 0, True, R);
  SetLength(Idx, K);
  for J := 0 to K - 1 do Idx[J] := R.Idx[J];
  for I := 0 to 5 do dP[I] := 0;
  SetLength(G, K);
  for J := 0 to K - 1 do G[J] := RngNormal(Rng);
  GateBackward(R, 0, True, @G[0], @dP[0]);
  Move(R.Probs[0], P[0], 24);
  Err := 0;
  for J := 0 to K - 1 do
  begin
    I := Idx[J]; H := 1e-3;
    P[I] := P[I] + H; Lp := GateObjective(P, Idx, K, G);
    P[I] := P[I] - 2 * H; Lm := GateObjective(P, Idx, K, G);
    P[I] := P[I] + H;
    Num := (Lp - Lm) / (2 * H);
    Err := Max(Err, Abs(Num - dP[I]));
  end;
  Check(Err < 1e-3, Format('renorm gate backward vs finite difference, max err %.1e', [Err]));
  Finish('test_router');
end.
