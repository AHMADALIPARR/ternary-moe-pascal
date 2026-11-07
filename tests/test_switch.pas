// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Switch load-balance loss against an independent double-precision sum.
// Top-1 is the published formula. Top-k is the documented generalization.
// CPU only: this does not touch the packed GEMM.
program test_switch;

{$mode objfpc}{$H+}

uses SysUtils, Math, tmoe_tensor, tmoe_router, testutil;

function RefAux(const R: TRouting; Coef: Double): Double;
var
  E, I: Integer;
  F, P, S: Double;
begin
  if (R.T <= 0) or (R.K <= 0) or (R.E <= 0) then Exit(0);
  S := 0;
  for E := 0 to R.E - 1 do
  begin
    F := R.Counts[E] / (Double(R.T) * R.K);
    P := 0;
    for I := 0 to R.T - 1 do P := P + R.Probs[I * R.E + E];
    P := P / R.T;
    S := S + F * P;
  end;
  Result := Coef * R.E * S;
end;

var
  L, dP: TSingleArray;
  R: TRouting;
  T, E, K, I, J, Kept: Integer;
  A, Err: Double;
  Idx: TIntArray;
begin
  Section('no tokens');
  L := nil;
  SetLength(L, 1);
  Route(@L[0], 0, 4, 1, 1.25, False, R);
  Check((R.T = 0) and (Length(R.Counts) = 4) and (R.Counts[0] = 0), 'T=0 route does not divide');
  Check(AuxLoss(R, 0.01) = 0, 'T=0 aux is 0');
  dP := nil;
  SetLength(dP, 1);
  dP[0] := 123;
  AuxLossBackward(R, 0.01, @dP[0]);
  Check(dP[0] = 123, 'T=0 backward does not write');

  Section('ties and extreme logits');
  SetLength(L, 6); SetLength(Idx, 3);
  for I := 0 to 5 do L[I] := 4;
  TopK(@L[0], 6, 3, @Idx[0]);
  Check((Idx[0] = 0) and (Idx[1] = 1) and (Idx[2] = 2), 'equal logits: lowest indices win');
  L[0] := 1e20; L[1] := -1e20; L[2] := 1e20; L[3] := 0; L[4] := -1e10; L[5] := 50;
  Route(@L[0], 1, 6, 2, 0, False, R);
  A := 0;
  for I := 0 to 5 do A := A + R.Probs[I];
  Check((Abs(A - 1) < 1e-5) and (not IsNan(R.Probs[0])),
    Format('extreme logits softmax sums to %.6f', [A]));
  Check((R.Idx[0] = 0) and (R.Idx[1] = 2),
    Format('tie at +1e20 goes to the lower index (%d,%d)', [R.Idx[0], R.Idx[1]]));

  Section('top-1 switch formula');
  T := 8; E := 4; K := 1;
  SetLength(L, T * E);
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = I mod E, 1e-3, 0);
  Route(@L[0], T, E, K, 0, False, R);
  A := AuxLoss(R, 0.01);
  Check(Abs(A - 0.01) < 1e-6, Format('balanced top-1 aux %.6f = alpha', [A]));
  Check(Abs(A - RefAux(R, 0.01)) < 1e-6, 'matches independent sum (single result)');
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = 0, 40, -40);
  Route(@L[0], T, E, K, 0, False, R);
  A := AuxLoss(R, 0.01);
  Check(Abs(A - 0.04) < 1e-5, Format('collapsed top-1 aux %.6f = alpha*E', [A]));
  Check(Abs(AssignFraction(R, 0) - 1) < 1e-12, 'collapsed f_0 = 1');
  Check(AssignFraction(R, 1) = 0, 'unused expert f = 0');

  Section('pre-drop vs post-drop');
  T := 10; E := 4; K := 1;
  SetLength(L, T * E);
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = 0, 8, 0);
  Route(@L[0], T, E, K, 1.25, False, R);
  Kept := 0;
  for I := 0 to T - 1 do if R.Kept[I] then Inc(Kept);
  Check(Kept = 4, Format('capacity keeps %d', [Kept]));
  Check(Abs(AssignFraction(R, 0) - 1) < 1e-12, 'pre-drop f_0 = 1');
  Check(Abs(KeptFraction(R, 0) - 0.4) < 1e-12, Format('post-drop kept fraction %.3f', [KeptFraction(R, 0)]));
  A := AuxLoss(R, 0.01);
  Check(Abs(A - 0.04) < 1e-3,
    Format('aux uses pre-drop f, so collapse+capacity is still ~alpha*E (%.6f)', [A]));
  Check(Abs(A - 0.01 * 4 * KeptFraction(R, 0) * MeanProb(R, 0)) > 1e-3,
    'aux is not the post-drop product');

  Section('stop-gradient on counts');
  T := 5; E := 4; K := 2;
  SetLength(L, T * E);
  for I := 0 to High(L) do L[I] := 0.1 * (I mod 7) - 0.3;
  Route(@L[0], T, E, K, 0, False, R);
  SetLength(dP, T * E);
  for I := 0 to High(dP) do dP[I] := 0;
  AuxLossBackward(R, 0.01, @dP[0]);
  Err := 0;
  for J := 0 to E - 1 do
  begin
    A := 0.01 * E * AssignFraction(R, J) / T;
    for I := 0 to T - 1 do Err := Max(Err, Abs(dP[I * E + J] - A));
    if AssignFraction(R, J) = 0 then
      for I := 0 to T - 1 do
        if dP[I * E + J] <> 0 then Err := 1;
  end;
  Check(Err < 1e-8, Format('dL/dp = alpha*E*f/T with f fixed, max err %.1e', [Err]));

  Section('top-k generalization is not the top-1 formula');
  T := 4; E := 4; K := 2;
  SetLength(L, T * E);
  for I := 0 to T - 1 do
    for J := 0 to E - 1 do L[I * E + J] := IfThen(J = 0, 5, IfThen(J = 1, 4, 0));
  Route(@L[0], T, E, K, 0, True, R);
  Check(Abs(R.Gate[0] + R.Gate[1] - 1) < 1e-5, 'renorm over the K selected probs, before capacity');
  A := AuxLoss(R, 0.02);
  Check(Abs(A - RefAux(R, 0.02)) < 1e-9, Format('top-2 aux %.6f matches slot-fraction generalization', [A]));
  Check(Abs(AssignFraction(R, 0) - 0.5) < 1e-9, 'each token''s first slot is expert 0, f_0 = 1/K');
  // A published top-1 reading of the same tokens would set f from argmax only.
  // That is a different number; this test locks the distinction.
  Check(Abs(AssignFraction(R, 0) - 1) > 0.1, 'top-2 f_0 is not the top-1 argmax fraction');
  Finish('test_switch');
end.
