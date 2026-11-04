// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// MoE feed-forward block:  y = shared(x) + sum_{j<k} g_j * expert_{i_j}(x)
//   shared(x)  = Wd (silu(Wg x) * (Wu x))        float, always active
//   expert(x)  = same SwiGLU with ternary packed weights; the input and the
//                hidden activation are quantized to int8 per token.
unit tmoe_moe;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor, tmoe_ternary, tmoe_router;

type
  TMoeWeights = record
    ShGate, ShUp, ShDown: PSingle;   // float shared expert
    Router: PSingle;                 // E x D
    ExGate, ExUp, ExDown: array of TTernaryMatrix;
  end;

procedure SharedExpertForward(const C: TMoeConfig; const W: TMoeWeights; X, Y: PSingle);
procedure RoutedExpertForward(const C: TMoeConfig; const W: TMoeWeights; E: Integer; X, Y: PSingle);
// X: T x D (already normed). Y: T x D. Routing returned for inspection.
procedure MoeForward(const C: TMoeConfig; const W: TMoeWeights; X: PSingle; T: Integer;
  Y: PSingle; out R: TRouting);
function RoutedExpertBytes(const W: TMoeWeights; E: Integer): Int64;

implementation

uses tmoe_quant, tmoe_gemv;

procedure SharedExpertForward(const C: TMoeConfig; const W: TMoeWeights; X, Y: PSingle);
var
  G, U: TSingleArray;
  I: Integer;
begin
  SetLength(G, C.SharedHidden);
  SetLength(U, C.SharedHidden);
  MatVec(W.ShGate, C.SharedHidden, C.Dim, X, @G[0]);
  MatVec(W.ShUp, C.SharedHidden, C.Dim, X, @U[0]);
  for I := 0 to C.SharedHidden - 1 do G[I] := Silu(G[I]) * U[I];
  MatVec(W.ShDown, C.Dim, C.SharedHidden, @G[0], Y);
end;

// Batched routed expert: X is N x D rows, Y is N x D.
procedure ExpertRows(const C: TMoeConfig; const W: TMoeWeights; E, N: Integer; X, Y: PSingle);
var
  Xq, Hq: array of ShortInt;
  Xs, Hs, G, U: TSingleArray;
  I: Integer;
  H: Integer;
begin
  H := C.ExpertHidden;
  SetLength(G, N * H);
  SetLength(U, N * H);
  if C.ActQuant then
  begin
    SetLength(Xq, N * C.Dim);
    SetLength(Xs, N);
    QuantizeInt8Rows(X, N, C.Dim, @Xq[0], @Xs[0]);
    TernaryGemmInt8(W.ExGate[E], N, @Xq[0], @Xs[0], @G[0]);
    TernaryGemmInt8(W.ExUp[E], N, @Xq[0], @Xs[0], @U[0]);
  end
  else
    for I := 0 to N - 1 do
    begin
      TernaryGemvF32(W.ExGate[E], @X[PtrInt(I) * C.Dim], @G[I * H]);
      TernaryGemvF32(W.ExUp[E], @X[PtrInt(I) * C.Dim], @U[I * H]);
    end;
  for I := 0 to N * H - 1 do G[I] := Silu(G[I]) * U[I];
  if C.ActQuant then
  begin
    SetLength(Hq, N * H);
    SetLength(Hs, N);
    QuantizeInt8Rows(@G[0], N, H, @Hq[0], @Hs[0]);
    TernaryGemmInt8(W.ExDown[E], N, @Hq[0], @Hs[0], Y);
  end
  else
    for I := 0 to N - 1 do
      TernaryGemvF32(W.ExDown[E], @G[I * H], @Y[PtrInt(I) * C.Dim]);
end;

procedure RoutedExpertForward(const C: TMoeConfig; const W: TMoeWeights; E: Integer; X, Y: PSingle);
begin
  ExpertRows(C, W, E, 1, X, Y);
end;

procedure MoeForward(const C: TMoeConfig; const W: TMoeWeights; X: PSingle; T: Integer;
  Y: PSingle; out R: TRouting);
var
  Logits, GX, GY: TSingleArray;
  Slots: TIntArray;
  I, J, E, N, S, D: Integer;
  G: Single;
begin
  D := C.Dim;
  // shared expert
  for I := 0 to T - 1 do
    SharedExpertForward(C, W, @X[PtrInt(I) * D], @Y[PtrInt(I) * D]);
  // router
  SetLength(Logits, T * C.Experts);
  for I := 0 to T - 1 do
    MatVec(W.Router, C.Experts, D, @X[PtrInt(I) * D], @Logits[I * C.Experts]);
  Route(@Logits[0], T, C.Experts, C.TopK, C.CapacityFactor, C.RenormTopK, R);
  // routed experts: gather the kept slots of each expert, run, scatter
  SetLength(Slots, T * C.TopK);
  for E := 0 to C.Experts - 1 do
  begin
    N := 0;
    for S := 0 to T * C.TopK - 1 do
      if R.Kept[S] and (R.Idx[S] = E) then
      begin
        Slots[N] := S;
        Inc(N);
      end;
    if N = 0 then Continue;
    SetLength(GX, N * D);
    SetLength(GY, N * D);
    for J := 0 to N - 1 do
      Move(X[PtrInt(Slots[J] div C.TopK) * D], GX[J * D], D * SizeOf(Single));
    ExpertRows(C, W, E, N, @GX[0], @GY[0]);
    for J := 0 to N - 1 do
    begin
      I := Slots[J] div C.TopK;
      G := R.Gate[Slots[J]];
      for S := 0 to D - 1 do
        Y[PtrInt(I) * D + S] := Y[PtrInt(I) * D + S] + G * GY[J * D + S];
    end;
  end;
end;

function RoutedExpertBytes(const W: TMoeWeights; E: Integer): Int64;
begin
  Result := TernaryBytes(W.ExGate[E]) + TernaryBytes(W.ExUp[E]) + TernaryBytes(W.ExDown[E]);
end;

end.
