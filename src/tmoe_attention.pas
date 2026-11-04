// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Causal self-attention with grouped-query heads and NEOX RoPE, reading and
// appending to the KV cache. Processes T new tokens at positions
// StartPos .. StartPos+T-1.
unit tmoe_attention;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor, tmoe_kvcache;

type
  TAttnWeights = record
    Q, K, V, O: PSingle;   // [QD x D], [KVD x D], [KVD x D], [D x QD]
  end;

// A: T x Dim normed inputs. Out: T x Dim (attention output after Wo).
procedure AttentionForward(const C: TMoeConfig; const W: TAttnWeights; Layer: Integer;
  var Cache: TKVCache; A: PSingle; T, StartPos: Integer; Out_: PSingle);

implementation

procedure AttentionForward(const C: TMoeConfig; const W: TAttnWeights; Layer: Integer;
  var Cache: TKVCache; A: PSingle; T, StartPos: Integer; Out_: PSingle);
var
  QD, KVD, Group, I, H, KH, U, P: Integer;
  Q, O, Sc: TSingleArray;
  Kp, Vp, Qh, Oh: PSingle;
  InvSq, Pr: Single;
  J: Integer;
begin
  QD := C.Heads * C.HeadDim;
  KVD := C.KVHeads * C.HeadDim;
  Group := C.Heads div C.KVHeads;
  InvSq := 1.0 / Sqrt(C.HeadDim);
  SetLength(Q, T * QD);
  SetLength(O, T * QD);
  SetLength(Sc, StartPos + T);
  for I := 0 to T - 1 do
  begin
    P := StartPos + I;
    MatVec(W.Q, QD, C.Dim, @A[PtrInt(I) * C.Dim], @Q[I * QD]);
    MatVec(W.K, KVD, C.Dim, @A[PtrInt(I) * C.Dim], KPtr(Cache, Layer, P));
    MatVec(W.V, KVD, C.Dim, @A[PtrInt(I) * C.Dim], VPtr(Cache, Layer, P));
    for H := 0 to C.Heads - 1 do
      RopeNeox(@Q[I * QD + H * C.HeadDim], C.HeadDim, P, C.RopeTheta, False);
    for H := 0 to C.KVHeads - 1 do
      RopeNeox(@KPtr(Cache, Layer, P)[H * C.HeadDim], C.HeadDim, P, C.RopeTheta, False);
  end;
  for I := 0 to T - 1 do
  begin
    P := StartPos + I;
    for H := 0 to C.Heads - 1 do
    begin
      KH := H div Group;
      Qh := @Q[I * QD + H * C.HeadDim];
      for U := 0 to P do
      begin
        Kp := @KPtr(Cache, Layer, U)[KH * C.HeadDim];
        Sc[U] := Dot(Qh, Kp, C.HeadDim) * InvSq;
      end;
      SoftmaxInPlace(@Sc[0], P + 1);
      Oh := @O[I * QD + H * C.HeadDim];
      for J := 0 to C.HeadDim - 1 do Oh[J] := 0;
      for U := 0 to P do
      begin
        Vp := @VPtr(Cache, Layer, U)[KH * C.HeadDim];
        Pr := Sc[U];
        for J := 0 to C.HeadDim - 1 do Oh[J] := Oh[J] + Pr * Vp[J];
      end;
    end;
    MatVec(W.O, C.Dim, QD, @O[I * QD], @Out_[PtrInt(I) * C.Dim]);
  end;
end;

end.
