// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Tiny trainer. Full forward/backward of the model in float, AdamW,
// cross-entropy plus the router load-balance loss.
//
// Routed experts are trained with a straight-through estimator: the forward
// pass uses the absmean-ternary weights (and int8 fake-quantized
// activations), the backward pass treats both quantizers as identity, and
// the gradient lands on the float latent weights. The packed model built
// from those latents computes the same function.
unit tmoe_train;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor, tmoe_params, tmoe_router;

type
  TTrainOpts = record
    LR, MinLR, Beta1, Beta2, Eps, WeightDecay, GradClip: Single;
    WarmupSteps, TotalSteps: Integer;
  end;

  TLayerAct = record
    X0, A, Q, K, V, P, O, X1, B: TSingleArray;
    R1, R2: TSingleArray;
    SG, SU, SH: TSingleArray;
    BQ, EG, EU, EH, EHQ, EY: TSingleArray;
    Rt: TRouting;
  end;

  TTrainer = record
    Cfg: TMoeConfig;
    Store: ^TParamStore;
    Opt: TTrainOpts;
    WQ: TSingleArray;          // ternary view of the expert latents
    Acts: array of TLayerAct;
    XF, RF, NF, Logits, Probs: TSingleArray;
    T: Integer;
    LastCE, LastAux, LastGradNorm: Single;
  end;

function DefaultTrainOpts: TTrainOpts;
procedure InitTrainer(out Tr: TTrainer; var Store: TParamStore; const Opt: TTrainOpts);
procedure PrepareWeights(var Tr: TTrainer);
// Forward over one sequence; keeps activations. Returns CE + aux.
function ForwardSeq(var Tr: TTrainer; Tokens, Targets: PInteger; T: Integer): Single;
// Accumulates Scale * d(CE + aux) into Store.Grad.
procedure BackwardSeq(var Tr: TTrainer; Tokens, Targets: PInteger; T: Integer; Scale: Single);
procedure ZeroGrad(var Tr: TTrainer);
procedure AdamWStep(var Tr: TTrainer);
function LRAt(const O: TTrainOpts; Step: Integer): Single;
// One optimizer step over NSeq sequences of SeqLen+1 tokens laid out back to back.
function TrainStep(var Tr: TTrainer; Batch: PInteger; NSeq, SeqLen: Integer): Single;

implementation

uses Math, tmoe_ternary, tmoe_quant;

function DefaultTrainOpts: TTrainOpts;
begin
  Result.LR := 3e-3;
  Result.MinLR := 3e-4;
  Result.Beta1 := 0.9;
  Result.Beta2 := 0.95;
  Result.Eps := 1e-8;
  Result.WeightDecay := 0.01;
  Result.GradClip := 1.0;
  Result.WarmupSteps := 10;
  Result.TotalSteps := 200;
end;

procedure InitTrainer(out Tr: TTrainer; var Store: TParamStore; const Opt: TTrainOpts);
begin
  Tr.Cfg := Store.Cfg;
  Tr.Store := @Store;
  Tr.Opt := Opt;
  Tr.WQ := nil;
  Tr.Acts := nil;
  SetLength(Tr.WQ, Store.Total);
  SetLength(Tr.Acts, Tr.Cfg.Layers);
  if Length(Store.Grad) <> Store.Total then AllocTrainState(Store);
  Tr.T := 0;
end;

procedure PrepareWeights(var Tr: TTrainer);
var
  I: Integer;
  E: Integer;
  Base: PtrInt;
begin
  for I := 0 to High(Tr.Store^.Info) do
    with Tr.Store^.Info[I] do
      if Kind = pkTernary then
        for E := 0 to Count - 1 do
        begin
          Base := Offset + PtrInt(E) * Rows * Cols;
          if Tr.Cfg.TernaryExperts then
            FakeQuantAbsmean(@Tr.Store^.Data[Base], Rows, Cols, @Tr.WQ[Base])
          else
            Move(Tr.Store^.Data[Base], Tr.WQ[Base], PtrInt(Rows) * Cols * SizeOf(Single));
        end;
end;

function PW(var Tr: TTrainer; Ofs: PtrInt): PSingle; inline;
begin
  Result := @Tr.Store^.Data[Ofs];
end;

function PG(var Tr: TTrainer; Ofs: PtrInt): PSingle; inline;
begin
  Result := @Tr.Store^.Grad[Ofs];
end;

function SiluGrad(G: Single): Single; inline;
var S: Single;
begin
  S := Sigmoid(G);
  Result := S * (1 + G * (1 - S));
end;

procedure Zeros(var A: TSingleArray; N: PtrInt);
begin
  if Length(A) <> N then SetLength(A, N);
  FillZero(A);
end;

procedure ActQ(const C: TMoeConfig; X: PSingle; N: Integer; Y: PSingle);
begin
  if C.ActQuant then FakeQuantInt8Row(X, N, Y)
  else Move(X^, Y^, N * SizeOf(Single));
end;

// ---------------------------------------------------------------- forward

procedure AttnFwd(var Tr: TTrainer; L, T: Integer);
var
  C: TMoeConfig;
  QD, KVD, Grp, H, KH, I, U, J: Integer;
  InvSq, Pr: Single;
  Pp, Qh, Oh, Vp: PSingle;
  Ofs: TLayerOfs;
begin
  C := Tr.Cfg;
  Ofs := Tr.Store^.L[L];
  QD := C.Heads * C.HeadDim;
  KVD := C.KVHeads * C.HeadDim;
  Grp := C.Heads div C.KVHeads;
  InvSq := 1 / Sqrt(C.HeadDim);
  with Tr.Acts[L] do
  begin
    Zeros(Q, T * QD); Zeros(K, T * KVD); Zeros(V, T * KVD);
    Zeros(P, PtrInt(C.Heads) * T * T); Zeros(O, T * QD);
    MatMulNT(@A[0], T, C.Dim, PW(Tr, Ofs.Q), QD, @Q[0]);
    MatMulNT(@A[0], T, C.Dim, PW(Tr, Ofs.K), KVD, @K[0]);
    MatMulNT(@A[0], T, C.Dim, PW(Tr, Ofs.V), KVD, @V[0]);
    for I := 0 to T - 1 do
    begin
      for H := 0 to C.Heads - 1 do RopeNeox(@Q[I * QD + H * C.HeadDim], C.HeadDim, I, C.RopeTheta, False);
      for H := 0 to C.KVHeads - 1 do RopeNeox(@K[I * KVD + H * C.HeadDim], C.HeadDim, I, C.RopeTheta, False);
    end;
    for H := 0 to C.Heads - 1 do
    begin
      KH := H div Grp;
      for I := 0 to T - 1 do
      begin
        Pp := @P[(PtrInt(H) * T + I) * T];
        Qh := @Q[I * QD + H * C.HeadDim];
        for U := 0 to I do Pp[U] := Dot(Qh, @K[U * KVD + KH * C.HeadDim], C.HeadDim) * InvSq;
        SoftmaxInPlace(Pp, I + 1);
        Oh := @O[I * QD + H * C.HeadDim];
        for U := 0 to I do
        begin
          Pr := Pp[U];
          Vp := @V[U * KVD + KH * C.HeadDim];
          for J := 0 to C.HeadDim - 1 do Oh[J] := Oh[J] + Pr * Vp[J];
        end;
      end;
    end;
  end;
end;

procedure MoeFwd(var Tr: TTrainer; L, T: Integer; Y: PSingle);
var
  C: TMoeConfig;
  Ofs: TLayerOfs;
  D, Hs, He, TK, I, J, E, N, S: Integer;
  Logits, GX, GG, GU, GH, GHQ, GY: TSingleArray;
  Slots: TIntArray;
  Wg, Wu, Wd: PSingle;
  G: Single;
begin
  C := Tr.Cfg;
  Ofs := Tr.Store^.L[L];
  D := C.Dim; Hs := C.SharedHidden; He := C.ExpertHidden; TK := C.TopK;
  with Tr.Acts[L] do
  begin
    // shared expert
    Zeros(SG, T * Hs); Zeros(SU, T * Hs); Zeros(SH, T * Hs);
    MatMulNT(@B[0], T, D, PW(Tr, Ofs.ShGate), Hs, @SG[0]);
    MatMulNT(@B[0], T, D, PW(Tr, Ofs.ShUp), Hs, @SU[0]);
    for I := 0 to T * Hs - 1 do SH[I] := Silu(SG[I]) * SU[I];
    MatMulNT(@SH[0], T, Hs, PW(Tr, Ofs.ShDown), D, Y);
    // router
    SetLength(Logits, T * C.Experts);
    MatMulNT(@B[0], T, D, PW(Tr, Ofs.Router), C.Experts, @Logits[0]);
    Route(@Logits[0], T, C.Experts, TK, C.CapacityFactor, C.RenormTopK, Rt);
    // routed experts
    Zeros(BQ, T * D);
    for I := 0 to T - 1 do ActQ(C, @B[I * D], D, @BQ[I * D]);
    Zeros(EG, T * TK * He); Zeros(EU, T * TK * He); Zeros(EH, T * TK * He);
    Zeros(EHQ, T * TK * He); Zeros(EY, T * TK * D);
    SetLength(Slots, T * TK);
    for E := 0 to C.Experts - 1 do
    begin
      N := 0;
      for S := 0 to T * TK - 1 do
        if Rt.Kept[S] and (Rt.Idx[S] = E) then begin Slots[N] := S; Inc(N); end;
      if N = 0 then Continue;
      Wg := @Tr.WQ[Ofs.ExGate + PtrInt(E) * He * D];
      Wu := @Tr.WQ[Ofs.ExUp + PtrInt(E) * He * D];
      Wd := @Tr.WQ[Ofs.ExDown + PtrInt(E) * He * D];
      SetLength(GX, N * D); SetLength(GG, N * He); SetLength(GU, N * He);
      SetLength(GH, N * He); SetLength(GHQ, N * He); SetLength(GY, N * D);
      for J := 0 to N - 1 do Move(BQ[(Slots[J] div TK) * D], GX[J * D], D * SizeOf(Single));
      MatMulNT(@GX[0], N, D, Wg, He, @GG[0]);
      MatMulNT(@GX[0], N, D, Wu, He, @GU[0]);
      for I := 0 to N * He - 1 do GH[I] := Silu(GG[I]) * GU[I];
      for J := 0 to N - 1 do ActQ(C, @GH[J * He], He, @GHQ[J * He]);
      MatMulNT(@GHQ[0], N, He, Wd, D, @GY[0]);
      for J := 0 to N - 1 do
      begin
        S := Slots[J];
        Move(GG[J * He], EG[S * He], He * SizeOf(Single));
        Move(GU[J * He], EU[S * He], He * SizeOf(Single));
        Move(GH[J * He], EH[S * He], He * SizeOf(Single));
        Move(GHQ[J * He], EHQ[S * He], He * SizeOf(Single));
        Move(GY[J * D], EY[S * D], D * SizeOf(Single));
      end;
    end;
    // combine
    for S := 0 to T * TK - 1 do
      if Rt.Kept[S] then
      begin
        I := S div TK;
        G := Rt.Gate[S];
        for J := 0 to D - 1 do Y[I * D + J] := Y[I * D + J] + G * EY[S * D + J];
      end;
  end;
end;

function ForwardSeq(var Tr: TTrainer; Tokens, Targets: PInteger; T: Integer): Single;
var
  C: TMoeConfig;
  D, L, I, V: Integer;
  Y: TSingleArray;
  CE, Aux: Double;
  Ofs: TLayerOfs;
begin
  C := Tr.Cfg;
  D := C.Dim;
  V := C.Vocab;
  Tr.T := T;
  SetLength(Y, T * D);
  Aux := 0;
  for L := 0 to C.Layers - 1 do
    with Tr.Acts[L] do
    begin
      Ofs := Tr.Store^.L[L];
      Zeros(X0, T * D);
      if L = 0 then
        for I := 0 to T - 1 do
          Move(Tr.Store^.Data[Tr.Store^.Embed + PtrInt(Tokens[I]) * D], X0[I * D], D * SizeOf(Single))
      else
      begin
        Move(Tr.XF[0], X0[0], T * D * SizeOf(Single)); // previous layer output
      end;
      Zeros(R1, T); Zeros(A, T * D);
      for I := 0 to T - 1 do
      begin
        R1[I] := RMSInv(@X0[I * D], D, C.NormEps);
        RMSNorm(@X0[I * D], PW(Tr, Ofs.AttnNorm), D, C.NormEps, @A[I * D]);
      end;
      AttnFwd(Tr, L, T);
      Zeros(X1, T * D);
      MatMulNT(@O[0], T, C.Heads * C.HeadDim, PW(Tr, Ofs.O), D, @X1[0]);
      for I := 0 to T * D - 1 do X1[I] := X1[I] + X0[I];
      Zeros(R2, T); Zeros(B, T * D);
      for I := 0 to T - 1 do
      begin
        R2[I] := RMSInv(@X1[I * D], D, C.NormEps);
        RMSNorm(@X1[I * D], PW(Tr, Ofs.FfnNorm), D, C.NormEps, @B[I * D]);
      end;
      MoeFwd(Tr, L, T, @Y[0]);
      Aux := Aux + AuxLoss(Rt, C.AuxCoef);
      Zeros(Tr.XF, T * D);
      for I := 0 to T * D - 1 do Tr.XF[I] := X1[I] + Y[I];
    end;
  Zeros(Tr.RF, T); Zeros(Tr.NF, T * D);
  Zeros(Tr.Logits, PtrInt(T) * V); Zeros(Tr.Probs, PtrInt(T) * V);
  for I := 0 to T - 1 do
  begin
    Tr.RF[I] := RMSInv(@Tr.XF[I * D], D, C.NormEps);
    RMSNorm(@Tr.XF[I * D], PW(Tr, Tr.Store^.OutNorm), D, C.NormEps, @Tr.NF[I * D]);
  end;
  MatMulNT(@Tr.NF[0], T, D, PW(Tr, Tr.Store^.Output), V, @Tr.Logits[0]);
  Move(Tr.Logits[0], Tr.Probs[0], PtrInt(T) * V * SizeOf(Single));
  CE := 0;
  for I := 0 to T - 1 do
  begin
    SoftmaxInPlace(@Tr.Probs[PtrInt(I) * V], V);
    if Targets <> nil then
      CE := CE - Ln(Max(Tr.Probs[PtrInt(I) * V + Targets[I]], 1e-30));
  end;
  CE := CE / T;
  Tr.LastCE := CE;
  Tr.LastAux := Aux;
  Result := CE + Aux;
end;

// --------------------------------------------------------------- backward

procedure RmsBack(X, G: PSingle; R: Single; dY: PSingle; N: Integer; dX, dG: PSingle);
var
  I: Integer;
  S, K: Single;
begin
  S := 0;
  for I := 0 to N - 1 do S := S + G[I] * dY[I] * X[I];
  K := R * R * R * S / N;
  for I := 0 to N - 1 do
  begin
    dG[I] := dG[I] + dY[I] * X[I] * R;
    dX[I] := dX[I] + R * G[I] * dY[I] - X[I] * K;
  end;
end;

procedure MoeBack(var Tr: TTrainer; L, T: Integer; dY, dB: PSingle; Scale: Single);
var
  C: TMoeConfig;
  Ofs: TLayerOfs;
  D, Hs, He, TK, I, J, E, N, S: Integer;
  dSH, dSG, dSU, dGate, dProbs, dLog, GX, GG, GU, GHQ, dGY, dGH, dGG, dGU, dGX: TSingleArray;
  Slots: TIntArray;
  Wg, Wu, Wd: PSingle;
  H: Single;
begin
  C := Tr.Cfg;
  Ofs := Tr.Store^.L[L];
  D := C.Dim; Hs := C.SharedHidden; He := C.ExpertHidden; TK := C.TopK;
  with Tr.Acts[L] do
  begin
    // shared expert
    Zeros(dSH, T * Hs);
    MatMulBackX(dY, T, D, PW(Tr, Ofs.ShDown), Hs, @dSH[0]);
    MatMulBackW(dY, T, D, @SH[0], Hs, PG(Tr, Ofs.ShDown));
    Zeros(dSG, T * Hs); Zeros(dSU, T * Hs);
    for I := 0 to T * Hs - 1 do
    begin
      dSG[I] := dSH[I] * SU[I] * SiluGrad(SG[I]);
      dSU[I] := dSH[I] * Silu(SG[I]);
    end;
    MatMulBackX(@dSG[0], T, Hs, PW(Tr, Ofs.ShGate), D, dB);
    MatMulBackX(@dSU[0], T, Hs, PW(Tr, Ofs.ShUp), D, dB);
    MatMulBackW(@dSG[0], T, Hs, @B[0], D, PG(Tr, Ofs.ShGate));
    MatMulBackW(@dSU[0], T, Hs, @B[0], D, PG(Tr, Ofs.ShUp));
    // gates
    Zeros(dGate, T * TK);
    for S := 0 to T * TK - 1 do
      if Rt.Kept[S] then dGate[S] := Dot(@dY[(S div TK) * D], @EY[S * D], D);
    // experts (STE through both quantizers)
    SetLength(Slots, T * TK);
    for E := 0 to C.Experts - 1 do
    begin
      N := 0;
      for S := 0 to T * TK - 1 do
        if Rt.Kept[S] and (Rt.Idx[S] = E) then begin Slots[N] := S; Inc(N); end;
      if N = 0 then Continue;
      Wg := @Tr.WQ[Ofs.ExGate + PtrInt(E) * He * D];
      Wu := @Tr.WQ[Ofs.ExUp + PtrInt(E) * He * D];
      Wd := @Tr.WQ[Ofs.ExDown + PtrInt(E) * He * D];
      SetLength(GX, N * D); SetLength(GG, N * He); SetLength(GU, N * He); SetLength(GHQ, N * He);
      SetLength(dGY, N * D);
      for J := 0 to N - 1 do
      begin
        S := Slots[J];
        Move(BQ[(S div TK) * D], GX[J * D], D * SizeOf(Single));
        Move(EG[S * He], GG[J * He], He * SizeOf(Single));
        Move(EU[S * He], GU[J * He], He * SizeOf(Single));
        Move(EHQ[S * He], GHQ[J * He], He * SizeOf(Single));
        for I := 0 to D - 1 do dGY[J * D + I] := Rt.Gate[S] * dY[(S div TK) * D + I];
      end;
      Zeros(dGH, N * He);
      MatMulBackX(@dGY[0], N, D, Wd, He, @dGH[0]);
      MatMulBackW(@dGY[0], N, D, @GHQ[0], He, PG(Tr, Ofs.ExDown + PtrInt(E) * He * D));
      Zeros(dGG, N * He); Zeros(dGU, N * He);
      for I := 0 to N * He - 1 do
      begin
        H := dGH[I];
        dGG[I] := H * GU[I] * SiluGrad(GG[I]);
        dGU[I] := H * Silu(GG[I]);
      end;
      Zeros(dGX, N * D);
      MatMulBackX(@dGG[0], N, He, Wg, D, @dGX[0]);
      MatMulBackX(@dGU[0], N, He, Wu, D, @dGX[0]);
      MatMulBackW(@dGG[0], N, He, @GX[0], D, PG(Tr, Ofs.ExGate + PtrInt(E) * He * D));
      MatMulBackW(@dGU[0], N, He, @GX[0], D, PG(Tr, Ofs.ExUp + PtrInt(E) * He * D));
      for J := 0 to N - 1 do
      begin
        S := Slots[J];
        for I := 0 to D - 1 do
          dB[(S div TK) * D + I] := dB[(S div TK) * D + I] + dGX[J * D + I];
      end;
    end;
    // router: gates + aux loss -> probs -> logits
    Zeros(dProbs, T * C.Experts);
    for I := 0 to T - 1 do GateBackward(Rt, I, C.RenormTopK, @dGate[I * TK], @dProbs[I * C.Experts]);
    AuxLossBackward(Rt, C.AuxCoef * Scale, @dProbs[0]);
    Zeros(dLog, T * C.Experts);
    for I := 0 to T - 1 do
      SoftmaxBackward(@Rt.Probs[I * C.Experts], @dProbs[I * C.Experts], C.Experts, @dLog[I * C.Experts]);
    MatMulBackX(@dLog[0], T, C.Experts, PW(Tr, Ofs.Router), D, dB);
    MatMulBackW(@dLog[0], T, C.Experts, @B[0], D, PG(Tr, Ofs.Router));
  end;
end;

procedure AttnBack(var Tr: TTrainer; L, T: Integer; dX1, dA: PSingle);
var
  C: TMoeConfig;
  Ofs: TLayerOfs;
  QD, KVD, Grp, H, KH, I, U, J: Integer;
  InvSq, Sm, Ds: Single;
  dOut, dQ, dK, dV, dP: TSingleArray;
  Pp, dOh, Qh, dQh: PSingle;
begin
  C := Tr.Cfg;
  Ofs := Tr.Store^.L[L];
  QD := C.Heads * C.HeadDim;
  KVD := C.KVHeads * C.HeadDim;
  Grp := C.Heads div C.KVHeads;
  InvSq := 1 / Sqrt(C.HeadDim);
  with Tr.Acts[L] do
  begin
    Zeros(dOut, T * QD);
    MatMulBackX(dX1, T, C.Dim, PW(Tr, Ofs.O), QD, @dOut[0]);
    MatMulBackW(dX1, T, C.Dim, @O[0], QD, PG(Tr, Ofs.O));
    Zeros(dQ, T * QD); Zeros(dK, T * KVD); Zeros(dV, T * KVD);
    SetLength(dP, T);
    for H := 0 to C.Heads - 1 do
    begin
      KH := H div Grp;
      for I := 0 to T - 1 do
      begin
        Pp := @P[(PtrInt(H) * T + I) * T];
        dOh := @dOut[I * QD + H * C.HeadDim];
        Qh := @Q[I * QD + H * C.HeadDim];
        dQh := @dQ[I * QD + H * C.HeadDim];
        Sm := 0;
        for U := 0 to I do
        begin
          dP[U] := Dot(dOh, @V[U * KVD + KH * C.HeadDim], C.HeadDim);
          Sm := Sm + Pp[U] * dP[U];
          for J := 0 to C.HeadDim - 1 do
            dV[U * KVD + KH * C.HeadDim + J] := dV[U * KVD + KH * C.HeadDim + J] + Pp[U] * dOh[J];
        end;
        for U := 0 to I do
        begin
          Ds := Pp[U] * (dP[U] - Sm) * InvSq;
          if Ds = 0 then Continue;
          for J := 0 to C.HeadDim - 1 do
          begin
            dQh[J] := dQh[J] + Ds * K[U * KVD + KH * C.HeadDim + J];
            dK[U * KVD + KH * C.HeadDim + J] := dK[U * KVD + KH * C.HeadDim + J] + Ds * Qh[J];
          end;
        end;
      end;
    end;
    // RoPE is a rotation: its backward is the inverse rotation
    for I := 0 to T - 1 do
    begin
      for H := 0 to C.Heads - 1 do RopeNeox(@dQ[I * QD + H * C.HeadDim], C.HeadDim, I, C.RopeTheta, True);
      for H := 0 to C.KVHeads - 1 do RopeNeox(@dK[I * KVD + H * C.HeadDim], C.HeadDim, I, C.RopeTheta, True);
    end;
    MatMulBackX(@dQ[0], T, QD, PW(Tr, Ofs.Q), C.Dim, dA);
    MatMulBackX(@dK[0], T, KVD, PW(Tr, Ofs.K), C.Dim, dA);
    MatMulBackX(@dV[0], T, KVD, PW(Tr, Ofs.V), C.Dim, dA);
    MatMulBackW(@dQ[0], T, QD, @A[0], C.Dim, PG(Tr, Ofs.Q));
    MatMulBackW(@dK[0], T, KVD, @A[0], C.Dim, PG(Tr, Ofs.K));
    MatMulBackW(@dV[0], T, KVD, @A[0], C.Dim, PG(Tr, Ofs.V));
  end;
end;

procedure BackwardSeq(var Tr: TTrainer; Tokens, Targets: PInteger; T: Integer; Scale: Single);
var
  C: TMoeConfig;
  D, V, I, L, J: Integer;
  dLogits, dX, dB, dA: TSingleArray;
  Ofs: TLayerOfs;
  Inv: Single;
begin
  C := Tr.Cfg;
  D := C.Dim;
  V := C.Vocab;
  // cross-entropy (mean over T) -> logits
  SetLength(dLogits, PtrInt(T) * V);
  Inv := Scale / T;
  for I := 0 to T - 1 do
  begin
    for J := 0 to V - 1 do dLogits[PtrInt(I) * V + J] := Tr.Probs[PtrInt(I) * V + J] * Inv;
    dLogits[PtrInt(I) * V + Targets[I]] := dLogits[PtrInt(I) * V + Targets[I]] - Inv;
  end;
  Zeros(dB, T * D);
  MatMulBackX(@dLogits[0], T, V, PW(Tr, Tr.Store^.Output), D, @dB[0]);
  MatMulBackW(@dLogits[0], T, V, @Tr.NF[0], D, PG(Tr, Tr.Store^.Output));
  Zeros(dX, T * D);
  for I := 0 to T - 1 do
    RmsBack(@Tr.XF[I * D], PW(Tr, Tr.Store^.OutNorm), Tr.RF[I], @dB[I * D], D, @dX[I * D],
      PG(Tr, Tr.Store^.OutNorm));
  for L := C.Layers - 1 downto 0 do
    with Tr.Acts[L] do
    begin
      Ofs := Tr.Store^.L[L];
      // X2 = X1 + moe(norm(X1)); dX currently holds dX2
      Zeros(dB, T * D);
      MoeBack(Tr, L, T, @dX[0], @dB[0], Scale);
      for I := 0 to T - 1 do
        RmsBack(@X1[I * D], PW(Tr, Ofs.FfnNorm), R2[I], @dB[I * D], D, @dX[I * D], PG(Tr, Ofs.FfnNorm));
      // dX now holds dX1. X1 = X0 + attn(norm(X0))
      Zeros(dA, T * D);
      AttnBack(Tr, L, T, @dX[0], @dA[0]);
      for I := 0 to T - 1 do
        RmsBack(@X0[I * D], PW(Tr, Ofs.AttnNorm), R1[I], @dA[I * D], D, @dX[I * D], PG(Tr, Ofs.AttnNorm));
    end;
  for I := 0 to T - 1 do
    for J := 0 to D - 1 do
      Tr.Store^.Grad[Tr.Store^.Embed + PtrInt(Tokens[I]) * D + J] :=
        Tr.Store^.Grad[Tr.Store^.Embed + PtrInt(Tokens[I]) * D + J] + dX[I * D + J];
end;

procedure ZeroGrad(var Tr: TTrainer);
begin
  FillZero(Tr.Store^.Grad);
end;

function LRAt(const O: TTrainOpts; Step: Integer): Single;
var Prog: Single;
begin
  if Step < O.WarmupSteps then Exit(O.LR * (Step + 1) / O.WarmupSteps);
  if O.TotalSteps <= O.WarmupSteps then Exit(O.LR);
  Prog := Min(1.0, (Step - O.WarmupSteps) / (O.TotalSteps - O.WarmupSteps));
  Result := O.MinLR + 0.5 * (O.LR - O.MinLR) * (1 + Cos(Pi * Prog));
end;

procedure AdamWStep(var Tr: TTrainer);
var
  S: ^TParamStore;
  I: Integer;
  J, Last: PtrInt;
  Norm, Clip, LR, B1c, B2c, G, Mh, Vh, WD: Double;
begin
  S := Tr.Store;
  Norm := 0;
  for J := 0 to S^.Total - 1 do Norm := Norm + Sqr(Double(S^.Grad[J]));
  Norm := Sqrt(Norm);
  Tr.LastGradNorm := Norm;
  Clip := 1;
  if (Tr.Opt.GradClip > 0) and (Norm > Tr.Opt.GradClip) then Clip := Tr.Opt.GradClip / Norm;
  LR := LRAt(Tr.Opt, S^.Step);
  Inc(S^.Step);
  B1c := 1 - Power(Tr.Opt.Beta1, S^.Step);
  B2c := 1 - Power(Tr.Opt.Beta2, S^.Step);
  for I := 0 to High(S^.Info) do
    with S^.Info[I] do
    begin
      if Kind in [pkDense, pkTernary] then WD := Tr.Opt.WeightDecay else WD := 0;
      Last := Offset + PtrInt(Count) * Rows * Cols - 1;
      for J := Offset to Last do
      begin
        G := S^.Grad[J] * Clip;
        S^.M[J] := Tr.Opt.Beta1 * S^.M[J] + (1 - Tr.Opt.Beta1) * G;
        S^.V[J] := Tr.Opt.Beta2 * S^.V[J] + (1 - Tr.Opt.Beta2) * G * G;
        Mh := S^.M[J] / B1c;
        Vh := S^.V[J] / B2c;
        S^.Data[J] := S^.Data[J] - LR * (Mh / (Sqrt(Vh) + Tr.Opt.Eps) + WD * S^.Data[J]);
      end;
    end;
end;

function TrainStep(var Tr: TTrainer; Batch: PInteger; NSeq, SeqLen: Integer): Single;
var
  I: Integer;
  Seq: PInteger;
  Total, CE, Aux: Double;
begin
  ZeroGrad(Tr);
  PrepareWeights(Tr);
  Total := 0; CE := 0; Aux := 0;
  for I := 0 to NSeq - 1 do
  begin
    Seq := @Batch[PtrInt(I) * (SeqLen + 1)];
    Total := Total + ForwardSeq(Tr, Seq, @Seq[1], SeqLen);
    CE := CE + Tr.LastCE;
    Aux := Aux + Tr.LastAux;
    BackwardSeq(Tr, Seq, @Seq[1], SeqLen, 1.0 / NSeq);
  end;
  AdamWStep(Tr);
  Tr.LastCE := CE / NSeq;
  Tr.LastAux := Aux / NSeq;
  Result := Total / NSeq;
end;

end.
