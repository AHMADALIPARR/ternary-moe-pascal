// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Inference model: token embedding, L x [RMSNorm -> GQA attention ->
// residual -> RMSNorm -> MoE (shared + routed ternary) -> residual],
// final RMSNorm and LM head. Routed experts are packed once at build time
// and stay resident in 2-bit form.
unit tmoe_model;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor, tmoe_ternary, tmoe_params, tmoe_kvcache,
  tmoe_attention, tmoe_moe, tmoe_router;

type
  TLayerWeights = record
    AttnNorm, FfnNorm: PSingle;
    Attn: TAttnWeights;
    Moe: TMoeWeights;
  end;

  TModel = record
    Cfg: TMoeConfig;
    Store: ^TParamStore;
    Embed, OutNorm, Output: PSingle;
    Layers: array of TLayerWeights;
  end;

// Packs the routed experts of Store (absmean ternary). Store must outlive M.
procedure BuildModel(out M: TModel; var Store: TParamStore);
// Runs T tokens at positions Cache.Len ..; appends to the cache.
// Logits receives T x Vocab (or only the last row when LastOnly).
procedure ModelForward(var M: TModel; var Cache: TKVCache; Tokens: PInteger; T: Integer;
  Logits: PSingle; LastOnly: Boolean);
// Bytes of weights touched to decode one token (top-k experts per layer).
function WeightBytesPerToken(const M: TModel): Int64;
function ResidentWeightBytes(const M: TModel): Int64;

implementation

procedure PackStack(var Store: TParamStore; Ofs: PtrInt; E, Rows, Cols: Integer;
  var Dest: array of TTernaryMatrix);
var I: Integer;
begin
  for I := 0 to E - 1 do
    QuantizeAbsmean(@Store.Data[Ofs + PtrInt(I) * Rows * Cols], Rows, Cols, Dest[I]);
end;

procedure BuildModel(out M: TModel; var Store: TParamStore);
var
  C: TMoeConfig;
  L: Integer;
  Lo: TLayerOfs;
begin
  C := Store.Cfg;
  M.Cfg := C;
  M.Store := @Store;
  M.Embed := @Store.Data[Store.Embed];
  M.OutNorm := @Store.Data[Store.OutNorm];
  M.Output := @Store.Data[Store.Output];
  M.Layers := nil;
  SetLength(M.Layers, C.Layers);
  for L := 0 to C.Layers - 1 do
  begin
    Lo := Store.L[L];
    M.Layers[L].AttnNorm := @Store.Data[Lo.AttnNorm];
    M.Layers[L].FfnNorm := @Store.Data[Lo.FfnNorm];
    M.Layers[L].Attn.Q := @Store.Data[Lo.Q];
    M.Layers[L].Attn.K := @Store.Data[Lo.K];
    M.Layers[L].Attn.V := @Store.Data[Lo.V];
    M.Layers[L].Attn.O := @Store.Data[Lo.O];
    with M.Layers[L].Moe do
    begin
      ShGate := @Store.Data[Lo.ShGate];
      ShUp := @Store.Data[Lo.ShUp];
      ShDown := @Store.Data[Lo.ShDown];
      Router := @Store.Data[Lo.Router];
      Dev := nil;
      SetLength(ExGate, C.Experts);
      SetLength(ExUp, C.Experts);
      SetLength(ExDown, C.Experts);
      PackStack(Store, Lo.ExGate, C.Experts, C.ExpertHidden, C.Dim, ExGate);
      PackStack(Store, Lo.ExUp, C.Experts, C.ExpertHidden, C.Dim, ExUp);
      PackStack(Store, Lo.ExDown, C.Experts, C.Dim, C.ExpertHidden, ExDown);
    end;
  end;
end;

procedure ModelForward(var M: TModel; var Cache: TKVCache; Tokens: PInteger; T: Integer;
  Logits: PSingle; LastOnly: Boolean);
var
  C: TMoeConfig;
  X, A, Y: TSingleArray;
  I, L, D, StartPos: Integer;
  R: TRouting;
begin
  C := M.Cfg;
  D := C.Dim;
  StartPos := Cache.Len;
  SetLength(X, T * D);
  SetLength(A, T * D);
  SetLength(Y, T * D);
  for I := 0 to T - 1 do
    Move(M.Embed[PtrInt(Tokens[I]) * D], X[I * D], D * SizeOf(Single));
  for L := 0 to C.Layers - 1 do
  begin
    for I := 0 to T - 1 do
      RMSNorm(@X[I * D], M.Layers[L].AttnNorm, D, C.NormEps, @A[I * D]);
    AttentionForward(C, M.Layers[L].Attn, L, Cache, @A[0], T, StartPos, @Y[0]);
    for I := 0 to T * D - 1 do X[I] := X[I] + Y[I];
    for I := 0 to T - 1 do
      RMSNorm(@X[I * D], M.Layers[L].FfnNorm, D, C.NormEps, @A[I * D]);
    MoeForward(C, M.Layers[L].Moe, @A[0], T, @Y[0], R);
    for I := 0 to T * D - 1 do X[I] := X[I] + Y[I];
  end;
  Cache.Len := StartPos + T;
  if LastOnly then
  begin
    RMSNorm(@X[(T - 1) * D], M.OutNorm, D, C.NormEps, @A[0]);
    MatVec(M.Output, C.Vocab, D, @A[0], Logits);
  end
  else
    for I := 0 to T - 1 do
    begin
      RMSNorm(@X[I * D], M.OutNorm, D, C.NormEps, @A[0]);
      MatVec(M.Output, C.Vocab, D, @A[0], @Logits[PtrInt(I) * C.Vocab]);
    end;
end;

function WeightBytesPerToken(const M: TModel): Int64;
var
  C: TMoeConfig;
  QD, KVD: Int64;
  PerLayer: Int64;
begin
  C := M.Cfg;
  QD := C.Heads * C.HeadDim;
  KVD := C.KVHeads * C.HeadDim;
  PerLayer := (QD * C.Dim * 2 + KVD * C.Dim * 2) * 4      // q,k,v,o fp32
            + Int64(2) * C.Dim * 4                        // norms
            + Int64(3) * C.SharedHidden * C.Dim * 4       // shared expert fp32
            + Int64(C.Experts) * C.Dim * 4                // router
            + Int64(C.TopK) * RoutedExpertBytes(M.Layers[0].Moe, 0);
  Result := PerLayer * C.Layers + Int64(C.Dim) * 4        // embedding row
          + Int64(C.Dim) * 4 + Int64(C.Vocab) * C.Dim * 4; // final norm + lm head
end;

function ResidentWeightBytes(const M: TModel): Int64;
var
  C: TMoeConfig;
  L, E: Integer;
begin
  C := M.Cfg;
  Result := 0;
  for L := 0 to C.Layers - 1 do
    for E := 0 to C.Experts - 1 do
      Inc(Result, RoutedExpertBytes(M.Layers[L].Moe, E));
  // everything that is not a routed expert stays fp32
  Inc(Result, (M.Store^.Total - Int64(C.Layers) * C.Experts * 3 * C.ExpertHidden * C.Dim) * 4);
end;

end.
