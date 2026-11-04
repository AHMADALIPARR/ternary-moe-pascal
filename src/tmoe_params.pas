// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Flat parameter store. Every trainable tensor is a slice of one float
// buffer; gradients and AdamW moments use the same offsets. Routed expert
// weights are kept here as float latents and quantized to ternary on use.
unit tmoe_params;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor;

type
  TParamKind = (pkNorm, pkEmbed, pkDense, pkRouter, pkTernary);

  TParamInfo = record
    Name: string;
    Kind: TParamKind;
    Count, Rows, Cols: Integer;  // Count > 1 for stacked experts
    Offset: PtrInt;              // in floats
  end;

  TLayerOfs = record
    AttnNorm, Q, K, V, O, FfnNorm: PtrInt;
    ShGate, ShUp, ShDown, Router: PtrInt;
    ExGate, ExUp, ExDown: PtrInt;   // E stacked matrices each
  end;

  TParamStore = record
    Cfg: TMoeConfig;
    Info: array of TParamInfo;
    L: array of TLayerOfs;
    Embed, OutNorm, Output: PtrInt;
    Total: PtrInt;
    Data, Grad, M, V: TSingleArray;
    Step: Integer;
  end;

procedure BuildLayout(out S: TParamStore; const C: TMoeConfig);
procedure InitParams(var S: TParamStore; Seed: QWord);
procedure AllocTrainState(var S: TParamStore);
function FindParam(const S: TParamStore; const Name: string): Integer;
function ParamPtr(var S: TParamStore; Ofs: PtrInt): PSingle; inline;

implementation

uses SysUtils, Math;

function ParamPtr(var S: TParamStore; Ofs: PtrInt): PSingle; inline;
begin
  Result := @S.Data[Ofs];
end;

procedure BuildLayout(out S: TParamStore; const C: TMoeConfig);
var
  N: Integer;
  Cur: PtrInt;

  function Add(const Name: string; Kind: TParamKind; Count, Rows, Cols: Integer): PtrInt;
  begin
    SetLength(S.Info, N + 1);
    S.Info[N].Name := Name;
    S.Info[N].Kind := Kind;
    S.Info[N].Count := Count;
    S.Info[N].Rows := Rows;
    S.Info[N].Cols := Cols;
    S.Info[N].Offset := Cur;
    Result := Cur;
    Inc(Cur, PtrInt(Count) * Rows * Cols);
    Inc(N);
  end;

var
  I, QD, KVD: Integer;
  P: string;
begin
  S.Cfg := C;
  S.Info := nil;
  S.L := nil;
  S.Data := nil; S.Grad := nil; S.M := nil; S.V := nil;
  S.Step := 0;
  N := 0;
  Cur := 0;
  QD := C.Heads * C.HeadDim;
  KVD := C.KVHeads * C.HeadDim;
  S.Embed := Add('tok_embd', pkEmbed, 1, C.Vocab, C.Dim);
  SetLength(S.L, C.Layers);
  for I := 0 to C.Layers - 1 do
  begin
    P := Format('blk.%d.', [I]);
    S.L[I].AttnNorm := Add(P + 'attn_norm', pkNorm, 1, 1, C.Dim);
    S.L[I].Q := Add(P + 'attn_q', pkDense, 1, QD, C.Dim);
    S.L[I].K := Add(P + 'attn_k', pkDense, 1, KVD, C.Dim);
    S.L[I].V := Add(P + 'attn_v', pkDense, 1, KVD, C.Dim);
    S.L[I].O := Add(P + 'attn_output', pkDense, 1, C.Dim, QD);
    S.L[I].FfnNorm := Add(P + 'ffn_norm', pkNorm, 1, 1, C.Dim);
    S.L[I].ShGate := Add(P + 'ffn_gate_shexp', pkDense, 1, C.SharedHidden, C.Dim);
    S.L[I].ShUp := Add(P + 'ffn_up_shexp', pkDense, 1, C.SharedHidden, C.Dim);
    S.L[I].ShDown := Add(P + 'ffn_down_shexp', pkDense, 1, C.Dim, C.SharedHidden);
    S.L[I].Router := Add(P + 'ffn_gate_inp', pkRouter, 1, C.Experts, C.Dim);
    S.L[I].ExGate := Add(P + 'ffn_gate_exps', pkTernary, C.Experts, C.ExpertHidden, C.Dim);
    S.L[I].ExUp := Add(P + 'ffn_up_exps', pkTernary, C.Experts, C.ExpertHidden, C.Dim);
    S.L[I].ExDown := Add(P + 'ffn_down_exps', pkTernary, C.Experts, C.Dim, C.ExpertHidden);
  end;
  S.OutNorm := Add('output_norm', pkNorm, 1, 1, C.Dim);
  S.Output := Add('output', pkDense, 1, C.Vocab, C.Dim);
  S.Total := Cur;
  SetLength(S.Data, S.Total);
end;

procedure InitParams(var S: TParamStore; Seed: QWord);
var
  R: TRng;
  I: Integer;
  J, N: PtrInt;
  Std: Single;
begin
  RngSeed(R, Seed);
  for I := 0 to High(S.Info) do
    with S.Info[I] do
    begin
      N := PtrInt(Count) * Rows * Cols;
      case Kind of
        pkNorm: Std := -1;
        pkEmbed: Std := 0.02;
      else
        Std := 1.0 / Sqrt(Cols);
      end;
      if (Pos('attn_output', Name) > 0) or (Pos('ffn_down', Name) > 0) then
        Std := Std / Sqrt(2.0 * S.Cfg.Layers);
      for J := 0 to N - 1 do
        if Std < 0 then S.Data[Offset + J] := 1.0
        else S.Data[Offset + J] := RngNormal(R) * Std;
    end;
end;

procedure AllocTrainState(var S: TParamStore);
begin
  SetLength(S.Grad, S.Total);
  SetLength(S.M, S.Total);
  SetLength(S.V, S.Total);
  FillZero(S.Grad);
  FillZero(S.M);
  FillZero(S.V);
end;

function FindParam(const S: TParamStore; const Name: string): Integer;
var I: Integer;
begin
  for I := 0 to High(S.Info) do
    if S.Info[I].Name = Name then Exit(I);
  Result := -1;
end;

end.
