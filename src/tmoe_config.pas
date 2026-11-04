// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Model configuration: sizes, MoE routing knobs and derived dimensions.
unit tmoe_config;

{$mode objfpc}{$H+}

interface

const
  TMOE_BLOCK = 128;   // ternary scale block (columns per scale)

type
  TMoeConfig = record
    Vocab: Integer;        // byte-level by default (256)
    Dim: Integer;          // model width
    Layers: Integer;
    Heads: Integer;        // query heads
    KVHeads: Integer;      // key/value heads (GQA)
    HeadDim: Integer;
    SharedHidden: Integer; // shared expert (dense SwiGLU) hidden size
    ExpertHidden: Integer; // routed ternary expert hidden size
    Experts: Integer;      // routed experts per layer
    TopK: Integer;
    CapacityFactor: Single; // <= 0 means no capacity limit
    RenormTopK: Boolean;    // renormalise the selected gate probabilities
    AuxCoef: Single;        // load-balance loss weight
    MaxSeq: Integer;
    RopeTheta: Single;
    NormEps: Single;
    TernaryExperts: Boolean; // False = float experts (used by gradient checks)
    ActQuant: Boolean;       // int8 per-token activations into routed experts
  end;

function TinyConfig: TMoeConfig;
function SmallConfig: TMoeConfig;
function ValidateConfig(const C: TMoeConfig; out Msg: string): Boolean;
function ConfigSummary(const C: TMoeConfig): string;

implementation

uses SysUtils;

function TinyConfig: TMoeConfig;
begin
  Result.Vocab := 256;
  Result.Dim := 128;
  Result.Layers := 2;
  Result.Heads := 4;
  Result.KVHeads := 2;
  Result.HeadDim := 32;
  Result.SharedHidden := 256;
  Result.ExpertHidden := 256;
  Result.Experts := 4;
  Result.TopK := 1;
  Result.CapacityFactor := 0;
  Result.RenormTopK := False;
  Result.AuxCoef := 0.01;
  Result.MaxSeq := 128;
  Result.RopeTheta := 10000;
  Result.NormEps := 1e-5;
  Result.TernaryExperts := True;
  Result.ActQuant := True;
end;

function SmallConfig: TMoeConfig;
begin
  Result := TinyConfig;
  Result.Dim := 512;
  Result.Layers := 8;
  Result.Heads := 8;
  Result.KVHeads := 2;
  Result.HeadDim := 64;
  Result.SharedHidden := 1024;
  Result.ExpertHidden := 1024;
  Result.Experts := 8;
  Result.MaxSeq := 512;
end;

function ValidateConfig(const C: TMoeConfig; out Msg: string): Boolean;
begin
  Msg := '';
  if (C.Dim <= 0) or (C.Layers <= 0) or (C.Vocab <= 0) then Msg := 'sizes must be positive'
  else if C.Heads * C.HeadDim <> C.Dim then Msg := 'Heads*HeadDim must equal Dim'
  else if (C.KVHeads <= 0) or (C.Heads mod C.KVHeads <> 0) then Msg := 'Heads must be a multiple of KVHeads'
  else if Odd(C.HeadDim) then Msg := 'HeadDim must be even (RoPE)'
  else if (C.Dim mod TMOE_BLOCK <> 0) or (C.ExpertHidden mod TMOE_BLOCK <> 0) then
    Msg := 'Dim and ExpertHidden must be multiples of 128'
  else if (C.Experts <= 0) or (C.TopK <= 0) or (C.TopK > C.Experts) then Msg := 'need 1 <= TopK <= Experts'
  else if C.MaxSeq <= 0 then Msg := 'MaxSeq must be positive';
  Result := Msg = '';
end;

function ConfigSummary(const C: TMoeConfig): string;
begin
  Result := Format('dim=%d layers=%d heads=%d/%d hd=%d shared=%d expert=%d E=%d top%d vocab=%d',
    [C.Dim, C.Layers, C.Heads, C.KVHeads, C.HeadDim, C.SharedHidden, C.ExpertHidden,
     C.Experts, C.TopK, C.Vocab]);
end;

end.
