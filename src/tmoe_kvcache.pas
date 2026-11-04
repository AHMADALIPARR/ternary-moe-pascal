// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Key/value cache: per layer, MaxSeq positions of KVHeads*HeadDim floats.
// Keys are stored after RoPE.
unit tmoe_kvcache;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor;

type
  TKVCache = record
    Layers, MaxSeq, KVDim: Integer;
    K, V: TSingleArray;
    Len: Integer;   // positions filled
  end;

procedure InitKVCache(out C: TKVCache; const Cfg: TMoeConfig);
procedure ResetKVCache(var C: TKVCache);
function KPtr(var C: TKVCache; Layer, Pos: Integer): PSingle; inline;
function VPtr(var C: TKVCache; Layer, Pos: Integer): PSingle; inline;
function KVCacheBytes(const C: TKVCache): Int64;

implementation

procedure InitKVCache(out C: TKVCache; const Cfg: TMoeConfig);
begin
  C.Layers := Cfg.Layers;
  C.MaxSeq := Cfg.MaxSeq;
  C.KVDim := Cfg.KVHeads * Cfg.HeadDim;
  C.K := nil;
  C.V := nil;
  SetLength(C.K, PtrInt(C.Layers) * C.MaxSeq * C.KVDim);
  SetLength(C.V, PtrInt(C.Layers) * C.MaxSeq * C.KVDim);
  C.Len := 0;
end;

procedure ResetKVCache(var C: TKVCache);
begin
  C.Len := 0;
end;

function KPtr(var C: TKVCache; Layer, Pos: Integer): PSingle; inline;
begin
  Result := @C.K[(PtrInt(Layer) * C.MaxSeq + Pos) * C.KVDim];
end;

function VPtr(var C: TKVCache; Layer, Pos: Integer): PSingle; inline;
begin
  Result := @C.V[(PtrInt(Layer) * C.MaxSeq + Pos) * C.KVDim];
end;

function KVCacheBytes(const C: TKVCache): Int64;
begin
  Result := (Int64(Length(C.K)) + Length(C.V)) * SizeOf(Single);
end;

end.
