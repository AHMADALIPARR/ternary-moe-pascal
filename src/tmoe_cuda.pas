// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Runtime binding to libtmoe_cuda.so (cuda/tmoe_cuda.h). The library is
// loaded on demand so every program also runs on machines without CUDA.
unit tmoe_cuda;

{$mode objfpc}{$H+}

interface

type
  TCudaInt = function: LongInt; cdecl;
  TCudaName = function(Dev: LongInt; Buf: PChar; Len: LongInt): LongInt; cdecl;
  TCudaErr = function: PChar; cdecl;
  TCudaActQuant = function(X: PSingle; Rows, Cols: LongInt; Q: PShortInt; Scales: PSingle): LongInt; cdecl;
  TCudaGemv = function(W: PByte; WS: PSingle; N, K: LongInt; Xq: PShortInt; Xs: Single; Y: PSingle): LongInt; cdecl;
  TCudaGemm = function(W: PByte; WS: PSingle; N, K: LongInt; Xq: PShortInt; Xs: PSingle; M: LongInt; Y: PSingle): LongInt; cdecl;
  TCudaMoe = function(X: PSingle; T, D, H, E, K: LongInt; SlotExpert: PLongInt; SlotGate: PSingle;
    WG: PByte; SG: PSingle; WU: PByte; SU: PSingle; WD: PByte; SD: PSingle; Y: PSingle): LongInt; cdecl;
  TCudaBench = function(N, K, Iters: LongInt; UsPerCall: PDouble): LongInt; cdecl;
  TCudaBenchGemm = function(M, N, K, Warmup, Iters, Kind: LongInt; Us: PDouble): LongInt; cdecl;
  TCudaInfo = function(Dev: LongInt; Buf: PChar; Len: LongInt): LongInt; cdecl;
  // Opaque device-resident packed experts. H is written by Create.
  TCudaResCreate = function(var H: Pointer; Experts, Dim, Hidden: LongInt;
    WG: PByte; SG: PSingle; WU: PByte; SU: PSingle; WD: PByte; SD: PSingle): LongInt; cdecl;
  TCudaResFwd = function(H: Pointer; X: PSingle; Tokens, TopK: LongInt;
    SlotExpert: PLongInt; SlotGate, Y: PSingle): LongInt; cdecl;
  TCudaResFree = procedure(H: Pointer); cdecl;

  TCudaApi = record
    Loaded: Boolean;
    Path: string;
    DeviceCount: TCudaInt;
    DeviceName: TCudaName;
    DeviceInfo: TCudaInfo;
    LastError: TCudaErr;
    ActQuant: TCudaActQuant;
    Gemv: TCudaGemv;
    Gemm: TCudaGemm;
    GemmNaive: TCudaGemm;
    MoeRouted: TCudaMoe;
    BenchGemv: TCudaBench;
    BenchGemm: TCudaBenchGemm;
    ResidentCreate: TCudaResCreate;
    ResidentForward: TCudaResFwd;
    ResidentDestroy: TCudaResFree;
  end;

var
  Cuda: TCudaApi;

// True when the library loaded and reports at least one device. Why holds the
// reason otherwise.
function CudaAvailable(out Why: string): Boolean;
function CudaDeviceName: string;
function CudaError: string;

implementation

uses SysUtils, dynlibs;

var
  Handle: TLibHandle = NilHandle;
  Tried: Boolean = False;

function LibCandidates: TStringArray;
var Env: string;
begin
  Result := nil;
  Env := GetEnvironmentVariable('TMOE_CUDA_LIB');
  if Env <> '' then
  begin
    SetLength(Result, 1);
    Result[0] := Env;
    Exit;
  end;
  SetLength(Result, 4);
  Result[0] := ExtractFilePath(ParamStr(0)) + 'libtmoe_cuda.so';
  Result[1] := ExtractFilePath(ParamStr(0)) + '../libtmoe_cuda.so';
  Result[2] := ExtractFilePath(ParamStr(0)) + '../build/libtmoe_cuda.so';
  Result[3] := 'libtmoe_cuda.so';
end;

function TryLoad(out Why: string): Boolean;
var
  C: TStringArray;
  I: Integer;
  function Sym(const N: string): Pointer;
  begin
    Result := GetProcedureAddress(Handle, N);
    if Result = nil then raise Exception.Create('missing symbol ' + N);
  end;
begin
  Why := '';
  if Tried then Exit(Cuda.Loaded);
  Tried := True;
  C := LibCandidates;
  for I := 0 to High(C) do
  begin
    if (Pos('/', C[I]) > 0) and not FileExists(C[I]) then Continue;
    Handle := LoadLibrary(C[I]);
    if Handle <> NilHandle then
    begin
      Cuda.Path := C[I];
      Break;
    end;
  end;
  if Handle = NilHandle then
  begin
    Why := 'libtmoe_cuda.so not found (build with make lib, or set TMOE_CUDA_LIB)';
    Exit(False);
  end;
  try
    Pointer(Cuda.DeviceCount) := Sym('tmoe_cuda_device_count');
    Pointer(Cuda.DeviceName) := Sym('tmoe_cuda_device_name');
    Pointer(Cuda.LastError) := Sym('tmoe_cuda_last_error');
    Pointer(Cuda.ActQuant) := Sym('tmoe_h_act_quant_int8');
    Pointer(Cuda.Gemv) := Sym('tmoe_h_ternary_gemv');
    Pointer(Cuda.Gemm) := Sym('tmoe_h_ternary_gemm');
    Pointer(Cuda.MoeRouted) := Sym('tmoe_h_moe_routed_forward');
    Pointer(Cuda.BenchGemv) := Sym('tmoe_h_bench_gemv');
    // Optional: an older library still answers the calls above.
    Pointer(Cuda.GemmNaive) := GetProcedureAddress(Handle, 'tmoe_h_ternary_gemm_naive');
    Pointer(Cuda.BenchGemm) := GetProcedureAddress(Handle, 'tmoe_h_bench_gemm');
    Pointer(Cuda.DeviceInfo) := GetProcedureAddress(Handle, 'tmoe_cuda_device_info');
    Pointer(Cuda.ResidentCreate) := GetProcedureAddress(Handle, 'tmoe_resident_create');
    Pointer(Cuda.ResidentForward) := GetProcedureAddress(Handle, 'tmoe_resident_forward');
    Pointer(Cuda.ResidentDestroy) := GetProcedureAddress(Handle, 'tmoe_resident_destroy');
  except
    on E: Exception do
    begin
      Why := E.Message;
      UnloadLibrary(Handle);
      Handle := NilHandle;
      Exit(False);
    end;
  end;
  Cuda.Loaded := True;
  Result := True;
end;

var
  LoadWhy: string = '';

function CudaAvailable(out Why: string): Boolean;
begin
  if not Tried then TryLoad(LoadWhy);
  Why := LoadWhy;
  if not Cuda.Loaded then Exit(False);
  if Cuda.DeviceCount() <= 0 then
  begin
    Why := 'no CUDA device';
    Exit(False);
  end;
  Result := True;
end;

function CudaDeviceName: string;
var Buf: array[0..255] of Char;
begin
  Result := '';
  if not Cuda.Loaded then Exit;
  FillChar(Buf, SizeOf(Buf), 0);
  if Cuda.DeviceName(0, @Buf[0], SizeOf(Buf)) = 0 then Result := StrPas(@Buf[0]);
end;

function CudaError: string;
begin
  if Cuda.Loaded then Result := StrPas(Cuda.LastError()) else Result := 'not loaded';
end;

end.
