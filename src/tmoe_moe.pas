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
    Dev: Pointer;                // device-resident packed experts, or nil
  end;

procedure SharedExpertForward(const C: TMoeConfig; const W: TMoeWeights; X, Y: PSingle);
procedure RoutedExpertForward(const C: TMoeConfig; const W: TMoeWeights; E: Integer; X, Y: PSingle);
// X: T x D (already normed). Y: T x D. Routing returned for inspection.
// When the device path is enabled and a CUDA device is present, the routed
// SwiGLU experts run on the packed GEMM and the result is added to the shared
// expert output already in Y. Any failure falls back to the CPU experts.
// The device path is off unless MoeSetDevice(True) was called, so tests keep
// the bit-exact CPU forward.
procedure MoeForward(const C: TMoeConfig; var W: TMoeWeights; X: PSingle; T: Integer;
  Y: PSingle; out R: TRouting);
function RoutedExpertBytes(const W: TMoeWeights; E: Integer): Int64;
procedure MoeSetDevice(Enable: Boolean);
function MoeDeviceWanted: Boolean;
procedure MoeReleaseDevice(var W: TMoeWeights);
function MoeDeviceStatus: string;

implementation

uses SysUtils, tmoe_quant, tmoe_gemv, tmoe_cuda;

var
  WantDevice: Boolean = False;
  DeviceWarned: Boolean = False;

procedure MoeSetDevice(Enable: Boolean);
begin
  WantDevice := Enable;
end;

function MoeDeviceWanted: Boolean;
begin
  Result := WantDevice;
end;

procedure MoeReleaseDevice(var W: TMoeWeights);
begin
  if W.Dev = nil then Exit;
  if Assigned(Cuda.ResidentDestroy) then Cuda.ResidentDestroy(W.Dev);
  W.Dev := nil;
end;

function MoeDeviceStatus: string;
var Why: string;
begin
  if not WantDevice then Exit('CPU (device path off; MoeSetDevice to enable)');
  if not CudaAvailable(Why) then Exit('CPU (' + Why + ')');
  if not Assigned(Cuda.ResidentCreate) then
    Exit('CPU (libtmoe_cuda.so has no tmoe_resident_create)');
  Result := 'CUDA ' + CudaDeviceName;
end;

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

procedure CopyStack(const Ms: array of TTernaryMatrix; DestB: PByte; DestS: PSingle);
var E, NB, NS: Integer;
begin
  NB := Length(Ms[0].Bits);
  NS := Length(Ms[0].Scales);
  for E := 0 to High(Ms) do
  begin
    Move(Ms[E].Bits[0], DestB[E * NB], NB);
    Move(Ms[E].Scales[0], DestS[E * NS], NS * SizeOf(Single));
  end;
end;

function UploadDevice(const C: TMoeConfig; var W: TMoeWeights): Boolean;
var
  BG, BU, BD: array of Byte;
  SG, SU, SD: TSingleArray;
  H: Pointer;
  Rc: LongInt;
  NB, NS: Integer;
begin
  Result := False;
  MoeReleaseDevice(W);
  if not WantDevice then Exit;
  if not Assigned(Cuda.ResidentCreate) then Exit;
  if Length(W.ExGate) <> C.Experts then Exit;
  NB := Length(W.ExGate[0].Bits);
  NS := Length(W.ExGate[0].Scales);
  SetLength(BG, NB * C.Experts);
  SetLength(SG, NS * C.Experts);
  CopyStack(W.ExGate, @BG[0], @SG[0]);
  NB := Length(W.ExUp[0].Bits);
  NS := Length(W.ExUp[0].Scales);
  SetLength(BU, NB * C.Experts);
  SetLength(SU, NS * C.Experts);
  CopyStack(W.ExUp, @BU[0], @SU[0]);
  NB := Length(W.ExDown[0].Bits);
  NS := Length(W.ExDown[0].Scales);
  SetLength(BD, NB * C.Experts);
  SetLength(SD, NS * C.Experts);
  CopyStack(W.ExDown, @BD[0], @SD[0]);
  H := nil;
  Rc := Cuda.ResidentCreate(H, C.Experts, C.Dim, C.ExpertHidden,
    @BG[0], @SG[0], @BU[0], @SU[0], @BD[0], @SD[0]);
  if Rc <> 0 then
  begin
    if H <> nil then Cuda.ResidentDestroy(H);
    if not DeviceWarned then
    begin
      WriteLn(StdErr, 'CUDA expert upload failed: ', CudaError, '; CPU experts');
      DeviceWarned := True;
    end;
    Exit;
  end;
  W.Dev := H;
  Result := True;
end;

// Adds the routed experts into Y, which already holds the shared expert.
// False means the caller should run the CPU loop.
function TryDeviceRouted(const C: TMoeConfig; var W: TMoeWeights; X, Y: PSingle;
  TT: Integer; const R: TRouting): Boolean;
var
  SE: TIntArray;
  Gate: TSingleArray;
  S: Integer;
  Why: string;
begin
  Result := False;
  if not WantDevice or not C.ActQuant then Exit;
  if TT <= 0 then Exit(True);
  if not CudaAvailable(Why) then Exit;
  if W.Dev = nil then
    if not UploadDevice(C, W) then Exit;
  SetLength(SE, TT * C.TopK);
  SetLength(Gate, TT * C.TopK);
  for S := 0 to TT * C.TopK - 1 do
  begin
    if R.Kept[S] then SE[S] := R.Idx[S] else SE[S] := -1;
    Gate[S] := R.Gate[S];
  end;
  if Cuda.ResidentForward(W.Dev, X, TT, C.TopK, @SE[0], @Gate[0], Y) <> 0 then
  begin
    if not DeviceWarned then
    begin
      WriteLn(StdErr, 'CUDA routed forward failed: ', CudaError, '; CPU experts');
      DeviceWarned := True;
    end;
    MoeReleaseDevice(W);
    Exit;
  end;
  Result := True;
end;

procedure MoeForward(const C: TMoeConfig; var W: TMoeWeights; X: PSingle; T: Integer;
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
  if TryDeviceRouted(C, W, X, Y, T, R) then Exit;
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
