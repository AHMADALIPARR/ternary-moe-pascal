// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Throughput and weight traffic of the packed inference path.
program tmoe_bench;

{$mode objfpc}{$H+}

uses SysUtils, BaseUnix, Unix, tmoe_config, tmoe_tensor, tmoe_params, tmoe_ternary, tmoe_gemv,
  tmoe_model, tmoe_moe, tmoe_kvcache, tmoe_cuda, tmoe_args;

function Clock: Double;
var Tv: TTimeVal;
begin
  fpgettimeofday(@Tv, nil);
  Result := Tv.tv_sec + Tv.tv_usec * 1e-6;
end;

var
  C: TMoeConfig;
  S: TParamStore;
  M: TModel;
  Cache: TKVCache;
  Tok: TIntArray;
  Logits: TSingleArray;
  Prefill, Decode, Reps, I, R, Next: Integer;
  T0, TP, TD, PrefTps, DecTps: Double;
  Why, Gpu: string;
  Mx: TTernaryMatrix;
  Xq: TShortArray;
  Y, Tmp: TSingleArray;
  Us: Double;
  N, K: Integer;
  Rng: TRng;
begin
  C := ArgConfig;
  Prefill := ArgInt('prefill', 64);
  Decode := ArgInt('decode', 64);
  Reps := ArgInt('reps', 5);
  if Prefill + Decode > C.MaxSeq then Decode := C.MaxSeq - Prefill;
  BuildLayout(S, C);
  InitParams(S, 1);
  BuildModel(M, S);
  InitKVCache(Cache, C);
  SetLength(Tok, Prefill);
  for I := 0 to Prefill - 1 do Tok[I] := (I * 31 + 7) mod C.Vocab;
  SetLength(Logits, C.Vocab);
  WriteLn(ConfigSummary(C));
  WriteLn(Format('parameters            %d', [S.Total]));
  WriteLn(Format('resident weights      %d bytes (%.2f MiB)', [ResidentWeightBytes(M), ResidentWeightBytes(M) / 1048576]));
  WriteLn(Format('weight bytes / token  %d (%.2f MiB)', [WeightBytesPerToken(M), WeightBytesPerToken(M) / 1048576]));
  WriteLn(Format('routed expert         %d bytes packed (fp32 would be %d)',
    [RoutedExpertBytes(M.Layers[0].Moe, 0), Int64(3) * C.ExpertHidden * C.Dim * 4]));
  // warm-up
  ResetKVCache(Cache);
  ModelForward(M, Cache, @Tok[0], Prefill, @Logits[0], True);
  TP := 1e30; TD := 1e30;
  for R := 1 to Reps do
  begin
    ResetKVCache(Cache);
    T0 := Clock;
    ModelForward(M, Cache, @Tok[0], Prefill, @Logits[0], True);
    if Clock - T0 < TP then TP := Clock - T0;
    T0 := Clock;
    Next := 1;
    for I := 1 to Decode do
      ModelForward(M, Cache, @Next, 1, @Logits[0], True);
    if Clock - T0 < TD then TD := Clock - T0;
  end;
  PrefTps := Prefill / TP;
  DecTps := Decode / TD;
  WriteLn(Format('cpu prefill  %d tok   %.1f tok/s  (best of %d)', [Prefill, PrefTps, Reps]));
  WriteLn(Format('cpu decode   %d tok   %.1f tok/s  (best of %d, 1 thread)', [Decode, DecTps, Reps]));
  WriteLn(Format('cpu decode weight read  %.1f MB/s', [DecTps * WeightBytesPerToken(M) / 1e6]));

  // single routed-expert GEMV, CPU
  N := C.ExpertHidden; K := C.Dim;
  RngSeed(Rng, 3);
  QuantizeAbsmean(@S.Data[S.L[0].ExGate], N, K, Mx);
  SetLength(Xq, K); SetLength(Y, N);
  for I := 0 to K - 1 do Xq[I] := RngInt(Rng, 255) - 127;
  R := 2000;
  T0 := Clock;
  for I := 1 to R do TernaryGemvInt8(Mx, @Xq[0], 0.01, @Y[0]);
  Us := (Clock - T0) * 1e6 / R;
  WriteLn(Format('cpu ternary GEMV %dx%d  %.2f us', [N, K, Us]));

  // the GPU bench shape, on the CPU
  N := 4096; K := 4096;
  SetLength(Tmp, N * K);
  for I := 0 to High(Tmp) do Tmp[I] := RngNormal(Rng);
  QuantizeAbsmean(@Tmp[0], N, K, Mx);
  Tmp := nil;
  SetLength(Xq, K); SetLength(Y, N);
  for I := 0 to K - 1 do Xq[I] := RngInt(Rng, 255) - 127;
  R := 20;
  T0 := Clock;
  for I := 1 to R do TernaryGemvInt8(Mx, @Xq[0], 0.01, @Y[0]);
  Us := (Clock - T0) * 1e6 / R;
  WriteLn(Format('cpu ternary GEMV %dx%d  %.2f us', [N, K, Us]));

  if CudaAvailable(Why) then
  begin
    Gpu := CudaDeviceName;
    if Cuda.BenchGemv(4096, 4096, 200, @Us) = 0 then
      WriteLn(Format('gpu %s ternary GEMV 4096x4096  %.2f us', [Gpu, Us]))
    else
      WriteLn('gpu bench failed: ', CudaError);
  end
  else
    WriteLn('gpu: ', Why);
end.
