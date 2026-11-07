// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Prints the static GEMM account and the Jetson Orin parameter table.
// Nothing here was timed on a device. --check only tests the identities.
program tmoe_cost;

{$mode objfpc}{$H+}

uses SysUtils, tmoe_account, tmoe_args;

procedure Die(const Msg: string);
begin
  WriteLn(StdErr, Msg);
  Halt(1);
end;

procedure Show(const Title: string; const A: TGemmAccount);
begin
  WriteLn(Title);
  if not A.Valid then
  begin
    WriteLn('  invalid shape (need M,N > 0 and K multiple of 128)');
    Exit;
  end;
  WriteLn(Format('  tiles %d x %d x %d (BM=%d BN=%d BK=%d, %d threads)',
    [A.TilesM, A.TilesN, A.TilesK, GEMM_BM, GEMM_BN, GEMM_BK, GEMM_THREADS]));
  WriteLn(Format('  issued bytes  weight %d  act %d  scale %d  total %d',
    [A.Weight, A.Act, A.Scale, A.Total]));
  WriteLn(Format('  naive bytes   weight %d  act %d  scale %d  total %d',
    [A.NaiveWeight, A.NaiveAct, A.NaiveScale, A.NaiveTotal]));
  if A.Weight > 0 then
    WriteLn(Format('  issued-weight ratio naive/tiled = %.2f  (BM=%d when M is a multiple)',
      [A.NaiveWeight / A.Weight, GEMM_BM]));
  WriteLn(Format('  dp4a issued %d  (x4 = %d padded MACs; useful MACs %d)',
    [A.Dp4aIssued, A.Dp4aIssued * 4, A.MacUseful]));
  WriteLn(Format('  blocks %d  barriers/block %d (sync events, not cycles)  smem %d B',
    [A.Blocks, A.BarriersPerBlock, A.SmemBytes]));
  if A.Total > 0 then
    WriteLn(Format('  useful MACs/issued-byte = %.2f', [A.MacUseful / A.Total]));
end;

procedure DramNote(const A: TGemmAccount; const Label_: string; BytesPerSec: Double);
begin
  if (not A.Valid) or (BytesPerSec <= 0) then Exit;
  // Lower bound only if every issued byte missed to DRAM and nothing overlapped.
  WriteLn(Format('  DRAM roof IF every issued byte missed, %s: %.3f us  (not a prediction)',
    [Label_, A.Total / BytesPerSec * 1e6]));
end;

function EstGBs(BusBits, MemMHz: Double): Double;
begin
  Result := BusBits / 8.0 * 2.0 * MemMHz / 1000.0;
end;

procedure SkuRow(const Name: string; TPC: Integer; GpuMHz, MemMHz, BusBits: Double);
begin
  WriteLn(Format('%-22s %4d %8.2f %8.0f %6.0f %10.2f',
    [Name, TPC, GpuMHz, MemMHz, BusBits, EstGBs(BusBits, MemMHz)]));
end;

procedure SkuTable;
begin
  WriteLn('Orin parameters (cited, not measured on this machine)');
  WriteLn('SM count below is 2 * nvpmodel "GPU TPC". The AGX brief says up to');
  WriteLn('8 TPCs and up to 16 SMs, i.e. 2 SMs/TPC. A power mode that gates');
  WriteLn('TPCs changes the SM count; do not assume the product-page core count.');
  WriteLn;
  WriteLn('Sources:');
  WriteLn('  NVIDIA Jetson AGX Orin Series Technical Brief (GPU: 128 CUDA cores/SM,');
  WriteLn('    up to 16 SMs, 192 KB L1/SM, 4 MB L2 on AGX, 204.8 GB/s at 256-bit');
  WriteLn('    LPDDR5 6400 Mb/s/pin).');
  WriteLn('  NVIDIA Jetson Linux Developer Guide r36.4, nvpmodel clock tables.');
  WriteLn('  CUDA Programming Guide, compute-capability tables: sm_87 has 64K');
  WriteLn('    32-bit registers/SM, 32 shared-memory banks, 164 KB smem/SM,');
  WriteLn('    163 KB smem/block. The resident-warp cell for 8.7 is blank in the');
  WriteLn('    rendered table; read maxThreadsPerMultiProcessor on the device.');
  WriteLn('  Product-page peaks (full DRAM rate, not a power-mode guarantee):');
  WriteLn('    AGX 256-bit 204.8 GB/s; NX 128-bit 102.4 GB/s.');
  WriteLn;
  WriteLn('Estimated EMC ceiling = bus_bytes * 2 * mem_MHz * 1e6.');
  WriteLn('That matches 204.8 GB/s at 256-bit and 3200 MHz. It is an estimate');
  WriteLn('of the mode''s max EMC, not a measured bandwidth. DVFS can sit lower.');
  WriteLn('MAXN is unconstrained and may throttle; the guide says it is not the');
  WriteLn('maximum-performance mode.');
  WriteLn;
  WriteLn(Format('%-22s %4s %8s %8s %6s %10s',
    ['mode', 'TPC', 'gpu MHz', 'mem MHz', 'bus', 'est GB/s']));
  SkuRow('AGX64 MAXN', 8, 1301.0, 3200, 256);
  SkuRow('AGX64 15W', 3, 408.0, 2133, 256);
  SkuRow('AGX64 30W default', 4, 612.0, 3200, 256);
  SkuRow('AGX64 50W', 8, 816.0, 3200, 256);
  SkuRow('AGX32 MAXN', 7, 930.75, 3200, 256);
  SkuRow('NX16 MAXN', 4, 918.0, 3200, 128);
  SkuRow('NX16 15W default', 2, 612.0, 3200, 128);
  SkuRow('Nano8 15W', 4, 624.75, 2133, 128);
  SkuRow('Nano4 10W', 2, 624.75, 2133, 64);
  WriteLn;
  WriteLn('Product pages also quote Nano 8GB at 102 GB/s and Nano 4GB at 51 GB/s.');
  WriteLn('Those match a higher EMC than the r36.4 15W/10W rows (Super / MaxN).');
  WriteLn('Use the row for the nvpmodel you actually set. Query it with:');
  WriteLn('  sudo nvpmodel -q');
  WriteLn('  tegrastats --interval 1000');
end;

var
  Why: string;
  M, N, K: Integer;
  A: TGemmAccount;
begin
  if HasFlag('check') then
  begin
    if not AccountsConsistent(Why) then Die(Why);
    WriteLn('tmoe_cost: account identities hold');
    Halt(0);
  end;
  SkuTable;
  WriteLn;
  WriteLn('Worked example, static only. Shape is one small-config gate GEMM');
  WriteLn('with 32 tokens on one expert (M=32, N=1024, K=512).');
  WriteLn('Assumptions labeled: AGX Orin 64GB, r36.4 nvpmodel, 2 SMs per TPC,');
  WriteLn('EMC ceiling scaled linearly from 204.8 GB/s at 3200 MHz. No device');
  WriteLn('was queried. The roof is not a predicted kernel time.');
  A := AccountGemm(32, 1024, 512);
  Show('small gate M=32 N=1024 K=512', A);
  DramNote(A, 'AGX64 MAXN est 204.8 GB/s', 204.8e9);
  DramNote(A, 'AGX64 30W est 204.8 GB/s (same EMC cap, fewer SMs)', 204.8e9);
  DramNote(A, 'AGX64 15W est 136.5 GB/s', 256 / 8 * 2 * 2133e6);
  WriteLn('  30W and MAXN share this EMC estimate; they do not share SM count');
  WriteLn('  (4 TPC / 8 SM vs 8 TPC / 16 SM) or GPU clock (612 vs 1301 MHz).');
  WriteLn('  Issued bytes are ~400 KiB, under the AGX 4 MB L2 cited in the brief,');
  WriteLn('  so a warm cache can make DRAM traffic smaller than the issued-byte');
  WriteLn('  count. The roof above assumes the opposite (every load misses).');
  WriteLn;
  A := AccountGemm(1, 4096, 4096);
  Show('GEMV-sized M=1 N=4096 K=4096', A);
  WriteLn('  Packed weights are N*K/4 = 4 MiB plus scales. fp32 weights would be');
  WriteLn('  N*K*4 = 64 MiB. Packing cuts the compulsory weight term by 16x and');
  WriteLn('  adds the scale vector (K/128 floats per row). It does not remove');
  WriteLn('  the load: a GEMV still streams the packed matrix. AGX L2 is 4 MB,');
  WriteLn('  so this matrix is the same order as L2 and may not stay resident.');
  DramNote(A, 'AGX64 full-rate 204.8 GB/s', 204.8e9);
  WriteLn;
  M := ArgInt('m', 0);
  N := ArgInt('n', 0);
  K := ArgInt('k', 0);
  if (M > 0) and (N > 0) and (K > 0) then
    Show(Format('requested M=%d N=%d K=%d', [M, N, K]), AccountGemm(M, N, K));
  WriteLn;
  WriteLn('No dp4a/clock figure is applied. The Orin brief''s "up to 5.3 FP32');
  WriteLn('TFLOPs" for AGX 64GB matches 2048 cores * 2 * 1.3 GHz and is not a');
  WriteLn('dp4a throughput. Barriers are not added onto that as if they were');
  WriteLn('serial extra cycles. Occupancy needs the compiler''s register count');
  WriteLn('(ptxas) and the device''s max warps; neither is invented here.');
end.
