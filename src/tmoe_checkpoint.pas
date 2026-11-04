// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Checkpoint file (little-endian), see docs/training.md:
//   "TMOECKP1"  u32 version  config  u32 flags  i32 step
//   u32 ntensors { u16 len, name, u8 kind, i32 count, i32 rows, i32 cols, i64 offset }
//   i64 nfloats, float32 params [, float32 adam_m, float32 adam_v when flags&1]
//   u32 nmatrices { i32 rows, i32 cols, u8 bits[rows*cols/4], f32 scales[rows*cols/128] }
//   u64 FNV-1a of every byte above
// The packed section holds the routed experts in resident 2-bit form,
// ordered layer, then gate/up/down, then expert.
unit tmoe_checkpoint;

{$mode objfpc}{$H+}

interface

uses tmoe_config, tmoe_tensor, tmoe_params, tmoe_ternary, tmoe_model;

const
  CKPT_FLAG_ADAM = 1;

type
  TTernaryArray = array of TTernaryMatrix;

procedure SaveCheckpoint(const FileName: string; var S: TParamStore; WithAdam: Boolean);
// Rebuilds the layout from the stored config. Packed experts are returned
// in the same order as written.
procedure LoadCheckpoint(const FileName: string; out S: TParamStore;
  out PackedExperts: TTernaryArray);
// Replace the packed experts of a built model with ones read from a file.
procedure UsePackedExperts(var M: TModel; const P: TTernaryArray);
function Fnv1a64(P: PByte; N: PtrInt): QWord;

implementation

uses Classes, SysUtils;

const
  MAGIC: array[0..7] of Char = 'TMOECKP1';
  VERSION = 1;

function Fnv1a64(P: PByte; N: PtrInt): QWord;
var I: PtrInt;
begin
  Result := QWord($CBF29CE484222325);
  for I := 0 to N - 1 do
  begin
    Result := Result xor P[I];
    Result := Result * QWord($100000001B3);
  end;
end;

procedure WI(St: TStream; V: LongInt); begin St.WriteBuffer(V, 4); end;
procedure WF(St: TStream; V: Single); begin St.WriteBuffer(V, 4); end;
procedure WB(St: TStream; V: Byte); begin St.WriteBuffer(V, 1); end;
function RI(St: TStream): LongInt; begin St.ReadBuffer(Result, 4); end;
function RF(St: TStream): Single; begin St.ReadBuffer(Result, 4); end;
function RB(St: TStream): Byte; begin St.ReadBuffer(Result, 1); end;

procedure WriteConfig(St: TStream; const C: TMoeConfig);
begin
  WI(St, C.Vocab); WI(St, C.Dim); WI(St, C.Layers); WI(St, C.Heads); WI(St, C.KVHeads);
  WI(St, C.HeadDim); WI(St, C.SharedHidden); WI(St, C.ExpertHidden); WI(St, C.Experts);
  WI(St, C.TopK); WF(St, C.CapacityFactor); WB(St, Ord(C.RenormTopK)); WF(St, C.AuxCoef);
  WI(St, C.MaxSeq); WF(St, C.RopeTheta); WF(St, C.NormEps);
  WB(St, Ord(C.TernaryExperts)); WB(St, Ord(C.ActQuant));
end;

function ReadConfig(St: TStream): TMoeConfig;
begin
  Result.Vocab := RI(St); Result.Dim := RI(St); Result.Layers := RI(St);
  Result.Heads := RI(St); Result.KVHeads := RI(St); Result.HeadDim := RI(St);
  Result.SharedHidden := RI(St); Result.ExpertHidden := RI(St); Result.Experts := RI(St);
  Result.TopK := RI(St); Result.CapacityFactor := RF(St); Result.RenormTopK := RB(St) <> 0;
  Result.AuxCoef := RF(St); Result.MaxSeq := RI(St); Result.RopeTheta := RF(St);
  Result.NormEps := RF(St); Result.TernaryExperts := RB(St) <> 0; Result.ActQuant := RB(St) <> 0;
end;

procedure SaveCheckpoint(const FileName: string; var S: TParamStore; WithAdam: Boolean);
var
  St: TMemoryStream;
  I, L, Kd, E, Rows, Cols: Integer;
  N: Word;
  Ofs: PtrInt;
  Tot: Int64;
  H: QWord;
  M: TTernaryMatrix;
begin
  St := TMemoryStream.Create;
  try
    St.WriteBuffer(MAGIC, 8);
    WI(St, VERSION);
    WriteConfig(St, S.Cfg);
    if WithAdam then WI(St, CKPT_FLAG_ADAM) else WI(St, 0);
    WI(St, S.Step);
    WI(St, Length(S.Info));
    for I := 0 to High(S.Info) do
      with S.Info[I] do
      begin
        N := Length(Name);
        St.WriteBuffer(N, 2);
        St.WriteBuffer(Name[1], N);
        WB(St, Ord(Kind));
        WI(St, Count); WI(St, Rows); WI(St, Cols);
        Tot := Offset;
        St.WriteBuffer(Tot, 8);
      end;
    Tot := S.Total;
    St.WriteBuffer(Tot, 8);
    St.WriteBuffer(S.Data[0], S.Total * SizeOf(Single));
    if WithAdam then
    begin
      St.WriteBuffer(S.M[0], S.Total * SizeOf(Single));
      St.WriteBuffer(S.V[0], S.Total * SizeOf(Single));
    end;
    WI(St, S.Cfg.Layers * 3 * S.Cfg.Experts);
    for L := 0 to S.Cfg.Layers - 1 do
      for Kd := 0 to 2 do
        for E := 0 to S.Cfg.Experts - 1 do
        begin
          case Kd of
            0: begin Ofs := S.L[L].ExGate; Rows := S.Cfg.ExpertHidden; Cols := S.Cfg.Dim; end;
            1: begin Ofs := S.L[L].ExUp; Rows := S.Cfg.ExpertHidden; Cols := S.Cfg.Dim; end;
          else
            begin Ofs := S.L[L].ExDown; Rows := S.Cfg.Dim; Cols := S.Cfg.ExpertHidden; end;
          end;
          QuantizeAbsmean(@S.Data[Ofs + PtrInt(E) * Rows * Cols], Rows, Cols, M);
          WI(St, Rows); WI(St, Cols);
          St.WriteBuffer(M.Bits[0], Length(M.Bits));
          St.WriteBuffer(M.Scales[0], Length(M.Scales) * SizeOf(Single));
        end;
    H := Fnv1a64(St.Memory, St.Size);
    St.WriteBuffer(H, 8);
    St.SaveToFile(FileName);
  finally
    St.Free;
  end;
end;

procedure LoadCheckpoint(const FileName: string; out S: TParamStore;
  out PackedExperts: TTernaryArray);
var
  St: TMemoryStream;
  Mg: array[0..7] of Char;
  C: TMoeConfig;
  Flags, NT, I, R, Cl: Integer;
  N: Word;
  Nm: string;
  Kind: Byte;
  Cnt, Rows, Cols: Integer;
  Ofs, Tot: Int64;
  H, Stored: QWord;
  Msg: string;
begin
  St := TMemoryStream.Create;
  try
    St.LoadFromFile(FileName);
    if St.Size < 16 then raise Exception.Create('checkpoint too small');
    H := Fnv1a64(St.Memory, St.Size - 8);
    Move(PByte(St.Memory)[St.Size - 8], Stored, 8);
    if H <> Stored then raise Exception.Create('checkpoint checksum mismatch');
    St.ReadBuffer(Mg, 8);
    if Mg <> MAGIC then raise Exception.Create('not a TMOE checkpoint');
    if RI(St) <> VERSION then raise Exception.Create('unsupported checkpoint version');
    C := ReadConfig(St);
    if not ValidateConfig(C, Msg) then raise Exception.Create('bad config: ' + Msg);
    BuildLayout(S, C);
    Flags := RI(St);
    S.Step := RI(St);
    NT := RI(St);
    if NT <> Length(S.Info) then raise Exception.Create('tensor count mismatch');
    for I := 0 to NT - 1 do
    begin
      St.ReadBuffer(N, 2);
      SetLength(Nm, N);
      if N > 0 then St.ReadBuffer(Nm[1], N);
      Kind := RB(St);
      Cnt := RI(St); Rows := RI(St); Cols := RI(St);
      St.ReadBuffer(Ofs, 8);
      if (Nm <> S.Info[I].Name) or (Kind <> Ord(S.Info[I].Kind)) or (Cnt <> S.Info[I].Count) or
         (Rows <> S.Info[I].Rows) or (Cols <> S.Info[I].Cols) or (Ofs <> S.Info[I].Offset) then
          raise Exception.Create('tensor table mismatch at ' + Nm);
    end;
    St.ReadBuffer(Tot, 8);
    if Tot <> S.Total then raise Exception.Create('parameter count mismatch');
    St.ReadBuffer(S.Data[0], S.Total * SizeOf(Single));
    if (Flags and CKPT_FLAG_ADAM) <> 0 then
    begin
      AllocTrainState(S);
      St.ReadBuffer(S.M[0], S.Total * SizeOf(Single));
      St.ReadBuffer(S.V[0], S.Total * SizeOf(Single));
    end;
    NT := RI(St);
    PackedExperts := nil;
    SetLength(PackedExperts, NT);
    for I := 0 to NT - 1 do
    begin
      R := RI(St); Cl := RI(St);
      InitTernary(PackedExperts[I], R, Cl);
      St.ReadBuffer(PackedExperts[I].Bits[0], Length(PackedExperts[I].Bits));
      St.ReadBuffer(PackedExperts[I].Scales[0], Length(PackedExperts[I].Scales) * SizeOf(Single));
    end;
  finally
    St.Free;
  end;
end;

procedure UsePackedExperts(var M: TModel; const P: TTernaryArray);
var L, E, I: Integer;
begin
  I := 0;
  for L := 0 to M.Cfg.Layers - 1 do
  begin
    for E := 0 to M.Cfg.Experts - 1 do begin M.Layers[L].Moe.ExGate[E] := P[I]; Inc(I); end;
    for E := 0 to M.Cfg.Experts - 1 do begin M.Layers[L].Moe.ExUp[E] := P[I]; Inc(I); end;
    for E := 0 to M.Cfg.Experts - 1 do begin M.Layers[L].Moe.ExDown[E] := P[I]; Inc(I); end;
  end;
end;

end.
