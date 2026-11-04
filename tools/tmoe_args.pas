// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// --name value command-line options for the tools.
unit tmoe_args;

{$mode objfpc}{$H+}

interface

uses tmoe_config;

function ArgStr(const Name, Def: string): string;
function ArgInt(const Name: string; Def: Integer): Integer;
function ArgFloat(const Name: string; Def: Double): Double;
function HasFlag(const Name: string): Boolean;
// --config tiny|small, plus --experts/--topk/--capacity overrides
function ArgConfig: TMoeConfig;

implementation

uses SysUtils;

function ArgStr(const Name, Def: string): string;
var I: Integer;
begin
  Result := Def;
  for I := 1 to ParamCount - 1 do
    if ParamStr(I) = '--' + Name then Exit(ParamStr(I + 1));
end;

function ArgInt(const Name: string; Def: Integer): Integer;
begin
  Result := StrToIntDef(ArgStr(Name, ''), Def);
end;

function ArgFloat(const Name: string; Def: Double): Double;
var Fs: TFormatSettings;
begin
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  Result := StrToFloatDef(ArgStr(Name, ''), Def, Fs);
end;

function HasFlag(const Name: string): Boolean;
var I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if ParamStr(I) = '--' + Name then Exit(True);
end;

function ArgConfig: TMoeConfig;
var Msg, N: string;
begin
  N := LowerCase(ArgStr('config', 'tiny'));
  if N = 'small' then Result := SmallConfig
  else if N = 'tiny' then Result := TinyConfig
  else
  begin
    WriteLn(StdErr, 'unknown --config ', N, ' (tiny or small)');
    Halt(2);
  end;
  Result.Experts := ArgInt('experts', Result.Experts);
  Result.TopK := ArgInt('topk', Result.TopK);
  Result.CapacityFactor := ArgFloat('capacity', Result.CapacityFactor);
  if HasFlag('renorm') then Result.RenormTopK := True;
  if not ValidateConfig(Result, Msg) then
  begin
    WriteLn(StdErr, 'bad config: ', Msg);
    Halt(2);
  end;
end;

end.
