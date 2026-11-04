// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Minimal assertion helpers shared by the test programs.
unit testutil;

{$mode objfpc}{$H+}

interface

procedure Check(Cond: Boolean; const What: string);
procedure Section(const Name: string);
// Prints the tally and exits with 1 on any failure, or 77 when skipped.
procedure Finish(const Name: string);
procedure Skip(const Name, Why: string);

implementation

uses SysUtils;

var
  Passed: Integer = 0;
  Failed: Integer = 0;

procedure Check(Cond: Boolean; const What: string);
begin
  if Cond then
  begin
    Inc(Passed);
    WriteLn('  ok    ', What);
  end
  else
  begin
    Inc(Failed);
    WriteLn('  FAIL  ', What);
  end;
end;

procedure Section(const Name: string);
begin
  WriteLn('[', Name, ']');
end;

procedure Finish(const Name: string);
begin
  WriteLn(Name, ': ', Passed, ' passed, ', Failed, ' failed');
  if Failed > 0 then Halt(1);
end;

procedure Skip(const Name, Why: string);
begin
  WriteLn(Name, ': SKIP (', Why, ')');
  Halt(0);
end;

end.
