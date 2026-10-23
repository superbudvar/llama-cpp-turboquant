@echo off
rem ============================================================================
rem  ggml-rpc-server launcher for the Windows RPC peer (192.168.5.2).
rem
rem  What it adds over running the exe by hand:
rem    * a timestamped, per-run log file next to the exe
rem    * every console line carries a timestamp
rem    * the "Accepted client connection" / "Client connection closed" pair is
rem      counted instead of printed - the fit solver opens one connection per
rem      probe, so a fit run otherwise fills the screen with identical lines
rem    * the raw, complete output is still written to the log
rem    * GGML_TQ_NATIVE is reported and can be forced, because that is the knob
rem      that decides whether TQ4_1S weights are expanded to q8_0 on load
rem
rem  Usage:
rem    rpc-server.bat                    run with the defaults below
rem    rpc-server.bat -p 50052           override the port
rem    rpc-server.bat --tq-native=0      let the server expand TQ4_1S to q8_0
rem                                       (fast prefill, 1.7x VRAM)
rem    rpc-server.bat --foreground       print raw output, no connection counter
rem
rem  GGML_TQ_NATIVE defaults to 1 here: the peer keeps TQ4_1S weights native.
rem  The peer has 8191 MiB of VRAM and the server-side default (expand to
rem  q8_0) costs 1.7x the VRAM, which does not fit for the 35B MoE.
rem
rem  Stop with Ctrl-C.
rem ============================================================================

setlocal

rem ---------------------------------------------------------------- settings
rem %~dp0 keeps the trailing backslash; %%~fI drops it.
for %%I in ("%~dp0.") do set "RPC_DIR=%%~fI"
set "RPC_EXE=%RPC_DIR%\ggml-rpc-server.exe"
set "RPC_HOST=0.0.0.0"
set "RPC_PORT=50052"
set "RPC_THREADS="
set "RPC_DEVICE="
set "RPC_LOG_DIR=%RPC_DIR%\logs"
set "RPC_FOREGROUND=0"
rem Default: keep TQ4_1S weights native on this peer (no 1.7x q8_0 expansion).
set "RPC_TQ_NATIVE=1"

rem ------------------------------------------------------------------- args
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-p"        (set "RPC_PORT=%~2"    & shift & shift & goto parse_args)
if /I "%~1"=="--port"    (set "RPC_PORT=%~2"    & shift & shift & goto parse_args)
if /I "%~1"=="-H"        (set "RPC_HOST=%~2"    & shift & shift & goto parse_args)
if /I "%~1"=="--host"    (set "RPC_HOST=%~2"    & shift & shift & goto parse_args)
if /I "%~1"=="-t"        (set "RPC_THREADS=%~2" & shift & shift & goto parse_args)
if /I "%~1"=="--threads" (set "RPC_THREADS=%~2" & shift & shift & goto parse_args)
if /I "%~1"=="-d"        (set "RPC_DEVICE=%~2"  & shift & shift & goto parse_args)
if /I "%~1"=="--device"  (set "RPC_DEVICE=%~2"  & shift & shift & goto parse_args)
if /I "%~1"=="--tq-native=1" (set "RPC_TQ_NATIVE=1" & shift & goto parse_args)
if /I "%~1"=="--tq-native=0" (set "RPC_TQ_NATIVE=0" & shift & goto parse_args)
if /I "%~1"=="--foreground"  (set "RPC_FOREGROUND=1" & shift & goto parse_args)
if /I "%~1"=="--help" goto usage
echo unknown option: %~1
goto usage
:args_done

:usage
echo Usage: %~nx0 [-H host] [-p port] [-t threads] [-d device]
echo                 [--tq-native=0^|1] [--foreground]
exit /b 1

rem ------------------------------------------------------------- preflight
if not exist "%RPC_EXE%" (
    echo ERROR: %RPC_EXE% not found.
    echo    Copy the built bin\ directory next to this script, or edit RPC_DIR.
    exit /b 1
)
if not exist "%RPC_LOG_DIR%" mkdir "%RPC_LOG_DIR%" 2>nul

rem One log file per run, so nothing is overwritten and runs stay separable.
rem The stamp comes from PowerShell so it is sortable and locale independent.
rem %date%/%time% have a locale dependent field order and can lose a token.
for /f "tokens=1" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%t"
if "%STAMP%"=="" set "STAMP=no-stamp"
set "LOG=%RPC_LOG_DIR%\rpc-server-%STAMP%.log"

rem The TQ4_1S expansion decision is made by the server process, so the env var
rem has to be set here, not on the client. The client's own GGML_TQ_NATIVE has
rem no effect on the weights that live on the peer.
rem
rem Default (native): the peer keeps the 5 bpw TQ4_1S bytes. The peer has
rem 8191 MiB of VRAM and the conversion would cost 1.7x that, so the expanded
rem copy does not fit for the 35B MoE. The price is 2x slower prefill on the
rem peer, because cuBLAS dequant-to-f16 needs a per-element inverse WHT.
rem With --tq-native=0 the server expands to q8_0 instead: fast prefill,
rem 1.7x the VRAM.
if not "%RPC_TQ_NATIVE%"=="" set "GGML_TQ_NATIVE=%RPC_TQ_NATIVE%"

rem Pass everything to the helper through the environment, so nothing has to be
rem quoted into the PowerShell command line.
set "RPC_LOG=%LOG%"
set "RPC_ARGV=-H %RPC_HOST% -p %RPC_PORT%"
if not "%RPC_THREADS%"=="" set "RPC_ARGV=%RPC_ARGV% -t %RPC_THREADS%"
if not "%RPC_DEVICE%"==""  set "RPC_ARGV=%RPC_ARGV% -d %RPC_DEVICE%"

rem ------------------------------------------------------------------ header
rem The exe's hash, so a log identifies its own build. Get-FileHash prints one
rem bare hex line, so there is nothing to parse out of a multi-line output.
set "SHA=unknown"
for /f "tokens=1" %%h in ('powershell -NoProfile -Command "$h = Get-FileHash -Algorithm SHA256 -LiteralPath $env:RPC_EXE; $h.Hash"') do set "SHA=%%h"

echo.
echo   ggml-rpc-server
echo   -----------
echo   exe             : %RPC_EXE%
echo   cmd             : %RPC_EXE% %RPC_ARGV%
echo   log             : %LOG%
echo   sha256          : %SHA%
echo   GGML_TQ_NATIVE  : %GGML_TQ_NATIVE%
if "%GGML_TQ_NATIVE%"=="1" echo                     (1 = keep TQ4_1S native, no 1.7x VRAM expansion)
if "%GGML_TQ_NATIVE%"=="0" echo                     (0 = expand TQ4_1S to q8_0, fast prefill, 1.7x VRAM)
if "%GGML_TQ_NATIVE%"==""  echo                     (unset = expand TQ4_1S to q8_0, fast prefill, 1.7x VRAM)
echo   stop            : Ctrl-C
echo.

rem Header into the log as well, so a log is self-describing.
> "%LOG%" echo ==== ggml-rpc-server started %date% %time% ====
>>"%LOG%" echo ==== log: %LOG%
>>"%LOG%" echo ==== cmd: %RPC_EXE% %RPC_ARGV%
>>"%LOG%" echo ==== sha256: %SHA%
>>"%LOG%" echo ==== GGML_TQ_NATIVE: %GGML_TQ_NATIVE%
>>"%LOG%" echo ==== raw output follows ====
>>"%LOG%" echo ====

if "%RPC_FOREGROUND%"=="1" goto foreground

rem -------------------------------------------------------------- filtered run
rem The server prints two lines per client connection and flushes after each,
rem so piping keeps the order and does not delay the lines.
rem Here the connection pair is counted; every other line is timestamped and
rem shown. The log keeps the raw text, unfiltered.
rem
rem Everything the PowerShell needs comes from the environment. Inside the
rem quoted argument cmd passes > & | through literally, so they are written
rem unescaped for PowerShell to interpret.
rem
rem RPC_ARGV is split on spaces before use: a PowerShell variable passed to a
rem native command is one argument, not a list, so it has to be split here.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Continue'; $log = $env:RPC_LOG; $raw = New-Object System.IO.StreamWriter($log, $true); $conn = 0; $closed = 0; $argv = $env:RPC_ARGV -split ' '; & $env:RPC_EXE @argv 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $line = $_.Exception.Message } else { $line = [string]$_ }; $line = $line.TrimEnd(); $raw.WriteLine($line); $raw.Flush(); $ts = (Get-Date).ToString('HH:mm:ss.fff'); if ($line -eq 'Accepted client connection') { $conn++; Write-Host ('[{0}] conn #{1} accepted' -f $ts, $conn) } elseif ($line -eq 'Client connection closed') { $closed++; Write-Host ('[{0}] conn #{1} closed ({2} still open)' -f $ts, $closed, ($conn - $closed)) } else { Write-Host ('[{0}] {1}' -f $ts, $line) } }; $raw.WriteLine('==== end of raw output ===='); $raw.Flush(); $raw.Close()"

echo.
echo Server stopped. Log: %LOG%
goto done

rem ----------------------------------------------------------- foreground run
:foreground
rem Raw output on the console, no counting, and the same file in the log.
rem Use this when you want to see exactly what the server prints, in order.
echo   (foreground mode: raw output; the log is still written)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Continue'; $log = $env:RPC_LOG; $raw = New-Object System.IO.StreamWriter($log, $true); $argv = $env:RPC_ARGV -split ' '; & $env:RPC_EXE @argv 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $line = $_.Exception.Message } else { $line = [string]$_ }; $line = $line.TrimEnd(); $raw.WriteLine($line); $raw.Flush(); Write-Host $line }; $raw.WriteLine('==== end of raw output ===='); $raw.Flush(); $raw.Close()"

echo.
echo Server stopped. Log: %LOG%

:done
endlocal
exit /b 0
