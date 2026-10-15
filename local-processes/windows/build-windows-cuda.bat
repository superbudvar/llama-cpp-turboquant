@echo off
setlocal
if "%~1"=="" (
  echo Usage: build-windows-cuda.bat -ProfilePath Z:\local-processes\profiles\llama-cpp-turboquant.json -ExpectedCommit SHA40 -ScratchSerial SERIAL -ScratchDriveLetter R -RunId ID
  exit /b 2
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
exit /b %ERRORLEVEL%
