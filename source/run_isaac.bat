@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0.." || exit /b 1
if not defined ISAAC_SIM_PATH if exist ".isaac-sim-path" set /p "ISAAC_SIM_PATH="<".isaac-sim-path"
if not defined ISAAC_SIM_PATH goto missing
if not exist "%ISAAC_SIM_PATH%\python.bat" goto missing
call "%ISAAC_SIM_PATH%\python.bat" %*
set "run_status=%errorlevel%"
popd
exit /b %run_status%

:missing
echo Set ISAAC_SIM_PATH to the Isaac Sim folder containing python.bat. See README.md. 1>&2
popd
exit /b 1
