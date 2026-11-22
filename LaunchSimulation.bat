@echo off
call "%~dp0source\run_isaac.bat" "%~dp0source\play_isaac.py" %*
exit /b %errorlevel%
