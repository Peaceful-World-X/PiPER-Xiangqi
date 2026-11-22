@echo off
call "%~dp0source\run_isaac.bat" "%~dp0source\render_isaac.py" --path-tracing %*
exit /b %errorlevel%
