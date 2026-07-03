@echo off
setlocal
set "KB_DIR=%~dp0"
python "%KB_DIR%kb_cli.py" %*
