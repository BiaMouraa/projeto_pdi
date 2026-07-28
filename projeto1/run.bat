@echo off
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PY%" (
    echo Python 3.12 nao encontrado em: %PY%
    echo Instale Python ou ajuste o caminho neste arquivo.
    pause
    exit /b 1
)
cd /d "%~dp0"
"%PY%" %*
