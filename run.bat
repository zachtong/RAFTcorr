@echo off
title RAFTcorr
cd /d "%~dp0"
set ENV_NAME=raftcorr-app

where conda >nul 2>&1
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate %ENV_NAME%
    ) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate %ENV_NAME%
    ) else (
        echo [ERROR] conda not found. Install Miniconda, then run install.bat.
        pause
        exit /b 1
    )
) else (
    call conda activate %ENV_NAME%
)
if %errorlevel% neq 0 (
    echo [ERROR] the environment %ENV_NAME% does not exist. Run install.bat first.
    pause
    exit /b 1
)

python -m raftcorr gui %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] RAFTcorr stopped with an error. Run install.bat if it was never installed.
    pause
)
