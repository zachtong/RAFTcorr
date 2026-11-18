@echo off
title RAFTcorr

:: Initialize conda if not already in PATH
where conda >nul 2>&1
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate raftcorr
    ) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate raftcorr
    ) else (
        echo [ERROR] conda not found. Please install Anaconda or Miniconda.
        pause
        exit /b 1
    )
) else (
    call conda activate raftcorr
)

:: Kill any stale RAFTcorr server before starting (prevents ghost process issue)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

python run_prod.py %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start RAFTcorr.
    echo Run install.bat first if you haven't.
    pause
)
