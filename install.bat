@echo off
setlocal EnableDelayedExpansion
title RAFTcorr installer
cd /d "%~dp0"

:: RAFTcorr 2 installer for Windows.
:: Creates the conda environment "raftcorr-app" (Python 3.11), installs the PyTorch build that
:: matches the NVIDIA driver (CPU build when there is no NVIDIA GPU), then RAFTcorr itself.
:: Requires Anaconda or Miniconda. Log: install.log

set ENV_NAME=raftcorr-app
set LOGFILE=%~dp0install.log
echo RAFTcorr installer > "%LOGFILE%"

:: ---- conda ----
where conda >nul 2>&1
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        set "CONDA=%USERPROFILE%\anaconda3\condabin\conda.bat"
    ) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        set "CONDA=%USERPROFILE%\miniconda3\condabin\conda.bat"
    ) else (
        echo [ERROR] conda was not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
        pause
        exit /b 1
    )
) else (
    set "CONDA=conda"
)

echo [1/5] Creating the conda environment %ENV_NAME% (Python 3.11)...
call "%CONDA%" env list | findstr /b /c:"%ENV_NAME% " >nul 2>&1
if %errorlevel% equ 0 (
    echo   environment exists, reusing it
) else (
    call "%CONDA%" create -n %ENV_NAME% python=3.11 -y >> "%LOGFILE%" 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] conda create failed, see install.log
        pause
        exit /b 1
    )
)
call "%CONDA%" activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo [ERROR] could not activate %ENV_NAME%
    pause
    exit /b 1
)
python -m pip install --upgrade pip >> "%LOGFILE%" 2>&1

echo [2/5] Detecting the GPU...
set CUDA_MAJOR=0
set CUDA_MINOR=0
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%L in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "NV_LINE=%%L"
    if defined NV_LINE (
        set "CUDA_PART=!NV_LINE:*CUDA Version: =!"
        for /f "tokens=1,2 delims=. " %%a in ("!CUDA_PART!") do (
            set CUDA_MAJOR=%%a
            set CUDA_MINOR=%%b
        )
    )
)
set NEED_CU128=0
for /f "tokens=*" %%G in ('nvidia-smi 2^>nul ^| findstr /i "RTX.50"') do set NEED_CU128=1
if !CUDA_MAJOR! geq 13 set NEED_CU128=1
if !CUDA_MAJOR! equ 12 if !CUDA_MINOR! geq 8 set NEED_CU128=1
if !NEED_CU128! equ 1 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    echo   driver supports CUDA !CUDA_MAJOR!.!CUDA_MINOR!: PyTorch with CUDA 12.8
) else if !CUDA_MAJOR! equ 12 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu126
    echo   driver supports CUDA !CUDA_MAJOR!.!CUDA_MINOR!: PyTorch with CUDA 12.6
) else if !CUDA_MAJOR! equ 11 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    echo   driver supports CUDA !CUDA_MAJOR!.!CUDA_MINOR!: PyTorch with CUDA 11.8
) else (
    echo   [WARNING] no NVIDIA GPU found: installing the CPU build. RAFTcorr runs, but slowly.
)

echo [3/5] Installing PyTorch (this downloads about 2.5 GB)...
pip install torch --index-url !TORCH_INDEX! >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch installation failed, see install.log
    pause
    exit /b 1
)

echo [4/5] Installing RAFTcorr...
pip install -e ".[all]" >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] RAFTcorr installation failed, see install.log
    pause
    exit /b 1
)

echo [5/5] Downloading the model files...
:: Not fatal: the application also downloads a model the first time it is used.
python -m raftcorr models --download
echo.
python -m raftcorr info
echo.
echo Done. Start RAFTcorr with run.bat. A model listed above as NOT INSTALLED is downloaded the
echo first time it is used; if it cannot be downloaded, put its file (RAFTcorr-s1.pth ...
echo RAFTcorr-s8.pth) into the "models" folder.
pause
