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
:: The same detection as the Windows installer (old and new nvidia-smi headers).
set TORCH_BUILD=cpu
for /f "tokens=1,* delims=|" %%A in ('python packaging\windows\setup_runtime.py --detect') do (
    set TORCH_BUILD=%%A
    echo   %%B
)
set TORCH_INDEX=https://download.pytorch.org/whl/!TORCH_BUILD!

echo [3/5] Installing PyTorch...
:: The rule of the Windows installer (setup_runtime.py torch_action): a PyTorch of the build chosen
:: above (the "+cu128" or "+cpu" of its version) is kept, so re-running downloads nothing. Any
:: other build, e.g. the CPU build after a driver update made CUDA usable, is uninstalled first:
:: "pip install --upgrade" would keep it as "already satisfied".
set TORCH_ACTION=install
for /f %%A in ('python packaging\windows\setup_runtime.py --torch-action !TORCH_BUILD!') do set TORCH_ACTION=%%A
if "!TORCH_ACTION!"=="keep" (
    echo   PyTorch !TORCH_BUILD! build already installed, keeping it
) else (
    if "!TORCH_ACTION!"=="replace" (
        echo   replacing the installed PyTorch with the !TORCH_BUILD! build
        pip uninstall -y torch >> "%LOGFILE%" 2>&1
    )
    echo   downloading PyTorch !TORCH_BUILD! ^(about 2.5 GB^)
    pip install torch --index-url !TORCH_INDEX! >> "%LOGFILE%" 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] PyTorch installation failed, see install.log
        pause
        exit /b 1
    )
)

echo [4/5] Installing RAFTcorr...
pip install -e . >> "%LOGFILE%" 2>&1
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
