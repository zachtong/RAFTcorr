@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   RAFTcorr Installer
echo ============================================
echo.
echo Requires: Anaconda/Miniconda, NVIDIA GPU with CUDA
echo.

:: Redirect all output to log file (while still showing on screen)
set "LOGFILE=%~dp0install.log"
echo Installation started: %date% %time% > "%LOGFILE%"
echo Working directory: %CD% >> "%LOGFILE%"

:: ---- Step 1: Check conda ----
echo [1/5] Checking for conda...
where conda >nul 2>&1
if !errorlevel! neq 0 (
    echo   conda not on PATH, checking common install locations...
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        echo   Found Anaconda3 at %USERPROFILE%\anaconda3
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate base
    ) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        echo   Found Miniconda3 at %USERPROFILE%\miniconda3
        call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate base
    ) else if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" (
        echo   Found Anaconda3 at %LOCALAPPDATA%\anaconda3
        call "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" activate base
    ) else if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" (
        echo   Found Miniconda3 at %LOCALAPPDATA%\miniconda3
        call "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" activate base
    ) else (
        echo.
        echo   ============================================
        echo   [ERROR] Conda not found!
        echo   ============================================
        echo.
        echo   RAFTcorr requires Anaconda or Miniconda to manage
        echo   Python environments and dependencies.
        echo.
        echo   Please install Miniconda ^(recommended^):
        echo     https://docs.anaconda.com/miniconda/
        echo.
        echo   Or Anaconda ^(full distribution^):
        echo     https://www.anaconda.com/download
        echo.
        echo   After installation:
        echo     1. Restart your terminal / command prompt
        echo     2. Run this installer again
        echo.
        echo   Checked locations:
        echo     - PATH
        echo     - %%USERPROFILE%%\anaconda3
        echo     - %%USERPROFILE%%\miniconda3
        echo     - %%LOCALAPPDATA%%\anaconda3
        echo     - %%LOCALAPPDATA%%\miniconda3
        echo.
        echo [ERROR] Conda not found. See above for instructions. >> "%LOGFILE%"
        pause
        exit /b 1
    )
)
echo [OK] conda found. >> "%LOGFILE%"
echo [OK] conda found.

:: ---- Step 2: Create/activate environment ----
echo [2/5] Setting up conda environment...
call conda info --envs | findstr /c:"raftcorr" >nul 2>&1
if !errorlevel! equ 0 (
    echo   Environment "raftcorr" already exists, activating...
    goto :env_ready
)

echo   Creating conda environment "raftcorr" with Python 3.10...
call conda create -n raftcorr python=3.10 -y >> "%LOGFILE%" 2>&1
if !errorlevel! neq 0 (
    echo.
    echo   ============================================
    echo   [ERROR] Failed to create conda environment!
    echo   ============================================
    echo.
    echo   Possible causes:
    echo     - Insufficient disk space
    echo     - Network connectivity issues
    echo     - Corrupted conda installation
    echo.
    echo   Try these fixes:
    echo     1. Run: conda clean --all
    echo     2. Check disk space: dir %USERPROFILE%\anaconda3
    echo     3. Update conda: conda update conda
    echo.
    echo   Full log: %LOGFILE%
    echo.
    echo [ERROR] conda create failed >> "%LOGFILE%"
    pause
    exit /b 1
)
echo [OK] Environment created.

:env_ready
echo.
call conda activate raftcorr
if !errorlevel! neq 0 (
    echo   [ERROR] Failed to activate raftcorr environment.
    echo   Try: conda init cmd.exe, then restart terminal.
    echo [ERROR] conda activate failed >> "%LOGFILE%"
    pause
    exit /b 1
)
echo [OK] Environment "raftcorr" activated.

:: ---- Step 3: Detect CUDA version from nvidia-smi ----
echo [3/5] Detecting GPU and CUDA...
set CUDA_MAJOR=0

:: First check if nvidia-smi exists
where nvidia-smi >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo   ============================================
    echo   [WARNING] nvidia-smi not found!
    echo   ============================================
    echo.
    echo   RAFTcorr requires an NVIDIA GPU with CUDA support.
    echo.
    echo   If you have an NVIDIA GPU:
    echo     1. Install the latest NVIDIA driver:
    echo        https://www.nvidia.com/drivers
    echo     2. Restart your computer
    echo     3. Run this installer again
    echo.
    echo   If you do NOT have an NVIDIA GPU:
    echo     RAFTcorr cannot run without CUDA acceleration.
    echo     A discrete NVIDIA GPU ^(GTX 1060 or better^) is required.
    echo.
    echo [WARNING] nvidia-smi not found >> "%LOGFILE%"
    pause
    exit /b 1
)

:: Run nvidia-smi and extract CUDA version
nvidia-smi >> "%LOGFILE%" 2>&1
for /f "tokens=*" %%L in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "NV_LINE=%%L"
if defined NV_LINE (
    set "CUDA_PART=!NV_LINE:*CUDA Version: =!"
    for /f "tokens=1 delims=. " %%a in ("!CUDA_PART!") do set CUDA_MAJOR=%%a
)

if !CUDA_MAJOR! equ 0 (
    echo.
    echo   ============================================
    echo   [ERROR] CUDA not detected!
    echo   ============================================
    echo.
    echo   nvidia-smi is available but CUDA version could not be read.
    echo.
    echo   Possible causes:
    echo     - Very old NVIDIA driver ^(update to latest^)
    echo     - GPU does not support CUDA
    echo.
    echo   Steps to fix:
    echo     1. Update NVIDIA driver: https://www.nvidia.com/drivers
    echo     2. Verify with: nvidia-smi
    echo     3. Look for "CUDA Version: X.Y" in the output
    echo.
    echo [ERROR] CUDA version parse failed >> "%LOGFILE%"
    pause
    exit /b 1
)

echo [OK] Detected CUDA !CUDA_MAJOR!.x
echo [OK] CUDA !CUDA_MAJOR!.x >> "%LOGFILE%"

:: ---- Step 4: Install PyTorch with CUDA ----
echo [4/5] Installing PyTorch with CUDA support...

:: Detect if GPU needs newer CUDA wheels (Blackwell / RTX 50-series)
set "NEED_CU128=0"

:: Method 1: Check GPU name from nvidia-smi output for RTX 50-series
for /f "tokens=*" %%G in ('nvidia-smi 2^>nul ^| findstr /i "RTX.50"') do set "NEED_CU128=1"

:: Method 2: CUDA 13+ always needs cu128 (Blackwell-era driver)
if !CUDA_MAJOR! geq 13 set "NEED_CU128=1"

if !NEED_CU128! equ 1 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    echo   Blackwell/RTX 50-series detected - using CUDA 12.8 wheels
) else if !CUDA_MAJOR! geq 12 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    echo   Using CUDA 12.4 wheels
) else (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    echo   Using CUDA 11.8 wheels
)

pip install torch torchvision torchaudio --index-url !TORCH_INDEX! >> "%LOGFILE%" 2>&1
if !errorlevel! neq 0 (
    echo.
    echo   ============================================
    echo   [ERROR] PyTorch installation failed!
    echo   ============================================
    echo.
    echo   Possible causes:
    echo     - No internet connection
    echo     - Proxy/firewall blocking PyPI or pytorch.org
    echo     - Insufficient disk space ^(PyTorch needs ~2.5 GB^)
    echo     - pip version too old
    echo.
    echo   Try these fixes:
    echo     1. Check internet: ping pypi.org
    echo     2. Update pip: python -m pip install --upgrade pip
    echo     3. Check disk space
    echo     4. If behind a proxy, set HTTP_PROXY and HTTPS_PROXY
    echo.
    echo   Full log: %LOGFILE%
    echo.
    echo [ERROR] PyTorch install failed >> "%LOGFILE%"
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

:: ---- Step 5: Install RAFTcorr + remaining dependencies ----
echo [5/5] Installing RAFTcorr and dependencies...
pip install -e . >> "%LOGFILE%" 2>&1
if !errorlevel! neq 0 (
    echo.
    echo   ============================================
    echo   [ERROR] RAFTcorr installation failed!
    echo   ============================================
    echo.
    echo   Possible causes:
    echo     - Missing build tools ^(Visual C++ Build Tools^)
    echo     - Dependency version conflicts
    echo     - Corrupted download
    echo.
    echo   Try these fixes:
    echo     1. Install Visual C++ Build Tools:
    echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo     2. Clear pip cache: pip cache purge
    echo     3. Try again: pip install -e .
    echo.
    echo   Full log: %LOGFILE%
    echo.
    echo [ERROR] pip install -e . failed >> "%LOGFILE%"
    pause
    exit /b 1
)
echo [OK] RAFTcorr installed.

:: ---- Verify installation ----
echo.
echo Verifying installation...
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>> "%LOGFILE%"
if !errorlevel! neq 0 (
    echo   [WARNING] PyTorch import check failed. See log for details.
) else (
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo   [WARNING] PyTorch installed but CUDA is not available!
        echo   GPU acceleration will not work. DIC processing will fail.
        echo   Try reinstalling PyTorch with the correct CUDA version.
    )
)

echo.
echo Installation log saved to: %LOGFILE%
echo.
echo ============================================
echo   Installation complete!
echo.
echo   To launch RAFTcorr:
echo     Double-click run.bat
echo     Or run:
echo       conda activate raftcorr
echo       python run_prod.py
echo ============================================
pause
