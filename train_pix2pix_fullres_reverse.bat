@echo off
setlocal

echo ========================================
echo  Pix2Pix FULL RESOLUTION Training
echo  (Reverse: Captured -> Original)
echo ========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

:: Check if PyTorch with CUDA is already installed
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo WARNING: Failed to install PyTorch with CUDA 12.4, trying CUDA 12.1...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    )
) else (
    echo PyTorch with CUDA already installed.
)

echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.

:: Verify CUDA
echo Checking GPU...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
echo.

:: Check if data exists
if not exist "data\original" (
    echo WARNING: data\original folder not found.
    echo Please add your training images before running.
    pause
    exit /b 1
)
if not exist "data\captured" (
    echo WARNING: data\captured folder not found.
    echo Please add your training images before running.
    pause
    exit /b 1
)

echo Data folders found.
echo.
echo ========================================
echo  FULL RESOLUTION Pix2Pix REVERSE Training
echo ========================================
echo.
echo This trains the REVERSE model at full resolution:
echo   Input:  Captured/printed images
echo   Output: Restored original images
echo.
echo Use this model to restore printed documents
echo back to their original digital quality.
echo.
echo Starting Pix2Pix full-resolution reverse training...
echo Output will be saved to: outputs_pix2pix_fullres_reverse\
echo.

python src/train_pix2pix.py --config configs/pix2pix_fullres_reverse_config.yaml

echo.
echo ========================================
echo  Training complete!
echo ========================================
echo.
echo Results saved to outputs_pix2pix_fullres_reverse\
echo.
pause
