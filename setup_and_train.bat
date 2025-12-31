@echo off
setlocal

echo ========================================
echo  Print-to-Camera Transformer Setup
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
echo Installing requirements...
pip install --upgrade pip
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
echo Data folders:
echo   - Input images:  data\original\
echo   - Output images: data\captured\
echo.

:: Check if data exists
if not exist "data\original" (
    echo WARNING: data\original folder not found.
    echo Please copy your prestineThumbnail files there.
)
if not exist "data\captured" (
    echo WARNING: data\captured folder not found.
    echo Please copy your warpThumbnail files there.
)

echo.
set /p RUN_TRAINING="Do you want to run the training script? (y/n): "

if /i "%RUN_TRAINING%"=="y" (
    echo.
    echo Starting training...
    python src/train.py --config configs/train_config.yaml
) else (
    echo.
    echo To run training later, use:
    echo   venv\Scripts\activate.bat
    echo   python src/train.py --config configs/train_config.yaml
)

echo.
pause
