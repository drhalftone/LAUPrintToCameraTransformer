@echo off
echo ========================================
echo Running Ablation Experiments
echo ========================================

cd /d "%~dp0"
call venv\Scripts\activate

echo.
echo [1/3] Training L1 Only...
echo ----------------------------------------
python src/train_pix2pix.py --config configs/ablation_l1_only.yaml

echo.
echo [2/3] Training L1 + Perceptual...
echo ----------------------------------------
python src/train_pix2pix.py --config configs/ablation_l1_perceptual.yaml

echo.
echo [3/3] Training L1 + GAN...
echo ----------------------------------------
python src/train_pix2pix.py --config configs/ablation_l1_gan.yaml

echo.
echo ========================================
echo All ablation training complete!
echo ========================================
echo.
echo Now run evaluate_ablation.bat to get metrics
pause
