@echo off
echo ========================================
echo Evaluating Ablation Models
echo ========================================

cd /d "%~dp0"
call venv\Scripts\activate

echo.
echo [1/4] Evaluating L1 Only...
python src/evaluate.py --checkpoint outputs_ablation_l1_only/checkpoint-final --data_dir data --original_subdir captured --captured_subdir original --output_dir evaluation_ablation_l1_only

echo.
echo [2/4] Evaluating L1 + Perceptual...
python src/evaluate.py --checkpoint outputs_ablation_l1_perceptual/checkpoint-final --data_dir data --original_subdir captured --captured_subdir original --output_dir evaluation_ablation_l1_perceptual

echo.
echo [3/4] Evaluating L1 + GAN...
python src/evaluate.py --checkpoint outputs_ablation_l1_gan/checkpoint-final --data_dir data --original_subdir captured --captured_subdir original --output_dir evaluation_ablation_l1_gan

echo.
echo [4/4] Evaluating Full Model (L1 + Perceptual + GAN)...
python src/evaluate.py --checkpoint outputs_pix2pix_reverse/checkpoint-final --data_dir data --original_subdir captured --captured_subdir original --output_dir evaluation_ablation_full

echo.
echo ========================================
echo Ablation evaluation complete!
echo ========================================
echo.
echo Results saved to:
echo   - evaluation_ablation_l1_only/
echo   - evaluation_ablation_l1_perceptual/
echo   - evaluation_ablation_l1_gan/
echo   - evaluation_ablation_full/
pause
