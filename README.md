# Print-to-Camera Transformer

Train models to transform images between original digital versions and their printed+captured counterparts using a machine vision system.

## Two Directions

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| **Forward** | Original digital | Simulated print | Preview how images will look after printing |
| **Reverse** | Captured print | Restored original | Remove print artifacts, color correction |

## Requirements

- Python 3.8+
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+ or 12.x

## Quick Start (Windows)

### Forward Model (Original → Captured)
```batch
train_pix2pix.bat
```

### Reverse Model (Captured → Original)
```batch
train_pix2pix_reverse.bat
```

Both scripts handle virtual environment setup, CUDA PyTorch installation, and training automatically.

## Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate.bat

# Install PyTorch with CUDA (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements.txt
```

## Data Setup

Place your paired images in:
- `data/original/` - Original digital images (e.g., `prestineThumbnail00001.tif`)
- `data/captured/` - Corresponding printed+captured images (e.g., `warpThumbnail00001.tif`)

Images are paired by their numeric suffix (e.g., `00001`), so different prefixes are OK.

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`

### Validate Data

```bash
python src/dataset.py --data_dir ./data
```

## Training

### Forward Model (Original → Captured)

Simulates how images will look after being printed and captured.

```bash
python src/train_pix2pix.py --config configs/pix2pix_config.yaml
```

Output: `outputs_pix2pix/`

### Reverse Model (Captured → Original)

Restores captured images back to original quality.

```bash
python src/train_pix2pix.py --config configs/pix2pix_reverse_config.yaml
```

Output: `outputs_pix2pix_reverse/`

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Batch size (4-8 for 16GB VRAM) |
| `learning_rate` | 2e-4 | Learning rate |
| `max_steps` | 10000 | Training steps |
| `use_gan` | true | Use PatchGAN discriminator |
| `use_perceptual` | true | Use VGG perceptual loss |
| `lambda_l1` | 100 | L1 loss weight (sharpness) |
| `lambda_perceptual` | 10 | Perceptual loss weight |
| `lambda_gan` | 1 | Adversarial loss weight |

**Memory Usage**: ~6-8GB VRAM for batch size 4

**Training Time**: ~35 minutes for 10K steps on RTX 4070 Ti

## Inference

### Forward Model
```bash
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input image.png \
  --output result.png
```

### Reverse Model
```bash
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix_reverse/checkpoint-final \
  --input captured_image.png \
  --output restored.png
```

### Batch Processing
```bash
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input ./input_folder \
  --output ./output_folder
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_size` | 512 | Processing resolution |
| `--device` | auto | Device (cuda/cpu) |

## Evaluation

Evaluate model quality with quantitative metrics and compare against traditional baselines.

The dataset contains 12,500 paired images. Evaluation uses a 90/10 train/validation split:
- **~11,250 images**: Used to fit baseline methods
- **~1,250 images**: Used for metric computation (statistically robust sample size)

```bash
python src/evaluate.py \
  --checkpoint ./outputs_pix2pix_reverse/checkpoint-final \
  --data_dir ./data \
  --output_dir ./evaluation_results \
  --visualize
```

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio. Measures pixel-wise reconstruction fidelity. | Higher |
| **SSIM** | Structural Similarity Index. Compares luminance, contrast, and structure. More perceptually relevant than PSNR. | Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity. Uses deep network features to measure perceptual distance. Correlates best with human judgment. | Lower |

**LPIPS** is particularly important for image restoration tasks because it captures perceptual quality that PSNR/SSIM miss. A slightly blurry image may have good PSNR but high (bad) LPIPS.

### Baseline Comparisons

The evaluation compares the trained model against traditional image correction methods:

| Baseline | Description |
|----------|-------------|
| **Identity** | No transformation (lower bound) |
| **Reinhard Color Transfer** | Mean/std matching in LAB color space |
| **Channel Regression** | Per-channel linear regression (y = ax + b) |
| **Histogram Matching** | CDF-based histogram matching per channel |

#### Identity
Returns the input unchanged. Establishes a **lower bound** - if the model doesn't beat this, it's not learning anything useful.

#### Reinhard Color Transfer
Matches the mean and standard deviation of each channel in **LAB color space**:
```
output_L = (input_L - mean(input_L)) × (std(target_L) / std(input_L)) + mean(target_L)
```
LAB separates lightness (L) from color (a, b), making the transfer more perceptually accurate than RGB. Target statistics are learned globally from training data; input statistics are computed per-image.

#### Channel Regression
Fits a linear model **y = ax + b** for each RGB channel using least-squares regression on all training pixel pairs:
```
slope = cov(input, target) / var(input)
intercept = mean(target) - slope × mean(input)
```
More flexible than color transfer - can correct cases where dark and bright pixels need different adjustments.

#### Histogram Matching
Transforms each channel's **cumulative distribution function (CDF)** to match the target distribution:
```
output = CDF_target_inverse(CDF_input(pixel_value))
```
The most sophisticated traditional method - can match arbitrary non-linear color transformations, not just linear ones.

#### Why Baselines Matter
If Pix2Pix significantly outperforms all baselines, it demonstrates the model is learning **spatial/structural corrections** (texture, sharpness, local artifacts) that color-only methods cannot achieve.

### Output Files

```
evaluation_results/
├── metrics.csv          # Raw metrics data
├── metrics.json         # Detailed results with metadata
├── report.txt           # Human-readable summary
├── table.tex            # LaTeX table for papers
├── metrics_comparison.png   # Bar chart visualization
└── sample_comparison.png    # Visual comparison grid
```

### References

- **SSIM**: Wang et al. (2004) "Image quality assessment: from error visibility to structural similarity" - IEEE TIP
- **LPIPS**: Zhang et al. (2018) "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" - CVPR
- **Reinhard**: Reinhard et al. (2001) "Color Transfer between Images" - IEEE CG&A

## Output

Training saves results to the output directory:
- `checkpoint-*/` - Model checkpoints (generator.pt, discriminator.pt)
- `results_step_*.png` - Validation samples at each checkpoint
- `results_final.png` - Final validation results with loss curve

Results image columns: **Input | Prediction | Ground Truth**

## Architecture

### Pix2Pix U-Net

```
Input (3ch) -> Encoder (64->128->256->512->1024) -> Decoder with skip connections -> Output (3ch)
```

- **Generator**: ~31M parameters (U-Net with skip connections)
- **Discriminator**: ~2.8M parameters (PatchGAN, 70x70 receptive field)
- **Losses**: L1 (reconstruction) + VGG Perceptual + Adversarial

### Diffusion-based (Experimental)

An alternative approach using Stable Diffusion with LoRA fine-tuning. See `src/train.py` for details.

```bash
python src/train.py --config configs/train_config.yaml
```

## License

MIT
