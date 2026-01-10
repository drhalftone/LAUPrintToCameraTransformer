# Print-to-Camera Transformer

Train models to transform images between original digital versions and their printed+captured counterparts using a machine vision system.

## Two Directions

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| **Forward** | Original digital | Simulated print | Preview how images will look after printing |
| **Reverse** | Captured print | Restored original | Remove print artifacts, color correction |

## Available Architectures

| Architecture | Resolution | GAN | Parameters | Best For |
|--------------|------------|-----|------------|----------|
| **Pix2Pix** | Fixed (512×512) or Full | Yes | ~34M | High perceptual quality |
| **NAFNet** | Full resolution | No | ~17M | Stable training, any image size |

Both architectures now support **full resolution** processing - no cropping or resizing required.

## Requirements

- Python 3.8+
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+ or 12.x

## Quick Start (Windows)

### Pix2Pix (Fixed 512×512 Resolution)

```batch
train_pix2pix.bat           # Forward: Original → Captured
train_pix2pix_reverse.bat   # Reverse: Captured → Original
```

### Pix2Pix (Full Resolution - No Resize)

```batch
train_pix2pix_fullres.bat           # Forward: Original → Captured
train_pix2pix_fullres_reverse.bat   # Reverse: Captured → Original
```

### NAFNet (Full Resolution - No Resize)

```batch
train_nafnet.bat            # Forward: Original → Captured
train_nafnet_reverse.bat    # Reverse: Captured → Original
```

All scripts handle virtual environment setup, CUDA PyTorch installation, and training automatically.

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

### Pix2Pix (Fixed Resolution)

Images are resized to 512×512 during training.

```bash
# Forward: Original → Captured
python src/train_pix2pix.py --config configs/pix2pix_config.yaml

# Reverse: Captured → Original
python src/train_pix2pix.py --config configs/pix2pix_reverse_config.yaml
```

Output: `outputs_pix2pix/` or `outputs_pix2pix_reverse/`

### Pix2Pix (Full Resolution)

Images are processed at their **native resolution** - no cropping or resizing.

```bash
# Forward: Original → Captured
python src/train_pix2pix.py --config configs/pix2pix_fullres_config.yaml

# Reverse: Captured → Original
python src/train_pix2pix.py --config configs/pix2pix_fullres_reverse_config.yaml
```

Output: `outputs_pix2pix_fullres/` or `outputs_pix2pix_fullres_reverse/`

#### Pix2Pix Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 (fixed) / 1 (fullres) | Batch size |
| `learning_rate` | 2e-4 | Learning rate |
| `max_steps` | 10000 | Training steps |
| `use_gan` | true | Use PatchGAN discriminator |
| `use_perceptual` | true | Use VGG perceptual loss |
| `lambda_l1` | 100 | L1 loss weight (sharpness) |
| `lambda_perceptual` | 10 | Perceptual loss weight |
| `lambda_gan` | 1 | Adversarial loss weight |
| `full_resolution` | false | Enable full resolution mode |
| `max_size` | null | Optional max dimension limit (fullres only) |

**Memory Usage**: ~6-8GB VRAM for batch size 4 at 512×512. Scales with image size in full resolution mode.

**Training Time**: ~35 minutes for 10K steps on RTX 4070 Ti

---

### NAFNet (Full Resolution)

Images are processed at their **native resolution** - no cropping or resizing. The model automatically pads images to be divisible by 16 and removes padding after processing.

```bash
# Forward: Original → Captured
python src/train_nafnet.py --config configs/nafnet_config.yaml

# Reverse: Captured → Original
python src/train_nafnet.py --config configs/nafnet_reverse_config.yaml

# Resume from checkpoint
python src/train_nafnet.py --config configs/nafnet_config.yaml --resume ./outputs_nafnet/checkpoint-5000
```

Output: `outputs_nafnet/` or `outputs_nafnet_reverse/`

#### NAFNet Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Batch size (1 for variable sizes, higher with same_size_batching) |
| `learning_rate` | 1e-3 | Learning rate (higher than Pix2Pix) |
| `max_steps` | 10000 | Training steps |
| `variant` | width32 | Model size: `lite`, `width32`, `width64` |
| `use_perceptual` | true | Use VGG perceptual loss |
| `lambda_l1` | 1.0 | L1 loss weight |
| `lambda_perceptual` | 0.1 | Perceptual loss weight |
| `max_size` | null | Optional max dimension limit |
| `same_size_batching` | true | Group same-size images for efficient batching |

#### NAFNet Variants

| Variant | Parameters | VRAM | Use Case |
|---------|------------|------|----------|
| `lite` | ~2M | Low | Fast experiments, limited GPU |
| `width32` | ~17M | Medium | **Recommended default** |
| `width64` | ~67M | High | Best quality, 24GB+ VRAM |

**Memory Usage**: Scales with image size. For 1920×1080 images, expect ~8-12GB VRAM.

**If you run out of VRAM**:
- Set `max_size: 1920` to limit maximum dimension
- Use `variant: lite` for smaller model
- Reduce `batch_size` to 1

## Inference

### Pix2Pix Inference

```bash
# Single image (fixed resolution)
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input image.png \
  --output result.png

# Single image (full resolution)
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix_fullres/checkpoint-final \
  --input image.png \
  --output result.png \
  --full_resolution

# Batch processing
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input ./input_folder \
  --output ./output_folder
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_size` | 512 | Processing resolution (ignored if --full_resolution) |
| `--full_resolution` | false | Process at native resolution (no resize) |
| `--device` | auto | Device (cuda/cpu) |

---

### NAFNet Inference

NAFNet processes images at their **original resolution**.

```bash
# Single image (full resolution)
python src/inference_nafnet.py \
  --checkpoint ./outputs_nafnet/checkpoint-final \
  --input image.png \
  --output result.png

# Batch processing
python src/inference_nafnet.py \
  --checkpoint ./outputs_nafnet/checkpoint-final \
  --input ./input_folder \
  --output ./output_folder

# For very large images, use tiled processing
python src/inference_nafnet.py \
  --checkpoint ./outputs_nafnet/checkpoint-final \
  --input large_image.png \
  --output result.png \
  --tiles --tile-size 512 --tile-overlap 64
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--variant` | width32 | Model variant (must match training) |
| `--device` | cuda | Device (cuda/cpu) |
| `--tiles` | false | Enable tiled processing for large images |
| `--tile-size` | 512 | Tile size for tiled processing |
| `--tile-overlap` | 64 | Overlap between tiles |

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
- **Resolution**: Fixed 512×512

---

### NAFNet (Nonlinear Activation Free Network)

```
Input (3ch) -> Encoder (4 levels) -> Middle blocks -> Decoder with skip connections -> Output (3ch)
```

- **Architecture**: Encoder-decoder with NAFBlocks (no nonlinear activations)
- **Key features**:
  - SimpleGate: Element-wise multiplication instead of ReLU/GELU
  - Simplified Channel Attention (SCA)
  - Layer Normalization instead of Batch Normalization
- **Losses**: L1 (reconstruction) + VGG Perceptual (no GAN)
- **Resolution**: Any size (fully convolutional)

NAFNet achieves state-of-the-art results on image restoration benchmarks with a simple, efficient design.

**Reference**: Chen et al. "Simple Baselines for Image Restoration" (ECCV 2022)

---

### Diffusion-based (Experimental)

An alternative approach using Stable Diffusion with LoRA fine-tuning. See `src/train.py` for details.

```bash
python src/train.py --config configs/train_config.yaml
```

## License

MIT
