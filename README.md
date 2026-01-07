# Print-to-Camera Transformer

Train a model to predict how images will look after being printed on a digital label printer and captured by a machine vision system.

## Training Approaches

This project supports two training approaches:

### 1. Pix2Pix U-Net (Recommended)

A direct image-to-image translation approach using a U-Net encoder-decoder with optional adversarial training. Best for this task as it directly learns the transformation without intermediate steps.

### 2. Diffusion-based (Experimental)

Uses the [Marigold](https://github.com/prs-eth/Marigold) architecture (fine-tuned Stable Diffusion with LoRA). More complex but may capture finer details in some cases.

## Requirements

- Python 3.8+
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+ or 12.x

## Quick Start (Windows)

Run the setup script which handles virtual environment creation and CUDA-enabled PyTorch installation:

```batch
setup_and_train.bat
```

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

### Pix2Pix (Recommended)

```bash
python src/train_pix2pix.py --config configs/pix2pix_config.yaml
```

**Key Parameters** (`configs/pix2pix_config.yaml`):

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

**Training Time**: ~15-20 minutes for 10K steps on RTX 4070 Ti

### Diffusion-based (Experimental)

```bash
python src/train.py --config configs/train_config.yaml
```

**Key Parameters** (`configs/train_config.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Batch size (1-2 for 12GB VRAM) |
| `learning_rate` | 5e-5 | Learning rate |
| `max_steps` | 5000 | Training steps |
| `lora_rank` | 32 | LoRA rank (higher = more capacity) |

**Memory Usage**: ~10-12GB VRAM for batch size 1

## Inference

### Pix2Pix Model

```bash
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input image.png \
  --output result.png
```

### Diffusion Model

```bash
python src/inference.py \
  --checkpoint ./outputs/checkpoint-final \
  --input image.png \
  --output result.png
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_steps` | 50 | DDIM denoising steps (diffusion only) |
| `--seed` | None | Random seed for reproducibility |

## Output

Training saves results to the output directory:
- `checkpoint-*/` - Model checkpoints
- `results_step_*.png` - Validation samples at each checkpoint
- `results_final.png` - Final validation results with loss curve

## Architecture

### Pix2Pix U-Net

```
Input (3ch) -> Encoder (64->128->256->512->1024) -> Decoder with skip connections -> Output (3ch)
```

- ~31M parameters
- Direct image-to-image mapping
- Trained with L1 + Perceptual + Adversarial losses

### Diffusion (Marigold-based)

Based on [Marigold](https://arxiv.org/abs/2312.02145):
1. Encode input image to latent space (VAE)
2. Concatenate with noisy target latent (8 channels)
3. U-Net denoises toward target
4. Decode output latent to image

LoRA fine-tuning trains ~0.76% of parameters for memory efficiency.

## License

MIT
