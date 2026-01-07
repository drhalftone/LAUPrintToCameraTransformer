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
