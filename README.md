# Print-to-Camera Transformer

A novel **hybrid diffusion-Pix2Pix framework** for bidirectional image transformation between original digital images and their printed+captured counterparts.

## Key Contribution: Hybrid Diffusion with Direct Supervision

This project introduces a novel approach that combines the generative power of latent diffusion models with the direct supervision of Pix2Pix-style losses:

1. **Marigold-style Conditioning**: Concatenates input image latents with noisy target latents (8-channel input)
2. **LoRA Fine-tuning**: Parameter-efficient training (~1.6M trainable params vs 860M total)
3. **Hybrid Loss Function**: Combines diffusion noise prediction with direct latent reconstruction and perceptual losses

This hybrid approach aims to leverage diffusion models' generative capabilities while maintaining the pixel-accurate supervision that makes Pix2Pix effective for paired image translation.

> **Note**: Diffusion model results are pending experimental validation. Pix2Pix baseline achieves 26.65 dB PSNR on the reverse task. Comparative results will be added once diffusion training is complete.

## Two Directions

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| **Forward** | Original digital | Simulated print | Preview how images will look after printing |
| **Reverse** | Captured print | Restored original | Remove print artifacts, color correction |

## Available Architectures

| Architecture | Type | Parameters | Status |
|--------------|------|------------|--------|
| **Diffusion (Ours)** | Hybrid diffusion + Pix2Pix losses | ~860M (LoRA: ~1.6M) | **Novel method** - results pending |
| **Pix2Pix** | U-Net + GAN | ~34M | Baseline - validated (26.65 dB PSNR) |
| **NAFNet** | Encoder-decoder (no GAN) | ~17M | Baseline - stable training |

---

## Novel Method: Hybrid Diffusion-Pix2Pix

Our main contribution is a hybrid training framework that combines:

### Architecture

```
Input latent (4ch) + Noisy target latent (4ch) → U-Net → Denoised target latent (4ch) → VAE Decode → Output
```

- **Base Model**: Stable Diffusion 2.1 with modified 8-channel input
- **Fine-tuning**: LoRA adapters on attention layers (~0.2% of parameters)
- **Conditioning**: Marigold-style latent concatenation

### Hybrid Loss Function

Unlike standard diffusion training (noise prediction only) or standard Pix2Pix (L1 + GAN), we combine both paradigms:

```python
L_total = λ_noise × L_noise + λ_latent × L_latent_recon + λ_perceptual × L_perceptual
```

| Loss Component | Description | Inspiration |
|----------------|-------------|-------------|
| **L_noise** | Standard diffusion noise prediction loss | Diffusion models |
| **L_latent_recon** | Direct L1 between predicted clean latent and target | Pix2Pix L1 loss |
| **L_perceptual** | VGG feature matching in pixel space | Perceptual loss literature |

The key insight is recovering the predicted clean latent from the noise prediction, enabling direct supervision during diffusion training.

### Training

```bash
python src/train.py --config configs/train_config.yaml
```

Output: `outputs/`

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Batch size |
| `gradient_accumulation_steps` | 8 | Effective batch = batch_size × this |
| `learning_rate` | 5e-5 | Learning rate |
| `max_steps` | 5000 | Training steps |
| `lora_rank` | 32 | LoRA rank (higher = more capacity) |
| `lambda_noise` | 1.0 | Noise prediction loss weight |
| `lambda_latent_recon` | 1.0 | Latent reconstruction loss weight |
| `lambda_perceptual` | 0.1 | Perceptual loss weight |

**Memory Usage**: ~10-12GB VRAM with gradient checkpointing and 8-bit Adam

---

## Baseline: Pix2Pix

Standard conditional GAN baseline for comparison.

### Results (Validated)

| Metric | Pix2Pix | Best Traditional |
|--------|---------|------------------|
| PSNR | **26.65 dB** | 18.07 dB |
| SSIM | **0.7454** | 0.4244 |
| LPIPS | **0.2483** | 0.5445 |

### Quick Start (Windows)

```batch
train_pix2pix.bat           # Forward: Original → Captured (512×512)
train_pix2pix_reverse.bat   # Reverse: Captured → Original (512×512)
train_pix2pix_fullres.bat   # Forward: Full resolution
train_pix2pix_fullres_reverse.bat   # Reverse: Full resolution
```

### Manual Training

```bash
# Fixed resolution (512×512)
python src/train_pix2pix.py --config configs/pix2pix_config.yaml
python src/train_pix2pix.py --config configs/pix2pix_reverse_config.yaml

# Full resolution
python src/train_pix2pix.py --config configs/pix2pix_fullres_config.yaml
python src/train_pix2pix.py --config configs/pix2pix_fullres_reverse_config.yaml
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 (fixed) / 1 (fullres) | Batch size |
| `learning_rate` | 2e-4 | Learning rate |
| `max_steps` | 10000 | Training steps |
| `lambda_l1` | 100 | L1 loss weight |
| `lambda_perceptual` | 10 | Perceptual loss weight |
| `lambda_gan` | 1 | Adversarial loss weight |

**Training Time**: ~35 minutes for 10K steps on RTX 4070 Ti

---

## Baseline: NAFNet

State-of-the-art image restoration baseline without GAN training.

### Quick Start (Windows)

```batch
train_nafnet.bat            # Forward: Original → Captured
train_nafnet_reverse.bat    # Reverse: Captured → Original
```

### Manual Training

```bash
python src/train_nafnet.py --config configs/nafnet_config.yaml
python src/train_nafnet.py --config configs/nafnet_reverse_config.yaml
```

#### Variants

| Variant | Parameters | Use Case |
|---------|------------|----------|
| `lite` | ~2M | Fast experiments |
| `width32` | ~17M | **Recommended** |
| `width64` | ~67M | Best quality (24GB+ VRAM) |

---

## Requirements

- Python 3.8+
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8+ or 12.x

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat  # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt
```

## Data Setup

Place paired images in:
- `data/original/` - Original digital images (e.g., `prestineThumbnail00001.tif`)
- `data/captured/` - Printed+captured images (e.g., `warpThumbnail00001.tif`)

Images are paired by numeric suffix. Supported formats: `.png`, `.jpg`, `.tif`, `.tiff`, `.bmp`, `.webp`

```bash
# Validate data
python src/dataset.py --data_dir ./data
```

## Evaluation

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
| **PSNR** | Peak Signal-to-Noise Ratio | Higher |
| **SSIM** | Structural Similarity Index | Higher |
| **LPIPS** | Learned Perceptual Similarity | Lower |

### Traditional Baselines

| Baseline | Description |
|----------|-------------|
| Identity | No transformation (lower bound) |
| Reinhard | Mean/std matching in LAB space |
| Channel Regression | Per-channel linear regression |
| Histogram Matching | CDF-based histogram matching |

## Inference

### Pix2Pix

```bash
python src/inference_pix2pix.py \
  --checkpoint ./outputs_pix2pix/checkpoint-final \
  --input image.png \
  --output result.png
```

### NAFNet

```bash
python src/inference_nafnet.py \
  --checkpoint ./outputs_nafnet/checkpoint-final \
  --input image.png \
  --output result.png
```

For large images, use tiled processing:
```bash
python src/inference_nafnet.py \
  --checkpoint ./outputs_nafnet/checkpoint-final \
  --input large_image.png \
  --output result.png \
  --tiles --tile-size 512 --tile-overlap 64
```

## Output

Training saves to the output directory:
- `checkpoint-*/` - Model checkpoints
- `results_step_*.png` - Validation samples at each checkpoint
- `results_final.png` - Final results with loss curve

## References

- **Pix2Pix**: Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
- **NAFNet**: Chen et al. "Simple Baselines for Image Restoration" (ECCV 2022)
- **Marigold**: Ke et al. "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024)
- **Perceptual Loss**: Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (ECCV 2016)

## License

MIT
