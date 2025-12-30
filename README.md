# Print-to-Camera Transformer

Train a diffusion model to predict how images will look after being printed on a digital label printer and captured by a machine vision system.

Uses the [Marigold](https://github.com/prs-eth/Marigold) architecture (fine-tuned Stable Diffusion with image conditioning) with LoRA for memory-efficient training.

## Requirements

- Python 3.8+
- NVIDIA GPU with 8-12GB VRAM
- CUDA 11.8+

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Place your paired images in:
- `data/original/` - Original digital images
- `data/captured/` - Corresponding printed+captured images

Images must have matching filenames (e.g., `image001.png` in both folders).

### Validate Data

```bash
python scripts/organize_data.py analyze --data_dir ./data
```

## Training

```bash
python src/train.py --config configs/train_config.yaml
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 1 | Batch size (1-2 for 8-12GB VRAM) |
| `--learning_rate` | 5e-5 | Learning rate |
| `--max_steps` | 20000 | Training steps |
| `--lora_rank` | 32 | LoRA rank (higher = more capacity) |

### Memory Usage

With default settings (LoRA + gradient checkpointing + fp16):
- ~8-10GB VRAM for batch size 1
- ~10-12GB VRAM for batch size 2

## Inference

```bash
python src/inference.py \
  --checkpoint ./outputs/checkpoint-final \
  --input image.png \
  --output result.png
```

### Batch Processing

```bash
python src/inference.py \
  --checkpoint ./outputs/checkpoint-final \
  --input ./input_folder \
  --output ./output_folder
```

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_steps` | 50 | DDIM denoising steps (fewer = faster) |
| `--ensemble` | 1 | Average N predictions (higher = smoother) |
| `--seed` | None | Random seed for reproducibility |

## Architecture

Based on [Marigold](https://arxiv.org/abs/2312.02145):
1. Encode input image to latent space (VAE)
2. Concatenate with noisy target latent (8 channels)
3. U-Net denoises toward target
4. Decode output latent to image

LoRA fine-tuning trains ~0.1% of parameters for memory efficiency.

## License

MIT
