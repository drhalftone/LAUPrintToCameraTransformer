"""
NAFNet Training Script for Print-to-Camera Transformation.

Trains NAFNet at full image resolution without cropping/resizing.
Uses L1 + optional perceptual loss (no GAN for stability).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nafnet_model import NAFNet, nafnet_width32, nafnet_width64, nafnet_lite
from unet_model import VGGPerceptualLoss
from dataset_fullres import get_fullres_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NAFNetTrainer:
    """Trainer for NAFNet image restoration."""

    def __init__(self, config: dict):
        self.config = config

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        # Set seed
        torch.manual_seed(config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training']['seed'])

        # Initialize models
        self._init_models()

        # Initialize data
        self._init_data()

        # Training state
        self.global_step = 0
        self.loss_history = []

        # Mixed precision
        self.scaler = GradScaler('cuda') if config['training'].get('mixed_precision') == 'fp16' else None

    def _init_models(self):
        """Initialize NAFNet model."""
        model_config = self.config['model']

        # Select model variant
        variant = model_config.get('variant', 'width32')
        if variant == 'width64':
            self.model = nafnet_width64()
        elif variant == 'lite':
            self.model = nafnet_lite()
        else:
            self.model = nafnet_width32()

        self.model = self.model.to(self.device)

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"NAFNet-{variant} parameters: {params:,}")

        # Perceptual loss (optional)
        self.use_perceptual = model_config.get('use_perceptual', True)
        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)
            logger.info("Using VGG perceptual loss")

        # Optimizer
        lr = self.config['training']['learning_rate']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.9),
            weight_decay=0.0
        )

        # Learning rate scheduler
        max_steps = self.config['training']['max_steps']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.01
        )

    def _init_data(self):
        """Initialize data loaders."""
        data_config = self.config['data']
        train_config = self.config['training']

        self.train_loader, self.val_loader = get_fullres_dataloaders(
            data_dir=data_config['data_dir'],
            batch_size=train_config['batch_size'],
            val_split=data_config['val_split'],
            num_workers=train_config.get('num_workers', 4),
            seed=train_config['seed'],
            original_subdir=data_config.get('original_subdir', 'original'),
            captured_subdir=data_config.get('captured_subdir', 'captured'),
            max_size=data_config.get('max_size', None),
            same_size_batching=train_config.get('same_size_batching', True),
        )

        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        # Log size distribution
        sizes = self.train_loader.dataset.sizes
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        logger.info(f"Image width range: {min(widths)} - {max(widths)}")
        logger.info(f"Image height range: {min(heights)} - {max(heights)}")

    def train_step(self, batch: dict) -> dict:
        """Perform a single training step."""
        input_images = batch['input_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)

        losses = {}
        use_amp = self.scaler is not None

        self.optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            # Forward pass
            output_images = self.model(input_images)

            # Handle padding if present (for variable-size batching)
            if 'pad_info' in batch:
                # Compute loss only on non-padded regions
                loss_l1 = 0
                for i, (h, w, pad_h, pad_w) in enumerate(batch['pad_info']):
                    loss_l1 += F.l1_loss(
                        output_images[i:i+1, :, :h, :w],
                        target_images[i:i+1, :, :h, :w]
                    )
                loss_l1 /= len(batch['pad_info'])
            else:
                loss_l1 = F.l1_loss(output_images, target_images)

            losses['l1_loss'] = loss_l1.item()

            # Perceptual loss
            if self.use_perceptual:
                if 'pad_info' in batch:
                    # Use cropped versions for perceptual loss
                    loss_perceptual = 0
                    for i, (h, w, _, _) in enumerate(batch['pad_info']):
                        loss_perceptual += self.perceptual_loss(
                            output_images[i:i+1, :, :h, :w],
                            target_images[i:i+1, :, :h, :w]
                        )
                    loss_perceptual /= len(batch['pad_info'])
                else:
                    loss_perceptual = self.perceptual_loss(output_images, target_images)
                losses['perceptual_loss'] = loss_perceptual.item()
            else:
                loss_perceptual = 0

            # Combined loss
            lambda_l1 = self.config['training'].get('lambda_l1', 1.0)
            lambda_perceptual = self.config['training'].get('lambda_perceptual', 0.1)

            total_loss = lambda_l1 * loss_l1 + lambda_perceptual * loss_perceptual

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        losses['total_loss'] = total_loss.item()

        return losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and return metrics."""
        self.model.eval()

        val_losses = {'l1_loss': 0, 'perceptual_loss': 0, 'count': 0}

        for batch in self.val_loader:
            input_images = batch['input_image'].to(self.device)
            target_images = batch['target_image'].to(self.device)

            output_images = self.model(input_images)

            # Handle padding
            if 'pad_info' in batch:
                for i, (h, w, _, _) in enumerate(batch['pad_info']):
                    val_losses['l1_loss'] += F.l1_loss(
                        output_images[i:i+1, :, :h, :w],
                        target_images[i:i+1, :, :h, :w]
                    ).item()
                    if self.use_perceptual:
                        val_losses['perceptual_loss'] += self.perceptual_loss(
                            output_images[i:i+1, :, :h, :w],
                            target_images[i:i+1, :, :h, :w]
                        ).item()
                    val_losses['count'] += 1
            else:
                val_losses['l1_loss'] += F.l1_loss(output_images, target_images).item() * input_images.size(0)
                if self.use_perceptual:
                    val_losses['perceptual_loss'] += self.perceptual_loss(output_images, target_images).item() * input_images.size(0)
                val_losses['count'] += input_images.size(0)

        self.model.train()

        count = val_losses['count']
        return {
            'val_l1_loss': val_losses['l1_loss'] / count,
            'val_perceptual_loss': val_losses['perceptual_loss'] / count if self.use_perceptual else 0,
        }

    def train(self):
        """Main training loop."""
        self.model.train()

        max_steps = self.config['training']['max_steps']
        log_steps = self.config['training']['log_steps']
        save_steps = self.config['training']['save_steps']
        val_steps = self.config['training'].get('val_steps', save_steps)
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting NAFNet training...")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  Output dir: {output_dir}")

        progress_bar = tqdm(total=max_steps, desc="Training")
        running_losses = {}

        # Infinite data iterator
        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch

        data_iter = iter(infinite_loader(self.train_loader))

        while self.global_step < max_steps:
            batch = next(data_iter)
            losses = self.train_step(batch)

            # Accumulate losses
            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0) + v

            self.global_step += 1
            progress_bar.update(1)

            # Update scheduler
            self.scheduler.step()

            # Logging
            if self.global_step % log_steps == 0:
                avg_losses = {k: v / log_steps for k, v in running_losses.items()}
                lr = self.scheduler.get_last_lr()[0]

                log_str = f"Step {self.global_step}: " + ", ".join(
                    f"{k}={v:.4f}" for k, v in avg_losses.items()
                ) + f", lr={lr:.2e}"
                logger.info(log_str)

                progress_bar.set_postfix(**avg_losses, lr=lr)
                self.loss_history.append((self.global_step, avg_losses.get('l1_loss', 0)))

                running_losses = {}

            # Validation
            if self.global_step % val_steps == 0:
                val_metrics = self.validate()
                logger.info(f"Validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            # Save checkpoint and results
            if self.global_step % save_steps == 0:
                self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
                self.save_results_image(output_dir / f"results_step_{self.global_step}.png")

        progress_bar.close()

        # Final save
        self.save_checkpoint(output_dir / "checkpoint-final")
        self.save_results_image(output_dir / "results_final.png")
        self.save_loss_plot(output_dir / "loss_curve.png")
        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "nafnet.pt")

        torch.save({
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config,
        }, path / "training_state.pt")

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path / "nafnet.pt", map_location=self.device))

        state = torch.load(path / "training_state.pt", map_location=self.device)
        self.global_step = state['global_step']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.loss_history = state.get('loss_history', [])

        logger.info(f"Checkpoint loaded from {path}, step {self.global_step}")

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4) -> list:
        """Generate validation samples."""
        self.model.eval()
        samples = []

        val_iter = iter(self.val_loader)
        for i in range(num_samples):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_img = batch['input_image'][:1].to(self.device)
            target_img = batch['target_image'][:1].to(self.device)

            output_img = self.model(input_img)

            # Handle padding if present
            if 'pad_info' in batch:
                h, w, _, _ = batch['pad_info'][0]
                input_img = input_img[:, :, :h, :w]
                output_img = output_img[:, :, :h, :w]
                target_img = target_img[:, :, :h, :w]

            # Convert to numpy [0, 255]
            input_np = ((input_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            output_np = ((output_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            target_np = ((target_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)

            samples.append((input_np, output_np, target_np))

        self.model.train()
        return samples

    def plot_loss_curve(self, figsize: Tuple[int, int] = (10, 5)) -> np.ndarray:
        """Plot loss curve and return as numpy array."""
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        if len(self.loss_history) > 0:
            steps, losses = zip(*self.loss_history)
            ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.5, label='L1 Loss')

            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Smoothed')
                ax.legend()

        ax.set_xlabel('Step')
        ax.set_ylabel('L1 Loss')
        ax.set_title('NAFNet Training Loss')
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return img.copy()

    def save_loss_plot(self, output_path: Path):
        """Save loss plot to file."""
        loss_img = self.plot_loss_curve()
        Image.fromarray(loss_img).save(output_path)
        logger.info(f"Loss plot saved to {output_path}")

    def save_results_image(self, output_path: Path):
        """Save results image with samples and loss curve."""
        logger.info("Generating validation samples...")
        samples = self.generate_samples(num_samples=4)

        if not samples:
            logger.warning("No samples to save")
            return

        # Scale down large images for visualization
        max_vis_size = 512
        scaled_samples = []
        for inp, out, tgt in samples:
            h, w = inp.shape[:2]
            if max(h, w) > max_vis_size:
                scale = max_vis_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                inp = np.array(Image.fromarray(inp).resize((new_w, new_h), Image.LANCZOS))
                out = np.array(Image.fromarray(out).resize((new_w, new_h), Image.LANCZOS))
                tgt = np.array(Image.fromarray(tgt).resize((new_w, new_h), Image.LANCZOS))
            scaled_samples.append((inp, out, tgt))

        # Create comparison grid
        img_h = max(s[0].shape[0] for s in scaled_samples)
        img_w = max(s[0].shape[1] for s in scaled_samples)

        grid_h = len(scaled_samples) * (img_h + 10) + 40  # Extra for title
        grid_w = 3 * (img_w + 10)
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        for i, (inp, out, tgt) in enumerate(scaled_samples):
            y = 40 + i * (img_h + 10)
            h, w = inp.shape[:2]

            # Center images if smaller than max
            x_offset = (img_w - w) // 2

            grid[y:y+h, x_offset:x_offset+w] = inp
            grid[y:y+h, img_w + 10 + x_offset:img_w + 10 + x_offset+w] = out
            grid[y:y+h, 2*(img_w + 10) + x_offset:2*(img_w + 10) + x_offset+w] = tgt

        # Create loss curve
        loss_curve = self.plot_loss_curve()

        # Combine
        total_height = grid_h + loss_curve.shape[0] + 20
        total_width = max(grid_w, loss_curve.shape[1])
        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # Add grid
        result[:grid_h, :grid_w] = grid

        # Add loss curve
        result[grid_h + 20:grid_h + 20 + loss_curve.shape[0], :loss_curve.shape[1]] = loss_curve

        # Save
        Image.fromarray(result).save(output_path)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NAFNet for print-to-camera transformation")
    parser.add_argument("--config", type=str, default="configs/nafnet_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = NAFNetTrainer(config)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    trainer.train()


if __name__ == "__main__":
    main()
