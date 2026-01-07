"""
Pix2Pix Training Script for Print-to-Camera Transformation.

Uses a U-Net generator with optional PatchGAN discriminator for
direct image-to-image translation.
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

from unet_model import UNet, PatchDiscriminator, VGGPerceptualLoss
from dataset import get_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Pix2PixTrainer:
    """Trainer for Pix2Pix-style image-to-image translation."""

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
        """Initialize generator and discriminator."""
        model_config = self.config['model']

        # Generator (U-Net)
        self.generator = UNet(
            in_channels=3,
            out_channels=3,
            features=model_config.get('features', 64)
        ).to(self.device)

        g_params = sum(p.numel() for p in self.generator.parameters())
        logger.info(f"Generator parameters: {g_params:,}")

        # Discriminator (optional)
        self.use_gan = model_config.get('use_gan', True)
        if self.use_gan:
            self.discriminator = PatchDiscriminator(in_channels=6).to(self.device)
            d_params = sum(p.numel() for p in self.discriminator.parameters())
            logger.info(f"Discriminator parameters: {d_params:,}")

        # Perceptual loss (optional)
        self.use_perceptual = model_config.get('use_perceptual', True)
        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)
            logger.info("Using VGG perceptual loss")

        # Optimizers
        lr = self.config['training']['learning_rate']
        self.optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

        if self.use_gan:
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=lr,
                betas=(0.5, 0.999)
            )

        # Learning rate schedulers
        max_steps = self.config['training']['max_steps']
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=max_steps, eta_min=lr * 0.01
        )
        if self.use_gan:
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d, T_max=max_steps, eta_min=lr * 0.01
            )

    def _init_data(self):
        """Initialize data loaders."""
        data_config = self.config['data']
        train_config = self.config['training']

        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=data_config['data_dir'],
            batch_size=train_config['batch_size'],
            image_size=data_config['image_size'],
            val_split=data_config['val_split'],
            num_workers=train_config.get('num_workers', 4),
            seed=train_config['seed'],
        )

        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

    def train_step(self, batch: dict) -> dict:
        """Perform a single training step."""
        input_images = batch['input_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)

        losses = {}
        use_amp = self.scaler is not None

        # Train Discriminator
        if self.use_gan:
            self.optimizer_d.zero_grad()

            with autocast('cuda', enabled=use_amp):
                # Generate fake images
                with torch.no_grad():
                    fake_images = self.generator(input_images)

                # Real loss
                pred_real = self.discriminator(input_images, target_images)
                loss_d_real = F.binary_cross_entropy_with_logits(
                    pred_real, torch.ones_like(pred_real)
                )

                # Fake loss
                pred_fake = self.discriminator(input_images, fake_images)
                loss_d_fake = F.binary_cross_entropy_with_logits(
                    pred_fake, torch.zeros_like(pred_fake)
                )

                loss_d = (loss_d_real + loss_d_fake) * 0.5

            if self.scaler:
                self.scaler.scale(loss_d).backward()
                self.scaler.step(self.optimizer_d)
            else:
                loss_d.backward()
                self.optimizer_d.step()

            losses['d_loss'] = loss_d.item()

        # Train Generator
        self.optimizer_g.zero_grad()

        with autocast('cuda', enabled=use_amp):
            fake_images = self.generator(input_images)

            # L1 loss (main reconstruction loss)
            loss_l1 = F.l1_loss(fake_images, target_images)
            losses['l1_loss'] = loss_l1.item()

            # Perceptual loss
            if self.use_perceptual:
                loss_perceptual = self.perceptual_loss(fake_images, target_images)
                losses['perceptual_loss'] = loss_perceptual.item()
            else:
                loss_perceptual = 0

            # GAN loss
            if self.use_gan:
                pred_fake = self.discriminator(input_images, fake_images)
                loss_gan = F.binary_cross_entropy_with_logits(
                    pred_fake, torch.ones_like(pred_fake)
                )
                losses['gan_loss'] = loss_gan.item()
            else:
                loss_gan = 0

            # Combined generator loss
            lambda_l1 = self.config['training'].get('lambda_l1', 100.0)
            lambda_perceptual = self.config['training'].get('lambda_perceptual', 10.0)
            lambda_gan = self.config['training'].get('lambda_gan', 1.0)

            loss_g = (
                lambda_l1 * loss_l1 +
                lambda_perceptual * loss_perceptual +
                lambda_gan * loss_gan
            )

        if self.scaler:
            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            loss_g.backward()
            self.optimizer_g.step()

        losses['g_loss'] = loss_g.item()

        return losses

    def train(self):
        """Main training loop."""
        self.generator.train()
        if self.use_gan:
            self.discriminator.train()

        max_steps = self.config['training']['max_steps']
        log_steps = self.config['training']['log_steps']
        save_steps = self.config['training']['save_steps']
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")

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

            # Update schedulers
            self.scheduler_g.step()
            if self.use_gan:
                self.scheduler_d.step()

            # Logging
            if self.global_step % log_steps == 0:
                avg_losses = {k: v / log_steps for k, v in running_losses.items()}
                lr = self.scheduler_g.get_last_lr()[0]

                log_str = f"Step {self.global_step}: " + ", ".join(
                    f"{k}={v:.4f}" for k, v in avg_losses.items()
                ) + f", lr={lr:.2e}"
                logger.info(log_str)

                progress_bar.set_postfix(**avg_losses, lr=lr)
                self.loss_history.append((self.global_step, avg_losses.get('l1_loss', 0)))

                running_losses = {}

            # Save checkpoint and results
            if self.global_step % save_steps == 0:
                self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
                self.save_results_image(output_dir / f"results_step_{self.global_step}.png")

        progress_bar.close()

        # Final save
        self.save_checkpoint(output_dir / "checkpoint-final")
        self.save_results_image(output_dir / "results_final.png")
        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.generator.state_dict(), path / "generator.pt")
        if self.use_gan:
            torch.save(self.discriminator.state_dict(), path / "discriminator.pt")

        torch.save({
            'global_step': self.global_step,
            'optimizer_g': self.optimizer_g.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'loss_history': self.loss_history,
        }, path / "training_state.pt")

        logger.info(f"Checkpoint saved to {path}")

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4) -> list:
        """Generate validation samples."""
        self.generator.eval()
        samples = []

        val_iter = iter(self.val_loader)
        for i in range(num_samples):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_img = batch['input_image'][:1].to(self.device)
            target_img = batch['target_image'][:1].to(self.device)

            output_img = self.generator(input_img)

            # Convert to numpy [0, 255]
            input_np = ((input_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            output_np = ((output_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            target_np = ((target_img[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)

            samples.append((input_np, output_np, target_np))

        self.generator.train()
        return samples

    def plot_loss_curve(self, figsize: Tuple[int, int] = (8, 4)) -> np.ndarray:
        """Plot loss curve and return as numpy array."""
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        if len(self.loss_history) > 0:
            steps, losses = zip(*self.loss_history)
            ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.7, label='L1 Loss')

            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Smoothed')
                ax.legend()

        ax.set_xlabel('Step')
        ax.set_ylabel('L1 Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return img.copy()

    def save_results_image(self, output_path: Path):
        """Save results image with samples and loss curve."""
        logger.info("Generating validation samples...")
        samples = self.generate_samples(num_samples=4)

        if not samples:
            logger.warning("No samples to save")
            return

        # Create comparison grid
        img_h, img_w = samples[0][0].shape[:2]
        grid_h = len(samples) * img_h
        grid_w = 3 * img_w
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        for i, (inp, out, tgt) in enumerate(samples):
            y = i * img_h
            grid[y:y+img_h, 0:img_w] = inp
            grid[y:y+img_h, img_w:2*img_w] = out
            grid[y:y+img_h, 2*img_w:3*img_w] = tgt

        # Create loss curve
        loss_curve = self.plot_loss_curve()

        # Combine
        total_height = grid_h + loss_curve.shape[0] + 60
        total_width = max(grid_w, loss_curve.shape[1])
        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # Add labels
        result[30:30+grid_h, :grid_w] = grid
        result[30+grid_h+30:30+grid_h+30+loss_curve.shape[0], :loss_curve.shape[1]] = loss_curve

        # Save
        Image.fromarray(result).save(output_path)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix for print-to-camera transformation")
    parser.add_argument("--config", type=str, default="configs/pix2pix_config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Pix2PixTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
