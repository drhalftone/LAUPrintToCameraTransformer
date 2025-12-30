"""
Print-to-Camera Transformer Training Script.

Fine-tunes Stable Diffusion using LoRA with Marigold-style conditioning
to predict printed+captured appearance from original images.
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrintToCameraTrainer:
    """Trainer for Print-to-Camera image transformation model."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set seed
        torch.manual_seed(config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training']['seed'])

        # Initialize models
        self._init_models()

        # Initialize optimizer and scheduler
        self._init_optimizer()

        # Initialize dataloaders
        self._init_data()

        # Mixed precision
        self.scaler = GradScaler() if config['training']['mixed_precision'] == 'fp16' else None

        # Training state
        self.global_step = 0

    def _init_models(self):
        """Initialize VAE, U-Net, and apply LoRA."""
        model_id = self.config['model']['pretrained_model']
        logger.info(f"Loading pretrained model: {model_id}")

        # Load VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

        # Modify U-Net input conv for 8-channel input (4 input latent + 4 noisy target latent)
        # Skip if already 8 channels (e.g., Marigold)
        if self.unet.config.in_channels == 4:
            self._modify_unet_input_channels()
        else:
            logger.info(f"U-Net already has {self.unet.config.in_channels} input channels, skipping modification")

        # Apply LoRA if configured
        if self.config['model']['use_lora']:
            self._apply_lora()

        self.unet.to(self.device)

        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            self.unet.enable_gradient_checkpointing()

        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # We don't need text encoder for this task (unconditional)
        # But we need dummy embeddings for the U-Net
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        # Create null text embedding (unconditional)
        with torch.no_grad():
            null_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            self.null_text_embedding = self.text_encoder(null_tokens)[0]

    def _modify_unet_input_channels(self):
        """Modify U-Net to accept 8-channel input (concatenated latents)."""
        old_conv = self.unet.conv_in
        new_conv = torch.nn.Conv2d(
            in_channels=8,  # 4 (input latent) + 4 (noisy target latent)
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        # Initialize new conv: copy weights for first 4 channels, zero for rest
        with torch.no_grad():
            new_conv.weight[:, :4, :, :] = old_conv.weight
            new_conv.weight[:, 4:, :, :] = 0
            new_conv.bias = old_conv.bias

        self.unet.conv_in = new_conv
        self.unet.config['in_channels'] = 8
        logger.info("Modified U-Net input conv to accept 8 channels")

    def _apply_lora(self):
        """Apply LoRA to U-Net attention layers."""
        lora_config = LoraConfig(
            r=self.config['model']['lora_rank'],
            lora_alpha=self.config['model']['lora_alpha'],
            lora_dropout=self.config['model']['lora_dropout'],
            target_modules=self.config['model']['lora_target_modules'],
        )
        self.unet = get_peft_model(self.unet, lora_config)
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        logger.info(f"LoRA applied. Trainable params: {trainable_params:,} / {total_params:,} "
                    f"({100 * trainable_params / total_params:.2f}%)")

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]

        # Use 8-bit Adam if available and configured
        if self.config['training']['use_8bit_adam']:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config['training']['learning_rate'],
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning("bitsandbytes not available, using regular AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config['training']['learning_rate'],
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )

        # Calculate total training steps
        self.max_steps = self.config['training']['max_steps']

        self.lr_scheduler = get_scheduler(
            self.config['training']['lr_scheduler'],
            optimizer=self.optimizer,
            num_warmup_steps=self.config['training']['lr_warmup_steps'],
            num_training_steps=self.max_steps,
        )

    def _init_data(self):
        """Initialize dataloaders."""
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            image_size=self.config['data']['image_size'],
            val_split=self.config['data']['val_split'],
            num_workers=4,
            seed=self.config['training']['seed'],
        )
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def train_step(self, batch: dict) -> float:
        """Perform a single training step."""
        self.unet.train()

        input_images = batch['input_image'].to(self.device)
        target_images = batch['target_image'].to(self.device)
        batch_size = input_images.shape[0]

        # Encode images to latent space
        input_latents = self.encode_images(input_images)
        target_latents = self.encode_images(target_images)

        # Sample noise and timesteps
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # Add noise to target latents
        noisy_target_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Concatenate input latents with noisy target latents (channel-wise)
        model_input = torch.cat([input_latents, noisy_target_latents], dim=1)

        # Expand null text embedding for batch
        encoder_hidden_states = self.null_text_embedding.expand(batch_size, -1, -1)

        # Forward pass
        use_amp = self.config['training']['mixed_precision'] == 'fp16'
        with autocast(enabled=use_amp):
            noise_pred = self.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            # Compute loss (predict noise)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

        return loss

    def train(self):
        """Main training loop."""
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        log_steps = self.config['training']['log_steps']
        save_steps = self.config['training']['save_steps']

        logger.info("Starting training...")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config['training']['batch_size'] * gradient_accumulation_steps}")

        progress_bar = tqdm(total=self.max_steps, desc="Training")
        running_loss = 0.0
        accumulated_loss = 0.0

        # Infinite data iterator
        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader(self.train_loader)

        while self.global_step < self.max_steps:
            batch = next(data_iter)

            # Training step
            loss = self.train_step(batch)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (after accumulation)
            if (self.global_step + 1) % gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                running_loss += accumulated_loss
                accumulated_loss = 0.0

            self.global_step += 1
            progress_bar.update(1)

            # Logging
            if self.global_step % log_steps == 0:
                avg_loss = running_loss / log_steps
                lr = self.lr_scheduler.get_last_lr()[0]
                logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                progress_bar.set_postfix(loss=avg_loss, lr=lr)
                running_loss = 0.0

            # Save checkpoint
            if self.global_step % save_steps == 0:
                self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

        progress_bar.close()
        self.save_checkpoint(output_dir / "checkpoint-final")
        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        if self.config['model']['use_lora']:
            self.unet.save_pretrained(path / "unet_lora")
        else:
            self.unet.save_pretrained(path / "unet")

        # Save config
        with open(path / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

        # Save optimizer and scheduler state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
        }, path / "training_state.pt")

        logger.info(f"Checkpoint saved to {path}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Print-to-Camera Transformer")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps from config")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="Override LoRA rank from config")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.lora_rank:
        config['model']['lora_rank'] = args.lora_rank

    # Create trainer and start training
    trainer = PrintToCameraTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
