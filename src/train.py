"""
Print-to-Camera Transformer Training Script.

Fine-tunes Stable Diffusion using LoRA with Marigold-style conditioning
to predict printed+captured appearance from original images.

Enhanced with Pix2Pix-style losses:
- Direct latent reconstruction loss (L1)
- Perceptual loss (VGG features)
- Support for v-prediction
- Fewer inference steps
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import get_dataloaders
from unet_model import VGGPerceptualLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrintToCameraTrainer:
    """Trainer for Print-to-Camera image transformation model."""

    def __init__(self, config: dict):
        self.config = config

        # Auto-detect device (cuda -> mps -> cpu)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

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

        # Initialize perceptual loss if enabled
        loss_config = config.get('loss', {})
        self.use_perceptual = loss_config.get('use_perceptual', True)
        self.use_latent_recon = loss_config.get('use_latent_recon', True)

        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)
            logger.info("Using VGG perceptual loss")

        # Loss weights
        self.lambda_noise = loss_config.get('lambda_noise', 1.0)
        self.lambda_latent_recon = loss_config.get('lambda_latent_recon', 1.0)
        self.lambda_perceptual = loss_config.get('lambda_perceptual', 0.1)
        self.perceptual_every = loss_config.get('perceptual_every', 4)  # Compute perceptual every N steps

        logger.info(f"Loss weights: noise={self.lambda_noise}, latent_recon={self.lambda_latent_recon}, perceptual={self.lambda_perceptual}")

        # Mixed precision
        self.scaler = GradScaler('cuda') if config['training']['mixed_precision'] == 'fp16' else None

        # Training state
        self.global_step = 0
        self.loss_history = []  # List of (step, loss) tuples

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

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to images."""
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
        return images

    @torch.no_grad()
    def predict_single(
        self,
        input_image: torch.Tensor,
        num_steps: int = 20,
        deterministic: bool = True,
        seed: int = None,
    ) -> torch.Tensor:
        """
        Run inference on a single input image.

        Args:
            input_image: Input tensor (B, C, H, W) in [-1, 1]
            num_steps: Number of denoising steps (fewer = faster, 10-20 recommended)
            deterministic: If True, use DDIM with eta=0 for reproducible output
            seed: Random seed for noise (only used if deterministic=False or for initial noise)

        Returns:
            Output image tensor (B, C, H, W) in [0, 1]
        """
        from diffusers import DDIMScheduler

        self.unet.eval()

        # Setup DDIM scheduler for inference
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        ddim_scheduler.set_timesteps(num_steps, device=self.device)

        # Encode input
        input_latents = self.encode_images(input_image)
        batch_size = input_latents.shape[0]

        # Start from random noise (with optional seed for reproducibility)
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latents = torch.randn(input_latents.shape, generator=generator, device=self.device, dtype=input_latents.dtype)
        else:
            latents = torch.randn_like(input_latents)

        # Text embedding
        encoder_hidden_states = self.null_text_embedding.expand(batch_size, -1, -1)

        # DDIM eta: 0 = deterministic, 1 = stochastic (like DDPM)
        eta = 0.0 if deterministic else 1.0

        # Denoising loop
        for t in ddim_scheduler.timesteps:
            model_input = torch.cat([input_latents, latents], dim=1)
            timestep = t.expand(batch_size).to(self.device)

            noise_pred = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            latents = ddim_scheduler.step(
                noise_pred, t, latents,
                eta=eta,
                return_dict=False
            )[0]

        # Decode to image
        output_image = self.decode_latents(latents)
        self.unet.train()
        return output_image

    @torch.no_grad()
    def generate_validation_samples(
        self,
        num_samples: int = 4,
        num_steps: int = 20,
        deterministic: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate validation samples (input, prediction, ground truth).

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            deterministic: Use deterministic inference for reproducible results
        """
        samples = []
        val_iter = iter(self.val_loader)

        for i in range(min(num_samples, len(self.val_loader.dataset))):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_img = batch['input_image'][:1].to(self.device)
            target_img = batch['target_image'][:1].to(self.device)

            # Generate prediction (use consistent seed for reproducible validation)
            pred_img = self.predict_single(
                input_img,
                num_steps=num_steps,
                deterministic=deterministic,
                seed=42 + i  # Different but reproducible seed per sample
            )

            # Convert to numpy (H, W, C) format
            input_np = (input_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            pred_np = pred_img[0].cpu().permute(1, 2, 0).numpy()
            target_np = (target_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

            samples.append((input_np, pred_np, target_np))

        return samples

    def create_comparison_grid(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """Create a grid of input | prediction | ground truth comparisons."""
        num_samples = len(samples)
        if num_samples == 0:
            return np.zeros((100, 300, 3), dtype=np.uint8)

        h, w = samples[0][0].shape[:2]
        grid_width = w * 3
        grid_height = h * num_samples

        grid = np.zeros((grid_height, grid_width, 3), dtype=np.float32)

        for i, (inp, pred, target) in enumerate(samples):
            y_start = i * h
            grid[y_start:y_start+h, 0:w] = inp
            grid[y_start:y_start+h, w:2*w] = pred
            grid[y_start:y_start+h, 2*w:3*w] = target

        return (grid * 255).astype(np.uint8)

    def plot_loss_curve(self, figsize: Tuple[int, int] = (8, 4)) -> np.ndarray:
        """Plot loss curve and return as numpy array."""
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        if len(self.loss_history) > 0:
            steps, losses = zip(*self.loss_history)
            ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.7)

            # Add smoothed line
            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label='Smoothed')
                ax.legend()

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        # Convert figure to numpy array
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
        plt.close(fig)

        return img.copy()

    def save_results_image(self, output_path: Path, num_samples: int = 4):
        """Save compiled results image with samples and loss curve."""
        logger.info("Generating validation samples...")
        samples = self.generate_validation_samples(num_samples=num_samples, num_steps=20)

        # Create comparison grid
        comparison_grid = self.create_comparison_grid(samples)

        # Create loss curve
        loss_curve = self.plot_loss_curve()

        # Get dimensions
        comp_h, comp_w = comparison_grid.shape[:2]
        loss_h, loss_w = loss_curve.shape[:2]

        # Create labels
        label_height = 30
        img_size = samples[0][0].shape[1] if samples else 100

        # Create final image
        total_width = max(comp_w, loss_w)
        total_height = label_height + comp_h + 20 + loss_h + 40

        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        # Create figure for proper text rendering
        fig, ax = plt.subplots(figsize=(total_width/100, total_height/100), dpi=100)
        ax.set_xlim(0, total_width)
        ax.set_ylim(total_height, 0)
        ax.axis('off')

        # Add column labels
        ax.text(img_size * 0.5, 20, 'Input', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(img_size * 1.5, 20, 'Prediction', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(img_size * 2.5, 20, 'Ground Truth', ha='center', va='center', fontsize=12, fontweight='bold')

        # Add comparison grid
        ax.imshow(comparison_grid, extent=[0, comp_w, label_height + comp_h, label_height])

        # Add loss curve
        loss_y_start = label_height + comp_h + 20
        ax.imshow(loss_curve, extent=[0, loss_w, loss_y_start + loss_h, loss_y_start])

        # Add step info
        ax.text(total_width/2, total_height - 10, f'Step: {self.global_step}',
                ha='center', va='center', fontsize=10)

        # Save
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

        logger.info(f"Results image saved to {output_path}")

    def _get_predicted_clean_latent(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the predicted clean latent from noise prediction.

        For epsilon-prediction: x0 = (xt - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        For v-prediction: x0 = sqrt(alpha_t) * xt - sqrt(1-alpha_t) * v
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        # Get alpha values for each timestep in batch
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        prediction_type = self.noise_scheduler.config.prediction_type

        if prediction_type == "epsilon":
            # x0 = (xt - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            pred_clean = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        elif prediction_type == "v_prediction":
            # x0 = sqrt(alpha_t) * xt - sqrt(1-alpha_t) * v
            pred_clean = sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * noise_pred
        else:
            # For "sample" prediction type, the model directly predicts x0
            pred_clean = noise_pred

        return pred_clean

    def train_step(self, batch: dict) -> dict:
        """Perform a single training step with Pix2Pix-style losses."""
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
        losses = {}

        with autocast('cuda', enabled=use_amp):
            noise_pred = self.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            # === Loss 1: Noise prediction loss (original diffusion loss) ===
            prediction_type = self.noise_scheduler.config.prediction_type
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                # v = sqrt(alpha_t) * eps - sqrt(1-alpha_t) * x0
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
                alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                target = torch.sqrt(alpha_t) * noise - torch.sqrt(1 - alpha_t) * target_latents
            else:
                target = target_latents  # sample prediction

            loss_noise = F.mse_loss(noise_pred, target, reduction="mean")
            losses['noise'] = loss_noise.item()

            # === Loss 2: Direct latent reconstruction loss (Pix2Pix-style) ===
            if self.use_latent_recon:
                pred_clean_latent = self._get_predicted_clean_latent(
                    noisy_target_latents, noise_pred, timesteps
                )
                loss_latent_recon = F.l1_loss(pred_clean_latent, target_latents)
                losses['latent_recon'] = loss_latent_recon.item()
            else:
                loss_latent_recon = 0

            # === Loss 3: Perceptual loss (decoded images) ===
            # Only compute periodically to save memory/compute
            if self.use_perceptual and (self.global_step % self.perceptual_every == 0):
                # Decode predicted clean latent to image space
                with torch.no_grad():
                    pred_clean_latent_detached = pred_clean_latent.detach() if self.use_latent_recon else \
                        self._get_predicted_clean_latent(noisy_target_latents, noise_pred, timesteps)

                # Decode with gradient for backprop
                pred_images = self.decode_latents_with_grad(pred_clean_latent)

                # Target images are in [-1, 1], convert to [0, 1] for VGG
                target_images_01 = (target_images + 1) / 2

                # VGG expects images in [-1, 1], so convert pred back
                pred_images_norm = pred_images * 2 - 1
                target_images_norm = target_images_01 * 2 - 1

                loss_perceptual = self.perceptual_loss(pred_images_norm, target_images_norm)
                losses['perceptual'] = loss_perceptual.item()
            else:
                loss_perceptual = 0

            # === Combined loss ===
            total_loss = (
                self.lambda_noise * loss_noise +
                self.lambda_latent_recon * loss_latent_recon +
                self.lambda_perceptual * loss_perceptual
            )
            losses['total'] = total_loss.item()

        return total_loss, losses

    def decode_latents_with_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images WITH gradient (for perceptual loss backprop)."""
        latents = latents / self.vae.config.scaling_factor
        # Use VAE decoder but allow gradients through
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def train(self):
        """Main training loop."""
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        log_steps = self.config['training']['log_steps']
        save_steps = self.config['training']['save_steps']

        logger.info("Starting training with Pix2Pix-style losses...")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config['training']['batch_size'] * gradient_accumulation_steps}")
        logger.info(f"  Latent reconstruction loss: {self.use_latent_recon}")
        logger.info(f"  Perceptual loss: {self.use_perceptual} (every {self.perceptual_every} steps)")

        progress_bar = tqdm(total=self.max_steps, desc="Training")
        running_losses = {}
        accumulated_loss = 0.0

        # Infinite data iterator
        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader(self.train_loader)

        while self.global_step < self.max_steps:
            batch = next(data_iter)

            # Training step - now returns (loss_tensor, losses_dict)
            loss, losses = self.train_step(batch)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            accumulated_loss += losses.get('total', loss.item())

            # Accumulate individual losses for logging
            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0) + v

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (after accumulation)
            if (self.global_step + 1) % gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                accumulated_loss = 0.0

            self.global_step += 1
            progress_bar.update(1)

            # Logging
            if self.global_step % log_steps == 0:
                avg_losses = {k: v / log_steps for k, v in running_losses.items()}
                lr = self.lr_scheduler.get_last_lr()[0]

                # Build log string
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
                logger.info(f"Step {self.global_step}: {loss_str}, lr={lr:.2e}")
                progress_bar.set_postfix(**avg_losses, lr=lr)

                # Record loss history (use total loss)
                self.loss_history.append((self.global_step, avg_losses.get('total', avg_losses.get('noise', 0))))

                running_losses = {}

            # Save checkpoint and results image
            if self.global_step % save_steps == 0:
                self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
                self.save_results_image(output_dir / f"results_step_{self.global_step}.png")

        progress_bar.close()
        self.save_checkpoint(output_dir / "checkpoint-final")
        self.save_results_image(output_dir / "results_final.png")
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
