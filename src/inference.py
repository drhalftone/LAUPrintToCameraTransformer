"""
Print-to-Camera Transformer Inference Script.

Generates predictions of how images will look after printing and capture.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import yaml

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrintToCameraPredictor:
    """Predictor for Print-to-Camera image transformation."""

    def __init__(
        self,
        checkpoint_path: str,
        base_model: str = "prs-eth/marigold-v1-0",
        device: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to trained checkpoint directory
            base_model: Base Stable Diffusion model ID
            device: Device to run on (auto-detected if None)
        """
        # Auto-detect device (cuda -> mps -> cpu)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        elif self.device.type == "mps":
            logger.info("Using MPS device (Apple Silicon)")
        else:
            logger.info(f"Using device: {self.device}")

        self.checkpoint_path = Path(checkpoint_path)
        self.base_model = base_model

        # Load config
        config_path = self.checkpoint_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None
            logger.warning("No config.yaml found in checkpoint, using defaults")

        # Load models
        self._load_models()

        # Image transforms
        self.image_size = 512
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize([-1], [2]),  # [-1, 1] -> [0, 1]
        ])

    def _load_models(self):
        """Load VAE, U-Net with LoRA, and scheduler."""
        logger.info(f"Loading base model: {self.base_model}")

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(self.base_model, subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval()

        # Load U-Net and modify input channels if needed
        self.unet = UNet2DConditionModel.from_pretrained(self.base_model, subfolder="unet")
        if self.unet.config.in_channels == 4:
            self._modify_unet_input_channels()
        else:
            logger.info(f"U-Net already has {self.unet.config.in_channels} input channels")

        # Load LoRA weights
        lora_path = self.checkpoint_path / "unet_lora"
        if lora_path.exists():
            logger.info(f"Loading LoRA weights from {lora_path}")
            self.unet = PeftModel.from_pretrained(self.unet, lora_path)
        else:
            # Try loading full U-Net weights
            unet_path = self.checkpoint_path / "unet"
            if unet_path.exists():
                logger.info(f"Loading full U-Net weights from {unet_path}")
                self.unet = UNet2DConditionModel.from_pretrained(unet_path)
            else:
                raise FileNotFoundError(f"No model weights found in {self.checkpoint_path}")

        self.unet.to(self.device)
        self.unet.eval()

        # Use DDIM scheduler for faster inference
        self.scheduler = DDIMScheduler.from_pretrained(self.base_model, subfolder="scheduler")

        # Load text encoder for null embeddings
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.base_model, subfolder="text_encoder")
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        # Create null text embedding
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
        """Modify U-Net to accept 8-channel input."""
        old_conv = self.unet.conv_in
        new_conv = torch.nn.Conv2d(
            in_channels=8,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :4, :, :] = old_conv.weight
            new_conv.weight[:, 4:, :, :] = 0
            new_conv.bias = old_conv.bias

        self.unet.conv_in = new_conv
        self.unet.config['in_channels'] = 8

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        with torch.no_grad():
            latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent).sample
        return image

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        num_inference_steps: int = 50,
        num_ensemble: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Predict printed+captured appearance from original image.

        Args:
            image: Input PIL Image
            num_inference_steps: Number of DDIM denoising steps
            num_ensemble: Number of predictions to ensemble (averaged)
            generator: Optional random generator for reproducibility

        Returns:
            Predicted PIL Image
        """
        # Preprocess image
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Encode input to latent space
        input_latent = self.encode_image(input_tensor)

        # Prepare for ensembling
        predictions = []

        for _ in range(num_ensemble):
            # Start from random noise
            latent_shape = input_latent.shape
            target_latent = torch.randn(latent_shape, device=self.device, generator=generator)

            # Set up scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

            # Denoising loop
            for t in self.scheduler.timesteps:
                # Concatenate input latent with noisy target latent
                model_input = torch.cat([input_latent, target_latent], dim=1)

                # Predict noise
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=self.null_text_embedding,
                    return_dict=False,
                )[0]

                # DDIM step
                target_latent = self.scheduler.step(noise_pred, t, target_latent).prev_sample

            predictions.append(target_latent)

        # Ensemble predictions (average)
        if num_ensemble > 1:
            target_latent = torch.stack(predictions).mean(dim=0)
        else:
            target_latent = predictions[0]

        # Decode to image
        output_tensor = self.decode_latent(target_latent)

        # Convert to PIL
        output_tensor = self.inverse_transform(output_tensor)
        output_tensor = output_tensor.clamp(0, 1)
        output_array = (output_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        output_image = Image.fromarray(output_array)

        return output_image

    def predict_batch(
        self,
        images: List[Image.Image],
        num_inference_steps: int = 50,
        show_progress: bool = True,
    ) -> List[Image.Image]:
        """
        Predict for a batch of images.

        Args:
            images: List of input PIL Images
            num_inference_steps: Number of DDIM steps
            show_progress: Show progress bar

        Returns:
            List of predicted PIL Images
        """
        results = []
        iterator = tqdm(images, desc="Predicting") if show_progress else images

        for img in iterator:
            result = self.predict(img, num_inference_steps=num_inference_steps)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Print-to-Camera Prediction")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--base_model", type=str, default="prs-eth/marigold-v1-0",
                        help="Base model (must match what was used for training)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image path or directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path or directory")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of DDIM inference steps")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of predictions to ensemble")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Initialize predictor
    predictor = PrintToCameraPredictor(args.checkpoint, base_model=args.base_model)

    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=predictor.device).manual_seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image prediction
        logger.info(f"Processing single image: {input_path}")
        image = Image.open(input_path)
        result = predictor.predict(
            image,
            num_inference_steps=args.num_steps,
            num_ensemble=args.ensemble,
            generator=generator,
        )
        result.save(output_path)
        logger.info(f"Saved result to {output_path}")

    elif input_path.is_dir():
        # Batch prediction
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        logger.info(f"Found {len(image_files)} images in {input_path}")

        for img_path in tqdm(image_files, desc="Processing"):
            image = Image.open(img_path)
            result = predictor.predict(
                image,
                num_inference_steps=args.num_steps,
                num_ensemble=args.ensemble,
                generator=generator,
            )

            out_file = output_path / f"{img_path.stem}_predicted.png"
            result.save(out_file)

        logger.info(f"Saved {len(image_files)} results to {output_path}")

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
