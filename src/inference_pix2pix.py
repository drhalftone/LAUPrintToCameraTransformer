"""
Inference script for Pix2Pix Print-to-Camera model.

Supports both fixed-resolution and full-resolution processing.
"""

import argparse
from pathlib import Path
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from unet_model import UNet


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load trained U-Net model from checkpoint."""
    model = UNet(in_channels=3, out_channels=3, features=64)

    state_dict = torch.load(checkpoint_path / "generator.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def process_image(
    model: UNet,
    image_path: Path,
    device: torch.device,
    image_size: int = 512,
    full_resolution: bool = False,
) -> Image.Image:
    """Process a single image through the model."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    if full_resolution:
        # Full resolution mode: no resizing, just normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Run inference (model handles padding automatically)
        with torch.no_grad():
            output_tensor = model(input_tensor, auto_pad=True)

        # Convert back to image
        output_np = ((output_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_np)

    else:
        # Fixed resolution mode: resize and crop
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output_tensor = model(input_tensor, auto_pad=False)

        # Convert back to image
        output_np = ((output_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_np)

        # Resize back to original size if needed
        if output_img.size != original_size:
            output_img = output_img.resize(original_size, Image.Resampling.BILINEAR)

    return output_img


def main():
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument("--image_size", type=int, default=512, help="Processing size (ignored if --full_resolution)")
    parser.add_argument("--full_resolution", action="store_true", help="Process at full resolution (no resize)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    if not (checkpoint_path / "generator.pt").exists():
        print(f"Error: generator.pt not found in {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    if args.full_resolution:
        print("Mode: FULL RESOLUTION (no resizing)")
    else:
        print(f"Mode: Fixed resolution ({args.image_size}x{args.image_size})")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Process single image or directory
    if input_path.is_file():
        print(f"Processing {input_path}")
        start_time = time.time()
        result = process_image(model, input_path, device, args.image_size, args.full_resolution)
        elapsed = time.time() - start_time
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"Saved to {output_path} ({elapsed:.2f}s)")

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        print(f"Processing {len(images)} images...")
        for img_path in images:
            start_time = time.time()
            result = process_image(model, img_path, device, args.image_size, args.full_resolution)
            elapsed = time.time() - start_time
            result.save(output_path / img_path.name)
            print(f"  {img_path.name} ({elapsed:.2f}s)")

        print(f"Results saved to {output_path}")

    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
