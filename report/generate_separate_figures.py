"""Generate separate original, scanned, and restored images for the paper."""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from unet_model import UNet


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load trained U-Net model from checkpoint."""
    model = UNet(in_channels=3, out_channels=3, features=64)
    state_dict = torch.load(checkpoint_path / "generator.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def process_image(model: UNet, image: Image.Image, device: torch.device, image_size: int = 512) -> Image.Image:
    """Process a single image through the model."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_np = ((output_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np)


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_dir = Path("../data")
    checkpoint_path = Path("../outputs_pix2pix_reverse/checkpoint-final")

    # Pick a good sample image (using index 5 for variety)
    sample_idx = "00005"
    original_path = data_dir / "original" / f"prestineThumbnail{sample_idx}.tif"
    captured_path = data_dir / "captured" / f"warpThumbnail{sample_idx}.tif"

    print(f"Loading original: {original_path}")
    print(f"Loading captured: {captured_path}")

    # Load images
    original_img = Image.open(original_path).convert("RGB")
    captured_img = Image.open(captured_path).convert("RGB")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Generate restored image (reverse model: captured -> original)
    print("Generating restored image...")
    restored_img = process_image(model, captured_img, device)

    # Apply same center crop transform used by model (resize then center crop to 512x512)
    from torchvision import transforms
    crop_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ])
    original_cropped = crop_transform(original_img)
    captured_cropped = crop_transform(captured_img)
    # restored_img is already 512x512 from model

    # Save as separate PNG files
    original_cropped.save("fig_original.png", quality=95)
    captured_cropped.save("fig_scanned.png", quality=95)
    restored_img.save("fig_restored.png", quality=95)

    print("Saved:")
    print("  - fig_original.png")
    print("  - fig_scanned.png")
    print("  - fig_restored.png")


if __name__ == "__main__":
    main()
