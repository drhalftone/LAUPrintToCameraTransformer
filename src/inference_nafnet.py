"""
NAFNet Inference Script for Print-to-Camera Transformation.

Runs inference on single images or batches at full resolution.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import time

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from nafnet_model import nafnet_width32, nafnet_width64, nafnet_lite, NAFNetLocal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NAFNetInference:
    """NAFNet inference wrapper."""

    def __init__(
        self,
        checkpoint_path: str,
        variant: str = "width32",
        device: str = "cuda",
        use_tiles: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ):
        """
        Initialize NAFNet inference.

        Args:
            checkpoint_path: Path to checkpoint directory or .pt file
            variant: Model variant ("lite", "width32", "width64")
            device: Device to run on ("cuda" or "cpu")
            use_tiles: Whether to use tiled processing for large images
            tile_size: Tile size for tiled processing
            tile_overlap: Overlap between tiles
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        if use_tiles:
            if variant == "width64":
                self.model = NAFNetLocal(
                    width=64, enc_blocks=(2, 2, 4, 8), middle_blocks=12, dec_blocks=(2, 2, 2, 2),
                    tile_size=tile_size, tile_overlap=tile_overlap
                )
            elif variant == "lite":
                self.model = NAFNetLocal(
                    width=32, enc_blocks=(1, 1, 1, 8), middle_blocks=1, dec_blocks=(1, 1, 1, 1),
                    tile_size=tile_size, tile_overlap=tile_overlap
                )
            else:
                self.model = NAFNetLocal(
                    width=32, enc_blocks=(1, 1, 1, 28), middle_blocks=1, dec_blocks=(1, 1, 1, 1),
                    tile_size=tile_size, tile_overlap=tile_overlap
                )
        else:
            if variant == "width64":
                self.model = nafnet_width64()
            elif variant == "lite":
                self.model = nafnet_lite()
            else:
                self.model = nafnet_width32()

        # Load weights
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            weights_path = checkpoint_path / "nafnet.pt"
        else:
            weights_path = checkpoint_path

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded NAFNet-{variant} from {weights_path}")

    @torch.no_grad()
    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Process a single image.

        Args:
            image: Input PIL Image (RGB)

        Returns:
            Output PIL Image (RGB)
        """
        # Convert to tensor
        img_tensor = transforms.ToTensor()(image.convert("RGB"))
        img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run inference
        output = self.model(img_tensor)

        # Convert back to PIL
        output = output[0].cpu().numpy()
        output = (output + 1) * 127.5  # Denormalize
        output = output.clip(0, 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)

        return Image.fromarray(output)

    def process_file(self, input_path: str, output_path: str):
        """Process a single image file."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Processing: {input_path}")
        start_time = time.time()

        image = Image.open(input_path).convert("RGB")
        w, h = image.size
        logger.info(f"  Input size: {w}x{h}")

        output = self.process_image(image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path, quality=95)

        elapsed = time.time() - start_time
        logger.info(f"  Saved to: {output_path} ({elapsed:.2f}s)")

    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return

        logger.info(f"Found {len(image_files)} images to process")
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            output_path = output_dir / f"{img_path.stem}_nafnet.png"
            self.process_file(str(img_path), str(output_path))


def main():
    parser = argparse.ArgumentParser(description="NAFNet inference for image restoration")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint directory or .pt file"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output image path or directory"
    )
    parser.add_argument(
        "--variant", type=str, default="width32",
        choices=["lite", "width32", "width64"],
        help="Model variant"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--tiles", action="store_true",
        help="Use tiled processing for large images"
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size for tiled processing"
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=64,
        help="Overlap between tiles"
    )
    args = parser.parse_args()

    inferencer = NAFNetInference(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        device=args.device,
        use_tiles=args.tiles,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        inferencer.process_directory(args.input, args.output)
    else:
        inferencer.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
