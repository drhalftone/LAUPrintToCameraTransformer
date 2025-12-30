"""
Image Alignment using Mutual Information Maximization.

Aligns captured images to original images by optimizing translation and scale
to maximize mutual information (minimize joint entropy).

Uses PyTorch for GPU-accelerated differentiable optimization.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MutualInformationLoss(nn.Module):
    """
    Differentiable Mutual Information loss using soft histograms.

    Maximizing MI = minimizing negative MI.
    """

    def __init__(self, num_bins: int = 64, sigma: float = 0.5, epsilon: float = 1e-10):
        """
        Args:
            num_bins: Number of histogram bins
            sigma: Gaussian kernel width for soft binning
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.epsilon = epsilon

        # Bin centers (0 to 1 range)
        self.register_buffer(
            'bin_centers',
            torch.linspace(0, 1, num_bins).view(1, 1, num_bins)
        )

    def _compute_soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft histogram using Gaussian kernels."""
        # x: (B, N) flattened image values in [0, 1]
        # Output: (B, num_bins) soft histogram

        x = x.unsqueeze(-1)  # (B, N, 1)

        # Gaussian kernel distance to each bin center
        diff = x - self.bin_centers  # (B, N, num_bins)
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        # Normalize weights per pixel
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)

        # Sum over pixels to get histogram
        hist = weights.sum(dim=1)  # (B, num_bins)

        # Normalize to probability distribution
        hist = hist / (hist.sum(dim=-1, keepdim=True) + self.epsilon)

        return hist

    def _compute_joint_histogram(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute joint soft histogram."""
        # x, y: (B, N) flattened image values in [0, 1]
        # Output: (B, num_bins, num_bins) joint histogram

        x = x.unsqueeze(-1)  # (B, N, 1)
        y = y.unsqueeze(-1)  # (B, N, 1)

        # Distance to bin centers
        diff_x = x - self.bin_centers  # (B, N, num_bins)
        diff_y = y - self.bin_centers  # (B, N, num_bins)

        # Gaussian weights
        weights_x = torch.exp(-0.5 * (diff_x / self.sigma) ** 2)
        weights_y = torch.exp(-0.5 * (diff_y / self.sigma) ** 2)

        # Normalize
        weights_x = weights_x / (weights_x.sum(dim=-1, keepdim=True) + self.epsilon)
        weights_y = weights_y / (weights_y.sum(dim=-1, keepdim=True) + self.epsilon)

        # Joint histogram: outer product and sum over pixels
        # (B, N, num_bins, 1) * (B, N, 1, num_bins) -> (B, N, num_bins, num_bins)
        joint = weights_x.unsqueeze(-1) * weights_y.unsqueeze(-2)
        joint = joint.sum(dim=1)  # (B, num_bins, num_bins)

        # Normalize
        joint = joint / (joint.sum(dim=(-1, -2), keepdim=True) + self.epsilon)

        return joint

    def _entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        p = p + self.epsilon
        return -torch.sum(p * torch.log(p), dim=-1)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Compute negative mutual information loss.

        Args:
            fixed: Fixed image (B, C, H, W) or (B, 1, H, W)
            moving: Moving image to align (same shape)

        Returns:
            Negative MI (to minimize)
        """
        # Convert to grayscale if needed
        if fixed.shape[1] == 3:
            fixed = 0.299 * fixed[:, 0] + 0.587 * fixed[:, 1] + 0.114 * fixed[:, 2]
            fixed = fixed.unsqueeze(1)
        if moving.shape[1] == 3:
            moving = 0.299 * moving[:, 0] + 0.587 * moving[:, 1] + 0.114 * moving[:, 2]
            moving = moving.unsqueeze(1)

        # Flatten spatial dimensions
        B = fixed.shape[0]
        fixed_flat = fixed.view(B, -1)  # (B, H*W)
        moving_flat = moving.view(B, -1)  # (B, H*W)

        # Clamp to [0, 1]
        fixed_flat = torch.clamp(fixed_flat, 0, 1)
        moving_flat = torch.clamp(moving_flat, 0, 1)

        # Compute histograms
        hist_fixed = self._compute_soft_histogram(fixed_flat)
        hist_moving = self._compute_soft_histogram(moving_flat)
        joint_hist = self._compute_joint_histogram(fixed_flat, moving_flat)

        # Compute entropies
        H_fixed = self._entropy(hist_fixed)  # (B,)
        H_moving = self._entropy(hist_moving)  # (B,)
        H_joint = self._entropy(joint_hist.view(B, -1))  # (B,)

        # Mutual Information = H(fixed) + H(moving) - H(joint)
        MI = H_fixed + H_moving - H_joint

        # Return negative MI (we want to maximize MI)
        return -MI.mean()


class ImageAligner(nn.Module):
    """
    Differentiable image alignment with translation and scale.
    """

    def __init__(
        self,
        init_tx: float = 0.0,
        init_ty: float = 0.0,
        init_scale: float = 1.0,
    ):
        super().__init__()

        # Learnable parameters
        self.tx = nn.Parameter(torch.tensor(init_tx))
        self.ty = nn.Parameter(torch.tensor(init_ty))
        self.log_scale = nn.Parameter(torch.tensor(np.log(init_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def get_transform_matrix(self) -> torch.Tensor:
        """Get 2x3 affine transform matrix."""
        s = self.scale

        # Affine matrix: scale then translate
        # [s, 0, tx]
        # [0, s, ty]
        matrix = torch.zeros(1, 2, 3, device=self.tx.device)
        matrix[0, 0, 0] = s
        matrix[0, 1, 1] = s
        matrix[0, 0, 2] = self.tx
        matrix[0, 1, 2] = self.ty

        return matrix

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transform to image.

        Args:
            image: (B, C, H, W) tensor

        Returns:
            Transformed image
        """
        B, C, H, W = image.shape

        # Get transform matrix and expand for batch
        matrix = self.get_transform_matrix()
        matrix = matrix.expand(B, -1, -1)

        # Create sampling grid
        grid = F.affine_grid(matrix, image.shape, align_corners=False)

        # Sample image
        output = F.grid_sample(
            image, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return output

    def get_params_dict(self) -> dict:
        """Get current parameters as dictionary."""
        return {
            'tx': self.tx.item(),
            'ty': self.ty.item(),
            'scale': self.scale.item(),
        }


def align_image_pair(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    num_steps: int = 200,
    lr: float = 0.01,
    num_bins: int = 64,
    scale_range: Tuple[float, float] = (0.97, 1.03),
    translation_range: float = 0.05,  # Fraction of image size
    verbose: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Align moving image to fixed image using MI maximization.

    Args:
        fixed: Fixed reference image (1, C, H, W)
        moving: Moving image to align (1, C, H, W)
        num_steps: Optimization steps
        lr: Learning rate
        num_bins: Histogram bins for MI
        scale_range: (min, max) allowed scale
        translation_range: Max translation as fraction of image size
        verbose: Print progress

    Returns:
        Tuple of (aligned_image, params_dict)
    """
    device = fixed.device

    # Initialize aligner
    aligner = ImageAligner().to(device)

    # MI loss
    mi_loss = MutualInformationLoss(num_bins=num_bins).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(aligner.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=lr * 0.1)

    best_loss = float('inf')
    best_params = None

    # Optimization loop
    iterator = tqdm(range(num_steps), desc="Aligning") if verbose else range(num_steps)

    for step in iterator:
        optimizer.zero_grad()

        # Apply transform
        aligned = aligner(moving)

        # Compute MI loss
        loss = mi_loss(fixed, aligned)

        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp parameters
        with torch.no_grad():
            # Clamp scale
            min_log_s = np.log(scale_range[0])
            max_log_s = np.log(scale_range[1])
            aligner.log_scale.clamp_(min_log_s, max_log_s)

            # Clamp translation (in normalized coords, so ±translation_range)
            aligner.tx.clamp_(-translation_range, translation_range)
            aligner.ty.clamp_(-translation_range, translation_range)

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = aligner.get_params_dict()

        if verbose and (step + 1) % 50 == 0:
            params = aligner.get_params_dict()
            tqdm.write(f"  Step {step+1}: MI={-loss.item():.4f}, "
                      f"tx={params['tx']*100:.2f}%, ty={params['ty']*100:.2f}%, "
                      f"scale={params['scale']:.4f}")

    # Apply best transform
    with torch.no_grad():
        aligner.tx.fill_(best_params['tx'])
        aligner.ty.fill_(best_params['ty'])
        aligner.log_scale.fill_(np.log(best_params['scale']))
        aligned = aligner(moving)

    # Convert translation to pixels for reporting
    H, W = fixed.shape[2], fixed.shape[3]
    best_params['tx_pixels'] = best_params['tx'] * W / 2
    best_params['ty_pixels'] = best_params['ty'] * H / 2

    return aligned, best_params


def load_image(path: Path, size: int = 512, device: str = 'cuda') -> torch.Tensor:
    """Load image as tensor."""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def save_image(tensor: torch.Tensor, path: Path):
    """Save tensor as image."""
    img = tensor[0].cpu().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(path)


def align_dataset(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    original_subdir: str = "original",
    captured_subdir: str = "captured",
    image_size: int = 512,
    num_steps: int = 200,
    device: str = 'cuda',
):
    """
    Align all image pairs in dataset.

    Args:
        data_dir: Root data directory
        output_dir: Output directory for aligned images (default: data_dir/captured_aligned)
        original_subdir: Subdirectory with original images
        captured_subdir: Subdirectory with captured images
        image_size: Size to process images at
        num_steps: Optimization steps per image
        device: Device to use
    """
    original_dir = data_dir / original_subdir
    captured_dir = data_dir / captured_subdir

    if output_dir is None:
        output_dir = data_dir / f"{captured_subdir}_aligned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    original_files = {f.stem.lower(): f for f in original_dir.iterdir()
                      if f.suffix.lower() in extensions}
    captured_files = {f.stem.lower(): f for f in captured_dir.iterdir()
                      if f.suffix.lower() in extensions}

    pairs = [(original_files[stem], captured_files[stem])
             for stem in original_files.keys() & captured_files.keys()]

    logger.info(f"Found {len(pairs)} image pairs to align")

    # Process each pair
    all_params = []

    for orig_path, cap_path in tqdm(pairs, desc="Aligning dataset"):
        # Load images
        fixed = load_image(orig_path, size=image_size, device=device)
        moving = load_image(cap_path, size=image_size, device=device)

        # Align
        aligned, params = align_image_pair(
            fixed, moving,
            num_steps=num_steps,
            verbose=False,
        )

        # Save aligned image
        out_path = output_dir / f"{cap_path.stem}_aligned.png"
        save_image(aligned, out_path)

        params['filename'] = cap_path.stem
        all_params.append(params)

        logger.info(f"  {cap_path.stem}: tx={params['tx_pixels']:.1f}px, "
                   f"ty={params['ty_pixels']:.1f}px, scale={params['scale']:.4f}")

    # Summary statistics
    tx_pixels = [p['tx_pixels'] for p in all_params]
    ty_pixels = [p['ty_pixels'] for p in all_params]
    scales = [p['scale'] for p in all_params]

    logger.info(f"\n=== Alignment Summary ===")
    logger.info(f"TX (pixels): mean={np.mean(tx_pixels):.2f}, std={np.std(tx_pixels):.2f}, "
               f"range=[{np.min(tx_pixels):.2f}, {np.max(tx_pixels):.2f}]")
    logger.info(f"TY (pixels): mean={np.mean(ty_pixels):.2f}, std={np.std(ty_pixels):.2f}, "
               f"range=[{np.min(ty_pixels):.2f}, {np.max(ty_pixels):.2f}]")
    logger.info(f"Scale: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}, "
               f"range=[{np.min(scales):.4f}, {np.max(scales):.4f}]")
    logger.info(f"\nAligned images saved to: {output_dir}")

    return all_params


def main():
    parser = argparse.ArgumentParser(description="Align images using Mutual Information")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Single pair alignment
    pair_parser = subparsers.add_parser("pair", help="Align a single image pair")
    pair_parser.add_argument("--fixed", type=str, required=True, help="Fixed reference image")
    pair_parser.add_argument("--moving", type=str, required=True, help="Moving image to align")
    pair_parser.add_argument("--output", type=str, required=True, help="Output aligned image")
    pair_parser.add_argument("--size", type=int, default=512, help="Processing size")
    pair_parser.add_argument("--steps", type=int, default=200, help="Optimization steps")
    pair_parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Dataset alignment
    dataset_parser = subparsers.add_parser("dataset", help="Align all pairs in dataset")
    dataset_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    dataset_parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    dataset_parser.add_argument("--size", type=int, default=512, help="Processing size")
    dataset_parser.add_argument("--steps", type=int, default=200, help="Optimization steps")
    dataset_parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    if args.command == "pair":
        # Single pair
        fixed = load_image(Path(args.fixed), size=args.size, device=args.device)
        moving = load_image(Path(args.moving), size=args.size, device=args.device)

        aligned, params = align_image_pair(
            fixed, moving,
            num_steps=args.steps,
            verbose=True,
        )

        save_image(aligned, Path(args.output))

        logger.info(f"\nAlignment results:")
        logger.info(f"  TX: {params['tx_pixels']:.2f} pixels")
        logger.info(f"  TY: {params['ty_pixels']:.2f} pixels")
        logger.info(f"  Scale: {params['scale']:.4f}")
        logger.info(f"Saved to: {args.output}")

    elif args.command == "dataset":
        # Full dataset
        output_dir = Path(args.output_dir) if args.output_dir else None
        align_dataset(
            data_dir=Path(args.data_dir),
            output_dir=output_dir,
            image_size=args.size,
            num_steps=args.steps,
            device=args.device,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
