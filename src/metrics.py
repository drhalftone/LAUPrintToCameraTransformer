"""
Image quality metrics for evaluation.

Implements PSNR, SSIM, and LPIPS for comparing predicted and target images.

References:
    PSNR: Standard signal processing metric, no specific citation needed.

    SSIM: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
          "Image quality assessment: from error visibility to structural similarity."
          IEEE Transactions on Image Processing, 13(4), 600-612.
          https://doi.org/10.1109/TIP.2003.819861

    LPIPS: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
           "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."
           IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
           https://arxiv.org/abs/1801.03924
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between predicted and target images.

    PSNR is defined as: 10 * log10(MAX^2 / MSE)

    This is a standard signal processing metric measuring reconstruction fidelity.
    Higher values indicate better quality (typically 20-40 dB for images).

    Args:
        pred: Predicted images, shape (B, C, H, W), range [-1, 1]
        target: Target images, same shape and range
        data_range: Range of the data (2.0 for [-1, 1] normalized images)

    Returns:
        Average PSNR in dB (higher is better)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Create a Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Create 2D window
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    window = window.expand(channels, 1, size, size)  # (C, 1, H, W)

    return window


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,
    K1: float = 0.01,
    K2: float = 0.03
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between predicted and target images.

    SSIM measures perceptual similarity by comparing luminance, contrast, and structure.
    Unlike PSNR, it accounts for the human visual system's sensitivity to structural
    information rather than just pixel-wise differences.

    Reference:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        "Image quality assessment: from error visibility to structural similarity."
        IEEE Transactions on Image Processing, 13(4), 600-612.

    Args:
        pred: Predicted images, shape (B, C, H, W), range [-1, 1]
        target: Target images, same shape and range
        window_size: Size of the Gaussian window (default 11, as in original paper)
        data_range: Range of the data (2.0 for [-1, 1] normalized images)
        K1, K2: Stability constants (defaults from original paper)

    Returns:
        Average SSIM (range 0-1, higher is better)
    """
    device = pred.device
    channels = pred.shape[1]

    # Stability constants
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Create Gaussian window
    window = _gaussian_window(window_size, 1.5, channels, device)

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_pred_target

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator

    return ssim_map.mean().item()


class ImageMetrics:
    """Unified interface for computing image quality metrics."""

    def __init__(self, device: Optional[torch.device] = None, lpips_net: str = 'alex'):
        """
        Args:
            device: torch device for computation
            lpips_net: LPIPS network backbone ('alex', 'vgg', 'squeeze')
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_net = lpips_net
        self._lpips_model = None

    def _get_lpips_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device)
                self._lpips_model.eval()
            except ImportError:
                raise ImportError(
                    "LPIPS package not installed. Install with: pip install lpips"
                )
        return self._lpips_model

    def psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between predicted and target tensors."""
        return psnr(pred.to(self.device), target.to(self.device))

    def ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Compute SSIM between predicted and target tensors."""
        return ssim(pred.to(self.device), target.to(self.device), window_size=window_size)

    def lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute LPIPS perceptual distance between predicted and target tensors.

        LPIPS uses deep features from a pretrained network (AlexNet by default)
        to measure perceptual similarity. It correlates better with human
        perception than PSNR/SSIM for many tasks.

        Reference:
            Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
            "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."
            CVPR 2018. https://arxiv.org/abs/1801.03924

        Returns:
            Average LPIPS distance (lower is better, 0 = identical)
        """
        lpips_model = self._get_lpips_model()
        pred = pred.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            # LPIPS expects images in [-1, 1] range, which matches our data
            distance = lpips_model(pred, target)

        return distance.mean().item()

    def compute_all(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all metrics at once.

        Args:
            pred: Predicted images, shape (B, C, H, W), range [-1, 1]
            target: Target images, same shape and range

        Returns:
            Dict with keys: 'psnr', 'ssim', 'lpips'
        """
        return {
            'psnr': self.psnr(pred, target),
            'ssim': self.ssim(pred, target),
            'lpips': self.lpips(pred, target)
        }


if __name__ == "__main__":
    # Quick test with random tensors
    print("Testing metrics module...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create test tensors (batch of 2, 3 channels, 64x64)
    pred = torch.randn(2, 3, 64, 64).to(device).clamp(-1, 1)
    target = torch.randn(2, 3, 64, 64).to(device).clamp(-1, 1)

    # Test individual functions
    print(f"PSNR: {psnr(pred, target):.2f} dB")
    print(f"SSIM: {ssim(pred, target):.4f}")

    # Test ImageMetrics class
    metrics = ImageMetrics(device=device)
    print(f"PSNR (class): {metrics.psnr(pred, target):.2f} dB")
    print(f"SSIM (class): {metrics.ssim(pred, target):.4f}")

    try:
        print(f"LPIPS: {metrics.lpips(pred, target):.4f}")
    except ImportError as e:
        print(f"LPIPS not available: {e}")

    # Test with identical images (should give perfect scores)
    print("\nWith identical images:")
    print(f"PSNR: {psnr(pred, pred):.2f} dB")
    print(f"SSIM: {ssim(pred, pred):.4f}")

    print("\nMetrics module test complete.")
