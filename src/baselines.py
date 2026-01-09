"""
Baseline transformation methods for comparison.

Implements traditional image transformation methods as baselines:
- Identity (no transformation)
- Histogram Matching
- Linear Color Transfer (mean/std matching in LAB space, following Reinhard et al. 2001)
- Channel-wise Linear Regression

References:
- Reinhard et al. (2001) "Color Transfer between Images", IEEE CG&A
- Gonzalez & Woods "Digital Image Processing" for histogram matching
"""

import torch
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod


# =============================================================================
# Color Space Conversion Utilities
# =============================================================================

def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to CIE LAB color space.

    Args:
        rgb: RGB tensor, shape (..., 3, H, W), range [0, 1]

    Returns:
        LAB tensor, same shape, L in [0, 100], a/b in roughly [-128, 128]
    """
    # Ensure we're working with float
    rgb = rgb.float()

    # Split channels - rgb is (..., 3, H, W)
    r, g, b = rgb[..., 0:1, :, :], rgb[..., 1:2, :, :], rgb[..., 2:3, :, :]

    # RGB to XYZ (sRGB with D65 illuminant)
    # First apply gamma correction (inverse sRGB companding)
    def srgb_to_linear(c):
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # Linear RGB to XYZ matrix (sRGB, D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # XYZ to LAB
    # Reference white D65
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, t ** (1/3), t / (3 * delta**2) + 4/29)

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)

    return torch.cat([L, a, b_ch], dim=-3)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert CIE LAB tensor to RGB color space.

    Args:
        lab: LAB tensor, shape (..., 3, H, W), L in [0, 100], a/b in roughly [-128, 128]

    Returns:
        RGB tensor, same shape, range [0, 1]
    """
    lab = lab.float()

    L, a, b_ch = lab[..., 0:1, :, :], lab[..., 1:2, :, :], lab[..., 2:3, :, :]

    # LAB to XYZ
    xn, yn, zn = 0.95047, 1.0, 1.08883

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_ch / 200

    delta = 6.0 / 29.0

    def f_inv(t):
        return torch.where(t > delta, t ** 3, 3 * delta**2 * (t - 4/29))

    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)

    # XYZ to linear RGB
    r_lin = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b_lin = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    # Linear RGB to sRGB (gamma correction)
    def linear_to_srgb(c):
        return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)

    r = linear_to_srgb(r_lin)
    g = linear_to_srgb(g_lin)
    b = linear_to_srgb(b_lin)

    rgb = torch.cat([r, g, b], dim=-3)
    return rgb.clamp(0, 1)


# =============================================================================
# Baseline Methods
# =============================================================================

class BaselineMethod(ABC):
    """Base class for baseline transformation methods."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Fit method parameters on training data (if applicable).

        Args:
            inputs: Input images, shape (N, C, H, W), range [-1, 1]
            targets: Target images, same shape and range
        """
        pass

    @abstractmethod
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        Transform images.

        Args:
            images: Input images, shape (B, C, H, W), range [-1, 1]

        Returns:
            Transformed images, same shape and range
        """
        raise NotImplementedError


class IdentityBaseline(BaselineMethod):
    """
    Identity transformation - no change.

    Lower bound baseline: how well does input match target without any
    transformation? Useful for understanding the difficulty of the task.
    """

    def __init__(self):
        super().__init__("Identity")

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        return images.clone()


class LinearColorTransferBaseline(BaselineMethod):
    """
    Linear color transfer using mean and std matching in LAB color space.

    Follows Reinhard et al. (2001) "Color Transfer between Images":
    For each LAB channel:
        output = (input - input_mean) * (target_std / input_std) + target_mean

    Using LAB space decorrelates the color channels, making the transfer
    more perceptually accurate than operating in RGB directly.

    Reference:
        Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001).
        Color transfer between images. IEEE Computer Graphics and Applications.
    """

    def __init__(self):
        super().__init__("Reinhard Color Transfer")
        self.target_mean: Optional[torch.Tensor] = None  # Shape (3,) in LAB
        self.target_std: Optional[torch.Tensor] = None   # Shape (3,) in LAB

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute global mean and std from targets in LAB space."""
        # Convert from [-1, 1] to [0, 1] for LAB conversion
        targets_01 = (targets + 1) / 2

        # Convert to LAB
        targets_lab = rgb_to_lab(targets_01)

        # Compute mean and std across all samples and spatial dimensions
        # targets_lab shape: (N, 3, H, W) with L, a, b channels
        self.target_mean = targets_lab.mean(dim=(0, 2, 3))  # (3,)
        self.target_std = targets_lab.std(dim=(0, 2, 3))    # (3,)

        # Avoid division by zero
        self.target_std = torch.clamp(self.target_std, min=1e-6)

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("Must call fit() before transform()")

        device = images.device
        target_mean = self.target_mean.to(device)
        target_std = self.target_std.to(device)

        # Convert from [-1, 1] to [0, 1]
        images_01 = (images + 1) / 2

        # Convert to LAB
        images_lab = rgb_to_lab(images_01)

        # Compute input statistics per image in LAB space
        # Shape: (B, 3)
        input_mean = images_lab.mean(dim=(2, 3))
        input_std = images_lab.std(dim=(2, 3))
        input_std = torch.clamp(input_std, min=1e-6)

        # Reshape for broadcasting: (B, 3, 1, 1)
        input_mean = input_mean.unsqueeze(-1).unsqueeze(-1)
        input_std = input_std.unsqueeze(-1).unsqueeze(-1)
        target_mean = target_mean.view(1, -1, 1, 1)
        target_std = target_std.view(1, -1, 1, 1)

        # Apply linear transfer in LAB space
        output_lab = (images_lab - input_mean) * (target_std / input_std) + target_mean

        # Convert back to RGB [0, 1]
        output_rgb = lab_to_rgb(output_lab)

        # Convert back to [-1, 1]
        output = output_rgb * 2 - 1

        return output.clamp(-1, 1)


class ChannelRegressionBaseline(BaselineMethod):
    """
    Per-channel linear regression from input to target.

    Fits a linear model y = ax + b for each channel separately,
    minimizing MSE on training data using least squares.
    """

    def __init__(self):
        super().__init__("Channel Regression")
        self.slopes: Optional[torch.Tensor] = None      # Shape (3,)
        self.intercepts: Optional[torch.Tensor] = None  # Shape (3,)

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Fit linear regression per channel using least squares."""
        # Flatten spatial dimensions: (N, C, H, W) -> (C, N*H*W)
        N, C, H, W = inputs.shape
        inputs_flat = inputs.permute(1, 0, 2, 3).reshape(C, -1)   # (C, N*H*W)
        targets_flat = targets.permute(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)

        slopes = []
        intercepts = []

        for c in range(C):
            x = inputs_flat[c]
            y = targets_flat[c]

            # Least squares: slope = cov(x,y) / var(x), intercept = mean(y) - slope * mean(x)
            x_mean = x.mean()
            y_mean = y.mean()

            cov_xy = ((x - x_mean) * (y - y_mean)).mean()
            var_x = ((x - x_mean) ** 2).mean()

            # Avoid division by zero
            if var_x < 1e-8:
                slope = torch.tensor(1.0)
                intercept = y_mean - x_mean
            else:
                slope = cov_xy / var_x
                intercept = y_mean - slope * x_mean

            slopes.append(slope.item())
            intercepts.append(intercept.item())

        self.slopes = torch.tensor(slopes)
        self.intercepts = torch.tensor(intercepts)

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.slopes is None or self.intercepts is None:
            raise RuntimeError("Must call fit() before transform()")

        device = images.device
        slopes = self.slopes.to(device).view(1, -1, 1, 1)
        intercepts = self.intercepts.to(device).view(1, -1, 1, 1)

        output = images * slopes + intercepts
        return output.clamp(-1, 1)


class HistogramMatchingBaseline(BaselineMethod):
    """
    Match histogram of input to average target histogram.

    Computes average histogram from training targets, then matches
    each input image's histogram to this reference using CDF matching.
    """

    def __init__(self, bins: int = 256):
        super().__init__("Histogram Matching")
        self.bins = bins
        self.reference_cdfs: Optional[List[np.ndarray]] = None  # Per-channel reference CDFs
        self.bin_edges: Optional[np.ndarray] = None

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute average target histogram per channel."""
        # Convert to numpy and scale to [0, 255]
        targets_np = ((targets.cpu().numpy() + 1) * 127.5).astype(np.float32)

        # targets_np shape: (N, C, H, W)
        N, C, H, W = targets_np.shape

        self.bin_edges = np.linspace(0, 256, self.bins + 1)
        self.reference_cdfs = []

        for c in range(C):
            # Flatten all target pixels for this channel
            channel_pixels = targets_np[:, c, :, :].flatten()

            # Compute histogram
            hist, _ = np.histogram(channel_pixels, bins=self.bin_edges, density=True)

            # Compute CDF
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]  # Normalize to [0, 1]

            self.reference_cdfs.append(cdf)

    def _match_histogram_channel(self, source: np.ndarray, reference_cdf: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference CDF for a single channel."""
        # source shape: (H, W) in [0, 255]

        # Compute source histogram and CDF
        source_hist, _ = np.histogram(source.flatten(), bins=self.bin_edges, density=True)
        source_cdf = np.cumsum(source_hist)
        source_cdf = source_cdf / source_cdf[-1]

        # Build mapping: for each source bin, find closest reference bin
        # Using interpolation on inverse CDF
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # Map source values through CDF matching
        # 1. Get CDF value for each pixel
        source_flat = source.flatten()
        source_bin_indices = np.clip(
            np.digitize(source_flat, self.bin_edges[1:-1]),
            0, self.bins - 1
        )
        source_cdf_values = source_cdf[source_bin_indices]

        # 2. Find corresponding value in reference CDF
        # Use linear interpolation: find where reference_cdf equals source_cdf_value
        matched_values = np.interp(source_cdf_values, reference_cdf, bin_centers)

        return matched_values.reshape(source.shape)

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.reference_cdfs is None:
            raise RuntimeError("Must call fit() before transform()")

        device = images.device

        # Convert to numpy [0, 255]
        images_np = ((images.cpu().numpy() + 1) * 127.5).astype(np.float32)

        B, C, H, W = images_np.shape
        output_np = np.zeros_like(images_np)

        for b in range(B):
            for c in range(C):
                output_np[b, c] = self._match_histogram_channel(
                    images_np[b, c],
                    self.reference_cdfs[c]
                )

        # Convert back to tensor [-1, 1]
        output = torch.from_numpy((output_np / 127.5) - 1).float().to(device)
        return output.clamp(-1, 1)


def get_all_baselines() -> List[BaselineMethod]:
    """Return list of all baseline methods."""
    return [
        IdentityBaseline(),
        LinearColorTransferBaseline(),
        ChannelRegressionBaseline(),
        HistogramMatchingBaseline(),
    ]


if __name__ == "__main__":
    # Quick test with random tensors
    print("Testing baselines module...")

    # Create test data
    torch.manual_seed(42)
    train_inputs = torch.randn(10, 3, 64, 64).clamp(-1, 1)
    train_targets = torch.randn(10, 3, 64, 64).clamp(-1, 1) * 0.8 + 0.1  # Different distribution

    test_inputs = torch.randn(2, 3, 64, 64).clamp(-1, 1)

    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test inputs shape: {test_inputs.shape}")

    # Test each baseline
    for baseline in get_all_baselines():
        print(f"\n{baseline.name}:")
        baseline.fit(train_inputs, train_targets)
        output = baseline.transform(test_inputs)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    print("\nBaselines module test complete.")
