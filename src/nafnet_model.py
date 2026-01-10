"""
NAFNet (Nonlinear Activation Free Network) for Image Restoration.

A simple, efficient architecture that achieves state-of-the-art results
without using nonlinear activation functions.

Reference: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022
https://arxiv.org/abs/2204.04676
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    """Simple Gate: splits channels and multiplies element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA) module."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """
    NAF Block: The core building block of NAFNet.

    Structure:
        LayerNorm -> Conv -> SimpleGate -> SCA -> Conv -> Skip
                 -> Conv -> SimpleGate -> Conv -> Skip (FFN path)
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()

        dw_channels = channels * dw_expand

        # Spatial mixing path
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=True)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, bias=True)

        # Channel mixing path (FFN)
        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1, bias=True)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1, bias=True)

        # Learnable scaling factors (beta)
        self.beta1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta1

        # Channel mixing (FFN)
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.gate2(y)
        y = self.conv5(y)
        x = x + y * self.beta2

        return x


class NAFNet(nn.Module):
    """
    NAFNet: Nonlinear Activation Free Network for Image Restoration.

    Encoder-decoder architecture with skip connections.
    Fully convolutional - handles any input size divisible by 2^num_encoders.

    Args:
        in_channels: Input image channels (default: 3 for RGB)
        out_channels: Output image channels (default: 3 for RGB)
        width: Base channel width (default: 32)
        num_encoders: Number of encoder/decoder levels (default: 4)
        enc_blocks: Number of NAFBlocks per encoder level
        middle_blocks: Number of NAFBlocks in the middle
        dec_blocks: Number of NAFBlocks per decoder level
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        num_encoders: int = 4,
        enc_blocks: Tuple[int, ...] = (1, 1, 1, 28),
        middle_blocks: int = 1,
        dec_blocks: Tuple[int, ...] = (1, 1, 1, 1),
    ):
        super().__init__()

        self.num_encoders = num_encoders
        self.pad_multiple = 2 ** num_encoders

        # Initial projection
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)

        # Encoders
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(num_encoders):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(enc_blocks[i])])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2, bias=True))
            chan *= 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])

        # Decoders
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(num_encoders):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=True),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(dec_blocks[i])])
            )

        # Output projection
        self.outro = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)

    def pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad input to be divisible by pad_multiple."""
        _, _, h, w = x.shape
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        return x, (pad_h, pad_w)

    def unpad(self, x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
        """Remove padding from output."""
        pad_h, pad_w = pad
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to multiple of 2^num_encoders
        x, pad = self.pad_to_multiple(x)

        # Initial projection
        x = self.intro(x)

        # Encoder path
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # Middle
        x = self.middle(x)

        # Decoder path with skip connections
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip  # Skip connection (addition, not concat)
            x = decoder(x)

        # Output projection
        x = self.outro(x)

        # Remove padding
        x = self.unpad(x, pad)

        return x


class NAFNetLocal(NAFNet):
    """
    NAFNet-Local: Processes image in overlapping tiles for very large images.

    Useful when full image doesn't fit in memory.
    """

    def __init__(self, *args, tile_size: int = 256, tile_overlap: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # If small enough, process directly
        if h <= self.tile_size and w <= self.tile_size:
            return super().forward(x)

        # Otherwise, process in tiles
        stride = self.tile_size - self.tile_overlap
        output = torch.zeros_like(x)
        count = torch.zeros_like(x)

        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # Get tile bounds
                i_end = min(i + self.tile_size, h)
                j_end = min(j + self.tile_size, w)
                i_start = max(0, i_end - self.tile_size)
                j_start = max(0, j_end - self.tile_size)

                # Process tile
                tile = x[:, :, i_start:i_end, j_start:j_end]
                tile_out = super().forward(tile)

                # Accumulate
                output[:, :, i_start:i_end, j_start:j_end] += tile_out
                count[:, :, i_start:i_end, j_start:j_end] += 1

        return output / count


# Lightweight NAFNet configurations for different use cases
def nafnet_width32(in_channels: int = 3, out_channels: int = 3) -> NAFNet:
    """NAFNet with width=32, good balance of speed and quality."""
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=32,
        num_encoders=4,
        enc_blocks=(1, 1, 1, 28),
        middle_blocks=1,
        dec_blocks=(1, 1, 1, 1),
    )


def nafnet_width64(in_channels: int = 3, out_channels: int = 3) -> NAFNet:
    """NAFNet with width=64, better quality, more VRAM."""
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=64,
        num_encoders=4,
        enc_blocks=(2, 2, 4, 8),
        middle_blocks=12,
        dec_blocks=(2, 2, 2, 2),
    )


def nafnet_lite(in_channels: int = 3, out_channels: int = 3) -> NAFNet:
    """Lightweight NAFNet for faster training and inference."""
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=32,
        num_encoders=4,
        enc_blocks=(1, 1, 1, 8),
        middle_blocks=1,
        dec_blocks=(1, 1, 1, 1),
    )


if __name__ == "__main__":
    # Test the model with various input sizes
    model = nafnet_width32()
    print(f"NAFNet-width32 parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test different input sizes
    test_sizes = [(256, 256), (512, 512), (720, 1280), (1080, 1920), (333, 555)]

    for h, w in test_sizes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test lite version
    model_lite = nafnet_lite()
    print(f"\nNAFNet-lite parameters: {sum(p.numel() for p in model_lite.parameters()):,}")

    # Test width64 version
    model_64 = nafnet_width64()
    print(f"NAFNet-width64 parameters: {sum(p.numel() for p in model_64.parameters()):,}")
