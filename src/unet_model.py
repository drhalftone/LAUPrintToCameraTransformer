"""
U-Net Model for Image-to-Image Translation.

A simple but effective encoder-decoder architecture with skip connections
for paired image transformation tasks like print-to-camera.

Supports both fixed-resolution and full-resolution modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concat skip + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for image-to-image translation.

    Architecture:
        - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512 -> 1024)
        - Decoder: 4 upsampling blocks with skip connections
        - Output: Same resolution as input, 3 channels (RGB)

    Supports full-resolution images with automatic padding.
    """

    # 4 downsampling layers means we need dimensions divisible by 2^4 = 16
    PAD_MULTIPLE = 16

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, features)        # 64
        self.enc2 = DownBlock(features, features * 2)       # 128
        self.enc3 = DownBlock(features * 2, features * 4)   # 256
        self.enc4 = DownBlock(features * 4, features * 8)   # 512

        # Bottleneck
        self.bottleneck = DownBlock(features * 8, features * 16)  # 1024

        # Decoder
        self.dec4 = UpBlock(features * 16, features * 8)    # 512
        self.dec3 = UpBlock(features * 8, features * 4)     # 256
        self.dec2 = UpBlock(features * 4, features * 2)     # 128
        self.dec1 = UpBlock(features * 2, features)         # 64

        # Output
        self.out = nn.Conv2d(features, out_channels, 1)

    def pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad input to be divisible by PAD_MULTIPLE."""
        _, _, h, w = x.shape
        pad_h = (self.PAD_MULTIPLE - h % self.PAD_MULTIPLE) % self.PAD_MULTIPLE
        pad_w = (self.PAD_MULTIPLE - w % self.PAD_MULTIPLE) % self.PAD_MULTIPLE

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

    def forward(self, x: torch.Tensor, auto_pad: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            auto_pad: If True, automatically pad input to required multiple
                      and remove padding from output. Set False for fixed-size inputs.
        """
        # Automatic padding for arbitrary input sizes
        if auto_pad:
            x, pad = self.pad_to_multiple(x)
        else:
            pad = (0, 0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out(d1)

        # Remove padding
        if auto_pad:
            out = self.unpad(out, pad)

        return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for adversarial training.

    Classifies 70x70 overlapping patches as real or fake.
    """

    def __init__(self, in_channels: int = 6):  # Input + Target concatenated
        super().__init__()

        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, normalize=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Use features from multiple layers
        self.slice1 = nn.Sequential(*list(vgg)[:5])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[5:10]) # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[10:19]) # relu3_4

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        # Input is [-1, 1], convert to [0, 1] then normalize for VGG
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)

        pred1, target1 = self.slice1(pred), self.slice1(target)
        pred2, target2 = self.slice2(pred1), self.slice2(target1)
        pred3, target3 = self.slice3(pred2), self.slice3(target2)

        loss = (
            nn.functional.l1_loss(pred1, target1) +
            nn.functional.l1_loss(pred2, target2) +
            nn.functional.l1_loss(pred3, target3)
        )
        return loss


if __name__ == "__main__":
    # Test the model with various input sizes
    model = UNet()
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test different input sizes (including non-multiples of 16)
    test_sizes = [(512, 512), (256, 256), (720, 1280), (1080, 1920), (333, 555)]

    print("\nTesting various input sizes with auto_pad=True:")
    for h, w in test_sizes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x, auto_pad=True)
        print(f"  Input: {x.shape} -> Output: {y.shape}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    print("\nTest passed: Output shapes match input shapes for all sizes!")

    # Test discriminator
    print("\nTesting PatchDiscriminator:")
    disc = PatchDiscriminator()
    print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")

    for h, w in test_sizes[:3]:
        x = torch.randn(1, 3, h, w)
        y = torch.randn(1, 3, h, w)
        with torch.no_grad():
            d = disc(x, y)
        print(f"  Input: ({h}, {w}) -> Discriminator output: {d.shape}")
