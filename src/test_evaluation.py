"""
Test script for evaluation modules.

Run with: python src/test_evaluation.py
"""

import torch
import numpy as np

print("=" * 60)
print("Testing Evaluation Modules")
print("=" * 60)

# =============================================================================
# Test 1: LAB Color Space Conversion
# =============================================================================
print("\n[1] Testing LAB color space conversion...")

from baselines import rgb_to_lab, lab_to_rgb

torch.manual_seed(42)
rgb = torch.rand(2, 3, 64, 64)  # Random RGB in [0, 1]

lab = rgb_to_lab(rgb)
rgb_recovered = lab_to_rgb(lab)

error = (rgb - rgb_recovered).abs().max().item()
print(f"    Roundtrip max error: {error:.6f}")
print(f"    LAB L range: [{lab[:,0].min():.1f}, {lab[:,0].max():.1f}] (expected: 0-100)")
print(f"    LAB a range: [{lab[:,1].min():.1f}, {lab[:,1].max():.1f}] (expected: ~-128 to 128)")
print(f"    LAB b range: [{lab[:,2].min():.1f}, {lab[:,2].max():.1f}] (expected: ~-128 to 128)")
assert error < 0.01, f"LAB roundtrip error too high: {error}"
print("    PASS")

# =============================================================================
# Test 2: Metrics with Known Values
# =============================================================================
print("\n[2] Testing metrics with known values...")

from metrics import psnr, ssim, ImageMetrics

# Identical images should have perfect scores
img = torch.rand(1, 3, 64, 64) * 2 - 1  # Range [-1, 1]

psnr_identical = psnr(img, img)
ssim_identical = ssim(img, img)

print(f"    PSNR (identical): {psnr_identical:.2f} dB (expected: inf)")
print(f"    SSIM (identical): {ssim_identical:.4f} (expected: 1.0)")
assert psnr_identical == float('inf'), f"PSNR of identical images should be inf, got {psnr_identical}"
assert ssim_identical > 0.999, f"SSIM of identical images should be ~1.0, got {ssim_identical}"
print("    PASS")

# Different images should have lower scores
img2 = torch.rand(1, 3, 64, 64) * 2 - 1
psnr_diff = psnr(img, img2)
ssim_diff = ssim(img, img2)

print(f"    PSNR (different): {psnr_diff:.2f} dB (expected: ~5-15)")
print(f"    SSIM (different): {ssim_diff:.4f} (expected: <1.0)")
assert psnr_diff < 20, f"PSNR of random images should be low, got {psnr_diff}"
assert ssim_diff < 0.5, f"SSIM of random images should be low, got {ssim_diff}"
print("    PASS")

# =============================================================================
# Test 3: Baseline Methods
# =============================================================================
print("\n[3] Testing baseline methods...")

from baselines import get_all_baselines

# Create synthetic training data
train_inputs = torch.rand(10, 3, 64, 64) * 2 - 1   # Range [-1, 1]
train_targets = torch.rand(10, 3, 64, 64) * 2 - 1
test_inputs = torch.rand(2, 3, 64, 64) * 2 - 1

for baseline in get_all_baselines():
    print(f"    {baseline.name}:")

    # Fit
    baseline.fit(train_inputs, train_targets)

    # Transform
    output = baseline.transform(test_inputs)

    # Validate output
    assert output.shape == test_inputs.shape, f"Shape mismatch: {output.shape} vs {test_inputs.shape}"
    assert output.min() >= -1.0 and output.max() <= 1.0, f"Output out of range: [{output.min()}, {output.max()}]"
    print(f"        Output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"        PASS")

# =============================================================================
# Test 4: LPIPS (optional - requires lpips package)
# =============================================================================
print("\n[4] Testing LPIPS...")

try:
    metrics = ImageMetrics()
    lpips_val = metrics.lpips(img, img2)
    print(f"    LPIPS (different images): {lpips_val:.4f}")

    lpips_identical = metrics.lpips(img, img)
    print(f"    LPIPS (identical): {lpips_identical:.4f} (expected: ~0)")
    assert lpips_identical < 0.01, f"LPIPS of identical images should be ~0, got {lpips_identical}"
    print("    PASS")
except ImportError:
    print("    SKIPPED - lpips package not installed")
    print("    Install with: pip install lpips")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nYou can now run the full evaluation:")
print("  python src/evaluate.py --checkpoint <path> --data_dir ./data")
