"""
Metrics visualization for evaluation results.

Generates publication-quality charts comparing methods across metrics.
"""

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import torch
from PIL import Image

if TYPE_CHECKING:
    from evaluate import EvaluationResult, Evaluator


def create_metrics_bar_chart(
    results: List["EvaluationResult"],
    output_path: Path,
    figsize: tuple = (14, 5)
) -> None:
    """
    Create grouped bar chart comparing all methods across metrics.

    Args:
        results: List of EvaluationResult from evaluation
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    methods = [r.method_name for r in results]
    n_methods = len(methods)

    # Use a colormap for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_methods))

    # Highlight 'Ours' with a different color
    for i, method in enumerate(methods):
        if "Ours" in method or "Pix2Pix" in method:
            colors[i] = plt.cm.Set1(0.1)  # Red-ish for our method

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # PSNR subplot (higher is better)
    ax = axes[0]
    psnr_values = [r.psnr for r in results]
    psnr_stds = [r.psnr_std for r in results]
    bars = ax.bar(range(n_methods), psnr_values, color=colors, yerr=psnr_stds, capsize=3)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR (higher is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = np.argmax(psnr_values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2)

    # SSIM subplot (higher is better)
    ax = axes[1]
    ssim_values = [r.ssim for r in results]
    ssim_stds = [r.ssim_std for r in results]
    bars = ax.bar(range(n_methods), ssim_values, color=colors, yerr=ssim_stds, capsize=3)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM (higher is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = np.argmax(ssim_values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2)

    # LPIPS subplot (lower is better)
    ax = axes[2]
    lpips_values = [r.lpips for r in results]
    lpips_stds = [r.lpips_std for r in results]
    bars = ax.bar(range(n_methods), lpips_values, color=colors, yerr=lpips_stds, capsize=3)
    ax.set_ylabel('LPIPS', fontsize=12)
    ax.set_title('LPIPS (lower is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best (lowest for LPIPS)
    best_idx = np.argmin(lpips_values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor from [-1, 1] to uint8 RGB array."""
    # tensor shape: (C, H, W) or (B, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch

    # Convert from [-1, 1] to [0, 255]
    img = ((tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def create_sample_comparison_grid(
    evaluator: "Evaluator",
    output_path: Path,
    num_samples: int = 4
) -> None:
    """
    Create grid showing input, all method outputs, and target.

    Layout:
    - Rows: Different samples
    - Columns: Input | Baseline1 | Baseline2 | ... | Pix2Pix | Target

    Args:
        evaluator: Evaluator instance with loaded model and baselines
        output_path: Path to save the figure
        num_samples: Number of sample images to show
    """
    # Collect samples from validation set
    samples = []
    val_iter = iter(evaluator.val_loader)

    for _ in range(min(num_samples, len(evaluator.val_loader.dataset))):
        try:
            batch = next(val_iter)
            samples.append(batch)
        except StopIteration:
            break

    if len(samples) == 0:
        print("No samples available for visualization")
        return

    # Methods to visualize (order matters for display)
    methods = [
        ("Input", lambda x: x),
    ]

    # Add baselines
    for baseline in evaluator.baselines:
        methods.append((baseline.name, baseline.transform))

    # Add our model last (before target)
    methods.append(("Pix2Pix (Ours)", lambda x: evaluator.model(x.to(evaluator.device))))

    # Add target
    methods.append(("Target", None))  # Special handling for target

    n_cols = len(methods)
    n_rows = len(samples)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, batch in enumerate(samples):
        input_img = batch['input_image']
        target_img = batch['target_image']

        for col_idx, (method_name, transform_fn) in enumerate(methods):
            ax = axes[row_idx, col_idx]

            if method_name == "Target":
                # Show target image
                img_array = tensor_to_image(target_img)
            elif method_name == "Input":
                # Show input image
                img_array = tensor_to_image(input_img)
            else:
                # Apply transformation
                with torch.no_grad():
                    output = transform_fn(input_img)
                    if isinstance(output, torch.Tensor) and output.device != torch.device('cpu'):
                        output = output.cpu()
                img_array = tensor_to_image(output)

            ax.imshow(img_array)
            ax.axis('off')

            # Add title to first row only
            if row_idx == 0:
                ax.set_title(method_name, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_improvement_table_figure(
    results: List["EvaluationResult"],
    output_path: Path,
    figsize: tuple = (10, 4)
) -> None:
    """
    Create a figure showing relative improvement of Pix2Pix over baselines.

    Args:
        results: List of EvaluationResult from evaluation
        output_path: Path to save the figure
        figsize: Figure size
    """
    # Find Pix2Pix result
    pix2pix_result = None
    baseline_results = []

    for r in results:
        if "Ours" in r.method_name or "Pix2Pix" in r.method_name:
            pix2pix_result = r
        else:
            baseline_results.append(r)

    if pix2pix_result is None:
        print("No Pix2Pix result found")
        return

    # Calculate improvements
    methods = []
    psnr_improvements = []
    ssim_improvements = []
    lpips_improvements = []

    for baseline in baseline_results:
        methods.append(baseline.method_name)

        # PSNR improvement (positive = better)
        psnr_improvements.append(pix2pix_result.psnr - baseline.psnr)

        # SSIM improvement (positive = better)
        ssim_improvements.append(pix2pix_result.ssim - baseline.ssim)

        # LPIPS improvement (negative = better, so we flip sign for display)
        lpips_improvements.append(baseline.lpips - pix2pix_result.lpips)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, psnr_improvements, width, label='PSNR (dB)', color='steelblue')
    bars2 = ax.bar(x, [s * 100 for s in ssim_improvements], width, label='SSIM (x100)', color='darkorange')
    bars3 = ax.bar(x + width, [l * 100 for l in lpips_improvements], width, label='LPIPS (x100, lower=better)', color='forestgreen')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement over baseline', fontsize=12)
    ax.set_title('Pix2Pix Improvement Over Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    # Test with mock data
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        method_name: str
        psnr: float
        ssim: float
        lpips: float
        psnr_std: float = 0.5
        ssim_std: float = 0.02
        lpips_std: float = 0.01
        n_samples: int = 10

    results = [
        MockResult("Pix2Pix (Ours)", 28.5, 0.89, 0.08),
        MockResult("Identity", 18.3, 0.61, 0.32),
        MockResult("Linear Color Transfer", 22.9, 0.75, 0.20),
        MockResult("Channel Regression", 23.1, 0.75, 0.19),
        MockResult("Histogram Matching", 21.7, 0.72, 0.21),
    ]

    print("Creating test bar chart...")
    create_metrics_bar_chart(results, Path("test_metrics_chart.png"))
    print("Saved to test_metrics_chart.png")

    print("\nCreating improvement table figure...")
    create_improvement_table_figure(results, Path("test_improvement.png"))
    print("Saved to test_improvement.png")
