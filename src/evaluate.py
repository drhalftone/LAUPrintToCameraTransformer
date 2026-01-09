"""
Evaluation script for Pix2Pix model and baseline comparisons.

Computes quantitative metrics (PSNR, SSIM, LPIPS) on the validation set
and compares the trained model against traditional baseline methods.
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Callable, Optional

import torch
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders
from unet_model import UNet
from metrics import ImageMetrics
from baselines import get_all_baselines, BaselineMethod


@dataclass
class EvaluationResult:
    """Results for a single method."""
    method_name: str
    psnr: float
    ssim: float
    lpips: float
    psnr_std: float = 0.0
    ssim_std: float = 0.0
    lpips_std: float = 0.0
    n_samples: int = 0


class Evaluator:
    """Evaluation orchestrator for model and baselines."""

    def __init__(
        self,
        checkpoint_path: Path,
        data_dir: Path,
        device: Optional[torch.device] = None,
        image_size: int = 512,
        original_subdir: str = "original",
        captured_subdir: str = "captured",
        pin_memory: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint directory (contains generator.pt)
            data_dir: Path to data directory
            device: Computation device
            image_size: Image size for evaluation
            original_subdir: Subdirectory for input images
            captured_subdir: Subdirectory for target images
            pin_memory: Whether to pin memory for CUDA transfers
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = ImageMetrics(device=self.device)
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir

        # Load model
        self.model = self._load_model(checkpoint_path)

        # Load data
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=str(data_dir),
            batch_size=1,  # Process one at a time for accurate metrics
            image_size=image_size,
            val_split=0.1,
            num_workers=0,  # Avoid multiprocessing issues on Windows
            original_subdir=original_subdir,
            captured_subdir=captured_subdir,
            pin_memory=pin_memory,
        )

        # Initialize baselines
        self.baselines = get_all_baselines()

    def _load_model(self, checkpoint_path: Path) -> UNet:
        """Load trained Pix2Pix generator."""
        model = UNet(in_channels=3, out_channels=3, features=64)

        generator_path = checkpoint_path / "generator.pt"
        if not generator_path.exists():
            raise FileNotFoundError(f"generator.pt not found in {checkpoint_path}")

        state_dict = torch.load(generator_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def fit_baselines(self, max_samples: int = 1000) -> None:
        """Fit baseline methods on training data.

        Args:
            max_samples: Maximum number of samples to use for fitting (for memory efficiency)
        """
        print(f"Fitting baseline methods on training data (using up to {max_samples} samples)...")

        # Collect training samples (limited for memory)
        inputs_list = []
        targets_list = []
        count = 0

        for batch in tqdm(self.train_loader, desc="Loading training data"):
            inputs_list.append(batch['input_image'])
            targets_list.append(batch['target_image'])
            count += batch['input_image'].shape[0]
            if count >= max_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:max_samples]
        targets = torch.cat(targets_list, dim=0)[:max_samples]

        print(f"Collected {len(inputs)} training samples for baseline fitting")

        # Fit each baseline
        for baseline in self.baselines:
            print(f"  Fitting {baseline.name}...")
            baseline.fit(inputs, targets)

    def evaluate_method(
        self,
        method_name: str,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> EvaluationResult:
        """Evaluate a single method on validation set."""

        psnr_values = []
        ssim_values = []
        lpips_values = []

        for batch in tqdm(self.val_loader, desc=f"Evaluating {method_name}", leave=False):
            inputs = batch['input_image'].to(self.device)
            targets = batch['target_image'].to(self.device)

            with torch.no_grad():
                preds = transform_fn(inputs)

            # Compute metrics for this sample
            psnr_values.append(self.metrics.psnr(preds, targets))
            ssim_values.append(self.metrics.ssim(preds, targets))
            lpips_values.append(self.metrics.lpips(preds, targets))

        return EvaluationResult(
            method_name=method_name,
            psnr=float(np.mean(psnr_values)),
            ssim=float(np.mean(ssim_values)),
            lpips=float(np.mean(lpips_values)),
            psnr_std=float(np.std(psnr_values)),
            ssim_std=float(np.std(ssim_values)),
            lpips_std=float(np.std(lpips_values)),
            n_samples=len(psnr_values)
        )

    def evaluate_all(self) -> List[EvaluationResult]:
        """Evaluate Pix2Pix model and all baselines."""

        results = []

        # Evaluate Pix2Pix model
        print("\nEvaluating Pix2Pix model...")
        pix2pix_result = self.evaluate_method(
            "Pix2Pix (Ours)",
            transform_fn=lambda x: self.model(x),
        )
        results.append(pix2pix_result)

        # Evaluate baselines
        for baseline in self.baselines:
            print(f"\nEvaluating {baseline.name}...")
            result = self.evaluate_method(
                baseline.name,
                transform_fn=baseline.transform,
            )
            results.append(result)

        return results


def format_results_table(results: List[EvaluationResult]) -> str:
    """Format results as ASCII table."""

    lines = [
        "",
        "=" * 80,
        f"{'Method':<25} {'PSNR (dB)':<15} {'SSIM':<15} {'LPIPS':<15}",
        "-" * 80,
    ]

    for r in results:
        psnr_str = f"{r.psnr:.2f} +/- {r.psnr_std:.2f}"
        ssim_str = f"{r.ssim:.4f} +/- {r.ssim_std:.4f}"
        lpips_str = f"{r.lpips:.4f} +/- {r.lpips_std:.4f}"
        lines.append(f"{r.method_name:<25} {psnr_str:<15} {ssim_str:<15} {lpips_str:<15}")

    lines.append("=" * 80)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - PSNR: Peak Signal-to-Noise Ratio (higher is better)")
    lines.append("  - SSIM: Structural Similarity Index (higher is better)")
    lines.append("  - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)")
    lines.append(f"  - Samples evaluated: {results[0].n_samples}")
    lines.append("")

    return "\n".join(lines)


def format_latex_table(results: List[EvaluationResult]) -> str:
    """Format results as LaTeX table for paper."""

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Quantitative comparison of methods}",
        "\\label{tab:results}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & PSNR (dB) $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\",
        "\\midrule",
    ]

    for r in results:
        # Bold the best values
        lines.append(f"{r.method_name} & {r.psnr:.2f} & {r.ssim:.4f} & {r.lpips:.4f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def save_results_csv(results: List[EvaluationResult], path: Path) -> None:
    """Save results to CSV file."""

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'PSNR_dB', 'PSNR_std', 'SSIM', 'SSIM_std', 'LPIPS', 'LPIPS_std', 'N_Samples'])
        for r in results:
            writer.writerow([
                r.method_name, r.psnr, r.psnr_std,
                r.ssim, r.ssim_std, r.lpips, r.lpips_std, r.n_samples
            ])


def save_results_json(results: List[EvaluationResult], path: Path, checkpoint_path: str, data_dir: str) -> None:
    """Save results to JSON file."""

    data = {
        'evaluation_date': datetime.now().isoformat(),
        'checkpoint': str(checkpoint_path),
        'data_dir': str(data_dir),
        'methods': [asdict(r) for r in results]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix model and baselines")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for evaluation")
    parser.add_argument("--original_subdir", type=str, default="original",
                        help="Subdirectory for input images")
    parser.add_argument("--captured_subdir", type=str, default="captured",
                        help="Subdirectory for target images")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization charts")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Only use pin_memory with CUDA
    pin_memory = device.type == "cuda"

    # Initialize evaluator
    print(f"\nLoading model from {args.checkpoint}")
    print(f"Loading data from {args.data_dir}")

    evaluator = Evaluator(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        device=device,
        image_size=args.image_size,
        original_subdir=args.original_subdir,
        captured_subdir=args.captured_subdir,
        pin_memory=pin_memory,
    )

    print(f"Training samples: {len(evaluator.train_loader.dataset)}")
    print(f"Validation samples: {len(evaluator.val_loader.dataset)}")

    # Fit baselines on training data
    evaluator.fit_baselines()

    # Run evaluation
    print("\n" + "=" * 50)
    print("Running evaluation...")
    print("=" * 50)

    results = evaluator.evaluate_all()

    # Output results
    table = format_results_table(results)
    print(table)

    # Save results
    save_results_csv(results, output_dir / "metrics.csv")
    save_results_json(results, output_dir / "metrics.json", args.checkpoint, args.data_dir)

    # Save text report
    with open(output_dir / "report.txt", 'w') as f:
        f.write(f"Evaluation Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(table)

    # Save LaTeX table
    latex_table = format_latex_table(results)
    with open(output_dir / "table.tex", 'w') as f:
        f.write(latex_table)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - metrics.csv: Raw metrics data")
    print(f"  - metrics.json: Detailed results with metadata")
    print(f"  - report.txt: Human-readable report")
    print(f"  - table.tex: LaTeX table for paper")

    # Generate visualizations
    if args.visualize:
        try:
            from visualize_metrics import create_metrics_bar_chart, create_sample_comparison_grid
            print("\nGenerating visualizations...")
            create_metrics_bar_chart(results, output_dir / "metrics_comparison.png")
            create_sample_comparison_grid(evaluator, output_dir / "sample_comparison.png", num_samples=4)
            print(f"  - metrics_comparison.png: Bar chart comparing methods")
            print(f"  - sample_comparison.png: Visual comparison grid")
        except ImportError as e:
            print(f"\nWarning: Could not generate visualizations: {e}")
        except Exception as e:
            print(f"\nWarning: Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
