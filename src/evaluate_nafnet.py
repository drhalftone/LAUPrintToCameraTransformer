"""
Evaluation script for NAFNet model.

Computes quantitative metrics (PSNR, SSIM, LPIPS) on the validation set
and compares against Pix2Pix and traditional baselines.
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
from nafnet_model import NAFNet, nafnet_width32
from unet_model import UNet
from metrics import ImageMetrics
from baselines import get_all_baselines


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


class NAFNetEvaluator:
    """Evaluation orchestrator for NAFNet, Pix2Pix, and baselines."""

    def __init__(
        self,
        nafnet_checkpoint_path: Path,
        pix2pix_checkpoint_path: Optional[Path],
        data_dir: Path,
        device: Optional[torch.device] = None,
        image_size: int = 512,
        original_subdir: str = "original",
        captured_subdir: str = "captured",
        pin_memory: bool = True,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = ImageMetrics(device=self.device)
        self.nafnet_checkpoint_path = nafnet_checkpoint_path
        self.pix2pix_checkpoint_path = pix2pix_checkpoint_path
        self.data_dir = data_dir

        # Load NAFNet model
        self.nafnet_model = self._load_nafnet(nafnet_checkpoint_path)

        # Load Pix2Pix model if available
        self.pix2pix_model = None
        if pix2pix_checkpoint_path and pix2pix_checkpoint_path.exists():
            try:
                self.pix2pix_model = self._load_pix2pix(pix2pix_checkpoint_path)
            except Exception as e:
                print(f"Warning: Could not load Pix2Pix model: {e}")

        # Load data
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=str(data_dir),
            batch_size=1,
            image_size=image_size,
            val_split=0.1,
            num_workers=0,
            original_subdir=original_subdir,
            captured_subdir=captured_subdir,
            pin_memory=pin_memory,
        )

        # Initialize baselines
        self.baselines = get_all_baselines()

    def _load_nafnet(self, checkpoint_path: Path) -> NAFNet:
        """Load trained NAFNet model."""
        model = nafnet_width32(in_channels=3, out_channels=3)

        nafnet_path = checkpoint_path / "nafnet.pt"
        if not nafnet_path.exists():
            raise FileNotFoundError(f"nafnet.pt not found in {checkpoint_path}")

        state_dict = torch.load(nafnet_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        print(f"Loaded NAFNet from {nafnet_path}")
        return model

    def _load_pix2pix(self, checkpoint_path: Path) -> UNet:
        """Load trained Pix2Pix generator."""
        model = UNet(in_channels=3, out_channels=3, features=64)

        generator_path = checkpoint_path / "generator.pt"
        if not generator_path.exists():
            raise FileNotFoundError(f"generator.pt not found in {checkpoint_path}")

        state_dict = torch.load(generator_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        print(f"Loaded Pix2Pix from {generator_path}")
        return model

    def fit_baselines(self, max_samples: int = 1000) -> None:
        """Fit baseline methods on training data."""
        print(f"Fitting baseline methods on training data (using up to {max_samples} samples)...")

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

    def evaluate_all(self, include_baselines: bool = True) -> List[EvaluationResult]:
        """Evaluate NAFNet, Pix2Pix, and all baselines."""

        results = []

        # Evaluate NAFNet model
        print("\nEvaluating NAFNet model...")
        nafnet_result = self.evaluate_method(
            "NAFNet (Ours)",
            transform_fn=lambda x: self.nafnet_model(x),
        )
        results.append(nafnet_result)

        # Evaluate Pix2Pix model if available
        if self.pix2pix_model is not None:
            print("\nEvaluating Pix2Pix model...")
            pix2pix_result = self.evaluate_method(
                "Pix2Pix",
                transform_fn=lambda x: self.pix2pix_model(x),
            )
            results.append(pix2pix_result)

        # Evaluate baselines
        if include_baselines:
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
        "=" * 85,
        f"{'Method':<25} {'PSNR (dB)':<20} {'SSIM':<20} {'LPIPS':<20}",
        "-" * 85,
    ]

    for r in results:
        psnr_str = f"{r.psnr:.2f} +/- {r.psnr_std:.2f}"
        ssim_str = f"{r.ssim:.4f} +/- {r.ssim_std:.4f}"
        lpips_str = f"{r.lpips:.4f} +/- {r.lpips_std:.4f}"
        lines.append(f"{r.method_name:<25} {psnr_str:<20} {ssim_str:<20} {lpips_str:<20}")

    lines.append("=" * 85)
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
        lines.append(f"{r.method_name} & {r.psnr:.2f} & {r.ssim:.4f} & {r.lpips:.4f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def save_results(results: List[EvaluationResult], output_dir: Path, nafnet_checkpoint: str, pix2pix_checkpoint: str, data_dir: str) -> None:
    """Save results in multiple formats."""

    # CSV
    with open(output_dir / "metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'PSNR_dB', 'PSNR_std', 'SSIM', 'SSIM_std', 'LPIPS', 'LPIPS_std', 'N_Samples'])
        for r in results:
            writer.writerow([
                r.method_name, r.psnr, r.psnr_std,
                r.ssim, r.ssim_std, r.lpips, r.lpips_std, r.n_samples
            ])

    # JSON
    data = {
        'evaluation_date': datetime.now().isoformat(),
        'nafnet_checkpoint': str(nafnet_checkpoint),
        'pix2pix_checkpoint': str(pix2pix_checkpoint),
        'data_dir': str(data_dir),
        'methods': [asdict(r) for r in results]
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(data, f, indent=2)

    # LaTeX table
    latex_table = format_latex_table(results)
    with open(output_dir / "table.tex", 'w') as f:
        f.write(latex_table)

    # Text report
    table = format_results_table(results)
    with open(output_dir / "report.txt", 'w') as f:
        f.write(f"Evaluation Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"NAFNet Checkpoint: {nafnet_checkpoint}\n")
        f.write(f"Pix2Pix Checkpoint: {pix2pix_checkpoint}\n")
        f.write(f"Data: {data_dir}\n")
        f.write(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NAFNet and compare with Pix2Pix")
    parser.add_argument("--nafnet_checkpoint", type=str, required=True,
                        help="Path to NAFNet checkpoint directory")
    parser.add_argument("--pix2pix_checkpoint", type=str, default=None,
                        help="Path to Pix2Pix checkpoint directory (optional)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results_nafnet",
                        help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for evaluation")
    parser.add_argument("--original_subdir", type=str, default="original",
                        help="Subdirectory for input images")
    parser.add_argument("--captured_subdir", type=str, default="captured",
                        help="Subdirectory for target images")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baseline evaluation")
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

    pin_memory = device.type == "cuda"

    # Initialize evaluator
    print(f"\nLoading NAFNet from {args.nafnet_checkpoint}")
    if args.pix2pix_checkpoint:
        print(f"Loading Pix2Pix from {args.pix2pix_checkpoint}")
    print(f"Loading data from {args.data_dir}")

    evaluator = NAFNetEvaluator(
        nafnet_checkpoint_path=Path(args.nafnet_checkpoint),
        pix2pix_checkpoint_path=Path(args.pix2pix_checkpoint) if args.pix2pix_checkpoint else None,
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
    if not args.no_baselines:
        evaluator.fit_baselines()

    # Run evaluation
    print("\n" + "=" * 50)
    print("Running evaluation...")
    print("=" * 50)

    results = evaluator.evaluate_all(include_baselines=not args.no_baselines)

    # Output results
    table = format_results_table(results)
    print(table)

    # Save results
    save_results(
        results, output_dir,
        args.nafnet_checkpoint,
        args.pix2pix_checkpoint or "N/A",
        args.data_dir
    )

    print(f"\nResults saved to {output_dir}/")
    print(f"  - metrics.csv: Raw metrics data")
    print(f"  - metrics.json: Detailed results with metadata")
    print(f"  - report.txt: Human-readable report")
    print(f"  - table.tex: LaTeX table for paper")


if __name__ == "__main__":
    main()
