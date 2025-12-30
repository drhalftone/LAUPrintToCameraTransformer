"""
Paired Image Dataset for Print-to-Camera Transformer Training.

Loads pairs of original images and their corresponding printed+captured versions.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PairedImageDataset(Dataset):
    """Dataset for paired original and captured images."""

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        data_dir: str,
        original_subdir: str = "original",
        captured_subdir: str = "captured",
        image_size: int = 512,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize the paired image dataset.

        Args:
            data_dir: Root directory containing original and captured subdirs
            original_subdir: Name of subdirectory with original images
            captured_subdir: Name of subdirectory with captured images
            image_size: Size to resize images to (square)
            split: Either "train" or "val"
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.original_dir = self.data_dir / original_subdir
        self.captured_dir = self.data_dir / captured_subdir
        self.image_size = image_size
        self.split = split

        # Find all paired images
        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired images found in {data_dir}. "
                f"Ensure {original_subdir}/ and {captured_subdir}/ contain images with matching filenames."
            )

        # Split into train/val
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.pairs)).tolist()
        n_val = int(len(self.pairs) * val_split)

        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.pairs = [self.pairs[i] for i in indices]

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching pairs of original and captured images."""
        pairs = []

        if not self.original_dir.exists():
            raise FileNotFoundError(f"Original images directory not found: {self.original_dir}")
        if not self.captured_dir.exists():
            raise FileNotFoundError(f"Captured images directory not found: {self.captured_dir}")

        # Get all original images
        original_files = {}
        for f in self.original_dir.iterdir():
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Use stem (filename without extension) as key
                original_files[f.stem.lower()] = f

        # Find matching captured images
        for f in self.captured_dir.iterdir():
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                stem = f.stem.lower()
                if stem in original_files:
                    pairs.append((original_files[stem], f))

        return sorted(pairs, key=lambda x: x[0].stem)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_path, captured_path = self.pairs[idx]

        # Load images
        original_img = Image.open(original_path).convert("RGB")
        captured_img = Image.open(captured_path).convert("RGB")

        # Apply transforms
        original_tensor = self.transform(original_img)
        captured_tensor = self.transform(captured_img)

        return {
            "input_image": original_tensor,
            "target_image": captured_tensor,
            "original_path": str(original_path),
            "captured_path": str(captured_path),
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    image_size: int = 512,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        image_size: Image size (square)
        val_split: Validation split fraction
        num_workers: Number of dataloader workers
        seed: Random seed

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = PairedImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        split="train",
        val_split=val_split,
        seed=seed,
    )

    val_dataset = PairedImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        split="val",
        val_split=val_split,
        seed=seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    try:
        dataset = PairedImageDataset(args.data_dir, split="train")
        print(f"Found {len(dataset)} training pairs")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample['input_image'].shape}")
            print(f"Target shape: {sample['target_image'].shape}")
            print(f"Original path: {sample['original_path']}")
            print(f"Captured path: {sample['captured_path']}")
    except Exception as e:
        print(f"Error: {e}")
