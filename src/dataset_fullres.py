"""
Full-Resolution Paired Image Dataset for NAFNet Training.

Loads pairs of images at their original resolution without cropping/resizing.
Supports variable-size images with proper batching strategies.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from torchvision import transforms
import random


class FullResolutionPairedDataset(Dataset):
    """
    Dataset for paired original and captured images at full resolution.

    No resizing or cropping - images are loaded at their native size.
    The model handles padding internally.
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

    def __init__(
        self,
        data_dir: str,
        original_subdir: str = "original",
        captured_subdir: str = "captured",
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        max_size: Optional[int] = None,  # Optional max dimension limit
        normalize: bool = True,  # Normalize to [-1, 1]
    ):
        """
        Initialize the full-resolution paired image dataset.

        Args:
            data_dir: Root directory containing original and captured subdirs
            original_subdir: Name of subdirectory with original images
            captured_subdir: Name of subdirectory with captured images
            split: Either "train" or "val"
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
            max_size: Optional maximum dimension (scales down if larger)
            normalize: Whether to normalize images to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.original_dir = self.data_dir / original_subdir
        self.captured_dir = self.data_dir / captured_subdir
        self.split = split
        self.max_size = max_size
        self.normalize = normalize

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

        # Cache image sizes for efficient batching
        self._cache_sizes()

    def _extract_numeric_id(self, filename: str) -> Optional[str]:
        """Extract numeric ID from filename."""
        import re
        match = re.search(r'(\d+)$', filename)
        return match.group(1) if match else None

    def _is_valid_image(self, path: Path) -> bool:
        """Check if an image file can be opened by PIL."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching pairs of original and captured images."""
        pairs = []
        skipped = []

        if not self.original_dir.exists():
            raise FileNotFoundError(f"Original images directory not found: {self.original_dir}")
        if not self.captured_dir.exists():
            raise FileNotFoundError(f"Captured images directory not found: {self.captured_dir}")

        # Get all original images, keyed by numeric ID
        original_files = {}
        for f in self.original_dir.iterdir():
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                numeric_id = self._extract_numeric_id(f.stem)
                if numeric_id:
                    original_files[numeric_id] = f

        # Find matching captured images by numeric ID
        for f in self.captured_dir.iterdir():
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                numeric_id = self._extract_numeric_id(f.stem)
                if numeric_id and numeric_id in original_files:
                    orig_path = original_files[numeric_id]
                    if self._is_valid_image(orig_path) and self._is_valid_image(f):
                        pairs.append((orig_path, f))
                    else:
                        skipped.append((orig_path.name, f.name))

        if skipped:
            print(f"Skipped {len(skipped)} corrupted image pairs")

        return sorted(pairs, key=lambda x: x[0].stem)

    def _cache_sizes(self):
        """Cache image sizes for efficient batching."""
        self.sizes = []
        for orig_path, _ in self.pairs:
            with Image.open(orig_path) as img:
                self.sizes.append(img.size)  # (width, height)

    def _get_size_bucket(self, idx: int) -> Tuple[int, int]:
        """Get size bucket for an image (for grouping similar sizes)."""
        w, h = self.sizes[idx]
        # Round to nearest 64 for bucketing
        bucket_w = ((w + 63) // 64) * 64
        bucket_h = ((h + 63) // 64) * 64
        return (bucket_w, bucket_h)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        original_path, captured_path = self.pairs[idx]

        # Load images
        original_img = Image.open(original_path).convert("RGB")
        captured_img = Image.open(captured_path).convert("RGB")

        # Optional scaling if max_size specified
        if self.max_size is not None:
            w, h = original_img.size
            if max(w, h) > self.max_size:
                scale = self.max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                original_img = original_img.resize((new_w, new_h), Image.LANCZOS)
                captured_img = captured_img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to tensor
        original_tensor = transforms.ToTensor()(original_img)  # [0, 1]
        captured_tensor = transforms.ToTensor()(captured_img)

        # Normalize to [-1, 1] if requested
        if self.normalize:
            original_tensor = original_tensor * 2 - 1
            captured_tensor = captured_tensor * 2 - 1

        return {
            "input_image": original_tensor,
            "target_image": captured_tensor,
            "original_path": str(original_path),
            "captured_path": str(captured_path),
            "original_size": self.sizes[idx],
        }


class SameSizeBatchSampler(Sampler):
    """
    Batch sampler that groups images by size.

    Images with the same (or similar) dimensions are batched together,
    allowing batch_size > 1 even with variable-size images.
    """

    def __init__(
        self,
        dataset: FullResolutionPairedDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by size bucket
        self.size_buckets = defaultdict(list)
        for idx in range(len(dataset)):
            bucket = dataset._get_size_bucket(idx)
            self.size_buckets[bucket].append(idx)

    def __iter__(self):
        # Create batches from each bucket
        batches = []

        for bucket, indices in self.size_buckets.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.size_buckets.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total


def collate_same_size(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for same-size batches."""
    return {
        "input_image": torch.stack([item["input_image"] for item in batch]),
        "target_image": torch.stack([item["target_image"] for item in batch]),
        "original_path": [item["original_path"] for item in batch],
        "captured_path": [item["captured_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


def collate_pad_to_max(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function that pads all images to the maximum size in the batch.

    Allows batching images of different sizes (less efficient but more flexible).
    """
    # Find max dimensions
    max_h = max(item["input_image"].shape[1] for item in batch)
    max_w = max(item["input_image"].shape[2] for item in batch)

    # Pad to multiple of 16 for NAFNet compatibility
    max_h = ((max_h + 15) // 16) * 16
    max_w = ((max_w + 15) // 16) * 16

    padded_inputs = []
    padded_targets = []
    pad_info = []

    for item in batch:
        h, w = item["input_image"].shape[1:]
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad input
        padded_input = torch.nn.functional.pad(
            item["input_image"], (0, pad_w, 0, pad_h), mode='reflect'
        )
        padded_inputs.append(padded_input)

        # Pad target
        padded_target = torch.nn.functional.pad(
            item["target_image"], (0, pad_w, 0, pad_h), mode='reflect'
        )
        padded_targets.append(padded_target)

        pad_info.append((h, w, pad_h, pad_w))

    return {
        "input_image": torch.stack(padded_inputs),
        "target_image": torch.stack(padded_targets),
        "original_path": [item["original_path"] for item in batch],
        "captured_path": [item["captured_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "pad_info": pad_info,  # (orig_h, orig_w, pad_h, pad_w)
    }


def get_fullres_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    original_subdir: str = "original",
    captured_subdir: str = "captured",
    pin_memory: bool = True,
    max_size: Optional[int] = None,
    same_size_batching: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for full-resolution images.

    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        val_split: Validation split fraction
        num_workers: Number of dataloader workers
        seed: Random seed
        original_subdir: Subdirectory for input images
        captured_subdir: Subdirectory for target images
        pin_memory: Whether to pin memory for CUDA transfers
        max_size: Optional maximum image dimension
        same_size_batching: If True, groups same-size images; if False, pads to max

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = FullResolutionPairedDataset(
        data_dir=data_dir,
        original_subdir=original_subdir,
        captured_subdir=captured_subdir,
        split="train",
        val_split=val_split,
        seed=seed,
        max_size=max_size,
    )

    val_dataset = FullResolutionPairedDataset(
        data_dir=data_dir,
        original_subdir=original_subdir,
        captured_subdir=captured_subdir,
        split="val",
        val_split=val_split,
        seed=seed,
        max_size=max_size,
    )

    if same_size_batching and batch_size > 1:
        # Use same-size batch sampler
        train_sampler = SameSizeBatchSampler(
            train_dataset, batch_size, drop_last=True, shuffle=True
        )
        val_sampler = SameSizeBatchSampler(
            val_dataset, batch_size, drop_last=False, shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_same_size,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_same_size,
        )
    else:
        # Use padding to max size in batch
        collate_fn = collate_pad_to_max if batch_size > 1 else collate_same_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    try:
        dataset = FullResolutionPairedDataset(args.data_dir, split="train")
        print(f"Found {len(dataset)} training pairs")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample['input_image'].shape}")
            print(f"Target shape: {sample['target_image'].shape}")
            print(f"Original size: {sample['original_size']}")
            print(f"Original path: {sample['original_path']}")

            # Test size buckets
            buckets = defaultdict(int)
            for i in range(len(dataset)):
                bucket = dataset._get_size_bucket(i)
                buckets[bucket] += 1

            print(f"\nSize buckets: {len(buckets)} unique sizes")
            for bucket, count in sorted(buckets.items(), key=lambda x: -x[1])[:5]:
                print(f"  {bucket}: {count} images")

    except Exception as e:
        print(f"Error: {e}")
