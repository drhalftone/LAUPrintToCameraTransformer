"""
Data Organization Helper Script.

Helps validate and organize paired images for training.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def find_images(directory: Path) -> dict:
    """Find all images in a directory, keyed by stem (filename without extension)."""
    images = {}
    if not directory.exists():
        return images

    for f in directory.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images[f.stem.lower()] = f
    return images


def validate_image(path: Path) -> Tuple[bool, str]:
    """Validate that an image can be opened and get its properties."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            return True, f"{width}x{height} {mode}"
    except Exception as e:
        return False, str(e)


def analyze_data(data_dir: Path, original_subdir: str = "original", captured_subdir: str = "captured"):
    """Analyze the data directory and report statistics."""
    original_dir = data_dir / original_subdir
    captured_dir = data_dir / captured_subdir

    print(f"\n{'='*60}")
    print("DATA ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"\nData directory: {data_dir}")

    # Check directory existence
    if not original_dir.exists():
        print(f"\n[ERROR] Original directory not found: {original_dir}")
        print(f"  Please create it and add your original images.")
        return

    if not captured_dir.exists():
        print(f"\n[ERROR] Captured directory not found: {captured_dir}")
        print(f"  Please create it and add your captured images.")
        return

    # Find images
    original_images = find_images(original_dir)
    captured_images = find_images(captured_dir)

    print(f"\n--- Image Counts ---")
    print(f"Original images:  {len(original_images)}")
    print(f"Captured images:  {len(captured_images)}")

    # Find pairs
    original_stems = set(original_images.keys())
    captured_stems = set(captured_images.keys())

    paired_stems = original_stems & captured_stems
    only_original = original_stems - captured_stems
    only_captured = captured_stems - original_stems

    print(f"\n--- Pairing Status ---")
    print(f"Paired images:    {len(paired_stems)}")
    print(f"Only in original: {len(only_original)}")
    print(f"Only in captured: {len(only_captured)}")

    # Report unpaired images
    if only_original:
        print(f"\n[WARNING] Images in original/ without matching captured/:")
        for stem in sorted(list(only_original)[:10]):
            print(f"  - {original_images[stem].name}")
        if len(only_original) > 10:
            print(f"  ... and {len(only_original) - 10} more")

    if only_captured:
        print(f"\n[WARNING] Images in captured/ without matching original/:")
        for stem in sorted(list(only_captured)[:10]):
            print(f"  - {captured_images[stem].name}")
        if len(only_captured) > 10:
            print(f"  ... and {len(only_captured) - 10} more")

    # Validate paired images
    if paired_stems:
        print(f"\n--- Validating Paired Images ---")
        valid_pairs = 0
        invalid_pairs = []
        dimensions = defaultdict(int)

        for stem in tqdm(sorted(paired_stems), desc="Validating"):
            orig_valid, orig_info = validate_image(original_images[stem])
            cap_valid, cap_info = validate_image(captured_images[stem])

            if orig_valid and cap_valid:
                valid_pairs += 1
                dimensions[orig_info] += 1
            else:
                invalid_pairs.append((stem, orig_info if not orig_valid else cap_info))

        print(f"\nValid pairs:   {valid_pairs}")
        print(f"Invalid pairs: {len(invalid_pairs)}")

        if invalid_pairs:
            print(f"\n[ERROR] Invalid image pairs:")
            for stem, error in invalid_pairs[:5]:
                print(f"  - {stem}: {error}")
            if len(invalid_pairs) > 5:
                print(f"  ... and {len(invalid_pairs) - 5} more")

        print(f"\n--- Image Dimensions ---")
        for dim, count in sorted(dimensions.items(), key=lambda x: -x[1]):
            print(f"  {dim}: {count} images")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if len(paired_stems) == 0:
        print("\n[ERROR] No paired images found!")
        print("  Ensure original/ and captured/ have images with matching filenames.")
    elif len(paired_stems) < 100:
        print(f"\n[WARNING] Only {len(paired_stems)} paired images found.")
        print("  Consider adding more data for better training results.")
        print("  Recommended: 1000+ pairs for good results.")
    else:
        print(f"\n[OK] Found {len(paired_stems)} paired images.")

    if only_original or only_captured:
        print("\n[INFO] Some images are unpaired. This is usually fine if intentional.")

    print(f"\n--- Next Steps ---")
    if len(paired_stems) > 0:
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start training: python src/train.py --config configs/train_config.yaml")
    else:
        print("1. Add original images to: data/original/")
        print("2. Add captured images to: data/captured/")
        print("3. Ensure filenames match (e.g., image001.png in both)")
        print("4. Re-run this script to validate")


def copy_and_organize(
    source_original: Path,
    source_captured: Path,
    dest_dir: Path,
    rename_pattern: str = "image_{:05d}",
):
    """Copy and organize images from source to destination with consistent naming."""
    dest_original = dest_dir / "original"
    dest_captured = dest_dir / "captured"

    dest_original.mkdir(parents=True, exist_ok=True)
    dest_captured.mkdir(parents=True, exist_ok=True)

    # Find paired images
    original_images = find_images(source_original)
    captured_images = find_images(source_captured)

    paired_stems = set(original_images.keys()) & set(captured_images.keys())

    print(f"Found {len(paired_stems)} paired images to copy")

    for i, stem in enumerate(tqdm(sorted(paired_stems), desc="Copying")):
        new_name = rename_pattern.format(i)

        # Copy original
        orig_src = original_images[stem]
        orig_dst = dest_original / f"{new_name}{orig_src.suffix}"
        shutil.copy2(orig_src, orig_dst)

        # Copy captured
        cap_src = captured_images[stem]
        cap_dst = dest_captured / f"{new_name}{cap_src.suffix}"
        shutil.copy2(cap_src, cap_dst)

    print(f"\nCopied {len(paired_stems)} pairs to {dest_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Data organization helper for Print-to-Camera Transformer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data directory")
    analyze_parser.add_argument("--data_dir", type=str, default="./data",
                                help="Data directory to analyze")
    analyze_parser.add_argument("--original", type=str, default="original",
                                help="Subdirectory name for original images")
    analyze_parser.add_argument("--captured", type=str, default="captured",
                                help="Subdirectory name for captured images")

    # Organize command
    organize_parser = subparsers.add_parser("organize", help="Copy and organize images")
    organize_parser.add_argument("--source_original", type=str, required=True,
                                 help="Source directory for original images")
    organize_parser.add_argument("--source_captured", type=str, required=True,
                                 help="Source directory for captured images")
    organize_parser.add_argument("--dest", type=str, default="./data",
                                 help="Destination data directory")
    organize_parser.add_argument("--pattern", type=str, default="image_{:05d}",
                                 help="Naming pattern for organized images")

    args = parser.parse_args()

    if args.command == "analyze" or args.command is None:
        data_dir = Path(args.data_dir if hasattr(args, 'data_dir') else "./data")
        original = args.original if hasattr(args, 'original') else "original"
        captured = args.captured if hasattr(args, 'captured') else "captured"
        analyze_data(data_dir, original, captured)

    elif args.command == "organize":
        copy_and_organize(
            Path(args.source_original),
            Path(args.source_captured),
            Path(args.dest),
            args.pattern,
        )


if __name__ == "__main__":
    main()
