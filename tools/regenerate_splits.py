"""Regenerate train/val splits with shuffled data."""

import argparse
import random
from pathlib import Path


def regenerate_splits(
    images_dir: Path,
    splits_dir: Path,
    split_ratio: float = 0.8,
    seed: int = 42,
    dataset3_images_dir: Path = Path("data/breastseg/dataset3_converted/images"),
    dataset3_list_file: Path | None = None,
) -> None:
    """Regenerate train/val splits from merged dataset."""
    rng = random.Random(seed)

    # Get all image files
    image_files = sorted([f.name for f in images_dir.glob("*.png")])

    print(f"Total images found: {len(image_files)}")

    # Identify Dataset3 files
    dataset3_files = set()
    if dataset3_images_dir.exists():
        dataset3_files = set([f.name for f in dataset3_images_dir.glob("*.png")])
        print(f"Dataset3 images: {len(dataset3_files)}")

    # Save Dataset3 file list for later use
    if dataset3_list_file and dataset3_files:
        dataset3_list_file.write_text("\n".join(sorted(dataset3_files)), encoding="utf-8")
        print(f"Dataset3 list saved to: {dataset3_list_file}")

    # Shuffle all files
    rng.shuffle(image_files)

    # Split
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Write splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train_files), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_files), encoding="utf-8")

    # Count Dataset3 samples in train/val
    train_d3 = sum(1 for f in train_files if f in dataset3_files)
    val_d3 = sum(1 for f in val_files if f in dataset3_files)

    print(f"\n[Done] Regenerated splits:")
    print(f"  Train: {len(train_files)} ({train_d3} from Dataset3)")
    print(f"  Val:   {len(val_files)} ({val_d3} from Dataset3)")
    print(f"  Saved to: {splits_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate train/val splits")
    parser.add_argument("--images-dir", type=Path, required=True, help="Images directory")
    parser.add_argument("--splits-dir", type=Path, required=True, help="Splits output directory")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset3-images-dir",
        type=Path,
        default=Path("data/breastseg/dataset3_converted/images"),
        help="Dataset3 image directory used to count and export hard-case IDs.",
    )
    parser.add_argument("--dataset3-list", type=Path, help="Output file for Dataset3 file list")
    args = parser.parse_args()

    regenerate_splits(
        args.images_dir,
        args.splits_dir,
        args.split_ratio,
        args.seed,
        args.dataset3_images_dir,
        args.dataset3_list,
    )


if __name__ == "__main__":
    main()
