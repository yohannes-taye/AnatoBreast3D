"""Convert zipped breast dataset into training-ready folder."""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path


def find_subdir(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def copy_tree(src: Path, dst: Path) -> list[str]:
    files = []
    for file in sorted(src.rglob("*")):
        if file.is_file():
            rel = file.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            files.append(str(rel).replace("\\", "/"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare breast dataset from zip.")
    parser.add_argument("--zip", required=True, help="Path to BreastSegDataset.zip")
    parser.add_argument("--out", required=True, help="Output root, e.g. ./data/breastseg/processed_merged")
    args = parser.parse_args()

    out_root = Path(args.out)
    images_dir = out_root / "images"
    masks_dir = out_root / "sem_masks"
    ann_dir = out_root / "annotations"
    splits_dir = out_root / "splits"
    for path in [images_dir, masks_dir, ann_dir, splits_dir]:
        path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(args.zip).extractall(tmp)
        tmp_root = Path(tmp)
        img_src = find_subdir(tmp_root, ["images", "JPEGImages"])
        mask_src = find_subdir(tmp_root, ["sem_masks", "SegmentationClass", "masks"])
        ann_src = find_subdir(tmp_root, ["annotations"])
        if img_src is None or mask_src is None or ann_src is None:
            raise FileNotFoundError("Zip must contain images/, sem_masks/, and annotations/ directories.")

        image_list = copy_tree(img_src, images_dir)
        copy_tree(mask_src, masks_dir)
        for json_file in ann_src.glob("*.json"):
            shutil.copy2(json_file, ann_dir / "coco_det.json")
            break

    random.shuffle(image_list)
    split_idx = int(len(image_list) * 0.8)
    train_split = image_list[:split_idx]
    val_split = image_list[split_idx:]
    (splits_dir / "train.txt").write_text("\n".join(train_split), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_split), encoding="utf-8")
    print(f"[PrepareData] {len(train_split)} train / {len(val_split)} val samples written to {out_root}")


if __name__ == "__main__":
    main()
