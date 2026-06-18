"""Convert LabelMe-style BreastSegDataset into training-ready format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def find_image(image_dir: Path, stem: str) -> Path | None:
    """Locate the original image file matching a JSON stem."""
    for ext in IMG_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def polygon_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    """Return xmin, ymin, xmax, ymax for a polygon."""
    arr = np.array(points, dtype=np.float32)
    x_min = float(arr[:, 0].min())
    y_min = float(arr[:, 1].min())
    x_max = float(arr[:, 0].max())
    y_max = float(arr[:, 1].max())
    return x_min, y_min, x_max, y_max


def draw_polygons(width: int, height: int, shapes: list[dict]) -> Image.Image:
    """Rasterize breast polygons into a binary mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        label = shape.get("label", "").lower()
        if "breast" not in label:
            continue
        points = shape.get("points", [])
        if len(points) >= 3:
            draw.polygon([tuple(pt) for pt in points], fill=1)
    return mask


def load_shapes(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "images": root / "images",
        "masks": root / "sem_masks",
        "ann": root / "annotations",
        "splits": root / "splits",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def convert_dataset(json_dir: Path, image_dir: Path, out_root: Path, split_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    dirs = ensure_dirs(out_root)
    json_files = sorted(json_dir.glob("*.json"))
    images_meta = []
    annotations = []
    ann_id = 1
    sample_names = []

    for idx, json_path in enumerate(json_files, start=1):
        stem = json_path.stem
        image_path = find_image(image_dir, stem)
        if image_path is None:
            print(f"[WARN] Missing image for {stem}, skipping.")
            continue

        data = load_shapes(json_path)
        width = int(data.get("imageWidth") or Image.open(image_path).width)
        height = int(data.get("imageHeight") or Image.open(image_path).height)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height))
        target_name = f"{stem}.png"
        image.save(dirs["images"] / target_name)

        mask = draw_polygons(width, height, data.get("shapes", []))
        mask.save(dirs["masks"] / target_name)

        nipple_shapes = [
            shape for shape in data.get("shapes", []) if "nipple" in shape.get("label", "").lower()
        ]
        for shape in nipple_shapes:
            x_min, y_min, x_max, y_max = polygon_to_bbox(shape.get("points", []))
            w = max(x_max - x_min, 1.0)
            h = max(y_max - y_min, 1.0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": 1,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        images_meta.append(
            {
                "id": idx,
                "file_name": target_name,
                "width": width,
                "height": height,
            }
        )
        sample_names.append(target_name)

    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "nipple"}],
    }
    with open(dirs["ann"] / "coco_det.json", "w", encoding="utf-8") as file:
        json.dump(coco, file)

    rng.shuffle(sample_names)
    split_idx = int(len(sample_names) * split_ratio)
    train = sample_names[:split_idx]
    val = sample_names[split_idx:]
    (dirs["splits"] / "train.txt").write_text("\n".join(train), encoding="utf-8")
    (dirs["splits"] / "val.txt").write_text("\n".join(val), encoding="utf-8")
    print(f"[Done] Converted {len(sample_names)} samples → {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BreastSegDataset from LabelMe JSON.")
    parser.add_argument("--json-dir", required=True, help="Directory containing LabelMe JSON files.")
    parser.add_argument("--image-dir", required=True, help="Directory containing raw images.")
    parser.add_argument("--out", default="./data/breastseg/processed_merged", help="Output directory.")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    convert_dataset(Path(args.json_dir), Path(args.image_dir), Path(args.out), args.split, args.seed)


if __name__ == "__main__":
    main()
