"""Offline data augmentation for the active-learning breast-seg dataset.

Grows a small curated set (e.g. 6 samples) up to a target count (e.g. 60) by
applying consistent geometric + photometric transforms to each original
image/mask/box triple. Augmented files are written **in place** next to the
originals and the COCO file, label JSONs, manifest and split files are kept in
sync.

The script is deterministic (``--seed``) and idempotent: every run first removes
any previously generated ``*_aug*`` artifacts, then regenerates from the
originals, so re-running never accumulates duplicates.

Why offline augmentation when the training loader already augments on the fly:
with only a handful of base images and ``drop_last=True`` the loader yields very
few optimizer steps per epoch. Raising the file count multiplies steps/epoch and
batch diversity. Keep transforms diverse but plausible so the 54 new frames are
distinct rather than near-duplicates.

Usage:
    python data_preprocess/augment_dataset.py \
        --data-root active_learning/dataset --target-count 60 --seed 23 [--preview]

Critical correctness notes:
- Masks are class-index maps with values {0, 1}; 255 means "ignore" in the
  loader. Geometric warps therefore fill the mask border with 0 (background) and
  the output is re-binarized to {0, 1}.
- COCO bbox is xywh; label JSON boxes are xyxy. Split files list ``<name>.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocess.aug_ops import affine_warp, horizontal_flip  # noqa: E402


AUG_TOKEN = "_aug"
MIN_BOX_SIDE_PX = 4.0


# --------------------------------------------------------------------------- IO

def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + replace, matching annotation_store."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_coco(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "nipple"}]}


def _dataset_paths(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "images": root / "images",
        "masks": root / "sem_masks",
        "labels": root / "labels",
        "ann": root / "annotations",
        "splits": root / "splits",
        "preview": root / "_aug_preview",
    }
    for key in ("images", "masks", "labels", "ann", "splits"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _is_augmented(name: str) -> bool:
    return AUG_TOKEN in Path(name).stem


# ------------------------------------------------------------------- geometry

def _geometric(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply hflip + affine consistently to image (BGR), mask and xyxy boxes.

    Uses the shared ``data_preprocess.aug_ops`` primitives with ``mask_border=0`` so
    revealed mask pixels become background (these masks are persisted to disk).
    """
    if params["flip"]:
        image, mask, boxes = horizontal_flip(image, mask, boxes)
    h, w = image.shape[:2]
    return affine_warp(
        image,
        mask,
        boxes,
        angle=params["angle"],
        scale=params["scale"],
        tx_px=params["tx"] * w,
        ty_px=params["ty"] * h,
        image_border=(128, 128, 128),
        mask_border=0,
        min_box_side=MIN_BOX_SIDE_PX,
    )


def _photometric(image: np.ndarray, params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast + HSV + gamma (+ optional blur/noise) on a BGR image."""
    img = image.astype(np.float32)
    img = (img - 128.0) * params["contrast"] + 128.0 + params["brightness"]
    img = np.clip(img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + params["hue"]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * params["sat"], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * params["val"], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    lut = np.clip((np.linspace(0, 1, 256) ** (1.0 / params["gamma"])) * 255.0, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, lut)

    if params["blur"]:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if params["noise"] > 0:
        noise = rng.normal(0.0, params["noise"], img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "flip": bool(rng.random() < 0.5),
        "angle": float(rng.uniform(-15.0, 15.0)),
        "scale": float(rng.uniform(0.85, 1.15)),
        "tx": float(rng.uniform(-0.08, 0.08)),
        "ty": float(rng.uniform(-0.08, 0.08)),
        "brightness": float(rng.uniform(-18.0, 18.0)),
        "contrast": float(rng.uniform(0.85, 1.15)),
        "hue": float(rng.uniform(-10.0, 10.0)),
        "sat": float(rng.uniform(0.85, 1.15)),
        "val": float(rng.uniform(0.85, 1.15)),
        "gamma": float(rng.uniform(0.8, 1.2)),
        "blur": bool(rng.random() < 0.25),
        "noise": float(rng.uniform(0.0, 4.0)) if rng.random() < 0.4 else 0.0,
    }


# ----------------------------------------------------------------- dataset ops

def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def _boxes_by_filename(coco: dict[str, Any]) -> dict[str, np.ndarray]:
    meta = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    grouped: dict[str, list[list[float]]] = {}
    for ann in coco.get("annotations", []):
        fname = meta.get(ann["image_id"])
        if fname is None:
            continue
        x, y, w, h = ann["bbox"]
        grouped.setdefault(fname, []).append([x, y, x + w, y + h])
    return {name: np.asarray(rows, dtype=np.float32) for name, rows in grouped.items()}


def _clean_previous(paths: dict[str, Path]) -> None:
    for directory, pattern in (
        (paths["images"], f"*{AUG_TOKEN}*.png"),
        (paths["masks"], f"*{AUG_TOKEN}*.png"),
        (paths["labels"], f"*{AUG_TOKEN}*.json"),
    ):
        for path in directory.glob(pattern):
            path.unlink()
    if paths["preview"].is_dir():
        for path in paths["preview"].glob("*.jpg"):
            path.unlink()


def _write_preview(paths: dict[str, Path], name: str, image_bgr: np.ndarray, mask01: np.ndarray, boxes: np.ndarray) -> None:
    paths["preview"].mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[mask01 > 0] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0.0)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), 2)
    cv2.imwrite(str(paths["preview"] / f"{name}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


def augment_dataset(root: Path, target_count: int, seed: int, preview: bool) -> dict[str, Any]:
    paths = _dataset_paths(root)
    coco_path = paths["ann"] / "coco_det.json"
    coco = _read_coco(coco_path)

    # 1. Split originals from any prior augmentations.
    original_images = [img for img in coco.get("images", []) if not _is_augmented(img["file_name"])]
    if not original_images:
        raise SystemExit(f"No original (non-augmented) images found in {coco_path}")
    original_names = [img["file_name"] for img in original_images]
    boxes_lookup = _boxes_by_filename(coco)

    # 2. Remove previously generated artifacts so the run is idempotent.
    _clean_previous(paths)

    # Rebuild COCO keeping only originals, with compact ids preserved.
    orig_img_ids = {img["id"] for img in original_images}
    new_coco: dict[str, Any] = {
        "images": list(original_images),
        "categories": coco.get("categories") or [{"id": 1, "name": "nipple"}],
        "annotations": [a for a in coco.get("annotations", []) if a["image_id"] in orig_img_ids],
    }
    next_image_id = max((img["id"] for img in new_coco["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in new_coco["annotations"]), default=0) + 1

    # Rebuild manifest keeping only original lines.
    manifest_path = paths["root"] / "manifest.jsonl"
    original_manifest: list[str] = []
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not _is_augmented(str(record.get("file_name", ""))):
                original_manifest.append(stripped)

    # 3. Plan how many augmentations per original (round-robin remainder).
    num_to_generate = max(0, target_count - len(original_names))
    rng = np.random.default_rng(seed)
    new_manifest: list[str] = []
    per_original_counter: dict[str, int] = {name: 0 for name in original_names}

    for index in range(num_to_generate):
        src_name = original_names[index % len(original_names)]
        src_stem = Path(src_name).stem
        per_original_counter[src_name] += 1
        aug_name = f"{src_stem}{AUG_TOKEN}{per_original_counter[src_name]:02d}.png"

        image = cv2.imread(str(paths["images"] / src_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(paths["masks"] / src_name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise SystemExit(f"Missing image or mask for original sample: {src_name}")
        mask = (mask > 0).astype(np.uint8)
        boxes = boxes_lookup.get(src_name, np.zeros((0, 4), dtype=np.float32)).copy()

        params = _sample_params(rng)
        aug_image, aug_mask, aug_boxes = _geometric(image, mask, boxes, params)
        aug_image = _photometric(aug_image, params, rng)
        aug_mask = (aug_mask > 0).astype(np.uint8)
        height, width = aug_mask.shape[:2]

        cv2.imwrite(str(paths["images"] / aug_name), aug_image)
        cv2.imwrite(str(paths["masks"] / aug_name), aug_mask)
        if preview:
            _write_preview(paths, Path(aug_name).stem, aug_image, aug_mask, aug_boxes)

        image_id = next_image_id
        next_image_id += 1
        new_coco["images"].append(
            {"id": image_id, "file_name": aug_name, "width": int(width), "height": int(height)}
        )
        for box in aug_boxes:
            x, y, w, h = _xyxy_to_xywh(box)
            new_coco["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

        label_payload = {
            "sample_id": Path(aug_name).stem,
            "file_name": aug_name,
            "source": "augmentation",
            "parent_file_name": src_name,
            "nipple_boxes_xyxy": [[float(v) for v in box] for box in aug_boxes],
            "transform": params,
            "created_at": time.time(),
        }
        _atomic_write_json(paths["labels"] / f"{Path(aug_name).stem}.json", label_payload)
        new_manifest.append(json.dumps(label_payload, ensure_ascii=False))

    # 4. Persist COCO, manifest and split files.
    _atomic_write_json(coco_path, new_coco)

    manifest_lines = original_manifest + new_manifest
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

    all_names = sorted(p.name for p in paths["images"].glob("*.png"))
    paths["splits"].joinpath("train.txt").write_text("\n".join(all_names), encoding="utf-8")
    paths["splits"].joinpath("val.txt").write_text("\n".join(sorted(original_names)), encoding="utf-8")

    return {
        "originals": len(original_names),
        "generated": num_to_generate,
        "total_images": len(all_names),
        "coco_images": len(new_coco["images"]),
        "coco_annotations": len(new_coco["annotations"]),
        "train_split": len(all_names),
        "val_split": len(original_names),
    }


def _verify(root: Path, summary: dict[str, Any]) -> None:
    """Cheap integrity asserts that guard the common footguns."""
    paths = _dataset_paths(root)
    images = sorted(p.name for p in paths["images"].glob("*.png"))
    masks = {p.name for p in paths["masks"].glob("*.png")}
    assert len(images) == summary["total_images"], "image count mismatch"
    for name in images:
        assert name in masks, f"missing mask for {name}"
        mask = cv2.imread(str(paths["masks"] / name), cv2.IMREAD_GRAYSCALE)
        unique = set(np.unique(mask).tolist())
        assert unique <= {0, 1}, f"mask {name} has non-binary values {unique}"
    coco = _read_coco(paths["ann"] / "coco_det.json")
    img_meta = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        meta = img_meta[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= meta["width"] and 0 <= y <= meta["height"], f"bbox out of frame in {meta['file_name']}"
        assert w > 0 and h > 0, f"degenerate bbox in {meta['file_name']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment the active-learning breast-seg dataset in place.")
    parser.add_argument("--data-root", default="active_learning/dataset", help="Dataset root directory.")
    parser.add_argument("--target-count", type=int, default=60, help="Total image count after augmentation.")
    parser.add_argument("--seed", type=int, default=23, help="RNG seed for reproducible augmentation.")
    parser.add_argument("--preview", action="store_true", help="Write overlay previews to <root>/_aug_preview.")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    summary = augment_dataset(root, args.target_count, args.seed, args.preview)
    _verify(root, summary)

    print("Augmentation complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  dataset_root: {root}")


if __name__ == "__main__":
    main()
