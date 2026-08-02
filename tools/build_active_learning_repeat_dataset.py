"""Build a physically repeated active-learning fine-tune dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.aug_ops import affine_warp, horizontal_flip  # noqa: E402


AUG_TOKEN = "_aug"
MIN_BOX_SIDE_PX = 4.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_dataset_file(root: Path, subdir: str, file_name: str) -> Path:
    path = root / subdir / file_name
    if path.is_file():
        return path
    stem = Path(file_name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = root / subdir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {subdir} file for {file_name}: {root / subdir}")


def _safe_prepare_dst(src: Path, dst: Path, overwrite: bool) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    project_resolved = PROJECT_ROOT.resolve()
    if dst_resolved == src_resolved:
        raise ValueError("Destination must not be the source dataset.")
    if dst_resolved == project_resolved:
        raise ValueError("Destination must not be the project root.")
    if project_resolved not in dst_resolved.parents:
        raise ValueError(f"Destination must stay inside the project: {dst}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    for subdir in ("images", "sem_masks", "labels", "annotations", "splits"):
        (dst / subdir).mkdir(parents=True, exist_ok=True)


def _normalize_date_from(value: str | None) -> str | None:
    if not value:
        return None
    token = "".join(ch for ch in value if ch.isdigit())
    if len(token) != 8:
        raise ValueError(f"--date-from must be YYYYMMDD or YYYY-MM-DD, got: {value}")
    return token


def _is_augmented_name(name: str) -> bool:
    return AUG_TOKEN in Path(name).stem


def _sample_date_key(sample_id: str, file_name: str) -> str | None:
    for value in (sample_id, Path(file_name).stem):
        token = value[:8]
        if len(token) == 8 and token.isdigit():
            return token
    return None


def _sample_aug_params(rng: np.random.Generator) -> dict[str, Any]:
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


def _photometric(image: np.ndarray, params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
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


def _make_augmented_assets(
    sample: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[list[float]], dict[str, Any]]:
    image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Missing image or mask for source sample: {sample['file_name']}")
    mask = (mask > 0).astype(np.uint8)
    boxes = np.asarray(sample["boxes"], dtype=np.float32)

    for _ in range(30):
        params = _sample_aug_params(rng)
        aug_image = image.copy()
        aug_mask = mask.copy()
        aug_boxes = boxes.copy()
        if params["flip"]:
            aug_image, aug_mask, aug_boxes = horizontal_flip(aug_image, aug_mask, aug_boxes)
        h, w = aug_image.shape[:2]
        aug_image, aug_mask, aug_boxes = affine_warp(
            aug_image,
            aug_mask,
            aug_boxes,
            angle=params["angle"],
            scale=params["scale"],
            tx_px=params["tx"] * w,
            ty_px=params["ty"] * h,
            image_border=(128, 128, 128),
            mask_border=0,
            min_box_side=MIN_BOX_SIDE_PX,
        )
        if aug_boxes.shape[0] == boxes.shape[0]:
            aug_image = _photometric(aug_image, params, rng)
            aug_mask = (aug_mask > 0).astype(np.uint8)
            return (
                aug_image,
                aug_mask,
                [[float(value) for value in box] for box in aug_boxes.tolist()],
                params,
            )

    raise ValueError(f"Unable to keep all nipple boxes after augmentation: {sample['label_path']}")


def _load_source_samples(
    src: Path,
    *,
    date_from: str | None = None,
    include_existing_augmented: bool = True,
) -> list[dict[str, Any]]:
    labels_dir = src / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
    samples: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        label = _read_json(label_path)
        sample_id = str(label.get("sample_id") or label_path.stem)
        file_name = str(label.get("file_name") or label.get("image") or f"{sample_id}.png")
        if not include_existing_augmented and _is_augmented_name(file_name):
            continue
        sample_date = _sample_date_key(sample_id, file_name)
        if date_from is not None and (sample_date is None or sample_date < date_from):
            continue
        boxes = label.get("nipple_boxes_xyxy") or []
        if len(boxes) != 2:
            raise ValueError(f"{label_path} has {len(boxes)} nipple boxes; expected exactly 2.")
        image_path = _resolve_dataset_file(src, "images", file_name)
        mask_path = _resolve_dataset_file(src, "sem_masks", file_name)
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(
            {
                "sample_id": sample_id,
                "file_name": image_path.name,
                "stem": image_path.stem,
                "label_path": label_path,
                "label": label,
                "boxes": boxes,
                "image_path": image_path,
                "mask_path": mask_path,
                "width": width,
                "height": height,
            }
        )
    if not samples:
        raise ValueError(f"No label json files found in {labels_dir}.")
    return samples


def _copy_sample(
    *,
    sample: dict[str, Any],
    dst: Path,
    new_stem: str,
    role: str,
    repeat_index: int,
    image_id: int,
    ann_id_start: int,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any], int]:
    new_file_name = f"{new_stem}.png"
    boxes = [[float(value) for value in box] for box in sample["boxes"]]
    width = int(sample["width"])
    height = int(sample["height"])
    transform: dict[str, Any] | None = None

    if augment:
        if rng is None:
            raise ValueError("rng is required when augment=True.")
        image_bgr, mask, boxes, transform = _make_augmented_assets(sample, rng)
        height, width = mask.shape[:2]
        cv2.imwrite(str(dst / "images" / new_file_name), image_bgr)
        cv2.imwrite(str(dst / "sem_masks" / new_file_name), mask)
    else:
        shutil.copy2(sample["image_path"], dst / "images" / new_file_name)
        shutil.copy2(sample["mask_path"], dst / "sem_masks" / new_file_name)

    label_payload = dict(sample["label"])
    label_payload.update(
        {
            "sample_id": new_stem,
            "file_name": new_file_name,
            "nipple_boxes_xyxy": boxes,
            "repeat_role": role,
            "repeat_index": repeat_index,
            "source_sample_id": sample["sample_id"],
            "source_file_name": sample["file_name"],
        }
    )
    if transform is not None:
        label_payload["source"] = "augmentation"
        label_payload["transform"] = transform
    _write_json(dst / "labels" / f"{new_stem}.json", label_payload)

    image_record = {
        "id": image_id,
        "file_name": new_file_name,
        "width": width,
        "height": height,
    }
    annotations: list[dict[str, Any]] = []
    ann_id = ann_id_start
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid box for {sample['label_path']}: {box}")
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return new_file_name, image_record, annotations, label_payload, ann_id


def build_repeat_dataset(
    src: Path,
    dst: Path,
    repeat: int,
    overwrite: bool,
    *,
    date_from: str | None = None,
    include_existing_augmented: bool = True,
    augment_train: bool = False,
    seed: int = 23,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    date_from = _normalize_date_from(date_from)
    src = src.resolve()
    dst = dst.resolve()
    _safe_prepare_dst(src, dst, overwrite)

    train_names: list[str] = []
    val_names: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    ann_id = 1
    samples = _load_source_samples(
        src,
        date_from=date_from,
        include_existing_augmented=include_existing_augmented,
    )
    rng = np.random.default_rng(seed)

    for sample in samples:
        for repeat_index in range(1, repeat + 1):
            suffix = "augtr" if augment_train else "tr"
            new_stem = f"{sample['stem']}_{suffix}{repeat_index:02d}"
            file_name, image_record, anns, label_payload, ann_id = _copy_sample(
                sample=sample,
                dst=dst,
                new_stem=new_stem,
                role="train_augmented" if augment_train else "train",
                repeat_index=repeat_index,
                image_id=image_id,
                ann_id_start=ann_id,
                augment=augment_train,
                rng=rng,
            )
            train_names.append(file_name)
            images.append(image_record)
            annotations.extend(anns)
            manifest_rows.append(label_payload)
            image_id += 1

        new_stem = f"{sample['stem']}_val"
        file_name, image_record, anns, label_payload, ann_id = _copy_sample(
            sample=sample,
            dst=dst,
            new_stem=new_stem,
            role="val",
            repeat_index=0,
            image_id=image_id,
            ann_id_start=ann_id,
        )
        val_names.append(file_name)
        images.append(image_record)
        annotations.extend(anns)
        manifest_rows.append(label_payload)
        image_id += 1

    (dst / "splits" / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (dst / "splits" / "val.txt").write_text("\n".join(val_names) + "\n", encoding="utf-8")
    (dst / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    _write_json(
        dst / "annotations" / "coco_det.json",
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "nipple"}],
        },
    )

    return {
        "source": str(src),
        "destination": str(dst),
        "source_samples": len(samples),
        "date_from": date_from,
        "include_existing_augmented": include_existing_augmented,
        "augment_train": augment_train,
        "seed": seed,
        "repeat": repeat,
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "images": len(images),
        "annotations": len(annotations),
        "manifest_rows": len(manifest_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repeated active-learning fine-tune dataset.")
    parser.add_argument("--src", default="active_learning/dataset", help="Source active-learning dataset root.")
    parser.add_argument("--dst", default="active_learning/dataset_repeat3", help="Destination dataset root.")
    parser.add_argument("--repeat", type=int, default=3, help="Train copies per source sample.")
    parser.add_argument("--overwrite", action="store_true", help="Replace destination if it already exists.")
    parser.add_argument("--date-from", help="Keep samples whose filename date is >= YYYYMMDD.")
    parser.add_argument(
        "--exclude-existing-augmented",
        action="store_true",
        help="Skip source files whose stem already contains '_aug'.",
    )
    parser.add_argument("--augment-train", action="store_true", help="Apply offline augmentation to train repeats.")
    parser.add_argument("--seed", type=int, default=23, help="RNG seed for train-repeat augmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_repeat_dataset(
        Path(args.src),
        Path(args.dst),
        args.repeat,
        args.overwrite,
        date_from=args.date_from,
        include_existing_augmented=not args.exclude_existing_augmented,
        augment_train=args.augment_train,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
