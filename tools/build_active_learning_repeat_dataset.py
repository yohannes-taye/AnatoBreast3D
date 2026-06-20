"""Build a physically repeated active-learning fine-tune dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _load_source_samples(src: Path) -> list[dict[str, Any]]:
    labels_dir = src / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
    samples: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        label = _read_json(label_path)
        sample_id = str(label.get("sample_id") or label_path.stem)
        file_name = str(label.get("file_name") or label.get("image") or f"{sample_id}.png")
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
) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any], int]:
    new_file_name = f"{new_stem}.png"
    shutil.copy2(sample["image_path"], dst / "images" / new_file_name)
    shutil.copy2(sample["mask_path"], dst / "sem_masks" / new_file_name)

    label_payload = dict(sample["label"])
    label_payload.update(
        {
            "sample_id": new_stem,
            "file_name": new_file_name,
            "nipple_boxes_xyxy": sample["boxes"],
            "repeat_role": role,
            "repeat_index": repeat_index,
            "source_sample_id": sample["sample_id"],
            "source_file_name": sample["file_name"],
        }
    )
    _write_json(dst / "labels" / f"{new_stem}.json", label_payload)

    image_record = {
        "id": image_id,
        "file_name": new_file_name,
        "width": sample["width"],
        "height": sample["height"],
    }
    annotations: list[dict[str, Any]] = []
    ann_id = ann_id_start
    for box in sample["boxes"]:
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


def build_repeat_dataset(src: Path, dst: Path, repeat: int, overwrite: bool) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("--repeat must be at least 1.")
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
    samples = _load_source_samples(src)

    for sample in samples:
        for repeat_index in range(1, repeat + 1):
            new_stem = f"{sample['stem']}_tr{repeat_index:02d}"
            file_name, image_record, anns, label_payload, ann_id = _copy_sample(
                sample=sample,
                dst=dst,
                new_stem=new_stem,
                role="train",
                repeat_index=repeat_index,
                image_id=image_id,
                ann_id_start=ann_id,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_repeat_dataset(Path(args.src), Path(args.dst), args.repeat, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
