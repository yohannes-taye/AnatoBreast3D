from __future__ import annotations

import base64
import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DATASET_DIR = Path("active_learning") / "dataset"


def _decode_data_url(value: str) -> bytes:
    if "," in value:
        _, value = value.split(",", 1)
    return base64.b64decode(value)


def _load_rgb(data_url: str) -> Image.Image:
    return Image.open(BytesIO(_decode_data_url(data_url))).convert("RGB")


def _load_mask(data_url: str, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(BytesIO(_decode_data_url(data_url))).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _ensure_dataset(root: Path) -> dict[str, Path]:
    paths = {
        "root": root / DATASET_DIR,
        "images": root / DATASET_DIR / "images",
        "masks": root / DATASET_DIR / "sem_masks",
        "ann": root / DATASET_DIR / "annotations",
        "splits": root / DATASET_DIR / "splits",
        "labels": root / DATASET_DIR / "labels",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _read_coco(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "nipple"}]}


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    x2 = max(0.0, min(width - 1.0, x2))
    y1 = max(0.0, min(height - 1.0, y1))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2, y2]


def _refresh_splits(paths: dict[str, Path]) -> list[str]:
    names = sorted(path.name for path in paths["images"].glob("*.png"))
    text = "\n".join(names)
    paths["splits"].joinpath("train.txt").write_text(text, encoding="utf-8")
    paths["splits"].joinpath("val.txt").write_text(text, encoding="utf-8")
    return names


def dataset_summary(project_root: Path) -> dict[str, Any]:
    paths = _ensure_dataset(project_root)
    coco = _read_coco(paths["ann"] / "coco_det.json")
    names = sorted(path.name for path in paths["images"].glob("*.png"))
    return {
        "dataset_dir": str(paths["root"]),
        "samples": len(names),
        "boxes": len(coco.get("annotations", [])),
        "latest": names[-1] if names else None,
    }


def save_annotation(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    paths = _ensure_dataset(project_root)
    camera_id = str(payload.get("camera_id") or "cam")
    frame = int(payload.get("frame") or 0)
    timestamp = float(payload.get("timestamp") or time.time())
    sample_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{camera_id}_f{frame:06d}_{uuid.uuid4().hex[:8]}"
    file_name = f"{sample_id}.png"

    image = _load_rgb(str(payload["image_png"]))
    width, height = image.size
    mask = _load_mask(str(payload["mask_png"]), image.size)
    boxes = [
        box
        for box in (
            _clamp_box(list(raw_box), width, height)
            for raw_box in payload.get("nipple_boxes_xyxy", [])
        )
        if box is not None
    ]

    image.save(paths["images"] / file_name)
    mask.save(paths["masks"] / file_name)

    coco_path = paths["ann"] / "coco_det.json"
    coco = _read_coco(coco_path)
    next_image_id = max([0] + [int(item["id"]) for item in coco.get("images", [])]) + 1
    next_ann_id = max([0] + [int(item["id"]) for item in coco.get("annotations", [])]) + 1
    coco.setdefault("images", []).append(
        {"id": next_image_id, "file_name": file_name, "width": width, "height": height}
    )
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        coco.setdefault("annotations", []).append(
            {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )
        next_ann_id += 1
    coco["categories"] = [{"id": 1, "name": "nipple"}]
    _write_json(coco_path, coco)

    label_payload = {
        "sample_id": sample_id,
        "file_name": file_name,
        "camera_id": camera_id,
        "frame": frame,
        "timestamp": timestamp,
        "nipple_boxes_xyxy": boxes,
        "source_manifest": payload.get("source_manifest", {}),
    }
    _write_json(paths["labels"] / f"{sample_id}.json", label_payload)
    with (paths["root"] / "manifest.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(label_payload, ensure_ascii=False) + "\n")

    names = _refresh_splits(paths)
    return {
        "ok": True,
        "sample_id": sample_id,
        "file_name": file_name,
        "samples": len(names),
        "boxes": len(coco.get("annotations", [])),
        "dataset_dir": str(paths["root"]),
    }
