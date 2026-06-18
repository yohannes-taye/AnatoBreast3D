"""Breast segmentation dataset with paired COCO-style boxes and masks."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.utils import cvtColor, preprocess_input

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
LOGGER = logging.getLogger(__name__)


@dataclass
class BreastSample:
    """Container returned by the dataset for collate_fn convenience."""

    image: Tensor
    gt_seg: Tensor
    gt_boxes: Tensor
    seg_valid: torch.BoolTensor
    det_valid: torch.BoolTensor
    image_id: str
    orig_size: Tuple[int, int]


class BreastCocoDataset(Dataset):
    """Loads paired semantic masks and detection boxes."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        input_size: Tuple[int, int] = (512, 512),
        multi_scale: Tuple[int, int] = (384, 640),
        train: bool = True,
        repeat_samples: Optional[Dict | List[Dict]] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.train = train
        self.image_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "sem_masks")
        self.split_file = os.path.join(data_root, "splits", f"{split}.txt")
        self.ann_path = os.path.join(data_root, "annotations", "coco_det.json")

        with open(self.split_file, "r", encoding="utf-8") as file:
            self.sample_ids = [line.strip() for line in file if line.strip()]
        if not self.sample_ids:
            raise ValueError(f"Split file {self.split_file} is empty.")

        # Optionally up-weight specific samples by duplicating them in the training list.
        # This keeps validation untouched because we only apply it when `train=True`.
        # Support both single dict and list of dicts for multiple repeat configurations
        if self.train and repeat_samples:
            # Convert single dict to list for uniform processing
            repeat_configs = [repeat_samples] if isinstance(repeat_samples, dict) else repeat_samples

            for repeat_cfg in repeat_configs:
                repeats = max(1, int(repeat_cfg.get("repeat", 1)))
                targets = set(repeat_cfg.get("ids", []))
                if repeats > 1 and targets:
                    extra = [sid for sid in self.sample_ids if sid in targets for _ in range(repeats - 1)]
                    if extra:
                        self.sample_ids.extend(extra)
                        LOGGER.info("Repeated %d samples x%d (total +%d).", len(targets), repeats, len(extra))

            # Shuffle after all repeats are applied
            with open(self.split_file, "r", encoding="utf-8") as file:
                original_count = len([line.strip() for line in file if line.strip()])
            if len(self.sample_ids) > original_count:
                random.shuffle(self.sample_ids)

        with open(self.ann_path, "r", encoding="utf-8") as file:
            coco = json.load(file)
        self.image_meta = {img["id"]: img for img in coco.get("images", [])}
        self.name_to_id = {img["file_name"]: img["id"] for img in coco.get("images", [])}
        self.boxes_by_image: Dict[str, List[List[float]]] = {}
        for ann in coco.get("annotations", []):
            image_id = ann["image_id"]
            img_info = self.image_meta.get(image_id)
            if not img_info:
                continue
            fname = img_info["file_name"]
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            self.boxes_by_image.setdefault(fname, []).append([x1, y1, x1 + w, y1 + h])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> BreastSample:
        sample_id = self.sample_ids[index]
        image_path = self._resolve_path(self.image_dir, sample_id)
        mask_path = self._resolve_path(self.mask_dir, sample_id, exts=(".png",))
        seg_valid = os.path.exists(mask_path)

        image = cvtColor(cv2.imread(image_path)[:, :, ::-1])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if seg_valid else None
        orig_size = (image.shape[0], image.shape[1])

        file_key = os.path.relpath(image_path, self.image_dir).replace("\\", "/")
        boxes = self.boxes_by_image.get(file_key, [])
        det_valid = torch.tensor(bool(file_key in self.boxes_by_image), dtype=torch.bool)
        boxes_array = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

        image, mask, boxes_array = self._augment(image, mask, boxes_array)
        tensor_image = torch.from_numpy(np.transpose(preprocess_input(image.astype(np.float32)), (2, 0, 1))).float()
        tensor_mask = torch.from_numpy(mask.astype(np.int64)) if mask is not None else torch.full(
            self.input_size, 255, dtype=torch.int64
        )
        tensor_boxes = torch.from_numpy(boxes_array).float()

        return BreastSample(
            image=tensor_image,
            gt_seg=tensor_mask,
            gt_boxes=tensor_boxes,
            seg_valid=torch.tensor(seg_valid, dtype=torch.bool),
            det_valid=det_valid,
            image_id=sample_id,
            orig_size=orig_size,
        )

    def _resolve_path(self, root: str, sample_id: str, exts: Sequence[str] = IMG_EXTENSIONS) -> str:
        base = Path(root) / sample_id
        if base.exists():
            return str(base)
        stem = base.stem
        for ext in exts:
            candidate = Path(root) / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Unable to locate file for id '{sample_id}' in {root}.")

    def _augment(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = image.copy()
        h, w = image.shape[:2]
        mask = mask.copy() if mask is not None else np.full((h, w), 255, dtype=np.uint8)

        if self.train:
            short = min(h, w)
            target_short = random.randint(self.multi_scale[0], self.multi_scale[1])
            scale = target_short / max(short, 1)
            image, mask, boxes = self._resize(image, mask, boxes, scale)
            image, mask, boxes = self._random_flip(image, mask, boxes)
            image, mask, boxes = self._random_affine(image, mask, boxes)
            image = self._color_jitter(image)

        image, mask, boxes = self._letterbox(image, mask, boxes)
        return image, mask, boxes

    def _resize(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boxes: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        boxes = boxes * scale
        return image, mask, boxes

    def _random_flip(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.train or random.random() > 0.5:
            return image, mask, boxes
        image = image[:, ::-1]
        mask = mask[:, ::-1]
        if boxes.size:
            width = image.shape[1]
            flipped = boxes.copy()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            boxes = flipped
        return image, mask, boxes

    def _random_affine(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.train:
            return image, mask, boxes
        h, w = image.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        center = (w / 2, h / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        if boxes.size:
            boxes = self._transform_boxes(boxes, matrix, w, h)
        return image, mask, boxes

    def _transform_boxes(self, boxes: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
        num_boxes = boxes.shape[0]
        points = np.ones((num_boxes * 4, 3), dtype=np.float32)
        corners = np.array(
            [
                [boxes[:, 0], boxes[:, 1]],
                [boxes[:, 2], boxes[:, 1]],
                [boxes[:, 2], boxes[:, 3]],
                [boxes[:, 0], boxes[:, 3]],
            ]
        ).transpose(2, 0, 1).reshape(-1, 2)
        points[:, :2] = corners
        transformed = points @ matrix.T
        transformed = transformed.reshape(num_boxes, 4, 2)
        xs = transformed[:, :, 0]
        ys = transformed[:, :, 1]
        x1 = np.clip(xs.min(axis=1), 0, width - 1)
        y1 = np.clip(ys.min(axis=1), 0, height - 1)
        x2 = np.clip(xs.max(axis=1), 0, width - 1)
        y2 = np.clip(ys.max(axis=1), 0, height - 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        valid = (x2 - x1 > 1) & (y2 - y1 > 1)
        return boxes[valid]

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        if not self.train:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-5, 5)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.9, 1.1), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.9, 1.1), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _letterbox(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        mask_canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized_img
        mask_canvas[dy : dy + new_h, dx : dx + new_w] = resized_mask

        if boxes.size:
            boxes = boxes * scale
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, target_w - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, target_h - 1)
        return canvas, mask_canvas, boxes


def breast_collate_fn(batch: Sequence[BreastSample]) -> Dict[str, object]:
    """Custom collate function for variable-size box tensors."""
    images = torch.stack([item.image for item in batch], dim=0)
    masks = torch.stack([item.gt_seg for item in batch], dim=0)
    boxes = [item.gt_boxes for item in batch]
    seg_valid = torch.stack([item.seg_valid for item in batch])
    det_valid = torch.stack([item.det_valid for item in batch])
    meta = [{"image_id": item.image_id, "orig_size": item.orig_size} for item in batch]
    return {
        "images": images,
        "gt_seg": masks,
        "gt_boxes": boxes,
        "seg_valid": seg_valid,
        "det_valid": det_valid,
        "meta": meta,
    }
