"""Dataset unit tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_loaders.breast_coco import BreastCocoDataset, breast_collate_fn


def create_dummy_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "sem_masks").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)

    images = []
    annotations = []
    for idx in range(2):
        img_path = root / "images" / f"{idx}.jpg"
        mask_path = root / "sem_masks" / f"{idx}.png"
        Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(img_path)
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(mask_path)
        images.append({"id": idx, "file_name": f"{idx}.jpg", "width": 64, "height": 64})
        annotations.append({"id": idx, "image_id": idx, "bbox": [10, 10, 20, 20], "category_id": 1})

    with open(root / "annotations" / "coco_det.json", "w", encoding="utf-8") as file:
        json.dump({"images": images, "annotations": annotations}, file)

    (root / "splits" / "train.txt").write_text("0.jpg\n1.jpg\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("0.jpg\n", encoding="utf-8")


def test_dataset_and_collate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        create_dummy_dataset(root)
        dataset = BreastCocoDataset(str(root), split="train", input_size=(64, 64), train=True)
        sample = dataset[0]
        assert isinstance(sample.image, torch.Tensor)
        assert sample.seg_valid
        loader_batch = breast_collate_fn([sample])
        assert loader_batch["images"].shape[0] == 1
        assert isinstance(loader_batch["gt_boxes"][0], torch.Tensor)
