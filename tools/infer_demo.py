"""Inference demo for MammaSegNet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from train_breastseg import build_model, load_config
from utils.utils import cvtColor, preprocess_input, resize_image


def list_images(path: str) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])


def preprocess(image: Image.Image, input_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    resized, nw, nh = resize_image(image, (input_size[1], input_size[0]))
    arr = np.array(resized, dtype=np.float32)
    tensor = torch.from_numpy(np.transpose(preprocess_input(arr), (2, 0, 1))).unsqueeze(0)
    return tensor, (nw, nh), image.size


def postprocess_mask(mask: torch.Tensor, orig_size: tuple[int, int], padded_shape: tuple[int, int], window: tuple[int, int]) -> np.ndarray:
    mask_np = mask.cpu().numpy().astype(np.uint8)
    h, w = padded_shape
    nw, nh = window
    pad_x = (w - nw) // 2
    pad_y = (h - nh) // 2
    crop = mask_np[pad_y : pad_y + nh, pad_x : pad_x + nw]
    crop = cv2.resize(crop, orig_size, interpolation=cv2.INTER_NEAREST)
    return crop


def remap_boxes(boxes: torch.Tensor, orig_size: tuple[int, int], padded_shape: tuple[int, int], window: tuple[int, int]) -> np.ndarray:
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.cpu().numpy()
    input_w, input_h = padded_shape[1], padded_shape[0]
    orig_w, orig_h = orig_size
    nw, nh = window
    scale = min(input_w / orig_w, input_h / orig_h)
    dx = (input_w - nw) / 2
    dy = (input_h - nh) / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h - 1)
    return boxes


def visualize(image: np.ndarray, mask: np.ndarray, boxes: np.ndarray, scores: torch.Tensor) -> np.ndarray:
    overlay = image.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.4
    overlay[mask == 1] = (alpha * color + (1 - alpha) * overlay[mask == 1]).astype(np.uint8)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        score = float(scores[idx].item()) if scores.numel() > idx else 0.0
        cv2.putText(
            overlay,
            f"{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a directory of images.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--save-dir", required=True, help="Directory to store visualizations.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold.")
    parser.add_argument("--topk", type=int, default=2, help="Top-k detections per image.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, init_from=None)
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(cfg["data"].get("input_size", (512, 512)))
    paths = list_images(args.input)
    for path in paths:
        image = cvtColor(Image.open(path))
        tensor, window, orig_size = preprocess(image, input_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            results = model(
                tensor,
                mode="predict",
                score_thr=args.score_thr,
                topk=args.topk,
                nms_iou=cfg["test"].get("nms_iou", 0.5),
            )
        result = results[0]
        mask = postprocess_mask(result["seg_mask"], orig_size, input_size, window)
        boxes = remap_boxes(result["detection"].boxes, orig_size, input_size, window)
        vis = visualize(np.array(image, dtype=np.uint8), mask, boxes, result["detection"].scores.cpu())
        out_path = save_dir / f"{path.stem}_vis.png"
        Image.fromarray(vis).save(out_path)
        print(f"[Infer] Saved {out_path}")


if __name__ == "__main__":
    main()

