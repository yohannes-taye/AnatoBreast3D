"""Shared low-level augmentation primitives for breast image/mask/box triples.

These pure functions apply a single geometric transform **consistently** to an RGB
image, its semantic mask, and its xyxy detection boxes. They do no random sampling
themselves: callers own the parameter sampling and the border-fill / min-box policy.

Two callers share these primitives but compose them differently:

- On-the-fly training augmentation (``data_preprocess.breast_coco``) samples gentle
  ranges and uses ``mask_border=255`` (a transient *ignore* label that the loss
  skips after letterboxing).
- Offline dataset augmentation (``data_preprocess.augment_dataset``) samples stronger ranges
  and uses ``mask_border=0`` because the mask is **persisted to disk** as a base
  sample, where 255 would be baked in as a permanent ignore region.
"""

from __future__ import annotations

import cv2
import numpy as np


def transform_boxes_xyxy(
    boxes: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
    min_side: float = 1.0,
) -> np.ndarray:
    """Affine-transform xyxy boxes by their four corners, clip, drop degenerate.

    Input shape/unit/frame: ``boxes`` is (N, 4) float xyxy in pixels; ``matrix`` is a
    (2, 3) cv2 affine; ``width``/``height`` bound the output frame.
    Return type: (M, 4) float xyxy with both sides strictly greater than
    ``min_side`` pixels (M <= N).
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    num = boxes.shape[0]
    corners = np.array(
        [
            [boxes[:, 0], boxes[:, 1]],
            [boxes[:, 2], boxes[:, 1]],
            [boxes[:, 2], boxes[:, 3]],
            [boxes[:, 0], boxes[:, 3]],
        ]
    ).transpose(2, 0, 1).reshape(-1, 2)
    points = np.ones((num * 4, 3), dtype=np.float32)
    points[:, :2] = corners
    transformed = (points @ matrix.T).reshape(num, 4, 2)
    xs = transformed[:, :, 0]
    ys = transformed[:, :, 1]
    x1 = np.clip(xs.min(axis=1), 0, width - 1)
    y1 = np.clip(ys.min(axis=1), 0, height - 1)
    x2 = np.clip(xs.max(axis=1), 0, width - 1)
    y2 = np.clip(ys.max(axis=1), 0, height - 1)
    out = np.stack([x1, y1, x2, y2], axis=1)
    keep = ((x2 - x1) > min_side) & ((y2 - y1) > min_side)
    return out[keep]


def horizontal_flip(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror image, mask and xyxy boxes left-right.

    Return type: contiguous copies of ``image`` and ``mask`` plus mirrored boxes.
    """
    image = image[:, ::-1].copy()
    mask = mask[:, ::-1].copy()
    if boxes.size:
        width = image.shape[1]
        flipped = boxes.copy()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        boxes = flipped
    return image, mask, boxes


def affine_warp(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: np.ndarray,
    *,
    angle: float,
    scale: float,
    tx_px: float,
    ty_px: float,
    image_border: tuple[int, int, int] = (128, 128, 128),
    mask_border: int = 255,
    min_box_side: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate/scale/translate image (linear), mask (nearest) and boxes consistently.

    Input shape/unit/frame: ``angle`` degrees, ``scale`` factor, ``tx_px``/``ty_px``
    pixel translations about the image centre. ``image_border`` fills revealed image
    pixels; ``mask_border`` fills revealed mask pixels (255 ignore for training,
    0 background for persisted samples).
    Return type: warped image, mask, and surviving xyxy boxes.
    """
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx_px
    matrix[1, 2] += ty_px
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=image_border)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=mask_border)
    boxes = transform_boxes_xyxy(boxes, matrix, w, h, min_side=min_box_side)
    return image, mask, boxes
