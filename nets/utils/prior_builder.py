"""Utility helpers to render prior tensors from boxes."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def build_prior_from_targets(
    gt_boxes: Optional[Tensor],
    height: int,
    width: int,
    sigma_ratio: float = 0.02,
) -> Tensor:
    """Create a 2xHxW tensor encoding target priors (gaussian + mask)."""
    return _render_prior(gt_boxes, height, width, sigma_ratio)


def build_prior_from_preds(
    pred_boxes: Optional[Tensor],
    height: int,
    width: int,
    sigma_ratio: float = 0.02,
) -> Tensor:
    """Create prior tensor from predicted boxes."""
    return _render_prior(pred_boxes, height, width, sigma_ratio)


def _render_prior(
    boxes: Optional[Tensor],
    height: int,
    width: int,
    sigma_ratio: float,
) -> Tensor:
    device = boxes.device if boxes is not None and boxes.numel() > 0 else torch.device("cpu")
    prior = torch.zeros((2, height, width), device=device)
    if boxes is None or boxes.numel() == 0:
        return prior

    boxes = boxes.to(device)
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    sigma = max(height, width) * sigma_ratio
    sigma_sq = sigma * sigma + 1e-6
    gaussian = prior[0]
    mask = prior[1]

    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        gaussian = torch.maximum(gaussian, torch.exp(-dist / (2 * sigma_sq)))

        x1_i = int(torch.clamp(torch.floor(x1), 0, width - 1))
        x2_i = int(torch.clamp(torch.ceil(x2), 0, width - 1))
        y1_i = int(torch.clamp(torch.floor(y1), 0, height - 1))
        y2_i = int(torch.clamp(torch.ceil(y2), 0, height - 1))
        mask[y1_i : y2_i + 1, x1_i : x2_i + 1] = 1.0

    prior[0] = gaussian
    prior[1] = mask
    return prior

