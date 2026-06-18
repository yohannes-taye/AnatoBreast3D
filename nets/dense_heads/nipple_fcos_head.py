"""Anchor-free FCOS style head dedicated to nipple detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Compute sigmoid focal loss for binary classification."""
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_factor * loss
    return loss


def _box_area(boxes: Tensor) -> Tensor:
    """Return area of boxes shaped [..., 4]."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def _generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise GIoU for two tensors of boxes."""
    # IoU
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # smallest enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def _nms(boxes: Tensor, scores: Tensor, iou_thr: float) -> Tensor:
    """Simple NMS implementation that mirrors torchvision.ops.nms."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[order[1:]], x1[i])
        yy1 = torch.maximum(y1[order[1:]], y1[i])
        xx2 = torch.minimum(x2[order[1:]], x2[i])
        yy2 = torch.minimum(y2[order[1:]], y2[i])
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        rem_areas = areas[order[1:]]
        union = areas[i] + rem_areas - inter
        iou = inter / union.clamp(min=1e-6)
        inds = torch.nonzero(iou <= iou_thr).squeeze(1)
        order = order[inds + 1]
    return boxes.new_tensor(keep, dtype=torch.long)


@dataclass
class DetectionPred:
    """Structure describing batched detection outputs."""

    boxes: Tensor
    scores: Tensor


class NippleFCOSHead(nn.Module):
    """Single-class FCOS head that predicts nipple boxes."""

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        strides: Sequence[int] | None = None,
        score_threshold: float = 0.3,
        nms_iou: float = 0.5,
        topk: int = 2,
    ) -> None:
        super().__init__()
        self.strides = strides or (4, 8, 16)
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.topk = topk

        self.cls_tower = self._make_tower(in_channels, feat_channels)
        self.bbox_tower = self._make_tower(in_channels, feat_channels)
        self.cls_logits = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)
        self._init_weights()

    @staticmethod
    def _make_tower(in_channels: int, feat_channels: int) -> nn.Sequential:
        layers = []
        for _ in range(4):
            layers.extend(
                [
                    nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, feat_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            in_channels = feat_channels
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for layer in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, inputs: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Return raw cls/bbox/centerness features for each pyramid level."""
        cls_scores: list[Tensor] = []
        bbox_preds: list[Tensor] = []
        centerness: list[Tensor] = []
        for feat in inputs:
            cls_feat = self.cls_tower(feat)
            bbox_feat = self.bbox_tower(feat)
            cls_scores.append(self.cls_logits(cls_feat))
            bbox_preds.append(F.relu(self.bbox_pred(bbox_feat)))
            centerness.append(self.centerness(bbox_feat))
        return cls_scores, bbox_preds, centerness

    def loss(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        centerness: Sequence[Tensor],
        gt_boxes: list[Tensor | None],
    ) -> dict[str, Tensor]:
        """Compute FCOS losses for a batch of samples."""
        points, strides = self._points_and_strides(cls_scores)
        batch_size = cls_scores[0].shape[0]
        cls_flat, bbox_flat, ctr_flat = self._flatten(cls_scores, bbox_preds, centerness)

        total_loss_cls = cls_flat.new_tensor(0.0)
        total_loss_bbox = cls_flat.new_tensor(0.0)
        total_loss_ctr = cls_flat.new_tensor(0.0)
        total_pos = cls_flat.new_tensor(0.0)

        for batch_idx in range(batch_size):
            gt = gt_boxes[batch_idx]
            if gt is None:
                continue
            cls_targets, bbox_targets, ctr_targets, pos_mask = self._assign(
                points,
                gt,
            )
            logits = cls_flat[batch_idx]
            bbox_pred = bbox_flat[batch_idx]
            ctr_pred = ctr_flat[batch_idx]

            total_loss_cls = total_loss_cls + _sigmoid_focal_loss(logits, cls_targets).sum()

            if pos_mask.sum() == 0:
                continue

            pos_idx = torch.nonzero(pos_mask).squeeze(1)
            pos_points = points[pos_idx]
            pos_bbox_targets = bbox_targets[pos_idx]
            pos_cls_targets = cls_targets[pos_idx]
            pos_ctr_targets = ctr_targets[pos_idx]

            total_pos = total_pos + pos_cls_targets.sum()
            total_loss_ctr = total_loss_ctr + F.binary_cross_entropy_with_logits(
                ctr_pred[pos_idx], pos_ctr_targets, reduction="sum"
            )

            pred_boxes = self._distance_to_boxes(pos_points, bbox_pred[pos_idx])
            target_boxes = self._distance_to_boxes(pos_points, pos_bbox_targets)
            giou = _generalized_box_iou(pred_boxes, target_boxes).diag()
            total_loss_bbox = total_loss_bbox + (1 - giou).sum()

        norm = torch.clamp(total_pos, min=1.0)
        return {
            "loss_cls": total_loss_cls / norm,
            "loss_bbox": total_loss_bbox / norm,
            "loss_ctr": total_loss_ctr / norm,
        }

    def predict(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        centerness: Sequence[Tensor],
        image_sizes: Sequence[tuple[int, int]],
        score_thr: float | None = None,
        topk: int | None = None,
        nms_iou: float | None = None,
    ) -> list[DetectionPred]:
        """Convert FCOS logits into NMS filtered predictions."""
        score_thr = score_thr if score_thr is not None else self.score_threshold
        topk = topk if topk is not None else self.topk
        nms_iou = nms_iou if nms_iou is not None else self.nms_iou

        points, _ = self._points_and_strides(cls_scores)
        cls_flat, bbox_flat, ctr_flat = self._flatten(cls_scores, bbox_preds, centerness)
        results: list[DetectionPred] = []
        for batch_idx, image_size in enumerate(image_sizes):
            logits = torch.sigmoid(cls_flat[batch_idx]).squeeze(-1)
            ctr = torch.sigmoid(ctr_flat[batch_idx]).squeeze(-1)
            scores = logits * ctr

            bbox_abs = self._distance_to_boxes(points, bbox_flat[batch_idx])
            bbox_abs[:, 0::2] = bbox_abs[:, 0::2].clamp(0, image_size[1])
            bbox_abs[:, 1::2] = bbox_abs[:, 1::2].clamp(0, image_size[0])

            if bbox_abs.numel() == 0:
                results.append(DetectionPred(boxes=bbox_abs, scores=scores))
                continue

            kept_scores, kept_idx = torch.topk(scores, k=min(topk, scores.numel()))
            keep = kept_scores > score_thr
            kept_scores = kept_scores[keep]
            kept_idx = kept_idx[keep]
            if kept_scores.numel() == 0:
                results.append(DetectionPred(boxes=bbox_abs.new_zeros((0, 4)), scores=bbox_abs.new_zeros((0,))))
                continue

            boxes = bbox_abs[kept_idx]
            keep_idx = _nms(boxes, kept_scores, nms_iou)
            results.append(DetectionPred(boxes=boxes[keep_idx], scores=kept_scores[keep_idx]))
        return results

    def _points_and_strides(self, cls_scores: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """Generate spatial point coordinates and stride tensor."""
        all_points = []
        all_strides = []
        device = cls_scores[0].device
        for level, stride in enumerate(self.strides):
            _, _, h, w = cls_scores[level].shape
            ys = torch.arange(h, device=device) + 0.5
            xs = torch.arange(w, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1) * stride
            all_points.append(points)
            all_strides.append(points.new_full((points.shape[0],), stride))
        return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)

    @staticmethod
    def _flatten(
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        centerness: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Flatten each prediction head from multi-level maps."""
        cls_batch = []
        bbox_batch = []
        ctr_batch = []
        for cls_feat, bbox_feat, ctr_feat in zip(cls_scores, bbox_preds, centerness):
            cls_batch.append(cls_feat.permute(0, 2, 3, 1).reshape(cls_feat.shape[0], -1, 1))
            bbox_batch.append(bbox_feat.permute(0, 2, 3, 1).reshape(bbox_feat.shape[0], -1, 4))
            ctr_batch.append(ctr_feat.permute(0, 2, 3, 1).reshape(ctr_feat.shape[0], -1, 1))
        return torch.cat(cls_batch, dim=1), torch.cat(bbox_batch, dim=1), torch.cat(ctr_batch, dim=1)

    @staticmethod
    def _distance_to_boxes(points: Tensor, distances: Tensor) -> Tensor:
        """Convert distances (l,t,r,b) with respect to points into absolute boxes."""
        x = points[:, 0]
        y = points[:, 1]
        l = distances[:, 0]
        t = distances[:, 1]
        r = distances[:, 2]
        b = distances[:, 3]
        x1 = x - l
        y1 = y - t
        x2 = x + r
        y2 = y + b
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _assign(
        self,
        points: Tensor,
        gt_boxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign each point to the nearest GT box."""
        device = points.device
        num_points = points.shape[0]
        cls_targets = torch.zeros((num_points, 1), device=device)
        bbox_targets = torch.zeros((num_points, 4), device=device)
        ctr_targets = torch.zeros((num_points, 1), device=device)
        pos_mask = torch.zeros((num_points,), dtype=torch.bool, device=device)

        if gt_boxes is None or gt_boxes.numel() == 0:
            return cls_targets, bbox_targets, ctr_targets, pos_mask

        boxes = gt_boxes.to(device)
        areas = _box_area(boxes)
        diff = torch.stack(
            (
                points[:, None, 0] - boxes[None, :, 0],
                points[:, None, 1] - boxes[None, :, 1],
                boxes[None, :, 2] - points[:, None, 0],
                boxes[None, :, 3] - points[:, None, 1],
            ),
            dim=-1,
        )
        inside = diff.min(dim=-1).values > 0
        if not inside.any():
            return cls_targets, bbox_targets, ctr_targets, pos_mask
        areas = areas.expand(num_points, -1).clone()
        areas[~inside] = float("inf")
        min_area, min_idx = areas.min(dim=1)
        assigned = torch.isfinite(min_area)
        pos_mask[assigned] = True
        bbox_targets[assigned] = diff[assigned, min_idx[assigned]]
        cls_targets[assigned] = 1.0

        l, t, r, b = bbox_targets[assigned].unbind(dim=-1)
        lr = torch.minimum(l, r) / torch.maximum(l, r).clamp(min=1e-6)
        tb = torch.minimum(t, b) / torch.maximum(t, b).clamp(min=1e-6)
        ctr = torch.sqrt(lr * tb)
        ctr_targets[assigned] = ctr.unsqueeze(-1)
        return cls_targets, bbox_targets, ctr_targets, pos_mask
