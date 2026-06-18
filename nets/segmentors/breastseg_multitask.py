"""Multi-task SegFormer model for breast segmentation and nipple detection."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nets.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from nets.decode_heads.segformer_head_with_prior import SegFormerHeadWithPrior
from nets.dense_heads.nipple_fcos_head import DetectionPred, NippleFCOSHead
from nets.necks.light_fpn import LightFPN
from nets.utils.prior_builder import build_prior_from_preds, build_prior_from_targets


BACKBONE_CFG = {
    "b0": {"channels": [32, 64, 160, 256], "embed": 256, "factory": mit_b0},
    "b1": {"channels": [64, 128, 320, 512], "embed": 256, "factory": mit_b1},
    "b2": {"channels": [64, 128, 320, 512], "embed": 768, "factory": mit_b2},
    "b3": {"channels": [64, 128, 320, 512], "embed": 768, "factory": mit_b3},
    "b4": {"channels": [64, 128, 320, 512], "embed": 768, "factory": mit_b4},
    "b5": {"channels": [64, 128, 320, 512], "embed": 768, "factory": mit_b5},
}


class BreastSegFormer(nn.Module):
    """Encapsulates backbone, decode head, and detection head."""

    def __init__(
        self,
        num_classes: int = 2,
        phi: str = "b2",
        init_from: Optional[str] = None,
        use_prior: bool = True,
        loss_weights: Optional[Dict[str, float]] = None,
        sigma_ratio: float = 0.02,
        det_head_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if phi not in BACKBONE_CFG:
            raise ValueError(f"Unsupported backbone '{phi}'.")
        cfg = BACKBONE_CFG[phi]
        self.backbone = cfg["factory"](pretrained=False)
        self.decode_head = SegFormerHeadWithPrior(
            num_classes=num_classes,
            in_channels=cfg["channels"],
            embedding_dim=cfg["embed"],
            use_prior=use_prior,
        )
        self.neck = LightFPN(cfg["channels"], out_channels=256)
        det_cfg = det_head_cfg or {}
        self.det_head = NippleFCOSHead(
            in_channels=256,
            feat_channels=256,
            strides=det_cfg.get("strides", (4, 8, 16)),
            score_threshold=det_cfg.get("score_thr", 0.3),
            nms_iou=det_cfg.get("nms_iou", 0.5),
            topk=det_cfg.get("topk", 2),
        )

        self.num_classes = num_classes
        self.sigma_ratio = sigma_ratio
        self.seg_loss_weight = (loss_weights or {}).get("seg", 1.0)
        self.det_loss_weight = (loss_weights or {}).get("det", 1.0)
        self.boundary_loss_weight = (loss_weights or {}).get("boundary", 0.2)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("boundary_kernel", kernel)

        if init_from:
            self.load_init_from_checkpoint(init_from)

    def forward(
        self,
        images: Tensor,
        gt_seg: Optional[Tensor] = None,
        gt_boxes: Optional[List[Optional[Tensor]]] = None,
        seg_valid: Optional[Tensor] = None,
        det_valid: Optional[Tensor] = None,
        mode: str = "loss",
        **kwargs: Dict,
    ):
        """Route to train or inference branches."""
        mode = mode or ("loss" if self.training else "predict")
        if mode == "loss":
            return self.forward_train(images, gt_seg, gt_boxes, seg_valid, det_valid)
        if mode == "predict":
            return self.simple_test(
                images,
                score_thr=kwargs.get("score_thr"),
                topk=kwargs.get("topk"),
                nms_iou=kwargs.get("nms_iou"),
            )
        raise ValueError(f"Unsupported mode '{mode}'.")

    def forward_train(
        self,
        images: Tensor,
        gt_seg: Optional[Tensor],
        gt_boxes: Optional[List[Optional[Tensor]]],
        seg_valid: Optional[Tensor],
        det_valid: Optional[Tensor],
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward with loss computation."""
        features = self.backbone(images)
        pyramid = self.neck(features)
        det_features = [feat for feat in pyramid[: len(self.det_head.strides)]]

        gt_boxes = gt_boxes or [None] * images.shape[0]
        det_valid = det_valid if det_valid is not None else torch.ones(len(gt_boxes), dtype=torch.bool, device=images.device)
        det_boxes_prepared: List[Optional[Tensor]] = []
        for idx, valid in enumerate(det_valid):
            if not bool(valid):
                det_boxes_prepared.append(None)
                continue
            boxes = gt_boxes[idx]
            det_boxes_prepared.append(boxes.to(images.device) if boxes is not None else torch.empty((0, 4), device=images.device))

        cls_scores, bbox_preds, centerness = self.det_head(det_features)
        det_losses = self.det_head.loss(cls_scores, bbox_preds, centerness, det_boxes_prepared)

        priors = self._build_batch_prior(det_boxes_prepared, images.shape[-2:], images.device)
        seg_logits = self.decode_head(features, prior=priors)
        seg_logits = F.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=True)
        seg_losses = self._compute_segmentation_losses(seg_logits, gt_seg, seg_valid)

        losses: Dict[str, Tensor] = {}
        if seg_losses is not None:
            losses["loss_seg"] = seg_losses["seg"] * self.seg_loss_weight
            if "boundary" in seg_losses:
                losses["loss_boundary"] = seg_losses["boundary"] * self.boundary_loss_weight
        if det_losses:
            losses["loss_det"] = (
                det_losses["loss_cls"] + det_losses["loss_bbox"] + det_losses["loss_ctr"]
            ) * self.det_loss_weight
        total_loss = torch.stack([value for value in losses.values()]) if losses else seg_logits.new_tensor(0.0)
        losses["loss_total"] = total_loss.sum() if losses else seg_logits.sum() * 0
        aux = {"seg_logits": seg_logits.detach()}
        return losses, aux

    @torch.no_grad()
    def simple_test(
        self,
        images: Tensor,
        score_thr: Optional[float] = None,
        topk: Optional[int] = None,
        nms_iou: Optional[float] = None,
    ) -> List[Dict[str, Tensor | DetectionPred]]:
        """Inference that outputs segmentation mask and detection predictions."""
        features = self.backbone(images)
        pyramid = self.neck(features)
        det_features = [feat for feat in pyramid[: len(self.det_head.strides)]]
        cls_scores, bbox_preds, centerness = self.det_head(det_features)
        sizes = [(images.shape[2], images.shape[3]) for _ in range(images.shape[0])]
        det_outputs = self.det_head.predict(
            cls_scores,
            bbox_preds,
            centerness,
            sizes,
            score_thr=score_thr,
            topk=topk,
            nms_iou=nms_iou,
        )

        priors = []
        for pred in det_outputs:
            boxes = pred.boxes if pred.boxes.numel() > 0 else None
            prior = build_prior_from_preds(boxes, images.shape[2], images.shape[3], self.sigma_ratio)
            priors.append(prior.to(images.device))
        priors_tensor = torch.stack(priors, dim=0) if priors else None
        seg_logits = self.decode_head(features, prior=priors_tensor)
        seg_logits = F.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=True)
        seg_masks = torch.argmax(seg_logits, dim=1)
        results: List[Dict[str, Tensor | DetectionPred]] = []
        for idx, pred in enumerate(det_outputs):
            results.append(
                {
                    "seg_mask": seg_masks[idx],
                    "logits": seg_logits[idx],
                    "detection": pred,
                }
            )
        return results

    def _build_batch_prior(
        self,
        gt_boxes: List[Optional[Tensor]],
        size_hw: Tuple[int, int],
        device: torch.device,
    ) -> Optional[Tensor]:
        if not gt_boxes:
            return None
        priors = []
        height, width = size_hw
        for boxes in gt_boxes:
            if boxes is None or boxes.numel() == 0:
                priors.append(torch.zeros((2, height, width), device=device))
            else:
                priors.append(build_prior_from_targets(boxes, height, width, self.sigma_ratio).to(device))
        return torch.stack(priors, dim=0)

    def _compute_segmentation_losses(
        self,
        logits: Tensor,
        gt_seg: Optional[Tensor],
        seg_valid: Optional[Tensor],
    ) -> Optional[Dict[str, Tensor]]:
        if gt_seg is None:
            return None
        batch_size = logits.shape[0]
        valid_mask = seg_valid if seg_valid is not None else torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        valid_idx = torch.nonzero(valid_mask).squeeze(1)
        if valid_idx.numel() == 0:
            return None

        ce_losses = []
        dice_losses = []
        boundary_losses = []
        for idx in valid_idx.tolist():
            ce_losses.append(self.ce_loss(logits[idx : idx + 1], gt_seg[idx : idx + 1]))
            dice_losses.append(self._dice_loss(logits[idx : idx + 1], gt_seg[idx : idx + 1]))
            boundary_losses.append(self._boundary_loss(logits[idx : idx + 1], gt_seg[idx : idx + 1]))
        seg_loss = torch.stack(ce_losses).mean() + torch.stack(dice_losses).mean()
        boundary_loss = torch.stack(boundary_losses).mean()
        return {"seg": seg_loss, "boundary": boundary_loss}

    def _dice_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.clone()
        ignore_mask = (target != 255).float().unsqueeze(1)
        target[target == 255] = 0
        target_oh = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        prob = torch.softmax(logits, dim=1)
        prob = prob * ignore_mask
        target_oh = target_oh * ignore_mask
        intersection = (prob * target_oh).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target_oh.sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        return dice.mean()

    def _boundary_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.clone()
        target[target == 255] = 0
        fg = (target == 1).float().unsqueeze(1)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        target_edge = torch.abs(F.conv2d(fg, self.boundary_kernel, padding=1))
        pred_edge = torch.abs(F.conv2d(prob, self.boundary_kernel, padding=1))
        return F.l1_loss(pred_edge, target_edge)

    def load_init_from_checkpoint(self, path: str) -> None:
        """Load pretrained MiT weights with strict=False."""
        if not path or not os.path.exists(path):
            print(f"[BreastSegFormer] Init path '{path}' not found, skipping.")
            return
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = {}
        skips = 0
        model_state = self.state_dict()
        for key, value in state_dict.items():
            name = key.replace("module.", "")
            if name in model_state and model_state[name].shape == value.shape:
                cleaned[name] = value
            else:
                skips += 1
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        print(
            f"[BreastSegFormer] Loaded {len(cleaned)} keys (skipped={skips}) from '{path}' "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
