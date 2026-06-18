"""Smoke tests for BreastSegFormer forward paths."""

from __future__ import annotations

import torch

from nets.segmentors.breastseg_multitask import BreastSegFormer


def test_forward_train_and_predict() -> None:
    model = BreastSegFormer(num_classes=2, phi="b0", init_from=None)
    model.train()
    images = torch.randn(2, 3, 128, 128)
    gt_seg = torch.zeros(2, 128, 128, dtype=torch.long)
    boxes = [
        torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
        torch.empty((0, 4), dtype=torch.float32),
    ]
    seg_valid = torch.tensor([True, False])
    det_valid = torch.tensor([True, True])
    losses, aux = model(
        images,
        gt_seg=gt_seg,
        gt_boxes=boxes,
        seg_valid=seg_valid,
        det_valid=det_valid,
        mode="loss",
    )
    assert "loss_total" in losses
    assert torch.isfinite(losses["loss_total"])
    assert "seg_logits" in aux

    model.eval()
    outputs = model(images, mode="predict")
    assert isinstance(outputs, list)
    assert "seg_mask" in outputs[0]
    assert "detection" in outputs[0]

