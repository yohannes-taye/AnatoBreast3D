"""Export BreastSegFormer to ONNX."""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from train_breastseg import build_model, load_config


class OnnxWrapper(torch.nn.Module):
    """Thin wrapper that exposes raw segmentation and detection logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.model.backbone(x)
        pyramid = self.model.neck(features)
        cls_scores, bbox_preds, centerness = self.model.det_head(pyramid[: len(self.model.det_head.strides)])
        seg_logits = self.model.decode_head(features, prior=None)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        cls_flat = torch.cat([feat.flatten(2) for feat in cls_scores], dim=2)
        bbox_flat = torch.cat([feat.flatten(2) for feat in bbox_preds], dim=2)
        ctr_flat = torch.cat([feat.flatten(2) for feat in centerness], dim=2)
        return seg_logits, cls_flat, bbox_flat, ctr_flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX graph.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--onnx", required=True, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, init_from=None)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    model.eval()
    wrapper = OnnxWrapper(model).to(device)

    input_size = tuple(cfg["data"].get("input_size", (512, 512)))
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["segmentation", "cls_logits", "bbox_preds", "centerness"],
        dynamic_axes={"images": {0: "batch"}, "segmentation": {0: "batch"}},
    )
    print(f"[Export] Saved ONNX model to {args.onnx}")


if __name__ == "__main__":
    main()

