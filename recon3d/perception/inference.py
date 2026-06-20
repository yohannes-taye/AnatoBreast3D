from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image

from recon3d.config import PerceptionConfig, PerceptionResult, RGBDFrame, Recon3DConfig


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.breastseg_infer import postprocess_mask, preprocess, remap_boxes  # noqa: E402
from tools.train_breastseg import build_model, load_config  # noqa: E402


LOGGER = logging.getLogger(__name__)


class BreastPerceptionAdapter:
    """Batch adapter around existing MammaSegNet + NippleFCOSHead inference."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        config: Recon3DConfig | PerceptionConfig | None = None,
    ) -> None:
        """Load the existing model without changing its behavior.

        Input shape/unit/frame: model consumes RGB tensors produced by the
        existing 512x512 letterbox preprocessing; output masks/boxes are remapped
        to aligned RGB/depth pixel coordinates.
        Return type: initialized adapter.
        """
        if config is None:
            perception_cfg = PerceptionConfig()
        elif isinstance(config, Recon3DConfig):
            perception_cfg = config.perception
        else:
            perception_cfg = config
        if checkpoint_path is not None:
            perception_cfg = PerceptionConfig(
                model_config_path=perception_cfg.model_config_path,
                checkpoint_path=Path(checkpoint_path),
                device=perception_cfg.device,
                score_thr=perception_cfg.score_thr,
                topk=perception_cfg.topk,
                nms_iou=perception_cfg.nms_iou,
            )
        if device is not None:
            perception_cfg = PerceptionConfig(
                model_config_path=perception_cfg.model_config_path,
                checkpoint_path=perception_cfg.checkpoint_path,
                device=device,
                score_thr=perception_cfg.score_thr,
                topk=perception_cfg.topk,
                nms_iou=perception_cfg.nms_iou,
            )
        self.config = perception_cfg
        self.model_cfg = load_config(str(perception_cfg.model_config_path))
        requested_device = perception_cfg.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for perception but unavailable; using CPU")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = build_model(self.model_cfg, self.device, init_from=None)
        checkpoint = torch.load(perception_cfg.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        self.model.eval()
        self.input_size = tuple(self.model_cfg["data"].get("input_size", (512, 512)))

    def predict_batch(self, frames: Mapping[str, RGBDFrame]) -> dict[str, PerceptionResult]:
        """Run existing RGB-only perception on a synchronized RGB-D frame set.

        Input shape/unit/frame: each frame rgb is HxWx3 uint8 RGB aligned to depth;
        depth is ignored by the model. Returned masks are HxW bool and boxes are
        Nx4 xyxy pixels in the original aligned RGB/depth coordinate system.
        Return type: mapping from camera ID to ``PerceptionResult``.
        """
        if not frames:
            return {}
        cam_ids = list(frames.keys())
        tensors: list[torch.Tensor] = []
        windows: dict[str, tuple[int, int]] = {}
        orig_sizes: dict[str, tuple[int, int]] = {}
        for cam_id in cam_ids:
            frame = frames[cam_id]
            image = Image.fromarray(frame.rgb, mode="RGB")
            tensor, window, orig_size = preprocess(image, self.input_size)
            tensors.append(tensor)
            windows[cam_id] = window
            orig_sizes[cam_id] = orig_size
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                batch,
                mode="predict",
                score_thr=self.config.score_thr,
                topk=self.config.topk,
                nms_iou=self.config.nms_iou,
            )

        results: dict[str, PerceptionResult] = {}
        for idx, cam_id in enumerate(cam_ids):
            output = outputs[idx]
            mask = postprocess_mask(output["seg_mask"], orig_sizes[cam_id], self.input_size, windows[cam_id])
            boxes = remap_boxes(output["detection"].boxes, orig_sizes[cam_id], self.input_size, windows[cam_id])
            scores = output["detection"].scores.detach().cpu().numpy().astype(np.float64, copy=True)
            results[cam_id] = PerceptionResult(
                cam_id=cam_id,
                breast_mask=(mask > 0),
                nipple_boxes_xyxy=boxes.astype(np.float64, copy=False),
                nipple_scores=scores,
                nipple_labels=None,
            )
        return results
