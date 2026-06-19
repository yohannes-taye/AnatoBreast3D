from __future__ import annotations

import json

import numpy as np

from recon3d.config import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthConfig,
    FusionConfig,
    PerceptionResult,
    Recon3DConfig,
    RGBDFrame,
)
from recon3d.pipeline import Recon3DPipeline


class _MaskOnlyPerception:
    def predict_batch(self, frames: dict[str, RGBDFrame]) -> dict[str, PerceptionResult]:
        return {
            cam_id: PerceptionResult(
                cam_id=cam_id,
                breast_mask=np.ones(frame.depth_mm.shape, dtype=bool),
                nipple_boxes_xyxy=np.empty((0, 4), dtype=np.float64),
                nipple_scores=np.empty((0,), dtype=np.float64),
                nipple_labels=None,
            )
            for cam_id, frame in frames.items()
        }


def test_pipeline_writes_expected_outputs(tmp_path) -> None:
    intr = CameraIntrinsics(width=8, height=6, fx=100.0, fy=100.0, cx=3.5, cy=2.5, distortion=None)
    depth_cfg = DepthConfig(depth_scale_m=0.001, min_depth_m=0.25, max_depth_m=2.0)
    rgb = np.full((6, 8, 3), 128, dtype=np.uint8)
    depth = np.full((6, 8), 1000, dtype=np.uint16)
    frames = {
        f"cam{i}": RGBDFrame(f"cam{i}", rgb, depth, 1_000_000 + i * 100_000, intr, depth_cfg)
        for i in range(3)
    }
    extrinsics = {cam_id: CameraExtrinsics(cam_id, np.eye(4, dtype=np.float64)) for cam_id in frames}
    config = Recon3DConfig(
        fusion=FusionConfig(voxel_size_m=0.005, outlier_nb_neighbors=3, outlier_std_ratio=2.0),
        output_dir=tmp_path,
    )

    result = Recon3DPipeline(config, extrinsics, perception=_MaskOnlyPerception()).run_once(frames)

    assert result.timings_ms["timestamp_drift_ms"] == 0.2
    assert (tmp_path / "fused_breast.ply").is_file()
    assert (tmp_path / "nipple_world.json").is_file()
    assert (tmp_path / "run_summary.json").is_file()
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["fused_cloud"]["points"] > 0
    assert summary["metrics"]["timestamp_drift_ms"] == 0.2


def test_pipeline_starts_viewer_when_enabled(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_start_viewer_server(**kwargs):
        calls.append(kwargs)
        return "http://127.0.0.1:8765/viewer/"

    monkeypatch.setattr("recon3d.pipeline.start_viewer_server", fake_start_viewer_server)
    config = Recon3DConfig(
        output_dir=tmp_path,
        enable_visualization=True,
        viewer_host="127.0.0.1",
        viewer_port=8899,
        viewer_open_browser=True,
    )

    pipeline = Recon3DPipeline(config, {}, perception=_MaskOnlyPerception())

    assert pipeline.viewer_url == "http://127.0.0.1:8765/viewer/"
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 8899,
            "output_dir": tmp_path,
            "open_browser": True,
        }
    ]
