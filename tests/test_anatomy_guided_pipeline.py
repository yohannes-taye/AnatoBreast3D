from __future__ import annotations

import json

import numpy as np

from anatomy_guided_recon3d import pipeline as pipeline_module
from anatomy_guided_recon3d.config import (
    AnatomyRegistrationConfig,
    CameraExtrinsics,
    CameraIntrinsics,
    DepthConfig,
    FusionConfig,
    NippleGuidedFusionConfig,
    NippleObservation,
    PerceptionResult,
    PostFusionCleanupConfig,
    QualityMetricsConfig,
    Recon3DConfig,
    RGBDFrame,
    TSDFConfig,
)
from anatomy_guided_recon3d.pipeline import Recon3DPipeline


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


def test_anatomy_pipeline_writes_quality_and_tsdf_diagnostics(tmp_path) -> None:
    intr = CameraIntrinsics(width=16, height=12, fx=120.0, fy=120.0, cx=7.5, cy=5.5, distortion=None)
    depth_cfg = DepthConfig(depth_scale_m=0.001, min_depth_m=0.25, max_depth_m=2.0)
    rgb = np.full((12, 16, 3), 160, dtype=np.uint8)
    depth = np.full((12, 16), 900, dtype=np.uint16)
    frames = {
        cam_id: RGBDFrame(cam_id, rgb, depth, 1_000_000 + index * 100_000, intr, depth_cfg)
        for index, cam_id in enumerate(("cam0", "cam1"))
    }
    extrinsics = {cam_id: CameraExtrinsics(cam_id, np.eye(4, dtype=np.float64)) for cam_id in frames}
    config = Recon3DConfig(
        fusion=FusionConfig(voxel_size_m=0.005, outlier_nb_neighbors=3, outlier_std_ratio=2.0),
        post_fusion_cleanup=PostFusionCleanupConfig(enabled=False),
        quality_metrics=QualityMetricsConfig(
            enabled=True,
            min_points=10,
            max_points_per_cloud=100,
            temporal_window=3,
        ),
        tsdf_fusion=TSDFConfig(enabled=False),
        output_dir=tmp_path,
    )

    result = Recon3DPipeline(config, extrinsics, perception=_MaskOnlyPerception()).run_once(frames)

    assert len(result.fused_cloud.points) > 0
    assert (tmp_path / "quality_metrics.json").is_file()
    assert (tmp_path / "tsdf_fusion.json").is_file()
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_metrics"]["surface_consistency"]["valid"] is True
    assert summary["tsdf_fusion"]["enabled"] is False
    assert summary["tsdf_fusion"]["source"] == "pointcloud"


def test_pipeline_passes_assigned_nipple_sides_to_guided_fusion(tmp_path, monkeypatch) -> None:
    intr = CameraIntrinsics(width=16, height=12, fx=120.0, fy=120.0, cx=7.5, cy=5.5, distortion=None)
    depth_cfg = DepthConfig(depth_scale_m=0.001, min_depth_m=0.25, max_depth_m=2.0)
    rgb = np.full((12, 16, 3), 160, dtype=np.uint8)
    depth = np.full((12, 16), 900, dtype=np.uint16)
    frames = {
        cam_id: RGBDFrame(cam_id, rgb, depth, 1_000_000 + index * 100_000, intr, depth_cfg)
        for index, cam_id in enumerate(("cam0", "cam1"))
    }
    extrinsics = {cam_id: CameraExtrinsics(cam_id, np.eye(4, dtype=np.float64)) for cam_id in frames}

    def fake_estimate_nipples(*args, **kwargs):
        cov = np.eye(3, dtype=np.float64) * 1e-6
        return [
            NippleObservation("cam0", "unknown", np.array([-0.03, 0.0, 1.0]), cov, 0.9, 50),
            NippleObservation("cam0", "unknown", np.array([0.03, 0.0, 1.0]), cov, 0.9, 50),
            NippleObservation("cam1", "unknown", np.array([-0.031, 0.0, 1.0]), cov, 0.9, 50),
            NippleObservation("cam1", "unknown", np.array([0.031, 0.0, 1.0]), cov, 0.9, 50),
        ]

    monkeypatch.setattr(pipeline_module, "estimate_nipple_observations", fake_estimate_nipples)
    config = Recon3DConfig(
        anatomy_registration=AnatomyRegistrationConfig(mode="translation_only"),
        fusion=FusionConfig(voxel_size_m=0.005, outlier_nb_neighbors=3, outlier_std_ratio=2.0),
        nipple_guided_fusion=NippleGuidedFusionConfig(missing_nipple_weight=0.55),
        post_fusion_cleanup=PostFusionCleanupConfig(enabled=False),
        quality_metrics=QualityMetricsConfig(enabled=False),
        output_dir=tmp_path,
    )

    Recon3DPipeline(config, extrinsics, perception=_MaskOnlyPerception()).run_once(frames)

    payload = json.loads((tmp_path / "nipple_guided_fusion.json").read_text(encoding="utf-8"))
    sides = {
        obs["side"]
        for decision in payload["view_decisions"].values()
        for obs in decision["observations"]
    }
    reasons = {decision["reason"] for decision in payload["view_decisions"].values()}
    assert sides == {"left", "right"}
    assert "no_nipple_observation_for_view" not in reasons
