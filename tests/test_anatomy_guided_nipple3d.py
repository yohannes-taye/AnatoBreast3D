from __future__ import annotations

import numpy as np

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthConfig,
    Nipple3DConfig,
    NippleObservation,
    PerceptionResult,
    RGBDFrame,
)
from anatomy_guided_recon3d.geometry.nipple3d import estimate_nipple_observations, fuse_nipple_observations


def _synthetic_nipple_frame(
    score: float = 0.9,
    center_depth_mm: int | None = None,
) -> tuple[RGBDFrame, CameraExtrinsics, PerceptionResult]:
    h, w = 96, 96
    intr = CameraIntrinsics(width=w, height=h, fx=120.0, fy=120.0, cx=48.0, cy=48.0)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - 48.0) ** 2 + (yy - 48.0) ** 2
    z_m = 1.0 - 0.05 * np.exp(-r2 / (2.0 * 8.0**2))
    depth = np.round(z_m * 1000.0).astype(np.uint16)
    if center_depth_mm is not None:
        depth[48, 48] = center_depth_mm
    mask = r2 < 30.0**2
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    frame = RGBDFrame(
        "cam0",
        rgb,
        depth,
        1_000_000,
        intr,
        DepthConfig(depth_scale_m=0.001, min_depth_m=0.2, max_depth_m=2.0),
    )
    extrinsic = CameraExtrinsics("cam0", np.eye(4, dtype=np.float64))
    perception = PerceptionResult(
        cam_id="cam0",
        breast_mask=mask,
        nipple_boxes_xyxy=np.array([[36.0, 36.0, 60.0, 60.0]], dtype=np.float64),
        nipple_scores=np.array([score], dtype=np.float64),
        nipple_labels=["right"],
    )
    return frame, extrinsic, perception


def _estimate(
    frame: RGBDFrame,
    extrinsic: CameraExtrinsics,
    perception: PerceptionResult,
    config: Nipple3DConfig,
):
    return estimate_nipple_observations(
        {frame.cam_id: frame},
        {frame.cam_id: perception},
        {frame.cam_id: extrinsic},
        config,
    )


def test_anatomy_nipple_tip_uses_robust_top_patch_not_single_depth_outlier() -> None:
    frame, extrinsic, perception = _synthetic_nipple_frame(center_depth_mm=900)

    observations = _estimate(
        frame,
        extrinsic,
        perception,
        Nipple3DConfig(
            dbscan_eps_m=0.08,
            min_cluster_points=5,
            tip_min_points=8,
            max_cluster_depth_mad_m=0.04,
            min_observation_confidence=0.01,
        ),
    )

    assert len(observations) == 1
    assert observations[0].point_world_m[2] > 0.93
    assert observations[0].point_world_m[2] < 0.97


def test_anatomy_nipple_observation_rejects_low_detector_score() -> None:
    frame, extrinsic, perception = _synthetic_nipple_frame(score=0.19)

    observations = _estimate(
        frame,
        extrinsic,
        perception,
        Nipple3DConfig(dbscan_eps_m=0.02, min_cluster_points=5, min_detector_score=0.2),
    )

    assert observations == []


def test_anatomy_nipple_observation_rejects_low_quality_confidence() -> None:
    frame, extrinsic, perception = _synthetic_nipple_frame(score=0.9)

    observations = _estimate(
        frame,
        extrinsic,
        perception,
        Nipple3DConfig(
            dbscan_eps_m=0.02,
            min_cluster_points=5,
            min_observation_confidence=0.95,
        ),
    )

    assert observations == []


def test_anatomy_nipple_fusion_rejects_inconsistent_two_camera_anchor() -> None:
    cov = np.eye(3, dtype=np.float64) * 1e-6
    observations = [
        NippleObservation("cam1", "left", np.array([0.0, 0.0, 1.0]), cov, 0.7, 100),
        NippleObservation("cam2", "left", np.array([0.0255, 0.0, 1.0]), cov, 0.8, 100),
    ]

    fused = fuse_nipple_observations(
        observations,
        Nipple3DConfig(max_pairwise_nipple_distance_m=0.015),
    )

    assert fused["left"].confidence == 0.0
    assert fused["left"].used_cameras == []
    assert sorted(fused["left"].rejected_cameras) == ["cam1", "cam2"]


def test_anatomy_nipple_fusion_keeps_consistent_pair_and_rejects_far_camera() -> None:
    cov = np.eye(3, dtype=np.float64) * 1e-6
    observations = [
        NippleObservation("cam0", "right", np.array([0.0, 0.0, 1.0]), cov, 0.7, 100),
        NippleObservation("cam1", "right", np.array([0.004, 0.0, 1.0]), cov, 0.7, 100),
        NippleObservation("cam2", "right", np.array([0.03, 0.0, 1.0]), cov, 0.9, 100),
    ]

    fused = fuse_nipple_observations(
        observations,
        Nipple3DConfig(max_pairwise_nipple_distance_m=0.015),
    )

    assert fused["right"].used_cameras == ["cam0", "cam1"]
    assert fused["right"].rejected_cameras == ["cam2"]
    assert fused["right"].confidence > 0.0
