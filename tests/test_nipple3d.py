from __future__ import annotations

import numpy as np

from recon3d.config import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthConfig,
    Nipple3DConfig,
    NippleObservation,
    PerceptionResult,
    RGBDFrame,
)
from recon3d.geometry.nipple3d import (
    estimate_nipple_observations,
    fuse_nipple_observations,
)


def _synthetic_frame(cam_id: str = "cam0", tx: float = 0.0) -> tuple[RGBDFrame, CameraExtrinsics, PerceptionResult]:
    h, w = 96, 96
    intr = CameraIntrinsics(width=w, height=h, fx=120.0, fy=120.0, cx=48.0, cy=48.0)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - 48.0) ** 2 + (yy - 48.0) ** 2
    z_m = 1.0 - 0.05 * np.exp(-r2 / (2.0 * 8.0**2))
    depth = np.round(z_m * 1000.0).astype(np.uint16)
    mask = r2 < 30.0**2
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    frame = RGBDFrame(cam_id, rgb, depth, 1_000_000, intr, DepthConfig(depth_scale_m=0.001, min_depth_m=0.2, max_depth_m=2.0))
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    extr = CameraExtrinsics(cam_id, T)
    perception = PerceptionResult(
        cam_id=cam_id,
        breast_mask=mask,
        nipple_boxes_xyxy=np.array([[36.0, 36.0, 60.0, 60.0]], dtype=np.float64),
        nipple_scores=np.array([0.9], dtype=np.float64),
        nipple_labels=["right"],
    )
    return frame, extr, perception


def test_nipple_tip_estimation_from_synthetic_patch() -> None:
    frame, extr, perception = _synthetic_frame()
    observations = estimate_nipple_observations(
        {frame.cam_id: frame},
        {frame.cam_id: perception},
        {frame.cam_id: extr},
        Nipple3DConfig(dbscan_eps_m=0.02, min_cluster_points=5),
    )
    assert len(observations) == 1
    point = observations[0].point_world_m
    assert abs(point[0]) < 0.02
    assert abs(point[1]) < 0.02
    assert point[2] < 0.96


def test_nipple_fusion_rejects_one_camera_outlier() -> None:
    cov = np.eye(3, dtype=np.float64) * 1e-6
    observations = [
        NippleObservation("cam0", "right", np.array([0.01, 0.0, 0.95]), cov, 0.9, 100),
        NippleObservation("cam1", "right", np.array([0.011, 0.0, 0.951]), cov, 0.9, 100),
        NippleObservation("cam2", "right", np.array([0.20, 0.0, 1.20]), cov, 0.9, 100),
    ]
    fused = fuse_nipple_observations(observations, Nipple3DConfig(mahalanobis_threshold=3.0))
    right = fused["right"]
    assert "cam2" in right.rejected_cameras
    assert np.linalg.norm(right.mean_world_m - np.array([0.0105, 0.0, 0.9505])) < 0.005


def test_nipple_fusion_handles_missing_camera() -> None:
    cov = np.eye(3, dtype=np.float64) * 1e-5
    observations = [
        NippleObservation("cam0", "left", np.array([-0.03, 0.0, 0.95]), cov, 0.8, 80),
    ]
    fused = fuse_nipple_observations(observations)
    assert fused["left"].used_cameras == ["cam0"]
    assert fused["right"].confidence == 0.0

