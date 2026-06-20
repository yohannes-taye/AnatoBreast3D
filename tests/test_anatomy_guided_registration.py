from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from anatomy_guided_recon3d.config import (
    AnatomyRegistrationConfig,
    CameraExtrinsics,
    FusedNipple,
    Nipple3DConfig,
    NippleObservation,
)
from anatomy_guided_recon3d.geometry.nipple_registration import (
    AnatomyRegistrationState,
    filter_side_mismatched_observations,
)


def _nipple(side: str, xyz: tuple[float, float, float]) -> FusedNipple:
    return FusedNipple(
        side=side,  # type: ignore[arg-type]
        mean_world_m=np.asarray(xyz, dtype=np.float64),
        covariance_world_m2=np.eye(3) * 1e-6,
        confidence=0.9,
        used_cameras=["cam0"],
        rejected_cameras=[],
    )


def _obs(cam_id: str, side: str, xyz: tuple[float, float, float]) -> NippleObservation:
    return NippleObservation(
        cam_id=cam_id,
        side=side,  # type: ignore[arg-type]
        point_world_m=np.asarray(xyz, dtype=np.float64),
        covariance_world_m2=np.eye(3) * 1e-6,
        confidence=0.9,
        num_points=50,
    )


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return cloud


def _cube_surface(offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    grid = np.linspace(-0.04, 0.04, 8)
    pts: list[list[float]] = []
    for x in grid:
        for y in grid:
            pts.append([x, y, 0.96])
            pts.append([x, y, 1.04])
    for x in grid:
        for z in np.linspace(0.96, 1.04, 8):
            pts.append([x, -0.04, z])
            pts.append([x, 0.04, z])
    for y in grid:
        for z in np.linspace(0.96, 1.04, 8):
            pts.append([-0.04, y, z])
            pts.append([0.04, y, z])
    return np.asarray(pts, dtype=np.float64) + np.asarray(offset, dtype=np.float64)


def test_side_gate_rejects_observation_closer_to_opposite_anchor() -> None:
    nipples = {
        "left": _nipple("left", (-0.03, 0.0, 1.0)),
        "right": _nipple("right", (0.03, 0.0, 1.0)),
    }
    observations = [
        _obs("cam0", "left", (0.028, 0.0, 1.0)),
        _obs("cam0", "right", (0.031, 0.0, 1.0)),
    ]

    kept, diagnostics = filter_side_mismatched_observations(
        observations,
        nipples,
        AnatomyRegistrationConfig(side_gate_margin_m=0.002),
    )

    assert len(kept) == 1
    assert kept[0].side == "right"
    assert diagnostics["rejected"][0]["reason"] == "closer_to_opposite_anchor"


def test_weighting_only_mode_is_not_supported() -> None:
    with pytest.raises(ValueError, match="mode must be off, translation_only, or nipple_init_robust_icp"):
        AnatomyRegistrationConfig(mode="weighting_only")  # type: ignore[arg-type]


def test_translation_registration_keeps_reference_fixed_and_corrects_secondary() -> None:
    nipples = {
        "left": _nipple("left", (-0.03, 0.0, 1.0)),
        "right": _nipple("right", (0.03, 0.0, 1.0)),
    }
    observations = [
        _obs("cam0", "left", (-0.02, 0.0, 1.0)),
        _obs("cam0", "right", (0.04, 0.0, 1.0)),
        _obs("cam1", "left", (-0.03, 0.0, 1.0)),
        _obs("cam1", "right", (0.03, 0.0, 1.0)),
    ]
    clouds = {"cam0": o3d.geometry.PointCloud(), "cam1": o3d.geometry.PointCloud()}
    extrinsics = {
        "cam0": CameraExtrinsics("cam0", np.eye(4)),
        "cam1": CameraExtrinsics("cam1", np.eye(4)),
    }

    refined, _, _, diagnostics = AnatomyRegistrationState().refine(
        clouds,
        extrinsics,
        observations,
        nipples,
        AnatomyRegistrationConfig(mode="translation_only", reference_camera="cam1", smooth_corrections=False),
        Nipple3DConfig(),
    )

    assert np.allclose(refined["cam1"].T_world_cam, np.eye(4))
    assert np.allclose(refined["cam0"].T_world_cam[:3, 3], [-0.01, 0.0, 0.0])
    assert diagnostics["cameras"]["cam0"]["reason"] == "nipple_translation_init"


def test_robust_icp_refines_from_nipple_translation_init() -> None:
    nipples = {
        "left": _nipple("left", (-0.03, 0.0, 1.0)),
        "right": _nipple("right", (0.03, 0.0, 1.0)),
    }
    observations = [
        _obs("cam0", "left", (-0.02, 0.0, 1.0)),
        _obs("cam0", "right", (0.04, 0.0, 1.0)),
        _obs("cam1", "left", (-0.03, 0.0, 1.0)),
        _obs("cam1", "right", (0.03, 0.0, 1.0)),
    ]
    clouds = {
        "cam0": _cloud(_cube_surface((0.01, 0.0, 0.0))),
        "cam1": _cloud(_cube_surface()),
    }
    extrinsics = {
        "cam0": CameraExtrinsics("cam0", np.eye(4)),
        "cam1": CameraExtrinsics("cam1", np.eye(4)),
    }

    refined, _, _, diagnostics = AnatomyRegistrationState().refine(
        clouds,
        extrinsics,
        observations,
        nipples,
        AnatomyRegistrationConfig(
            mode="nipple_init_robust_icp",
            reference_camera="cam1",
            smooth_corrections=False,
            icp_min_points=50,
            icp_voxel_size_m=0.004,
            icp_max_rmse_m=0.02,
        ),
        Nipple3DConfig(),
    )

    assert diagnostics["cameras"]["cam0"]["reason"] == "icp_refined"
    assert diagnostics["cameras"]["cam0"]["icp"]["history"][-1]["threshold_m"] == 0.025
    assert np.allclose(refined["cam0"].T_world_cam[:3, 3], [-0.01, 0.0, 0.0], atol=0.003)
