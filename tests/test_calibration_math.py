from __future__ import annotations

import numpy as np
import pytest

import recon3d.calibration.charuco as charuco
from recon3d.calibration.charuco import CharucoBoardSpec, PoseEstimate, estimate_extrinsics_from_charuco
from recon3d.calibration.io import (
    load_depth_configs,
    load_extrinsics,
    load_intrinsics,
    save_depth_configs,
    save_extrinsics,
    save_intrinsics,
)
from recon3d.config import CameraExtrinsics, CameraIntrinsics, DepthConfig, RGBDFrame


def test_t_world_cam_convention_and_inverse() -> None:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.0, 2.0, 3.0]
    p_cam = np.array([0.1, 0.2, 1.0, 1.0])
    p_world = T @ p_cam
    p_cam_roundtrip = np.linalg.inv(T) @ p_world
    assert np.allclose(p_cam_roundtrip, p_cam)


def test_calibration_json_roundtrip(tmp_path) -> None:
    intr = {"cam0": CameraIntrinsics(64, 48, 100.0, 101.0, 32.0, 24.0, np.zeros(5))}
    extr = {"cam0": CameraExtrinsics("cam0", np.eye(4))}
    depth = {"cam0": DepthConfig(0.001, 0.25, 2.0)}

    save_intrinsics(tmp_path / "intrinsics.json", intr)
    save_extrinsics(tmp_path / "extrinsics.json", extr)
    save_depth_configs(tmp_path / "depth.json", depth)

    intr_loaded = load_intrinsics(tmp_path / "intrinsics.json")
    extr_loaded = load_extrinsics(tmp_path / "extrinsics.json")
    depth_loaded = load_depth_configs(tmp_path / "depth.json")

    assert intr_loaded["cam0"].K.shape == (3, 3)
    assert np.allclose(extr_loaded["cam0"].T_world_cam, np.eye(4))
    assert depth_loaded["cam0"].depth_scale_m == 0.001


def test_charuco_reference_camera_world(monkeypatch: pytest.MonkeyPatch) -> None:
    intr = CameraIntrinsics(4, 3, 100.0, 100.0, 1.5, 1.0, None)
    depth_cfg = DepthConfig(0.001)
    frame = RGBDFrame(
        "cam0",
        np.zeros((3, 4, 3), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint16),
        0,
        intr,
        depth_cfg,
    )
    frames = {"cam0": frame, "cam1": RGBDFrame("cam1", frame.rgb, frame.depth_mm, 0, intr, depth_cfg)}
    T_cam0_board = np.eye(4, dtype=np.float64)
    T_cam1_board = np.eye(4, dtype=np.float64)
    T_cam1_board[:3, 3] = [1.0, 0.0, 0.0]
    poses = {
        "cam0": PoseEstimate(T_cam0_board, 0.1, 0.2, 10),
        "cam1": PoseEstimate(T_cam1_board, 0.1, 0.2, 10),
    }

    monkeypatch.setattr(charuco, "make_charuco_board", lambda spec: object())
    monkeypatch.setattr(charuco, "detect_charuco", lambda rgb, board: object())
    monkeypatch.setattr(charuco, "estimate_board_pose", lambda detection, board, intrinsics: poses[intrinsics_marker.pop(0)])

    intrinsics_marker = ["cam0", "cam1"]
    extrinsics, _ = estimate_extrinsics_from_charuco(frames, CharucoBoardSpec(5, 7, 0.04, 0.03), "first_camera")

    assert np.allclose(extrinsics["cam0"].T_world_cam, np.eye(4))
    expected_cam1 = np.eye(4, dtype=np.float64)
    expected_cam1[:3, 3] = [-1.0, 0.0, 0.0]
    assert np.allclose(extrinsics["cam1"].T_world_cam, expected_cam1)
