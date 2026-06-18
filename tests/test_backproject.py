from __future__ import annotations

import numpy as np

from recon3d.config import CameraIntrinsics, DepthConfig
from recon3d.geometry.backproject import backproject_depth_to_arrays


def test_backproject_synthetic_points_error_under_1mm() -> None:
    intr = CameraIntrinsics(width=3, height=2, fx=1000.0, fy=1000.0, cx=1.0, cy=0.5)
    depth_cfg = DepthConfig(depth_scale_m=0.001, min_depth_m=0.1, max_depth_m=2.0)
    depth = np.array([[1000, 1000, 1000], [1500, 0, 500]], dtype=np.uint16)
    mask = np.ones((2, 3), dtype=bool)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)

    points, _, pixels = backproject_depth_to_arrays(depth, mask, rgb, intr, depth_cfg)

    expected = []
    for u, v in pixels:
        z = depth[v, u] * 0.001
        expected.append([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z])
    expected_arr = np.asarray(expected, dtype=np.float64)
    assert points.shape == expected_arr.shape
    assert np.max(np.linalg.norm(points - expected_arr, axis=1)) < 0.001

