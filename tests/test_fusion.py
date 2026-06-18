from __future__ import annotations

import numpy as np

from recon3d.config import CameraExtrinsics, FusionConfig
from recon3d.geometry.backproject import make_point_cloud
from recon3d.geometry.fusion import PointCloudFusionBackend


def test_point_cloud_fusion_applies_t_world_cam() -> None:
    cloud0 = make_point_cloud(np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    cloud1 = make_point_cloud(np.array([[-1.0, 0.0, 1.0]], dtype=np.float64))
    T0 = np.eye(4, dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    T1[0, 3] = 1.0
    backend = PointCloudFusionBackend(FusionConfig(voxel_size_m=0.0001, outlier_nb_neighbors=20))

    fused = backend.fuse(
        {"cam0": cloud0, "cam1": cloud1},
        {
            "cam0": CameraExtrinsics("cam0", T0),
            "cam1": CameraExtrinsics("cam1", T1),
        },
    )

    points = np.asarray(fused.points)
    assert points.shape[0] >= 1
    assert np.linalg.norm(points.mean(axis=0) - np.array([0.0, 0.0, 1.0])) < 0.001

