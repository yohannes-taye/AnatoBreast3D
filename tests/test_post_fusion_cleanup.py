from __future__ import annotations

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import FusedNipple, PostFusionCleanupConfig
from anatomy_guided_recon3d.geometry.post_cleanup import cleanup_fused_cloud


def _cluster(center: tuple[float, float, float], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.normal(scale=0.002, size=(count, 3))


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.tile([[0.8, 0.6, 0.5]], (points.shape[0], 1)))
    return cloud


def _nipple(side: str, xyz: tuple[float, float, float]) -> FusedNipple:
    return FusedNipple(
        side=side,  # type: ignore[arg-type]
        mean_world_m=np.asarray(xyz, dtype=np.float64),
        covariance_world_m2=np.eye(3) * 1e-6,
        confidence=0.9,
        used_cameras=["cam0"],
        rejected_cameras=[],
    )


def test_cleanup_keeps_nipple_anchored_smaller_side_not_largest_only() -> None:
    left = _cluster((-0.05, 0.0, 0.8), 900, 1)
    right = _cluster((0.05, 0.0, 0.8), 300, 2)
    floater = _cluster((0.25, 0.25, 0.8), 80, 3)
    cloud = _cloud(np.vstack([left, right, floater]))
    nipples = {
        "left": _nipple("left", (-0.05, 0.0, 0.8)),
        "right": _nipple("right", (0.05, 0.0, 0.8)),
    }

    cleaned, diagnostics = cleanup_fused_cloud(
        cloud,
        nipples,
        PostFusionCleanupConfig(
            radius_outlier_enabled=False,
            dbscan_eps_m=0.01,
            dbscan_min_points=20,
            min_cluster_points=800,
            nipple_anchor_keep_radius_m=0.02,
            adjacency_distance_m=0.0,
        ),
    )

    assert len(cleaned.points) >= 1100
    assert len(cleaned.points) < len(cloud.points)
    selected = diagnostics["dbscan"]["selected_labels"]
    assert len(selected) == 2
    assert any(cluster["keep_by_nipple"] for cluster in diagnostics["dbscan"]["clusters"])
