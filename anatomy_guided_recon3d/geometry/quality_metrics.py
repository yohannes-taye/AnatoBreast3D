from __future__ import annotations

import copy
from collections import deque
from typing import Any, Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    FusedNipple,
    QualityMetricsConfig,
)


def _limited_cloud(
    cloud: o3d.geometry.PointCloud,
    config: QualityMetricsConfig,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return copy.deepcopy(cloud)
    reduced = cloud.voxel_down_sample(config.surface_voxel_size_m)
    count = len(reduced.points)
    if count <= config.max_points_per_cloud:
        return reduced
    indices = np.linspace(0, count - 1, config.max_points_per_cloud, dtype=np.int64)
    return reduced.select_by_index(indices.tolist())


def _world_metric_cloud(
    cloud_cam: o3d.geometry.PointCloud,
    extrinsic: CameraExtrinsics,
    config: QualityMetricsConfig,
) -> o3d.geometry.PointCloud:
    cloud = _limited_cloud(cloud_cam, config)
    cloud.transform(extrinsic.T_world_cam)
    return cloud


def _overlap_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_distance_m: float,
) -> np.ndarray:
    if len(source.points) == 0 or len(target.points) == 0:
        return np.empty((0,), dtype=np.float64)
    distances = np.asarray(source.compute_point_cloud_distance(target), dtype=np.float64)
    return distances[np.isfinite(distances) & (distances <= max_distance_m)]


def _symmetric_surface_metrics(
    a: o3d.geometry.PointCloud,
    b: o3d.geometry.PointCloud,
    config: QualityMetricsConfig,
) -> dict[str, Any] | None:
    if len(a.points) < config.min_points or len(b.points) < config.min_points:
        return None
    dist_ab = _overlap_distances(a, b, config.surface_max_distance_m)
    dist_ba = _overlap_distances(b, a, config.surface_max_distance_m)
    distances = np.concatenate([dist_ab, dist_ba])
    if distances.size == 0:
        return None
    return {
        "rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "mae_m": float(np.mean(distances)),
        "max_m": float(np.max(distances)),
        "overlap_points": int(distances.size),
        "overlap_ratio": float(
            0.5
            * (
                dist_ab.size / max(1, len(a.points))
                + dist_ba.size / max(1, len(b.points))
            )
        ),
    }


def compute_cross_camera_surface_consistency(
    clouds_cam: Mapping[str, o3d.geometry.PointCloud],
    extrinsics: Mapping[str, CameraExtrinsics],
    config: QualityMetricsConfig,
) -> dict[str, Any]:
    """Measure corrected per-camera surface agreement without using nipple residuals."""
    if not config.enabled:
        return {"enabled": False}
    world_clouds = {
        cam_id: _world_metric_cloud(cloud, extrinsics[cam_id], config)
        for cam_id, cloud in sorted(clouds_cam.items())
        if cam_id in extrinsics and len(cloud.points) >= config.min_points
    }
    pairs: list[dict[str, Any]] = []
    cam_ids = sorted(world_clouds)
    for i, cam_a in enumerate(cam_ids):
        for cam_b in cam_ids[i + 1 :]:
            metrics = _symmetric_surface_metrics(world_clouds[cam_a], world_clouds[cam_b], config)
            if metrics is None:
                pairs.append(
                    {
                        "pair": [cam_a, cam_b],
                        "valid": False,
                        "reason": "insufficient_overlap",
                    }
                )
                continue
            pairs.append({"pair": [cam_a, cam_b], "valid": True, **metrics})

    valid = [pair for pair in pairs if pair.get("valid")]
    if not valid:
        return {
            "enabled": True,
            "valid": False,
            "reason": "no_valid_camera_pairs",
            "pairs": pairs,
        }
    weights = np.asarray([max(1, int(pair["overlap_points"])) for pair in valid], dtype=np.float64)
    rmse = np.asarray([float(pair["rmse_m"]) for pair in valid], dtype=np.float64)
    overlap = np.asarray([float(pair["overlap_ratio"]) for pair in valid], dtype=np.float64)
    return {
        "enabled": True,
        "valid": True,
        "mean_rmse_m": float(np.mean(rmse)),
        "weighted_rmse_m": float(np.average(rmse, weights=weights)),
        "mean_overlap_ratio": float(np.mean(overlap)),
        "valid_pairs": len(valid),
        "pairs": pairs,
    }


class TemporalRepeatabilityTracker:
    """Rolling output stability metrics for static replay/phantom checks."""

    def __init__(self, config: QualityMetricsConfig) -> None:
        self.config = config
        self._nipple_history: dict[str, deque[np.ndarray]] = {
            "left": deque(maxlen=config.temporal_window),
            "right": deque(maxlen=config.temporal_window),
        }
        self._centroid_history: deque[np.ndarray] = deque(maxlen=config.temporal_window)
        self._previous_cloud: o3d.geometry.PointCloud | None = None

    def update(
        self,
        fused_cloud: o3d.geometry.PointCloud,
        nipples: Mapping[str, FusedNipple],
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}

        diagnostics: dict[str, Any] = {
            "enabled": True,
            "window": int(self.config.temporal_window),
        }
        cloud = _limited_cloud(fused_cloud, self.config)
        if len(cloud.points) >= self.config.min_points:
            points = np.asarray(cloud.points, dtype=np.float64)
            centroid = np.mean(points, axis=0)
            self._centroid_history.append(centroid)
            if self._previous_cloud is not None and len(self._previous_cloud.points) >= self.config.min_points:
                metrics = _symmetric_surface_metrics(cloud, self._previous_cloud, self.config)
                if metrics is not None:
                    diagnostics["surface_to_previous"] = metrics
            self._previous_cloud = copy.deepcopy(cloud)
        else:
            diagnostics["surface_to_previous"] = {"valid": False, "reason": "insufficient_points"}

        if len(self._centroid_history) >= 2:
            centroids = np.asarray(self._centroid_history, dtype=np.float64)
            diagnostics["cloud_centroid_std_m"] = float(np.mean(np.std(centroids, axis=0)))
            diagnostics["cloud_centroid_window"] = int(len(self._centroid_history))

        nipple_diag: dict[str, Any] = {}
        for side in ("left", "right"):
            nipple = nipples.get(side)
            if nipple is None or nipple.confidence <= 0.0:
                nipple_diag[side] = {"valid": False, "reason": "missing_nipple"}
                continue
            self._nipple_history[side].append(np.asarray(nipple.mean_world_m, dtype=np.float64))
            history = np.asarray(self._nipple_history[side], dtype=np.float64)
            if len(history) < 2:
                nipple_diag[side] = {"valid": True, "window": int(len(history)), "std_m": None}
                continue
            std_axis = np.std(history, axis=0)
            nipple_diag[side] = {
                "valid": True,
                "window": int(len(history)),
                "std_axis_m": std_axis.tolist(),
                "std_m": float(np.linalg.norm(std_axis)),
            }
        diagnostics["nipple_repeatability"] = nipple_diag
        return diagnostics
