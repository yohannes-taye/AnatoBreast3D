from __future__ import annotations

import copy
from collections import deque
from typing import Any, Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    FusedNipple,
    NippleObservation,
    QualityMetricsConfig,
)


# Distinct RGB colours used to paint each camera's cloud in the debug PLY so
# layering / double-walls between cameras are visible at a glance.
_DEBUG_CAMERA_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.12, 0.12),
    (0.15, 0.70, 0.22),
    (0.20, 0.45, 0.95),
    (0.95, 0.75, 0.12),
    (0.72, 0.20, 0.85),
    (0.12, 0.80, 0.80),
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
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
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


def compute_nipple_pairwise_metrics(
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    gate_m: float,
) -> dict[str, Any]:
    """Per-side multi-camera nipple agreement: the hard ``<= gate_m`` acceptance metric.

    Input shape/unit/frame: ``observations`` are corrected world-frame nipple points
    (one per camera per side); ``nipples`` is the corrected fused consensus.
    Return type: per-side per-camera coords, max/mean pairwise distance, residual to
    consensus, and pass/fail against ``gate_m`` (the multi-camera ``max pairwise``).
    """
    by_side: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    for obs in observations:
        if obs.side in by_side and np.isfinite(obs.point_world_m).all():
            by_side[obs.side][obs.cam_id] = np.asarray(obs.point_world_m, dtype=np.float64)

    result: dict[str, Any] = {"gate_m": float(gate_m)}
    overall_max: float | None = None
    overall_pass = True
    for side in ("left", "right"):
        cams = by_side[side]
        cam_ids = sorted(cams)
        pairs: list[dict[str, Any]] = []
        max_pairwise: float | None = None
        for i, cam_a in enumerate(cam_ids):
            for cam_b in cam_ids[i + 1 :]:
                distance = float(np.linalg.norm(cams[cam_a] - cams[cam_b]))
                pairs.append({"pair": [cam_a, cam_b], "distance_m": distance})
                max_pairwise = distance if max_pairwise is None else max(max_pairwise, distance)
        entry: dict[str, Any] = {
            "camera_count": len(cam_ids),
            "cameras_m": {cam_id: cams[cam_id].tolist() for cam_id in cam_ids},
            "pairwise": pairs,
            "max_pairwise_m": max_pairwise,
            "mean_pairwise_m": float(np.mean([p["distance_m"] for p in pairs])) if pairs else None,
        }
        nipple = nipples.get(side)
        if nipple is not None and nipple.confidence > 0.0 and np.isfinite(nipple.mean_world_m).all():
            consensus = np.asarray(nipple.mean_world_m, dtype=np.float64)
            residuals = {cam_id: float(np.linalg.norm(cams[cam_id] - consensus)) for cam_id in cam_ids}
            entry["consensus_m"] = consensus.tolist()
            entry["residual_to_consensus_m"] = residuals
            entry["max_residual_to_consensus_m"] = max(residuals.values()) if residuals else None
        side_pass = max_pairwise is None or max_pairwise <= gate_m
        entry["pass"] = bool(side_pass)
        result[side] = entry
        if max_pairwise is not None:
            overall_max = max_pairwise if overall_max is None else max(overall_max, max_pairwise)
            overall_pass = overall_pass and side_pass

    result["left_max_pairwise_m"] = result["left"]["max_pairwise_m"]
    result["right_max_pairwise_m"] = result["right"]["max_pairwise_m"]
    result["max_pairwise_m"] = overall_max
    result["pass"] = bool(overall_pass and overall_max is not None)
    return result


def export_colored_camera_ply(
    clouds_cam: Mapping[str, o3d.geometry.PointCloud],
    extrinsics: Mapping[str, CameraExtrinsics],
    path: Any,
    config: QualityMetricsConfig,
) -> dict[str, Any]:
    """Write one PLY with each camera's corrected world cloud painted a distinct colour.

    Input shape/unit/frame: per-camera clouds are camera-frame metres; ``extrinsics``
    are the corrected ``T_world_cam`` so layering between cameras is visible in world.
    Return type: ``{written, path, points_per_camera, colors}``.
    """
    combined = o3d.geometry.PointCloud()
    points_per_camera: dict[str, int] = {}
    colors: dict[str, list[float]] = {}
    for index, cam_id in enumerate(sorted(clouds_cam)):
        cloud = clouds_cam[cam_id]
        if cam_id not in extrinsics or len(cloud.points) == 0:
            continue
        world = copy.deepcopy(cloud)
        world.transform(extrinsics[cam_id].T_world_cam)
        color = _DEBUG_CAMERA_COLORS[index % len(_DEBUG_CAMERA_COLORS)]
        world.paint_uniform_color(list(color))
        combined += world
        points_per_camera[cam_id] = len(world.points)
        colors[cam_id] = list(color)
    if len(combined.points) == 0:
        return {"written": False, "reason": "empty", "points_per_camera": points_per_camera}
    written = bool(o3d.io.write_point_cloud(str(path), combined))
    return {
        "written": written,
        "path": str(path),
        "points_per_camera": points_per_camera,
        "colors": colors,
    }
