from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import FusedNipple, PostFusionCleanupConfig


@dataclass(frozen=True)
class _ClusterInfo:
    label: int
    indices: np.ndarray
    count: int
    min_bound: np.ndarray
    max_bound: np.ndarray
    centroid: np.ndarray
    nearest_nipple_m: float | None
    nearest_nipple_side: str | None


def _valid_nipple_points(nipples: Mapping[str, FusedNipple]) -> dict[str, np.ndarray]:
    return {
        side: nipple.mean_world_m
        for side, nipple in nipples.items()
        if side in {"left", "right"} and nipple.confidence > 0.0 and np.isfinite(nipple.mean_world_m).all()
    }


def _aabb_distance(a: _ClusterInfo, b: _ClusterInfo) -> float:
    gap = np.maximum(np.maximum(a.min_bound - b.max_bound, b.min_bound - a.max_bound), 0.0)
    return float(np.linalg.norm(gap))


def _cluster_infos(
    points: np.ndarray,
    labels: np.ndarray,
    nipples: Mapping[str, FusedNipple],
) -> list[_ClusterInfo]:
    nipple_points = _valid_nipple_points(nipples)
    infos: list[_ClusterInfo] = []
    for label in sorted(int(value) for value in set(labels.tolist()) if int(value) >= 0):
        indices = np.nonzero(labels == label)[0]
        if indices.size == 0:
            continue
        cluster_points = points[indices]
        nearest_dist: float | None = None
        nearest_side: str | None = None
        for side, nipple_point in nipple_points.items():
            distances = np.linalg.norm(cluster_points - nipple_point.reshape(1, 3), axis=1)
            min_dist = float(np.min(distances)) if distances.size else None
            if min_dist is not None and (nearest_dist is None or min_dist < nearest_dist):
                nearest_dist = min_dist
                nearest_side = side
        infos.append(
            _ClusterInfo(
                label=label,
                indices=indices,
                count=int(indices.size),
                min_bound=cluster_points.min(axis=0),
                max_bound=cluster_points.max(axis=0),
                centroid=cluster_points.mean(axis=0),
                nearest_nipple_m=nearest_dist,
                nearest_nipple_side=nearest_side,
            )
        )
    return infos


def _select_clusters(
    infos: list[_ClusterInfo],
    config: PostFusionCleanupConfig,
) -> tuple[set[int], list[dict[str, Any]]]:
    selected: set[int] = set()
    payload: list[dict[str, Any]] = []
    for info in infos:
        keep_by_size = info.count >= config.min_cluster_points
        keep_by_nipple = (
            config.preserve_nipple_clusters
            and info.nearest_nipple_m is not None
            and info.nearest_nipple_m <= config.nipple_anchor_keep_radius_m
        )
        if keep_by_size or keep_by_nipple:
            selected.add(info.label)
        payload.append(
            {
                "label": info.label,
                "points": info.count,
                "nearest_nipple_m": info.nearest_nipple_m,
                "nearest_nipple_side": info.nearest_nipple_side,
                "keep_by_size": keep_by_size,
                "keep_by_nipple": keep_by_nipple,
                "keep_by_adjacency": False,
            }
        )

    if config.adjacency_distance_m <= 0 or not selected:
        return selected, payload

    info_by_label = {info.label: info for info in infos}
    payload_by_label = {entry["label"]: entry for entry in payload}
    queue: deque[int] = deque(sorted(selected))
    while queue:
        current_label = queue.popleft()
        current = info_by_label[current_label]
        for other in infos:
            if other.label in selected:
                continue
            if _aabb_distance(current, other) <= config.adjacency_distance_m:
                selected.add(other.label)
                queue.append(other.label)
                payload_by_label[other.label]["keep_by_adjacency"] = True
    return selected, payload


def cleanup_fused_cloud(
    cloud: o3d.geometry.PointCloud,
    nipples: Mapping[str, FusedNipple],
    config: PostFusionCleanupConfig,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    """Remove floating islands while preserving nipple-anchored breast clusters."""
    diagnostics: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "points_input": int(len(cloud.points)),
    }
    if not config.enabled or len(cloud.points) == 0:
        diagnostics["points_output"] = int(len(cloud.points))
        return cloud, diagnostics

    cleaned = cloud
    if config.radius_outlier_enabled and len(cleaned.points) > config.min_neighbors:
        radius_input = cleaned
        before = len(cleaned.points)
        radius_cleaned, indices = cleaned.remove_radius_outlier(
            nb_points=int(config.min_neighbors),
            radius=float(config.radius_m),
        )
        retained_ratio = len(radius_cleaned.points) / max(before, 1)
        if retained_ratio < config.min_retained_ratio:
            cleaned = radius_input
            fallback = "retained_ratio_below_min_keep_input"
        else:
            cleaned = radius_cleaned
            fallback = None
        diagnostics["radius_outlier"] = {
            "enabled": True,
            "points_before": int(before),
            "points_after": int(len(cleaned.points)),
            "removed": int(before - len(cleaned.points)),
            "retained_ratio": float(retained_ratio),
            "fallback": fallback,
        }
    else:
        diagnostics["radius_outlier"] = {"enabled": False}

    if not config.dbscan_enabled or len(cleaned.points) < config.dbscan_min_points:
        diagnostics["dbscan"] = {"enabled": False}
        diagnostics["points_output"] = int(len(cleaned.points))
        return cleaned, diagnostics

    points = np.asarray(cleaned.points)
    labels = np.asarray(
        cleaned.cluster_dbscan(
            eps=float(config.dbscan_eps_m),
            min_points=int(config.dbscan_min_points),
            print_progress=False,
        ),
        dtype=np.int32,
    )
    infos = _cluster_infos(points, labels, nipples)
    selected, cluster_payload = _select_clusters(infos, config)
    if not selected:
        diagnostics["dbscan"] = {
            "enabled": True,
            "clusters": cluster_payload,
            "selected_labels": [],
            "noise_points": int(np.count_nonzero(labels < 0)),
            "fallback": "no_selected_clusters_keep_input",
        }
        diagnostics["points_output"] = int(len(cleaned.points))
        return cleaned, diagnostics

    keep_indices = np.nonzero(np.isin(labels, list(selected)))[0]
    output = cleaned.select_by_index(keep_indices.tolist())
    diagnostics["dbscan"] = {
        "enabled": True,
        "clusters": cluster_payload,
        "selected_labels": sorted(int(label) for label in selected),
        "noise_points": int(np.count_nonzero(labels < 0)),
        "points_before": int(len(cleaned.points)),
        "points_after": int(len(output.points)),
        "removed": int(len(cleaned.points) - len(output.points)),
    }
    diagnostics["points_output"] = int(len(output.points))
    return output, diagnostics
