from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal, Mapping, cast

import cv2
import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    Nipple3DConfig,
    NippleObservation,
    PerceptionResult,
    RGBDFrame,
    Side,
    FusedNipple,
)
from anatomy_guided_recon3d.geometry.backproject import backproject_depth_to_arrays


def _clip_box(box_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box_xyxy.astype(np.float64).tolist()
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(width - 1, int(np.ceil(x2)))
    iy2 = min(height - 1, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _cluster_labels(points_cam_m: np.ndarray, eps_m: float, min_cluster_points: int) -> np.ndarray:
    if points_cam_m.shape[0] < min_cluster_points:
        return np.full((points_cam_m.shape[0],), -1, dtype=np.int32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_cam_m)
    labels = np.asarray(
        cloud.cluster_dbscan(eps=eps_m, min_points=min_cluster_points, print_progress=False),
        dtype=np.int32,
    )
    return labels


def _select_cluster(points_cam_m: np.ndarray, labels: np.ndarray, min_cluster_points: int) -> np.ndarray | None:
    candidates: list[tuple[float, int, int]] = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        idx = np.nonzero(labels == label)[0]
        if idx.size < min_cluster_points:
            continue
        centroid_z = float(points_cam_m[idx, 2].mean())
        candidates.append((centroid_z, -int(idx.size), int(label)))
    if not candidates:
        return None
    _, _, selected_label = min(candidates)
    return np.nonzero(labels == selected_label)[0]


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _filter_cluster_by_depth_mad(
    points_cam_m: np.ndarray,
    pixels_uv: np.ndarray,
    cfg: Nipple3DConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    z = points_cam_m[:, 2]
    mad = _median_absolute_deviation(z)
    if mad > cfg.max_cluster_depth_mad_m:
        return None
    sigma = max(1.4826 * mad, 0.0015)
    gate_m = max(0.006, min(0.04, 3.5 * sigma))
    median_z = float(np.median(z))
    keep = np.abs(z - median_z) <= gate_m
    if int(np.count_nonzero(keep)) < cfg.min_cluster_points:
        return None
    return points_cam_m[keep], pixels_uv[keep]


def _robust_tip_candidate_cam(
    cluster_cam_m: np.ndarray,
    normal_cam: np.ndarray,
    cfg: Nipple3DConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    centroid = cluster_cam_m.mean(axis=0)
    signed = (cluster_cam_m - centroid) @ normal_cam
    top_count = max(cfg.tip_min_points, int(np.ceil(cluster_cam_m.shape[0] * cfg.tip_top_fraction)))
    top_count = min(top_count, cluster_cam_m.shape[0])
    if top_count < cfg.tip_min_points:
        return None
    order = np.argsort(signed)
    top_idx = order[-top_count:]
    top_signed = signed[top_idx]
    top_mad = _median_absolute_deviation(top_signed)
    if top_mad > cfg.max_tip_signed_mad_m:
        top_median = float(np.median(top_signed))
        sigma = max(1.4826 * top_mad, 0.0015)
        local_gate = min(cfg.max_tip_signed_mad_m * 3.0, 3.0 * sigma)
        refined_idx = top_idx[np.abs(top_signed - top_median) <= local_gate]
        if refined_idx.size < cfg.tip_min_points:
            return None
        top_idx = refined_idx
        top_signed = signed[top_idx]
        top_mad = _median_absolute_deviation(top_signed)
        if top_mad > cfg.max_tip_signed_mad_m:
            return None
    candidate_cam = np.median(cluster_cam_m[top_idx], axis=0)
    return candidate_cam, top_idx, top_mad


def _robust_center_candidate_cam(
    cluster_cam_m: np.ndarray,
    cluster_pixels_uv: np.ndarray,
    box_xyxy: np.ndarray,
    cfg: Nipple3DConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    x1, y1, x2, y2 = box_xyxy.astype(np.float64).tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = max(0.5 * (x2 - x1), 1.0)
    half_h = max(0.5 * (y2 - y1), 1.0)
    dx = (cluster_pixels_uv[:, 0].astype(np.float64) - cx) / half_w
    dy = (cluster_pixels_uv[:, 1].astype(np.float64) - cy) / half_h
    radial = np.sqrt(dx * dx + dy * dy)
    center_idx = np.nonzero(radial <= cfg.center_patch_fraction)[0]
    if center_idx.size < cfg.center_patch_min_points:
        order = np.argsort(radial)
        keep_count = min(cluster_cam_m.shape[0], cfg.center_patch_min_points)
        if keep_count < cfg.center_patch_min_points:
            return None
        center_idx = order[:keep_count]
    center_points = cluster_cam_m[center_idx]
    z_mad = _median_absolute_deviation(center_points[:, 2])
    if z_mad > cfg.max_cluster_depth_mad_m:
        return None
    candidate_cam = np.median(center_points, axis=0)
    return candidate_cam, center_idx, z_mad


def _pca_normal(points_cam_m: np.ndarray) -> np.ndarray:
    centered = points_cam_m - points_cam_m.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points_cam_m.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return normal / norm


def _orient_normal_away_from_camera(normal: np.ndarray, centroid_cam_m: np.ndarray) -> np.ndarray:
    ray_norm = float(np.linalg.norm(centroid_cam_m))
    if ray_norm == 0.0:
        return normal
    ray_dir = centroid_cam_m / ray_norm
    oriented = normal.copy()
    if float(np.dot(oriented, -ray_dir)) <= 0.0:
        oriented *= -1.0
    return oriented


def _side_from_label(label: str | None) -> Side:
    if label is None:
        return "unknown"
    low = label.lower()
    if "left" in low or low in {"l", "lt"}:
        return "left"
    if "right" in low or low in {"r", "rt"}:
        return "right"
    return "unknown"


def _candidate_covariance_world(
    cluster_cam_m: np.ndarray,
    cluster_pixels_uv: np.ndarray,
    candidate_cam_m: np.ndarray,
    normal_cam: np.ndarray,
    mask: np.ndarray,
    score: float,
    quality_scale: float,
    T_world_cam: np.ndarray,
    jitter_m2: float,
) -> tuple[np.ndarray, float]:
    centroid = cluster_cam_m.mean(axis=0)
    z = max(float(centroid[2]), 1e-6)
    ray = centroid / max(float(np.linalg.norm(centroid)), 1e-6)
    view_quality = max(float(np.dot(normal_cam, -ray)), 0.05)

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    u = np.clip(cluster_pixels_uv[:, 0], 0, mask.shape[1] - 1)
    v = np.clip(cluster_pixels_uv[:, 1], 0, mask.shape[0] - 1)
    boundary_px = float(np.median(distance[v, u])) if cluster_pixels_uv.size else 0.0
    boundary_factor = 1.0 + max(0.0, 5.0 - boundary_px) / 5.0 * 4.0

    detector_quality = max(float(score), 0.05)
    count_quality = min(1.0, cluster_cam_m.shape[0] / 200.0)
    confidence = float(
        np.clip(detector_quality * view_quality * count_quality * quality_scale / boundary_factor, 0.0, 1.0)
    )

    depth_std = max(0.0015, 0.003 * z)
    normal_var = (depth_std**2) / (view_quality**2)
    tangent_var = (depth_std * 1.5) ** 2
    scale = boundary_factor / detector_quality

    n = normal_cam / max(float(np.linalg.norm(normal_cam)), 1e-6)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = ref - np.dot(ref, n) * n
    t1 /= max(float(np.linalg.norm(t1)), 1e-6)
    t2 = np.cross(n, t1)
    basis = np.column_stack((t1, t2, n))
    cov_cam = basis @ np.diag([tangent_var * scale, tangent_var * scale, normal_var * scale]) @ basis.T
    cov_cam += np.eye(3) * jitter_m2
    R = T_world_cam[:3, :3]
    cov_world = R @ cov_cam @ R.T
    cov_world = 0.5 * (cov_world + cov_world.T) + np.eye(3) * jitter_m2
    return cov_world, confidence


def estimate_observation_from_box(
    frame: RGBDFrame,
    perception: PerceptionResult,
    extrinsic: CameraExtrinsics,
    box_xyxy: np.ndarray,
    score: float,
    side: Side,
    config: Nipple3DConfig | None = None,
) -> NippleObservation | None:
    """Estimate one nipple tip observation from one detection box.

    Input shape/unit/frame: box is xyxy pixels in aligned RGB/depth; frame depth
    is raw units converted to camera metres; output point is world metres.
    Return type: ``NippleObservation`` or ``None`` when clustering fails.
    """
    cfg = config or Nipple3DConfig()
    if float(score) < cfg.min_detector_score:
        return None
    h, w = perception.breast_mask.shape
    clipped = _clip_box(box_xyxy, w, h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    local_mask = np.zeros_like(perception.breast_mask, dtype=bool)
    local_mask[y1 : y2 + 1, x1 : x2 + 1] = perception.breast_mask[y1 : y2 + 1, x1 : x2 + 1]
    points_cam, _, pixels_uv = backproject_depth_to_arrays(
        frame.depth_mm,
        local_mask,
        frame.rgb,
        frame.intrinsics,
        frame.depth_config,
    )
    if points_cam.shape[0] < cfg.min_cluster_points:
        return None
    labels = _cluster_labels(points_cam, cfg.dbscan_eps_m, cfg.min_cluster_points)
    cluster_idx = _select_cluster(points_cam, labels, cfg.min_cluster_points)
    if cluster_idx is None:
        return None
    cluster = points_cam[cluster_idx]
    cluster_pixels = pixels_uv[cluster_idx]
    filtered = _filter_cluster_by_depth_mad(cluster, cluster_pixels, cfg)
    if filtered is None:
        return None
    cluster, cluster_pixels = filtered
    centroid = cluster.mean(axis=0)
    normal = _orient_normal_away_from_camera(_pca_normal(cluster), centroid)
    tip_candidate = _robust_tip_candidate_cam(cluster, normal, cfg)
    center_candidate = _robust_center_candidate_cam(cluster, cluster_pixels, box_xyxy, cfg)
    if tip_candidate is None and center_candidate is None:
        return None
    if tip_candidate is None:
        candidate_cam, support_idx, center_mad = center_candidate  # type: ignore[misc]
        quality_scale = float(np.clip(cfg.max_cluster_depth_mad_m / max(center_mad, 0.001), 0.25, 1.0))
    elif center_candidate is None:
        candidate_cam, support_idx, top_mad = tip_candidate
        quality_scale = float(np.clip(cfg.max_tip_signed_mad_m / max(top_mad, 0.001), 0.25, 1.0))
    else:
        tip_cam, tip_idx, top_mad = tip_candidate
        center_cam, center_idx, center_mad = center_candidate
        disagreement_m = float(np.linalg.norm(tip_cam - center_cam))
        center_weight = float(cfg.center_candidate_weight)
        candidate_cam = center_weight * center_cam + (1.0 - center_weight) * tip_cam
        support_idx = center_idx
        top_quality = cfg.max_tip_signed_mad_m / max(top_mad, 0.001)
        center_quality = cfg.max_cluster_depth_mad_m / max(center_mad, 0.001)
        if disagreement_m <= cfg.max_tip_center_distance_m:
            disagreement_quality = 1.0
        else:
            disagreement_quality = max(0.5, cfg.max_tip_center_distance_m / max(disagreement_m, 0.001))
        quality_scale = float(np.clip(min(center_quality, max(0.5, top_quality), disagreement_quality), 0.25, 1.0))
    candidate_h = np.append(candidate_cam, 1.0)
    candidate_world = (extrinsic.T_world_cam @ candidate_h)[:3]
    cov_world, confidence = _candidate_covariance_world(
        cluster,
        cluster_pixels[support_idx],
        candidate_cam,
        normal,
        perception.breast_mask,
        score,
        quality_scale,
        extrinsic.T_world_cam,
        cfg.covariance_jitter_m2,
    )
    if confidence < cfg.min_observation_confidence:
        return None
    return NippleObservation(
        cam_id=frame.cam_id,
        side=side,
        point_world_m=candidate_world,
        covariance_world_m2=cov_world,
        confidence=confidence,
        num_points=int(cluster.shape[0]),
    )


def estimate_nipple_observations(
    frames: Mapping[str, RGBDFrame],
    perceptions: Mapping[str, PerceptionResult],
    extrinsics: Mapping[str, CameraExtrinsics],
    config: Nipple3DConfig | None = None,
) -> list[NippleObservation]:
    """Estimate per-camera nipple observations from perception outputs.

    Input shape/unit/frame: frames are aligned RGB-D; perception boxes/masks are
    aligned RGB pixels; extrinsics map camera to world metres.
    Return type: list of ``NippleObservation`` in world frame.
    """
    observations: list[NippleObservation] = []
    for cam_id, frame in frames.items():
        if cam_id not in perceptions or cam_id not in extrinsics:
            continue
        perception = perceptions[cam_id]
        for idx, box in enumerate(perception.nipple_boxes_xyxy):
            label = perception.nipple_labels[idx] if perception.nipple_labels is not None else None
            side = _side_from_label(label)
            obs = estimate_observation_from_box(
                frame,
                perception,
                extrinsics[cam_id],
                box,
                float(perception.nipple_scores[idx]),
                side,
                config,
            )
            if obs is not None:
                observations.append(obs)
    return observations


def assign_observation_sides(
    observations: Iterable[NippleObservation],
    left_right_axis_world: np.ndarray,
) -> list[NippleObservation]:
    """Assign left/right sides when detector labels are unavailable.

    Input shape/unit/frame: observations are world metres; axis is a world-frame
    unit vector. Smaller projection is assigned ``left`` and larger projection
    is assigned ``right`` for the configured rig convention.
    Return type: list of observations with side set to left/right when possible.
    """
    obs = list(observations)
    labelled = [item for item in obs if item.side in {"left", "right"}]
    unknown = [item for item in obs if item.side == "unknown"]
    if not unknown:
        return obs
    axis = np.array(left_right_axis_world, dtype=np.float64, copy=True)
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    sorted_unknown = sorted(unknown, key=lambda item: float(np.dot(item.point_world_m, axis)))
    assigned: list[NippleObservation] = []
    if len(sorted_unknown) == 1:
        item = sorted_unknown[0]
        projection = float(np.dot(item.point_world_m, axis))
        side: Side = "left" if projection <= 0.0 else "right"
        assigned.append(
            NippleObservation(
                cam_id=item.cam_id,
                side=side,
                point_world_m=item.point_world_m,
                covariance_world_m2=item.covariance_world_m2,
                confidence=item.confidence,
                num_points=item.num_points,
            )
        )
    else:
        midpoint = 0.5 * (
            float(np.dot(sorted_unknown[0].point_world_m, axis))
            + float(np.dot(sorted_unknown[-1].point_world_m, axis))
        )
        for item in sorted_unknown:
            side = "left" if float(np.dot(item.point_world_m, axis)) <= midpoint else "right"
            assigned.append(
                NippleObservation(
                    cam_id=item.cam_id,
                    side=side,
                    point_world_m=item.point_world_m,
                    covariance_world_m2=item.covariance_world_m2,
                    confidence=item.confidence,
                    num_points=item.num_points,
                )
            )
    return labelled + assigned


def _information_fuse(observations: list[NippleObservation], jitter_m2: float) -> tuple[np.ndarray, np.ndarray]:
    info = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for obs in observations:
        cov = obs.covariance_world_m2 + np.eye(3) * jitter_m2
        inv_cov = np.linalg.inv(cov)
        info += inv_cov
        rhs += inv_cov @ obs.point_world_m
    cov_fused = np.linalg.inv(info + np.eye(3) * jitter_m2)
    mean = cov_fused @ rhs
    cov_fused = 0.5 * (cov_fused + cov_fused.T) + np.eye(3) * jitter_m2
    return mean, cov_fused


def _max_pairwise_distance_m(observations: list[NippleObservation]) -> float:
    max_dist = 0.0
    for i, lhs in enumerate(observations):
        for rhs in observations[i + 1 :]:
            dist = float(np.linalg.norm(lhs.point_world_m - rhs.point_world_m))
            max_dist = max(max_dist, dist)
    return max_dist


def _best_consistent_subset(
    observations: list[NippleObservation],
    max_pairwise_distance_m: float,
) -> tuple[list[NippleObservation], list[NippleObservation]]:
    if len(observations) <= 1:
        return list(observations), []
    best: list[NippleObservation] = []
    best_score = -1.0
    count = len(observations)
    for mask in range(1, 1 << count):
        subset = [observations[idx] for idx in range(count) if mask & (1 << idx)]
        if len(subset) < 2:
            continue
        if _max_pairwise_distance_m(subset) > max_pairwise_distance_m:
            continue
        score = float(len(subset) * 10.0 + sum(obs.confidence for obs in subset))
        if score > best_score:
            best = subset
            best_score = score
    if not best:
        return [], list(observations)
    best_ids = {id(obs) for obs in best}
    rejected = [obs for obs in observations if id(obs) not in best_ids]
    return best, rejected


def fuse_side_observations(
    side: str,
    observations: list[NippleObservation],
    config: Nipple3DConfig | None = None,
) -> FusedNipple:
    """Fuse observations for one nipple side with Mahalanobis outlier rejection.

    Input shape/unit/frame: observations are world metres with covariance m^2.
    Return type: ``FusedNipple`` for ``left`` or ``right``.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    nipple_side = cast(Literal["left", "right"], side)
    cfg = config or Nipple3DConfig()
    if not observations:
        return FusedNipple(
            side=nipple_side,
            mean_world_m=np.zeros(3, dtype=np.float64),
            covariance_world_m2=np.eye(3, dtype=np.float64) * 1e6,
            confidence=0.0,
            used_cameras=[],
            rejected_cameras=[],
        )
    if len(observations) < cfg.min_fused_observations:
        return FusedNipple(
            side=nipple_side,
            mean_world_m=np.zeros(3, dtype=np.float64),
            covariance_world_m2=np.eye(3, dtype=np.float64) * 1e6,
            confidence=0.0,
            used_cameras=[],
            rejected_cameras=[obs.cam_id for obs in observations],
        )
    used = list(observations)
    rejected: list[NippleObservation] = []
    if len(used) > 2:
        points = np.asarray([obs.point_world_m for obs in used], dtype=np.float64)
        median = np.median(points, axis=0)
        dists = np.linalg.norm(points - median, axis=1)
        mad = float(np.median(np.abs(dists - np.median(dists))))
        robust_gate_m = max(0.03, 6.0 * mad)
        robust_kept = [obs for obs, dist in zip(used, dists) if float(dist) <= robust_gate_m]
        robust_rejected = [obs for obs, dist in zip(used, dists) if float(dist) > robust_gate_m]
        if robust_kept:
            used = robust_kept
            rejected.extend(robust_rejected)

    consistent, pairwise_rejected = _best_consistent_subset(used, cfg.max_pairwise_nipple_distance_m)
    if len(used) >= 2 and not consistent:
        rejected.extend(pairwise_rejected)
        return FusedNipple(
            side=nipple_side,
            mean_world_m=np.zeros(3, dtype=np.float64),
            covariance_world_m2=np.eye(3, dtype=np.float64) * 1e6,
            confidence=0.0,
            used_cameras=[],
            rejected_cameras=[obs.cam_id for obs in rejected],
        )
    if consistent:
        used = consistent
        rejected.extend(pairwise_rejected)

    mean, cov = _information_fuse(used, cfg.covariance_jitter_m2)
    if len(used) > 2:
        inv_cov = np.linalg.inv(cov + np.eye(3) * cfg.covariance_jitter_m2)
        kept: list[NippleObservation] = []
        for obs in used:
            delta = obs.point_world_m - mean
            dist = float(np.sqrt(delta.T @ inv_cov @ delta))
            if dist > cfg.mahalanobis_threshold:
                rejected.append(obs)
            else:
                kept.append(obs)
        if kept:
            used = kept
            mean, cov = _information_fuse(used, cfg.covariance_jitter_m2)
    confidence = float(np.clip(np.mean([obs.confidence for obs in used]) if used else 0.0, 0.0, 1.0))
    return FusedNipple(
        side=nipple_side,
        mean_world_m=mean,
        covariance_world_m2=cov,
        confidence=confidence,
        used_cameras=[obs.cam_id for obs in used],
        rejected_cameras=[obs.cam_id for obs in rejected],
    )


def fuse_nipple_observations(
    observations: Iterable[NippleObservation],
    config: Nipple3DConfig | None = None,
) -> dict[str, FusedNipple]:
    """Fuse all nipple observations into left/right estimates.

    Input shape/unit/frame: observations are world metres with covariance m^2.
    Return type: dict with ``left`` and ``right`` ``FusedNipple`` entries.
    """
    cfg = config or Nipple3DConfig()
    assigned = assign_observation_sides(list(observations), cfg.left_right_axis_world)
    grouped: dict[str, list[NippleObservation]] = defaultdict(list)
    for obs in assigned:
        if obs.side in {"left", "right"}:
            grouped[obs.side].append(obs)
    return {
        "left": fuse_side_observations("left", grouped["left"], cfg),
        "right": fuse_side_observations("right", grouped["right"], cfg),
    }


def nipple_json_payload(timestamp_ns: int, nipples: Mapping[str, FusedNipple]) -> dict:
    """Create JSON-ready nipple payload.

    Input shape/unit/frame: timestamp is nanoseconds; nipple fields are world
    metres and metre squared covariance.
    Return type: JSON-compatible dict.
    """
    payload: dict[str, object] = {"timestamp_ns": int(timestamp_ns)}
    for side in ("left", "right"):
        nipple = nipples.get(side)
        if nipple is None:
            nipple = fuse_side_observations(side, [])
        payload[side] = {
            "mean_world_m": nipple.mean_world_m.tolist(),
            "covariance_world_m2": nipple.covariance_world_m2.tolist(),
            "confidence": nipple.confidence,
            "used_cameras": nipple.used_cameras,
            "rejected_cameras": nipple.rejected_cameras,
        }
    return payload

