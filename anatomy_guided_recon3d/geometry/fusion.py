from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping, Protocol

import cv2
import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    DepthConfig,
    FusedNipple,
    FusionConfig,
    NippleGuidedFusionConfig,
    NippleObservation,
    PerceptionResult,
    RGBDFrame,
)
from anatomy_guided_recon3d.geometry.backproject import backproject_masked_depth


class FusionBackend(Protocol):
    """Protocol for multi-camera point cloud fusion backends."""

    def fuse(
        self,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
    ) -> o3d.geometry.Geometry:
        """Fuse camera-frame clouds into world-frame geometry.

        Input shape/unit/frame: each cloud is in its camera frame in metres;
        extrinsics provide ``T_world_cam`` for each camera ID.
        Return type: Open3D world-frame geometry.
        """
        ...


def _copy_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return copy.deepcopy(cloud)


def transform_cloud_to_world(
    cloud_cam: o3d.geometry.PointCloud,
    extrinsic: CameraExtrinsics,
) -> o3d.geometry.PointCloud:
    """Transform one cloud from camera to world frame.

    Input shape/unit/frame: cloud points are camera-frame metres; ``T_world_cam``
    maps camera homogeneous points to world homogeneous points.
    Return type: copied world-frame ``PointCloud``.
    """
    cloud_world = _copy_cloud(cloud_cam)
    cloud_world.transform(extrinsic.T_world_cam)
    return cloud_world


def merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge point clouds without downsampling.

    Input shape/unit/frame: all clouds are in a common frame in metres.
    Return type: merged ``PointCloud``.
    """
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged += cloud
    return merged


class PointCloudFusionBackend:
    """Default low-latency world-frame point cloud fusion backend."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
    ) -> o3d.geometry.PointCloud:
        """Fuse camera-frame clouds into one world-frame point cloud.

        Input shape/unit/frame: camera clouds are in metres; extrinsics map
        camera frame to world frame. Missing extrinsics raise ``KeyError``.
        Return type: world-frame ``open3d.geometry.PointCloud``.
        """
        world_clouds: list[o3d.geometry.PointCloud] = []
        for cam_id, cloud in clouds_cam.items():
            if cam_id not in extrinsics:
                raise KeyError(f"Missing extrinsics for camera '{cam_id}'")
            if len(cloud.points) == 0:
                continue
            world_clouds.append(transform_cloud_to_world(cloud, extrinsics[cam_id]))
        merged = merge_clouds(world_clouds)
        if len(merged.points) == 0:
            return merged
        if self.config.voxel_size_m > 0:
            merged = merged.voxel_down_sample(self.config.voxel_size_m)
        if len(merged.points) > self.config.outlier_nb_neighbors:
            merged, _ = merged.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio,
            )
        if self.config.estimate_normals and len(merged.points) > 0:
            merged.estimate_normals()
        return merged


def point_cloud_bounds(cloud: o3d.geometry.PointCloud) -> dict[str, list[float] | int]:
    """Compute simple world-frame point cloud bounds.

    Input shape/unit/frame: cloud is in metres.
    Return type: dict with point count and min/max bounds.
    """
    points = np.asarray(cloud.points)
    if points.size == 0:
        return {"points": 0, "min_bound_m": [], "max_bound_m": []}
    return {
        "points": int(points.shape[0]),
        "min_bound_m": points.min(axis=0).tolist(),
        "max_bound_m": points.max(axis=0).tolist(),
    }


def _flying_edge_mask(depth_m: np.ndarray, valid: np.ndarray, threshold_m: float) -> np.ndarray:
    outliers = np.zeros_like(depth_m, dtype=bool)
    pairs = [
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ]
    for current, neighbour in pairs:
        edge = (
            valid[current]
            & valid[neighbour]
            & (np.abs(depth_m[current] - depth_m[neighbour]) > threshold_m)
        )
        outliers[current] |= edge
    return outliers


def clean_nipple_guided_mask(
    frame: RGBDFrame,
    perception: PerceptionResult,
    config: NippleGuidedFusionConfig,
) -> np.ndarray:
    """Clean one per-view breast mask before RGB-D back-projection.

    Input shape/unit/frame: mask/depth are aligned RGB pixels; depth is converted
    to camera-frame metres. Return type: HxW boolean mask.
    """
    mask = perception.breast_mask.astype(bool)
    if config.mask_erode_px > 0:
        kernel_size = config.mask_erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    valid = frame.depth_mm > 0
    depth_m = frame.depth_mm.astype(np.float64) * frame.depth_config.depth_scale_m
    flying = _flying_edge_mask(depth_m, valid, config.depth_edge_threshold_m)
    return mask & valid & ~flying


def _stable_seed(cam_id: str, base_seed: int) -> int:
    digest = hashlib.sha1(cam_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") ^ int(base_seed)


def _subsample_cloud(cloud: o3d.geometry.PointCloud, fraction: float, seed: int) -> o3d.geometry.PointCloud:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    point_count = len(cloud.points)
    if point_count == 0 or fraction >= 0.999:
        return cloud
    keep_count = max(1, int(round(point_count * fraction)))
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(point_count, size=keep_count, replace=False))
    return cloud.select_by_index(keep.tolist())


def _valid_fused_nipples(nipples: Mapping[str, FusedNipple]) -> dict[str, FusedNipple]:
    return {
        side: nipple
        for side, nipple in nipples.items()
        if side in {"left", "right"} and nipple.confidence > 0.0 and np.isfinite(nipple.mean_world_m).all()
    }


def _observation_payload(obs: NippleObservation, fused: Mapping[str, FusedNipple]) -> dict[str, Any]:
    fused_nipple = fused.get(obs.side)
    error_m = None
    if fused_nipple is not None and fused_nipple.confidence > 0.0:
        error_m = float(np.linalg.norm(obs.point_world_m - fused_nipple.mean_world_m))
    return {
        "cam_id": obs.cam_id,
        "side": obs.side,
        "point_world_m": obs.point_world_m.tolist(),
        "confidence": float(obs.confidence),
        "num_points": int(obs.num_points),
        "error_to_fused_m": error_m,
    }


def make_nipple_guided_view_decisions(
    camera_ids: list[str],
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    config: NippleGuidedFusionConfig,
) -> dict[str, dict[str, Any]]:
    """Score and optionally reject camera views using fused nipple anchors.

    Input shape/unit/frame: observations and fused nipples are world-frame
    metres. Return type: JSON-ready per-camera decisions.
    """
    valid_nipples = _valid_fused_nipples(nipples)
    by_camera: dict[str, list[NippleObservation]] = {cam_id: [] for cam_id in camera_ids}
    for obs in observations:
        by_camera.setdefault(obs.cam_id, []).append(obs)

    if not config.enabled or not valid_nipples:
        reason = "disabled" if not config.enabled else "no_valid_fused_nipple_fallback"
        return {
            cam_id: {
                "accepted": True,
                "weight": 1.0,
                "reason": reason,
                "best_error_m": None,
                "observations": [_observation_payload(obs, nipples) for obs in by_camera.get(cam_id, [])],
            }
            for cam_id in camera_ids
        }

    decisions: dict[str, dict[str, Any]] = {}
    for cam_id in camera_ids:
        cam_obs = [obs for obs in by_camera.get(cam_id, []) if obs.side in valid_nipples]
        obs_payload = [_observation_payload(obs, nipples) for obs in by_camera.get(cam_id, [])]
        if not cam_obs:
            decisions[cam_id] = {
                "accepted": True,
                "weight": float(config.missing_nipple_weight),
                "reason": "no_nipple_observation_for_view",
                "best_error_m": None,
                "observations": obs_payload,
            }
            continue

        scored: list[tuple[float, float, NippleObservation]] = []
        for obs in cam_obs:
            fused_nipple = valid_nipples[obs.side]
            error_m = float(np.linalg.norm(obs.point_world_m - fused_nipple.mean_world_m))
            consistency = float(np.exp(-0.5 * (error_m / max(config.nipple_gate_m, 1e-6)) ** 2))
            score = float(np.clip(obs.confidence * consistency, 0.0, 1.0))
            scored.append((score, error_m, obs))
        best_score, best_error_m, best_obs = max(scored, key=lambda item: item[0])
        used_by_fusion = any(cam_id in nipple.used_cameras for nipple in valid_nipples.values())
        rejected_by_fusion = any(cam_id in nipple.rejected_cameras for nipple in valid_nipples.values())
        consistent = best_error_m <= config.nipple_gate_m or used_by_fusion
        accepted = consistent or not config.reject_inconsistent_views
        if rejected_by_fusion and not used_by_fusion and config.reject_inconsistent_views:
            accepted = False
        if accepted:
            weight = max(float(best_score), float(config.min_consistent_weight))
            reason = "nipple_consistent"
            if not consistent:
                reason = "inconsistent_but_rejection_disabled"
        else:
            weight = 0.0
            reason = "nipple_inconsistent_view_rejected"
        decisions[cam_id] = {
            "accepted": bool(accepted),
            "weight": float(np.clip(weight, 0.0, 1.0)),
            "reason": reason,
            "best_error_m": float(best_error_m),
            "best_side": best_obs.side,
            "observations": obs_payload,
        }

    accepted_count = sum(1 for decision in decisions.values() if decision["accepted"])
    if accepted_count < config.min_accepted_views:
        for decision in decisions.values():
            decision["accepted"] = True
            decision["weight"] = max(float(decision["weight"]), float(config.missing_nipple_weight))
            decision["reason"] = "min_accepted_views_fallback"
    return decisions


def build_nipple_guided_clouds(
    frames: Mapping[str, RGBDFrame],
    perceptions: Mapping[str, PerceptionResult],
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    config: NippleGuidedFusionConfig,
) -> tuple[dict[str, o3d.geometry.PointCloud], dict[str, dict[str, Any]], dict[str, int]]:
    """Build per-camera clouds after nipple-guided mask cleaning and scoring.

    Input shape/unit/frame: frames are aligned RGB-D in camera frame; nipple
    observations/anchors are world-frame metres. Return type:
    ``(clouds_cam, view_decisions, point_counts)``.
    """
    camera_ids = list(frames.keys())
    view_decisions = make_nipple_guided_view_decisions(camera_ids, observations, nipples, config)
    clouds: dict[str, o3d.geometry.PointCloud] = {}
    point_counts: dict[str, int] = {}
    for cam_id, frame in frames.items():
        if cam_id not in perceptions:
            continue
        decision = view_decisions[cam_id]
        if not decision["accepted"]:
            point_counts[cam_id] = 0
            continue
        mask = clean_nipple_guided_mask(frame, perceptions[cam_id], config)
        depth_config = DepthConfig(
            depth_scale_m=frame.depth_config.depth_scale_m,
            min_depth_m=config.depth_min_m,
            max_depth_m=config.depth_max_m,
        )
        cloud = backproject_masked_depth(
            frame.depth_mm,
            mask,
            frame.rgb,
            frame.intrinsics,
            depth_config,
            estimate_normals=False,
        )
        weighted = _subsample_cloud(
            cloud,
            float(decision["weight"]),
            _stable_seed(cam_id, config.random_seed),
        )
        clouds[cam_id] = weighted
        point_counts[cam_id] = len(weighted.points)
    return clouds, view_decisions, point_counts

