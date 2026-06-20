from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CloudCleanupConfig,
    FusedNipple,
    NippleGuidedFusionConfig,
    NippleObservation,
    PerceptionResult,
    RGBDFrame,
)
from anatomy_guided_recon3d.geometry.cloud_cleanup import backproject_cleaned_breast_cloud


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
    metres. Return type: JSON-ready per-camera decisions and weights.
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


def build_guided_cleaned_clouds(
    frames: Mapping[str, RGBDFrame],
    perceptions: Mapping[str, PerceptionResult],
    view_decisions: Mapping[str, Mapping[str, Any]],
    cleanup_config: CloudCleanupConfig,
    guide_config: NippleGuidedFusionConfig,
) -> tuple[dict[str, o3d.geometry.PointCloud], dict[str, int]]:
    """Back-project cleaned breast clouds and apply guided per-view weights.

    Input shape/unit/frame: frame clouds are camera-frame metres. Return type:
    ``(clouds_cam, weighted_point_counts)`` for the downstream fusion backend.
    """
    clouds: dict[str, o3d.geometry.PointCloud] = {}
    point_counts: dict[str, int] = {}
    for cam_id, frame in frames.items():
        if cam_id not in perceptions:
            continue
        decision = view_decisions.get(cam_id, {"accepted": True, "weight": 1.0})
        if not bool(decision.get("accepted", True)):
            point_counts[cam_id] = 0
            continue
        cloud = backproject_cleaned_breast_cloud(frame, perceptions[cam_id].breast_mask, cleanup_config)
        weighted = _subsample_cloud(
            cloud,
            float(decision.get("weight", 1.0)),
            _stable_seed(cam_id, guide_config.random_seed),
        )
        clouds[cam_id] = weighted
        point_counts[cam_id] = len(weighted.points)
    return clouds, point_counts
