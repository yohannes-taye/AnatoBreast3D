from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    AnatomyRegistrationConfig,
    CameraExtrinsics,
    FusedNipple,
    Nipple3DConfig,
    NippleObservation,
)
from anatomy_guided_recon3d.geometry.nipple3d import fuse_nipple_observations


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _readonly_transform(value: np.ndarray) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _rotation_angle_deg(T: np.ndarray) -> float:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    trace = float(np.trace(R))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _translation_norm_m(T: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(T[:3, 3], dtype=np.float64)))


def _transform_cloud_cam_to_world(
    cloud_cam: o3d.geometry.PointCloud,
    extrinsic: CameraExtrinsics,
    delta_world: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud_world = copy.deepcopy(cloud_cam)
    cloud_world.transform(delta_world @ extrinsic.T_world_cam)
    return cloud_world


def _downsample_for_icp(cloud: o3d.geometry.PointCloud, voxel_size_m: float) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return copy.deepcopy(cloud)
    down = cloud.voxel_down_sample(voxel_size_m) if voxel_size_m > 0 else copy.deepcopy(cloud)
    if len(down.points) > 0:
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size_m * 3.0, 1e-4),
                max_nn=30,
            )
        )
    return down


def _make_icp_estimator(config: AnatomyRegistrationConfig):
    registration = o3d.pipelines.registration
    if config.robust_kernel == "none":
        return registration.TransformationEstimationPointToPlane()
    if config.robust_kernel == "huber":
        loss = registration.HuberLoss(k=config.robust_kernel_k_m)
    else:
        loss = registration.TukeyLoss(k=config.robust_kernel_k_m)
    return registration.TransformationEstimationPointToPlane(loss)


def _matrix_from_rtvec(rotation_vec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rotation_vec, dtype=np.float64).reshape(3, 1))
    T = _identity()
    T[:3, :3] = R
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _blend_transforms(previous: np.ndarray, raw: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    prev_rvec, _ = cv2.Rodrigues(np.asarray(previous[:3, :3], dtype=np.float64))
    raw_rvec, _ = cv2.Rodrigues(np.asarray(raw[:3, :3], dtype=np.float64))
    rvec = (1.0 - alpha) * prev_rvec.reshape(3) + alpha * raw_rvec.reshape(3)
    translation = (1.0 - alpha) * previous[:3, 3] + alpha * raw[:3, 3]
    return _matrix_from_rtvec(rvec, translation)


def _valid_nipple_anchors(nipples: Mapping[str, FusedNipple]) -> dict[str, FusedNipple]:
    return {
        side: nipple
        for side, nipple in nipples.items()
        if side in {"left", "right"} and nipple.confidence > 0.0 and np.isfinite(nipple.mean_world_m).all()
    }


def filter_side_mismatched_observations(
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    config: AnatomyRegistrationConfig,
) -> tuple[list[NippleObservation], dict[str, Any]]:
    """Reject side-labelled observations that are closer to the opposite anchor."""
    anchors = _valid_nipple_anchors(nipples)
    if len(anchors) < 2:
        return observations, {"enabled": False, "reason": "missing_two_anchors", "rejected": []}
    kept: list[NippleObservation] = []
    rejected: list[dict[str, Any]] = []
    for obs in observations:
        if obs.side not in {"left", "right"}:
            kept.append(obs)
            continue
        other_side = "right" if obs.side == "left" else "left"
        own = anchors.get(obs.side)
        other = anchors.get(other_side)
        if own is None or other is None:
            kept.append(obs)
            continue
        own_dist = float(np.linalg.norm(obs.point_world_m - own.mean_world_m))
        other_dist = float(np.linalg.norm(obs.point_world_m - other.mean_world_m))
        if other_dist + config.side_gate_margin_m < own_dist:
            rejected.append(
                {
                    "cam_id": obs.cam_id,
                    "side": obs.side,
                    "own_distance_m": own_dist,
                    "opposite_distance_m": other_dist,
                    "reason": "closer_to_opposite_anchor",
                }
            )
            continue
        kept.append(obs)
    return kept, {"enabled": True, "rejected": rejected, "kept": len(kept)}


def transform_observations_by_camera_delta(
    observations: list[NippleObservation],
    deltas_world: Mapping[str, np.ndarray],
) -> list[NippleObservation]:
    """Apply per-camera world-frame correction deltas to nipple observations."""
    corrected: list[NippleObservation] = []
    for obs in observations:
        delta = deltas_world.get(obs.cam_id)
        if delta is None:
            corrected.append(obs)
            continue
        point = (delta @ np.append(obs.point_world_m, 1.0))[:3]
        R = delta[:3, :3]
        cov = R @ obs.covariance_world_m2 @ R.T
        corrected.append(
            NippleObservation(
                cam_id=obs.cam_id,
                side=obs.side,
                point_world_m=point,
                covariance_world_m2=cov,
                confidence=obs.confidence,
                num_points=obs.num_points,
            )
        )
    return corrected


def _camera_observations(
    cam_id: str,
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    config: AnatomyRegistrationConfig,
) -> list[NippleObservation]:
    anchors = _valid_nipple_anchors(nipples)
    return [
        obs
        for obs in observations
        if obs.cam_id == cam_id
        and obs.side in anchors
        and obs.confidence >= config.min_nipple_confidence
        and np.isfinite(obs.point_world_m).all()
    ]


def _nipple_residuals(
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    delta_world: np.ndarray | None = None,
) -> tuple[float | None, dict[str, float]]:
    anchors = _valid_nipple_anchors(nipples)
    residuals: list[float] = []
    by_side: dict[str, float] = {}
    for obs in observations:
        anchor = anchors.get(obs.side)
        if anchor is None:
            continue
        point = obs.point_world_m
        if delta_world is not None:
            point = (delta_world @ np.append(point, 1.0))[:3]
        residual = float(np.linalg.norm(point - anchor.mean_world_m))
        residuals.append(residual)
        by_side[obs.side] = residual
    if not residuals:
        return None, by_side
    return float(np.mean(residuals)), by_side


def _translation_init_from_nipples(
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
) -> np.ndarray | None:
    anchors = _valid_nipple_anchors(nipples)
    offsets: list[np.ndarray] = []
    weights: list[float] = []
    for obs in observations:
        anchor = anchors.get(obs.side)
        if anchor is None:
            continue
        offsets.append(anchor.mean_world_m - obs.point_world_m)
        weights.append(max(float(obs.confidence), 1e-3))
    if not offsets:
        return None
    offset = np.average(np.asarray(offsets, dtype=np.float64), axis=0, weights=np.asarray(weights, dtype=np.float64))
    delta = _identity()
    delta[:3, 3] = offset
    return delta


def _delta_passes_basic_gates(
    delta: np.ndarray,
    observations: list[NippleObservation],
    nipples: Mapping[str, FusedNipple],
    before_residual: float | None,
    config: AnatomyRegistrationConfig,
) -> tuple[bool, str, float | None]:
    after_residual, _ = _nipple_residuals(observations, nipples, delta)
    if _translation_norm_m(delta) > config.max_translation_m:
        return False, "translation_over_gate", after_residual
    if _rotation_angle_deg(delta) > config.max_rotation_deg:
        return False, "rotation_over_gate", after_residual
    if (
        before_residual is not None
        and after_residual is not None
        and before_residual > 1e-6
        and after_residual > before_residual * config.residual_improvement_ratio
    ):
        return False, "nipple_residual_worse", after_residual
    return True, "accepted", after_residual


@dataclass
class _CameraState:
    raw_delta_world: np.ndarray = field(default_factory=_identity)
    smoothed_delta_world: np.ndarray = field(default_factory=_identity)
    stable_frames: int = 0
    frozen: bool = False


class AnatomyRegistrationState:
    """Stateful quasi-static online extrinsic refinement."""

    def __init__(self) -> None:
        self._states: dict[str, _CameraState] = {}
        self._frame_index = 0

    def current_deltas(self, camera_ids: list[str]) -> dict[str, np.ndarray]:
        return {
            cam_id: self._states.get(cam_id, _CameraState()).smoothed_delta_world.copy()
            for cam_id in camera_ids
        }

    def current_extrinsics(
        self,
        extrinsics: Mapping[str, CameraExtrinsics],
        camera_ids: list[str] | None = None,
    ) -> dict[str, CameraExtrinsics]:
        ids = camera_ids or list(extrinsics.keys())
        deltas = self.current_deltas(ids)
        return {
            cam_id: CameraExtrinsics(cam_id, deltas.get(cam_id, _identity()) @ extrinsics[cam_id].T_world_cam)
            for cam_id in ids
            if cam_id in extrinsics
        }

    def refine(
        self,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
        observations: list[NippleObservation],
        nipples: Mapping[str, FusedNipple],
        config: AnatomyRegistrationConfig,
        nipple_config: Nipple3DConfig,
    ) -> tuple[dict[str, CameraExtrinsics], list[NippleObservation], dict[str, FusedNipple], dict[str, Any]]:
        camera_ids = [cam_id for cam_id in sorted(clouds_cam) if cam_id in extrinsics]
        if config.mode == "off":
            deltas = {cam_id: _identity() for cam_id in camera_ids}
            corrected_observations = transform_observations_by_camera_delta(observations, deltas)
            corrected_nipples = fuse_nipple_observations(corrected_observations, nipple_config)
            return (
                {cam_id: extrinsics[cam_id] for cam_id in camera_ids},
                corrected_observations,
                corrected_nipples,
                {
                    "mode": config.mode,
                    "enabled": False,
                    "reason": "registration_disabled",
                    "deltas": {cam_id: _identity().tolist() for cam_id in camera_ids},
                },
            )

        reference = config.reference_camera if config.reference_camera in camera_ids else (camera_ids[0] if camera_ids else None)
        diagnostics: dict[str, Any] = {
            "mode": config.mode,
            "topology": "reference_star",
            "reference_camera": reference,
            "frame_index": self._frame_index,
            "cameras": {},
        }

        target_cloud: o3d.geometry.PointCloud | None = None
        if reference is not None and reference in clouds_cam and reference in extrinsics:
            target_cloud = _downsample_for_icp(
                _transform_cloud_cam_to_world(clouds_cam[reference], extrinsics[reference], _identity()),
                config.icp_voxel_size_m,
            )

        raw_deltas: dict[str, np.ndarray] = {}
        smoothed_deltas: dict[str, np.ndarray] = {}
        for cam_id in camera_ids:
            state = self._states.setdefault(cam_id, _CameraState())
            cam_obs = _camera_observations(cam_id, observations, nipples, config)
            before_residual, before_by_side = _nipple_residuals(cam_obs, nipples)
            run_recheck = self._frame_index % max(1, config.recheck_interval_frames) == 0
            can_skip_frozen = state.frozen and not run_recheck

            if cam_id == reference:
                raw_delta = _identity()
                reason = "reference_camera_fixed"
                icp_payload: dict[str, Any] = {"enabled": False, "reason": reason}
            elif can_skip_frozen:
                raw_delta = state.raw_delta_world.copy()
                reason = "frozen"
                icp_payload = {"enabled": False, "reason": reason}
            else:
                raw_delta, reason, icp_payload = self._estimate_camera_delta(
                    cam_id,
                    clouds_cam,
                    extrinsics,
                    cam_obs,
                    nipples,
                    target_cloud,
                    reference,
                    state,
                    config,
                )

            accepted, gate_reason, after_residual, after_by_side = self._gate_delta(
                raw_delta,
                cam_obs,
                nipples,
                before_residual,
                config,
            )
            if not accepted:
                raw_delta = _identity() if state.smoothed_delta_world is None else state.smoothed_delta_world.copy()
                reason = gate_reason

            smoothed = raw_delta
            if config.smooth_corrections and cam_id != reference:
                smoothed = _blend_transforms(state.smoothed_delta_world, raw_delta, config.correction_alpha)

            delta_change = np.linalg.inv(state.smoothed_delta_world) @ smoothed
            if (
                accepted
                and _translation_norm_m(delta_change) <= config.freeze_translation_epsilon_m
                and _rotation_angle_deg(delta_change) <= config.freeze_rotation_epsilon_deg
            ):
                state.stable_frames += 1
            else:
                state.stable_frames = 0
            if config.freeze_after_frames > 0 and state.stable_frames >= config.freeze_after_frames:
                state.frozen = True
            elif run_recheck and state.frozen:
                state.frozen = False

            state.raw_delta_world = raw_delta.copy()
            state.smoothed_delta_world = smoothed.copy()
            raw_deltas[cam_id] = raw_delta
            smoothed_deltas[cam_id] = smoothed
            diagnostics["cameras"][cam_id] = {
                "accepted": bool(accepted),
                "reason": reason,
                "nipple_count": len(cam_obs),
                "before_residual_m": before_residual,
                "before_residual_by_side_m": before_by_side,
                "after_raw_residual_m": after_residual,
                "after_raw_residual_by_side_m": after_by_side,
                "raw_translation_m": raw_delta[:3, 3].tolist(),
                "raw_translation_norm_m": _translation_norm_m(raw_delta),
                "raw_rotation_deg": _rotation_angle_deg(raw_delta),
                "smoothed_translation_m": smoothed[:3, 3].tolist(),
                "smoothed_translation_norm_m": _translation_norm_m(smoothed),
                "smoothed_rotation_deg": _rotation_angle_deg(smoothed),
                "stable_frames": state.stable_frames,
                "frozen": state.frozen,
                "icp": icp_payload,
                "raw_delta_world": raw_delta.tolist(),
                "smoothed_delta_world": smoothed.tolist(),
            }

        self._frame_index += 1
        refined = {
            cam_id: CameraExtrinsics(cam_id, smoothed_deltas.get(cam_id, _identity()) @ extrinsics[cam_id].T_world_cam)
            for cam_id in camera_ids
        }
        corrected_observations = transform_observations_by_camera_delta(observations, smoothed_deltas)
        corrected_nipples = fuse_nipple_observations(corrected_observations, nipple_config)
        diagnostics["corrected_nipple_anchors"] = {
            side: {
                "mean_world_m": nipple.mean_world_m.tolist(),
                "confidence": float(nipple.confidence),
                "used_cameras": list(nipple.used_cameras),
                "rejected_cameras": list(nipple.rejected_cameras),
            }
            for side, nipple in corrected_nipples.items()
        }
        return refined, corrected_observations, corrected_nipples, diagnostics

    def _estimate_camera_delta(
        self,
        cam_id: str,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
        cam_obs: list[NippleObservation],
        nipples: Mapping[str, FusedNipple],
        target_cloud: o3d.geometry.PointCloud | None,
        reference: str | None,
        state: _CameraState,
        config: AnatomyRegistrationConfig,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        nipple_init = _translation_init_from_nipples(cam_obs, nipples)
        if nipple_init is None:
            nipple_init = state.smoothed_delta_world.copy()
            init_reason = "warm_start_without_nipple"
        else:
            init_reason = "nipple_translation_init"

        if config.mode == "translation_only":
            return nipple_init, init_reason, {"enabled": False, "reason": "translation_only"}

        if target_cloud is None or reference is None:
            return nipple_init, "icp_no_reference_target", {"enabled": False, "reason": "no_reference_target"}
        if len(clouds_cam.get(cam_id, o3d.geometry.PointCloud()).points) < config.icp_min_points:
            return nipple_init, "icp_source_too_small", {"enabled": False, "reason": "source_too_small"}
        if len(target_cloud.points) < config.icp_min_points:
            return nipple_init, "icp_target_too_small", {"enabled": False, "reason": "target_too_small"}

        source_world = _downsample_for_icp(
            _transform_cloud_cam_to_world(clouds_cam[cam_id], extrinsics[cam_id], nipple_init),
            config.icp_voxel_size_m,
        )
        if len(source_world.points) < config.icp_min_points:
            return nipple_init, "icp_downsampled_source_too_small", {"enabled": False, "reason": "downsampled_source_too_small"}

        estimator = _make_icp_estimator(config)
        iterations = config.icp_max_iterations_warm if state.stable_frames > 0 else config.icp_max_iterations_cold
        current = _identity()
        history: list[dict[str, Any]] = []
        try:
            for threshold in config.icp_correspondence_m:
                result = o3d.pipelines.registration.registration_icp(
                    source_world,
                    target_cloud,
                    float(threshold),
                    current,
                    estimator,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
                )
                current = np.asarray(result.transformation, dtype=np.float64)
                history.append(
                    {
                        "threshold_m": float(threshold),
                        "fitness": float(result.fitness),
                        "inlier_rmse_m": float(result.inlier_rmse),
                        "correspondences": int(len(result.correspondence_set)),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - ICP is optional refinement
            return nipple_init, "icp_failed_exception", {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}

        final_metrics = history[-1] if history else {}
        if float(final_metrics.get("fitness", 0.0)) < config.icp_min_fitness:
            return nipple_init, "icp_low_fitness_fallback", {"enabled": True, "history": history, "accepted": False}
        if float(final_metrics.get("inlier_rmse_m", float("inf"))) > config.icp_max_rmse_m:
            return nipple_init, "icp_high_rmse_fallback", {"enabled": True, "history": history, "accepted": False}

        icp_delta = current @ nipple_init
        before_residual, _ = _nipple_residuals(cam_obs, nipples)
        icp_ok, icp_gate_reason, icp_residual = _delta_passes_basic_gates(
            icp_delta,
            cam_obs,
            nipples,
            before_residual,
            config,
        )
        init_ok, init_gate_reason, init_residual = _delta_passes_basic_gates(
            nipple_init,
            cam_obs,
            nipples,
            before_residual,
            config,
        )
        if not icp_ok and init_ok:
            return (
                nipple_init,
                "icp_rejected_nipple_translation_fallback",
                {
                    "enabled": True,
                    "history": history,
                    "accepted": False,
                    "fallback": "nipple_translation_init",
                    "icp_gate_reason": icp_gate_reason,
                    "icp_residual_m": icp_residual,
                    "translation_init_gate_reason": init_gate_reason,
                    "translation_init_residual_m": init_residual,
                },
            )

        return icp_delta, "icp_refined", {"enabled": True, "history": history, "accepted": True}

    def _gate_delta(
        self,
        delta: np.ndarray,
        observations: list[NippleObservation],
        nipples: Mapping[str, FusedNipple],
        before_residual: float | None,
        config: AnatomyRegistrationConfig,
    ) -> tuple[bool, str, float | None, dict[str, float]]:
        translation_m = _translation_norm_m(delta)
        rotation_deg = _rotation_angle_deg(delta)
        after_residual, after_by_side = _nipple_residuals(observations, nipples, delta)
        if translation_m > config.max_translation_m:
            return False, "translation_over_gate", after_residual, after_by_side
        if rotation_deg > config.max_rotation_deg:
            return False, "rotation_over_gate", after_residual, after_by_side
        if (
            before_residual is not None
            and after_residual is not None
            and before_residual > 1e-6
            and after_residual > before_residual * config.residual_improvement_ratio
        ):
            return False, "nipple_residual_worse", after_residual, after_by_side
        return True, "accepted", after_residual, after_by_side
