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


def _downsample_for_icp(
    cloud: o3d.geometry.PointCloud,
    voxel_size_m: float,
    outlier_nb_neighbors: int = 0,
    outlier_std_ratio: float = 0.0,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return copy.deepcopy(cloud)
    down = cloud.voxel_down_sample(voxel_size_m) if voxel_size_m > 0 else copy.deepcopy(cloud)
    if len(down.points) > max(10, outlier_nb_neighbors) and outlier_nb_neighbors > 0 and outlier_std_ratio > 0.0:
        down, _ = down.remove_statistical_outlier(
            nb_neighbors=int(outlier_nb_neighbors),
            std_ratio=float(outlier_std_ratio),
        )
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


def _average_transform(transforms: Mapping[str, np.ndarray]) -> np.ndarray:
    if not transforms:
        return _identity()
    rvecs: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for transform in transforms.values():
        rvec, _ = cv2.Rodrigues(np.asarray(transform[:3, :3], dtype=np.float64))
        rvecs.append(rvec.reshape(3))
        translations.append(np.asarray(transform[:3, 3], dtype=np.float64))
    return _matrix_from_rtvec(
        np.mean(np.asarray(rvecs, dtype=np.float64), axis=0),
        np.mean(np.asarray(translations, dtype=np.float64), axis=0),
    )


def _median_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        return _identity()
    rvecs: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for transform in transforms:
        rvec, _ = cv2.Rodrigues(np.asarray(transform[:3, :3], dtype=np.float64))
        rvecs.append(rvec.reshape(3))
        translations.append(np.asarray(transform[:3, 3], dtype=np.float64))
    return _matrix_from_rtvec(
        np.median(np.asarray(rvecs, dtype=np.float64), axis=0),
        np.median(np.asarray(translations, dtype=np.float64), axis=0),
    )


def _gauge_normalize_deltas(deltas: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not deltas:
        return {}
    gauge_inv = np.linalg.inv(_average_transform(deltas))
    return {cam_id: gauge_inv @ delta for cam_id, delta in deltas.items()}


def _merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        if len(cloud.points) > 0:
            merged += cloud
    return merged


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
    raw_history: list[np.ndarray] = field(default_factory=list)
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

        if config.registration_topology == "multiway":
            return self._refine_multiway(
                camera_ids,
                clouds_cam,
                extrinsics,
                observations,
                nipples,
                config,
                nipple_config,
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
                config.icp_statistical_outlier_nb_neighbors,
                config.icp_statistical_outlier_std_ratio,
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

    def _refine_multiway(
        self,
        camera_ids: list[str],
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
        observations: list[NippleObservation],
        nipples: Mapping[str, FusedNipple],
        config: AnatomyRegistrationConfig,
        nipple_config: Nipple3DConfig,
    ) -> tuple[dict[str, CameraExtrinsics], list[NippleObservation], dict[str, FusedNipple], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "mode": config.mode,
            "topology": "multiway",
            "reference_camera": None,
            "frame_index": self._frame_index,
            "gauge_normalized": True,
            "iterations": [],
            "cameras": {},
        }
        if not camera_ids:
            return {}, [], {}, diagnostics

        current_deltas: dict[str, np.ndarray] = {}
        camera_payloads: dict[str, dict[str, Any]] = {}
        skipped_frozen: set[str] = set()
        for cam_id in camera_ids:
            state = self._states.setdefault(cam_id, _CameraState())
            cam_obs = _camera_observations(cam_id, observations, nipples, config)
            nipple_init = _translation_init_from_nipples(cam_obs, nipples)
            if state.frozen:
                initial = state.raw_delta_world.copy()
                initial_reason = "frozen"
                skipped_frozen.add(cam_id)
            elif nipple_init is not None:
                initial = nipple_init
                initial_reason = "nipple_translation_init"
            else:
                initial = state.smoothed_delta_world.copy()
                initial_reason = "warm_start_without_valid_nipple"
            current_deltas[cam_id] = initial
            before_residual, before_by_side = _nipple_residuals(cam_obs, nipples)
            camera_payloads[cam_id] = {
                "nipple_count": len(cam_obs),
                "before_residual_m": before_residual,
                "before_residual_by_side_m": before_by_side,
                "initial_reason": initial_reason,
                "icp": {"enabled": False, "reason": "translation_only" if config.mode == "translation_only" else None},
            }

        current_deltas = _gauge_normalize_deltas(current_deltas)

        if config.mode != "translation_only":
            for iteration_idx in range(int(config.multiway_iterations)):
                iteration_payload: dict[str, Any] = {"index": iteration_idx, "cameras": {}}
                next_deltas = dict(current_deltas)
                for cam_id in camera_ids:
                    if cam_id in skipped_frozen:
                        iteration_payload["cameras"][cam_id] = {"accepted": True, "reason": "frozen"}
                        continue
                    candidate, payload = self._estimate_multiway_camera_delta(
                        cam_id,
                        camera_ids,
                        clouds_cam,
                        extrinsics,
                        current_deltas,
                        observations,
                        nipples,
                        config,
                    )
                    next_deltas[cam_id] = candidate
                    iteration_payload["cameras"][cam_id] = payload
                    camera_payloads[cam_id]["icp"] = payload
                current_deltas = _gauge_normalize_deltas(next_deltas)
                diagnostics["iterations"].append(iteration_payload)

        current_deltas = _gauge_normalize_deltas(current_deltas)
        if config.nipple_polish_enabled:
            current_deltas, diagnostics["nipple_polish"] = self._polish_deltas_to_nipple_consensus(
                camera_ids,
                current_deltas,
                observations,
                config,
                nipple_config,
            )
        else:
            diagnostics["nipple_polish"] = {"enabled": False}
        smoothed_deltas: dict[str, np.ndarray] = {}
        raw_deltas: dict[str, np.ndarray] = {}
        for cam_id in camera_ids:
            state = self._states.setdefault(cam_id, _CameraState())
            cam_obs = _camera_observations(cam_id, observations, nipples, config)
            before_residual = camera_payloads[cam_id]["before_residual_m"]
            raw_delta = current_deltas.get(cam_id, state.smoothed_delta_world.copy())
            accepted, gate_reason, after_residual, after_by_side = self._gate_delta(
                raw_delta,
                cam_obs,
                nipples,
                before_residual,
                config,
            )
            reason = "multiway_icp_refined" if config.mode != "translation_only" else camera_payloads[cam_id]["initial_reason"]
            if not accepted:
                raw_delta = state.smoothed_delta_world.copy()
                reason = gate_reason
            elif config.freeze_after_frames > 0:
                if not state.frozen:
                    state.raw_history.append(raw_delta.copy())
                    if len(state.raw_history) > int(config.freeze_after_frames):
                        state.raw_history = state.raw_history[-int(config.freeze_after_frames) :]
                    if len(state.raw_history) >= int(config.freeze_after_frames):
                        raw_delta = _median_transform(state.raw_history)
                        state.frozen = True
                        state.stable_frames = int(config.freeze_after_frames)
                        reason = f"{reason}_median_frozen"
                else:
                    state.raw_history.append(raw_delta.copy())
                    if len(state.raw_history) > int(config.freeze_after_frames):
                        state.raw_history = state.raw_history[-int(config.freeze_after_frames) :]

            smoothed = raw_delta
            if config.smooth_corrections and not state.frozen:
                smoothed = _blend_transforms(state.smoothed_delta_world, raw_delta, config.correction_alpha)

            delta_change = np.linalg.inv(state.smoothed_delta_world) @ smoothed
            if not state.frozen:
                if (
                    accepted
                    and _translation_norm_m(delta_change) <= config.freeze_translation_epsilon_m
                    and _rotation_angle_deg(delta_change) <= config.freeze_rotation_epsilon_deg
                ):
                    state.stable_frames += 1
                else:
                    state.stable_frames = 0

            state.raw_delta_world = raw_delta.copy()
            state.smoothed_delta_world = smoothed.copy()
            raw_deltas[cam_id] = raw_delta
            smoothed_deltas[cam_id] = smoothed
            diagnostics["cameras"][cam_id] = {
                "accepted": bool(accepted),
                "reason": reason,
                "nipple_count": int(camera_payloads[cam_id]["nipple_count"]),
                "before_residual_m": before_residual,
                "before_residual_by_side_m": camera_payloads[cam_id]["before_residual_by_side_m"],
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
                "freeze_history_frames": len(state.raw_history),
                "icp": camera_payloads[cam_id]["icp"],
                "raw_delta_world": raw_delta.tolist(),
                "smoothed_delta_world": smoothed.tolist(),
            }

        self._frame_index += 1
        diagnostics["all_frozen"] = bool(camera_ids) and all(
            self._states.get(cam_id, _CameraState()).frozen for cam_id in camera_ids
        )
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

    def _polish_deltas_to_nipple_consensus(
        self,
        camera_ids: list[str],
        current_deltas: Mapping[str, np.ndarray],
        observations: list[NippleObservation],
        config: AnatomyRegistrationConfig,
        nipple_config: Nipple3DConfig,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        corrected_observations = transform_observations_by_camera_delta(observations, current_deltas)
        corrected_nipples = fuse_nipple_observations(corrected_observations, nipple_config)
        anchors = _valid_nipple_anchors(corrected_nipples)
        payload: dict[str, Any] = {
            "enabled": True,
            "anchors": {
                side: {
                    "confidence": float(nipple.confidence),
                    "used_cameras": list(nipple.used_cameras),
                    "rejected_cameras": list(nipple.rejected_cameras),
                }
                for side, nipple in corrected_nipples.items()
            },
            "cameras": {},
        }
        if not anchors:
            payload["reason"] = "no_valid_corrected_anchor"
            return dict(current_deltas), payload

        by_camera: dict[str, list[tuple[np.ndarray, float, str]]] = {cam_id: [] for cam_id in camera_ids}
        for obs in corrected_observations:
            if obs.cam_id not in by_camera or obs.side not in anchors:
                continue
            if obs.confidence < config.min_nipple_confidence:
                continue
            offset = anchors[obs.side].mean_world_m - obs.point_world_m
            by_camera[obs.cam_id].append((offset, max(float(obs.confidence), 1e-3), obs.side))

        polished: dict[str, np.ndarray] = dict(current_deltas)
        for cam_id in camera_ids:
            items = by_camera.get(cam_id, [])
            if not items:
                payload["cameras"][cam_id] = {"applied": False, "reason": "no_camera_nipple_observation"}
                continue
            offsets = np.asarray([item[0] for item in items], dtype=np.float64)
            weights = np.asarray([item[1] for item in items], dtype=np.float64)
            offset = np.average(offsets, axis=0, weights=weights)
            raw_norm = float(np.linalg.norm(offset))
            if raw_norm > config.nipple_polish_max_translation_m:
                offset = offset / max(raw_norm, 1e-9) * config.nipple_polish_max_translation_m
            offset = offset * config.nipple_polish_alpha
            polish = _identity()
            polish[:3, 3] = offset
            polished[cam_id] = polish @ current_deltas.get(cam_id, _identity())
            payload["cameras"][cam_id] = {
                "applied": True,
                "sides": [item[2] for item in items],
                "raw_offset_norm_m": raw_norm,
                "applied_offset_m": offset.tolist(),
                "applied_offset_norm_m": float(np.linalg.norm(offset)),
            }

        polished = _gauge_normalize_deltas(polished)
        payload["gauge_normalized"] = True
        return polished, payload

    def _estimate_multiway_camera_delta(
        self,
        cam_id: str,
        camera_ids: list[str],
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
        current_deltas: Mapping[str, np.ndarray],
        observations: list[NippleObservation],
        nipples: Mapping[str, FusedNipple],
        config: AnatomyRegistrationConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        base_delta = current_deltas.get(cam_id, _identity())
        cam_obs = _camera_observations(cam_id, observations, nipples, config)
        source_cloud = clouds_cam.get(cam_id)
        if source_cloud is None or len(source_cloud.points) < config.icp_min_points:
            return base_delta, {"enabled": True, "accepted": False, "reason": "source_too_small"}

        target_parts: list[o3d.geometry.PointCloud] = []
        for other_id in camera_ids:
            if other_id == cam_id:
                continue
            other_cloud = clouds_cam.get(other_id)
            if other_cloud is None or len(other_cloud.points) < config.icp_min_points:
                continue
            target_parts.append(
                _transform_cloud_cam_to_world(
                    other_cloud,
                    extrinsics[other_id],
                    current_deltas.get(other_id, _identity()),
                )
            )
        if not target_parts:
            return base_delta, {"enabled": True, "accepted": False, "reason": "target_missing"}

        source_world = _downsample_for_icp(
            _transform_cloud_cam_to_world(source_cloud, extrinsics[cam_id], base_delta),
            config.icp_voxel_size_m,
            config.icp_statistical_outlier_nb_neighbors,
            config.icp_statistical_outlier_std_ratio,
        )
        target_world = _downsample_for_icp(
            _merge_clouds(target_parts),
            config.icp_voxel_size_m,
            config.icp_statistical_outlier_nb_neighbors,
            config.icp_statistical_outlier_std_ratio,
        )
        if len(source_world.points) < config.icp_min_points:
            return base_delta, {"enabled": True, "accepted": False, "reason": "downsampled_source_too_small"}
        if len(target_world.points) < config.icp_min_points:
            return base_delta, {"enabled": True, "accepted": False, "reason": "downsampled_target_too_small"}

        estimator = _make_icp_estimator(config)
        state = self._states.setdefault(cam_id, _CameraState())
        iterations = config.icp_max_iterations_warm if state.stable_frames > 0 else config.icp_max_iterations_cold
        correction = _identity()
        history: list[dict[str, Any]] = []
        try:
            for threshold in config.icp_correspondence_m:
                result = o3d.pipelines.registration.registration_icp(
                    source_world,
                    target_world,
                    float(threshold),
                    correction,
                    estimator,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
                )
                correction = np.asarray(result.transformation, dtype=np.float64)
                history.append(
                    {
                        "threshold_m": float(threshold),
                        "fitness": float(result.fitness),
                        "inlier_rmse_m": float(result.inlier_rmse),
                        "correspondences": int(len(result.correspondence_set)),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - ICP is optional refinement
            return base_delta, {"enabled": True, "accepted": False, "reason": "icp_exception", "error": f"{type(exc).__name__}: {exc}"}

        final_metrics = history[-1] if history else {}
        if float(final_metrics.get("fitness", 0.0)) < config.icp_min_fitness:
            return base_delta, {"enabled": True, "accepted": False, "reason": "low_fitness", "history": history}
        if float(final_metrics.get("inlier_rmse_m", float("inf"))) > config.icp_max_rmse_m:
            return base_delta, {"enabled": True, "accepted": False, "reason": "high_rmse", "history": history}

        candidate = correction @ base_delta
        candidate = _blend_transforms(base_delta, candidate, config.multiway_update_alpha)
        before_residual, _ = _nipple_residuals(cam_obs, nipples)
        ok, gate_reason, residual = _delta_passes_basic_gates(candidate, cam_obs, nipples, before_residual, config)
        if not ok:
            return (
                base_delta,
                {
                    "enabled": True,
                    "accepted": False,
                    "reason": gate_reason,
                    "history": history,
                    "candidate_residual_m": residual,
                },
            )
        return (
            candidate,
            {
                "enabled": True,
                "accepted": True,
                "reason": "multiway_icp_refined",
                "history": history,
                "target_cameras": [other_id for other_id in camera_ids if other_id != cam_id],
                "candidate_residual_m": residual,
            },
        )

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
            config.icp_statistical_outlier_nb_neighbors,
            config.icp_statistical_outlier_std_ratio,
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
