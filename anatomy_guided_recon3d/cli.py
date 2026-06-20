from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np
import yaml

from anatomy_guided_recon3d.calibration.charuco import (
    CharucoBoardSpec,
    estimate_extrinsics_from_charuco,
    summarize_reprojection_errors,
)
from anatomy_guided_recon3d.calibration.io import load_extrinsics, save_extrinsics
from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    AnatomyRegistrationConfig,
    CloudCleanupConfig,
    FusionConfig,
    Nipple3DConfig,
    NippleGuidedFusionConfig,
    PerceptionConfig,
    PostFusionCleanupConfig,
    QualityMetricsConfig,
    Recon3DConfig,
    SurfaceConfig,
    TSDFConfig,
    VisualizationConfig,
)
from anatomy_guided_recon3d.pipeline import Recon3DPipeline
from anatomy_guided_recon3d.sensors.femto_bolt import FemtoBoltCamera, HardwareUnavailableError, OrbbecBagCamera, ReplayRGBDCamera
from anatomy_guided_recon3d.sensors.rig import MultiFemtoBoltRig


LOGGER = logging.getLogger(__name__)


BOARD_NAME_RE = re.compile(r"^charuco_(?P<x>\d+)x(?P<y>\d+)_(?P<square_mm>\d+(?:\.\d+)?)mm$")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_recon_config(path: Path, output_override: Path | None = None) -> tuple[Recon3DConfig, Path]:
    """Load a reconstruction CLI config.

    Input shape/unit/frame: YAML/JSON config paths are filesystem paths; numeric
    fields use metres, milliseconds, and detector confidence units.
    Return type: ``(Recon3DConfig, extrinsics_path)``.
    """
    raw = _load_yaml_or_json(path)
    perception_raw = raw.get("perception", {})
    cleanup_raw = raw.get("cloud_cleanup", {})
    fusion_raw = raw.get("fusion", {})
    surface_raw = raw.get("surface", {})
    visualization_raw = raw.get("visualization", {})
    nipple_raw = raw.get("nipple3d", {})
    nipple_guided_raw = raw.get("nipple_guided_fusion", {})
    registration_raw = raw.get("anatomy_registration", {})
    post_cleanup_raw = raw.get("post_fusion_cleanup", {})
    tsdf_raw = raw.get("tsdf_fusion", {})
    quality_raw = raw.get("quality_metrics", {})
    output_dir = output_override or (Path(raw["output_dir"]) if raw.get("output_dir") else None)
    config = Recon3DConfig(
        perception=PerceptionConfig(
            model_config_path=Path(
                perception_raw.get("model_config_path", "configs/breastseg/breastseg_b2_inplace.py")
            ),
            checkpoint_path=Path(perception_raw.get("checkpoint_path", "checkpoint/breastseg.pth")),
            device=str(perception_raw.get("device", "cuda")),
            score_thr=float(perception_raw.get("score_thr", 0.3)),
            topk=int(perception_raw.get("topk", 2)),
            nms_iou=float(perception_raw.get("nms_iou", 0.5)),
        ),
        cloud_cleanup=CloudCleanupConfig(
            mask_erode_px=int(cleanup_raw.get("mask_erode_px", 4)),
            flying_depth_edge_threshold_m=float(cleanup_raw.get("flying_depth_edge_threshold_m", 0.015)),
            min_depth_m=float(cleanup_raw.get("min_depth_m", 0.25)),
            max_depth_m=float(cleanup_raw.get("max_depth_m", 1.2)),
            edge_grazing_max_angle_deg=float(cleanup_raw.get("edge_grazing_max_angle_deg", 80.0)),
        ),
        fusion=FusionConfig(
            voxel_size_m=float(fusion_raw.get("voxel_size_m", 0.002)),
            outlier_nb_neighbors=int(fusion_raw.get("outlier_nb_neighbors", 20)),
            outlier_std_ratio=float(fusion_raw.get("outlier_std_ratio", 2.0)),
            estimate_normals=bool(fusion_raw.get("estimate_normals", False)),
        ),
        surface=SurfaceConfig(
            mesh_enabled=bool(surface_raw.get("mesh_enabled", False)),
            poisson_depth=int(surface_raw.get("poisson_depth", 8)),
            density_trim_quantile=float(surface_raw.get("density_trim_quantile", 0.1)),
            sample_points=int(surface_raw.get("sample_points", 80000)),
            normal_radius_m=float(surface_raw.get("normal_radius_m", 0.01)),
            normal_max_nn=int(surface_raw.get("normal_max_nn", 30)),
            min_points=int(surface_raw.get("min_points", 500)),
        ),
        visualization=VisualizationConfig(
            rerun_log_cameras=bool(visualization_raw.get("rerun_log_cameras", False)),
        ),
        nipple3d=Nipple3DConfig(
            dbscan_eps_m=float(nipple_raw.get("dbscan_eps_m", 0.005)),
            min_cluster_points=int(nipple_raw.get("min_cluster_points", 30)),
            covariance_jitter_m2=float(nipple_raw.get("covariance_jitter_m2", 1e-8)),
            left_right_axis_world=nipple_raw.get("left_right_axis_world", [1.0, 0.0, 0.0]),
            mahalanobis_threshold=float(nipple_raw.get("mahalanobis_threshold", 3.0)),
            nipple_apex_fraction=float(nipple_raw.get("nipple_apex_fraction", 0.15)),
        ),
        nipple_guided_fusion=NippleGuidedFusionConfig(
            enabled=bool(nipple_guided_raw.get("enabled", True)),
            nipple_gate_m=float(nipple_guided_raw.get("nipple_gate_m", 0.05)),
            reject_inconsistent_views=bool(nipple_guided_raw.get("reject_inconsistent_views", True)),
            missing_nipple_weight=float(nipple_guided_raw.get("missing_nipple_weight", 0.55)),
            min_consistent_weight=float(nipple_guided_raw.get("min_consistent_weight", 0.35)),
            min_accepted_views=int(nipple_guided_raw.get("min_accepted_views", 1)),
            random_seed=int(nipple_guided_raw.get("random_seed", 11)),
        ),
        anatomy_registration=AnatomyRegistrationConfig(
            mode=str(registration_raw.get("mode", "nipple_init_robust_icp")),
            reference_camera=registration_raw.get("reference_camera", "cam1"),
            min_nipple_confidence=float(registration_raw.get("min_nipple_confidence", 0.15)),
            side_gate_margin_m=float(registration_raw.get("side_gate_margin_m", 0.005)),
            max_translation_m=float(registration_raw.get("max_translation_m", 0.04)),
            max_rotation_deg=float(registration_raw.get("max_rotation_deg", 8.0)),
            residual_improvement_ratio=float(registration_raw.get("residual_improvement_ratio", 1.05)),
            smooth_corrections=bool(registration_raw.get("smooth_corrections", True)),
            correction_alpha=float(registration_raw.get("correction_alpha", 0.35)),
            freeze_after_frames=int(registration_raw.get("freeze_after_frames", 20)),
            recheck_interval_frames=int(registration_raw.get("recheck_interval_frames", 30)),
            freeze_translation_epsilon_m=float(registration_raw.get("freeze_translation_epsilon_m", 0.001)),
            freeze_rotation_epsilon_deg=float(registration_raw.get("freeze_rotation_epsilon_deg", 0.25)),
            icp_voxel_size_m=float(registration_raw.get("icp_voxel_size_m", 0.006)),
            icp_correspondence_m=tuple(registration_raw.get("icp_correspondence_m", [0.05, 0.035, 0.025])),
            icp_max_iterations_cold=int(registration_raw.get("icp_max_iterations_cold", 20)),
            icp_max_iterations_warm=int(registration_raw.get("icp_max_iterations_warm", 3)),
            icp_min_points=int(registration_raw.get("icp_min_points", 250)),
            icp_min_fitness=float(registration_raw.get("icp_min_fitness", 0.25)),
            icp_max_rmse_m=float(registration_raw.get("icp_max_rmse_m", 0.012)),
            robust_kernel=str(registration_raw.get("robust_kernel", "tukey")),
            robust_kernel_k_m=float(registration_raw.get("robust_kernel_k_m", 0.01)),
        ),
        post_fusion_cleanup=PostFusionCleanupConfig(
            enabled=bool(post_cleanup_raw.get("enabled", True)),
            radius_outlier_enabled=bool(post_cleanup_raw.get("radius_outlier_enabled", True)),
            radius_m=float(post_cleanup_raw.get("radius_m", 0.012)),
            min_neighbors=int(post_cleanup_raw.get("min_neighbors", 8)),
            min_retained_ratio=float(post_cleanup_raw.get("min_retained_ratio", 0.25)),
            dbscan_enabled=bool(post_cleanup_raw.get("dbscan_enabled", True)),
            dbscan_eps_m=float(post_cleanup_raw.get("dbscan_eps_m", 0.012)),
            dbscan_min_points=int(post_cleanup_raw.get("dbscan_min_points", 40)),
            nipple_anchor_keep_radius_m=float(post_cleanup_raw.get("nipple_anchor_keep_radius_m", 0.035)),
            min_cluster_points=int(post_cleanup_raw.get("min_cluster_points", 800)),
            adjacency_distance_m=float(post_cleanup_raw.get("adjacency_distance_m", 0.02)),
            preserve_nipple_clusters=bool(post_cleanup_raw.get("preserve_nipple_clusters", True)),
        ),
        tsdf_fusion=TSDFConfig(
            enabled=bool(tsdf_raw.get("enabled", False)),
            voxel_length_m=float(tsdf_raw.get("voxel_length_m", 0.004)),
            sdf_trunc_m=float(tsdf_raw.get("sdf_trunc_m", 0.025)),
            depth_trunc_m=float(tsdf_raw.get("depth_trunc_m", 1.2)),
            color=bool(tsdf_raw.get("color", True)),
        ),
        quality_metrics=QualityMetricsConfig(
            enabled=bool(quality_raw.get("enabled", True)),
            surface_voxel_size_m=float(quality_raw.get("surface_voxel_size_m", 0.008)),
            surface_max_distance_m=float(quality_raw.get("surface_max_distance_m", 0.025)),
            min_points=int(quality_raw.get("min_points", 80)),
            max_points_per_cloud=int(quality_raw.get("max_points_per_cloud", 1500)),
            temporal_window=int(quality_raw.get("temporal_window", 30)),
            nipple_pairwise_gate_m=float(quality_raw.get("nipple_pairwise_gate_m", 0.005)),
            debug_ply_enabled=bool(quality_raw.get("debug_ply_enabled", False)),
            debug_ply_interval_frames=int(quality_raw.get("debug_ply_interval_frames", 10)),
        ),
        output_dir=output_dir,
        sync_max_drift_ms=float(raw.get("sync_max_drift_ms", 5.0)),
    )
    extrinsics_path = Path(raw.get("extrinsics_path", "configs/recon3d/extrinsics.json"))
    return config, extrinsics_path


def _build_replay_rig(bag: Path, max_drift_ms: float) -> MultiFemtoBoltRig:
    camera_dirs = sorted(path for path in bag.iterdir() if path.is_dir() and (path / "metadata.json").is_file())
    if not camera_dirs:
        raise FileNotFoundError(f"No replay camera folders with metadata.json found under {bag}")
    cameras = [ReplayRGBDCamera(path.name, path) for path in camera_dirs]
    return MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=max_drift_ms)


def _parse_bag_entries(entries: list[str] | None) -> dict[str, Path]:
    """Parse ``cam_id=path.bag`` CLI values."""
    bags: dict[str, Path] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Bag entries must use cam_id=path.bag form, got: {entry}")
        cam_id, raw_path = entry.split("=", 1)
        cam_id = cam_id.strip()
        if not cam_id:
            raise ValueError(f"Bag entry has empty camera id: {entry}")
        if cam_id in bags:
            raise ValueError(f"Duplicate bag camera id: {cam_id}")
        bags[cam_id] = Path(raw_path)
    return bags


def _load_bag_session(path: Path) -> dict[str, Path]:
    """Load a Recon3D SDK-bag session manifest."""
    manifest_path = path / "session.json" if path.is_dir() else path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"Bag session has no cameras: {manifest_path}")
    bags: dict[str, Path] = {}
    for entry in cameras:
        if not isinstance(entry, dict):
            raise ValueError(f"Bag session camera entries must be mappings: {manifest_path}")
        cam_id = str(entry.get("id") or entry.get("cam_id") or "")
        rel_bag = entry.get("bag") or entry.get("bag_path")
        if not cam_id or not rel_bag:
            raise ValueError(f"Bag session camera missing id/bag: {entry}")
        bag_path = Path(rel_bag)
        if not bag_path.is_absolute():
            bag_path = manifest_path.parent / bag_path
        bags[cam_id] = bag_path
    return bags


def _timestamp_source_family(source: str | None) -> str:
    source = str(source or "")
    if "global_timestamp" in source:
        return "global"
    if "timestamp_us" in source or "timestamp_ms" in source:
        return "hardware"
    if "metadata" in source or "sensor" in source:
        return "metadata"
    if "system" in source:
        return "system"
    return "none"


def _summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _orbbec_sdk_version() -> str | None:
    try:
        import pyorbbecsdk as ob  # type: ignore
    except Exception:  # noqa: BLE001 - optional hardware dependency
        return None
    return str(getattr(ob, "__version__", "") or "") or None


def _frame_timestamp_stats(frames: dict[str, Any]) -> tuple[dict[str, int], float, int]:
    timestamps = {cam_id: int(frame.timestamp_ns) for cam_id, frame in frames.items()}
    missing = [cam_id for cam_id, timestamp_ns in timestamps.items() if timestamp_ns <= 0]
    if missing:
        raise RuntimeError(f"Synchronized frame set has missing SDK timestamps: {missing}")
    drift_ms = (max(timestamps.values()) - min(timestamps.values())) / 1_000_000.0
    common_t0_ns = max(timestamps.values())
    return timestamps, drift_ms, common_t0_ns


def _sync_event_brief(rig: MultiFemtoBoltRig) -> str:
    event = getattr(rig, "last_sync_event", None)
    if not isinstance(event, dict):
        return "no sync diagnostics yet"
    parts = [f"reason={event.get('drop_reason') or 'none'}"]
    best_pool = event.get("best_pool_match")
    if isinstance(best_pool, dict):
        best_drift_ms = best_pool.get("best_drift_ms")
        if best_drift_ms is not None:
            parts.append(f"best_pool_drift={float(best_drift_ms):.3f}ms")
        candidate_counts = best_pool.get("candidate_counts")
        if candidate_counts:
            parts.append(f"candidates={candidate_counts}")
    for key in ("chosen_pre_conversion_drift_ms", "pre_conversion_drift_ms"):
        value = event.get(key)
        if value is not None:
            parts.append(f"{key}={float(value):.3f}ms")
            break
    selected_drifts = event.get("selected_source_drifts_ms")
    if isinstance(selected_drifts, dict) and selected_drifts:
        compact = ", ".join(
            f"{name}={float(value):.3f}ms"
            for name, value in sorted(selected_drifts.items())[:3]
        )
        parts.append(f"source_drifts=({compact})")
    return " ".join(parts)


def _print_record_status(message: str, *args: Any) -> None:
    text = message % args if args else message
    print(f"[record] {text}", flush=True)


def _wait_for_record_ready_t0(
    rig: MultiFemtoBoltRig,
    *,
    timeout_s: float,
    log_interval_s: float,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("--record-ready-timeout-s must be positive")
    if log_interval_s <= 0:
        raise ValueError("--record-ready-log-interval-s must be positive")

    start = time.perf_counter()
    deadline = start + timeout_s
    next_log = start + log_interval_s
    attempts = 0
    dropped = 0
    last_error = ""
    _print_record_status(
        "Record status: phase=ready elapsed=0.0s/%.1fs attempts=0 dropped=0 "
        "bag_writing=not_started",
        timeout_s,
    )
    while True:
        attempts += 1
        try:
            frames = rig.read_synchronized_frames()
        except RuntimeError as exc:
            dropped += 1
            last_error = str(exc)
            now = time.perf_counter()
            if now >= deadline:
                raise RuntimeError(
                    "Timed out waiting for record-ready synchronized T0 before creating bag recorders. "
                    f"attempts={attempts} dropped={dropped} last_error={last_error!r} "
                    f"latest_sync='{_sync_event_brief(rig)}'. Check Orbbec metadata timestamps, "
                    "hardware sync cabling/modes, and camera stream readiness."
                ) from exc
            if now >= next_log:
                _print_record_status(
                    "Record status: phase=ready elapsed=%.1fs/%.1fs attempts=%d dropped=%d "
                    "bag_writing=not_started last_error=%s sync=%s",
                    now - start,
                    timeout_s,
                    attempts,
                    dropped,
                    last_error,
                    _sync_event_brief(rig),
                )
                next_log = now + log_interval_s
            continue
        except StopIteration as exc:
            raise RuntimeError("Camera stream ended before record-ready synchronized T0 was found") from exc

        timestamps, drift_ms, t0_ns = _frame_timestamp_stats(frames)
        elapsed_s = time.perf_counter() - start
        _print_record_status(
            "Record status: phase=ready complete=true elapsed=%.2fs attempts=%d dropped=%d "
            "t0_ns=%d drift=%.3fms timestamps=%s",
            elapsed_s,
            attempts,
            dropped,
            t0_ns,
            drift_ms,
            timestamps,
        )
        return {
            "ready_t0_ns": t0_ns,
            "ready_t0_timestamps_ns": timestamps,
            "ready_t0_drift_ms": drift_ms,
            "ready_attempts": attempts,
            "ready_dropped": dropped,
            "ready_elapsed_s": elapsed_s,
        }


def _powershell_arg(value: str | Path) -> str:
    text = str(value)
    if text == "":
        return "''"
    if re.search(r"[\s'`$&|<>;(){}\[\]]", text):
        return "'" + text.replace("'", "''") + "'"
    return text


def _record_replay_command(config_path: Path, bag_session: Path) -> str:
    replay_out = Path("runs") / f"{bag_session.name}_replay"
    parts: list[str | Path] = [
        "conda",
        "run",
        "-n",
        "breastseg",
        "python",
        "-m",
        "anatomy_guided_recon3d.cli",
        "replay",
        "--config",
        config_path,
        "--bag-session",
        bag_session,
        "--out",
        replay_out,
        "--sync-diagnostics",
        "--probe-frames",
        "180",
        "--wait-timeout-ms",
        "5000",
    ]
    return " ".join(_powershell_arg(part) for part in parts)


def _format_size_bytes(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def _bag_size_status(bag_paths: dict[str, Path]) -> str:
    parts: list[str] = []
    for cam_id, bag_path in sorted(bag_paths.items()):
        try:
            size_text = _format_size_bytes(bag_path.stat().st_size)
        except FileNotFoundError:
            size_text = "pending"
        parts.append(f"{cam_id}={size_text}")
    return " ".join(parts)


def _drift_status(drift_values_ms: list[float]) -> str:
    if not drift_values_ms:
        return "n/a"
    latest = drift_values_ms[-1]
    return (
        f"last={latest:.3f}ms "
        f"min={min(drift_values_ms):.3f}ms "
        f"max={max(drift_values_ms):.3f}ms "
        f"mean={sum(drift_values_ms) / len(drift_values_ms):.3f}ms"
    )


def _log_recording_status(
    *,
    phase: str = "recording",
    start_perf: float,
    duration_s: float,
    accepted: int,
    dropped: int,
    drift_values_ms: list[float],
    bag_paths: dict[str, Path],
    last_error: str | None = None,
) -> None:
    elapsed_s = max(0.0, time.perf_counter() - start_perf)
    remaining_s = max(0.0, duration_s - elapsed_s)
    total = accepted + dropped
    accepted_fps = accepted / elapsed_s if elapsed_s > 0 else 0.0
    accept_rate = (accepted / total * 100.0) if total > 0 else 0.0
    last_error_text = f" last_error={last_error[:180]}" if last_error else ""
    _print_record_status(
        "Record status: phase=%s elapsed=%.1fs/%.1fs remaining=%.1fs "
        "accepted=%d dropped=%d accepted_fps=%.2f accept_rate=%.1f%% drift=(%s) bags=(%s)%s",
        phase,
        elapsed_s,
        duration_s,
        remaining_s,
        accepted,
        dropped,
        accepted_fps,
        accept_rate,
        _drift_status(drift_values_ms),
        _bag_size_status(bag_paths),
        last_error_text,
    )


def _probe_orbbec_bag_timestamps(
    bags: dict[str, Path],
    *,
    align_mode: str,
    wait_timeout_ms: int,
    playback_rate: float | None,
    probe_frames: int,
    max_drift_ms: float,
) -> tuple[int, dict[str, Any]]:
    """Probe SDK playback timestamps and choose the common T0 crop timestamp."""
    if len(bags) == 0:
        raise ValueError("No Orbbec bag files provided")
    samples: dict[str, list[dict[str, Any]]] = {}
    for cam_id, bag_path in sorted(bags.items()):
        camera = OrbbecBagCamera(
            cam_id,
            bag_path,
            align_mode=align_mode,
            wait_timeout_ms=wait_timeout_ms,
            playback_rate=playback_rate,
        )
        camera.start()
        try:
            cam_samples: list[dict[str, Any]] = []
            while len(cam_samples) < probe_frames:
                try:
                    frameset = camera.grab_next_frameset()
                except StopIteration:
                    break
                diagnostics = camera.frameset_diagnostics(frameset)
                if diagnostics.get("complete") and int(diagnostics.get("device_timestamp_ns") or 0) > 0:
                    cam_samples.append(diagnostics)
            samples[cam_id] = cam_samples
        finally:
            camera.stop()

    missing = [cam_id for cam_id, values in samples.items() if not values]
    if missing:
        raise RuntimeError(f"No timestamped complete frames found in bags: {missing}")

    families = {
        cam_id: {
            _timestamp_source_family(str(sample.get("timestamp_source")))
            for sample in values[: min(len(values), probe_frames)]
        }
        for cam_id, values in samples.items()
    }
    bad = {
        cam_id: sorted(values)
        for cam_id, values in families.items()
        if not values or values <= {"none"} or "system" in values
    }
    if bad:
        raise RuntimeError(
            "Bag replay timestamps are not reliable for sync; "
            f"system/empty timestamp sources found: {bad}"
        )
    non_system_families = {family for values in families.values() for family in values if family != "none"}
    if len(non_system_families) > 1:
        raise RuntimeError(
            "Bag replay timestamps mix clock families and cannot be compared safely: "
            f"{ {cam_id: sorted(values) for cam_id, values in families.items()} }"
        )

    for cam_id, values in samples.items():
        timestamps = [int(value["device_timestamp_ns"]) for value in values]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"Bag timestamps are not strictly increasing for {cam_id}")

    first_timestamps = {
        cam_id: int(values[0]["device_timestamp_ns"])
        for cam_id, values in samples.items()
    }
    t0_floor_ns = max(first_timestamps.values())
    timestamps_by_cam = {
        cam_id: [int(value["device_timestamp_ns"]) for value in values]
        for cam_id, values in samples.items()
    }
    candidate_targets = sorted(
        {
            timestamp_ns
            for values in timestamps_by_cam.values()
            for timestamp_ns in values
            if timestamp_ns >= t0_floor_ns
        }
    )
    best_target_ns: int | None = None
    best_after_t0: dict[str, int] | None = None
    best_after_sources: dict[str, str] | None = None
    best_drift_ms: float | None = None
    for target_ns in candidate_targets:
        candidate_after: dict[str, int] = {}
        candidate_sources: dict[str, str] = {}
        complete = True
        for cam_id, timestamps in timestamps_by_cam.items():
            index = bisect_left(timestamps, target_ns)
            if index >= len(timestamps):
                complete = False
                break
            candidate_after[cam_id] = timestamps[index]
            candidate_sources[cam_id] = str(samples[cam_id][index].get("timestamp_source"))
        if not complete:
            continue
        candidate_drift_ms = (max(candidate_after.values()) - min(candidate_after.values())) / 1_000_000.0
        if best_drift_ms is None or candidate_drift_ms < best_drift_ms:
            best_target_ns = target_ns
            best_after_t0 = candidate_after
            best_after_sources = candidate_sources
            best_drift_ms = candidate_drift_ms
        if candidate_drift_ms <= max_drift_ms:
            best_target_ns = target_ns
            best_after_t0 = candidate_after
            best_after_sources = candidate_sources
            best_drift_ms = candidate_drift_ms
            break

    if best_target_ns is None or best_after_t0 is None or best_after_sources is None or best_drift_ms is None:
        raise RuntimeError(
            f"Probe did not include a complete common replay window; "
            f"increase --probe-frames (current {probe_frames})"
        )

    t0_ns = best_target_ns
    first_after_t0 = best_after_t0
    first_after_sources = best_after_sources
    drift_ms = best_drift_ms
    if drift_ms > max_drift_ms:
        raise RuntimeError(
            f"Best common-T0 bag timestamp drift {drift_ms:.3f} ms exceeds limit {max_drift_ms:.3f} ms "
            f"within --probe-frames={probe_frames}; increase --probe-frames or inspect hardware sync"
        )

    summary = {
        "probe_frames": probe_frames,
        "first_timestamps_ns": first_timestamps,
        "t0_floor_ns": t0_floor_ns,
        "t0_ns": t0_ns,
        "first_after_t0_ns": first_after_t0,
        "first_after_t0_sources": first_after_sources,
        "first_after_t0_drift_ms": drift_ms,
        "timestamp_source_families": {cam_id: sorted(values) for cam_id, values in families.items()},
    }
    return t0_ns, summary


def _build_orbbec_bag_rig(
    bags: dict[str, Path],
    *,
    config: Recon3DConfig,
    align_mode: str,
    wait_timeout_ms: int,
    playback_rate: float | None,
    probe_frames: int,
    sync_diagnostics_path: Path | None,
) -> tuple[MultiFemtoBoltRig, dict[str, Any]]:
    t0_ns, summary = _probe_orbbec_bag_timestamps(
        bags,
        align_mode=align_mode,
        wait_timeout_ms=wait_timeout_ms,
        playback_rate=playback_rate,
        probe_frames=probe_frames,
        max_drift_ms=config.sync_max_drift_ms,
    )
    cameras = [
        OrbbecBagCamera(
            cam_id,
            bag_path,
            align_mode=align_mode,
            wait_timeout_ms=wait_timeout_ms,
            playback_rate=playback_rate,
            start_timestamp_ns=t0_ns,
        )
        for cam_id, bag_path in sorted(bags.items())
    ]
    rig = MultiFemtoBoltRig(
        cameras,
        max_timestamp_drift_ms=config.sync_max_drift_ms,
        sync_diagnostics_path=sync_diagnostics_path,
    )
    return rig, summary


def _optional_pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive")
    return width, height


def _get_list(value: Any, name: str) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _hardware_camera_specs(args: argparse.Namespace, raw_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Build hardware camera specs from config plus CLI overrides.

    Input shape/unit/frame: config values are YAML/JSON mappings; serials are
    Orbbec device serial numbers.
    Return type: list of ``(cam_id, FemtoBoltCamera kwargs)``.
    """
    hardware_raw = raw_config.get("hardware", {})
    if hardware_raw is None:
        hardware_raw = {}
    if not isinstance(hardware_raw, dict):
        raise ValueError("hardware config must be a mapping")

    align_mode = args.align_mode or str(hardware_raw.get("align_mode", "hw"))
    color_format = args.color_format or str(hardware_raw.get("color_format", "RGB"))
    color_fps = args.color_fps if args.color_fps is not None else int(hardware_raw.get("color_fps", 0))
    color_size = _optional_pair(args.color_size if args.color_size else hardware_raw.get("color_size"), "color_size")
    color_auto_exposure = (
        args.color_auto_exposure
        if getattr(args, "color_auto_exposure", None) is not None
        else hardware_raw.get("color_auto_exposure")
    )
    color_exposure = (
        args.color_exposure
        if getattr(args, "color_exposure", None) is not None
        else hardware_raw.get("color_exposure")
    )
    color_gain = args.color_gain if getattr(args, "color_gain", None) is not None else hardware_raw.get("color_gain")
    apply_sync_config = bool(hardware_raw.get("apply_sync_config", True)) and not bool(args.no_sync_config)
    wait_timeout_ms = int(hardware_raw.get("wait_timeout_ms", 1000))

    cameras_raw = _get_list(hardware_raw.get("cameras"), "hardware.cameras")
    specs: list[tuple[str, dict[str, Any]]] = []
    if cameras_raw and not args.cameras:
        for index, entry in enumerate(cameras_raw):
            if not isinstance(entry, dict):
                raise ValueError("hardware.cameras entries must be mappings")
            cam_id = str(entry.get("id") or entry.get("cam_id") or f"cam{index}")
            specs.append(
                (
                    cam_id,
                    {
                        "serial_number": entry.get("serial_number") or entry.get("serial"),
                        "sync_mode": str(entry.get("sync_mode", hardware_raw.get("sync_mode", "standalone"))),
                        "align_mode": str(entry.get("align_mode", align_mode)),
                        "color_size": _optional_pair(entry.get("color_size", color_size), "camera.color_size"),
                        "color_fps": int(entry.get("color_fps", color_fps)),
                        "color_format": str(entry.get("color_format", color_format)),
                        "color_auto_exposure": entry.get("color_auto_exposure", color_auto_exposure),
                        "color_exposure": entry.get("color_exposure", color_exposure),
                        "color_gain": entry.get("color_gain", color_gain),
                        "wait_timeout_ms": int(entry.get("wait_timeout_ms", wait_timeout_ms)),
                        "apply_sync_config": bool(entry.get("apply_sync_config", apply_sync_config)),
                    },
                )
            )
    else:
        cam_ids = args.cameras or hardware_raw.get("camera_ids") or ["cam0", "cam1", "cam2"]
        cam_ids = [str(cam_id) for cam_id in cam_ids]
        serials = args.serials or hardware_raw.get("serials") or [None] * len(cam_ids)
        sync_modes = args.sync_modes or hardware_raw.get("sync_modes") or ["standalone"] * len(cam_ids)
        if len(serials) != len(cam_ids):
            raise ValueError("--serials / hardware.serials must match camera count")
        if len(sync_modes) != len(cam_ids):
            raise ValueError("--sync-modes / hardware.sync_modes must match camera count")
        for cam_id, serial, sync_mode in zip(cam_ids, serials, sync_modes):
            specs.append(
                (
                    cam_id,
                    {
                        "serial_number": serial,
                        "sync_mode": str(sync_mode),
                        "align_mode": align_mode,
                        "color_size": color_size,
                        "color_fps": color_fps,
                        "color_format": color_format,
                        "color_auto_exposure": color_auto_exposure,
                        "color_exposure": color_exposure,
                        "color_gain": color_gain,
                        "wait_timeout_ms": wait_timeout_ms,
                        "apply_sync_config": apply_sync_config,
                    },
                )
            )

    if len(specs) > 1 and any(not kwargs.get("serial_number") for _, kwargs in specs):
        raise ValueError("Multi-camera hardware run requires one serial_number per camera")
    return specs


def _parse_board_spec(
    board_name: str,
    square_length_m: float | None = None,
    marker_length_m: float | None = None,
    dictionary: str = "DICT_5X5_100",
) -> CharucoBoardSpec:
    match = BOARD_NAME_RE.match(board_name)
    if match is None:
        raise ValueError(
            f"Unsupported board name '{board_name}'. Expected format like charuco_5x7_40mm"
        )
    square_m = square_length_m or (float(match.group("square_mm")) / 1000.0)
    marker_m = marker_length_m or (square_m * 0.75)
    return CharucoBoardSpec(
        squares_x=int(match.group("x")),
        squares_y=int(match.group("y")),
        square_length_m=float(square_m),
        marker_length_m=float(marker_m),
        dictionary_name=_parse_dictionary_name(dictionary),
    )


def _parse_dictionary_name(dictionary: str) -> int:
    name = dictionary.upper()
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"
    value = getattr(cv2.aruco, name, None)
    if value is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary}")
    return int(value)


def _read_calibration_frame_set(cameras: list[FemtoBoltCamera], settle_frames: int) -> dict[str, Any]:
    for _ in range(max(0, settle_frames)):
        for camera in cameras:
            camera.read_aligned_frame()
    return {camera.cam_id: camera.read_aligned_frame() for camera in cameras}


def _read_calibration_frame_set_sequential(
    camera_specs: list[tuple[str, dict[str, Any]]],
    settle_frames: int,
) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for cam_id, kwargs in camera_specs:
        camera_kwargs = dict(kwargs)
        camera_kwargs["apply_sync_config"] = False
        camera = FemtoBoltCamera(cam_id, **camera_kwargs)
        camera.start()
        try:
            for _ in range(max(0, settle_frames)):
                camera.read_aligned_frame()
            frames[cam_id] = camera.read_aligned_frame()
        finally:
            camera.stop()
    return frames


def _sync_diagnostics_path(args: argparse.Namespace, output_dir: Path | None) -> Path | None:
    if not getattr(args, "sync_diagnostics", False):
        return None
    raw_path = getattr(args, "sync_diagnostics_path", None)
    if raw_path:
        return Path(raw_path)
    if output_dir is not None:
        return Path(output_dir) / "sync_diagnostics.jsonl"
    return Path("sync_diagnostics.jsonl")


def _charuco_capture_score(
    required_camera_ids: set[str],
    poses: dict[str, Any],
    summary: dict[str, float | int],
    board_spec: CharucoBoardSpec,
) -> float:
    max_corners_per_camera = (board_spec.squares_x - 1) * (board_spec.squares_y - 1)
    detected = set(poses)
    missing_count = len(required_camera_ids - detected)
    total_corners = sum(min(int(pose.num_corners), max_corners_per_camera) for pose in poses.values())
    expected_corners = len(required_camera_ids) * max_corners_per_camera
    corner_deficit = max(0, expected_corners - total_corners)
    return (
        1000.0 * missing_count
        + 10.0 * corner_deficit
        + float(summary["mean_reprojection_error_px"])
        + 0.1 * float(summary["max_reprojection_error_px"])
    )


def cmd_replay(args: argparse.Namespace) -> None:
    """Run reconstruction from saved RGB-D data.

    Input shape/unit/frame: SDK bag replay uses one Orbbec ``.bag`` per camera;
    legacy replay uses one RGB/depth PNG folder per camera.
    Return type: ``None``.
    """
    config_path = Path(args.config)
    raw_config = _load_yaml_or_json(config_path)
    hardware_raw = raw_config.get("hardware", {}) or {}
    if not isinstance(hardware_raw, dict):
        raise ValueError("hardware config must be a mapping")
    config, extrinsics_path = load_recon_config(config_path, Path(args.out) if args.out else None)
    extrinsics = load_extrinsics(extrinsics_path)
    bag_session = Path(args.bag_session) if args.bag_session else None
    explicit_bags = _parse_bag_entries(args.bags)
    legacy_or_session_bag = Path(args.bag) if args.bag else None
    source_count = int(bag_session is not None) + int(bool(explicit_bags)) + int(legacy_or_session_bag is not None)
    if source_count == 0:
        raise ValueError("Provide --bag-session, --bags cam0=... cam1=..., or legacy --bag directory")
    if source_count > 1:
        raise ValueError("Use exactly one replay source: --bag-session, --bags, or --bag")

    sdk_bags: dict[str, Path] = {}
    if bag_session is not None:
        sdk_bags = _load_bag_session(bag_session)
    elif explicit_bags:
        sdk_bags = explicit_bags
    elif legacy_or_session_bag is not None and (
        legacy_or_session_bag.suffix.lower() == ".json"
        or (legacy_or_session_bag.is_dir() and (legacy_or_session_bag / "session.json").is_file())
    ):
        sdk_bags = _load_bag_session(legacy_or_session_bag)

    if sdk_bags:
        align_mode = args.align_mode or str(hardware_raw.get("align_mode", "hw"))
        wait_timeout_ms = (
            int(args.wait_timeout_ms)
            if args.wait_timeout_ms is not None
            else int(hardware_raw.get("wait_timeout_ms", 1000))
        )
        rig, timestamp_summary = _build_orbbec_bag_rig(
            sdk_bags,
            config=config,
            align_mode=align_mode,
            wait_timeout_ms=wait_timeout_ms,
            playback_rate=args.playback_rate,
            probe_frames=int(args.probe_frames),
            sync_diagnostics_path=_sync_diagnostics_path(args, config.output_dir),
        )
        LOGGER.info("Orbbec bag timestamp probe: %s", json.dumps(timestamp_summary, sort_keys=True))
    elif legacy_or_session_bag is not None:
        if legacy_or_session_bag.suffix.lower() == ".bag":
            raise ValueError("Single .bag replay is ambiguous; use --bags cam_id=path.bag for every camera")
        rig = _build_replay_rig(legacy_or_session_bag, config.sync_max_drift_ms)
    else:
        raise ValueError("No replay source resolved")

    pipeline = Recon3DPipeline(config, extrinsics, rig=rig, **_pipeline_visualization_kwargs(args))
    max_frames = getattr(args, "max_frames", None)
    if max_frames is None:
        pipeline.run_stream()
        return

    rig.start()
    processed = 0
    try:
        while processed < int(max_frames):
            try:
                frames = rig.read_synchronized_frames()
            except StopIteration:
                break
            except RuntimeError as exc:
                LOGGER.warning("Skipping frame set: %s", exc)
                continue
            pipeline.run_once(frames)
            processed += 1
    finally:
        rig.stop()
    LOGGER.info("Replay processed %d frame sets", processed)


def cmd_run(args: argparse.Namespace) -> None:
    """Run reconstruction from real Femto Bolt cameras.

    Input shape/unit/frame: hardware frames must be aligned RGB-D with depth mm.
    Return type: ``None``.
    """
    config_path = Path(args.config)
    raw_config = _load_yaml_or_json(config_path)
    config, extrinsics_path = load_recon_config(config_path, Path(args.out) if args.out else None)
    camera_specs = _hardware_camera_specs(args, raw_config)
    if args.identity_extrinsics:
        if len(camera_specs) != 1:
            raise ValueError("--identity-extrinsics is only allowed for single-camera diagnostics")
        extrinsics = {
            cam_id: CameraExtrinsics(cam_id, np.eye(4, dtype=np.float64))
            for cam_id, _ in camera_specs
        }
    else:
        if not extrinsics_path.is_file():
            raise FileNotFoundError(
                f"Formal multi-camera reconstruction requires calibrated extrinsics: {extrinsics_path}"
            )
        extrinsics = load_extrinsics(extrinsics_path)
    cameras = [FemtoBoltCamera(cam_id, **kwargs) for cam_id, kwargs in camera_specs]
    rig = MultiFemtoBoltRig(
        cameras,
        max_timestamp_drift_ms=config.sync_max_drift_ms,
        sync_diagnostics_path=_sync_diagnostics_path(args, config.output_dir),
    )
    pipeline = Recon3DPipeline(config, extrinsics, rig=rig, **_pipeline_visualization_kwargs(args))
    if args.once:
        rig.start()
        try:
            frames = rig.read_synchronized_frames()
            pipeline.run_once(frames)
        finally:
            rig.stop()
    else:
        pipeline.run_stream()


def cmd_record(args: argparse.Namespace) -> None:
    """Record raw Orbbec SDK bag files from synchronized hardware streams.

    Input shape/unit/frame: records one SDK ``.bag`` per camera and validates
    synchronized frame sets using the same live-stream rig path.
    Return type: ``None``.
    """
    config_path = Path(args.config)
    raw_config = _load_yaml_or_json(config_path)
    config, _ = load_recon_config(config_path, Path(args.out))
    camera_specs = _hardware_camera_specs(args, raw_config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(args.warmup_frames) < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if float(args.duration) <= 0:
        raise ValueError("--duration must be positive")
    if float(args.record_ready_timeout_s) <= 0:
        raise ValueError("--record-ready-timeout-s must be positive")
    if float(args.record_ready_log_interval_s) <= 0:
        raise ValueError("--record-ready-log-interval-s must be positive")
    if float(args.record_status_interval_s) <= 0:
        raise ValueError("--record-status-interval-s must be positive")
    if float(args.record_sync_max_drift_ms) <= 0:
        raise ValueError("--record-sync-max-drift-ms must be positive")
    duration_s = float(args.duration)
    status_interval_s = float(args.record_status_interval_s)
    record_sync_max_drift_ms = float(args.record_sync_max_drift_ms)

    cameras: list[FemtoBoltCamera] = []
    manifest_cameras: list[dict[str, Any]] = []
    bag_paths: dict[str, Path] = {}
    for cam_id, kwargs in camera_specs:
        bag_path = out_dir / f"{cam_id}.bag"
        if bag_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing bag: {bag_path} (pass --overwrite)")
        if bag_path.exists():
            bag_path.unlink()
        bag_paths[cam_id] = bag_path
        camera_kwargs = dict(kwargs)
        cameras.append(FemtoBoltCamera(cam_id, **camera_kwargs))
        manifest_cameras.append(
            {
                "id": cam_id,
                "bag": bag_path.name,
                "serial_number": camera_kwargs.get("serial_number"),
                "sync_mode": camera_kwargs.get("sync_mode"),
                "align_mode": camera_kwargs.get("align_mode"),
                "color_size": list(camera_kwargs["color_size"]) if camera_kwargs.get("color_size") else None,
                "color_fps": camera_kwargs.get("color_fps"),
                "color_format": camera_kwargs.get("color_format"),
                "wait_timeout_ms": camera_kwargs.get("wait_timeout_ms"),
            }
        )

    rig = MultiFemtoBoltRig(
        cameras,
        max_timestamp_drift_ms=record_sync_max_drift_ms,
        sync_diagnostics_path=_sync_diagnostics_path(args, out_dir),
    )
    accepted = 0
    dropped = 0
    drift_values_ms: list[float] = []
    last_timestamps_ns: dict[str, int] = {}
    stream_started_at_utc = datetime.now(timezone.utc).isoformat()
    record_started_at_utc: str | None = None
    record_start_perf = 0.0
    ready_info: dict[str, Any] = {}
    warmup_accepted = 0
    warmup_dropped = 0
    stop_reason = "duration_complete"
    last_recording_error = ""
    LOGGER.info("Starting SDK bag recording stream: cameras=%s out=%s", [cam.cam_id for cam in cameras], out_dir)
    _print_record_status(
        "Live status is printed to stdout. record_sync_max_drift_ms=%.3f pipeline_sync_max_drift_ms=%.3f. "
        "If using `conda run`, add `--no-capture-output` or `--live-stream` before `-n breastseg`.",
        record_sync_max_drift_ms,
        config.sync_max_drift_ms,
    )
    rig.start()
    try:
        ready_info = _wait_for_record_ready_t0(
            rig,
            timeout_s=float(args.record_ready_timeout_s),
            log_interval_s=float(args.record_ready_log_interval_s),
        )

        for _ in range(int(args.warmup_frames)):
            try:
                rig.read_synchronized_frames()
            except RuntimeError as exc:
                warmup_dropped += 1
                LOGGER.warning("Post-T0 warmup skipped frame set: %s", exc)
            else:
                warmup_accepted += 1
        if int(args.warmup_frames) > 0:
            LOGGER.info(
                "Post-T0 warmup completed before bag writing: accepted=%d dropped=%d",
                warmup_accepted,
                warmup_dropped,
            )

        LOGGER.info(
            "Creating SDK bag recorders after synchronized T0; first written frames will be post-T0"
        )
        for camera in cameras:
            camera.begin_recording(
                bag_paths[camera.cam_id],
                compression=not bool(args.no_compression),
            )
        record_started_at_utc = datetime.now(timezone.utc).isoformat()
        record_start_perf = time.perf_counter()
        deadline = record_start_perf + duration_s
        next_status_perf = record_start_perf + status_interval_s
        LOGGER.info(
            "Recording SDK bags for %.3f seconds from ready_t0_ns=%s",
            duration_s,
            ready_info.get("ready_t0_ns"),
        )
        _log_recording_status(
            start_perf=record_start_perf,
            duration_s=duration_s,
            accepted=accepted,
            dropped=dropped,
            drift_values_ms=drift_values_ms,
            bag_paths=bag_paths,
            last_error=last_recording_error,
        )
        while time.perf_counter() < deadline:
            try:
                frames = rig.read_synchronized_frames()
            except RuntimeError as exc:
                dropped += 1
                last_recording_error = str(exc)
                LOGGER.debug("Recording skipped frame set: %s", exc)
                now = time.perf_counter()
                if now >= next_status_perf:
                    _log_recording_status(
                        start_perf=record_start_perf,
                        duration_s=duration_s,
                        accepted=accepted,
                        dropped=dropped,
                        drift_values_ms=drift_values_ms,
                        bag_paths=bag_paths,
                        last_error=last_recording_error,
                    )
                    last_recording_error = ""
                    next_status_perf = now + status_interval_s
                continue
            timestamps, drift_ms, _ = _frame_timestamp_stats(frames)
            drift_values_ms.append(drift_ms)
            last_timestamps_ns = timestamps
            accepted += 1
            now = time.perf_counter()
            if now >= next_status_perf:
                _log_recording_status(
                    start_perf=record_start_perf,
                    duration_s=duration_s,
                    accepted=accepted,
                    dropped=dropped,
                    drift_values_ms=drift_values_ms,
                    bag_paths=bag_paths,
                    last_error=last_recording_error,
                )
                last_recording_error = ""
                next_status_perf = now + status_interval_s
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if record_start_perf > 0:
            LOGGER.warning(
                "Manual stop requested with Ctrl+C; stopping recorders and writing session manifest"
            )
        else:
            LOGGER.warning(
                "Manual stop requested with Ctrl+C before bag writing started; stopping cameras"
            )
    finally:
        for camera in cameras:
            try:
                camera.pause_recording()
            except Exception:  # noqa: BLE001 - continue teardown for every camera
                LOGGER.exception("[%s] failed to pause recorder", camera.cam_id)
        rig.stop()

    elapsed_s = max(0.0, time.perf_counter() - record_start_perf) if record_start_perf > 0 else 0.0
    if record_start_perf > 0:
        _log_recording_status(
            phase="stopped",
            start_perf=record_start_perf,
            duration_s=duration_s,
            accepted=accepted,
            dropped=dropped,
            drift_values_ms=drift_values_ms,
            bag_paths=bag_paths,
            last_error=last_recording_error,
        )
    elif stop_reason == "keyboard_interrupt":
        LOGGER.warning("Stopped before bag writing started; no replayable bag session was written")
        return
    manifest = {
        "schema": "recon3d_orbbec_bag_session_v1",
        "created_at_utc": stream_started_at_utc,
        "stream_started_at_utc": stream_started_at_utc,
        "record_started_at_utc": record_started_at_utc,
        "duration_s": elapsed_s,
        "requested_duration_s": duration_s,
        "stop_reason": stop_reason,
        "manual_stop": stop_reason == "keyboard_interrupt",
        "completed": stop_reason == "duration_complete",
        "pyorbbecsdk_version": _orbbec_sdk_version(),
        "config": str(config_path),
        "sync_max_drift_ms": config.sync_max_drift_ms,
        "pipeline_sync_max_drift_ms": config.sync_max_drift_ms,
        "record_sync_max_drift_ms": record_sync_max_drift_ms,
        "record_compression": not bool(args.no_compression),
        "record_started_after_t0": True,
        "ready_t0_ns": ready_info.get("ready_t0_ns"),
        "ready_t0_timestamps_ns": ready_info.get("ready_t0_timestamps_ns", {}),
        "ready_t0_drift_ms": ready_info.get("ready_t0_drift_ms"),
        "ready_attempts": ready_info.get("ready_attempts", 0),
        "ready_dropped": ready_info.get("ready_dropped", 0),
        "ready_elapsed_s": ready_info.get("ready_elapsed_s", 0.0),
        "warmup_after_t0": {
            "requested_frame_sets": int(args.warmup_frames),
            "accepted_frame_sets": warmup_accepted,
            "dropped_frame_sets": warmup_dropped,
        },
        "cameras": manifest_cameras,
        "record_stats": {
            "accepted_frame_sets": accepted,
            "dropped_frame_sets": dropped,
            "status_interval_s": status_interval_s,
            "timestamp_drift_ms": _summarize_values(drift_values_ms),
            "last_timestamps_ns": last_timestamps_ns,
        },
    }
    manifest_path = out_dir / "session.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Recorded SDK bag session: %s accepted=%d dropped=%d drift_ms=%s",
        manifest_path,
        accepted,
        dropped,
        manifest["record_stats"]["timestamp_drift_ms"],
    )
    if accepted == 0:
        if stop_reason == "keyboard_interrupt":
            LOGGER.warning(
                "Manual stop happened before any synchronized frame set was validated; replay may fail. "
                "Inspect %s and the bag sizes before using it as a benchmark run.",
                manifest_path,
            )
        else:
            raise RuntimeError(f"Recording produced no synchronized frame sets; inspect {manifest_path}")
    replay_command = _record_replay_command(config_path, out_dir)
    LOGGER.info("Replay command:\n%s", replay_command)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Collect Charuco frames and estimate extrinsics.

    Input shape/unit/frame: RGB images are color camera pixels; output transform
    convention is p_world = T_world_cam @ p_cam_h.
    Return type: ``None``.
    """
    config_path = Path(args.config)
    raw_config = _load_yaml_or_json(config_path)
    board_spec = _parse_board_spec(args.board, args.square_length_m, args.marker_length_m, args.dictionary)
    camera_specs = _hardware_camera_specs(args, raw_config)
    required_camera_ids = {cam_id for cam_id, _ in camera_specs}
    best: tuple[float, dict[str, CameraExtrinsics], dict[str, Any], dict[str, Any]] | None = None

    LOGGER.info(
        "Starting formal Charuco calibration: board=%s dictionary=%s square=%.4fm marker=%.4fm world=%s cameras=%s",
        args.board,
        args.dictionary,
        board_spec.square_length_m,
        board_spec.marker_length_m,
        args.world,
        sorted(required_camera_ids),
    )
    if args.simultaneous:
        cameras = [FemtoBoltCamera(cam_id, **kwargs) for cam_id, kwargs in camera_specs]
        rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=float(args.max_drift_ms))
        rig.start()
        try:
            for capture_idx in range(int(args.captures)):
                frames = _read_calibration_frame_set(cameras, int(args.settle_frames))
                extrinsics, poses = estimate_extrinsics_from_charuco(frames, board_spec, world=str(args.world))
                summary = summarize_reprojection_errors(poses)
                detected = set(extrinsics)
                missing = sorted(required_camera_ids - detected)
                corners = {cam_id: pose.num_corners for cam_id, pose in poses.items()}
                LOGGER.info(
                    "Calibration capture %d/%d: detected=%s missing=%s corners=%s mean=%.3fpx max=%.3fpx",
                    capture_idx + 1,
                    args.captures,
                    sorted(detected),
                    missing,
                    corners,
                    float(summary["mean_reprojection_error_px"]),
                    float(summary["max_reprojection_error_px"]),
                )
                if missing and not args.allow_partial:
                    continue
                score = _charuco_capture_score(required_camera_ids, poses, summary, board_spec)
                if best is None or score < best[0]:
                    best = (score, extrinsics, poses, summary)
        finally:
            rig.stop()
    else:
        for capture_idx in range(int(args.captures)):
            frames = _read_calibration_frame_set_sequential(camera_specs, int(args.settle_frames))
            extrinsics, poses = estimate_extrinsics_from_charuco(frames, board_spec, world=str(args.world))
            summary = summarize_reprojection_errors(poses)
            detected = set(extrinsics)
            missing = sorted(required_camera_ids - detected)
            corners = {cam_id: pose.num_corners for cam_id, pose in poses.items()}
            LOGGER.info(
                "Calibration capture %d/%d: detected=%s missing=%s corners=%s mean=%.3fpx max=%.3fpx",
                capture_idx + 1,
                args.captures,
                sorted(detected),
                missing,
                corners,
                float(summary["mean_reprojection_error_px"]),
                float(summary["max_reprojection_error_px"]),
            )
            if missing and not args.allow_partial:
                continue
            score = _charuco_capture_score(required_camera_ids, poses, summary, board_spec)
            if best is None or score < best[0]:
                best = (score, extrinsics, poses, summary)

    if best is None:
        raise RuntimeError(
            "No valid Charuco capture contained all required cameras. "
            "Move the board so every RGB camera sees it clearly, then run calibration again."
        )

    _, extrinsics, poses, summary = best
    out_path = Path(args.out)
    save_extrinsics(out_path, extrinsics)
    report = {
        "board": {
            "name": args.board,
            "dictionary": args.dictionary,
            "squares_x": board_spec.squares_x,
            "squares_y": board_spec.squares_y,
            "square_length_m": board_spec.square_length_m,
            "marker_length_m": board_spec.marker_length_m,
        },
        "world": args.world,
        "summary": summary,
        "cameras": {
            cam_id: {
                "mean_reprojection_error_px": pose.mean_reprojection_error_px,
                "max_reprojection_error_px": pose.max_reprojection_error_px,
                "num_corners": pose.num_corners,
            }
            for cam_id, pose in poses.items()
        },
    }
    report_path = out_path.with_name(f"{out_path.stem}_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Saved calibrated extrinsics: %s", out_path)
    LOGGER.info("Saved calibration report: %s", report_path)


def _add_sync_diagnostics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sync-diagnostics",
        action="store_true",
        help="Write SDK FIFO/timestamp diagnostics to JSONL without changing capture policy.",
    )
    parser.add_argument(
        "--sync-diagnostics-path",
        default=None,
        help="JSONL path for --sync-diagnostics; defaults to <out>/sync_diagnostics.jsonl.",
    )


def _add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Publish live_manifest.json and a rerun stream for viewer/live.html.",
    )
    parser.add_argument("--rerun-web-port", type=int, default=9090, help="Port hosting rerun web assets.")
    parser.add_argument("--rerun-grpc-port", type=int, default=9876, help="Port hosting rerun data.")
    parser.add_argument("--rerun-mem", default="25%", help="rerun server memory budget.")
    parser.add_argument(
        "--rerun-log-cameras",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish per-camera depth/mask/box overlays into Rerun; RGB previews stay in the browser Cameras panel.",
    )
    parser.add_argument("--viewer-rgb-width", type=int, default=640, help="Width of RGB/mask preview images.")


def _pipeline_visualization_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "visualize": bool(getattr(args, "visualize", False)),
        "rerun_web_port": int(getattr(args, "rerun_web_port", 9090)),
        "rerun_grpc_port": int(getattr(args, "rerun_grpc_port", 9876)),
        "rerun_memory": str(getattr(args, "rerun_mem", "25%")),
        "rerun_log_cameras": getattr(args, "rerun_log_cameras", None),
        "viewer_rgb_width": int(getattr(args, "viewer_rgb_width", 640)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser.

    Input shape/unit/frame: CLI strings.
    Return type: ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description="Orbbec RGB-D multi-camera reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Collect Charuco frames and estimate extrinsics")
    calibrate.add_argument("--board", default="charuco_5x7_40mm")
    calibrate.add_argument("--config", default="configs/recon3d/default.yaml")
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--world", default="board", help="Use 'board', 'first_camera', or a camera ID.")
    calibrate.add_argument("--dictionary", default="DICT_5X5_100")
    calibrate.add_argument("--captures", type=int, default=8, help="Number of frame sets to evaluate.")
    calibrate.add_argument("--settle-frames", type=int, default=2, help="Frames to discard before each capture.")
    calibrate.add_argument("--max-drift-ms", type=float, default=1000.0)
    calibrate.add_argument("--square-length-m", type=float, default=None)
    calibrate.add_argument("--marker-length-m", type=float, default=None)
    calibrate.add_argument("--allow-partial", action="store_true", help="Write extrinsics even if not all cameras detect.")
    calibrate.add_argument("--simultaneous", action="store_true", help="Capture all cameras at the same time.")
    calibrate.add_argument("--cameras", nargs="*", default=None)
    calibrate.add_argument("--serials", nargs="*", default=None)
    calibrate.add_argument("--sync-modes", nargs="*", default=None)
    calibrate.add_argument("--align-mode", choices=["hw", "sw"], default=None)
    calibrate.add_argument("--color-size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    calibrate.add_argument("--color-fps", type=int, default=None)
    calibrate.add_argument("--color-format", default=None)
    calibrate.add_argument("--color-auto-exposure", action=argparse.BooleanOptionalAction, default=None)
    calibrate.add_argument("--color-exposure", type=int, default=None)
    calibrate.add_argument("--color-gain", type=int, default=None)
    calibrate.add_argument("--no-sync-config", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    record = sub.add_parser("record", help="Record synchronized RGB-D data")
    record.add_argument("--config", default="configs/recon3d/default.yaml")
    record.add_argument("--duration", "--duration-s", dest="duration", type=float, default=30.0)
    record.add_argument("--out", required=True)
    record.add_argument("--cameras", nargs="*", default=None)
    record.add_argument("--serials", nargs="*", default=None, help="Orbbec serials matching --cameras order.")
    record.add_argument("--sync-modes", nargs="*", default=None, help="Sync modes matching --cameras order.")
    record.add_argument("--align-mode", choices=["hw", "sw"], default=None, help="Depth-to-color alignment mode.")
    record.add_argument("--color-size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    record.add_argument("--color-fps", type=int, default=None)
    record.add_argument("--color-format", default=None)
    record.add_argument("--color-auto-exposure", action=argparse.BooleanOptionalAction, default=None)
    record.add_argument("--color-exposure", type=int, default=None)
    record.add_argument("--color-gain", type=int, default=None)
    record.add_argument("--no-sync-config", action="store_true", help="Do not write Orbbec multi-device sync config.")
    record.add_argument("--no-compression", action="store_true", help="Disable SDK bag compression.")
    record.add_argument("--warmup-frames", type=int, default=2, help="Synchronized frame sets to consume before stats.")
    record.add_argument(
        "--record-ready-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for all cameras to produce one synchronized T0 before bag writing starts.",
    )
    record.add_argument(
        "--record-ready-log-interval-s",
        type=float,
        default=2.0,
        help="Seconds between progress logs while waiting for record-ready T0.",
    )
    record.add_argument(
        "--record-status-interval-s",
        type=float,
        default=2.0,
        help="Seconds between live recording status lines after bag writing starts.",
    )
    record.add_argument(
        "--record-sync-max-drift-ms",
        type=float,
        default=1000.0,
        help=(
            "Loose drift limit used only by record-side ready/status monitoring. "
            "Replay/pipeline still uses sync_max_drift_ms from config."
        ),
    )
    record.add_argument("--overwrite", action="store_true", help="Overwrite existing cam*.bag files in --out.")
    _add_sync_diagnostics_args(record)
    record.set_defaults(func=cmd_record)

    run = sub.add_parser("run", help="Run real-time reconstruction from cameras")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=False)
    run.add_argument("--cameras", nargs="*", default=None)
    run.add_argument("--serials", nargs="*", default=None, help="Orbbec serials matching --cameras order.")
    run.add_argument("--sync-modes", nargs="*", default=None, help="Sync modes matching --cameras order.")
    run.add_argument("--align-mode", choices=["hw", "sw"], default=None, help="Depth-to-color alignment mode.")
    run.add_argument("--color-size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    run.add_argument("--color-fps", type=int, default=None)
    run.add_argument("--color-format", default=None)
    run.add_argument("--color-auto-exposure", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--color-exposure", type=int, default=None)
    run.add_argument("--color-gain", type=int, default=None)
    run.add_argument("--no-sync-config", action="store_true", help="Do not write Orbbec multi-device sync config.")
    run.add_argument("--once", action="store_true", help="Capture and reconstruct one synchronized frame set.")
    run.add_argument(
        "--identity-extrinsics",
        action="store_true",
        help="Use identity extrinsics for a single-camera diagnostic run only.",
    )
    _add_sync_diagnostics_args(run)
    _add_visualization_args(run)
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="Replay saved RGB-D data")
    replay.add_argument("--config", required=True)
    replay.add_argument("--bag", required=False, help="Legacy PNG replay dir, or SDK bag session dir/json.")
    replay.add_argument("--bag-session", default=None, help="SDK bag session directory or session.json.")
    replay.add_argument("--bags", nargs="*", default=None, help="SDK bag entries: cam0=path.bag cam1=path.bag ...")
    replay.add_argument("--out", required=False)
    replay.add_argument("--align-mode", choices=["hw", "sw"], default=None, help="Depth-to-color alignment mode.")
    replay.add_argument("--wait-timeout-ms", type=int, default=None, help="SDK playback frame wait timeout.")
    replay.add_argument("--playback-rate", type=float, default=None, help="SDK playback rate; omit for recorded rate.")
    replay.add_argument("--probe-frames", type=int, default=180, help="Frames per bag to probe before T0 crop.")
    replay.add_argument("--max-frames", type=int, default=None, help="Optional limit for replay smoke tests.")
    _add_sync_diagnostics_args(replay)
    _add_visualization_args(replay)
    replay.set_defaults(func=cmd_replay)
    return parser


def main() -> None:
    """CLI entry point.

    Input shape/unit/frame: command-line arguments.
    Return type: ``None``.
    """
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except HardwareUnavailableError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

