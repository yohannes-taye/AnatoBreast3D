from __future__ import annotations

from collections import deque
import json
import logging
import socket
import threading
import time
from typing import Mapping

import cv2
import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    FusedNipple,
    Recon3DConfig,
    ReconstructionResult,
    RGBDFrame,
)
from anatomy_guided_recon3d.calibration.io import save_extrinsics
from anatomy_guided_recon3d.geometry.cloud_cleanup import backproject_cleaned_breast_cloud
from anatomy_guided_recon3d.geometry.fusion import FusionBackend, PointCloudFusionBackend, point_cloud_bounds
from anatomy_guided_recon3d.geometry.guided_fusion import (
    apply_guided_weights_to_clouds,
    make_nipple_guided_view_decisions,
)
from anatomy_guided_recon3d.geometry.nipple_registration import (
    AnatomyRegistrationState,
    filter_side_mismatched_observations,
)
from anatomy_guided_recon3d.geometry.nipple3d import (
    assign_observation_sides,
    estimate_nipple_observations,
    fuse_nipple_observations,
    nipple_json_payload,
)
from anatomy_guided_recon3d.geometry.post_cleanup import cleanup_fused_cloud
from anatomy_guided_recon3d.geometry.quality_metrics import (
    TemporalRepeatabilityTracker,
    compute_cross_camera_surface_consistency,
)
from anatomy_guided_recon3d.geometry.surface import optional_poisson_surface_points
from anatomy_guided_recon3d.geometry.tsdf_fusion import fuse_tsdf
from anatomy_guided_recon3d.perception.inference import BreastPerceptionAdapter
from anatomy_guided_recon3d.sensors.rig import MultiFemtoBoltRig
from anatomy_guided_recon3d.smoothing.temporal import OneEuroFilter3D


LOGGER = logging.getLogger(__name__)


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _choose_rerun_ports(web_port: int, grpc_port: int) -> tuple[int, int]:
    for offset in range(20):
        candidate_web = int(web_port) + offset
        candidate_grpc = int(grpc_port) + offset
        if _port_available(candidate_web) and _port_available(candidate_grpc):
            return candidate_web, candidate_grpc
    raise OSError(f"No free rerun port pair found from {web_port}/{grpc_port}")


def _nipple_payload(nipple: FusedNipple) -> dict[str, object]:
    return {
        "mean_world_m": nipple.mean_world_m.tolist(),
        "confidence": float(nipple.confidence),
        "used_cameras": list(nipple.used_cameras),
        "rejected_cameras": list(nipple.rejected_cameras),
    }


def _build_rerun_blueprint(rrb, cam_ids: list[str], log_cameras: bool):
    if not log_cameras:
        return rrb.Blueprint(
            rrb.Spatial3DView(origin="world", name="Fused 3D"),
            collapse_panels=True,
        )
    depth_views = [rrb.Spatial2DView(origin=f"depth/{cam_id}", name=f"{cam_id} depth") for cam_id in cam_ids]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="world", name="Fused 3D"),
            rrb.Grid(contents=depth_views, grid_columns=max(1, len(cam_ids))),
            row_shares=[2.0, 1.4],
        ),
        collapse_panels=True,
    )


class Recon3DPipeline:
    """RGB-D multi-camera breast reconstruction pipeline."""

    def __init__(
        self,
        config: Recon3DConfig,
        extrinsics: Mapping[str, CameraExtrinsics],
        *,
        rig: MultiFemtoBoltRig | None = None,
        perception: BreastPerceptionAdapter | None = None,
        fusion_backend: FusionBackend | None = None,
        visualize: bool = False,
        rerun_web_port: int = 9090,
        rerun_grpc_port: int = 9876,
        rerun_memory: str = "25%",
        rerun_log_cameras: bool | None = None,
        viewer_rgb_width: int = 640,
    ) -> None:
        """Create the reconstruction pipeline.

        Input shape/unit/frame: extrinsics are camera-to-world transforms in
        metres; rig returns synchronized aligned RGB-D frames.
        Return type: ``Recon3DPipeline``.
        """
        self.config = config
        self.extrinsics = dict(extrinsics)
        self._camera_centers_world_m = [
            extrinsic.T_world_cam[:3, 3]
            for extrinsic in self.extrinsics.values()
        ]
        self.rig = rig
        self.perception = perception or BreastPerceptionAdapter(config=config)
        self.fusion_backend = fusion_backend or PointCloudFusionBackend(config.fusion)
        self._filters = {"left": OneEuroFilter3D(), "right": OneEuroFilter3D()}
        self.visualize = bool(visualize)
        self.rerun_web_port = int(rerun_web_port)
        self.rerun_grpc_port = int(rerun_grpc_port)
        self.rerun_memory = str(rerun_memory)
        self.rerun_log_cameras = (
            bool(config.visualization.rerun_log_cameras)
            if rerun_log_cameras is None
            else bool(rerun_log_cameras)
        )
        self.viewer_rgb_width = int(viewer_rgb_width)
        self._frame_index = 0
        self._rr = None
        self._rerun_url: str | None = None
        self._rerun_error: str | None = None
        self._rerun_started = False
        self._last_nipple_guided_fusion: dict[str, object] | None = None
        self._last_nipple_registration: dict[str, object] | None = None
        self._last_final_cleanup: dict[str, object] | None = None
        self._last_tsdf_fusion: dict[str, object] | None = None
        self._registration_state = AnatomyRegistrationState()
        self._repeatability_tracker = TemporalRepeatabilityTracker(config.quality_metrics)
        history_len = max(1, int(config.quality_metrics.temporal_window))
        self._nipple_anchor_history: dict[str, deque[FusedNipple]] = {
            "left": deque(maxlen=history_len),
            "right": deque(maxlen=history_len),
        }
        self._last_nipple_anchor_stabilization: dict[str, object] | None = None
        self._last_quality_metrics: dict[str, object] | None = None
        self._last_refined_extrinsics: dict[str, CameraExtrinsics] | None = None

    def _validate_frames(self, frames: Mapping[str, RGBDFrame]) -> int:
        if not frames:
            raise ValueError("No RGB-D frames provided")
        missing = sorted(set(frames) - set(self.extrinsics))
        if missing:
            raise KeyError(f"Missing extrinsics for cameras: {missing}")
        timestamps = [frame.timestamp_ns for frame in frames.values()]
        drift_ms = (max(timestamps) - min(timestamps)) / 1_000_000.0
        if drift_ms > self.config.sync_max_drift_ms:
            raise RuntimeError(f"Frame timestamp drift {drift_ms:.3f} ms exceeds limit")
        return int(round(float(np.median(timestamps))))

    def _smooth_nipples(self, nipples: dict[str, FusedNipple], timestamp_ns: int) -> dict[str, FusedNipple]:
        smoothed: dict[str, FusedNipple] = {}
        for side, nipple in nipples.items():
            if nipple.confidence <= 0.0:
                smoothed[side] = nipple
                continue
            mean = self._filters[side].filter(nipple.mean_world_m, timestamp_ns)
            smoothed[side] = FusedNipple(
                side=nipple.side,
                mean_world_m=mean,
                covariance_world_m2=nipple.covariance_world_m2,
                confidence=nipple.confidence,
                used_cameras=nipple.used_cameras,
                rejected_cameras=nipple.rejected_cameras,
            )
        return smoothed

    def _stabilize_nipples(self, nipples: dict[str, FusedNipple]) -> dict[str, FusedNipple]:
        stabilized: dict[str, FusedNipple] = {}
        diagnostics: dict[str, object] = {"enabled": True, "sides": {}}
        min_cameras = int(self.config.nipple3d.min_fused_observations)
        jitter = float(self.config.nipple3d.covariance_jitter_m2)
        for side in ("left", "right"):
            current = nipples[side]
            history = self._nipple_anchor_history[side]
            current_valid = (
                current.confidence > 0.0
                and len(current.used_cameras) >= min_cameras
                and np.isfinite(current.mean_world_m).all()
            )
            if current_valid:
                history.append(current)

            if history:
                points = np.asarray([item.mean_world_m for item in history], dtype=np.float64)
                mean = np.median(points, axis=0)
                covariances = np.asarray([item.covariance_world_m2 for item in history], dtype=np.float64)
                covariance = np.mean(covariances, axis=0)
                if points.shape[0] >= 2:
                    centered = points - mean.reshape(1, 3)
                    covariance += centered.T @ centered / max(points.shape[0] - 1, 1)
                covariance = 0.5 * (covariance + covariance.T) + np.eye(3, dtype=np.float64) * jitter
                confidence = float(np.median([item.confidence for item in history]))
                if not current_valid:
                    confidence *= 0.85
                used_cameras = sorted({cam for item in history for cam in item.used_cameras})
                rejected_cameras = list(current.rejected_cameras)
                stabilized[side] = FusedNipple(
                    side=side,
                    mean_world_m=mean,
                    covariance_world_m2=covariance,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    used_cameras=used_cameras,
                    rejected_cameras=rejected_cameras,
                )
                source = "current_temporal_median" if current_valid else "history_fallback"
            else:
                stabilized[side] = current
                source = "current_invalid_no_history"

            diagnostics["sides"][side] = {
                "source": source,
                "current_valid": bool(current_valid),
                "history_frames": int(len(history)),
                "current": _nipple_payload(current),
                "stabilized": _nipple_payload(stabilized[side]),
            }
        self._last_nipple_anchor_stabilization = diagnostics
        return stabilized

    def _write_cloud_and_nipples(self, result: ReconstructionResult) -> None:
        out_dir = self.config.output_dir
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_path = out_dir / "fused_breast.ply"
        if not o3d.io.write_point_cloud(str(ply_path), result.fused_cloud):
            raise OSError(f"Failed to write fused point cloud: {ply_path}")
        payload = nipple_json_payload(result.timestamp_ns, result.nipples)
        (out_dir / "nipple_world.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._last_nipple_guided_fusion is not None:
            (out_dir / "nipple_guided_fusion.json").write_text(
                json.dumps(self._last_nipple_guided_fusion, indent=2),
                encoding="utf-8",
            )
        if self._last_nipple_registration is not None:
            (out_dir / "nipple_registration.json").write_text(
                json.dumps(self._last_nipple_registration, indent=2),
                encoding="utf-8",
            )
        if self._last_final_cleanup is not None:
            (out_dir / "final_cleanup.json").write_text(
                json.dumps(self._last_final_cleanup, indent=2),
                encoding="utf-8",
            )
        if self._last_tsdf_fusion is not None:
            (out_dir / "tsdf_fusion.json").write_text(
                json.dumps(self._last_tsdf_fusion, indent=2),
                encoding="utf-8",
            )
        if self._last_quality_metrics is not None:
            (out_dir / "quality_metrics.json").write_text(
                json.dumps(self._last_quality_metrics, indent=2),
                encoding="utf-8",
            )
        if (
            self._last_refined_extrinsics is not None
            and self._last_nipple_registration is not None
            and bool(self._last_nipple_registration.get("all_frozen", False))
        ):
            save_extrinsics(out_dir / "extrinsics_refined.json", self._last_refined_extrinsics)

    def _write_summary(self, result: ReconstructionResult) -> None:
        out_dir = self.config.output_dir
        if out_dir is None:
            return
        summary = {
            "timestamp_ns": result.timestamp_ns,
            "timings_ms": result.timings_ms,
            "fused_cloud": point_cloud_bounds(result.fused_cloud),
            "metrics": {
                "timestamp_drift_ms": result.timings_ms.get("timestamp_drift_ms"),
                "calibration_mean_reprojection_error_px": "not_measured",
                "calibration_max_reprojection_error_px": "not_measured",
                "phantom_repeatability_std_mm": "not_measured",
                "single_frame_latency_ms": result.timings_ms.get("total_ms"),
            },
        }
        if self._last_nipple_guided_fusion is not None:
            summary["nipple_guided_fusion"] = self._last_nipple_guided_fusion
        if self._last_nipple_registration is not None:
            summary["nipple_registration"] = self._last_nipple_registration
        if self._last_final_cleanup is not None:
            summary["final_cleanup"] = self._last_final_cleanup
        if self._last_tsdf_fusion is not None:
            summary["tsdf_fusion"] = self._last_tsdf_fusion
        if self._last_quality_metrics is not None:
            summary["quality_metrics"] = self._last_quality_metrics
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _ensure_rerun_stream(self, cam_ids: list[str]) -> None:
        if self._rerun_started or not self.visualize:
            return
        self._rerun_started = True
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            web_port, grpc_port = _choose_rerun_ports(self.rerun_web_port, self.rerun_grpc_port)
            rr.init("breast_recon3d")
            blueprint = _build_rerun_blueprint(rrb, cam_ids, self.rerun_log_cameras)

            def serve_rerun() -> None:
                try:
                    rr.serve_web(
                        open_browser=False,
                        web_port=web_port,
                        grpc_port=grpc_port,
                        default_blueprint=blueprint,
                        server_memory_limit=self.rerun_memory,
                    )
                except Exception as exc:  # noqa: BLE001 - reported through manifest/logs
                    self._rerun_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning("Recon3D rerun server stopped: %s", self._rerun_error)

            threading.Thread(
                target=serve_rerun,
                name="recon3d-rerun-server",
                daemon=True,
            ).start()
            time.sleep(0.2)
            if self._rerun_error:
                raise RuntimeError(self._rerun_error)
            self._rr = rr
            self._rerun_url = (
                f"http://127.0.0.1:{web_port}"
                f"?url=rerun%2Bhttp://127.0.0.1:{grpc_port}/proxy"
            )
            LOGGER.info("Recon3D rerun stream started at %s", self._rerun_url)
        except Exception as exc:  # noqa: BLE001 - visualization must not break reconstruction
            self._rr = None
            self._rerun_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Recon3D rerun stream disabled: %s", self._rerun_error)

    def _publish_rerun_frame(
        self,
        frames: Mapping[str, RGBDFrame],
        perceptions: Mapping[str, object],
        result: ReconstructionResult,
        frame_wall_s: float,
    ) -> None:
        rr = self._rr
        if rr is None:
            return
        rr.set_time("frame", sequence=self._frame_index)
        rr.set_time("time", timestamp=frame_wall_s)
        if self.rerun_log_cameras:
            for cam_id, frame in sorted(frames.items()):
                perception = perceptions.get(cam_id)
                depth_units_per_meter = 1.0 / frame.depth_config.depth_scale_m
                rr.log(
                    f"depth/{cam_id}",
                    rr.DepthImage(
                        frame.depth_mm,
                        meter=depth_units_per_meter,
                        colormap="turbo",
                        depth_range=[self.config.cloud_cleanup.min_depth_m, self.config.cloud_cleanup.max_depth_m],
                    ),
                )
                if perception is None:
                    continue
                rr.log(f"cams/{cam_id}/seg", rr.SegmentationImage(perception.breast_mask.astype(np.uint8), opacity=0.5))
                boxes = np.asarray(perception.nipple_boxes_xyxy, dtype=np.float32)
                if len(boxes):
                    labels = [f"nipple {float(score):.2f}" for score in perception.nipple_scores]
                    rr.log(
                        f"cams/{cam_id}/nipple",
                        rr.Boxes2D(
                            mins=boxes[:, :2],
                            sizes=boxes[:, 2:] - boxes[:, :2],
                            colors=[(255, 64, 64)] * len(boxes),
                            labels=labels,
                        ),
                    )
                else:
                    rr.log(
                        f"cams/{cam_id}/nipple",
                        rr.Boxes2D(mins=np.zeros((0, 2), dtype=np.float32), sizes=np.zeros((0, 2), dtype=np.float32)),
                    )

        points = np.asarray(result.fused_cloud.points)
        colors = np.asarray(result.fused_cloud.colors)
        kwargs = {"radii": 0.0015}
        if len(points) and len(colors) == len(points):
            kwargs["colors"] = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        rr.log("world/breast", rr.Points3D(points, **kwargs))
        for name, value in result.timings_ms.items():
            rr.log(f"timing/{name}", rr.Scalars(float(value)))

    def _publish_viewer_frame(
        self,
        frames: Mapping[str, RGBDFrame],
        perceptions: Mapping[str, object],
        result: ReconstructionResult,
        frame_wall_s: float,
    ) -> None:
        out_dir = self.config.output_dir
        if out_dir is None or not self.visualize:
            return
        live_dir = out_dir / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        cam_ids = sorted(frames)
        self._ensure_rerun_stream(cam_ids)

        display: dict[str, dict[str, int]] = {}
        perception_payload: dict[str, dict[str, object]] = {}
        rgb_paths: dict[str, str] = {}
        mask_paths: dict[str, str] = {}
        depth_paths: dict[str, str] = {}
        target_width = max(1, int(self.viewer_rgb_width))
        depth_min_m = float(self.config.cloud_cleanup.min_depth_m)
        depth_max_m = float(self.config.cloud_cleanup.max_depth_m)
        depth_span_m = max(depth_max_m - depth_min_m, 1e-6)
        for cam_id, frame in sorted(frames.items()):
            scale = target_width / float(frame.rgb.shape[1])
            target_height = max(1, int(round(frame.rgb.shape[0] * scale)))
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(live_dir / f"{cam_id}_rgb.jpg"),
                cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            rgb_paths[cam_id] = f"live/{cam_id}_rgb.jpg"
            display[cam_id] = {
                "width": target_width,
                "height": target_height,
                "source_width": int(frame.rgb.shape[1]),
                "source_height": int(frame.rgb.shape[0]),
            }
            depth_m = frame.depth_mm.astype(np.float32) * float(frame.depth_config.depth_scale_m)
            depth_valid = np.isfinite(depth_m) & (depth_m > 0)
            depth_norm = np.clip((depth_m - depth_min_m) / depth_span_m, 0.0, 1.0)
            depth_u8 = ((1.0 - depth_norm) * 255.0).astype(np.uint8)
            depth_u8[~depth_valid] = 0
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            depth_color[~depth_valid] = 0
            cv2.imwrite(
                str(live_dir / f"{cam_id}_depth.jpg"),
                cv2.resize(depth_color, (target_width, target_height), interpolation=cv2.INTER_NEAREST),
                [cv2.IMWRITE_JPEG_QUALITY, 86],
            )
            depth_paths[cam_id] = f"live/{cam_id}_depth.jpg"
            perception = perceptions.get(cam_id)
            if perception is None:
                continue
            mask = perception.breast_mask.astype(np.uint8)
            cv2.imwrite(
                str(live_dir / f"{cam_id}_mask.png"),
                cv2.resize(mask * 255, (target_width, target_height), interpolation=cv2.INTER_NEAREST),
            )
            mask_paths[cam_id] = f"live/{cam_id}_mask.png"
            perception_payload[cam_id] = {
                "mask_pixels": int(mask.sum()),
                "mask_bbox_xyxy": _mask_bbox(mask.astype(bool)),
                "nipple_boxes_xyxy": np.asarray(perception.nipple_boxes_xyxy, dtype=float).tolist(),
                "nipple_scores": np.asarray(perception.nipple_scores, dtype=float).tolist(),
            }

        try:
            self._publish_rerun_frame(frames, perceptions, result, frame_wall_s)
        except Exception as exc:  # noqa: BLE001 - keep static viewer assets usable
            self._rerun_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Recon3D rerun frame publish failed: %s", self._rerun_error)

        total_ms = float(result.timings_ms.get("total_ms") or 0.0)
        manifest = {
            "timestamp": frame_wall_s,
            "frame": self._frame_index,
            "points": int(len(result.fused_cloud.points)),
            "fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0,
            "cameras": cam_ids,
            "rerun_url": self._rerun_url,
            "rgb": rgb_paths,
            "mask": mask_paths,
            "depth": depth_paths,
            "display": display,
            "perception": perception_payload,
            "timings_ms": result.timings_ms,
        }
        if self._last_nipple_guided_fusion is not None:
            manifest["nipple_guided_fusion"] = self._last_nipple_guided_fusion
        if self._last_nipple_registration is not None:
            manifest["nipple_registration"] = self._last_nipple_registration
        if self._last_final_cleanup is not None:
            manifest["final_cleanup"] = self._last_final_cleanup
        if self._last_tsdf_fusion is not None:
            manifest["tsdf_fusion"] = self._last_tsdf_fusion
        if self._last_quality_metrics is not None:
            manifest["quality_metrics"] = self._last_quality_metrics
        if self._rerun_error:
            manifest["rerun_error"] = self._rerun_error
        (out_dir / "live_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._frame_index += 1

    def run_once(self, frames: Mapping[str, RGBDFrame]) -> ReconstructionResult:
        """Run one synchronized RGB-D frame set through reconstruction.

        Input shape/unit/frame: frames are aligned RGB HxWx3 and depth HxW raw
        millimetres; output cloud/nipples are world-frame metres.
        Return type: ``ReconstructionResult``.
        """
        frame_wall_s = time.time()
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        timestamp_ns = self._validate_frames(frames)
        timestamps = [frame.timestamp_ns for frame in frames.values()]
        timings["timestamp_drift_ms"] = (max(timestamps) - min(timestamps)) / 1_000_000.0

        start = time.perf_counter()
        perceptions = self.perception.predict_batch(frames)
        timings["perception_ms"] = _now_ms(start)

        start = time.perf_counter()
        observations = estimate_nipple_observations(frames, perceptions, self.extrinsics, self.config.nipple3d)
        observations = assign_observation_sides(observations, self.config.nipple3d.left_right_axis_world)
        nipples = fuse_nipple_observations(observations, self.config.nipple3d)
        timings["nipple3d_ms"] = _now_ms(start)

        start = time.perf_counter()
        if self.config.anatomy_registration.mode == "off":
            gated_observations = observations
            side_gate = {"enabled": False, "reason": "registration_mode_off"}
        else:
            gated_observations, side_gate = filter_side_mismatched_observations(
                observations,
                nipples,
                self.config.anatomy_registration,
            )
            if len(gated_observations) != len(observations):
                nipples = fuse_nipple_observations(gated_observations, self.config.nipple3d)
        timings["side_gate_ms"] = _now_ms(start)

        start = time.perf_counter()
        clouds = {
            cam_id: backproject_cleaned_breast_cloud(
                frame,
                perceptions[cam_id].breast_mask,
                self.config.cloud_cleanup,
            )
            for cam_id, frame in frames.items()
            if cam_id in perceptions
        }
        timings["backproject_ms"] = _now_ms(start)

        start = time.perf_counter()
        refined_extrinsics, corrected_observations, corrected_nipples, registration_diag = self._registration_state.refine(
            clouds,
            self.extrinsics,
            gated_observations,
            nipples,
            self.config.anatomy_registration,
            self.config.nipple3d,
        )
        registration_diag["side_gate"] = side_gate
        self._last_nipple_registration = registration_diag
        self._last_refined_extrinsics = refined_extrinsics
        timings["registration_ms"] = _now_ms(start)

        stabilized_nipples = (
            corrected_nipples
            if self.config.anatomy_registration.mode == "off"
            else self._stabilize_nipples(corrected_nipples)
        )

        start = time.perf_counter()
        if self.config.anatomy_registration.mode == "off":
            view_decisions = {
                cam_id: {
                    "accepted": True,
                    "weight": 1.0,
                    "reason": "registration_mode_off",
                    "best_error_m": None,
                    "observations": [],
                }
                for cam_id in frames
            }
            weighted_clouds = dict(clouds)
            guided_point_counts = {cam_id: len(cloud.points) for cam_id, cloud in weighted_clouds.items()}
        else:
            view_decisions = make_nipple_guided_view_decisions(
                list(frames.keys()),
                corrected_observations,
                stabilized_nipples,
                self.config.nipple_guided_fusion,
            )
            weighted_clouds, guided_point_counts = apply_guided_weights_to_clouds(
                clouds,
                view_decisions,
                self.config.nipple_guided_fusion,
            )
        timings["guided_decisions_ms"] = _now_ms(start)
        self._last_nipple_guided_fusion = {
            "enabled": self.config.anatomy_registration.mode != "off" and bool(self.config.nipple_guided_fusion.enabled),
            "mode": self.config.anatomy_registration.mode,
            "initial_fused_nipple_anchors": {
                side: _nipple_payload(nipple)
                for side, nipple in nipples.items()
            },
            "corrected_fused_nipple_anchors": {
                side: _nipple_payload(nipple)
                for side, nipple in corrected_nipples.items()
            },
            "stabilized_fused_nipple_anchors": {
                side: _nipple_payload(nipple)
                for side, nipple in stabilized_nipples.items()
            },
            "nipple_anchor_stabilization": self._last_nipple_anchor_stabilization,
            "view_decisions": view_decisions,
            "weighted_point_counts": guided_point_counts,
        }

        start = time.perf_counter()
        self._last_tsdf_fusion = {
            "enabled": bool(self.config.tsdf_fusion.enabled),
            "source": "pointcloud",
        }
        if self.config.tsdf_fusion.enabled:
            tsdf_frames = {
                cam_id: frame
                for cam_id, frame in frames.items()
                if bool(view_decisions.get(cam_id, {}).get("accepted", False))
                and float(view_decisions.get(cam_id, {}).get("weight", 0.0) or 0.0) > 0.0
            }
            tsdf_perceptions = {
                cam_id: perceptions[cam_id]
                for cam_id in tsdf_frames
                if cam_id in perceptions
            }
            tsdf_extrinsics = {
                cam_id: refined_extrinsics[cam_id]
                for cam_id in tsdf_frames
                if cam_id in refined_extrinsics
            }
            tsdf_cloud, tsdf_diag = fuse_tsdf(
                tsdf_frames,
                tsdf_perceptions,
                tsdf_extrinsics,
                self.config.cloud_cleanup,
                self.config.tsdf_fusion,
            )
            tsdf_diag["skipped_by_guided_decision"] = [
                cam_id
                for cam_id in sorted(frames)
                if cam_id not in tsdf_frames
            ]
            if len(tsdf_cloud.points):
                fused_cloud = tsdf_cloud
                tsdf_diag["source"] = "tsdf"
            else:
                fused_cloud = self.fusion_backend.fuse(weighted_clouds, refined_extrinsics)
                tsdf_diag["source"] = "pointcloud_fallback"
            self._last_tsdf_fusion = tsdf_diag
        else:
            fused_cloud = self.fusion_backend.fuse(weighted_clouds, refined_extrinsics)
        timings["fusion_ms"] = _now_ms(start)

        start = time.perf_counter()
        fused_cloud, cleanup_diag = cleanup_fused_cloud(
            fused_cloud,
            stabilized_nipples,
            self.config.post_fusion_cleanup,
        )
        self._last_final_cleanup = cleanup_diag
        timings["post_cleanup_ms"] = _now_ms(start)

        start = time.perf_counter()
        camera_centers_world_m = [
            extrinsic.T_world_cam[:3, 3]
            for extrinsic in refined_extrinsics.values()
        ]
        fused_cloud = optional_poisson_surface_points(
            fused_cloud,
            camera_centers_world_m,
            self.config.surface,
        )
        timings["surface_ms"] = _now_ms(start)

        # Unify per-camera colours: paint the fused breast body one flat colour so
        # exposure/white-balance differences don't show as a patchwork, but keep the
        # real RGB of the areola + nipple (near the fused nipple anchors) so they stay
        # visible. Affects both the Rerun view and the written PLY.
        uniform_color = self.config.visualization.uniform_breast_color
        if uniform_color is not None and len(fused_cloud.points) > 0:
            points = np.asarray(fused_cloud.points)
            painted = np.tile(np.asarray(uniform_color, dtype=np.float64), (len(points), 1))
            radius = float(self.config.visualization.areola_radius_m)
            if radius > 0.0:
                areola = np.zeros(len(points), dtype=bool)
                for side in ("left", "right"):
                    nipple = stabilized_nipples.get(side)
                    if nipple is not None and nipple.confidence > 0.0 and np.isfinite(nipple.mean_world_m).all():
                        areola |= np.linalg.norm(points - np.asarray(nipple.mean_world_m), axis=1) <= radius
                areola_color = self.config.visualization.uniform_areola_color
                if areola_color is not None:
                    painted[areola] = np.asarray(areola_color, dtype=np.float64)
                else:
                    real_colors = np.asarray(fused_cloud.colors)
                    if real_colors.shape == painted.shape:
                        painted[areola] = real_colors[areola]
            fused_cloud.colors = o3d.utility.Vector3dVector(painted)

        start = time.perf_counter()
        nipples = self._smooth_nipples(stabilized_nipples, timestamp_ns)
        timings["smoothing_ms"] = _now_ms(start)

        start = time.perf_counter()
        self._last_quality_metrics = {
            "surface_consistency": compute_cross_camera_surface_consistency(
                clouds,
                refined_extrinsics,
                self.config.quality_metrics,
            ),
            "temporal_repeatability": self._repeatability_tracker.update(fused_cloud, nipples),
        }
        timings["quality_metrics_ms"] = _now_ms(start)

        start = time.perf_counter()
        result = ReconstructionResult(
            timestamp_ns=timestamp_ns,
            fused_cloud=fused_cloud,
            nipples=nipples,
            timings_ms=timings,
        )
        self._write_cloud_and_nipples(result)
        timings["write_ms"] = _now_ms(start)
        timings["total_ms"] = _now_ms(total_start)
        result = ReconstructionResult(
            timestamp_ns=timestamp_ns,
            fused_cloud=fused_cloud,
            nipples=nipples,
            timings_ms=timings,
        )
        self._write_summary(result)
        self._publish_viewer_frame(frames, perceptions, result, frame_wall_s)
        LOGGER.info(
            "Recon3D frame complete: total=%.2f ms perception=%.2f ms points=%d",
            timings["total_ms"],
            timings.get("perception_ms", 0.0),
            len(fused_cloud.points),
        )
        return result

    def run_stream(self) -> None:
        """Run continuously from the configured rig.

        Input shape/unit/frame: rig frames are aligned RGB-D with timestamps.
        Return type: ``None``.
        """
        if self.rig is None:
            raise RuntimeError("run_stream requires a rig")
        self.rig.start()
        try:
            while True:
                try:
                    frames = self.rig.read_synchronized_frames()
                except StopIteration:
                    break
                except RuntimeError as exc:
                    LOGGER.warning("Skipping frame set: %s", exc)
                    continue
                self.run_once(frames)
        finally:
            self.rig.stop()

