from __future__ import annotations

import json
import logging
import time
from typing import Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    FusedNipple,
    Recon3DConfig,
    ReconstructionResult,
    RGBDFrame,
)
from anatomy_guided_recon3d.geometry.fusion import (
    FusionBackend,
    PointCloudFusionBackend,
    build_nipple_guided_clouds,
    point_cloud_bounds,
)
from anatomy_guided_recon3d.geometry.nipple3d import (
    estimate_nipple_observations,
    fuse_nipple_observations,
    nipple_json_payload,
)
from anatomy_guided_recon3d.perception.inference import BreastPerceptionAdapter
from anatomy_guided_recon3d.sensors.rig import MultiFemtoBoltRig
from anatomy_guided_recon3d.smoothing.temporal import OneEuroFilter3D


LOGGER = logging.getLogger(__name__)


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


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
    ) -> None:
        """Create the reconstruction pipeline.

        Input shape/unit/frame: extrinsics are camera-to-world transforms in
        metres; rig returns synchronized aligned RGB-D frames.
        Return type: ``Recon3DPipeline``.
        """
        self.config = config
        self.extrinsics = dict(extrinsics)
        self.rig = rig
        self.perception = perception or BreastPerceptionAdapter(config=config)
        self.fusion_backend = fusion_backend or PointCloudFusionBackend(config.fusion)
        self._filters = {"left": OneEuroFilter3D(), "right": OneEuroFilter3D()}
        self._last_nipple_guided_fusion: dict[str, object] = {}

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
        (out_dir / "nipple_guided_fusion.json").write_text(
            json.dumps(self._last_nipple_guided_fusion, indent=2),
            encoding="utf-8",
        )

    def _write_summary(self, result: ReconstructionResult) -> None:
        out_dir = self.config.output_dir
        if out_dir is None:
            return
        summary = {
            "timestamp_ns": result.timestamp_ns,
            "timings_ms": result.timings_ms,
            "fused_cloud": point_cloud_bounds(result.fused_cloud),
            "nipple_guided_fusion": self._last_nipple_guided_fusion,
            "metrics": {
                "timestamp_drift_ms": result.timings_ms.get("timestamp_drift_ms"),
                "calibration_mean_reprojection_error_px": "not_measured",
                "calibration_max_reprojection_error_px": "not_measured",
                "phantom_repeatability_std_mm": "not_measured",
                "single_frame_latency_ms": result.timings_ms.get("total_ms"),
            },
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run_once(self, frames: Mapping[str, RGBDFrame]) -> ReconstructionResult:
        """Run one synchronized RGB-D frame set through reconstruction.

        Input shape/unit/frame: frames are aligned RGB HxWx3 and depth HxW raw
        millimetres; output cloud/nipples are world-frame metres.
        Return type: ``ReconstructionResult``.
        """
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
        nipples = fuse_nipple_observations(observations, self.config.nipple3d)
        timings["nipple3d_ms"] = _now_ms(start)

        start = time.perf_counter()
        clouds, view_decisions, point_counts = build_nipple_guided_clouds(
            frames,
            perceptions,
            observations,
            nipples,
            self.config.nipple_guided_fusion,
        )
        timings["nipple_guided_clouds_ms"] = _now_ms(start)
        self._last_nipple_guided_fusion = {
            "method": "nipple_guided_multi_view_mask_and_surface_fusion",
            "enabled": self.config.nipple_guided_fusion.enabled,
            "num_observations": len(observations),
            "view_decisions": view_decisions,
            "point_counts": point_counts,
            "guide_config": {
                "mask_erode_px": self.config.nipple_guided_fusion.mask_erode_px,
                "depth_edge_threshold_m": self.config.nipple_guided_fusion.depth_edge_threshold_m,
                "depth_min_m": self.config.nipple_guided_fusion.depth_min_m,
                "depth_max_m": self.config.nipple_guided_fusion.depth_max_m,
                "nipple_gate_m": self.config.nipple_guided_fusion.nipple_gate_m,
                "reject_inconsistent_views": self.config.nipple_guided_fusion.reject_inconsistent_views,
                "missing_nipple_weight": self.config.nipple_guided_fusion.missing_nipple_weight,
                "min_consistent_weight": self.config.nipple_guided_fusion.min_consistent_weight,
                "min_accepted_views": self.config.nipple_guided_fusion.min_accepted_views,
            },
        }

        start = time.perf_counter()
        fused_cloud = self.fusion_backend.fuse(clouds, self.extrinsics)
        timings["fusion_ms"] = _now_ms(start)
        self._last_nipple_guided_fusion["fused_cloud"] = point_cloud_bounds(fused_cloud)

        start = time.perf_counter()
        nipples = self._smooth_nipples(nipples, timestamp_ns)
        timings["smoothing_ms"] = _now_ms(start)

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
        LOGGER.info(
            "Nipple-guided Recon3D frame complete: total=%.2f ms perception=%.2f ms points=%d",
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

