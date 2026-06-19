from __future__ import annotations

import atexit
import json
import logging
from pathlib import Path
import threading
import time
from typing import Mapping
from urllib.parse import quote
import webbrowser

import numpy as np
import open3d as o3d

from recon3d.config import (
    CameraExtrinsics,
    FusedNipple,
    Recon3DConfig,
    ReconstructionResult,
    RGBDFrame,
)
from recon3d.geometry.backproject import backproject_masked_depth
from recon3d.geometry.fusion import FusionBackend, PointCloudFusionBackend, point_cloud_bounds
from recon3d.geometry.nipple3d import (
    estimate_nipple_observations,
    fuse_nipple_observations,
    nipple_json_payload,
)
from recon3d.perception.inference import BreastPerceptionAdapter
from recon3d.sensors.rig import MultiFemtoBoltRig
from recon3d.smoothing.temporal import OneEuroFilter3D


LOGGER = logging.getLogger(__name__)
_VIEWER_LOCK = threading.Lock()
_VIEWER_SERVER: object | None = None
_VIEWER_THREAD: threading.Thread | None = None
_VIEWER_ROOT: Path | None = None
_VIEWER_BASE_URL: str | None = None
_VIEWER_ATEXIT_REGISTERED = False


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _is_viewer_running() -> bool:
    return _VIEWER_SERVER is not None and _VIEWER_THREAD is not None and _VIEWER_THREAD.is_alive()


def _viewer_url_for_output(base_url: str, root: Path, output_dir: Path | None) -> str:
    if output_dir is None:
        return base_url
    try:
        rel = output_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return base_url
    base = f"../{rel.as_posix().rstrip('/')}/"
    return f"{base_url}?base={quote(base, safe='')}"


def stop_viewer_server() -> None:
    """Stop the embedded Recon3D viewer server if this process started it."""
    global _VIEWER_BASE_URL, _VIEWER_ROOT, _VIEWER_SERVER, _VIEWER_THREAD
    with _VIEWER_LOCK:
        server = _VIEWER_SERVER
        if server is None:
            return
        shutdown = getattr(server, "shutdown", None)
        server_close = getattr(server, "server_close", None)
        if callable(shutdown):
            shutdown()
        if callable(server_close):
            server_close()
        _VIEWER_SERVER = None
        _VIEWER_THREAD = None
        _VIEWER_ROOT = None
        _VIEWER_BASE_URL = None


def start_viewer_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: Path | None = None,
    open_browser: bool = False,
) -> str:
    """Start the browser viewer/API server in a background thread.

    Input shape/unit/frame: host/port are HTTP bind settings; output_dir is the
    reconstruction output folder to load in the viewer when it is under the
    project root.
    Return type: viewer URL string.
    """
    global _VIEWER_ATEXIT_REGISTERED, _VIEWER_BASE_URL, _VIEWER_ROOT, _VIEWER_SERVER, _VIEWER_THREAD
    with _VIEWER_LOCK:
        if _is_viewer_running() and _VIEWER_BASE_URL is not None and _VIEWER_ROOT is not None:
            url = _viewer_url_for_output(_VIEWER_BASE_URL, _VIEWER_ROOT, output_dir)
            if open_browser:
                webbrowser.open(url)
            return url

        from tools.serve_recon3d_viewer import ROOT, make_server

        server = None
        chosen_port = port
        for candidate_port in range(port, port + 20):
            try:
                server = make_server(host, candidate_port, ROOT)
                chosen_port = candidate_port
                break
            except OSError:
                continue
        if server is None:
            raise RuntimeError(f"No free viewer port found from {port} to {port + 19}")

        thread = threading.Thread(target=server.serve_forever, name="recon3d-viewer-server", daemon=True)
        thread.start()
        _VIEWER_SERVER = server
        _VIEWER_THREAD = thread
        _VIEWER_ROOT = ROOT
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        _VIEWER_BASE_URL = f"http://{browser_host}:{chosen_port}/viewer/"
        if not _VIEWER_ATEXIT_REGISTERED:
            atexit.register(stop_viewer_server)
            _VIEWER_ATEXIT_REGISTERED = True

        url = _viewer_url_for_output(_VIEWER_BASE_URL, ROOT, output_dir)
        LOGGER.info("Recon3D viewer server started at %s", url)
        if open_browser:
            webbrowser.open(url)
        return url


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
        self.viewer_url: str | None = None
        if self.config.enable_visualization:
            try:
                self.viewer_url = start_viewer_server(
                    host=self.config.viewer_host,
                    port=self.config.viewer_port,
                    output_dir=self.config.output_dir,
                    open_browser=self.config.viewer_open_browser,
                )
            except Exception as exc:  # noqa: BLE001 - visualization should not stop reconstruction
                LOGGER.warning("Recon3D viewer server could not start: %s", exc)

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
        clouds = {
            cam_id: backproject_masked_depth(
                frame.depth_mm,
                perceptions[cam_id].breast_mask,
                frame.rgb,
                frame.intrinsics,
                frame.depth_config,
                estimate_normals=False,
            )
            for cam_id, frame in frames.items()
            if cam_id in perceptions
        }
        timings["backproject_ms"] = _now_ms(start)

        start = time.perf_counter()
        observations = estimate_nipple_observations(frames, perceptions, self.extrinsics, self.config.nipple3d)
        nipples = fuse_nipple_observations(observations, self.config.nipple3d)
        timings["nipple3d_ms"] = _now_ms(start)

        start = time.perf_counter()
        fused_cloud = self.fusion_backend.fuse(clouds, self.extrinsics)
        timings["fusion_ms"] = _now_ms(start)

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
