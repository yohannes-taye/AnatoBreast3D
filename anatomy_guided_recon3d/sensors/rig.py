from __future__ import annotations

import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Protocol, Sequence

from anatomy_guided_recon3d.config import RGBDFrame


LOGGER = logging.getLogger(__name__)


class SyncDiagnosticsWriter:
    """Append sync diagnostics events to a JSONL file without affecting capture."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._warned = False

    def write(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("wall_time_s", time.time())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - diagnostics must not interrupt capture
            if not self._warned:
                LOGGER.warning("Failed to write sync diagnostics to %s: %s", self.path, exc)
                self._warned = True


class RGBDCamera(Protocol):
    """Protocol for aligned RGB-D camera wrappers."""

    cam_id: str

    def start(self) -> None:
        """Start capture.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        ...

    def stop(self) -> None:
        """Stop capture.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        ...

    def read_aligned_frame(self) -> RGBDFrame:
        """Read an aligned RGB-D frame.

        Input shape/unit/frame: RGB HxWx3 uint8 aligned to depth HxW raw mm.
        Return type: ``RGBDFrame``.
        """
        ...


class MultiFemtoBoltRig:
    """Three-camera synchronized RGB-D rig manager."""

    def __init__(
        self,
        cameras: Sequence[RGBDCamera],
        max_timestamp_drift_ms: float = 5.0,
        sync_diagnostics_path: Path | str | None = None,
    ) -> None:
        """Create a multi-camera rig.

        Input shape/unit/frame: cameras return aligned RGB-D frames with hardware
        timestamps in nanoseconds; drift threshold is milliseconds.
        Return type: ``MultiFemtoBoltRig``.
        """
        if len(cameras) == 0:
            raise ValueError("At least one camera is required")
        if max_timestamp_drift_ms <= 0:
            raise ValueError("max_timestamp_drift_ms must be positive")
        cam_ids = [camera.cam_id for camera in cameras]
        duplicate_ids = sorted({cam_id for cam_id in cam_ids if cam_ids.count(cam_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate camera IDs are not allowed: {duplicate_ids}")
        self.cameras = list(cameras)
        self.max_timestamp_drift_ns = int(max_timestamp_drift_ms * 1_000_000)
        self._sync_diagnostics = (
            SyncDiagnosticsWriter(sync_diagnostics_path)
            if sync_diagnostics_path is not None
            else None
        )
        self._sync_diagnostics_round = 0
        self._last_sync_event: dict[str, Any] | None = None

    def enable_sync_diagnostics(self, path: Path | str) -> None:
        """Enable JSONL diagnostics for subsequent synchronized reads."""
        self._sync_diagnostics = SyncDiagnosticsWriter(path)

    @property
    def sync_diagnostics_enabled(self) -> bool:
        return self._sync_diagnostics is not None

    @property
    def last_sync_event(self) -> dict[str, Any] | None:
        """Most recent synchronized-read diagnostics event, if one was produced."""
        return self._last_sync_event

    def start(self) -> None:
        """Start all cameras.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        started: list[RGBDCamera] = []
        # Hardware-sync rigs should have secondary cameras streaming before the
        # primary starts emitting trigger pulses. Standalone/mock/replay cameras
        # keep their original order because they do not expose ``sync_mode``.
        cameras = sorted(
            self.cameras,
            key=lambda camera: 1 if getattr(camera, "sync_mode", "") == "primary" else 0,
        )
        try:
            for camera in cameras:
                camera.start()
                started.append(camera)
            sync_camera = next(
                (
                    camera
                    for camera in started
                    if getattr(camera, "sync_mode", "standalone") not in {"standalone", "free_run"}
                    and hasattr(camera, "enable_multi_device_sync")
                ),
                None,
            )
            if sync_camera is not None:
                sync_camera.enable_multi_device_sync()
            for camera in started:
                resume_playback = getattr(camera, "resume_playback_stream", None)
                if callable(resume_playback):
                    resume_playback()
        except Exception:
            for camera in reversed(started):
                try:
                    camera.stop()
                except Exception:
                    LOGGER.exception("Failed to stop camera '%s' after rig start failure", camera.cam_id)
            raise

    def stop(self) -> None:
        """Stop all cameras.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        for camera in self.cameras:
            camera.stop()

    @staticmethod
    def _timestamp_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
        timestamps = [
            int(frame.get("device_timestamp_ns") or 0)
            for frame in frames
            if int(frame.get("device_timestamp_ns") or 0) > 0
        ]
        if not timestamps:
            return {"count": len(frames), "first_ns": None, "last_ns": None, "span_ms": None}
        return {
            "count": len(frames),
            "first_ns": timestamps[0],
            "last_ns": timestamps[-1],
            "span_ms": (max(timestamps) - min(timestamps)) / 1_000_000.0,
        }

    def _next_diagnostics_event(self, path: str) -> dict[str, Any]:
        round_index = self._sync_diagnostics_round
        self._sync_diagnostics_round += 1
        return {
            "event": "sync_read",
            "path": path,
            "round_index": round_index,
            "max_timestamp_drift_ms": self.max_timestamp_drift_ns / 1_000_000.0,
            "cameras": {},
        }

    def _write_sync_diagnostics(self, event: dict[str, Any]) -> None:
        self._last_sync_event = event
        if self._sync_diagnostics is not None:
            self._sync_diagnostics.write(event)

    def _camera_timestamp_candidates(self, camera_event: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        drain = camera_event.get("drain", {})
        for index, frame in enumerate(drain.get("drained_frames") or []):
            timestamp_ns = int(frame.get("device_timestamp_ns") or 0)
            if timestamp_ns > 0:
                candidates.append(
                    {
                        "timestamp_ns": timestamp_ns,
                        "source": "drain",
                        "index": index,
                        "timestamp_source": frame.get("timestamp_source"),
                    }
                )
        selected = camera_event.get("grab", {}).get("selected_frame")
        if selected:
            timestamp_ns = int(selected.get("device_timestamp_ns") or 0)
            if timestamp_ns > 0:
                candidates.append(
                    {
                        "timestamp_ns": timestamp_ns,
                        "source": "selected",
                        "index": None,
                        "timestamp_source": selected.get("timestamp_source"),
                    }
                )
        return candidates

    @staticmethod
    def _pool_match_summary(
        camera_ids: list[str],
        pools: dict[str, list[dict[str, Any]]],
        combo: tuple[dict[str, Any], ...] | None,
        drift_ns: int | None,
    ) -> dict[str, Any]:
        if combo is None or drift_ns is None:
            return {
                "available": False,
                "candidate_counts": {cam_id: len(values) for cam_id, values in pools.items()},
                "best_drift_ms": None,
                "timestamps_ns": {},
                "sources": {},
            }
        return {
            "available": True,
            "candidate_counts": {cam_id: len(values) for cam_id, values in pools.items()},
            "best_drift_ms": drift_ns / 1_000_000.0,
            "timestamps_ns": {
                cam_id: int(item["timestamp_ns"])
                for cam_id, item in zip(camera_ids, combo)
            },
            "sources": {
                cam_id: {
                    "source": item["source"],
                    "index": item["index"],
                    "timestamp_source": item.get("timestamp_source"),
                }
                for cam_id, item in zip(camera_ids, combo)
            },
        }

    def _best_match_from_pools(
        self,
        pools: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...] | None]:
        camera_ids = [camera.cam_id for camera in self.cameras]
        if any(not values for values in pools.values()):
            return self._pool_match_summary(camera_ids, pools, None, None), None

        best_combo = None
        best_drift_ns = None
        best_median_ns = None
        for combo in product(*(pools[cam_id] for cam_id in camera_ids)):
            timestamps = [int(item["timestamp_ns"]) for item in combo]
            drift_ns = max(timestamps) - min(timestamps)
            median_ns = sorted(timestamps)[len(timestamps) // 2]
            if (
                best_drift_ns is None
                or drift_ns < best_drift_ns
                or (drift_ns == best_drift_ns and median_ns > int(best_median_ns or 0))
            ):
                best_drift_ns = drift_ns
                best_median_ns = median_ns
                best_combo = combo

        return self._pool_match_summary(camera_ids, pools, best_combo, best_drift_ns), best_combo

    def _best_pool_match(self, cameras_event: dict[str, Any]) -> dict[str, Any]:
        pools = {
            camera.cam_id: self._camera_timestamp_candidates(cameras_event.get(camera.cam_id, {}))
            for camera in self.cameras
        }
        summary, _ = self._best_match_from_pools(pools)
        return summary

    def _selected_source_drifts_ms(self, cameras_event: dict[str, Any]) -> dict[str, float]:
        sources = {
            "selected": "device_timestamp_ns",
            "color_global": "color_global_timestamp_ns",
            "depth_global": "depth_global_timestamp_ns",
            "color_timestamp_us": "color_timestamp_us_ns",
            "depth_timestamp_us": "depth_timestamp_us_ns",
            "color_timestamp_ms": "color_timestamp_ms_ns",
            "depth_timestamp_ms": "depth_timestamp_ms_ns",
            "color_system": "color_system_timestamp_ns",
            "depth_system": "depth_system_timestamp_ns",
        }
        drifts: dict[str, float] = {}
        for source_name, key in sources.items():
            timestamps: list[int] = []
            for camera in self.cameras:
                frame = cameras_event.get(camera.cam_id, {}).get("grab", {}).get("selected_frame") or {}
                timestamp_ns = int(frame.get(key) or 0)
                if timestamp_ns > 0:
                    timestamps.append(timestamp_ns)
            if len(timestamps) == len(self.cameras):
                drifts[source_name] = (max(timestamps) - min(timestamps)) / 1_000_000.0
        return drifts

    @staticmethod
    def _json_safe_drain(drain: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in drain.items()
            if key not in {"frameset", "raw_frames", "candidates"}
        }

    def _grab_candidate(
        self,
        camera: RGBDCamera,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if hasattr(camera, "grab_next_frameset_with_diagnostics"):
            frameset, grab = camera.grab_next_frameset_with_diagnostics()
        else:
            grab_start = time.perf_counter()
            frameset = camera.grab_next_frameset()
            grab = {"grab_wait_ms": (time.perf_counter() - grab_start) * 1000.0}
        selected = grab.get("selected_frame")
        if selected is None and hasattr(camera, "frameset_diagnostics"):
            selected = camera.frameset_diagnostics(frameset)
            grab["selected_frame"] = selected
        timestamp_ns = int((selected or {}).get("device_timestamp_ns") or 0)
        if timestamp_ns <= 0:
            timestamp_ns = camera.frameset_device_timestamp_ns(frameset)
        return {
            "frameset": frameset,
            "timestamp_ns": timestamp_ns,
            "source": "selected",
            "index": None,
            "timestamp_source": (selected or {}).get("timestamp_source"),
            "diagnostics": selected or {},
        }, grab

    def _read_fifo_pool_matched_frames(self) -> dict[str, RGBDFrame]:
        event = self._next_diagnostics_event("fifo_pool_matching")
        cameras_event = event["cameras"]
        written = False
        try:
            candidate_pools: dict[str, list[dict[str, Any]]] = {}
            old_selected_timestamps: dict[str, int] = {}
            for camera in self.cameras:
                if hasattr(camera, "drain_frameset_pool_with_diagnostics"):
                    candidates, drain = camera.drain_frameset_pool_with_diagnostics()
                    drain = self._json_safe_drain(drain)
                elif hasattr(camera, "flush_frames_with_diagnostics"):
                    camera.flush_frames()
                    candidates = []
                    drain = {
                        "cam_id": camera.cam_id,
                        "drain_count": None,
                        "candidate_count": 0,
                        "drained_frames": [],
                    }
                else:
                    camera.flush_frames()
                    candidates = []
                    drain = {
                        "cam_id": camera.cam_id,
                        "drain_count": None,
                        "candidate_count": 0,
                        "drained_frames": [],
                    }
                drained_frames = list(drain.get("drained_frames") or [])
                drain["drained_summary"] = self._timestamp_summary(drained_frames)
                cameras_event[camera.cam_id] = {"drain": drain}

                selected_candidate, grab = self._grab_candidate(camera)
                old_selected_timestamps[camera.cam_id] = int(selected_candidate["timestamp_ns"])
                cameras_event[camera.cam_id]["grab"] = grab
                cameras_event[camera.cam_id]["selected_timestamp_ns"] = int(
                    selected_candidate["timestamp_ns"]
                )
                if int(selected_candidate["timestamp_ns"]) > 0:
                    candidates = list(candidates) + [selected_candidate]
                candidate_pools[camera.cam_id] = candidates

            old_selected_values = [value for value in old_selected_timestamps.values() if value > 0]
            old_selected_drift_ns = (
                max(old_selected_values) - min(old_selected_values)
                if len(old_selected_values) == len(self.cameras)
                else None
            )
            chosen_match, chosen_combo = self._best_match_from_pools(candidate_pools)
            event["old_flush_next_selected_timestamps_ns"] = old_selected_timestamps
            event["old_flush_next_selected_drift_ms"] = (
                old_selected_drift_ns / 1_000_000.0
                if old_selected_drift_ns is not None
                else None
            )
            event["fifo_pool_depth_by_camera"] = {
                cam_id: len(candidates) for cam_id, candidates in candidate_pools.items()
            }
            event["best_pool_match"] = chosen_match
            event["chosen_pool_match"] = chosen_match
            event["selected_source_drifts_ms"] = self._selected_source_drifts_ms(cameras_event)

            if chosen_combo is None:
                event["selected_timestamps_ns"] = {}
                event["pre_conversion_drift_ms"] = None
                event["chosen_pre_conversion_drift_ms"] = None
                event["accepted"] = False
                event["drop_reason"] = "no_fifo_pool_match"
                self._write_sync_diagnostics(event)
                written = True
                raise RuntimeError("Dropped frame set: no complete timestamped FIFO pool match")

            chosen_by_cam = {
                camera.cam_id: candidate
                for camera, candidate in zip(self.cameras, chosen_combo)
            }
            chosen_timestamps = {
                cam_id: int(candidate["timestamp_ns"])
                for cam_id, candidate in chosen_by_cam.items()
            }
            chosen_values = list(chosen_timestamps.values())
            chosen_drift_ns = max(chosen_values) - min(chosen_values)
            chosen_drift_ms = chosen_drift_ns / 1_000_000.0
            old_selected_drift_ms = event["old_flush_next_selected_drift_ms"]
            event["selected_timestamps_ns"] = chosen_timestamps
            event["chosen_timestamps_ns"] = chosen_timestamps
            event["pre_conversion_drift_ms"] = chosen_drift_ms
            event["chosen_pre_conversion_drift_ms"] = chosen_drift_ms
            event["selected_bad_but_fifo_good"] = (
                old_selected_drift_ms is not None
                and old_selected_drift_ms > self.max_timestamp_drift_ns / 1_000_000.0
                and chosen_drift_ns <= self.max_timestamp_drift_ns
            )
            if chosen_drift_ns > self.max_timestamp_drift_ns:
                event["accepted"] = False
                event["drop_reason"] = "timestamp_drift"
                self._write_sync_diagnostics(event)
                written = True
                LOGGER.warning(
                    "Dropping RGB-D frame set: FIFO match drift %.3f ms exceeds %.3f ms",
                    chosen_drift_ms,
                    self.max_timestamp_drift_ns / 1_000_000.0,
                )
                raise RuntimeError("Dropped frame set due to timestamp drift")

            frames: dict[str, RGBDFrame] = {}
            for camera in self.cameras:
                candidate = chosen_by_cam[camera.cam_id]
                if hasattr(camera, "frameset_to_rgbd_with_diagnostics"):
                    frame, conversion = camera.frameset_to_rgbd_with_diagnostics(
                        candidate["frameset"],
                        timestamp_ns=int(candidate["timestamp_ns"]),
                    )
                else:
                    convert_start = time.perf_counter()
                    frame = camera.frameset_to_rgbd(
                        candidate["frameset"],
                        timestamp_ns=int(candidate["timestamp_ns"]),
                    )
                    conversion = {"total_ms": (time.perf_counter() - convert_start) * 1000.0}
                frames[camera.cam_id] = frame
                cameras_event[camera.cam_id]["conversion"] = conversion

            event["conversion_total_ms"] = sum(
                float(cameras_event[camera.cam_id].get("conversion", {}).get("total_ms") or 0.0)
                for camera in self.cameras
            )
            event["accepted"] = True
            event["drop_reason"] = None
            self._write_sync_diagnostics(event)
            written = True
            return frames
        except Exception as exc:
            if not written:
                event["accepted"] = False
                event["drop_reason"] = "exception"
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                self._write_sync_diagnostics(event)
            raise

    def _read_fallback_frames_with_diagnostics(self) -> dict[str, RGBDFrame]:
        event = self._next_diagnostics_event("single_call")
        written = False
        try:
            frames = {camera.cam_id: camera.read_aligned_frame() for camera in self.cameras}
            timestamps = [frame.timestamp_ns for frame in frames.values()]
            drift = max(timestamps) - min(timestamps)
            event["selected_timestamps_ns"] = {
                cam_id: frame.timestamp_ns for cam_id, frame in frames.items()
            }
            event["pre_conversion_drift_ms"] = drift / 1_000_000.0
            if drift > self.max_timestamp_drift_ns:
                event["accepted"] = False
                event["drop_reason"] = "timestamp_drift"
                self._write_sync_diagnostics(event)
                written = True
                LOGGER.warning(
                    "Dropping RGB-D frame set: timestamp drift %.3f ms exceeds %.3f ms",
                    drift / 1_000_000.0,
                    self.max_timestamp_drift_ns / 1_000_000.0,
                )
                raise RuntimeError("Dropped frame set due to timestamp drift")
            event["accepted"] = True
            event["drop_reason"] = None
            self._write_sync_diagnostics(event)
            written = True
            return frames
        except Exception as exc:
            if not written:
                event["accepted"] = False
                event["drop_reason"] = "exception"
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                self._write_sync_diagnostics(event)
            raise

    def _read_multiphase_frames_with_diagnostics(self) -> dict[str, RGBDFrame]:
        event = self._next_diagnostics_event("multi_phase")
        cameras_event = event["cameras"]
        written = False
        try:
            for camera in self.cameras:
                if hasattr(camera, "flush_frames_with_diagnostics"):
                    drain = camera.flush_frames_with_diagnostics()
                else:
                    camera.flush_frames()
                    drain = {"cam_id": camera.cam_id, "drain_count": None, "drained_frames": []}
                drained_frames = list(drain.get("drained_frames") or [])
                drain["drained_summary"] = self._timestamp_summary(drained_frames)
                cameras_event[camera.cam_id] = {"drain": drain}

            raw: list[tuple[RGBDCamera, Any, dict[str, Any]]] = []
            for camera in self.cameras:
                if hasattr(camera, "grab_next_frameset_with_diagnostics"):
                    frameset, grab = camera.grab_next_frameset_with_diagnostics()
                else:
                    grab_start = time.perf_counter()
                    frameset = camera.grab_next_frameset()
                    grab = {"grab_wait_ms": (time.perf_counter() - grab_start) * 1000.0}
                raw.append((camera, frameset, grab))
                cameras_event[camera.cam_id]["grab"] = grab

            stamped = []
            selected_timestamps = {}
            for camera, frameset, grab in raw:
                timestamp_ns = camera.frameset_device_timestamp_ns(frameset)
                stamped.append((camera, frameset, timestamp_ns))
                selected_timestamps[camera.cam_id] = timestamp_ns
                cameras_event[camera.cam_id]["selected_timestamp_ns"] = timestamp_ns
                if "selected_frame" not in grab and hasattr(camera, "frameset_diagnostics"):
                    cameras_event[camera.cam_id]["grab"]["selected_frame"] = camera.frameset_diagnostics(frameset)

            timestamps = list(selected_timestamps.values())
            drift = max(timestamps) - min(timestamps)
            event["selected_timestamps_ns"] = selected_timestamps
            event["pre_conversion_drift_ms"] = drift / 1_000_000.0
            event["best_pool_match"] = self._best_pool_match(cameras_event)
            event["selected_source_drifts_ms"] = self._selected_source_drifts_ms(cameras_event)

            frames: dict[str, RGBDFrame] = {}
            for camera, frameset, timestamp_ns in stamped:
                if hasattr(camera, "frameset_to_rgbd_with_diagnostics"):
                    frame, conversion = camera.frameset_to_rgbd_with_diagnostics(
                        frameset,
                        timestamp_ns=timestamp_ns,
                    )
                else:
                    convert_start = time.perf_counter()
                    frame = camera.frameset_to_rgbd(frameset, timestamp_ns=timestamp_ns)
                    conversion = {"total_ms": (time.perf_counter() - convert_start) * 1000.0}
                frames[camera.cam_id] = frame
                cameras_event[camera.cam_id]["conversion"] = conversion

            event["conversion_total_ms"] = sum(
                float(cameras_event[camera.cam_id].get("conversion", {}).get("total_ms") or 0.0)
                for camera in self.cameras
            )
            if drift > self.max_timestamp_drift_ns:
                event["accepted"] = False
                event["drop_reason"] = "timestamp_drift"
                self._write_sync_diagnostics(event)
                written = True
                LOGGER.warning(
                    "Dropping RGB-D frame set: timestamp drift %.3f ms exceeds %.3f ms",
                    drift / 1_000_000.0,
                    self.max_timestamp_drift_ns / 1_000_000.0,
                )
                raise RuntimeError("Dropped frame set due to timestamp drift")

            event["accepted"] = True
            event["drop_reason"] = None
            self._write_sync_diagnostics(event)
            written = True
            return frames
        except Exception as exc:
            if not written:
                event["accepted"] = False
                event["drop_reason"] = "exception"
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                self._write_sync_diagnostics(event)
            raise

    def read_synchronized_frames(self) -> dict[str, RGBDFrame]:
        """Read one synchronized frame set.

        Input shape/unit/frame: each camera provides timestamp_ns; accepted drift
        must be below configured threshold.
        Return type: mapping from camera ID to ``RGBDFrame``.
        """
        # Hardware-sync exposure and host arrival are different clocks in practice.
        # The SDK polling FIFO can contain a better cross-camera timestamp match
        # than the next frame each pipeline returns after a flush, so the hardware
        # path drains queued raw frame sets, keeps those raw references alive, adds
        # one blocking next frame per camera, and converts only the best timestamp
        # combination.
        if all(
            hasattr(camera, "drain_frameset_pool_with_diagnostics")
            and hasattr(camera, "grab_next_frameset")
            and hasattr(camera, "frameset_device_timestamp_ns")
            and hasattr(camera, "frameset_to_rgbd")
            for camera in self.cameras
        ):
            return self._read_fifo_pool_matched_frames()
        if all(
            hasattr(camera, "flush_frames")
            and hasattr(camera, "grab_next_frameset")
            and hasattr(camera, "frameset_device_timestamp_ns")
            and hasattr(camera, "frameset_to_rgbd")
            for camera in self.cameras
        ):
            if self._sync_diagnostics is not None:
                return self._read_multiphase_frames_with_diagnostics()
            for camera in self.cameras:
                camera.flush_frames()
            raw = [(camera, camera.grab_next_frameset()) for camera in self.cameras]
            stamped = [(camera, fs, camera.frameset_device_timestamp_ns(fs)) for camera, fs in raw]
            frames = {
                camera.cam_id: camera.frameset_to_rgbd(fs, timestamp_ns=ts)
                for camera, fs, ts in stamped
            }
        else:
            if self._sync_diagnostics is not None:
                return self._read_fallback_frames_with_diagnostics()
            frames = {camera.cam_id: camera.read_aligned_frame() for camera in self.cameras}
        timestamps = [frame.timestamp_ns for frame in frames.values()]
        drift = max(timestamps) - min(timestamps)
        if drift > self.max_timestamp_drift_ns:
            LOGGER.warning(
                "Dropping RGB-D frame set: timestamp drift %.3f ms exceeds %.3f ms",
                drift / 1_000_000.0,
                self.max_timestamp_drift_ns / 1_000_000.0,
            )
            raise RuntimeError("Dropped frame set due to timestamp drift")
        return frames

