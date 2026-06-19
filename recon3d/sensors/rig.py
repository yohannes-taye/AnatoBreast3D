from __future__ import annotations

import logging
from typing import Protocol, Sequence

from recon3d.config import RGBDFrame


LOGGER = logging.getLogger(__name__)


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

    def __init__(self, cameras: Sequence[RGBDCamera], max_timestamp_drift_ms: float = 5.0) -> None:
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

    def read_synchronized_frames(self) -> dict[str, RGBDFrame]:
        """Read one synchronized frame set.

        Input shape/unit/frame: each camera provides timestamp_ns; accepted drift
        must be below configured threshold.
        Return type: mapping from camera ID to ``RGBDFrame``.
        """
        # Read all cameras on the same hardware-synced trigger in four phases:
        #   1. flush every camera's queued backlog (fast, back-to-back),
        #   2. block for the next frame from each -- after the flush that is the next
        #      synchronized trigger, identical across cameras,
        #   3. capture every camera's device timestamp immediately, while all raw
        #      frame sets are still the freshly grabbed trigger,
        #   4. align/decode each raw set.
        # Phases 1-3 run back-to-back with no heavy work in between: that is what
        # keeps the trigger identical. The timestamps must be captured before any
        # decode because materialising one camera's frame can recycle the SDK buffers
        # the others still reference, which would otherwise advance their timestamps
        # by whole frame periods. Cameras without the multi-phase API (replay/mock)
        # use the single-call read.
        if all(
            hasattr(camera, "flush_frames")
            and hasattr(camera, "grab_next_frameset")
            and hasattr(camera, "frameset_device_timestamp_ns")
            and hasattr(camera, "frameset_to_rgbd")
            for camera in self.cameras
        ):
            for camera in self.cameras:
                camera.flush_frames()
            raw = [(camera, camera.grab_next_frameset()) for camera in self.cameras]
            stamped = [(camera, fs, camera.frameset_device_timestamp_ns(fs)) for camera, fs in raw]
            frames = {
                camera.cam_id: camera.frameset_to_rgbd(fs, timestamp_ns=ts)
                for camera, fs, ts in stamped
            }
        else:
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
