from __future__ import annotations

import logging
from typing import Protocol, Sequence

from anatomy_guided_recon3d.config import RGBDFrame


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

