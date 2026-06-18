from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from recon3d.config import CameraIntrinsics, DepthConfig, RGBDFrame
from recon3d.sensors.femto_bolt import MockRGBDCamera, ReplayRGBDCamera
from recon3d.sensors.rig import MultiFemtoBoltRig


class _FailingCamera:
    cam_id = "bad"

    def start(self) -> None:
        raise RuntimeError("start failed")

    def stop(self) -> None:
        pass

    def read_aligned_frame(self) -> RGBDFrame:
        raise AssertionError("not used")


class _StartedCamera:
    cam_id = "good"

    def __init__(self) -> None:
        self.stopped = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def read_aligned_frame(self) -> RGBDFrame:
        raise AssertionError("not used")


def _metadata(width: int = 4, height: int = 3) -> dict[str, object]:
    return {
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 1.5,
            "cy": 1.0,
            "distortion": None,
        },
        "depth_config": {
            "depth_scale_m": 0.001,
            "min_depth_m": 0.25,
            "max_depth_m": 3.0,
        },
        "timestamps_ns": [1_000],
    }


def test_replay_camera_validates_missing_frame_files(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ReplayRGBDCamera("cam0", tmp_path)


def test_replay_camera_reads_aligned_frame(tmp_path: Path) -> None:
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    rgb = np.full((3, 4, 3), 127, dtype=np.uint8)
    depth = np.full((3, 4), 1000, dtype=np.uint16)
    cv2.imwrite(str(tmp_path / "rgb" / "000000.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "depth" / "000000.png"), depth)

    camera = ReplayRGBDCamera("cam0", tmp_path)
    frame = camera.read_aligned_frame()

    assert frame.cam_id == "cam0"
    assert frame.rgb.shape == (3, 4, 3)
    assert frame.depth_mm.shape == (3, 4)
    assert frame.timestamp_ns == 1_000


def test_mock_camera_rejects_empty_frames() -> None:
    with pytest.raises(ValueError):
        MockRGBDCamera("cam0", [])


def test_rig_rejects_duplicate_camera_ids() -> None:
    intr = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    depth_cfg = DepthConfig(depth_scale_m=0.001)
    frame = RGBDFrame(
        cam_id="cam0",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_mm=np.ones((2, 2), dtype=np.uint16),
        timestamp_ns=0,
        intrinsics=intr,
        depth_config=depth_cfg,
    )
    with pytest.raises(ValueError):
        MultiFemtoBoltRig([MockRGBDCamera("cam0", [frame]), MockRGBDCamera("cam0", [frame])])


def test_rig_stops_started_cameras_when_later_start_fails() -> None:
    good = _StartedCamera()
    with pytest.raises(RuntimeError):
        MultiFemtoBoltRig([good, _FailingCamera()]).start()
    assert good.stopped
