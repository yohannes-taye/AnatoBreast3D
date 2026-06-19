from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from recon3d.config import CameraIntrinsics, DepthConfig, RGBDFrame
from recon3d.sensors.femto_bolt import FemtoBoltCamera, MockRGBDCamera, ReplayRGBDCamera
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


class _RawFrameSet:
    def __init__(self, timestamp_ns: int) -> None:
        self.timestamp_ns = timestamp_ns


class _TimestampFrame:
    def get_timestamp(self) -> int:
        return 1

    def get_timestamp_us(self) -> int:
        return 1_000

    def get_global_timestamp_us(self) -> int:
        return 5_000

    def get_system_timestamp_us(self) -> int:
        return 9_000

    def has_metadata(self, metadata_type) -> bool:
        return False


class _TimestampFrameSet:
    def get_color_frame(self) -> _TimestampFrame:
        return _TimestampFrame()

    def get_depth_frame(self) -> None:
        return None


class _MultiPhaseDiagnosticCamera:
    def __init__(self, cam_id: str, drained_timestamps_ns: list[int], grabbed_timestamp_ns: int) -> None:
        self.cam_id = cam_id
        self.drained_timestamps_ns = list(drained_timestamps_ns)
        self.grabbed_timestamp_ns = int(grabbed_timestamp_ns)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def flush_frames(self) -> None:
        pass

    def _frame_diagnostics(self, frameset: _RawFrameSet) -> dict[str, object]:
        return {
            "device_timestamp_ns": frameset.timestamp_ns,
            "timestamp_source": "mock",
            "system_timestamp_ns": frameset.timestamp_ns + 10,
            "has_color": True,
            "has_depth": True,
            "complete": True,
        }

    def drain_frameset_pool_with_diagnostics(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        candidates = []
        drained_frames = []
        for index, timestamp_ns in enumerate(self.drained_timestamps_ns):
            frameset = _RawFrameSet(timestamp_ns)
            diagnostics = self._frame_diagnostics(frameset)
            drained_frames.append(diagnostics)
            candidates.append(
                {
                    "frameset": frameset,
                    "timestamp_ns": timestamp_ns,
                    "source": "drain",
                    "index": index,
                    "timestamp_source": "mock",
                    "diagnostics": diagnostics,
                }
            )
        return candidates, {
            "cam_id": self.cam_id,
            "drain_count": len(self.drained_timestamps_ns),
            "candidate_count": len(candidates),
            "drain_ms": 0.1,
            "hit_max_drain": False,
            "max_drain": 128,
            "drained_frames": drained_frames,
        }

    def flush_frames_with_diagnostics(self) -> dict[str, object]:
        _, drain = self.drain_frameset_pool_with_diagnostics()
        return {
            "cam_id": self.cam_id,
            **drain,
        }

    def grab_next_frameset(self) -> _RawFrameSet:
        return _RawFrameSet(self.grabbed_timestamp_ns)

    def grab_next_frameset_with_diagnostics(self) -> tuple[_RawFrameSet, dict[str, object]]:
        frameset = self.grab_next_frameset()
        return frameset, {
            "cam_id": self.cam_id,
            "grab_wait_ms": 0.2,
            "selected_frame": self._frame_diagnostics(frameset),
        }

    def frameset_device_timestamp_ns(self, frameset: _RawFrameSet) -> int:
        return frameset.timestamp_ns

    def frameset_to_rgbd(self, frameset: _RawFrameSet, timestamp_ns: int | None = None) -> RGBDFrame:
        return _make_frame(self.cam_id, frameset.timestamp_ns if timestamp_ns is None else timestamp_ns)

    def frameset_to_rgbd_with_diagnostics(
        self,
        frameset: _RawFrameSet,
        timestamp_ns: int | None = None,
    ) -> tuple[RGBDFrame, dict[str, object]]:
        frame = self.frameset_to_rgbd(frameset, timestamp_ns=timestamp_ns)
        return frame, {
            "cam_id": self.cam_id,
            "timestamp_ns": frame.timestamp_ns,
            "align_ms": 0.0,
            "decode_ms": 0.3,
            "total_ms": 0.3,
        }

    def read_aligned_frame(self) -> RGBDFrame:
        raise AssertionError("not used")


def _make_frame(cam_id: str, timestamp_ns: int) -> RGBDFrame:
    intr = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    depth_cfg = DepthConfig(depth_scale_m=0.001)
    return RGBDFrame(
        cam_id=cam_id,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_mm=np.ones((2, 2), dtype=np.uint16),
        timestamp_ns=timestamp_ns,
        intrinsics=intr,
        depth_config=depth_cfg,
    )


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


def test_rig_sync_diagnostics_disabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "sync_diagnostics.jsonl"
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [1_000, 2_000], 10_000),
        _MultiPhaseDiagnosticCamera("cam1", [1_100, 2_100], 10_100),
        _MultiPhaseDiagnosticCamera("cam2", [900, 1_900], 9_900),
    ]

    frames = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0).read_synchronized_frames()

    assert sorted(frames) == ["cam0", "cam1", "cam2"]
    assert not path.exists()


def test_rig_selects_better_fifo_pool_match(tmp_path: Path) -> None:
    path = tmp_path / "sync_diagnostics.jsonl"
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [1_000_000_000], 10_000_000_000),
        _MultiPhaseDiagnosticCamera("cam1", [1_001_000_000], 10_080_000_000),
        _MultiPhaseDiagnosticCamera("cam2", [999_000_000], 10_160_000_000),
    ]
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0, sync_diagnostics_path=path)

    frames = rig.read_synchronized_frames()
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert {cam_id: frame.timestamp_ns for cam_id, frame in frames.items()} == {
        "cam0": 1_000_000_000,
        "cam1": 1_001_000_000,
        "cam2": 999_000_000,
    }
    assert event["old_flush_next_selected_drift_ms"] == 160.0
    assert event["chosen_pre_conversion_drift_ms"] == 2.0
    assert event["selected_bad_but_fifo_good"] is True
    assert event["chosen_pool_match"]["sources"]["cam0"]["source"] == "drain"


def test_rig_fifo_tie_break_prefers_newer_median_timestamp() -> None:
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [1_000, 2_000], 10_000),
        _MultiPhaseDiagnosticCamera("cam1", [1_100, 2_100], 10_100),
        _MultiPhaseDiagnosticCamera("cam2", [900, 1_900], 9_900),
    ]

    frames = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0).read_synchronized_frames()

    assert {cam_id: frame.timestamp_ns for cam_id, frame in frames.items()} == {
        "cam0": 10_000,
        "cam1": 10_100,
        "cam2": 9_900,
    }


def test_rig_rejects_fifo_pool_match_over_budget_without_conversion(tmp_path: Path) -> None:
    path = tmp_path / "sync_diagnostics.jsonl"
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [], 1_000_000_000),
        _MultiPhaseDiagnosticCamera("cam1", [], 1_020_000_000),
        _MultiPhaseDiagnosticCamera("cam2", [], 1_040_000_000),
    ]
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0, sync_diagnostics_path=path)

    with pytest.raises(RuntimeError, match="timestamp drift"):
        rig.read_synchronized_frames()
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert event["accepted"] is False
    assert event["drop_reason"] == "timestamp_drift"
    assert event["chosen_pre_conversion_drift_ms"] == 40.0
    assert "conversion" not in event["cameras"]["cam0"]


def test_rig_writes_sync_diagnostics_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sync_diagnostics.jsonl"
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [1_000, 2_000], 10_000),
        _MultiPhaseDiagnosticCamera("cam1", [1_100, 2_100], 10_100),
        _MultiPhaseDiagnosticCamera("cam2", [900, 1_900], 9_900),
    ]
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0, sync_diagnostics_path=path)

    frames = rig.read_synchronized_frames()
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert sorted(frames) == ["cam0", "cam1", "cam2"]
    assert event["path"] == "fifo_pool_matching"
    assert event["round_index"] == 0
    assert event["accepted"] is True
    assert event["drop_reason"] is None
    assert event["pre_conversion_drift_ms"] == 0.0002
    assert event["selected_timestamps_ns"] == {"cam0": 10_000, "cam1": 10_100, "cam2": 9_900}
    assert event["chosen_pre_conversion_drift_ms"] == 0.0002
    assert event["old_flush_next_selected_drift_ms"] == 0.0002
    assert event["fifo_pool_depth_by_camera"] == {"cam0": 3, "cam1": 3, "cam2": 3}
    assert event["selected_bad_but_fifo_good"] is False
    assert event["chosen_pool_match"]["best_drift_ms"] == 0.0002
    assert event["best_pool_match"]["best_drift_ms"] == 0.0002
    assert event["best_pool_match"]["candidate_counts"] == {"cam0": 3, "cam1": 3, "cam2": 3}
    assert event["cameras"]["cam0"]["drain"]["drain_count"] == 2
    assert event["cameras"]["cam0"]["drain"]["candidate_count"] == 2
    assert event["cameras"]["cam0"]["drain"]["drained_frames"][0]["device_timestamp_ns"] == 1_000
    assert event["cameras"]["cam0"]["drain"]["drained_summary"]["count"] == 2
    assert event["cameras"]["cam1"]["grab"]["selected_frame"]["device_timestamp_ns"] == 10_100
    assert event["cameras"]["cam2"]["conversion"]["total_ms"] == 0.3


def test_rig_sync_diagnostics_write_failure_does_not_interrupt(tmp_path: Path) -> None:
    cameras = [
        _MultiPhaseDiagnosticCamera("cam0", [], 10_000),
        _MultiPhaseDiagnosticCamera("cam1", [], 10_100),
        _MultiPhaseDiagnosticCamera("cam2", [], 9_900),
    ]
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=5.0, sync_diagnostics_path=tmp_path)

    frames = rig.read_synchronized_frames()

    assert sorted(frames) == ["cam0", "cam1", "cam2"]


def test_femto_timestamp_diagnostics_prefers_global_timestamp() -> None:
    camera = object.__new__(FemtoBoltCamera)
    frameset = _TimestampFrameSet()

    diagnostics = camera.frameset_diagnostics(frameset)

    assert diagnostics["device_timestamp_ns"] == 5_000_000
    assert diagnostics["timestamp_source"] == "color_global_timestamp_us"
    assert diagnostics["color_timestamp_us_ns"] == 1_000_000
    assert diagnostics["color_system_timestamp_ns"] == 9_000_000
    assert camera.frameset_device_timestamp_ns(frameset) == 5_000_000
