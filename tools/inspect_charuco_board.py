from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon3d.calibration.charuco import CharucoBoardSpec, detect_charuco, make_charuco_board
from recon3d.cli import _hardware_camera_specs, _parse_dictionary_name
from recon3d.sensors.femto_bolt import FemtoBoltCamera
from recon3d.sensors.rig import MultiFemtoBoltRig


COMMON_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
]


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _detect_marker_count(rgb, dictionary_name: str) -> int:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(_parse_dictionary_name(dictionary_name))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        _, ids, _ = detector.detectMarkers(gray)
    else:
        _, ids, _ = aruco.detectMarkers(gray, dictionary)
    return 0 if ids is None else int(len(ids))


def _capture_one(camera_spec: tuple[str, dict], settle_frames: int):
    cam_id, kwargs = camera_spec
    camera_kwargs = dict(kwargs)
    camera_kwargs["apply_sync_config"] = False
    camera = FemtoBoltCamera(cam_id, **camera_kwargs)
    camera.start()
    try:
        for _ in range(max(0, settle_frames)):
            camera.read_aligned_frame()
        return camera.read_aligned_frame()
    finally:
        camera.stop()


def _capture_simultaneous(camera_specs: list[tuple[str, dict]], settle_frames: int):
    cameras = [FemtoBoltCamera(cam_id, **kwargs) for cam_id, kwargs in camera_specs]
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=1000.0)
    rig.start()
    try:
        for _ in range(max(0, settle_frames)):
            for camera in cameras:
                camera.read_aligned_frame()
        return {camera.cam_id: camera.read_aligned_frame() for camera in cameras}
    finally:
        rig.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect which ArUco dictionary matches the visible Charuco board.")
    parser.add_argument("--config", default="configs/recon3d/default.yaml")
    parser.add_argument("--out", default="runs/charuco_inspect")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, default=0.052)
    parser.add_argument("--marker-length-m", type=float, default=0.039)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--dictionary", action="append", default=None)
    parser.add_argument("--color-size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--color-fps", type=int, default=None)
    parser.add_argument("--color-format", default=None)
    parser.add_argument("--align-mode", choices=["hw", "sw"], default=None)
    parser.add_argument("--simultaneous", action="store_true")
    parser.add_argument("--no-sync-config", action="store_true")
    args = parser.parse_args()

    raw = _load_yaml(Path(args.config))
    specs_args = Namespace(
        cameras=None,
        serials=None,
        sync_modes=None,
        align_mode=args.align_mode,
        color_size=args.color_size,
        color_fps=args.color_fps,
        color_format=args.color_format,
        no_sync_config=True if args.no_sync_config else not args.simultaneous,
    )
    specs = _hardware_camera_specs(specs_args, raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionaries = args.dictionary or COMMON_DICTIONARIES
    if args.simultaneous:
        frames = _capture_simultaneous(specs, args.settle_frames)
    else:
        frames = {}
        for spec in specs:
            frame = _capture_one(spec, args.settle_frames)
            frames[frame.cam_id] = frame
    for frame in frames.values():
        cv2.imwrite(str(out_dir / f"{frame.cam_id}_rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved RGB previews to {out_dir}")
    for dictionary in dictionaries:
        board = make_charuco_board(
            CharucoBoardSpec(
                args.squares_x,
                args.squares_y,
                args.square_length_m,
                args.marker_length_m,
                _parse_dictionary_name(dictionary),
            )
        )
        marker_counts = {}
        charuco_counts = {}
        for cam_id, frame in frames.items():
            marker_counts[cam_id] = _detect_marker_count(frame.rgb, dictionary)
            detection = detect_charuco(frame.rgb, board)
            charuco_counts[cam_id] = 0 if detection is None else int(len(detection.ids))
        marker_total = sum(marker_counts.values())
        charuco_total = sum(charuco_counts.values())
        print(
            f"{dictionary}: markers={marker_total} {marker_counts} | "
            f"charuco={charuco_total} {charuco_counts}"
        )


if __name__ == "__main__":
    main()
