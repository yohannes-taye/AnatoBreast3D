from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import yaml

from recon3d.calibration.charuco import (
    CharucoBoardSpec,
    estimate_extrinsics_from_charuco,
    summarize_reprojection_errors,
)
from recon3d.calibration.io import load_extrinsics, save_extrinsics
from recon3d.config import CameraExtrinsics, FusionConfig, Nipple3DConfig, PerceptionConfig, Recon3DConfig
from recon3d.pipeline import Recon3DPipeline
from recon3d.sensors.femto_bolt import FemtoBoltCamera, HardwareUnavailableError, ReplayRGBDCamera
from recon3d.sensors.rig import MultiFemtoBoltRig


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
    fusion_raw = raw.get("fusion", {})
    nipple_raw = raw.get("nipple3d", {})
    output_dir = output_override or (Path(raw["output_dir"]) if raw.get("output_dir") else None)
    config = Recon3DConfig(
        perception=PerceptionConfig(
            model_config_path=Path(
                perception_raw.get("model_config_path", "configs/breastseg/breastseg_b2_inplace.py")
            ),
            checkpoint_path=Path(perception_raw.get("checkpoint_path", "work_dirs/breastseg_b2/best.pth")),
            device=str(perception_raw.get("device", "cuda")),
            score_thr=float(perception_raw.get("score_thr", 0.3)),
            topk=int(perception_raw.get("topk", 2)),
            nms_iou=float(perception_raw.get("nms_iou", 0.5)),
        ),
        fusion=FusionConfig(
            voxel_size_m=float(fusion_raw.get("voxel_size_m", 0.002)),
            outlier_nb_neighbors=int(fusion_raw.get("outlier_nb_neighbors", 20)),
            outlier_std_ratio=float(fusion_raw.get("outlier_std_ratio", 2.0)),
            estimate_normals=bool(fusion_raw.get("estimate_normals", False)),
        ),
        nipple3d=Nipple3DConfig(
            dbscan_eps_m=float(nipple_raw.get("dbscan_eps_m", 0.005)),
            min_cluster_points=int(nipple_raw.get("min_cluster_points", 30)),
            covariance_jitter_m2=float(nipple_raw.get("covariance_jitter_m2", 1e-8)),
            left_right_axis_world=nipple_raw.get("left_right_axis_world", [1.0, 0.0, 0.0]),
            mahalanobis_threshold=float(nipple_raw.get("mahalanobis_threshold", 3.0)),
        ),
        output_dir=output_dir,
        sync_max_drift_ms=float(raw.get("sync_max_drift_ms", 5.0)),
        enable_visualization=bool(raw.get("enable_visualization", False)),
    )
    extrinsics_path = Path(raw.get("extrinsics_path", "configs/recon3d/extrinsics.json"))
    return config, extrinsics_path


def _build_replay_rig(bag: Path, max_drift_ms: float) -> MultiFemtoBoltRig:
    camera_dirs = sorted(path for path in bag.iterdir() if path.is_dir() and (path / "metadata.json").is_file())
    if not camera_dirs:
        raise FileNotFoundError(f"No replay camera folders with metadata.json found under {bag}")
    cameras = [ReplayRGBDCamera(path.name, path) for path in camera_dirs]
    return MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=max_drift_ms)


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
    """Run reconstruction from saved replay RGB-D data.

    Input shape/unit/frame: replay bag stores RGB PNG and raw depth PNG per camera.
    Return type: ``None``.
    """
    config, extrinsics_path = load_recon_config(Path(args.config), Path(args.out) if args.out else None)
    extrinsics = load_extrinsics(extrinsics_path)
    rig = _build_replay_rig(Path(args.bag), config.sync_max_drift_ms)
    pipeline = Recon3DPipeline(config, extrinsics, rig=rig)
    pipeline.run_stream()


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
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=config.sync_max_drift_ms)
    pipeline = Recon3DPipeline(config, extrinsics, rig=rig)
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
    """Record raw synchronized RGB-D data from hardware.

    Input shape/unit/frame: hardware frames are aligned RGB-D with depth mm.
    Return type: ``None``.
    """
    raise HardwareUnavailableError(
        "Recording requires verified pyorbbecsdk hardware integration; replay/mock support is available now"
    )


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
    record.add_argument("--duration", type=float, default=30.0)
    record.add_argument("--out", required=True)
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
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="Replay saved RGB-D data")
    replay.add_argument("--config", required=True)
    replay.add_argument("--bag", required=True)
    replay.add_argument("--out", required=False)
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
