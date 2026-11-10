"""Live calibrated framing preview and on-demand SDK bag capture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback

import cv2
import numpy as np
import yaml

from anatomy_guided_recon3d.calibration.io import camera_intrinsics_to_dict, load_extrinsics
from anatomy_guided_recon3d.sensors.femto_bolt import FemtoBoltCamera
from anatomy_guided_recon3d.sensors.rig import MultiFemtoBoltRig
from anatomy_guided_recon3d.tracking.capture import enable_common_clock
from anatomy_guided_recon3d.tracking.runtime import PROJECT_ROOT, replace_file, status, write_json
from anatomy_guided_recon3d.tracking.sequence import prepare_frame


def publish_image(path: Path, rgb: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Cannot encode camera preview")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.tobytes())
    replace_file(temporary, path)


def run(output: Path, config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hardware = raw["hardware"]
    calibration_path = (PROJECT_ROOT / raw["extrinsics_path"]).resolve()
    extrinsics = load_extrinsics(calibration_path)
    devices = enable_common_clock()
    cameras = [FemtoBoltCamera(
        item["id"], serial_number=item["serial_number"], sync_mode=item.get("sync_mode", "standalone"),
        align_mode="sw", color_size=tuple(hardware.get("color_size", [1280, 720])),
        color_fps=int(hardware.get("color_fps", 30)), color_auto_exposure=True, apply_sync_config=False,
    ) for item in hardware["cameras"]]
    for camera in cameras:
        if camera.cam_id not in extrinsics:
            raise ValueError(f"No calibration for {camera.cam_id}")
    rig = MultiFemtoBoltRig(cameras, max_timestamp_drift_ms=50)
    session = None
    record_start = 0.0
    accepted = 0
    drops = 0
    drifts = []
    last_publish = 0.0
    command_id = None
    stop_reason = "duration_complete"
    metadata = []
    perception = None
    masks = {}
    last_mask = 0.0
    status(output, "starting", "Starting calibrated camera preview")
    rig.start()
    try:
        started = time.monotonic()
        while True:
            command_path = output / "command.json"
            command = json.loads(command_path.read_text()) if command_path.exists() else {}
            fresh = command.get("id") != command_id
            if fresh:
                command_id = command.get("id")
                if command.get("action") == "stop":
                    stop_reason = "user_stop"
                    break
                if command.get("action") == "record" and session is None:
                    # Only accept capture once a complete calibrated preview has
                    # been published and frame-time checks have passed.
                    if not (output / "preview.json").exists():
                        raise RuntimeError("Wait for the live preview before starting a recording")
                    duration = float(command.get("duration_s", 6))
                    if not 1 <= duration <= 60:
                        raise ValueError("Record between 1 and 60 seconds")
                    session = PROJECT_ROOT / "data" / "Replay" / ("tapip3d-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
                    session.mkdir(parents=True, exist_ok=False)
                    shutil.copy2(calibration_path, session / "extrinsics.json")
                    for camera in cameras:
                        camera.begin_recording(session / f"{camera.cam_id}.bag")
                    record_start = time.monotonic()
                    accepted = drops = 0
                    drifts = []
            if session is not None and time.monotonic() - record_start >= duration:
                break
            try:
                frames = rig.read_synchronized_frames()
            except RuntimeError as exc:
                drops += 1
                if time.monotonic() - started > 20 and last_publish == 0:
                    raise RuntimeError(f"Camera timing did not settle: {exc}") from exc
                continue
            drift = (max(frame.timestamp_ns for frame in frames.values()) - min(frame.timestamp_ns for frame in frames.values())) / 1e6
            if session is not None:
                accepted += 1
                drifts.append(drift)
            if time.monotonic() - last_publish < 0.2:
                continue
            last_publish = time.monotonic()
            foreground_only = bool(command.get("foreground_only", True))
            if foreground_only and (not masks or time.monotonic() - last_mask > .5):
                if perception is None:
                    from anatomy_guided_recon3d.cli import load_recon_config
                    from anatomy_guided_recon3d.perception.inference import BreastPerceptionAdapter
                    perception = BreastPerceptionAdapter(config=load_recon_config(config_path)[0])
                predictions = perception.predict_batch(frames)
                masks = {key: value.breast_mask for key, value in predictions.items()}
                last_mask = time.monotonic()
            views = []
            for camera in cameras:
                frame = frames[camera.cam_id]
                image = cv2.resize(frame.rgb, (640, 360), interpolation=cv2.INTER_AREA)
                publish_image(output / f"{camera.cam_id}_rgb.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                depth_m = frame.depth_mm.astype(np.float32) * frame.depth_config.depth_scale_m
                depth_display = cv2.applyColorMap((np.clip((depth_m - .25) / 1.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                depth_display[depth_m <= 0] = 0
                publish_image(output / f"{camera.cam_id}_depth.jpg", cv2.resize(depth_display, (640, 360), interpolation=cv2.INTER_NEAREST))
                mask = masks[camera.cam_id] if foreground_only else np.ones(depth_m.shape, bool)
                prepared = prepare_frame(frame, mask, extrinsics[camera.cam_id].T_world_cam, 512)
                depth = prepared["depths"]
                y, x = np.where(prepared["mask"] & (depth >= float(command.get("min_depth_m", .3))) & (depth <= float(command.get("max_depth_m", 1.0))))
                indices = np.linspace(0, max(0, len(y) - 1), min(5000, len(y))).round().astype(int)
                y, x = y[indices], x[indices]
                xyz = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(prepared["intrinsics"]).T
                xyz *= depth[y, x, None]
                T = extrinsics[camera.cam_id].T_world_cam
                xyz = xyz @ T[:3, :3].T + T[:3, 3]
                views.append(dict(id=camera.cam_id, serial=camera.serial_number, xyz=xyz.round(5).tolist(),
                                  rgb=f"{camera.cam_id}_rgb.jpg", depth=f"{camera.cam_id}_depth.jpg",
                                  valid_depth_percent=round(100 * float(np.mean(depth_m > 0)), 1),
                                  timestamp_ns=frame.timestamp_ns))
            metadata = [dict(id=camera.cam_id, serial_number=camera.serial_number, bag=f"{camera.cam_id}.bag",
                             align_mode="sw", sync_mode="standalone", intrinsics=camera_intrinsics_to_dict(frames[camera.cam_id].intrinsics))
                        for camera in cameras]
            elapsed = time.monotonic() - record_start if session is not None else 0
            state = "recording" if session is not None else "preview"
            write_json(output / "preview.json", dict(state=state, timestamp=time.time(), cameras=views,
                       calibration=str(calibration_path), calibration_loaded=True, drift_ms=round(drift, 3),
                       foreground_only=foreground_only,
                       elapsed_s=round(elapsed, 1), remaining_s=round(max(0, duration-elapsed), 1) if session else 0))
            status(output, state, "Recording RGB-D" if session else "Preview ready — frame the phantom and inspect the overlap",
                   drift_ms=round(drift, 3), elapsed_s=round(elapsed, 1), session=str(session.relative_to(PROJECT_ROOT)) if session else None)
    finally:
        ended = time.monotonic()
        if session is not None:
            status(output, "saving", "Saving the RGB-D recording and calibration", session=str(session.relative_to(PROJECT_ROOT)))
        for camera in cameras:
            camera.pause_recording()
        rig.stop()
        if session is not None:
            write_json(session / "session.json", dict(
                schema="recon3d_orbbec_bag_session_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                duration_s=round(ended-record_start, 3), completed=accepted > 0, cameras=metadata,
                accepted_frame_sets=accepted, dropped_frame_sets=drops, stop_reason=stop_reason,
                timestamp_clock="SDK global timestamps; standalone cameras, not hardware trigger synchronization",
                calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                timestamp_drift_ms=dict(max=max(drifts), mean=float(np.mean(drifts))) if drifts else None,
            ))
            status(output, "recorded", "Recording saved", session=str(session.relative_to(PROJECT_ROOT)), frames=accepted)
        else:
            status(output, "stopped", "Camera preview stopped")
    del devices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/recon3d/tracking_capture.yaml")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    try:
        run(args.out, args.config)
    except Exception as exc:
        status(args.out, "failed", str(exc), error=type(exc).__name__)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
