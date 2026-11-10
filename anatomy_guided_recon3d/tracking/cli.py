"""Track a recorded RGB-D sequence using the released TAPIP3D checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2
import numpy as np

from anatomy_guided_recon3d.tracking.runtime import PROJECT_ROOT, check_paths, read_settings, status
from anatomy_guided_recon3d.tracking.sequence import prepare_frame, save_sequence


def capture_recorded(session: Path, output: Path, camera_id: str | None, config_path: Path,
                     settings: dict, skip: int = 0, mask_path: Path | None = None) -> Path:
    from anatomy_guided_recon3d.calibration.io import load_extrinsics
    from anatomy_guided_recon3d.cli import load_recon_config
    from anatomy_guided_recon3d.sensors.femto_bolt import OrbbecBagCamera, ReplayRGBDCamera

    root = session.parent if session.is_file() else session
    config, extrinsics_path = load_recon_config(config_path)
    if (root / "extrinsics.json").is_file():
        extrinsics_path = root / "extrinsics.json"
    extrinsics = load_extrinsics(extrinsics_path)
    manifest_path = root / "session.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cameras = {str(entry.get("id") or entry.get("cam_id")): entry for entry in manifest["cameras"]}
        camera_id = camera_id or next(iter(cameras))
        if camera_id not in cameras:
            raise ValueError(f"Camera {camera_id} is not in this recording")
        entry = cameras[camera_id]
        bag = Path(entry.get("bag") or entry.get("bag_path") or "")
        if not bag.suffix.lower() == ".bag":
            raise ValueError("This session does not contain Orbbec SDK bag recordings")
        camera = OrbbecBagCamera(camera_id, root / bag, align_mode="sw")
    else:
        candidates = sorted(p.name for p in root.iterdir() if (p / "metadata.json").is_file())
        if not candidates:
            raise ValueError("Expected an SDK bag session or camera folders with RGB/depth PNGs and metadata.json")
        camera_id = camera_id or candidates[0]
        if camera_id not in candidates:
            raise ValueError(f"Camera {camera_id} is not in this recording")
        camera = ReplayRGBDCamera(camera_id, root / camera_id)
    if camera_id not in extrinsics:
        raise ValueError(f"No saved calibration for {camera_id}")
    count, stride = int(settings["frames"]), int(settings["stride"])
    if not 2 <= count <= 96 or not 1 <= stride <= 30 or skip < 0:
        raise ValueError("Use 2–96 frames, stride 1–30, and a nonnegative skip")
    raw_frames = []
    previous = -1
    camera.start()
    try:
        index = 0
        while len(raw_frames) < count:
            try:
                frame = camera.read_aligned_frame()
            except StopIteration:
                break
            index += 1
            if index <= skip or (index - skip - 1) % stride:
                continue
            if frame.timestamp_ns <= previous:
                continue
            raw_frames.append(frame)
            previous = frame.timestamp_ns
            status(output, "preparing", f"Read {len(raw_frames)} of {count} RGB-D frames", frames=len(raw_frames), camera=camera_id)
    finally:
        camera.stop()
    if len(raw_frames) < 2:
        raise ValueError("The recording did not contain two usable temporal frames")
    # Finish playback before loading/running segmentation. SDK playback is
    # real-time; doing GPU work while it runs silently skips buffered frames.
    first = raw_frames[0]
    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read breast mask: {mask_path}")
        mask = mask > 0
    else:
        from anatomy_guided_recon3d.perception.inference import BreastPerceptionAdapter
        model = BreastPerceptionAdapter(config=config)
        mask = model.predict_batch({camera_id: first})[camera_id].breast_mask
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cv2.imwrite(str(output / "first_rgb.jpg"), cv2.cvtColor(first.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "first_mask.png"), mask.astype(np.uint8) * 255)
    frames = [prepare_frame(frame, mask if index == 0 else np.ones(frame.depth_mm.shape, dtype=bool),
                           extrinsics[camera_id].T_world_cam, int(settings["max_side"]))
              for index, frame in enumerate(raw_frames)]
    destination = output / "input.npz"
    save_sequence(destination, frames, camera_id, int(settings["max_points"]))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path, help="SDK bag session or legacy RGB-D camera folders")
    source.add_argument("--input", type=Path, help="Already prepared, calibrated TAPIP3D NPZ")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", help="One physical camera over time; run other cameras separately")
    parser.add_argument("--config", type=Path, default=Path("configs/recon3d/tracking_capture.yaml"))
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--mask", type=Path, help="Optional first-frame breast mask, otherwise use breast segmentation")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        settings = read_settings(args.settings)
        check_paths(settings)
        source_path = args.input or capture_recorded(args.session, args.out, args.camera, args.config,
                                                     settings, args.skip, args.mask)
        if args.prepare_only:
            status(args.out, "prepared", "RGB-D sequence prepared", input=str(source_path))
            return
        command = [settings["python"], "-m", "anatomy_guided_recon3d.tracking.worker", "--input", str(source_path.resolve()),
                   "--out", str(args.out.resolve())]
        if args.settings:
            command += ["--settings", str(args.settings.resolve())]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except Exception as exc:
        # Preserve the worker's more specific error, if it already published one.
        existing = args.out / "status.json"
        if not existing.exists() or json.loads(existing.read_text())["state"] != "failed":
            status(args.out, "failed", str(exc), error=type(exc).__name__)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
