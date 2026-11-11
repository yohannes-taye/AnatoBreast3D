"""Prepare synchronized, seekable RGB-D inspection frames from a recording."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import gzip
from pathlib import Path
import time
import traceback

import cv2
import numpy as np

from anatomy_guided_recon3d.calibration.io import load_extrinsics
from anatomy_guided_recon3d.sensors.femto_bolt import OrbbecBagCamera, ReplayRGBDCamera
from anatomy_guided_recon3d.tracking.preview_geometry import add_camera_gaps, camera_frustum, make_depth_surface
from anatomy_guided_recon3d.tracking.recording import publish_image, publish_mask
from anatomy_guided_recon3d.tracking.runtime import PROJECT_ROOT, capture_config, status, write_json
from anatomy_guided_recon3d.tracking.sequence import prepare_frame


def collect_samples(camera, interval_ns: int = 200_000_000, limit: int = 300) -> list:
    """Read playback before doing GPU work, so decoding does not drop frames."""
    result = []
    bucket = None
    best_distance = None
    camera.start()
    try:
        while True:
            try:
                frame = camera.read_aligned_frame()
            except StopIteration:
                break
            current = (frame.timestamp_ns + interval_ns//2)//interval_ns
            distance = abs(frame.timestamp_ns-current*interval_ns)
            if bucket is None or current > bucket:
                if len(result) >= limit:
                    break
                result.append(frame)
                bucket, best_distance = current, distance
            elif current == bucket and distance < best_distance:
                result[-1] = frame
                best_distance = distance
    finally:
        camera.stop()
    return result


def synchronize_samples(streams: dict, tolerance_ns: int = 50_000_000) -> list[dict]:
    """Pair real timestamps; do not equate unsynchronized camera frame indices."""
    if not streams or any(not frames for frames in streams.values()):
        raise ValueError("The recording has an empty camera stream")
    reference = next(iter(streams))
    timestamps = {name: [frame.timestamp_ns for frame in frames] for name, frames in streams.items()}
    for values in timestamps.values():
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("Recorded camera timestamps are not strictly increasing")
    pairs = []
    for frame in streams[reference]:
        pair = {reference: frame}
        for name, frames in streams.items():
            if name == reference:
                continue
            position = bisect_left(timestamps[name], frame.timestamp_ns)
            candidates = [i for i in (position-1, position) if 0 <= i < len(frames)]
            index = min(candidates, key=lambda i: abs(timestamps[name][i]-frame.timestamp_ns))
            if abs(timestamps[name][index]-frame.timestamp_ns) > tolerance_ns:
                break
            pair[name] = frames[index]
        if len(pair) == len(streams):
            pairs.append(pair)
    if len(pairs) < 2:
        raise ValueError("No shared RGB-D timeline within 50 ms. Check the recording's camera clocks")
    return pairs


def prepare(session: Path, output: Path, config_path: Path) -> None:
    from anatomy_guided_recon3d.cli import load_recon_config
    from anatomy_guided_recon3d.perception.inference import BreastPerceptionAdapter

    root = session.parent if session.is_file() else session
    config, extrinsics_path = load_recon_config(config_path)
    calibration = root / "extrinsics.json" if (root / "extrinsics.json").is_file() else extrinsics_path
    extrinsics = load_extrinsics(calibration)
    manifest_path = root / "session.json"
    cameras = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["cameras"]:
            name = str(entry.get("id") or entry.get("cam_id"))
            bag = entry.get("bag") or entry.get("bag_path")
            if not bag:
                raise ValueError("The session has no SDK bag for " + name)
            cameras[name] = OrbbecBagCamera(name, root / bag, align_mode="sw", wait_timeout_ms=5000)
    else:
        cameras = {path.name: ReplayRGBDCamera(path.name, path) for path in sorted(root.iterdir())
                   if (path / "metadata.json").is_file()}
    if not cameras or set(cameras)-set(extrinsics):
        raise ValueError("The recording needs a saved calibration for every camera")
    streams = {}
    for name, camera in cameras.items():
        status(output, "preparing", f"Reading {name} RGB-D recording")
        streams[name] = collect_samples(camera)
    pairs = synchronize_samples(streams)
    model = BreastPerceptionAdapter(config=config)
    times = []
    first_time = min(frame.timestamp_ns for frame in pairs[0].values())
    for index, frames in enumerate(pairs):
        status(output, "preparing", f"Preparing playback frame {index+1} of {len(pairs)}", done=index, total=len(pairs))
        perceptions = model.predict_batch(frames)
        views = []
        directory = output / "images"
        directory.mkdir(exist_ok=True)
        for camera_index, (name, frame) in enumerate(frames.items()):
            stem = f"{index:04d}_{name}"
            rgb_path = directory / (stem + "_rgb.jpg")
            depth_path = directory / (stem + "_depth.jpg")
            mask_path = directory / (stem + "_mask.png")
            publish_image(rgb_path, cv2.cvtColor(cv2.resize(frame.rgb, (640, 360)), cv2.COLOR_RGB2BGR))
            depth_m = frame.depth_mm.astype(np.float32)*frame.depth_config.depth_scale_m
            depth_color = cv2.applyColorMap((np.clip((depth_m-.25), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_color[depth_m <= 0] = 0
            publish_image(depth_path, cv2.resize(depth_color, (640, 360), interpolation=cv2.INTER_NEAREST))
            mask = perceptions[name].breast_mask
            publish_mask(mask_path, mask, camera_index)
            prepared = prepare_frame(frame, mask, extrinsics[name].T_world_cam, 512)
            geometry = make_depth_surface(prepared, "scene", "fine", .1, 3.)
            views.append(dict(id=name, serial=name, **geometry, T_world_cam=extrinsics[name].T_world_cam.tolist(),
                              frustum=camera_frustum(prepared),
                              rgb=rgb_path.relative_to(output).as_posix(), depth=depth_path.relative_to(output).as_posix(),
                              mask=mask_path.relative_to(output).as_posix(), timestamp_ns=frame.timestamp_ns,
                              valid_depth_percent=round(100*float(np.mean(depth_m > 0)), 1)))
        add_camera_gaps(views)
        timestamps = [frame.timestamp_ns for frame in frames.values()]
        elapsed = (min(timestamps)-first_time)/1e9
        times.append(round(elapsed, 4))
        payload = dict(
            state="playback", timestamp=time.time(), time_s=elapsed, cameras=views, geometry_version=2,
            calibration=str(calibration), calibration_loaded=True, drift_ms=(max(timestamps)-min(timestamps))/1e6,
            scope="scene", detail="fine", full_scene=True, depth_range_m=[.1, 3.], foreground_only=False,
        )
        destination = output / "frames" / f"{index:04d}.json.gz"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(gzip.compress(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"), compresslevel=1))
    write_json(output / "manifest.json", dict(schema="anatobreast3d_rgbd_playback_v1", source=str(root),
               frame_count=len(times), times_s=times, duration_s=times[-1], cameras=list(cameras),
               truncated=any(len(frames) >= 300 for frames in streams.values()),
               detail="fine", frames=[f"frames/{i:04d}.json.gz" for i in range(len(times))]))
    status(output, "complete", f"Playback ready: {len(times)} frames from {len(cameras)} cameras", frames=len(times),
           duration_s=times[-1], manifest="manifest.json", source=str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=capture_config())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        prepare(args.session, args.out, args.config)
    except Exception as exc:
        status(args.out, "failed", str(exc), error=type(exc).__name__)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
