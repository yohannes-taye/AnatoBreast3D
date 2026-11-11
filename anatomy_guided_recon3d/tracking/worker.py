"""Run the released TAPIP3D network in a separate, optional Python process."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import hashlib

import numpy as np
import cv2

from anatomy_guided_recon3d.tracking.runtime import check_paths, read_settings, status, write_json
from anatomy_guided_recon3d.tracking.sequence import validate_sequence


def project_world_tracks(coords: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> list:
    """Project world-metre predictions back onto the matching rectified RGB grid."""
    cameras = np.einsum('tij,tnj->tni', extrinsics[:,:3,:3], coords) + extrinsics[:,:3,3][:,None,:]
    projected = np.einsum('tij,tnj->tni', intrinsics, cameras)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[:,:,:2] / projected[:,:,2:]
    valid = np.isfinite(pixels).all(axis=-1) & (cameras[:,:,2] > 0)
    safe = pixels.astype(object)
    safe[~valid] = None
    return safe.tolist()


def make_preview(data, coords: np.ndarray, visible: np.ndarray) -> dict:
    """Export real world trajectories plus sampled measured depth for context."""
    contexts = []
    for depth, rgb, K, E in zip(data["depths"], data["video"], data["intrinsics"], data["extrinsics"]):
        y, x = np.where(depth > 0)
        selected = np.linspace(0, max(0, len(x) - 1), min(1400, len(x))).round().astype(int)
        y, x = y[selected], x[selected]
        xyz = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(K).T
        xyz *= depth[y, x, None]
        Twc = np.linalg.inv(E)
        xyz = xyz @ Twc[:3, :3].T + Twc[:3, 3]
        contexts.append({"xyz": xyz.round(5).tolist(), "rgb": rgb[y, x].tolist()})
    displacement = np.linalg.norm(coords - data["query_point"][None, :, 1:], axis=-1) * 1000
    finite = np.isfinite(coords).all(axis=-1)
    visible = visible & finite
    safe_coords = coords.astype(object)
    safe_coords[~finite] = None
    displacement = np.where(visible, displacement, 0)
    return {
        "schema": "anatobreast3d_tapip3d_tracks_v1", "coordinate_frame": "world", "length_unit": "metre",
        "camera_id": str(data.get("camera_id", "unknown")), "frame_count": len(coords), "point_count": coords.shape[1],
        "coords": safe_coords.tolist(), "visible": visible.tolist(), "displacement_mm": displacement.round(3).tolist(),
        "query_points": data["query_point"].tolist(), "context": contexts,
        "timestamps_ns": data.get("timestamps_ns", np.arange(len(coords), dtype=np.int64)).tolist(),
    }


def run(input_path: Path, output: Path, settings: dict) -> dict:
    check_paths(settings)
    with np.load(input_path, allow_pickle=False) as archive:
        data = dict(archive)
    frame_count, point_count = validate_sequence(data)
    status(output, "loading", "Loading TAPIP3D", frames=frame_count, points=point_count)
    # Upstream uses generic top-level names (models, utils, datasets). They must
    # never be imported into the reconstruction server's Python process.
    sys.path.insert(0, settings["upstream_repo"])
    import torch
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Tracking is configured for CUDA, but CUDA is unavailable in the tracking environment")
    if settings["knn_backend"] == "torch":
        from anatomy_guided_recon3d.tracking.torch_knn import install
        install()
    from utils.inference_utils import inference
    import models
    from copy import deepcopy
    started = time.perf_counter()
    checkpoint = torch.load(settings["checkpoint"], map_location="cpu", weights_only=False)
    model_config = deepcopy(checkpoint["cfg"]["model"])
    # The released checkpoint includes ALL encoder weights. Initializing an
    # unrelated pretrained encoder first only downloads weights then overwrites
    # them. Disable that construction-time download, then load strictly.
    model_config["encoder"]["pretrained"] = False
    model = models.from_config(model_config, image_size=checkpoint["cfg"]["train_dataset"]["resolution"])
    model.load_state_dict(checkpoint["weight"], strict=True)
    model.set_eval_mode("raw")
    model.eval().to(settings["device"])
    del checkpoint
    # The upstream model supports chunked image encoding and maintains point
    # identities through its own overlapping windows. Do not restart queries
    # independently for each window of a long recording.
    original_encode = model.encode_rgbs
    def encoder_chunks(rgb_obs, chunk_size=None):
        return original_encode(rgb_obs, chunk_size=chunk_size or int(settings.get("encoder_chunk_size", 8)))
    model.encode_rgbs = encoder_chunks
    device = settings["device"]
    inference_data = dict(data)
    if frame_count < model.seq_len:
        for key in ("video", "depths", "intrinsics", "extrinsics"):
            inference_data[key] = np.concatenate([data[key], np.repeat(data[key][-1:], model.seq_len-frame_count, axis=0)])
    video = torch.from_numpy(inference_data["video"]).permute(0, 3, 1, 2).float().to(device) / 255.0
    tensor = {key: torch.from_numpy(inference_data[key]).float().to(device)
              for key in ("depths", "intrinsics", "extrinsics", "query_point")}
    status(output, "tracking", f"Tracking {point_count} points through {frame_count} frames", frames=frame_count, points=point_count)
    # FP32 works on RTX 20-series as well as newer GPUs. The released CLI's
    # unconditional BF16 autocast is unsupported on the workstation's RTX 2070.
    with torch.inference_mode():
        coords, visible = inference(
            model=model, video=video, **tensor, num_iters=int(settings["num_iters"]),
            grid_size=int(settings["support_grid_size"]), vis_threshold=float(settings["visibility_threshold"]),
        )
    inference_frames = len(inference_data["video"])
    coords = coords.float().cpu().numpy().reshape(inference_frames, point_count, 3)[:frame_count]
    visible = visible.cpu().numpy().reshape(inference_frames, point_count)[:frame_count].astype(bool)
    visible &= np.isfinite(coords).all(axis=-1)
    elapsed = time.perf_counter() - started
    np.savez_compressed(output / "tracks.npz", coords_world_m=coords, visible=visible,
                        query_points=data["query_point"], timestamps_ns=data.get("timestamps_ns", np.arange(frame_count)),
                        coordinate_frame=np.asarray("world"), length_unit=np.asarray("metre"))
    with (output / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp_ns", "point_id", "x_world_m", "y_world_m", "z_world_m", "visible"])
        times = data.get("timestamps_ns", np.arange(frame_count))
        for t in range(frame_count):
            for n in range(point_count):
                writer.writerow([t, int(times[t]), n, *coords[t, n].tolist(), bool(visible[t, n])])
    result = make_preview(data, coords, visible)
    safe_pixels = project_world_tracks(coords, data["intrinsics"], data["extrinsics"])
    frame_dir = output / "frames"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []
    for index, rgb in enumerate(data["video"]):
        path = frame_dir / f"{index:04d}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError("Could not save tracking review frame")
        frame_paths.append(path.relative_to(output).as_posix())
    result.update(pixel_tracks=safe_pixels, rgb_frames=frame_paths,
                  video_size=[int(data["video"].shape[2]), int(data["video"].shape[1])],
                  camera_T_world_cam=np.linalg.inv(data["extrinsics"][0]).tolist())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=settings["upstream_repo"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    result["method"] = {"name": "TAPIP3D", "upstream_revision": revision, "knn_backend": settings["knn_backend"],
                        "precision": "float32", "elapsed_s": round(elapsed, 3),
                        "visibility_threshold": settings["visibility_threshold"],
                        "encoder_chunk_size": settings.get("encoder_chunk_size", 8),
                        "window_frames": model.seq_len, "identity_policy": "upstream overlapping windows; first-frame query IDs",
                        "inference_frames": inference_frames, "seed_source": settings.get("seed_source", "breast segmentation or supplied mask")}
    timestamps = data.get("timestamps_ns", np.arange(frame_count))
    initial = data["query_point"][None,:,1:]
    motion = np.linalg.norm(coords-initial, axis=-1)*1000
    with Path(settings["checkpoint"]).open("rb") as handle:
        checkpoint_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(output / "tracking_report.json", dict(frames=frame_count, points=point_count,
               duration_s=float((timestamps[-1]-timestamps[0])/1e9), visible_fraction=float(visible.mean()),
               median_displacement_mm=float(np.median(motion[visible])) if visible.any() else None,
               max_displacement_mm=float(np.max(motion[visible])) if visible.any() else None,
               nonfinite_predictions=int((~np.isfinite(coords).all(axis=-1)).sum()),
               checkpoint_sha256=checkpoint_sha, method=result["method"],
               limitation="Integration check; geometric accuracy and occlusion recovery require measured ground truth"))
    write_json(output / "result.json", result)
    status(output, "complete", f"Tracked {point_count} points through {frame_count} frames", frames=frame_count,
           points=point_count, elapsed_s=round(elapsed, 3), result="result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        run(args.input, args.out, read_settings(args.settings))
    except Exception as exc:
        status(args.out, "failed", str(exc), error=type(exc).__name__)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
