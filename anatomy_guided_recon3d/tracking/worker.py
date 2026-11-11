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

import numpy as np

from anatomy_guided_recon3d.tracking.runtime import check_paths, read_settings, status, write_json
from anatomy_guided_recon3d.tracking.sequence import validate_sequence


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
    device = settings["device"]
    video = torch.from_numpy(data["video"]).permute(0, 3, 1, 2).float().to(device) / 255.0
    tensor = {key: torch.from_numpy(data[key]).float().to(device)
              for key in ("depths", "intrinsics", "extrinsics", "query_point")}
    status(output, "tracking", f"Tracking {point_count} points through {frame_count} frames", frames=frame_count, points=point_count)
    # FP32 works on RTX 20-series as well as newer GPUs. The released CLI's
    # unconditional BF16 autocast is unsupported on the workstation's RTX 2070.
    with torch.inference_mode():
        coords, visible = inference(
            model=model, video=video, **tensor, num_iters=int(settings["num_iters"]),
            grid_size=int(settings["support_grid_size"]), vis_threshold=float(settings["visibility_threshold"]),
        )
    coords = coords.float().cpu().numpy().reshape(frame_count, point_count, 3)
    visible = visible.cpu().numpy().reshape(frame_count, point_count).astype(bool)
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
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=settings["upstream_repo"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    result["method"] = {"name": "TAPIP3D", "upstream_revision": revision, "knn_backend": settings["knn_backend"],
                        "precision": "float32", "elapsed_s": round(elapsed, 3),
                        "visibility_threshold": settings["visibility_threshold"]}
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
