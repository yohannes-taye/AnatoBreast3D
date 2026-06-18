"""Capture one frame per camera and fuse the FULL scene into a world-frame cloud.

This is a calibration/plumbing sanity check (not the breast pipeline): it skips
the breast mask and fuses all valid depth, so the three cameras' clouds should
visually overlap if the extrinsics are correct. Output goes to the location the
browser viewer reads (runs/recon3d_live/), reusing the real geometry + fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon3d.config import CameraIntrinsics, DepthConfig, FusionConfig  # noqa: E402
from recon3d.geometry.backproject import backproject_masked_depth  # noqa: E402
from recon3d.geometry.fusion import PointCloudFusionBackend, point_cloud_bounds  # noqa: E402
from recon3d.calibration.io import load_extrinsics  # noqa: E402
from recon3d.sensors.femto_bolt import FemtoBoltCamera  # noqa: E402

DEFAULT_CAMERAS = {"cam0": "CL8T754004D", "cam1": "CL8T7540197", "cam2": "CL8T75400XA"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture + full-scene 3-camera fusion for the viewer")
    ap.add_argument("--out", default="runs/recon3d_live")
    ap.add_argument("--extrinsics", default="configs/recon3d/extrinsics.json")
    ap.add_argument("--stride", type=int, default=2, help="Pixel subsample to keep the cloud light.")
    ap.add_argument("--voxel", type=float, default=0.004, help="Fusion voxel size (m).")
    ap.add_argument("--max-depth", type=float, default=1.2, help="Clip depth beyond this (m).")
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--align-mode", choices=["hw", "sw"], default="sw")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--serials", nargs="*", default=None)
    args = ap.parse_args()

    cams = dict(zip(args.cameras, args.serials)) if args.cameras and args.serials else dict(DEFAULT_CAMERAS)
    extrinsics = load_extrinsics(Path(args.extrinsics))
    s = max(1, args.stride)
    t0 = time.perf_counter()

    clouds = {}
    drift_lo, drift_hi = None, None
    for cam_id, serial in cams.items():
        cam = FemtoBoltCamera(cam_id, serial_number=serial, align_mode=args.align_mode)
        try:
            cam.start()
            for _ in range(8):
                cam.read_aligned_frame()
            f = cam.read_aligned_frame()
        except Exception as exc:  # noqa: BLE001
            print(f"{cam_id}: capture failed: {exc!r}")
            cam.stop()
            continue
        cam.stop()
        if cam_id not in extrinsics:
            print(f"{cam_id}: no extrinsics, skipping")
            continue
        ts = f.timestamp_ns
        drift_lo = ts if drift_lo is None else min(drift_lo, ts)
        drift_hi = ts if drift_hi is None else max(drift_hi, ts)

        rgb = np.ascontiguousarray(f.rgb[::s, ::s])
        depth = np.ascontiguousarray(f.depth_mm[::s, ::s])
        K = f.intrinsics
        intr = CameraIntrinsics(width=rgb.shape[1], height=rgb.shape[0],
                                fx=K.fx / s, fy=K.fy / s, cx=K.cx / s, cy=K.cy / s)
        dcfg = DepthConfig(depth_scale_m=f.depth_config.depth_scale_m,
                           min_depth_m=args.min_depth, max_depth_m=args.max_depth)
        mask = np.ones(depth.shape, dtype=bool)  # full scene
        cloud = backproject_masked_depth(depth, mask, rgb, intr, dcfg)
        clouds[cam_id] = cloud
        print(f"{cam_id}: {len(cloud.points)} pts (board/scene), depth_scale={dcfg.depth_scale_m:.6f}")

    if not clouds:
        raise SystemExit("No clouds captured")

    fused = PointCloudFusionBackend(FusionConfig(voxel_size_m=args.voxel)).fuse(clouds, extrinsics)
    total_ms = (time.perf_counter() - t0) * 1000.0
    drift_ms = ((drift_hi - drift_lo) / 1e6) if (drift_lo is not None and drift_hi is not None) else 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out / "fused_breast.ply"), fused)
    if not ok:
        raise SystemExit("Failed to write point cloud")
    (out / "run_summary.json").write_text(json.dumps({
        "timestamp_ns": int(drift_hi or 0),
        "timings_ms": {"total_ms": total_ms, "perception_ms": 0.0,
                       "fusion_ms": total_ms, "timestamp_drift_ms": drift_ms},
        "fused_cloud": point_cloud_bounds(fused),
        "note": "full-scene calibration-check fusion (no breast mask)",
    }, indent=2), encoding="utf-8")
    (out / "nipple_world.json").write_text(json.dumps({}, indent=2), encoding="utf-8")

    b = point_cloud_bounds(fused)
    print(f"\nFused {len(fused.points)} points -> {out/'fused_breast.ply'}")
    print(f"bounds (m): {b}")
    print(f"per-camera timestamp drift: {drift_ms:.1f} ms")


if __name__ == "__main__":
    main()
