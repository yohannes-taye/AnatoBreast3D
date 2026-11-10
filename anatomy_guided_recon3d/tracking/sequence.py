"""Bridge aligned Orbbec RGB-D into TAPIP3D's world-coordinate contract."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anatomy_guided_recon3d.config import RGBDFrame


def prepare_frame(frame: RGBDFrame, mask: np.ndarray, T_world_cam: np.ndarray, max_side: int = 512) -> dict:
    """Rectify one aligned RGB-D frame; depth and world coordinates are metres.

    TAPIP3D expects world-to-camera extrinsics, the inverse of AnatoBreast3D's
    camera-to-world calibration. Use the fixed rig calibration, never the
    per-frame anatomy/ICP correction, which could absorb actual deformation.
    """
    if mask.shape != frame.depth_mm.shape:
        raise ValueError("Tracking mask must align with the RGB-D image")
    if not 128 <= max_side <= 1024:
        raise ValueError("Tracking image size must be between 128 and 1024 pixels")
    source_h, source_w = mask.shape
    scale = min(1.0, max_side / max(source_h, source_w))
    width = max(32, int(round(source_w * scale / 8)) * 8)
    height = max(32, int(round(source_h * scale / 8)) * 8)
    sx, sy = width / source_w, height / source_h
    K = np.array(frame.intrinsics.K, dtype=np.float64, copy=True)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    distortion = frame.intrinsics.distortion
    if distortion is not None and np.any(distortion):
        map_x, map_y = cv2.initUndistortRectifyMap(
            frame.intrinsics.K, distortion, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        rgb = cv2.remap(frame.rgb, map_x, map_y, cv2.INTER_LINEAR)
        depth = cv2.remap(frame.depth_mm, map_x, map_y, cv2.INTER_NEAREST)
        mask = cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    else:
        rgb = cv2.resize(frame.rgb, (width, height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(frame.depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32) * float(frame.depth_config.depth_scale_m)
    valid = np.isfinite(depth) & (depth >= frame.depth_config.min_depth_m) & (depth <= frame.depth_config.max_depth_m)
    depth[~valid] = 0
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64)
    if T_world_cam.shape != (4, 4) or not np.isfinite(T_world_cam).all():
        raise ValueError("Tracking requires a finite 4x4 camera-to-world calibration")
    if not np.allclose(T_world_cam[3], [0, 0, 0, 1]):
        raise ValueError("Invalid camera-to-world homogeneous transform")
    return {
        "video": rgb, "depths": depth, "intrinsics": K.astype(np.float32),
        "extrinsics": np.linalg.inv(T_world_cam).astype(np.float32),
        "mask": mask.astype(bool) & valid, "timestamp_ns": int(frame.timestamp_ns),
    }


def seed_queries(frame: dict, max_points: int = 128) -> np.ndarray:
    """Select spatially distributed breast pixels with valid, interior depth."""
    if not 1 <= max_points <= 1024:
        raise ValueError("Track between 1 and 1024 points")
    valid = frame["mask"].astype(np.uint8)
    # Avoid unreliable mask boundaries and depth discontinuities at initialization.
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0
    depth = frame["depths"]
    near = cv2.erode(depth, np.ones((3, 3), np.uint8))
    far = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    valid &= (near > 0) & ((far - near) <= 0.02)
    y, x = np.where(valid)
    if not len(x):
        raise ValueError("No valid interior breast depth for tracking; check the segmentation and depth alignment")
    # Choose the valid pixel nearest each cell centre, avoiding an ordering bias.
    n = int(np.ceil(np.sqrt(max_points * 2)))
    chosen = []
    rows = np.arange(y.min(), y.max() + 1)
    columns = np.arange(x.min(), x.max() + 1)
    for yy in np.array_split(rows, min(n, len(rows))):
        for xx in np.array_split(columns, min(n, len(columns))):
            cy, cx = (yy[0] + yy[-1]) / 2, (xx[0] + xx[-1]) / 2
            indices = np.flatnonzero((y >= yy[0]) & (y <= yy[-1]) & (x >= xx[0]) & (x <= xx[-1]))
            if len(indices):
                chosen.append(indices[np.argmin((x[indices] - cx) ** 2 + (y[indices] - cy) ** 2)])
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_points).round().astype(int)]
    pixels = np.column_stack([x[chosen], y[chosen], np.ones(len(chosen))])
    camera = (pixels @ np.linalg.inv(frame["intrinsics"]).T) * depth[y[chosen], x[chosen], None]
    T = np.linalg.inv(frame["extrinsics"])
    world = camera @ T[:3, :3].T + T[:3, 3]
    return np.column_stack([np.zeros(len(world)), world]).astype(np.float32)


def save_sequence(path: Path, frames: list[dict], cam_id: str, max_points: int = 128) -> None:
    """Save one physical camera over time; never treat different views as time."""
    if len(frames) < 2:
        raise ValueError("Point tracking needs at least two temporal frames")
    timestamps = np.asarray([frame["timestamp_ns"] for frame in frames], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Tracking frames must have strictly increasing timestamps")
    data = {name: np.stack([frame[name] for frame in frames]) for name in ("video", "depths", "intrinsics", "extrinsics")}
    data.update(query_point=seed_queries(frames[0], max_points), timestamps_ns=timestamps,
                first_mask=frames[0]["mask"], camera_id=np.asarray(cam_id),
                coordinate_frame=np.asarray("world"), length_unit=np.asarray("metre"))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def validate_sequence(data) -> tuple[int, int]:
    """Reject malformed or uncalibrated inputs before allocating GPU memory."""
    video, depth, K, E, query = [data[key] for key in ("video", "depths", "intrinsics", "extrinsics", "query_point")]
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError("video must be uint8 T x H x W x 3 RGB")
    T, H, W, _ = video.shape
    if not 2 <= T <= 96 or H % 8 or W % 8:
        raise ValueError("Use 2–96 frames and image dimensions divisible by 8")
    if depth.shape != (T, H, W) or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depths must be finite nonnegative T x H x W camera Z depths in metres")
    if not np.any(depth[0] > 0):
        raise ValueError("The first frame has no valid depth")
    if K.shape != (T, 3, 3) or E.shape != (T, 4, 4) or not np.isfinite(K).all() or not np.isfinite(E).all():
        raise ValueError("Provide finite per-frame camera intrinsics and world-to-camera extrinsics")
    if np.any(K[:, (0, 1), (0, 1)] <= 0) or not np.allclose(K[:, 2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    if not np.allclose(E[:, 3], [0, 0, 0, 1]) or np.any(np.abs(np.linalg.det(E[:, :3, :3]) - 1) > 0.01):
        raise ValueError("Invalid rigid world-to-camera extrinsics")
    rotation = E[:, :3, :3]
    if not np.allclose(rotation @ np.swapaxes(rotation, 1, 2), np.eye(3), atol=.01):
        raise ValueError("World-to-camera rotation must be orthonormal")
    if query.ndim != 2 or query.shape[1] != 4 or not 1 <= len(query) <= 1024 or not np.isfinite(query).all():
        raise ValueError("query_point must contain 1–1024 [frame, world X, world Y, world Z] queries")
    if np.any(query[:, 0] != 0):
        raise ValueError("This integration initializes all point identities at the first captured frame")
    return T, len(query)
