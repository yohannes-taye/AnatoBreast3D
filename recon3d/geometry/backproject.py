from __future__ import annotations

import numpy as np
import open3d as o3d

from recon3d.config import CameraIntrinsics, DepthConfig


def valid_depth_mask(depth_raw: np.ndarray, depth_config: DepthConfig) -> np.ndarray:
    """Return valid raw depth mask.

    Input shape/unit/frame: depth_raw is HxW raw depth aligned to RGB; raw units
    are converted using ``depth_config.depth_scale_m`` to camera-frame metres.
    Return type: HxW bool ndarray.
    """
    depth_m = depth_raw.astype(np.float64) * depth_config.depth_scale_m
    valid = (
        np.isfinite(depth_m)
        & (depth_raw > 0)
        & (depth_m >= depth_config.min_depth_m)
        & (depth_m <= depth_config.max_depth_m)
    )
    return np.asarray(valid, dtype=bool)


def backproject_depth_to_arrays(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_config: DepthConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project masked aligned depth to camera-frame arrays.

    Input shape/unit/frame: depth_raw is HxW raw depth aligned to RGB; mask is
    HxW bool/0-1; rgb is HxWx3 uint8 RGB; intrinsics are color-camera pixels.
    Output points are camera-frame metres and colors are RGB in [0, 1].
    Return type: ``(points_cam_m, colors, pixels_uv)``.
    """
    if depth_raw.ndim != 2:
        raise ValueError("depth_raw must be HxW")
    if mask.shape != depth_raw.shape:
        raise ValueError("mask must match depth_raw shape")
    if rgb.shape[:2] != depth_raw.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 and match depth_raw shape")
    if (intrinsics.width, intrinsics.height) != (depth_raw.shape[1], depth_raw.shape[0]):
        raise ValueError("intrinsics must match depth image size")

    keep = mask.astype(bool) & valid_depth_mask(depth_raw, depth_config)
    if not keep.any():
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
        )

    v, u = np.nonzero(keep)
    z = depth_raw[v, u].astype(np.float64) * depth_config.depth_scale_m
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float64)
    colors = rgb[v, u].astype(np.float64) / 255.0
    pixels = np.column_stack((u, v)).astype(np.int32)
    return points, colors, pixels


def make_point_cloud(
    points_m: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    estimate_normals: bool = False,
) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud.

    Input shape/unit/frame: points_m is Nx3 in metres; colors is optional Nx3 RGB
    in [0, 1] in the same point order.
    Return type: ``open3d.geometry.PointCloud``.
    """
    if points_m.ndim != 2 or points_m.shape[1] != 3:
        raise ValueError("points_m must be Nx3")
    if not np.isfinite(points_m).all():
        raise ValueError("points_m must be finite")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_m.astype(np.float64, copy=False))
    if colors is not None:
        if colors.shape != points_m.shape:
            raise ValueError("colors must be Nx3 and match points_m")
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64, copy=False))
    if estimate_normals and len(cloud.points) > 0:
        cloud.estimate_normals()
    return cloud


def backproject_masked_depth(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_config: DepthConfig,
    *,
    estimate_normals: bool = False,
) -> o3d.geometry.PointCloud:
    """Back-project masked aligned depth to an Open3D camera-frame cloud.

    Input shape/unit/frame: depth_raw is HxW raw depth aligned to RGB; mask is
    HxW bool/0-1; rgb is HxWx3 uint8 RGB. Output points are camera-frame metres.
    Return type: ``open3d.geometry.PointCloud``.
    """
    points, colors, _ = backproject_depth_to_arrays(depth_raw, mask, rgb, intrinsics, depth_config)
    return make_point_cloud(points, colors, estimate_normals=estimate_normals)
