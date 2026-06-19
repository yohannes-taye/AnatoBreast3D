from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from recon3d.config import CloudCleanupConfig, DepthConfig, RGBDFrame
from recon3d.geometry.backproject import backproject_masked_depth, valid_depth_mask


def flying_depth_edges(depth_m: np.ndarray, valid: np.ndarray, threshold_m: float) -> np.ndarray:
    """Detect depth discontinuities likely to create flying edge points.

    Input shape/unit/frame: depth_m is HxW metres aligned to RGB; valid is HxW
    bool on the same grid; threshold_m is a neighbour depth jump in metres.
    Return type: HxW bool mask where true means reject this depth pixel.
    """
    if threshold_m <= 0:
        return np.zeros_like(valid, dtype=bool)
    edges = np.zeros_like(valid, dtype=bool)
    pairs = (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    )
    for here, there in pairs:
        local = valid[here] & valid[there] & (np.abs(depth_m[here] - depth_m[there]) > threshold_m)
        edges[here] |= local
    return edges


def cleanup_depth_config(frame_depth_config: DepthConfig, config: CloudCleanupConfig) -> DepthConfig:
    """Apply pipeline cleanup depth range while preserving SDK depth scale.

    Input shape/unit/frame: depth scale is metres per raw unit from the SDK.
    Return type: ``DepthConfig`` with the cleanup min/max depth range.
    """
    return DepthConfig(
        depth_scale_m=frame_depth_config.depth_scale_m,
        min_depth_m=config.min_depth_m,
        max_depth_m=config.max_depth_m,
    )


def cleaned_breast_mask(
    depth_raw: np.ndarray,
    breast_mask: np.ndarray,
    depth_config: DepthConfig,
    config: CloudCleanupConfig,
) -> np.ndarray:
    """Return the breast mask after erosion, depth range, and edge cleanup.

    Input shape/unit/frame: mask and depth are HxW aligned RGB/depth pixels;
    depth raw units are converted by ``depth_config`` into metres.
    Return type: HxW bool mask suitable for back-projection.
    """
    mask = breast_mask.astype(bool, copy=False)
    if config.mask_erode_px > 0:
        size = config.mask_erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

    ranged_depth_config = cleanup_depth_config(depth_config, config)
    valid = valid_depth_mask(depth_raw, ranged_depth_config)
    depth_m = depth_raw.astype(np.float64) * ranged_depth_config.depth_scale_m
    edge_reject = flying_depth_edges(depth_m, valid, config.flying_depth_edge_threshold_m)
    return mask & valid & ~edge_reject


def backproject_cleaned_breast_cloud(
    frame: RGBDFrame,
    breast_mask: np.ndarray,
    config: CloudCleanupConfig,
) -> o3d.geometry.PointCloud:
    """Back-project one cleaned per-camera breast mask to camera-frame points.

    Input shape/unit/frame: frame RGB/depth are aligned; output cloud is
    camera-frame metres with RGB colours.
    Return type: ``open3d.geometry.PointCloud``.
    """
    cleaned_mask = cleaned_breast_mask(frame.depth_mm, breast_mask, frame.depth_config, config)
    return backproject_masked_depth(
        frame.depth_mm,
        cleaned_mask,
        frame.rgb,
        frame.intrinsics,
        cleanup_depth_config(frame.depth_config, config),
        estimate_normals=False,
    )
