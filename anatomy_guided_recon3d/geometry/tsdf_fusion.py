from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import (
    CameraExtrinsics,
    CloudCleanupConfig,
    PerceptionResult,
    RGBDFrame,
    TSDFConfig,
)
from anatomy_guided_recon3d.geometry.cloud_cleanup import cleaned_breast_mask


def _masked_depth(frame: RGBDFrame, perception: PerceptionResult, cleanup: CloudCleanupConfig) -> np.ndarray:
    mask = cleaned_breast_mask(frame.depth_mm, perception.breast_mask, frame.depth_config, cleanup)
    depth = np.array(frame.depth_mm, copy=True)
    depth[~mask] = 0
    return depth


def fuse_tsdf(
    frames: Mapping[str, RGBDFrame],
    perceptions: Mapping[str, PerceptionResult],
    extrinsics: Mapping[str, CameraExtrinsics],
    cleanup: CloudCleanupConfig,
    config: TSDFConfig,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    """Integrate corrected masked RGB-D frames into a TSDF point cloud."""
    diagnostics: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "integrated_cameras": [],
        "errors": [],
    }
    if not config.enabled:
        diagnostics["points"] = 0
        return o3d.geometry.PointCloud(), diagnostics

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(config.voxel_length_m),
        sdf_trunc=float(config.sdf_trunc_m),
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if config.color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        ),
    )
    for cam_id, frame in sorted(frames.items()):
        perception = perceptions.get(cam_id)
        extrinsic = extrinsics.get(cam_id)
        if perception is None or extrinsic is None:
            continue
        depth = _masked_depth(frame, perception, cleanup)
        if not np.any(depth > 0):
            diagnostics["errors"].append({"cam_id": cam_id, "reason": "empty_masked_depth"})
            continue
        color_img = o3d.geometry.Image(np.ascontiguousarray(frame.rgb))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1.0 / float(frame.depth_config.depth_scale_m),
            depth_trunc=float(config.depth_trunc_m),
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            frame.intrinsics.width,
            frame.intrinsics.height,
            frame.intrinsics.fx,
            frame.intrinsics.fy,
            frame.intrinsics.cx,
            frame.intrinsics.cy,
        )
        try:
            volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic.T_world_cam))
            diagnostics["integrated_cameras"].append(cam_id)
        except Exception as exc:  # noqa: BLE001 - TSDF is an optional final-quality backend
            diagnostics["errors"].append({"cam_id": cam_id, "reason": f"{type(exc).__name__}: {exc}"})

    if not diagnostics["integrated_cameras"]:
        diagnostics["points"] = 0
        return o3d.geometry.PointCloud(), diagnostics
    cloud = volume.extract_point_cloud()
    diagnostics["points"] = int(len(cloud.points))
    return cloud, diagnostics
