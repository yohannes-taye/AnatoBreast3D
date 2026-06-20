from __future__ import annotations

import copy
from typing import Mapping, Protocol

import numpy as np
import open3d as o3d

from anatomy_guided_recon3d.config import CameraExtrinsics, FusionConfig


class FusionBackend(Protocol):
    """Protocol for multi-camera point cloud fusion backends."""

    def fuse(
        self,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
    ) -> o3d.geometry.Geometry:
        """Fuse camera-frame clouds into world-frame geometry.

        Input shape/unit/frame: each cloud is in its camera frame in metres;
        extrinsics provide ``T_world_cam`` for each camera ID.
        Return type: Open3D world-frame geometry.
        """
        ...


def _copy_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return copy.deepcopy(cloud)


def transform_cloud_to_world(
    cloud_cam: o3d.geometry.PointCloud,
    extrinsic: CameraExtrinsics,
) -> o3d.geometry.PointCloud:
    """Transform one cloud from camera to world frame.

    Input shape/unit/frame: cloud points are camera-frame metres; ``T_world_cam``
    maps camera homogeneous points to world homogeneous points.
    Return type: copied world-frame ``PointCloud``.
    """
    cloud_world = _copy_cloud(cloud_cam)
    cloud_world.transform(extrinsic.T_world_cam)
    return cloud_world


def merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge point clouds without downsampling.

    Input shape/unit/frame: all clouds are in a common frame in metres.
    Return type: merged ``PointCloud``.
    """
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged += cloud
    return merged


class PointCloudFusionBackend:
    """Default low-latency world-frame point cloud fusion backend."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        clouds_cam: Mapping[str, o3d.geometry.PointCloud],
        extrinsics: Mapping[str, CameraExtrinsics],
    ) -> o3d.geometry.PointCloud:
        """Fuse camera-frame clouds into one world-frame point cloud.

        Input shape/unit/frame: camera clouds are in metres; extrinsics map
        camera frame to world frame. Missing extrinsics raise ``KeyError``.
        Return type: world-frame ``open3d.geometry.PointCloud``.
        """
        world_clouds: list[o3d.geometry.PointCloud] = []
        for cam_id, cloud in clouds_cam.items():
            if cam_id not in extrinsics:
                raise KeyError(f"Missing extrinsics for camera '{cam_id}'")
            if len(cloud.points) == 0:
                continue
            world_clouds.append(transform_cloud_to_world(cloud, extrinsics[cam_id]))
        merged = merge_clouds(world_clouds)
        if len(merged.points) == 0:
            return merged
        if self.config.voxel_size_m > 0:
            merged = merged.voxel_down_sample(self.config.voxel_size_m)
        if len(merged.points) > self.config.outlier_nb_neighbors:
            merged, _ = merged.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio,
            )
        if self.config.estimate_normals and len(merged.points) > 0:
            merged.estimate_normals()
        return merged


def point_cloud_bounds(cloud: o3d.geometry.PointCloud) -> dict[str, list[float] | int]:
    """Compute simple world-frame point cloud bounds.

    Input shape/unit/frame: cloud is in metres.
    Return type: dict with point count and min/max bounds.
    """
    points = np.asarray(cloud.points)
    if points.size == 0:
        return {"points": 0, "min_bound_m": [], "max_bound_m": []}
    return {
        "points": int(points.shape[0]),
        "min_bound_m": points.min(axis=0).tolist(),
        "max_bound_m": points.max(axis=0).tolist(),
    }

