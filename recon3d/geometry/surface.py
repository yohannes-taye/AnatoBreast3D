from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import open3d as o3d

from recon3d.config import SurfaceConfig


def optional_poisson_surface_points(
    cloud: o3d.geometry.PointCloud,
    camera_centers_world_m: Sequence[np.ndarray],
    config: SurfaceConfig,
) -> o3d.geometry.PointCloud:
    """Optionally smooth a fused cloud through Poisson surface reconstruction.

    Input shape/unit/frame: cloud and camera centres are world-frame metres.
    Return type: point cloud; when mesh is enabled, the mesh is sampled back to
    points so downstream PLY/viewer behaviour stays unchanged.
    """
    if not config.mesh_enabled or len(cloud.points) <= config.min_points:
        return cloud

    source = copy.deepcopy(cloud)
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.normal_radius_m,
            max_nn=config.normal_max_nn,
        )
    )
    if camera_centers_world_m:
        center = np.mean(np.asarray(camera_centers_world_m, dtype=np.float64), axis=0)
        source.orient_normals_towards_camera_location(center)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        source,
        depth=config.poisson_depth,
    )
    densities_np = np.asarray(densities)
    if densities_np.size and config.density_trim_quantile > 0:
        threshold = float(np.quantile(densities_np, config.density_trim_quantile))
        mesh.remove_vertices_by_mask(densities_np < threshold)
    mesh = mesh.crop(source.get_axis_aligned_bounding_box())
    if len(mesh.triangles) == 0:
        return cloud
    return mesh.sample_points_uniformly(config.sample_points)
