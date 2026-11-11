"""Observed depth-grid surfaces for live inspection; never fill missing depth."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


SCOPES = ("breast_area", "breast", "scene")
DETAIL_STRIDES = {"fast": 8, "balanced": 4, "fine": 2}


def camera_frustum(prepared: dict) -> dict:
    """Actual off-centre pinhole rays, expressed in the calibrated world frame."""
    height, width = prepared["depths"].shape
    pixels = np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]], np.float64)
    rays = pixels @ np.linalg.inv(prepared["intrinsics"]).T
    Twc = np.linalg.inv(prepared["extrinsics"])
    return dict(origin_world_m=Twc[:3,3].tolist(), rays_world=(rays @ Twc[:3,:3].T).tolist())


def make_depth_surface(prepared: dict, scope: str = "breast_area", detail: str = "balanced",
                       near: float = .3, far: float = 1.) -> dict:
    """Back-project matched RGB/depth pixels and join valid adjacent samples.

    Coordinates are world metres. A 20 mm camera-depth discontinuity prevents a
    triangle, so holes and separate depth layers are not bridged. Breast-area
    mode uses an enlarged detector bounding box, not an invented completion.
    """
    if scope not in SCOPES or detail not in DETAIL_STRIDES:
        raise ValueError("Unsupported 3D area or detail")
    depth = prepared["depths"]
    mask = prepared["mask"].astype(bool)
    breast_area = np.zeros(depth.shape, bool)
    y, x = np.where(mask)
    if len(x):
        padx = max(12, int((x.max() - x.min() + 1) * .5))
        pady = max(12, int((y.max() - y.min() + 1) * .5))
        breast_area[max(0, y.min()-pady):y.max()+pady+1, max(0, x.min()-padx):x.max()+padx+1] = True
    region = mask if scope == "breast" else breast_area if scope == "breast_area" else np.ones(depth.shape, bool)
    step = DETAIL_STRIDES[detail]
    y, x = np.mgrid[:depth.shape[0]:step, :depth.shape[1]:step]
    z = depth[y, x]
    valid = region[y, x] & np.isfinite(z) & (z >= near) & (z <= far) & (z > 0)
    index = np.full(valid.shape, -1, np.int32)
    index[valid] = np.arange(valid.sum(), dtype=np.int32)
    yy, xx = y[valid], x[valid]
    camera = np.column_stack((xx, yy, np.ones(len(xx)))) @ np.linalg.inv(prepared["intrinsics"]).T
    camera *= depth[yy, xx, None]
    Twc = np.linalg.inv(prepared["extrinsics"])
    xyz = camera @ Twc[:3, :3].T + Twc[:3, 3]
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    triangles = np.concatenate((np.stack((a, c, b), axis=-1).reshape(-1, 3),
                                np.stack((b, c, d), axis=-1).reshape(-1, 3)))
    triangles = triangles[(triangles >= 0).all(axis=1)]
    depths = depth[yy, xx]
    if len(triangles):
        triangles = triangles[np.ptp(depths[triangles], axis=1) <= .02]
    return dict(xyz=xyz.round(5).tolist(), colors=prepared["video"][yy, xx].tolist(),
                depth_m=depths.round(5).tolist(), triangles=triangles.ravel().tolist(),
                region=(mask[yy, xx].astype(np.uint8) + 2*breast_area[yy, xx].astype(np.uint8)).tolist(),
                points=int(len(xyz)), faces=int(len(triangles)), scope=scope, detail=detail)


def add_camera_gaps(views: list[dict]) -> None:
    """Nearest sampled surface distance, leaving non-overlapping areas grey."""
    for i, view in enumerate(views):
        source = np.asarray(view["xyz"], dtype=np.float64).reshape(-1, 3)
        others = [np.asarray(other["xyz"], dtype=np.float64).reshape(-1, 3)
                  for j, other in enumerate(views) if i != j and other["xyz"]]
        gap = np.full(len(source), -1., np.float64)
        if len(source) and others:
            distance, _ = cKDTree(np.concatenate(others)).query(source, distance_upper_bound=.03, workers=1)
            supported = np.isfinite(distance)
            gap[supported] = distance[supported] * 1000
        view["gap_mm"] = gap.round(2).tolist()
        matches = gap >= 0
        view["overlap"] = dict(matched_percent=round(100*float(matches.mean()), 1) if len(gap) else 0,
                               median_gap_mm=round(float(np.median(gap[matches])), 2) if matches.any() else None)
