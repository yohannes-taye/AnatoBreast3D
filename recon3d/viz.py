from __future__ import annotations

import copy
import logging

import numpy as np
import open3d as o3d

from recon3d.config import FusedNipple


LOGGER = logging.getLogger(__name__)


class Open3DRGBDViewer:
    """Optional non-headless Open3D viewer for fused RGB-D reconstruction."""

    def __init__(self, enabled: bool = False) -> None:
        """Create a viewer.

        Input shape/unit/frame: enabled controls whether an Open3D window is created.
        Return type: ``Open3DRGBDViewer``.
        """
        self.enabled = enabled
        self._vis: o3d.visualization.Visualizer | None = None
        self._cloud_added = False

    def start(self) -> None:
        """Start visualization if enabled.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        if not self.enabled or self._vis is not None:
            return
        self._vis = o3d.visualization.Visualizer()
        created = self._vis.create_window(window_name="Recon3D RGB-D Fusion", width=1280, height=960)
        if not created:
            LOGGER.warning("Open3D visualization window could not be created; disabling viewer")
            self.enabled = False
            self._vis = None

    def update(self, cloud: o3d.geometry.PointCloud, nipples: dict[str, FusedNipple] | None = None) -> None:
        """Update visualization.

        Input shape/unit/frame: cloud and nipple points are world-frame metres.
        Return type: ``None``.
        """
        if not self.enabled:
            return
        self.start()
        if self._vis is None:
            return
        display = copy.deepcopy(cloud)
        if not self._cloud_added:
            self._vis.add_geometry(display)
            self._cloud_added = True
        else:
            self._vis.clear_geometries()
            self._vis.add_geometry(display)
        if nipples:
            for side, nipple in nipples.items():
                if nipple.confidence <= 0.0:
                    continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
                sphere.translate(nipple.mean_world_m)
                sphere.paint_uniform_color([1.0, 0.0, 0.0] if side == "left" else [0.0, 1.0, 0.0])
                self._vis.add_geometry(sphere)
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        """Close visualization.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
            self._cloud_added = False


def make_nipple_marker_points(nipples: dict[str, FusedNipple]) -> np.ndarray:
    """Return nipple marker coordinates for external visualization.

    Input shape/unit/frame: nipple estimates are world-frame metres.
    Return type: Nx3 ndarray in world metres.
    """
    points = [nipple.mean_world_m for nipple in nipples.values() if nipple.confidence > 0.0]
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)
