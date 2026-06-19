from __future__ import annotations

import numpy as np

from recon3d.config import CloudCleanupConfig, DepthConfig
from recon3d.geometry.cloud_cleanup import cleaned_breast_mask, flying_depth_edges


def test_flying_depth_edges_rejects_discontinuities_without_wrapping() -> None:
    depth_m = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    valid = np.ones_like(depth_m, dtype=bool)

    edges = flying_depth_edges(depth_m, valid, threshold_m=0.2)

    assert bool(edges[1, 1])
    assert not bool(edges[0, 0])


def test_cleaned_breast_mask_applies_depth_range_and_edge_rejection() -> None:
    depth = np.array(
        [
            [1000, 1000, 1000, 1000, 1000],
            [1000, 1000, 1000, 1000, 1000],
            [1000, 1000, 1500, 1000, 1000],
            [1000, 1000, 1000, 1000, 1000],
            [1000, 1000, 1000, 1000, 2000],
        ],
        dtype=np.uint16,
    )
    mask = np.ones_like(depth, dtype=bool)
    depth_config = DepthConfig(depth_scale_m=0.001, min_depth_m=0.25, max_depth_m=3.0)
    cleanup_config = CloudCleanupConfig(
        mask_erode_px=0,
        flying_depth_edge_threshold_m=0.2,
        min_depth_m=0.25,
        max_depth_m=1.2,
    )

    cleaned = cleaned_breast_mask(depth, mask, depth_config, cleanup_config)

    assert cleaned[0, 0]
    assert not cleaned[2, 2]
    assert not cleaned[4, 4]
