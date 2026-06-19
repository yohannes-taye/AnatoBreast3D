from __future__ import annotations

from recon3d.cli import load_recon_config


def test_load_recon_config_reads_full_nipple_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output_dir: runs/test
extrinsics_path: configs/recon3d/extrinsics.json
nipple3d:
  dbscan_eps_m: 0.007
  min_cluster_points: 12
  covariance_jitter_m2: 2.0e-8
  left_right_axis_world: [0.0, 1.0, 0.0]
  mahalanobis_threshold: 2.5
cloud_cleanup:
  mask_erode_px: 3
  flying_depth_edge_threshold_m: 0.02
  min_depth_m: 0.3
  max_depth_m: 1.1
surface:
  mesh_enabled: true
  poisson_depth: 7
  density_trim_quantile: 0.2
  sample_points: 4000
  normal_radius_m: 0.02
  normal_max_nn: 12
  min_points: 20
""",
        encoding="utf-8",
    )

    config, extrinsics_path = load_recon_config(config_path)

    assert str(extrinsics_path).replace("\\", "/") == "configs/recon3d/extrinsics.json"
    assert config.nipple3d.dbscan_eps_m == 0.007
    assert config.nipple3d.min_cluster_points == 12
    assert config.nipple3d.covariance_jitter_m2 == 2.0e-8
    assert config.nipple3d.left_right_axis_world.tolist() == [0.0, 1.0, 0.0]
    assert config.nipple3d.mahalanobis_threshold == 2.5
    assert config.cloud_cleanup.mask_erode_px == 3
    assert config.cloud_cleanup.flying_depth_edge_threshold_m == 0.02
    assert config.cloud_cleanup.min_depth_m == 0.3
    assert config.cloud_cleanup.max_depth_m == 1.1
    assert config.surface.mesh_enabled is True
    assert config.surface.poisson_depth == 7
    assert config.surface.density_trim_quantile == 0.2
    assert config.surface.sample_points == 4000
    assert config.surface.normal_radius_m == 0.02
    assert config.surface.normal_max_nn == 12
    assert config.surface.min_points == 20
