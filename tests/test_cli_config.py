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
