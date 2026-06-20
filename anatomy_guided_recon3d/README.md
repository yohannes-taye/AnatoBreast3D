# Anatomy-Guided Recon3D

This package is a full copied branch of `recon3d` with one intended algorithmic
change: multi-view breast surface reconstruction is now guided by nipple
landmarks.

The original `recon3d` package is not modified. This package keeps the same
camera, calibration, perception, replay, and CLI structure, but changes the
runtime reconstruction pipeline:

```text
synchronized RGB-D frames
  -> breast mask and nipple prediction per camera
  -> 3D nipple observations in the calibrated world frame
  -> fused left/right nipple anatomical anchors
  -> guided per-view cloud decisions and weights
  -> cleaned breast clouds using the same cleanup as recon3d
  -> weighted point-cloud fusion and optional surface smoothing
```

Run one real three-camera frame:

```powershell
conda activate breastseg
python -m anatomy_guided_recon3d.cli run `
  --config configs\recon3d\default.yaml `
  --out runs\anatomy_guided_recon3d `
  --align-mode sw `
  --color-size 1280 720 `
  --no-sync-config `
  --once
```

Useful outputs:

```text
runs/anatomy_guided_recon3d/
  fused_breast.ply
  nipple_world.json
  nipple_guided_fusion.json
  run_summary.json
```

`nipple_guided_fusion.json` is the key diagnostic file. It reports each camera's
accepted/rejected state, point count, nipple consistency error, and fusion
weight.

Optional YAML section:

```yaml
nipple_guided_fusion:
  enabled: true
  nipple_gate_m: 0.05
  reject_inconsistent_views: true
  missing_nipple_weight: 0.55
  min_consistent_weight: 0.35
  min_accepted_views: 1
  random_seed: 11
```
