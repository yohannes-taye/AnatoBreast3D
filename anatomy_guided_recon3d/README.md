# Nipple-Guided Recon3D

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
  -> per-view mask/RGB-D consistency scoring
  -> optional inconsistent-view rejection
  -> weighted mask-cleaned point-cloud fusion
```

Run one real three-camera frame:

```powershell
conda activate breastseg
python -m nipple_guided_recon3D run `
  --config configs\recon3d\default.yaml `
  --out runs\nipple_guided_recon3D `
  --align-mode sw `
  --color-size 1280 720 `
  --no-sync-config `
  --once
```

Useful outputs:

```text
runs/nipple_guided_recon3D/
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
  mask_erode_px: 4
  depth_edge_threshold_m: 0.015
  depth_min_m: 0.25
  depth_max_m: 1.2
  nipple_gate_m: 0.05
  reject_inconsistent_views: true
  missing_nipple_weight: 0.55
  min_consistent_weight: 0.35
  min_accepted_views: 1
```
