# Medical-breast-segmentation-and-reconstruction

Orbbec Femto Bolt RGB-D multi-camera reconstruction for real-time breast surface
fusion and 3D nipple localisation. The existing `BreastSegFormer` and
`NippleFCOSHead` perception models are kept unchanged; this version replaces the
old RGB-only reconstruction path with calibrated RGB-D fusion.

## Environment

Use the existing conda environment:

```bash
conda activate segformer
pip install -r requirements.txt
```

Do not create a new environment or upgrade core perception dependencies unless
there is a specific compatibility reason.
`requirements.txt` installs `opencv-contrib-python` because Charuco calibration
needs `cv2.aruco`. In an existing environment that already has `opencv-python`,
replace the OpenCV wheel instead of keeping both packages installed.

On the Orbbec hardware workstation, install the optional camera SDK after the
base requirements:

```bash
pip install -r requirements-hardware.txt
```

If `pyorbbecsdk` is not available from pip on that machine, keep using replay or
mock mode and install the SDK from Orbbec's supported distribution channel.

## Pipeline

```text
3x aligned Orbbec RGB-D frames
  -> existing BreastSegFormer + NippleFCOSHead on RGB
  -> breast mask + nipple boxes
  -> masked depth back-projection
  -> T_world_cam point cloud fusion
  -> fused_breast.ply + nipple_world.json
```

Real hardware access is isolated in `recon3d/sensors/femto_bolt.py`. Replay and
mock RGB-D execution work without physical cameras.

## Workflows

```bash
python -m recon3d.cli calibrate --board charuco_5x7_40mm --out configs/recon3d/extrinsics.json
python -m recon3d.cli record --duration 30 --out data/raw/session_001
python -m recon3d.cli replay --config configs/recon3d/default.yaml --bag data/raw/session_001 --out runs/session_001_replay
python -m recon3d.cli run --config configs/recon3d/default.yaml --out runs/live
```

Current hardware capture and online calibration commands require final
`pyorbbecsdk` binding work on the target machine. The core pipeline, replay,
geometry, fusion, IO, and tests do not require cameras.

## Coordinate Convention

```text
World frame: right-handed, selected by Charuco board or reference camera.

          +Y_world
             ^
             |
             o----> +X_world
            /
      +Z_world

Camera frame: OpenCV RGB-D convention, +Z_cam points forward from the camera.

T_world_cam maps camera-frame homogeneous points into world coordinates:

    p_world = T_world_cam @ [x_cam, y_cam, z_cam, 1]^T
```

## Outputs

```text
fused_breast.ply   world-frame fused breast point cloud, metres
nipple_world.json  left/right mean_world_m, covariance_world_m2, confidence,
                   used_cameras, rejected_cameras
run_summary.json   timings and measurable metrics; replay marks hardware-only
                   metrics as not_measured
```

## Validation

```bash
conda run -n segformer python -m pytest
conda run -n segformer ruff check .
conda run -n segformer mypy recon3d --strict
```

More details:

- `docs/QUICKSTART.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/DATA_LAYOUT.md`
