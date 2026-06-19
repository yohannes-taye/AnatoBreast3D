# Quickstart: Orbbec RGB-D Reconstruction

This repository is now structured around the Orbbec Femto Bolt RGB-D path. The
segmentation and nipple detection models remain the existing implementation.

## 1. Environment

```bash
conda activate breastseg
python --version
pip install -r requirements.txt
```

Do not reinstall or upgrade `torch`, `torchvision`, CUDA, `mmcv`, or existing
segmentation dependencies unless you are intentionally rebuilding the perception
environment.

`requirements.txt` is the base install file for segmentation, replay/mock RGB-D
reconstruction, calibration math, tests, linting, and typing. It uses
`opencv-contrib-python==4.10.0.84` so Charuco calibration has `cv2.aruco`. If an
existing environment already contains `opencv-python`, remove that OpenCV wheel
before installing the requirements to avoid mixed OpenCV packages.

Optional hardware dependency:

```bash
pip install -r requirements-hardware.txt
```

Only install hardware requirements on the machine connected to the Orbbec
cameras. Replay mode does not need `pyorbbecsdk`. If the package is not
available from pip on the target machine, install the SDK from Orbbec's supported
distribution channel and keep the replay/mock path unchanged.

## 2. Configure Reconstruction

Use the template:

```bash
configs/recon3d/default.yaml
```

Important fields:

```yaml
extrinsics_path: configs/recon3d/extrinsics.json
perception:
  model_config_path: configs/breastseg/breastseg_b2_inplace.py
  checkpoint_path: checkpoint/breastseg.pth
fusion:
  voxel_size_m: 0.002
nipple3d:
  dbscan_eps_m: 0.005
  min_cluster_points: 30
```

`extrinsics.json` must contain `T_world_cam` for each camera. Do not use fake
extrinsics for metric evaluation.

## 3. Calibrate

```bash
python -m recon3d.cli calibrate \
  --board charuco_5x7_40mm \
  --out configs/recon3d/extrinsics.json
```

Calibration uses OpenCV ArUco/Charuco APIs and exports camera-to-world
transforms. Mean reprojection error should be below 1 px; max reprojection error
should be below 2 px. Online capture still depends on the final `pyorbbecsdk`
binding on the target machine.

## 4. Record

```bash
python -m recon3d.cli record --config configs/recon3d/default.yaml --out data/Replay/orbbec_bag_test --duration-s 30 --color-size 1280 720 --color-fps 30
```

The target recording layout is one SDK bag session:

```text
session.json
cam0.bag
cam1.bag
cam2.bag
```

Recording waits for a synchronized T0 before bag writing starts.

## 5. Replay

```powershell
python -m recon3d.cli replay --config configs\recon3d\default.yaml --bag-session data\Replay\orbbec_bag_720p_100s --out runs\orbbec_bag_720p_100s_replay --visualize --probe-frames 180 --wait-timeout-ms 5000
```

Replay runs the same reconstruction pipeline without physical cameras. This is
the preferred first test path for debugging model, geometry, and IO behavior.
The browser viewer also has a Replay panel that lists SDK bag sessions from
`data\Replay\*\session.json` and starts the same CLI replay command for the
selected session.

## 6. Real-Time Run

Use two terminals for the normal live workflow. The viewer/API server serves
`viewer/live.html`; the formal CLI run publishes `live_manifest.json`, preview
images, and the embedded Rerun stream.

```powershell
python tools\serve_recon3d_viewer.py --host 127.0.0.1 --port 8765 --open
```

```powershell
python -m recon3d.cli run --config configs\recon3d\default.yaml --out runs\live --visualize
```

For formal headless reconstruction, run the pipeline directly:

```bash
python -m recon3d.cli run \
  --config configs/recon3d/default.yaml \
  --out runs/live \
  --cameras cam0 cam1 cam2
```

The browser dashboard is served separately at
`http://127.0.0.1:8765/viewer/live.html`.

Expected hardware mode:

```text
Depth: NFOV 640x576 @ 30 fps
RGB:   1280x720 or 1920x1080 @ 30 fps
Depth unit: millimetres
Timestamp drift target: < 5 ms
```

## 7. Outputs

```text
runs/<session>/fused_breast.ply
runs/<session>/nipple_world.json
runs/<session>/run_summary.json
```

`nipple_world.json` contains left/right 3D world coordinates, covariance,
confidence, used cameras, and rejected cameras.

## 8. Validate

```bash
conda run -n breastseg python -m pytest
conda run -n breastseg ruff check .
conda run -n breastseg mypy recon3d --strict
```
