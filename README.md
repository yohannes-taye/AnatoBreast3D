# BreastSegTrack

## Markerless Patient-Specific Breast Surface Mapping for Deformation-Aware Mixed-Reality Surgical Guidance

**Authors:** Luyao Xia, Zirui Xiao, Huidong Bai, Mark Billinghurst

**Affiliation:** Auckland Bioengineering Institute, University of Auckland,
Auckland, New Zealand

BreastSegTrack is an academic and engineering prototype for markerless,
patient-specific breast surface mapping from calibrated multi-view RGB-D cameras.
The system combines breast-specific RGB segmentation, nipple localisation,
depth-to-color aligned Orbbec capture, Charuco-based multi-camera calibration,
masked point-cloud fusion, live browser visualization, and an interactive
active-learning loop for improving segmentation quality during use.

The long-term target is deformation-aware mixed-reality surgical guidance:
instead of relying on a single static overlay, the system reconstructs and tracks
the visible breast surface so that patient-specific 3D coordinates can be updated
as the breast is positioned or manipulated.

This repository is a research prototype for technical evaluation, algorithm
development, and non-diagnostic visualization. It is not a clinical decision
system and should not be used for autonomous diagnosis or surgical action.

## Contributions

1. **Breast segmentation and nipple recognition.** BreastSegTrack performs
   breast-specific RGB segmentation and nipple landmark detection from each camera
   view, providing both the visible breast boundary and anatomical anchor points
   needed for downstream tracking.
2. **Nipple-guided multi-view RGB mask fusion.** Nipple detections are used as
   cross-view anatomical anchors to associate and fuse breast masks from multiple
   RGB views, improving left-right consistency and reducing ambiguity when the
   visible breast boundary changes during manipulation.
3. **Mask-guided RGB-D breast surface reconstruction.** The fused RGB mask
   representation is combined with depth-to-color aligned Orbbec depth maps, then
   back-projected through calibrated camera intrinsics and `T_world_cam`
   extrinsics to reconstruct a patient-specific 3D breast surface.
4. **Deformation-aware MR foundation.** The reconstructed surface and nipple
   anchors are designed to support downstream deformable registration and
   overlay-ready 3D coordinates for mixed-reality surgical guidance.

## Engineering Scope

The current repository implements the acquisition, perception, calibration,
RGB-D reconstruction, visualization, and active-learning parts of BreastSegTrack:

- three-camera Orbbec Femto Bolt RGB-D capture;
- SDK depth-to-color alignment for per-camera RGB-D consistency;
- multitask breast segmentation and nipple detection;
- Charuco-based multi-camera extrinsic calibration;
- mask-filtered depth back-projection and multi-view point-cloud fusion;
- browser-based live visualization of RGB, depth, masks, nipple detections, and
  fused 3D surface;
- interactive active-learning annotation and fine-tuning infrastructure.

Downstream surgical registration, MR headset integration, and formal clinical
workflow validation are treated as research extensions on top of this engineering
base.

## System Overview

```text
3x Orbbec Femto Bolt cameras
  -> SDK RGB-D capture with depth-to-color alignment
  -> RGB breast segmentation and nipple detection
  -> mask-filtered depth back-projection
  -> Charuco T_world_cam transformation
  -> multi-view point-cloud fusion and optional surface cleanup
  -> patient-specific breast surface map
  -> deformation-aware tracking / MR overlay coordinates
  -> live browser visualization and active-learning annotation
```

The mathematical back-projection for each valid masked depth pixel is:

```text
z = depth_raw * depth_scale_m
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_world = T_world_cam @ [x, y, z, 1]^T
```

After SDK depth-to-color alignment, the RGB image, segmentation mask, and depth
map must have the same height and width. This is the key contract that allows
`mask[v, u]` to select the correct depth sample.

## Repository Layout

```text
configs/
  breastseg/                 RGB perception model configs
  recon3d/                   camera rig, fusion, and extrinsic configs
recon3d/
  sensors/                   Orbbec, replay, mock, and multi-camera rig code
  calibration/               Charuco detection and extrinsic estimation
  perception/                BreastSeg inference adapter
  geometry/                  masked back-projection, fusion, nipple 3D logic
  smoothing/                 temporal filters
  pipeline.py                reference RGB-D reconstruction pipeline
tools/
  live_recon.py              formal live three-camera viewer pipeline
  serve_recon3d_viewer.py    browser viewer and active-learning API server
  inspect_charuco_board.py   board/camera inspection utility
  train_breastseg.py         segmentation and detection training entry point
viewer/
  live.html                  realtime browser dashboard
  active_learning.js/css     annotation UI mounted inside the live viewer
active_learning/
  dataset/                   saved live labels in BreastSeg training format
  configs/                   short fine-tuning config
  training.py                API-triggered fine-tuning launcher
tests/                       unit tests for geometry, sensors, pipeline, CLI
```

## Hardware Setup

The current validated rig uses three Orbbec Femto Bolt RGB-D cameras:

| Camera ID | Serial number |
| --- | --- |
| `cam0` | `CL8T754004D` |
| `cam1` | `CL8T7540197` |
| `cam2` | `CL8T75400XA` |

Depth-to-color alignment is handled inside `recon3d/sensors/femto_bolt.py` using
the Orbbec SDK. This alignment is per camera and uses the factory calibration
between the color and depth sensors. Moving the external cameras does not change
the internal D2C calibration, but it does invalidate the multi-camera extrinsics.

For dynamic multi-view reconstruction, hardware sync is strongly preferred. The
project supports primary/secondary Orbbec sync roles, but the physical wiring and
stored device roles must match the camera order used by the live script and YAML
configuration.

## Environment

Use the existing conda environment when available:

```powershell
conda activate breastseg
pip install -r requirements.txt
```

On the Orbbec workstation, install the hardware bindings after the base
environment is clean:

```powershell
pip install -r requirements-hardware.txt
```

Notes:

- `requirements.txt` pins `numpy==1.26.4`, `open3d==0.18.0`, PyTorch CUDA 11.8,
  and `opencv-contrib-python` because Charuco calibration requires `cv2.aruco`.
- `requirements-hardware.txt` adds `pyorbbecsdk`, `rerun-sdk==0.23.1`, and a
  NumPy-1.x-compatible `pyarrow`.
- Keep only one OpenCV wheel in the environment. If `opencv-python` is already
  installed, replace it with `opencv-contrib-python`.

## Live Operation

The browser dashboard is a separate viewer process. Use the root startup script
when you want the live pipeline and the single visualization entry together:

```powershell
cd E:\OneDrive\CodeSpace\TDSegmentation\Breast-Segmentation\AntatoBreast3D
powershell -ExecutionPolicy Bypass -File .\start_recon3d_live.ps1 -OpenBrowser
```

The dashboard is served at:

```text
http://127.0.0.1:8765/viewer/live.html
```

The script opens one PowerShell window for the browser viewer/API server and one
for realtime capture, segmentation, fusion, and Rerun streaming. Logs are written
under `runs\recon3d_live\logs\`.

For formal headless reconstruction without the browser dashboard, run the
pipeline directly:

```powershell
cd E:\OneDrive\CodeSpace\TDSegmentation\Breast-Segmentation\AntatoBreast3D
conda activate breastseg
python -m recon3d.cli run --config configs\recon3d\default.yaml --out runs\recon3d_live
```

For manual debugging, start the same two processes in separate PowerShell or
conda terminals.

Terminal 1: browser viewer and active-learning API server.

```powershell
cd E:\OneDrive\CodeSpace\TDSegmentation\Breast-Segmentation\AntatoBreast3D
conda activate breastseg
python tools\serve_recon3d_viewer.py --host 127.0.0.1 --port 8765 --open
```

Terminal 2: realtime camera capture, segmentation, fusion, and live manifest
writer.

```powershell
cd E:\OneDrive\CodeSpace\TDSegmentation\Breast-Segmentation\AntatoBreast3D
conda activate breastseg
python tools\live_recon.py --out runs\recon3d_live --config runs\_tmp_live.yaml --color-size 1280 720 --reuse-existing-sync-config --rerun-web-port 9090 --rerun-grpc-port 9876
```

Open the live dashboard:

```text
http://127.0.0.1:8765/viewer/live.html
```

The viewer server is responsible for the web page and Active Learning API. The
camera pipeline is responsible for Orbbec capture, reconstruction, Rerun
streaming, and `live_manifest.json`. If the UI shows `server offline`, restart
`tools\serve_recon3d_viewer.py`; that message does not necessarily mean the
cameras stopped.

Manual live-view checks:

- Fresh `live_manifest.json` with `rerun_url`: the Rerun iframe stays visible.
- Stale `live_manifest.json` with `rerun_url`: the iframe stays visible and the
  header shows a stale frame status.
- Manifest missing `rerun_url`: the page shows `Old live manifest` once without
  flickering, including the base path it is reading.
- Manifest with `rerun_error`: the page shows the Rerun error and points to
  `runs\recon3d_live\logs\`.

## Calibration

The checked-in calibration uses a 5 x 5 Charuco board with 50 mm square pitch,
40 mm ArUco markers, and `DICT_7X7_50`. The current report is stored at
`configs/recon3d/extrinsics_report.json` and records:

```text
mean reprojection error: 0.251 px
max reprojection error:  0.783 px
detected corners:        16 per camera
world frame:             Charuco board
```

Recalibrate whenever any camera, mount, tripod, or board-relative rig geometry is
moved. The following command matches the current physical board dimensions:

```powershell
python -m recon3d.cli calibrate `
  --config configs\recon3d\default.yaml `
  --out configs\recon3d\extrinsics.json `
  --board charuco_5x5_50mm `
  --dictionary DICT_7X7_50 `
  --square-length-m 0.050 `
  --marker-length-m 0.040 `
  --world board `
  --captures 12 `
  --settle-frames 3 `
  --max-drift-ms 1000 `
  --simultaneous `
  --no-sync-config `
  --align-mode sw `
  --color-size 1280 720 `
  --cameras cam1 cam2 cam0 `
  --serials CL8T7540197 CL8T75400XA CL8T754004D `
  --sync-modes secondary secondary primary
```

After calibration, inspect `configs/recon3d/extrinsics_report.json`. For this
setup, all three cameras should detect the board and report sub-pixel reprojection
error. Low reprojection error with too few corners is not a reliable calibration.

## Outputs

Formal reconstruction outputs are written under the selected run directory:

```text
runs/recon3d_live/
  fused_breast.ply       fused world-frame point cloud, metres
  live_manifest.json     frame index, FPS, paths, perception summaries, timings
  nipple_world.json      3D nipple output for the reference pipeline path
  live/
    cam0_rgb.jpg         latest RGB frame for annotation
    cam0_mask.png        latest predicted breast mask
    cam1_rgb.jpg
    cam1_mask.png
    cam2_rgb.jpg
    cam2_mask.png
```

`tools/live_recon.py` also streams RGB, depth, segmentation overlays, 2D nipple
boxes, timing, and the fused point cloud to the embedded Rerun viewer used by
`viewer/live.html`.

## Active Learning

The Active Learning panel in the live dashboard is designed for fast correction
of weak segmentation results:

1. Select `cam0`, `cam1`, or `cam2`.
2. Click `Annotate`.
3. Use `Prediction` to seed the editable mask from `runs/recon3d_live/live/cam*_mask.png`.
4. Add or erase mask regions and edit nipple boxes.
5. Click `Save Label`.
6. Launch fine-tuning when enough corrected samples have been saved.

Saved labels are written to:

```text
active_learning/dataset/
  images/*.png
  sem_masks/*.png
  annotations/coco_det.json
  splits/train.txt
  splits/val.txt
  manifest.jsonl
```

Fine-tuning is launched by the viewer API through `active_learning/training.py`.
The generated command reuses the standard training entry point:

```powershell
python tools\train_breastseg.py --config active_learning\configs\breastseg_active.py --data-root active_learning\dataset --init-from checkpoint\breastseg.pth --work-dir active_learning\checkpoints\<run_id>
```

The best checkpoint for a fine-tuning run is saved as:

```text
active_learning/checkpoints/<run_id>/best.pth
```

## Research and Engineering Notes

- The quality of 3D reconstruction is bounded by the RGB mask quality. Boundary
  errors in segmentation become depth boundary artifacts and can appear as point
  cloud drag or flying edges.
- Nipple detections are used as anatomical anchors for cross-view consistency and
  downstream deformable registration, but they must be visually verified when the
  RGB detector is uncertain or partially occluded.
- Erosion, depth-gradient filtering, voxel downsampling, statistical outlier
  removal, Poisson meshing, cropping, and Laplacian smoothing are engineering
  post-processing options. They can improve visual smoothness, but they should
  not be treated as a replacement for correct segmentation, calibration, and sync.
- If cameras are moved, update extrinsics. If only exposure, focus, or software
  segmentation thresholds are changed, extrinsics do not need to be recomputed.
- If the live FPS is low, profile `timings_ms` in `live_manifest.json` first.
  Perception, point-cloud fusion, mesh generation, and disk writes are the usual
  bottlenecks.
- The current live viewer prioritizes RGB/mask/2D nipple visualization, fused
  cloud inspection, and annotation. The reference `Recon3DPipeline` contains the
  metric 3D nipple fusion path used by the broader BreastSegTrack research
  pipeline.
- The reported `28 FPS` result is a research deployment target and depends on the
  exact model, hardware, resolution, visualization mode, and post-processing
  configuration. When optimizing this engineering repository, use
  `live_manifest.json` timings as the source of truth.

## Validation

Run the automated checks:

```powershell
conda activate breastseg
python -m pytest
ruff check .
mypy recon3d --strict
```

Useful runtime inspections:

```powershell
python tools\inspect_charuco_board.py
python -m recon3d.cli run --config configs\recon3d\default.yaml --out runs\formal_live --once
```

## Documentation

Additional project notes:

- `docs/QUICKSTART.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/DATA_LAYOUT.md`
- `active_learning/README.md`

## Citation

If this system is used in academic work, cite the project as:

```text
Xia, L., Xiao, Z., Bai, H., and Billinghurst, M. BreastSegTrack: Markerless
Patient-Specific Breast Surface Mapping for Deformation-Aware Mixed-Reality
Surgical Guidance. Auckland Bioengineering Institute, University of Auckland.
```

For reproducibility, report the hardware calibration, Charuco board geometry,
segmentation checkpoint, depth alignment mode, camera resolution, synchronization
mode, and post-processing settings used for each experiment.

## License

This project is released under the MIT License. See `LICENSE`.
