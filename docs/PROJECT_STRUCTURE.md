# Project Structure

This project contains two connected parts:

1. A 2D perception stack for breast segmentation and nipple detection.
2. A post-perception RGB-D reconstruction stack for multi-camera 3D fusion and 3D nipple localisation.

The published project name is
`Medical-breast-segmentation-and-reconstruction`, and the pipeline described
here follows the current code layout.

## Pipeline Overview

```text
LabelMe / processed breast images
  -> data_preprocess/convert_breast_labelme.py or data_preprocess/prepare_data.py
  -> data/breastseg/processed_merged
  -> data_preprocess.BreastCocoDataset
  -> tools/train_breastseg.py
  -> nets.segmentors.MammaSegNet
  -> work_dirs/breastseg_b2/best.pth
  -> tools/breastseg_infer.py or recon3d.perception.BreastPerceptionAdapter
  -> RGB breast masks + nipple boxes
  -> recon3d.geometry.backproject
  -> recon3d.geometry.nipple3d
  -> recon3d.geometry.fusion
  -> fused_breast.ply + nipple_world.json + run_summary.json
```

The important boundary is that `nets/` remains a 2D RGB perception model. The
3D package in `recon3d/` reuses that model through an adapter and then combines
its masks/boxes with aligned depth, intrinsics, and `T_world_cam` extrinsics.

## Top-Level Layout

```text
Medical-breast-segmentation-and-reconstruction/
|-- checkpoint/                 # optional local weights; ignored except for README instructions
|-- configs/                    # 2D model and 3D reconstruction configuration files
|-- data/                       # local datasets and replay/recorded data roots
|-- data_preprocess/            # dataset prep + PyTorch dataset/collate/augmentation
|-- docs/                       # project documentation and operating notes
|-- logs/                       # local runtime logs grouped by source
|-- nets/                       # MammaSegNet, MiT backbone, heads, necks, losses/utilities
|-- recon3d/                    # Orbbec RGB-D reconstruction package
|-- tests/                      # unit, smoke, and single-view demo assets
|-- tools/                      # command-line data, training, validation, inference, export tools
|-- utils/                      # legacy helper code used by training/inference
|-- viewer/                     # browser UI, Rerun iframe shell, active-learning frontend
|-- work_dirs/                  # experiment checkpoints, best model, TensorBoard event files
|-- requirements.txt            # base software dependencies
`-- requirements-hardware.txt   # optional Orbbec hardware dependencies
```

## Folder Roles In The Pipeline

| Folder | Pipeline role | Main code/data | Notes |
| --- | --- | --- | --- |
| `.claude/` | Development metadata only. | Local assistant config. | No effect on data processing, model training, or reconstruction. |
| `.vscode/` | Developer/editor convenience. | Workspace settings. | Safe to ignore for runtime behavior. |
| `checkpoint/` | Deployed/named perception weights, separate from raw training output. | `breastseg.pth`, `README.md`. | `configs/recon3d/default.yaml` loads `checkpoint/breastseg.pth`. Training still writes `work_dirs/breastseg_b2/best.pth`; copy/rename it here to deploy. Future models can use new subfolders. |
| `configs/` | Configuration source for both pipeline halves. | `breastseg/breastseg_b2_inplace.py`, `recon3d/default.yaml`. | `breastseg` controls dataset paths, model size, loss weights, optimizer, sampling repeats, and inference thresholds. `recon3d` controls checkpoint path, device, fusion voxel size, nipple 3D thresholds, output path, and sync drift. |
| `data/` | Local data root consumed by data preparation, loaders, and offline replay. | `data/breastseg/raw_combined`, `data/Replay`, `dataset3_converted`, `processed_merged`. | `processed_merged` is the default training root: `images/`, `sem_masks/`, `annotations/coco_det.json`, `splits/train.txt`, `splits/val.txt`. SDK bag sessions used by the viewer Replay panel live under `data/Replay/<session>/session.json`. |
| `data_preprocess/` | Offline dataset prep plus model-ready batch loading. | `breast_coco.py`, `aug_ops.py`, `convert_breast_labelme.py`, `prepare_data.py`, `regenerate_splits.py`, `augment_dataset.py`. | `BreastCocoDataset` reads images, segmentation masks, COCO nipple boxes, split files, hard-sample repeats, applies augmentation + letterbox resize, and returns `images`, `gt_seg`, `gt_boxes`, validity masks, and metadata. `aug_ops.py` holds the geometric augmentation primitives shared by on-the-fly (`breast_coco.py`) and offline (`augment_dataset.py`) augmentation; the CLI scripts convert, split, and augment datasets. |
| `docs/` | Human-facing project knowledge base. | `QUICKSTART.md`, `DATA_LAYOUT.md`, `PROJECT_STRUCTURE.md`. | Documents environment setup, data layout, pipeline commands, coordinate conventions, and folder responsibilities. |
| `logs/` | Local runtime logs grouped by source. | `training/`, `orbbec_sdk/`, `README.MD`. | Training writes to `logs/training/`. Orbbec SDK logs should be kept under `logs/orbbec_sdk/` when collected manually; the SDK may still auto-create a capitalized `Log/` folder when running from the project root. Generated log files are ignored by git. |
| `nets/` | Core 2D neural network implementation. | `backbone.py`, `segmentors/`, `decode_heads/`, `dense_heads/`, `necks/`, `utils/`. | Defines the MiT backbone, MammaSeg segmentation head, LightFPN, FCOS-style nipple detector, Gaussian prior builder, and multi-task `MammaSegNet`. |
| `recon3d/` | 3D reconstruction and localisation stack. | `pipeline.py`, `cli.py`, `config.py`, plus subpackages. | Takes synchronized aligned RGB-D frames, runs the existing 2D model, cleans masks/depth, back-projects masked depth, estimates/fuses nipple observations, fuses point clouds, optionally applies surface smoothing, publishes live viewer artifacts, and writes reconstruction outputs. |
| `tests/` | Regression protection and demo assets. | `test_pipeline.py`, `test_backproject.py`, `test_fusion.py`, `test_nipple3d.py`, `test_sensors.py`, `singleview/`. | Synthetic tests cover 3D math, replay cameras, config loading, perception pipeline integration, and single-view image/video inference helpers. Hardware is not required for the current tests. |
| `tools/` | Operational command-line scripts. | `run_recon3d_live.py`, `serve_recon3d_viewer.py`, `train_breastseg.py`, `breastseg_infer.py`, `validate.py`, `export_onnx.py`, `model_summary.py`. | Thin launchers plus the 2D model toolchain; offline dataset prep lives in `data_preprocess/`. `serve_recon3d_viewer.py` serves the browser UI and can launch SDK bag replay through `recon3d.cli replay`; formal 3D runtime behavior belongs in `recon3d.cli` and `recon3d.pipeline`. |
| `utils/` | Legacy helper utilities. | `utils.py`, `utils_metrics.py`, `callbacks.py`. | Current multi-task training still uses helpers such as `seed_everything`, `cvtColor`, `resize_image`, and `preprocess_input`. |
| `viewer/` | Browser visualization frontend. | `live.html`, `active_learning.js`, `active_learning.css`. | `live.html` is the single visualization page for live and SDK bag replay. It embeds the Rerun stream from `live_manifest.json`, exposes the Replay panel, and hosts the active-learning UI. |
| `work_dirs/` | Experiment output root. | `breastseg_b2/best.pth`, `epoch_*.pth`, `tensorboard/`. | Training checkpoints and TensorBoard event files live here. Perception loads the deployed copy at `checkpoint/breastseg.pth`. |
## `configs/`

```text
configs/
|-- breastseg/
|   `-- breastseg_b2_inplace.py
`-- recon3d/
    `-- default.yaml
```

`configs/breastseg/breastseg_b2_inplace.py` is a Python config dictionary. It
sets the B2 backbone, two-class segmentation, nipple detector thresholds, loss
weights, hard-sample repeat lists, optimizer, AMP, gradient accumulation, and
validation metric weighting.

`configs/recon3d/default.yaml` is the RGB-D runtime config. It binds the 3D
pipeline to the 2D perception checkpoint, extrinsics file, sync tolerance,
cloud cleanup, fusion parameters, optional surface smoothing, nipple 3D
clustering/fusion parameters, and output folder. The formal runtime entry point
is `python -m recon3d.cli`; the browser viewer is served separately from
`viewer/live.html`.

## `data/`

```text
data/
`-- breastseg/
    |-- raw_combined/
    |   |-- images/
    |   `-- labelme/
    |-- dataset3_converted/
    |   |-- images/
    |   |-- sem_masks/
    |   |-- annotations/
    |   `-- splits/
    `-- processed_merged/
        |-- images/
        |-- sem_masks/
        |-- annotations/coco_det.json
        |-- splits/train.txt
        |-- splits/val.txt
        `-- dataset3_files.txt
```

`raw_combined` is the annotation/source area. `convert_breast_labelme.py`
rasterizes LabelMe breast polygons into semantic masks and converts nipple
polygons into COCO-format detection boxes.

`dataset3_converted` is an already converted dataset subset. The training config
contains a dedicated `dataset3_repeat` hard-case list to increase sampling
frequency for these examples.

`processed_merged` is the active default training dataset. It is read directly by
`BreastCocoDataset`, so its folder names are part of the training contract.

## `nets/`

```text
nets/
|-- backbone.py
|-- mammaseg.py
|-- segmentors/
|   `-- breastseg_multitask.py
|-- decode_heads/
|   `-- mammaseg_prior_head.py
|-- dense_heads/
|   `-- nipple_fcos_head.py
|-- necks/
|   `-- light_fpn.py
`-- utils/
    `-- prior_builder.py
```

`segmentors/breastseg_multitask.py` is the central model assembly point. It
combines:

- MiT backbone from `backbone.py`.
- `LightFPN` for multi-scale detector features.
- `NippleFCOSHead` for single-class nipple box detection.
- `MammaSegPriorHead` for segmentation, optionally conditioned by Gaussian
  priors rendered from GT or predicted nipple boxes.
- Segmentation CE/Dice/boundary losses plus detection FCOS losses.

During training, GT nipple boxes build segmentation priors. During inference,
the detector predicts boxes first, the model converts those boxes into priors,
and the segmentation head uses them to refine the breast mask.

## `recon3d/`

```text
recon3d/
|-- config.py
|-- cli.py
|-- pipeline.py
|-- calibration/
|   |-- charuco.py
|   `-- io.py
|-- geometry/
|   |-- backproject.py
|   |-- fusion.py
|   `-- nipple3d.py
|-- perception/
|   `-- inference.py
|-- sensors/
|   |-- femto_bolt.py
|   `-- rig.py
`-- smoothing/
    `-- temporal.py
```

`recon3d/config.py` defines immutable data containers and runtime configs:
camera intrinsics, depth conversion, RGB-D frames, perception results,
extrinsics, nipple observations, fused nipples, and reconstruction outputs.

`recon3d/cli.py` provides the command entrypoints:

- `replay`: read saved RGB-D camera folders and run the full 3D pipeline.
- `run`: run from live Femto Bolt camera wrappers.
- `record`: placeholder for verified hardware recording integration.
- `calibrate`: placeholder for verified online capture, with calibration math
  implemented in `calibration/`.

`recon3d/pipeline.py` orchestrates one synchronized frame set:

```text
validate frames and timestamp drift
  -> BreastPerceptionAdapter.predict_batch
  -> backproject_masked_depth for each camera
  -> estimate_nipple_observations
  -> fuse_nipple_observations
  -> PointCloudFusionBackend.fuse
  -> OneEuroFilter3D smoothing
  -> write fused_breast.ply, nipple_world.json, run_summary.json
```

Subpackages:

| Subfolder | Role |
| --- | --- |
| `calibration/` | Charuco detection, solvePnP pose estimation, reprojection error summaries, and JSON IO for intrinsics/extrinsics/depth configs. It uses the convention `p_world = T_world_cam @ p_cam_h`. |
| `geometry/` | Metric geometry operations: valid depth masking, depth back-projection to camera-frame point clouds, camera-to-world transform, point cloud merge/downsample/outlier removal, 3D nipple candidate extraction, covariance estimation, side assignment, outlier rejection, and final left/right fusion. |
| `perception/` | Adapter between `recon3d` and the existing 2D model. It letterboxes RGB frames with `tools.breastseg_infer` preprocessing, runs `MammaSegNet`, and remaps masks/boxes back to original aligned RGB/depth pixels. |
| `sensors/` | Camera abstraction layer. `ReplayRGBDCamera` and `MockRGBDCamera` are functional now; `FemtoBoltCamera` guards the unfinished `pyorbbecsdk` hardware integration. `MultiFemtoBoltRig` synchronizes camera reads by timestamp drift. |
| `smoothing/` | Low-latency temporal filters for 3D nipple coordinates, currently One Euro filtering in the main pipeline and an optional constant-velocity Kalman filter. |

All geometry outputs are in metres. Raw depth PNGs are treated as raw depth
units, normally millimetres, and converted using `DepthConfig.depth_scale_m`.

## `data_preprocess/`

```text
data_preprocess/
|-- breast_coco.py            # BreastCocoDataset + collate (runtime training data)
|-- aug_ops.py                # shared geometric augmentation primitives
|-- convert_breast_labelme.py # LabelMe -> COCO conversion
|-- prepare_data.py           # zip -> processed dataset
|-- regenerate_splits.py      # train/val split generation
|-- augment_dataset.py        # offline dataset augmentation
`-- __init__.py
```

`data_preprocess/` holds both offline dataset preparation (convert / split /
augment) and the runtime `BreastCocoDataset` used by training. `aug_ops.py`
provides the geometric augmentation primitives shared by `breast_coco.py`
(on-the-fly) and `augment_dataset.py` (offline), so the two paths stay in sync.

## `tools/`

```text
tools/
|-- train_breastseg.py
|-- validate.py
|-- breastseg_infer.py
|-- export_onnx.py
|-- model_summary.py
`-- __init__.py
```

Typical 2D workflow:

```text
convert_breast_labelme.py / prepare_data.py
  -> regenerate_splits.py if needed
  -> train_breastseg.py
  -> validate.py
  -> breastseg_infer.py
  -> export_onnx.py if deployment needs ONNX
```

`train_breastseg.py` is the main training entry. It loads config, builds
`BreastCocoDataset`, builds `MammaSegNet`, trains with AdamW/cosine LR/AMP,
evaluates Dice, mIoU, AP, AP-small, writes logs to `logs/training/`,
checkpoints to `work_dirs/`, and TensorBoard events to
`work_dirs/<experiment>/tensorboard`.

`breastseg_infer.py` is the batch-friendly single-view inference script used both by
humans and by `recon3d.perception.inference` for preprocessing/postprocessing
helpers.

## `tests/`

```text
tests/
|-- test_backproject.py
|-- test_calibration_math.py
|-- test_cli_config.py
|-- test_dataset.py
|-- test_forward.py
|-- test_fusion.py
|-- test_nipple3d.py
|-- test_pipeline.py
|-- test_sensors.py
|-- run_breastseg_image_tests.py
|-- run_breastseg_video_tests.py
`-- singleview/
    |-- input_vis/
    `-- output_vis/
```

The unit tests mainly protect the new RGB-D path and key model/data contracts.
`singleview/` stores small image/video assets and visual outputs for quick
manual inspection of 2D perception.

## Runtime Outputs

2D training outputs:

```text
logs/training/train_breastseg_b2_*.log
work_dirs/breastseg_b2/epoch_*.pth
work_dirs/breastseg_b2/best.pth
work_dirs/breastseg_b2/tensorboard/
```

Orbbec SDK hardware logs:

```text
logs/orbbec_sdk/OrbbecSDK.log*.txt
Log/OrbbecSDK.log*.txt
```

Prefer `logs/orbbec_sdk/` for collected SDK logs. The capitalized `Log/`
directory is kept ignored because some Orbbec SDK runs create it automatically.

3D reconstruction outputs:

```text
runs/<session>/fused_breast.ply
runs/<session>/nipple_world.json
runs/<session>/run_summary.json
```

`fused_breast.ply` is the fused world-frame point cloud. `nipple_world.json`
stores left/right world coordinates, covariance, confidence, used cameras, and
rejected cameras. `run_summary.json` stores timing and measurable run metrics.

## Standard Commands

```bash
python data_preprocess/convert_breast_labelme.py --json-dir ./data/breastseg/raw_combined/labelme --image-dir ./data/breastseg/raw_combined/images --out ./data/breastseg/processed_merged
python data_preprocess/regenerate_splits.py --images-dir ./data/breastseg/processed_merged/images --splits-dir ./data/breastseg/processed_merged/splits
python tools/train_breastseg.py --config configs/breastseg/breastseg_b2_inplace.py
python tools/breastseg_infer.py --config configs/breastseg/breastseg_b2_inplace.py --checkpoint checkpoint/breastseg.pth --input tests/singleview/input_vis --save-dir tests/singleview/output_vis
python -m recon3d.cli replay --config configs/recon3d/default.yaml --bag-session data/Replay/orbbec_bag_720p_100s --out runs/orbbec_bag_720p_100s_replay --visualize --probe-frames 180 --wait-timeout-ms 5000
python -m pytest
ruff check .
mypy recon3d --strict
```
