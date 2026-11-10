# RGB-D recording and TAPIP3D surface tracking

Open `/viewer/recording.html` on the local AnatoBreast3D viewer server, or choose
**Record & track** from the reconstruction viewer.

## Recording screen

1. Start camera preview. The workstation profile opens the two calibrated
   Femto Bolt cameras (`cam1` / `CL8T75401EJ`, `cam2` / `CL8T75400XA`).
2. Position the phantom using the two colour views. The depth slider overlays
   the aligned depth image on colour to help inspect boundaries.
3. Inspect the cyan and orange point clouds in the 3D overlap view. Drag to
   rotate, scroll to zoom, and hide either camera to compare its surface.
   The updated preview uses breast segmentation to focus the 3D view. The
   foreground toggle is available when the running server supports it.
4. Set the duration and start recording. Stop & save ends a clip early.
   Saving releases the cameras and writes a new SDK bag session plus a copy of
   its calibration. Start preview again to frame another clip.
5. Select a saved recording and camera, then track breast surface points.
   View 3D tracks provides a frame slider and playback; CSV and NPZ preserve
   point identities, world coordinates, and predicted visibility.

Move the phantom to change framing. Moving a camera invalidates its saved
pose and requires recalibration. A loaded calibration and a small timing
difference do not establish geometric accuracy. Look for double surfaces and
misaligned boundaries before recording movement.

Recordings are created under `data/Replay/tapip3d-*`. Preview assets, logs,
prepared sequences, and tracking results live under `runs/point_tracking`.
Both locations are already excluded from Git. The SDK bags retain the captured
colour/depth streams; JPEG previews and downsampled tracking inputs do not
replace them. Never treat the initial photo-based reconstruction as temporal
RGB-D footage.

## Calibrated workstation profile

`configs/recon3d/tracking_capture.yaml` references the physically verified
two-camera calibration saved on 6 November 2026. Its source is outside this
repository, in the workspace's `tools/orbbec/calibration` folder. A copy is
saved with every recording. The profile uses 1280 × 720 colour at 30 fps and
SDK software depth-to-colour alignment. It leaves hardware trigger modes
unchanged and enables SDK global timestamps plus device-clock synchronization.
These are standalone cameras; this does not turn them into a hardware-triggered
rig. The preview rejects frame pairs more than 50 ms apart.

The first six-second capture on 11 November 2026 saved both bags with 100
accepted monitored frame pairs, no rejected pairs, and a maximum measured
timestamp difference of 16.289 ms. The original recorder's session duration
includes finalization time; the new recording screen measures clip duration
before finalization. A stationary capture is an integration check, not a
validation of deformation tracking.

## Optional runtime

The official project is <https://tapip3d.github.io/> and its source is
<https://github.com/zbw001/TAPIP3D> (Apache 2.0). This integration was checked
against revision `4cb7e69a1687f67d56ec3e506768f51f2c581b46` and the authors'
`tapip3d_final.pth` checkpoint from <https://huggingface.co/zbww/tapip3d>.

Configure the source checkout, checkpoint, and Python executable in
`configs/tracking/tapip3d.json`. Paths are relative to the project root;
`ANATOBREAST3D_TRACKING_PYTHON` can override the executable. These large
dependencies are kept outside this repository. The current workspace has them
under `../tools/tracking/`.

For another workstation, create an isolated Python 3.10/3.11 environment with
the application's compatible CUDA PyTorch, torchvision, NumPy, OpenCV, and
Orbbec SDK. Install `requirements-tracking.txt` there, clone the pinned source,
and download the checkpoint. On this workstation the isolated environment
reuses the existing perception environment's installed packages through a
`.pth` entry, with tracking-specific additions installed separately. Existing
perception packages were not replaced.

Measured RGB-D inputs do not need MegaSAM or monocular depth prediction. The
adapter runs the released network and loads every checkpoint tensor strictly.
It skips downloading an unrelated encoder initialization that would immediately
be overwritten by the checkpoint. FP32 inference supports the RTX 2070; the
upstream demo's unconditional BF16 autocast does not. The optional `torch` KNN
backend implements exact, per-batch Euclidean neighbour search without compiling
pointops2. `native` selects the upstream extension where it is installed.
Benchmark equivalence and real-time throughput have not been established.

## Data contract and coordinates

Tracking follows one physical camera across time. Run each camera separately;
this integration does not associate track identities between cameras. All
queries initialize at the first selected frame, inside the breast mask and
away from invalid depths and discontinuities. The target is up to 128 points
over 16 frames, sampling every third source frame by default.

RGB, depth, and masks are rectified together, then resized with corresponding
camera intrinsics. Raw depth is converted to camera Z depth in metres. Camera
poses are the fixed recorded calibration, inverted from `T_world_cam` into the
world-to-camera transforms expected by TAPIP3D. Per-frame nipple/ICP alignment
is deliberately excluded because it could absorb real surface movement.

Playback frames are collected before segmentation and preparation. Loading a
segmentation model during real-time SDK playback otherwise skips initial
frames. Actual source timestamps are retained in the prepared input and result.

Prepared NPZ keys: `video` (T,H,W,3 uint8 RGB), `depths` (T,H,W float32 metres),
`intrinsics` (T,3,3), `extrinsics` (T,4,4 world-to-camera), `query_point`
(N,4: first-frame index and world XYZ), `timestamps_ns`, and `first_mask`.
The optional tracker runs in a separate process so the upstream generic
`models`, `datasets`, and `utils` modules cannot collide with this application.

Outputs include `coords_world_m` and `visible` in `tracks.npz`, a row per
frame/point in `tracks.csv`, and a browser preview in `result.json`. Hidden or
non-finite predictions are not displayed as observed surface points. The
movement colours represent displacement from initialization, not a confidence
score or an accuracy measurement. Continuous live inference and cross-window
identity persistence remain a later stage; the current live mode is framing and
recording, followed by offline tracking.

## Command line and verification

```powershell
python -m anatomy_guided_recon3d.tracking.capture --config configs/recon3d/tracking_capture.yaml --duration 6 --out data/Replay/my_clip --no-sync-config
python -m anatomy_guided_recon3d.tracking.cli --session data/Replay/my_clip --camera cam2 --out runs/point_tracking/my_clip_cam2
python -m pytest tests/test_point_tracking.py tests/test_viewer_server.py tests/test_anatomy_guided_pipeline.py -q
```

The tracking CLI also accepts legacy camera folders containing RGB/depth PNGs
and `metadata.json`, an explicit first-frame mask, or an already prepared NPZ.
Session-local calibration takes precedence over the config calibration.

`tools/check_recording_ui.cjs` verifies actual camera previews, depth overlay,
camera visibility controls, and tracking playback in Chrome. It accepts
`VIEWER_URL`, `CHROME_PATH`, and `PLAYWRIGHT_MODULE` environment variables.
Its optional `--record` check creates a two-second recording and restarts the
preview; run that option only when ready to capture.

Three.js r160 and OrbitControls are included for the local 3D view, under
`viewer/vendor/LICENSE` (MIT). All viewer resources are served locally with
explicit JavaScript MIME types and disabled caching.
