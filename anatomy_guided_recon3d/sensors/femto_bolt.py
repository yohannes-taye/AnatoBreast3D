from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from anatomy_guided_recon3d.calibration.io import camera_intrinsics_from_dict, depth_config_from_dict
from anatomy_guided_recon3d.config import CameraIntrinsics, DepthConfig, RGBDFrame


LOGGER = logging.getLogger(__name__)

# Optional Orbbec SDK v2 binding. We catch *any* exception, not just ImportError,
# because on Windows the native import can fail with OSError/RuntimeError when the
# Femto Bolt Microsoft depth-engine DLL (depthengine_2_0.dll) is missing from PATH.
try:
    import pyorbbecsdk as _ob  # type: ignore
except Exception as _exc:  # noqa: BLE001 - SDK is an optional hardware dependency
    _ob = None
    _OB_IMPORT_ERROR: Exception | None = _exc
else:
    _OB_IMPORT_ERROR = None

# String -> OBMultiDeviceSyncMode attribute name. Wired hardware sync uses one
# PRIMARY plus N SECONDARY cameras connected by the Orbbec sync hub/cable.
_SYNC_MODE_NAMES = {
    "standalone": "STANDALONE",
    "free_run": "FREE_RUN",
    "primary": "PRIMARY",
    "secondary": "SECONDARY",
    "secondary_synced": "SECONDARY_SYNCED",
    "software_triggering": "SOFTWARE_TRIGGERING",
    "hardware_triggering": "HARDWARE_TRIGGERING",
}

_CONTEXT = None  # lazily created singleton pyorbbecsdk.Context shared by all cameras


def _get_context():
    """Return the shared process-wide Orbbec ``Context`` (created lazily).

    Input shape/unit/frame: no input.
    Return type: ``pyorbbecsdk.Context``.
    """
    global _CONTEXT
    if _ob is None:
        raise HardwareUnavailableError(
            "pyorbbecsdk is not importable; cannot create an Orbbec Context"
        )
    if _CONTEXT is None:
        _CONTEXT = _ob.Context()
    return _CONTEXT


def _color_frame_to_rgb(color_frame) -> np.ndarray:
    """Convert an Orbbec color frame to an HxWx3 uint8 RGB array.

    Input shape/unit/frame: SDK color frame in RGB/BGR/MJPG/YUYV/UYVY/NV12/NV21/I420.
    Return type: HxWx3 uint8 ndarray in RGB channel order.
    """
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    if fmt == _ob.OBFormat.RGB:
        return data.reshape(height, width, 3).copy()
    if fmt == _ob.OBFormat.BGR:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
    if fmt == _ob.OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode MJPG color frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == _ob.OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV)
    if hasattr(_ob.OBFormat, "YUY2") and fmt == _ob.OBFormat.YUY2:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
    if fmt == _ob.OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_UYVY)
    if fmt == _ob.OBFormat.NV12:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)
    if fmt == _ob.OBFormat.NV21:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV21)
    if fmt == _ob.OBFormat.I420:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_I420)
    raise RuntimeError(f"Unsupported Orbbec color format: {fmt}")


def _stream_profile_len(profile_list) -> int:
    """Return stream profile count across pyorbbecsdk minor API variants."""
    try:
        return len(profile_list)
    except TypeError:
        return int(profile_list.get_count())


def _stream_profile_at(profile_list, index: int):
    """Return one stream profile across pyorbbecsdk minor API variants."""
    try:
        return profile_list[index]
    except TypeError:
        return profile_list.get_stream_profile_by_index(index)


class HardwareUnavailableError(RuntimeError):
    """Raised when real Orbbec hardware capture is requested but unavailable."""


class FemtoBoltCamera:
    """Orbbec Femto Bolt RGB-D camera with depth-to-color (D2C) alignment.

    Wraps pyorbbecsdk v2. Every ``read_aligned_frame`` returns RGB and depth on a
    single shared pixel grid (the color grid), so a 2D segmentation mask indexes
    the depth directly with no further alignment. For a multi-camera rig, set one
    camera ``sync_mode="primary"`` and the rest ``sync_mode="secondary"`` and wire
    them with the Orbbec sync hub; start secondaries before the primary.
    """

    def __init__(
        self,
        cam_id: str,
        *,
        serial_number: str | None = None,
        sync_mode: str = "standalone",
        color_size: tuple[int, int] | None = None,
        color_fps: int = 0,
        color_format: str = "RGB",
        color_auto_exposure: bool | None = None,
        color_exposure: int | None = None,
        color_gain: int | None = None,
        align_mode: str = "hw",
        wait_timeout_ms: int = 1000,
        apply_sync_config: bool = True,
    ) -> None:
        """Create a hardware camera wrapper.

        Input shape/unit/frame: ``color_size`` is (width, height) pixels (``None``
        lets the SDK choose a D2C-supported pair); ``align_mode`` is ``"hw"`` for
        on-device D2C or ``"sw"`` for host-side ``AlignFilter``; ``sync_mode`` is a
        key of ``_SYNC_MODE_NAMES``.
        Return type: ``FemtoBoltCamera``.
        """
        self.cam_id = cam_id
        self.serial_number = serial_number
        self.sync_mode = sync_mode.lower()
        if self.sync_mode not in _SYNC_MODE_NAMES:
            raise ValueError(f"Unknown sync_mode '{sync_mode}'; expected one of {sorted(_SYNC_MODE_NAMES)}")
        self.align_mode = align_mode.lower()
        if self.align_mode not in ("hw", "sw"):
            raise ValueError("align_mode must be 'hw' or 'sw'")
        self.color_size = color_size
        self.color_fps = int(color_fps)
        self.color_format = color_format.upper()
        self.color_auto_exposure = color_auto_exposure
        self.color_exposure = color_exposure
        self.color_gain = color_gain
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.apply_sync_config = apply_sync_config

        self._sdk_available = self._check_sdk()
        self._running = False
        self._pipeline = None
        self._device = None
        self._align = None
        self._intrinsics: CameraIntrinsics | None = None
        self._depth_config = DepthConfig(depth_scale_m=0.001)

    @staticmethod
    def _check_sdk() -> bool:
        """Return whether the pyorbbecsdk binding imported successfully."""
        return _ob is not None

    def _resolve_device(self):
        device_list = _get_context().query_devices()
        count = device_list.get_count()
        if count == 0:
            raise HardwareUnavailableError("No Orbbec devices found on this host")
        if self.serial_number:
            try:
                device = device_list.get_device_by_serial_number(self.serial_number)
                if device is not None:
                    return device
            except Exception:  # noqa: BLE001 - fall back to manual scan across SDK versions
                pass
            for index in range(count):
                device = device_list.get_device_by_index(index)
                if device.get_device_info().get_serial_number() == self.serial_number:
                    return device
            raise HardwareUnavailableError(f"Orbbec device with serial '{self.serial_number}' not found")
        return device_list.get_device_by_index(0)

    def _select_color_profile(self):
        profiles = self._pipeline.get_stream_profile_list(_ob.OBSensorType.COLOR_SENSOR)
        width, height = self.color_size or (0, 0)
        fmt = getattr(_ob.OBFormat, self.color_format, None)
        if fmt is not None:
            try:
                return profiles.get_video_stream_profile(width, height, fmt, self.color_fps)
            except Exception:  # noqa: BLE001 - requested format/size may be unsupported
                LOGGER.warning(
                    "[%s] color profile %s @ %sx%s unavailable; using default profile",
                    self.cam_id,
                    self.color_format,
                    width,
                    height,
                )
        return profiles.get_default_video_stream_profile()

    def _select_hw_d2c_profiles(self):
        """Select a hardware-D2C-compatible color/depth profile pair.

        Hardware D2C does not support every arbitrary color/depth pair. The SDK
        exposes valid depth profiles for each color profile; use that list
        instead of enabling a default depth profile and hoping it aligns.
        """
        color_profiles = self._pipeline.get_stream_profile_list(_ob.OBSensorType.COLOR_SENSOR)
        requested_color = self._select_color_profile()
        candidates = [requested_color]
        for idx in range(_stream_profile_len(color_profiles)):
            profile = _stream_profile_at(color_profiles, idx)
            if profile is requested_color:
                continue
            try:
                if self.color_size is not None:
                    video = profile.as_video_stream_profile()
                    if (video.get_width(), video.get_height()) != self.color_size:
                        continue
                fmt = getattr(_ob.OBFormat, self.color_format, None)
                if fmt is not None and profile.get_format() != fmt:
                    continue
            except Exception:  # noqa: BLE001 - skip profiles that do not expose video metadata consistently
                continue
            candidates.append(profile)

        last_error: Exception | None = None
        for color_profile in candidates:
            try:
                depth_profiles = self._pipeline.get_d2c_depth_profile_list(color_profile, _ob.OBAlignMode.HW_MODE)
                if _stream_profile_len(depth_profiles) > 0:
                    return color_profile, _stream_profile_at(depth_profiles, 0)
            except Exception as exc:  # noqa: BLE001 - try the next color profile
                last_error = exc
                continue
        raise HardwareUnavailableError(
            f"[{self.cam_id}] no hardware-D2C-compatible color/depth profile pair found"
            + (f" ({last_error})" if last_error else "")
        )

    def _set_bool_property_if_supported(self, property_id, value: bool, name: str) -> None:
        if self._device is None:
            return
        try:
            if self._device.is_property_supported(property_id, _ob.OBPermissionType.PERMISSION_READ_WRITE):
                self._device.set_bool_property(property_id, bool(value))
        except Exception as exc:  # noqa: BLE001 - camera property support varies by firmware
            LOGGER.warning("[%s] failed to set %s=%s: %s", self.cam_id, name, value, exc)

    def _set_int_property_if_supported(self, property_id, value: int, name: str) -> None:
        if self._device is None:
            return
        try:
            if self._device.is_property_supported(property_id, _ob.OBPermissionType.PERMISSION_READ_WRITE):
                self._device.set_int_property(property_id, int(value))
        except Exception as exc:  # noqa: BLE001 - camera property support varies by firmware
            LOGGER.warning("[%s] failed to set %s=%s: %s", self.cam_id, name, value, exc)

    def _apply_color_controls(self) -> None:
        if self.color_auto_exposure is not None:
            self._set_bool_property_if_supported(
                _ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                bool(self.color_auto_exposure),
                "color_auto_exposure",
            )
        if self.color_exposure is not None:
            self._set_bool_property_if_supported(
                _ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                False,
                "color_auto_exposure",
            )
            self._set_int_property_if_supported(
                _ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
                int(self.color_exposure),
                "color_exposure",
            )
        if self.color_gain is not None:
            self._set_int_property_if_supported(
                _ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT,
                int(self.color_gain),
                "color_gain",
            )

    @staticmethod
    def _intrinsics_from_profile(color_profile) -> CameraIntrinsics:
        intr = color_profile.as_video_stream_profile().get_intrinsic()
        return CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            distortion=None,
        )

    @staticmethod
    def _rescale_intrinsics(intr: CameraIntrinsics, new_w: int, new_h: int) -> CameraIntrinsics:
        sx = new_w / intr.width
        sy = new_h / intr.height
        return CameraIntrinsics(
            width=new_w,
            height=new_h,
            fx=intr.fx * sx,
            fy=intr.fy * sy,
            cx=intr.cx * sx,
            cy=intr.cy * sy,
            distortion=None,
        )

    def start(self) -> None:
        """Start hardware capture with D2C alignment (and sync role if configured).

        Input shape/unit/frame: produces depth-to-color aligned RGB-D streams.
        Return type: ``None``.
        """
        if not self._sdk_available:
            raise HardwareUnavailableError(
                "pyorbbecsdk is not importable; install it (pip install pyorbbecsdk) and the Femto "
                "Bolt depth-engine DLL, or use ReplayRGBDCamera/MockRGBDCamera. "
                f"Original import error: {_OB_IMPORT_ERROR!r}"
            )
        self._device = self._resolve_device()
        self._apply_color_controls()

        # Apply multi-device hardware sync role before streaming (skip for standalone).
        if self.apply_sync_config and self.sync_mode != "standalone":
            try:
                sync_config = self._device.get_multi_device_sync_config()
                sync_config.mode = getattr(_ob.OBMultiDeviceSyncMode, _SYNC_MODE_NAMES[self.sync_mode])
                self._device.set_multi_device_sync_config(sync_config)
            except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
                raise HardwareUnavailableError(
                    f"[{self.cam_id}] failed to set multi-device sync mode '{self.sync_mode}': {exc}"
                ) from exc

        self._pipeline = _ob.Pipeline(self._device)
        config = _ob.Config()

        use_software_align = self.align_mode == "sw"
        if self.align_mode == "hw":
            try:
                color_profile, depth_profile = self._select_hw_d2c_profiles()
                config.enable_stream(depth_profile)
                config.enable_stream(color_profile)
                config.set_align_mode(_ob.OBAlignMode.HW_MODE)
            except HardwareUnavailableError as exc:
                LOGGER.warning("[%s] hardware D2C unavailable; falling back to software D2C: %s", self.cam_id, exc)
                config = _ob.Config()
                use_software_align = True
        if use_software_align:
            color_profile = self._select_color_profile()
            config.enable_stream(color_profile)
            depth_profiles = self._pipeline.get_stream_profile_list(_ob.OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
            if hasattr(_ob, "OBFrameAggregateOutputMode"):
                config.set_frame_aggregate_output_mode(_ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            config.set_align_mode(_ob.OBAlignMode.DISABLE)
            self._align = _ob.AlignFilter(align_to_stream=_ob.OBStreamType.COLOR_STREAM)

        self._intrinsics = self._intrinsics_from_profile(color_profile)
        self._pipeline.start(config)
        self._running = True
        LOGGER.info("[%s] Femto Bolt started (sync=%s, align=%s)", self.cam_id, self.sync_mode, self.align_mode)

    def stop(self) -> None:
        """Stop hardware capture.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - best-effort teardown
                LOGGER.exception("[%s] error while stopping pipeline", self.cam_id)
            self._pipeline = None
        self._align = None
        self._device = None

    def enable_multi_device_sync(self, timeout_ms: int = 60000) -> None:
        """Enable Orbbec context-level multi-device synchronization.

        Input shape/unit/frame: timeout is milliseconds accepted by the SDK.
        Return type: ``None``.
        """
        _get_context().enable_multi_device_sync(int(timeout_ms))

    def read_aligned_frame(self) -> RGBDFrame:
        """Read one D2C-aligned RGB-D frame from hardware.

        Input shape/unit/frame: hardware output is RGB HxWx3 uint8 and depth HxW
        raw units on the same (color) grid; raw depth -> metres via depth scale.
        Return type: ``RGBDFrame``.
        """
        if not self._running or self._pipeline is None:
            raise RuntimeError(f"Camera '{self.cam_id}' is not started")
        frames = self._pipeline.wait_for_frames(self.wait_timeout_ms)
        if frames is None:
            raise RuntimeError(f"Camera '{self.cam_id}' timed out waiting for frames")
        if self._align is not None:
            frames = self._align.process(frames)
            if hasattr(frames, "as_frame_set"):
                frames = frames.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError(f"Camera '{self.cam_id}' returned an incomplete RGB-D frame set")

        rgb = _color_frame_to_rgb(color_frame)
        depth_h = depth_frame.get_height()
        depth_w = depth_frame.get_width()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_h, depth_w)
        if rgb.shape[:2] != depth_raw.shape:
            raise RuntimeError(
                f"[{self.cam_id}] D2C alignment mismatch: rgb {rgb.shape[:2]} vs depth "
                f"{depth_raw.shape}; pick a D2C-supported color/depth resolution pair"
            )

        # depth_scale converts raw depth units -> millimetres; fold mm -> metres in.
        depth_scale_mm = float(depth_frame.get_depth_scale())
        self._depth_config = DepthConfig(depth_scale_m=depth_scale_mm * 0.001)

        intrinsics = self._intrinsics
        if intrinsics is None:
            raise RuntimeError(f"Camera '{self.cam_id}' has no intrinsics; call start() first")
        if (intrinsics.width, intrinsics.height) != (rgb.shape[1], rgb.shape[0]):
            intrinsics = self._rescale_intrinsics(intrinsics, rgb.shape[1], rgb.shape[0])
            self._intrinsics = intrinsics

        timestamp_us = int(color_frame.get_timestamp_us())
        if timestamp_us <= 0 and hasattr(depth_frame, "get_timestamp_us"):
            timestamp_us = int(depth_frame.get_timestamp_us())
        if timestamp_us <= 0 and hasattr(color_frame, "get_system_timestamp_us"):
            timestamp_us = int(color_frame.get_system_timestamp_us())
        timestamp_ns = max(timestamp_us * 1000, 0)
        return RGBDFrame(
            cam_id=self.cam_id,
            rgb=rgb,
            depth_mm=depth_raw,
            timestamp_ns=timestamp_ns,
            intrinsics=intrinsics,
            depth_config=self._depth_config,
        )

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return color intrinsics (valid after ``start``).

        Input shape/unit/frame: SDK color camera intrinsics in pixels.
        Return type: ``CameraIntrinsics``.
        """
        if self._intrinsics is None:
            raise HardwareUnavailableError("Call start() before get_intrinsics()")
        return self._intrinsics

    def get_depth_config(self) -> DepthConfig:
        """Return depth conversion settings.

        Input shape/unit/frame: metre scale; refreshed from the SDK on each frame
        read, defaults to a millimetre raw unit before the first read.
        Return type: ``DepthConfig``.
        """
        return self._depth_config


class ReplayRGBDCamera:
    """File-backed RGB-D replay camera compatible with ``FemtoBoltCamera``."""

    def __init__(self, cam_id: str, root: Path) -> None:
        """Create a replay camera.

        Input shape/unit/frame: root contains ``metadata.json`` plus ``rgb`` and
        ``depth`` folders for this camera; depth PNGs are raw millimetres.
        Return type: ``ReplayRGBDCamera``.
        """
        self.cam_id = cam_id
        self.root = Path(root)
        meta_path = self.root / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Replay camera metadata missing: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("intrinsics", "depth_config", "timestamps_ns"):
            if key not in metadata:
                raise ValueError(f"Replay metadata missing required key '{key}': {meta_path}")
        self.intrinsics = camera_intrinsics_from_dict(metadata["intrinsics"])
        self.depth_config = depth_config_from_dict(metadata["depth_config"])
        self.timestamps = [int(value) for value in metadata["timestamps_ns"]]
        if not self.timestamps:
            raise ValueError(f"Replay camera has no timestamps: {meta_path}")
        self.rgb_paths = [self.root / "rgb" / f"{idx:06d}.png" for idx in range(len(self.timestamps))]
        self.depth_paths = [self.root / "depth" / f"{idx:06d}.png" for idx in range(len(self.timestamps))]
        missing = [path for pair in zip(self.rgb_paths, self.depth_paths) for path in pair if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:4])
            raise FileNotFoundError(f"Replay camera '{cam_id}' has missing frame files: {preview}")
        self._index = 0
        self._running = False

    def start(self) -> None:
        """Start replay.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        self._running = True
        self._index = 0

    def stop(self) -> None:
        """Stop replay.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        self._running = False

    def read_aligned_frame(self) -> RGBDFrame:
        """Read one replay RGB-D frame.

        Input shape/unit/frame: RGB PNG is HxWx3 RGB; depth PNG is HxW raw mm.
        Return type: ``RGBDFrame``.
        """
        if self._index >= len(self.timestamps):
            raise StopIteration(f"Replay camera '{self.cam_id}' exhausted")
        rgb_bgr = cv2.imread(str(self.rgb_paths[self._index]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.depth_paths[self._index]), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth is None:
            raise FileNotFoundError(f"Missing replay frame {self._index} for camera {self.cam_id}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        frame = RGBDFrame(
            cam_id=self.cam_id,
            rgb=rgb,
            depth_mm=depth,
            timestamp_ns=self.timestamps[self._index],
            intrinsics=self.intrinsics,
            depth_config=self.depth_config,
        )
        self._index += 1
        return frame

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return replay color intrinsics.

        Input shape/unit/frame: pixel intrinsics loaded from JSON.
        Return type: ``CameraIntrinsics``.
        """
        return self.intrinsics

    def get_depth_config(self) -> DepthConfig:
        """Return replay depth conversion settings.

        Input shape/unit/frame: metre scale loaded from JSON.
        Return type: ``DepthConfig``.
        """
        return self.depth_config


class MockRGBDCamera:
    """In-memory replay camera for tests and examples."""

    def __init__(self, cam_id: str, frames: list[RGBDFrame]) -> None:
        """Create an in-memory mock camera.

        Input shape/unit/frame: frames are aligned RGB-D frames for one camera.
        Return type: ``MockRGBDCamera``.
        """
        if not frames:
            raise ValueError("MockRGBDCamera requires at least one RGBDFrame")
        self.cam_id = cam_id
        self.frames = list(frames)
        self._index = 0

    def start(self) -> None:
        """Start mock replay.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        self._index = 0

    def stop(self) -> None:
        """Stop mock replay.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """

    def read_aligned_frame(self) -> RGBDFrame:
        """Read one in-memory RGB-D frame.

        Input shape/unit/frame: stored frame arrays are aligned RGB/depth.
        Return type: ``RGBDFrame``.
        """
        if self._index >= len(self.frames):
            raise StopIteration(f"Mock camera '{self.cam_id}' exhausted")
        frame = self.frames[self._index]
        self._index += 1
        return frame

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return mock color intrinsics.

        Input shape/unit/frame: pixel intrinsics from first frame.
        Return type: ``CameraIntrinsics``.
        """
        return self.frames[0].intrinsics

    def get_depth_config(self) -> DepthConfig:
        """Return mock depth config.

        Input shape/unit/frame: metre scale from first frame.
        Return type: ``DepthConfig``.
        """
        return self.frames[0].depth_config


def create_rgbd_camera(
    cam_id: str,
    *,
    replay_root: Path | str | None = None,
    mock_frames: Sequence[RGBDFrame] | Iterable[RGBDFrame] | None = None,
    prefer_hardware: bool = True,
    **femto_kwargs,
):
    """Build an RGB-D camera, gracefully degrading when the SDK is unavailable.

    Input shape/unit/frame: ``femto_kwargs`` are forwarded to ``FemtoBoltCamera``
    (e.g. ``serial_number``, ``sync_mode``, ``align_mode``); ``replay_root`` and
    ``mock_frames`` are fallbacks used when pyorbbecsdk cannot be imported.
    Return type: ``FemtoBoltCamera`` | ``ReplayRGBDCamera`` | ``MockRGBDCamera``.
    """
    if prefer_hardware and FemtoBoltCamera._check_sdk():
        try:
            return FemtoBoltCamera(cam_id, **femto_kwargs)
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing the rig
            LOGGER.warning("[%s] FemtoBoltCamera unavailable (%s); falling back", cam_id, exc)
    if replay_root is not None:
        return ReplayRGBDCamera(cam_id, Path(replay_root))
    if mock_frames is not None:
        return MockRGBDCamera(cam_id, list(mock_frames))
    raise HardwareUnavailableError(
        f"[{cam_id}] pyorbbecsdk unavailable and no replay_root/mock_frames provided for fallback. "
        f"Original import error: {_OB_IMPORT_ERROR!r}"
    )

