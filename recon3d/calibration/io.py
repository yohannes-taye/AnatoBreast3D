from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from recon3d.config import CameraExtrinsics, CameraIntrinsics, DepthConfig


COORDINATE_CONVENTION = "p_world = T_world_cam @ p_cam_h"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def camera_intrinsics_to_dict(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    """Serialize color intrinsics.

    Input shape/unit/frame: pixels and OpenCV distortion coefficients.
    Return type: JSON-compatible dict.
    """
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion": None if intrinsics.distortion is None else intrinsics.distortion.tolist(),
    }


def camera_intrinsics_from_dict(data: Mapping[str, Any]) -> CameraIntrinsics:
    """Deserialize color intrinsics.

    Input shape/unit/frame: JSON fields in pixel units.
    Return type: ``CameraIntrinsics``.
    """
    distortion_raw = data.get("distortion")
    distortion = None if distortion_raw is None else np.asarray(distortion_raw, dtype=np.float64)
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        distortion=distortion,
    )


def camera_extrinsics_to_dict(extrinsics: CameraExtrinsics) -> dict[str, Any]:
    """Serialize camera-to-world transform.

    Input shape/unit/frame: 4x4 ``T_world_cam`` in metres.
    Return type: JSON-compatible dict.
    """
    return {
        "cam_id": extrinsics.cam_id,
        "T_world_cam": extrinsics.T_world_cam.tolist(),
        "coordinate_convention": COORDINATE_CONVENTION,
    }


def camera_extrinsics_from_dict(cam_id: str, data: Mapping[str, Any]) -> CameraExtrinsics:
    """Deserialize camera-to-world transform.

    Input shape/unit/frame: 4x4 ``T_world_cam`` in metres.
    Return type: ``CameraExtrinsics``.
    """
    stored_cam_id = str(data.get("cam_id", cam_id))
    T_world_cam = np.asarray(data["T_world_cam"], dtype=np.float64)
    return CameraExtrinsics(cam_id=stored_cam_id, T_world_cam=T_world_cam)


def depth_config_to_dict(depth_config: DepthConfig) -> dict[str, Any]:
    """Serialize depth conversion settings.

    Input shape/unit/frame: metre scale and metre depth range.
    Return type: JSON-compatible dict.
    """
    return {
        "depth_scale_m": depth_config.depth_scale_m,
        "min_depth_m": depth_config.min_depth_m,
        "max_depth_m": depth_config.max_depth_m,
    }


def depth_config_from_dict(data: Mapping[str, Any]) -> DepthConfig:
    """Deserialize depth conversion settings.

    Input shape/unit/frame: metre scale and metre depth range.
    Return type: ``DepthConfig``.
    """
    return DepthConfig(
        depth_scale_m=float(data["depth_scale_m"]),
        min_depth_m=float(data.get("min_depth_m", 0.25)),
        max_depth_m=float(data.get("max_depth_m", 3.0)),
    )


def save_intrinsics(path: Path, intrinsics: Mapping[str, CameraIntrinsics]) -> None:
    """Save color intrinsics in JSON.

    Input shape/unit/frame: mapping from camera ID to pixel intrinsics.
    Return type: ``None``.
    """
    data = {
        "type": "recon3d_intrinsics",
        "cameras": {cam_id: camera_intrinsics_to_dict(value) for cam_id, value in intrinsics.items()},
    }
    _write_json(path, data)


def load_intrinsics(path: Path) -> dict[str, CameraIntrinsics]:
    """Load color intrinsics from JSON.

    Input shape/unit/frame: JSON file with pixel intrinsics.
    Return type: mapping from camera ID to ``CameraIntrinsics``.
    """
    data = _read_json(path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"intrinsics JSON must contain cameras object: {path}")
    return {cam_id: camera_intrinsics_from_dict(value) for cam_id, value in cameras.items()}


def save_extrinsics(path: Path, extrinsics: Mapping[str, CameraExtrinsics]) -> None:
    """Save camera-to-world extrinsics in JSON.

    Input shape/unit/frame: mapping from camera ID to 4x4 ``T_world_cam`` in metres.
    Return type: ``None``.
    """
    data = {
        "type": "recon3d_extrinsics",
        "coordinate_convention": COORDINATE_CONVENTION,
        "cameras": {cam_id: camera_extrinsics_to_dict(value) for cam_id, value in extrinsics.items()},
    }
    _write_json(path, data)


def load_extrinsics(path: Path) -> dict[str, CameraExtrinsics]:
    """Load camera-to-world extrinsics from JSON.

    Input shape/unit/frame: JSON file with 4x4 ``T_world_cam`` in metres.
    Return type: mapping from camera ID to ``CameraExtrinsics``.
    """
    data = _read_json(path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"extrinsics JSON must contain cameras object: {path}")
    return {cam_id: camera_extrinsics_from_dict(cam_id, value) for cam_id, value in cameras.items()}


def save_depth_configs(path: Path, depth_configs: Mapping[str, DepthConfig]) -> None:
    """Save depth conversion settings in JSON.

    Input shape/unit/frame: mapping from camera ID to metre depth settings.
    Return type: ``None``.
    """
    data = {
        "type": "recon3d_depth_configs",
        "cameras": {cam_id: depth_config_to_dict(value) for cam_id, value in depth_configs.items()},
    }
    _write_json(path, data)


def load_depth_configs(path: Path) -> dict[str, DepthConfig]:
    """Load depth conversion settings from JSON.

    Input shape/unit/frame: JSON file with metre depth settings.
    Return type: mapping from camera ID to ``DepthConfig``.
    """
    data = _read_json(path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"depth config JSON must contain cameras object: {path}")
    return {cam_id: depth_config_from_dict(value) for cam_id, value in cameras.items()}
