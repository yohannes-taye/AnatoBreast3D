from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from anatomy_guided_recon3d.config import CameraExtrinsics, CameraIntrinsics, RGBDFrame


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Charuco board dimensions.

    Input shape/unit/frame: square and marker lengths are metres in board frame.
    Return type: immutable board specification.
    """

    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary_name: int = cv2.aruco.DICT_5X5_100


@dataclass(frozen=True)
class CharucoDetection:
    """Detected Charuco corners for one RGB image.

    Input shape/unit/frame: corners are Nx1x2 pixels; ids are Nx1 Charuco IDs.
    Return type: immutable detection container.
    """

    corners_px: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class PoseEstimate:
    """Camera pose relative to a Charuco board.

    Input shape/unit/frame: ``T_cam_board`` maps board metres to camera metres.
    Return type: immutable pose with reprojection errors in pixels.
    """

    T_cam_board: np.ndarray
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    num_corners: int


def make_charuco_board(spec: CharucoBoardSpec) -> Any:
    """Create an OpenCV Charuco board.

    Input shape/unit/frame: board dimensions in metres.
    Return type: OpenCV ``cv2.aruco.CharucoBoard`` object.
    """
    aruco: Any = getattr(cv2, "aruco")
    dictionary = aruco.getPredefinedDictionary(spec.dictionary_name)
    return aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )


def detect_charuco(rgb: np.ndarray, board: Any) -> CharucoDetection | None:
    """Detect Charuco corners in an RGB image.

    Input shape/unit/frame: rgb is HxWx3 uint8 RGB in color camera pixels.
    Return type: ``CharucoDetection`` or ``None``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    aruco: Any = getattr(cv2, "aruco")
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        count = 0 if charuco_ids is None else len(charuco_ids)
    else:
        dictionary = board.getDictionary()
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
        if ids is None or len(corners) == 0:
            return None
        count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if count is None or count < 4 or charuco_corners is None or charuco_ids is None:
        return None
    return CharucoDetection(corners_px=charuco_corners.astype(np.float64), ids=charuco_ids.astype(np.int32))


def _board_object_points(board: Any, ids: np.ndarray) -> np.ndarray:
    chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    flat_ids = ids.reshape(-1).astype(np.int64)
    if flat_ids.max(initial=-1) >= chessboard_corners.shape[0]:
        raise ValueError("Charuco ID outside board corner range")
    return chessboard_corners[flat_ids]


def _rt_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def estimate_board_pose(
    detection: CharucoDetection,
    board: Any,
    intrinsics: CameraIntrinsics,
) -> PoseEstimate:
    """Estimate camera pose relative to the Charuco board.

    Input shape/unit/frame: detection corners are pixels; board points are metres;
    returned transform maps board frame to camera frame.
    Return type: ``PoseEstimate``.
    """
    object_points = _board_object_points(board, detection.ids)
    image_points = detection.corners_px.reshape(-1, 2).astype(np.float64)
    dist = np.zeros((0,), dtype=np.float64) if intrinsics.distortion is None else np.asarray(intrinsics.distortion, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for Charuco detection")
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, intrinsics.K, dist, rvec, tvec)
    except cv2.error as exc:
        LOGGER.warning("cv2.solvePnPRefineLM failed; using solvePnP result: %s", exc)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics.K, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return PoseEstimate(
        T_cam_board=_rt_to_transform(rvec, tvec),
        mean_reprojection_error_px=float(errors.mean()),
        max_reprojection_error_px=float(errors.max()),
        num_corners=int(object_points.shape[0]),
    )


def estimate_extrinsics_from_charuco(
    frames: Mapping[str, RGBDFrame],
    board_spec: CharucoBoardSpec,
    world: str = "board",
) -> tuple[dict[str, CameraExtrinsics], dict[str, PoseEstimate]]:
    """Estimate ``T_world_cam`` from synchronized Charuco observations.

    Input shape/unit/frame: frames contain RGB images in camera pixels; board
    dimensions are metres. With ``world='board'``, board frame is world frame.
    With ``world='first_camera'`` or a camera ID, that camera frame becomes world.
    Return type: ``(extrinsics, poses)`` dictionaries keyed by camera ID.
    """
    board = make_charuco_board(board_spec)
    poses: dict[str, PoseEstimate] = {}
    for cam_id, frame in frames.items():
        detection = detect_charuco(frame.rgb, board)
        if detection is None:
            continue
        poses[cam_id] = estimate_board_pose(detection, board, frame.intrinsics)

    extrinsics: dict[str, CameraExtrinsics] = {}
    if world == "board":
        for cam_id, pose in poses.items():
            T_world_cam = np.linalg.inv(pose.T_cam_board)
            extrinsics[cam_id] = CameraExtrinsics(cam_id=cam_id, T_world_cam=T_world_cam)
        return extrinsics, poses

    if not poses:
        return {}, poses
    reference_cam_id = next(iter(poses)) if world == "first_camera" else world
    if reference_cam_id not in poses:
        raise ValueError(f"Reference world camera '{reference_cam_id}' was not detected")
    T_ref_board = poses[reference_cam_id].T_cam_board
    for cam_id, pose in poses.items():
        T_world_cam = T_ref_board @ np.linalg.inv(pose.T_cam_board)
        extrinsics[cam_id] = CameraExtrinsics(cam_id=cam_id, T_world_cam=T_world_cam)
    return extrinsics, poses


def summarize_reprojection_errors(poses: Mapping[str, PoseEstimate]) -> dict[str, float | int]:
    """Summarize calibration reprojection errors.

    Input shape/unit/frame: pose errors are pixels.
    Return type: dict with mean/max error and warning flags.
    """
    if not poses:
        return {
            "num_cameras": 0,
            "mean_reprojection_error_px": float("nan"),
            "max_reprojection_error_px": float("nan"),
            "mean_error_gt_1px": 0,
            "max_error_gt_2px": 0,
        }
    means = np.array([pose.mean_reprojection_error_px for pose in poses.values()], dtype=np.float64)
    maxes = np.array([pose.max_reprojection_error_px for pose in poses.values()], dtype=np.float64)
    mean_error = float(means.mean())
    max_error = float(maxes.max())
    return {
        "num_cameras": len(poses),
        "mean_reprojection_error_px": mean_error,
        "max_reprojection_error_px": max_error,
        "mean_error_gt_1px": int(mean_error > 1.0),
        "max_error_gt_2px": int(max_error > 2.0),
    }

