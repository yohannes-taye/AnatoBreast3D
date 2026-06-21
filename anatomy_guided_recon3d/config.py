from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


Side = Literal["left", "right", "unknown"]


def _readonly_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type | None = np.float64,
    name: str = "array",
    finite: bool = True,
) -> np.ndarray:
    """Validate and return a read-only defensive ndarray copy.

    Input shape/unit/frame: arbitrary numeric ndarray, units depend on caller.
    Return type: read-only ``np.ndarray`` with optional dtype conversion.
    """
    arr = np.array(value, dtype=dtype, copy=True) if dtype is not None else np.array(value, copy=True)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, got {arr.ndim}")
    if finite and np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite numeric values")
    arr.setflags(write=False)
    return arr


def _validate_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class CameraIntrinsics:
    """Color camera intrinsics.

    Input shape/unit/frame: width/height in pixels; fx/fy/cx/cy in pixels;
    distortion is OpenCV coefficients for the color camera.
    Return type: immutable dataclass with read-only distortion ndarray.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("CameraIntrinsics width/height must be positive")
        for name in ("fx", "fy"):
            _validate_positive(float(getattr(self, name)), name)
        for name in ("cx", "cy"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.distortion is not None:
            object.__setattr__(
                self,
                "distortion",
                _readonly_array(self.distortion, ndim=1, name="distortion"),
            )

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 color intrinsic matrix in pixel units."""
        return _readonly_array(
            np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]),
            shape=(3, 3),
            name="K",
        )


@dataclass(frozen=True)
class DepthConfig:
    """Depth conversion and validity range.

    Input shape/unit/frame: depth_scale_m converts raw depth units to metres;
    min/max depth are metres in the camera frame.
    Return type: immutable dataclass.
    """

    depth_scale_m: float
    min_depth_m: float = 0.25
    max_depth_m: float = 3.0

    def __post_init__(self) -> None:
        _validate_positive(float(self.depth_scale_m), "depth_scale_m")
        _validate_positive(float(self.min_depth_m), "min_depth_m")
        _validate_positive(float(self.max_depth_m), "max_depth_m")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera-to-world transform.

    Input shape/unit/frame: ``T_world_cam`` is 4x4, metres, mapping camera-frame
    homogeneous points to world-frame points via ``p_world = T_world_cam @ p_cam_h``.
    Return type: immutable dataclass with read-only transform ndarray.
    """

    cam_id: str
    T_world_cam: np.ndarray

    def __post_init__(self) -> None:
        if not self.cam_id:
            raise ValueError("cam_id must be non-empty")
        T = _readonly_array(self.T_world_cam, shape=(4, 4), name="T_world_cam")
        if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError("T_world_cam last row must be [0, 0, 0, 1]")
        object.__setattr__(self, "T_world_cam", T)


@dataclass(frozen=True)
class RGBDFrame:
    """Aligned RGB-D frame.

    Input shape/unit/frame: rgb is HxWx3 uint8 RGB; depth_mm is HxW raw depth
    aligned to RGB, normally uint16 millimetres; timestamp_ns is nanoseconds.
    Return type: immutable dataclass with read-only image arrays.
    """

    cam_id: str
    rgb: np.ndarray
    depth_mm: np.ndarray
    timestamp_ns: int
    intrinsics: CameraIntrinsics
    depth_config: DepthConfig

    def __post_init__(self) -> None:
        if not self.cam_id:
            raise ValueError("cam_id must be non-empty")
        rgb = _readonly_array(self.rgb, ndim=3, dtype=np.uint8, name="rgb", finite=False)
        if rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
        depth = _readonly_array(self.depth_mm, ndim=2, dtype=None, name="depth_mm", finite=True)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"rgb/depth shapes must match, got {rgb.shape[:2]} and {depth.shape}")
        if (self.intrinsics.width, self.intrinsics.height) != (rgb.shape[1], rgb.shape[0]):
            raise ValueError("intrinsics width/height must match aligned RGB-D frame size")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_mm", depth)


@dataclass(frozen=True)
class PerceptionResult:
    """Per-camera perception result in aligned RGB pixel coordinates.

    Input shape/unit/frame: breast_mask is HxW bool/0-1 aligned to RGB/depth;
    nipple_boxes_xyxy is Nx4 pixel boxes; scores are N detector confidences.
    Return type: immutable dataclass with read-only mask/box/score arrays.
    """

    cam_id: str
    breast_mask: np.ndarray
    nipple_boxes_xyxy: np.ndarray
    nipple_scores: np.ndarray
    nipple_labels: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.cam_id:
            raise ValueError("cam_id must be non-empty")
        mask = _readonly_array(self.breast_mask.astype(bool, copy=False), ndim=2, dtype=bool, name="breast_mask")
        boxes = _readonly_array(self.nipple_boxes_xyxy, ndim=2, name="nipple_boxes_xyxy")
        if boxes.shape[1] != 4:
            raise ValueError("nipple_boxes_xyxy must have shape Nx4")
        scores = _readonly_array(self.nipple_scores, ndim=1, name="nipple_scores")
        if scores.shape[0] != boxes.shape[0]:
            raise ValueError("nipple_scores length must match boxes")
        if self.nipple_labels is not None and len(self.nipple_labels) != boxes.shape[0]:
            raise ValueError("nipple_labels length must match boxes")
        object.__setattr__(self, "breast_mask", mask)
        object.__setattr__(self, "nipple_boxes_xyxy", boxes)
        object.__setattr__(self, "nipple_scores", scores)
        if self.nipple_labels is not None:
            object.__setattr__(self, "nipple_labels", list(self.nipple_labels))


@dataclass(frozen=True)
class NippleObservation:
    """Single camera nipple tip observation.

    Input shape/unit/frame: point_world_m is 3-vector in world metres;
    covariance_world_m2 is 3x3 covariance in world metre squared.
    Return type: immutable dataclass with read-only arrays.
    """

    cam_id: str
    side: Side
    point_world_m: np.ndarray
    covariance_world_m2: np.ndarray
    confidence: float
    num_points: int

    def __post_init__(self) -> None:
        if self.side not in {"left", "right", "unknown"}:
            raise ValueError("side must be left, right, or unknown")
        point = _readonly_array(self.point_world_m, shape=(3,), name="point_world_m")
        cov = _readonly_array(self.covariance_world_m2, shape=(3, 3), name="covariance_world_m2")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        if self.num_points < 0:
            raise ValueError("num_points must be non-negative")
        object.__setattr__(self, "point_world_m", point)
        object.__setattr__(self, "covariance_world_m2", cov)


@dataclass(frozen=True)
class FusedNipple:
    """Fused left/right nipple estimate.

    Input shape/unit/frame: mean_world_m is 3-vector in world metres;
    covariance_world_m2 is 3x3 covariance in world metre squared.
    Return type: immutable dataclass with read-only arrays.
    """

    side: Literal["left", "right"]
    mean_world_m: np.ndarray
    covariance_world_m2: np.ndarray
    confidence: float
    used_cameras: list[str]
    rejected_cameras: list[str]

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        mean = _readonly_array(self.mean_world_m, shape=(3,), name="mean_world_m")
        cov = _readonly_array(self.covariance_world_m2, shape=(3, 3), name="covariance_world_m2")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "mean_world_m", mean)
        object.__setattr__(self, "covariance_world_m2", cov)
        object.__setattr__(self, "used_cameras", list(self.used_cameras))
        object.__setattr__(self, "rejected_cameras", list(self.rejected_cameras))


@dataclass(frozen=True)
class ReconstructionResult:
    """One reconstructed RGB-D frame set.

    Input shape/unit/frame: fused_cloud is in world metres; nipples are keyed
    by ``left``/``right``; timings are milliseconds.
    Return type: immutable dataclass.
    """

    timestamp_ns: int
    fused_cloud: Any
    nipples: dict[str, FusedNipple]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object.__setattr__(self, "nipples", dict(self.nipples))
        object.__setattr__(self, "timings_ms", dict(self.timings_ms))


@dataclass(frozen=True)
class Nipple3DConfig:
    """Nipple 3D estimation parameters.

    Input shape/unit/frame: eps and distances are metres; left_right_axis_world
    is a 3-vector in world frame.
    Return type: immutable dataclass with read-only axis ndarray.
    """

    dbscan_eps_m: float = 0.005
    min_cluster_points: int = 30
    covariance_jitter_m2: float = 1e-8
    left_right_axis_world: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    mahalanobis_threshold: float = 3.0

    def __post_init__(self) -> None:
        _validate_positive(self.dbscan_eps_m, "dbscan_eps_m")
        if self.min_cluster_points <= 0:
            raise ValueError("min_cluster_points must be positive")
        _validate_positive(self.covariance_jitter_m2, "covariance_jitter_m2")
        axis = _readonly_array(self.left_right_axis_world, shape=(3,), name="left_right_axis_world")
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            raise ValueError("left_right_axis_world must be non-zero")
        object.__setattr__(self, "left_right_axis_world", _readonly_array(axis / norm, shape=(3,), name="left_right_axis_world"))


@dataclass(frozen=True)
class FusionConfig:
    """Point cloud fusion parameters.

    Input shape/unit/frame: voxel size and outlier radius are metres in world frame.
    Return type: immutable dataclass.
    """

    voxel_size_m: float = 0.002
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    estimate_normals: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.voxel_size_m, "voxel_size_m")
        if self.outlier_nb_neighbors <= 0:
            raise ValueError("outlier_nb_neighbors must be positive")
        _validate_positive(self.outlier_std_ratio, "outlier_std_ratio")


@dataclass(frozen=True)
class CloudCleanupConfig:
    """Per-camera mask/depth cleanup before back-projection.

    Input shape/unit/frame: erosion is pixels on the aligned RGB/depth grid;
    depth edge threshold and depth range are metres in camera frame.
    Return type: immutable dataclass.
    """

    mask_erode_px: int = 4
    flying_depth_edge_threshold_m: float = 0.015
    min_depth_m: float = 0.25
    max_depth_m: float = 1.2

    def __post_init__(self) -> None:
        if self.mask_erode_px < 0:
            raise ValueError("mask_erode_px must be non-negative")
        if self.flying_depth_edge_threshold_m < 0:
            raise ValueError("flying_depth_edge_threshold_m must be non-negative")
        _validate_positive(self.min_depth_m, "min_depth_m")
        _validate_positive(self.max_depth_m, "max_depth_m")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")


@dataclass(frozen=True)
class NippleGuidedFusionConfig:
    """Anatomy-guided multi-view fusion decisions.

    Input shape/unit/frame: distances are metres. The nipple gate controls how
    far a per-view 3D nipple observation may be from the fused anatomical anchor
    before the view is down-weighted or rejected.
    Return type: immutable dataclass.
    """

    enabled: bool = True
    nipple_gate_m: float = 0.05
    reject_inconsistent_views: bool = True
    missing_nipple_weight: float = 0.55
    min_consistent_weight: float = 0.35
    min_accepted_views: int = 1
    random_seed: int = 11

    def __post_init__(self) -> None:
        _validate_positive(self.nipple_gate_m, "nipple_gate_m")
        for name in ("missing_nipple_weight", "min_consistent_weight"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_accepted_views <= 0:
            raise ValueError("min_accepted_views must be positive")


@dataclass(frozen=True)
class AnatomyRegistrationConfig:
    """Online anatomy-guided extrinsic refinement.

    Input shape/unit/frame: correction transforms are small world-frame
    ``Delta T_world_cam`` updates composed before calibrated extrinsics.
    Return type: immutable dataclass.
    """

    mode: Literal["off", "translation_only", "nipple_init_robust_icp"] = "nipple_init_robust_icp"
    reference_camera: str | None = "cam1"
    min_nipple_confidence: float = 0.15
    side_gate_margin_m: float = 0.005
    max_translation_m: float = 0.04
    max_rotation_deg: float = 8.0
    residual_improvement_ratio: float = 1.05
    smooth_corrections: bool = True
    correction_alpha: float = 0.35
    freeze_after_frames: int = 20
    recheck_interval_frames: int = 30
    freeze_translation_epsilon_m: float = 0.001
    freeze_rotation_epsilon_deg: float = 0.25
    icp_voxel_size_m: float = 0.006
    icp_correspondence_m: tuple[float, ...] = (0.05, 0.035, 0.025)
    icp_max_iterations_cold: int = 20
    icp_max_iterations_warm: int = 3
    icp_min_points: int = 250
    icp_min_fitness: float = 0.25
    icp_max_rmse_m: float = 0.012
    robust_kernel: Literal["none", "huber", "tukey"] = "tukey"
    robust_kernel_k_m: float = 0.01

    def __post_init__(self) -> None:
        if self.mode not in {"off", "translation_only", "nipple_init_robust_icp"}:
            raise ValueError("mode must be off, translation_only, or nipple_init_robust_icp")
        if not (0.0 <= self.min_nipple_confidence <= 1.0):
            raise ValueError("min_nipple_confidence must be in [0, 1]")
        if self.side_gate_margin_m < 0:
            raise ValueError("side_gate_margin_m must be non-negative")
        _validate_positive(self.max_translation_m, "max_translation_m")
        _validate_positive(self.max_rotation_deg, "max_rotation_deg")
        _validate_positive(self.residual_improvement_ratio, "residual_improvement_ratio")
        if not (0.0 <= self.correction_alpha <= 1.0):
            raise ValueError("correction_alpha must be in [0, 1]")
        if self.freeze_after_frames < 0:
            raise ValueError("freeze_after_frames must be non-negative")
        if self.recheck_interval_frames <= 0:
            raise ValueError("recheck_interval_frames must be positive")
        _validate_positive(self.freeze_translation_epsilon_m, "freeze_translation_epsilon_m")
        _validate_positive(self.freeze_rotation_epsilon_deg, "freeze_rotation_epsilon_deg")
        _validate_positive(self.icp_voxel_size_m, "icp_voxel_size_m")
        if not self.icp_correspondence_m:
            raise ValueError("icp_correspondence_m must not be empty")
        for value in self.icp_correspondence_m:
            _validate_positive(float(value), "icp_correspondence_m")
        if self.icp_max_iterations_cold <= 0 or self.icp_max_iterations_warm <= 0:
            raise ValueError("icp max iterations must be positive")
        if self.icp_min_points <= 0:
            raise ValueError("icp_min_points must be positive")
        if not (0.0 <= self.icp_min_fitness <= 1.0):
            raise ValueError("icp_min_fitness must be in [0, 1]")
        _validate_positive(self.icp_max_rmse_m, "icp_max_rmse_m")
        if self.robust_kernel not in {"none", "huber", "tukey"}:
            raise ValueError("robust_kernel must be none, huber, or tukey")
        _validate_positive(self.robust_kernel_k_m, "robust_kernel_k_m")
        object.__setattr__(self, "icp_correspondence_m", tuple(float(v) for v in self.icp_correspondence_m))


@dataclass(frozen=True)
class PostFusionCleanupConfig:
    """Fused-cloud cleanup after transform/merge/voxel/statistical fusion."""

    enabled: bool = True
    radius_outlier_enabled: bool = True
    radius_m: float = 0.012
    min_neighbors: int = 8
    min_retained_ratio: float = 0.25
    dbscan_enabled: bool = True
    dbscan_eps_m: float = 0.012
    dbscan_min_points: int = 40
    nipple_anchor_keep_radius_m: float = 0.035
    min_cluster_points: int = 800
    adjacency_distance_m: float = 0.02
    preserve_nipple_clusters: bool = True

    def __post_init__(self) -> None:
        _validate_positive(self.radius_m, "radius_m")
        if self.min_neighbors <= 0:
            raise ValueError("min_neighbors must be positive")
        if not (0.0 <= self.min_retained_ratio <= 1.0):
            raise ValueError("min_retained_ratio must be in [0, 1]")
        _validate_positive(self.dbscan_eps_m, "dbscan_eps_m")
        if self.dbscan_min_points <= 0:
            raise ValueError("dbscan_min_points must be positive")
        _validate_positive(self.nipple_anchor_keep_radius_m, "nipple_anchor_keep_radius_m")
        if self.min_cluster_points <= 0:
            raise ValueError("min_cluster_points must be positive")
        if self.adjacency_distance_m < 0:
            raise ValueError("adjacency_distance_m must be non-negative")


@dataclass(frozen=True)
class TSDFConfig:
    """Optional final-quality TSDF integration path."""

    enabled: bool = False
    voxel_length_m: float = 0.004
    sdf_trunc_m: float = 0.025
    depth_trunc_m: float = 1.2
    color: bool = True

    def __post_init__(self) -> None:
        _validate_positive(self.voxel_length_m, "voxel_length_m")
        _validate_positive(self.sdf_trunc_m, "sdf_trunc_m")
        _validate_positive(self.depth_trunc_m, "depth_trunc_m")


@dataclass(frozen=True)
class QualityMetricsConfig:
    """Objective online diagnostics for A/B reconstruction comparisons."""

    enabled: bool = True
    surface_voxel_size_m: float = 0.008
    surface_max_distance_m: float = 0.025
    min_points: int = 80
    max_points_per_cloud: int = 1500
    temporal_window: int = 30

    def __post_init__(self) -> None:
        _validate_positive(self.surface_voxel_size_m, "surface_voxel_size_m")
        _validate_positive(self.surface_max_distance_m, "surface_max_distance_m")
        if self.min_points <= 0:
            raise ValueError("min_points must be positive")
        if self.max_points_per_cloud <= 0:
            raise ValueError("max_points_per_cloud must be positive")
        if self.temporal_window <= 1:
            raise ValueError("temporal_window must be greater than 1")


@dataclass(frozen=True)
class SurfaceConfig:
    """Optional world-frame surface smoothing after multi-camera fusion.

    Input shape/unit/frame: normal search radius is metres; Poisson depth,
    density trim quantile, and sample count are Open3D surface parameters.
    Return type: immutable dataclass.
    """

    mesh_enabled: bool = False
    poisson_depth: int = 8
    density_trim_quantile: float = 0.1
    sample_points: int = 80000
    normal_radius_m: float = 0.01
    normal_max_nn: int = 30
    min_points: int = 500

    def __post_init__(self) -> None:
        if self.poisson_depth <= 0:
            raise ValueError("poisson_depth must be positive")
        if not (0.0 <= self.density_trim_quantile < 1.0):
            raise ValueError("density_trim_quantile must be in [0, 1)")
        if self.sample_points <= 0:
            raise ValueError("sample_points must be positive")
        _validate_positive(self.normal_radius_m, "normal_radius_m")
        if self.normal_max_nn <= 0:
            raise ValueError("normal_max_nn must be positive")
        if self.min_points < 0:
            raise ValueError("min_points must be non-negative")


@dataclass(frozen=True)
class VisualizationConfig:
    """Browser/Rerun visualization payload controls."""

    rerun_log_cameras: bool = False
    uniform_breast_color: tuple[float, float, float] | None = None
    uniform_areola_color: tuple[float, float, float] | None = None
    areola_radius_m: float = 0.02

    def __post_init__(self) -> None:
        for name in ("uniform_breast_color", "uniform_areola_color"):
            color = getattr(self, name)
            if color is not None and (len(color) != 3 or not all(0.0 <= float(c) <= 1.0 for c in color)):
                raise ValueError(f"{name} must be 3 RGB floats in [0, 1]")
        if self.areola_radius_m < 0.0:
            raise ValueError("areola_radius_m must be non-negative")


@dataclass(frozen=True)
class PerceptionConfig:
    """Existing MammaSegNet adapter configuration.

    Input shape/unit/frame: paths are local filesystem paths; score threshold
    is unitless detector confidence.
    Return type: immutable dataclass.
    """

    model_config_path: Path = Path("configs/breastseg/breastseg_b2_inplace.py")
    checkpoint_path: Path = Path("checkpoint/breastseg.pth")
    device: str = "cuda"
    score_thr: float = 0.3
    topk: int = 2
    nms_iou: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_config_path", Path(self.model_config_path))
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if not (0.0 <= self.score_thr <= 1.0):
            raise ValueError("score_thr must be in [0, 1]")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if not (0.0 <= self.nms_iou <= 1.0):
            raise ValueError("nms_iou must be in [0, 1]")


@dataclass(frozen=True)
class Recon3DConfig:
    """Top-level RGB-D reconstruction configuration.

    Input shape/unit/frame: output_dir is local filesystem; sync threshold is
    milliseconds across camera timestamps.
    Return type: immutable dataclass.
    """

    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    cloud_cleanup: CloudCleanupConfig = field(default_factory=CloudCleanupConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    nipple3d: Nipple3DConfig = field(default_factory=Nipple3DConfig)
    nipple_guided_fusion: NippleGuidedFusionConfig = field(default_factory=NippleGuidedFusionConfig)
    anatomy_registration: AnatomyRegistrationConfig = field(default_factory=AnatomyRegistrationConfig)
    post_fusion_cleanup: PostFusionCleanupConfig = field(default_factory=PostFusionCleanupConfig)
    tsdf_fusion: TSDFConfig = field(default_factory=TSDFConfig)
    quality_metrics: QualityMetricsConfig = field(default_factory=QualityMetricsConfig)
    output_dir: Path | None = None
    sync_max_drift_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        _validate_positive(self.sync_max_drift_ms, "sync_max_drift_ms")
