"""Multi-distance depth correction calibration for Femto Bolt cameras.

The Charuco board (caliper-confirmed 50 mm squares) provides an independent metric
truth: solvePnP on the board gives each corner's true camera-frame depth Z without
using the depth sensor at all. The ToF depth gives the measured Z at the same pixel.
Collecting (z_measured, z_true) pairs at several distances lets us fit, per camera,

    z_true = a * z_measured + b        (scale a + fixed offset b)

which a single-distance capture cannot separate.

Workflow
--------
1. Put the board where a camera sees it, run ``capture`` (repeat ~5-8 times with the
   board at different distances over ~0.4-0.7 m and a few tilts). Each run appends
   pairs to the accumulator JSON.

       python tools/calibrate_depth.py capture

2. Solve and write the per-camera correction:

       python tools/calibrate_depth.py fit

The result is written to configs/recon3d/depth_correction.json. ``fit`` also reports
whether a pure-scale model (b ~ 0, which fits DepthConfig.depth_scale_m directly) is
enough, or whether the offset b is needed.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon3d.calibration.charuco import (  # noqa: E402
    CharucoBoardSpec,
    detect_charuco,
    estimate_board_pose,
    make_charuco_board,
)
from recon3d.sensors.femto_bolt import FemtoBoltCamera  # noqa: E402

LOGGER = logging.getLogger("calibrate_depth")

# Defaults for this workstation; override with --serials / --cameras.
DEFAULT_CAMERAS = {
    "cam0": "CL8T754004D",
    "cam1": "CL8T7540197",
    "cam2": "CL8T75400XA",
}


def _board(args: argparse.Namespace) -> object:
    dict_id = getattr(cv2.aruco, args.dictionary)
    return make_charuco_board(CharucoBoardSpec(5, 5, args.square_length_m, args.marker_length_m, dict_id))


def _sample_depth_m(depth: np.ndarray, u: float, v: float, scale_m: float, win: int = 4) -> float | None:
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(0, vi - win):min(h, vi + win + 1), max(0, ui - win):min(w, ui + win + 1)].astype(np.float64)
    patch = patch[patch > 0]
    return float(np.median(patch)) * scale_m if patch.size >= 6 else None


def _cameras(args: argparse.Namespace) -> dict[str, str]:
    if args.cameras and args.serials:
        if len(args.cameras) != len(args.serials):
            raise SystemExit("--cameras and --serials must have equal length")
        return dict(zip(args.cameras, args.serials))
    return dict(DEFAULT_CAMERAS)


def cmd_capture(args: argparse.Namespace) -> None:
    board = _board(args)
    accum = Path(args.accum)
    accum.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = json.loads(accum.read_text(encoding="utf-8")) if accum.is_file() else []

    new_pairs = 0
    for cam_id, serial in _cameras(args).items():
        cam = FemtoBoltCamera(cam_id, serial_number=serial, align_mode=args.align_mode)
        try:
            cam.start()
            for _ in range(args.settle):
                cam.read_aligned_frame()
            frame = cam.read_aligned_frame()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s: capture failed: %r", cam_id, exc)
            cam.stop()
            continue
        cam.stop()

        det = detect_charuco(frame.rgb, board)
        if det is None:
            LOGGER.info("%s: board not detected (skipped)", cam_id)
            continue
        pose = estimate_board_pose(det, board, frame.intrinsics)
        if pose.mean_reprojection_error_px > args.max_reproj_px:
            LOGGER.warning(
                "%s: skipped, reproj %.2f px > %.2f (hold the board still / reduce tilt & blur)",
                cam_id, pose.mean_reprojection_error_px, args.max_reproj_px,
            )
            continue
        obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[det.ids.reshape(-1)]
        homog = np.c_[obj, np.ones(len(obj))]
        cam_true = (pose.T_cam_board @ homog.T).T[:, :3]  # true corner pos in camera frame
        scale = frame.depth_config.depth_scale_m
        n = 0
        for i, px in enumerate(det.corners_px.reshape(-1, 2)):
            z_meas = _sample_depth_m(frame.depth_mm, float(px[0]), float(px[1]), scale)
            if z_meas is None:
                continue
            records.append({"cam": cam_id, "z_meas": z_meas, "z_true": float(cam_true[i, 2])})
            n += 1
            new_pairs += 1
        LOGGER.info("%s: +%d pairs (board ~%.1f cm, reproj %.2f px)", cam_id, n,
                    pose.T_cam_board[2, 3] * 100, pose.mean_reprojection_error_px)

    accum.write_text(json.dumps(records, indent=2), encoding="utf-8")
    # quick per-camera coverage summary
    for cam_id in sorted({r["cam"] for r in records}):
        zs = np.array([r["z_meas"] for r in records if r["cam"] == cam_id])
        LOGGER.info("  %s total pairs=%d  z_meas range=[%.3f, %.3f] m", cam_id, zs.size, zs.min(), zs.max())
    LOGGER.info("Added %d pairs -> %s", new_pairs, accum)


def _fit_line(zm: np.ndarray, zt: np.ndarray, trim_sigma: float = 3.0) -> tuple[float, float, float, int]:
    """Robust least squares z_true = a*z_meas + b with one outlier-trim pass."""
    A = np.c_[zm, np.ones_like(zm)]
    sol, *_ = np.linalg.lstsq(A, zt, rcond=None)
    resid = zt - A @ sol
    keep = np.abs(resid) <= trim_sigma * resid.std() if resid.std() > 0 else np.ones_like(resid, bool)
    A2 = np.c_[zm[keep], np.ones(keep.sum())]
    sol, *_ = np.linalg.lstsq(A2, zt[keep], rcond=None)
    rms = float(np.sqrt(np.mean((zt[keep] - A2 @ sol) ** 2)))
    return float(sol[0]), float(sol[1]), rms, int(keep.sum())


def cmd_fit(args: argparse.Namespace) -> None:
    records = json.loads(Path(args.accum).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for cam_id in sorted({r["cam"] for r in records}):
        zm = np.array([r["z_meas"] for r in records if r["cam"] == cam_id])
        zt = np.array([r["z_true"] for r in records if r["cam"] == cam_id])
        if zm.size < args.min_pairs:
            LOGGER.warning("%s: only %d pairs (< %d); capture more distances", cam_id, zm.size, args.min_pairs)
            continue
        a, b, rms, n = _fit_line(zm, zt)
        # scale-only model (b = 0): a0 = sum(zt*zm)/sum(zm^2)
        a0 = float(np.dot(zt, zm) / np.dot(zm, zm))
        rms0 = float(np.sqrt(np.mean((zt - a0 * zm) ** 2)))
        span = float(zm.max() - zm.min())
        LOGGER.info(
            "%s: n=%d z_meas=[%.3f,%.3f]m  |  scale+offset: a=%.4f b=%+.4fm rms=%.1fmm  |  "
            "scale-only: a=%.4f rms=%.1fmm",
            cam_id, n, zm.min(), zm.max(), a, b, rms * 1000, a0, rms0 * 1000,
        )
        out[cam_id] = {
            "model": "z_true = a*z_meas + b  (metres)",
            "a": a, "b_m": b, "rms_mm": rms * 1000,
            "scale_only_a": a0, "scale_only_rms_mm": rms0 * 1000,
            "n_pairs": n, "z_meas_span_m": span,
        }

    if not out:
        raise SystemExit("No camera had enough pairs. Run more `capture` passes at varied distances.")

    # Recommendation: pure scale is enough if |b| small and offset barely helps.
    for cam_id, r in out.items():
        offset_helps = (abs(r["b_m"]) > 0.003) and (r["scale_only_rms_mm"] - r["rms_mm"] > 1.0)
        r["recommended"] = "scale+offset" if offset_helps else "scale-only"
        if any(r["z_meas_span_m"] < 0.08 for r in out.values()):
            r["warning"] = "z range < 8cm; offset estimate is weak — capture wider distance spread"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", args.out)
    LOGGER.info("Recommendation per camera: %s", {c: r["recommended"] for c, r in out.items()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-distance Femto Bolt depth correction calibration")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--accum", default="runs/depth_calib/pairs.json")
        sp.add_argument("--square-length-m", type=float, default=0.050)
        sp.add_argument("--marker-length-m", type=float, default=0.040)
        sp.add_argument("--dictionary", default="DICT_7X7_50")

    cap = sub.add_parser("capture", help="Capture one set and append (z_meas, z_true) pairs")
    common(cap)
    cap.add_argument("--cameras", nargs="*", default=None)
    cap.add_argument("--serials", nargs="*", default=None)
    cap.add_argument("--align-mode", choices=["hw", "sw"], default="sw")
    cap.add_argument("--settle", type=int, default=8)
    cap.add_argument("--max-reproj-px", type=float, default=2.0,
                     help="Skip a camera's frame if board PnP reprojection exceeds this.")
    cap.set_defaults(func=cmd_capture)

    fit = sub.add_parser("fit", help="Fit per-camera depth correction and write JSON")
    common(fit)
    fit.add_argument("--out", default="configs/recon3d/depth_correction.json")
    fit.add_argument("--min-pairs", type=int, default=30)
    fit.set_defaults(func=cmd_fit)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
