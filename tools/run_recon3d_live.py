from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _require_formal_inputs(config_path: Path) -> None:
    raw = _load_yaml(config_path)
    extrinsics_path = Path(raw.get("extrinsics_path", "configs/recon3d/extrinsics.json"))
    if not extrinsics_path.is_file():
        raise FileNotFoundError(
            "Formal three-camera reconstruction needs calibrated extrinsics. "
            f"Missing file: {extrinsics_path}"
        )
    hardware = raw.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("Config must contain a hardware mapping")
    cameras = hardware.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 3:
        raise ValueError("Formal live run expects exactly three hardware cameras in hardware.cameras")
    missing = [
        str(entry.get("id", f"cam{index}"))
        for index, entry in enumerate(cameras)
        if not isinstance(entry, dict) or not entry.get("serial_number")
    ]
    if missing:
        raise ValueError(f"Missing serial_number for cameras: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal three-camera Recon3D pipeline.")
    parser.add_argument("--config", default="configs/recon3d/default.yaml")
    parser.add_argument("--out", default="runs/recon3d_live")
    parser.add_argument("--once", action="store_true", help="Capture and reconstruct one synchronized frame set.")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish live_manifest.json and the embedded rerun stream for viewer/live.html.",
    )
    parser.add_argument("--rerun-web-port", type=int, default=9090, help="Port hosting rerun web assets.")
    parser.add_argument("--rerun-grpc-port", type=int, default=9876, help="Port hosting rerun data.")
    parser.add_argument("--rerun-mem", default="25%", help="rerun server memory budget.")
    parser.add_argument("--viewer-rgb-width", type=int, default=640, help="Width of RGB/mask preview images.")
    parser.add_argument("--color-size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--color-fps", type=int, default=None)
    parser.add_argument("--color-format", default=None)
    parser.add_argument("--sync-diagnostics", action="store_true", help="Write FIFO/timestamp diagnostics JSONL.")
    parser.add_argument(
        "--no-sync-config",
        action="store_true",
        help="Do not write Orbbec multi-device sync settings before capture.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    _require_formal_inputs(config_path)

    command = [
        sys.executable,
        "-m",
        "recon3d.cli",
        "run",
        "--config",
        str(config_path),
        "--out",
        args.out,
    ]
    if args.once:
        command.append("--once")
    if args.visualize:
        command.append("--visualize")
        command += ["--rerun-web-port", str(args.rerun_web_port)]
        command += ["--rerun-grpc-port", str(args.rerun_grpc_port)]
        command += ["--rerun-mem", str(args.rerun_mem)]
        command += ["--viewer-rgb-width", str(args.viewer_rgb_width)]
    if args.color_size is not None:
        command += ["--color-size", str(args.color_size[0]), str(args.color_size[1])]
    if args.color_fps is not None:
        command += ["--color-fps", str(args.color_fps)]
    if args.color_format is not None:
        command += ["--color-format", str(args.color_format)]
    if args.sync_diagnostics:
        command.append("--sync-diagnostics")
    if args.no_sync_config:
        command.append("--no-sync-config")

    print("Running formal Recon3D pipeline:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
