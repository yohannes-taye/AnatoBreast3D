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
    if args.no_sync_config:
        command.append("--no-sync-config")

    print("Running formal Recon3D pipeline:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
