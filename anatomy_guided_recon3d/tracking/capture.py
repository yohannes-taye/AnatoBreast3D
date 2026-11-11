"""Record RGB-D with the existing recorder and a shared SDK timestamp clock."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import shutil


def enable_common_clock() -> list:
    from anatomy_guided_recon3d.sensors.femto_bolt import _get_context

    context = _get_context()
    devices = context.query_devices()
    keep_alive = []
    for index in range(devices.get_count()):
        device = devices.get_device_by_index(index)
        if not device.is_global_timestamp_supported():
            raise RuntimeError("This camera does not support the shared SDK timestamp clock")
        device.enable_global_timestamp(True)
        keep_alive.append(device)
    if not keep_alive:
        raise RuntimeError("No RGB-D cameras are connected")
    context.enable_multi_device_sync(60000)
    return keep_alive


def main() -> None:
    import yaml
    from anatomy_guided_recon3d.tracking.runtime import capture_config

    from anatomy_guided_recon3d.cli import main as record_main

    arguments = sys.argv[1:]
    if "--help" in arguments or "-h" in arguments:
        sys.argv = [sys.argv[0], "record", "--help"]
        record_main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=capture_config())
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args(arguments)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    calibration = Path(config["extrinsics_path"])
    if not calibration.is_file():
        raise FileNotFoundError("Recording requires your saved rig calibration: "+str(calibration))
    args.out.mkdir(parents=True,exist_ok=True)
    snapshot=args.out/"extrinsics.json"
    if snapshot.exists() and snapshot.read_bytes()!=calibration.read_bytes():
        raise FileExistsError("This output folder contains a different calibration; choose a new recording folder")
    if calibration.resolve()!=snapshot.resolve():
        shutil.copy2(calibration,snapshot)
    keep_alive = enable_common_clock()
    # Pass the resolved default explicitly: the base recorder has its own default.
    if not any(value == "--config" or value.startswith("--config=") for value in arguments):
        arguments = ["--config", str(args.config), *arguments]
    sys.argv = [sys.argv[0], "record", *arguments]
    record_main()


if __name__ == "__main__":
    main()
