"""Record RGB-D with the existing recorder and a shared SDK timestamp clock."""
from __future__ import annotations

import sys


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
    keep_alive = enable_common_clock()
    from anatomy_guided_recon3d.cli import main as record_main
    sys.argv.insert(1, "record")
    record_main()


if __name__ == "__main__":
    main()
