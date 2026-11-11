"""Create a local recording profile from the user's calibrated camera rig."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cameras",action="store_true")
    parser.add_argument("--extrinsics",type=Path)
    parser.add_argument("--camera",action="append",default=[],metavar="ID=SERIAL")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if args.list_cameras:
        import pyorbbecsdk as sdk
        context=sdk.Context()
        devices=context.query_devices()
        for i in range(devices.get_count()):
            info=devices.get_device_by_index(i).get_device_info()
            print(info.get_name(),info.get_serial_number())
        if not args.extrinsics:
            return
    if not args.extrinsics:
        parser.error("Provide --extrinsics path/to/your/calibration.json")
    from anatomy_guided_recon3d.calibration.io import load_extrinsics
    calibrated=load_extrinsics(args.extrinsics)
    source=json.loads(args.extrinsics.read_text(encoding="utf-8"))
    if source.get("length_unit","meters") not in ("m","meter","meters","metre","metres"):
        parser.error("Extrinsic translations must be stored in metres")
    cameras={name:entry.get("serial_number") for name,entry in source["cameras"].items()}
    if args.camera:
        cameras={}
        for value in args.camera:
            if "=" not in value:
                parser.error("Use --camera cam1=SERIAL_NUMBER")
            name,serial=value.split("=",1)
            if not name or not serial:
                parser.error("Camera ID and serial must be nonempty")
            cameras[name]=serial
    if not cameras or set(cameras)-set(calibrated) or not all(cameras.values()):
        parser.error("Supply --camera ID=SERIAL for each selected camera, using the IDs in your calibration")
    if len(set(cameras.values()))!=len(cameras):
        parser.error("Each camera must have a different physical serial number")
    profile=ROOT/"configs/recon3d/tracking_capture.local.yaml"
    snapshot=ROOT/"configs/recon3d/extrinsics.local.json"
    if (profile.exists() or snapshot.exists()) and not args.overwrite:
        parser.error("Local rig settings exist. Use --overwrite only when replacing them intentionally")
    profile.parent.mkdir(exist_ok=True)
    if args.extrinsics.resolve()!=snapshot.resolve():
        shutil.copy2(args.extrinsics,snapshot)
    config=yaml.safe_load((ROOT/"configs/recon3d/tracking_capture.yaml").read_text())
    config["extrinsics_path"]=snapshot.relative_to(ROOT).as_posix()
    config["hardware"]["cameras"]=[dict(id=name,serial_number=serial,sync_mode="standalone") for name,serial in cameras.items()]
    profile.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    print("Saved",profile)
    print("The recorder will use your local calibration and camera serials automatically.")


if __name__=="__main__":
    main()
