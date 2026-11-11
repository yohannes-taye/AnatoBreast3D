"""Install the optional recorder/tracker into the Python environment running this script."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REVISION = "4cb7e69a1687f67d56ec3e506768f51f2c581b46"
CHECKPOINT_SHA256 = "3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e"


def run(*args):
    subprocess.run(list(map(str,args)), cwd=ROOT, check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-base",action="store_true",help="Use an already installed application environment")
    parser.add_argument("--skip-downloads",action="store_true",help="Only install dependencies and write local settings")
    parser.add_argument("--overwrite-settings",action="store_true")
    args=parser.parse_args()
    if sys.version_info[:2] not in ((3,10),(3,11)) or sys.maxsize<=2**32:
        raise SystemExit("Use 64-bit Python 3.11 (3.10 also supported by the pinned dependencies).")
    if sys.prefix==sys.base_prefix:
        raise SystemExit("Create a virtual environment first. Follow docs/RECORDING_SETUP.md, steps 1–2.")
    settings_path=ROOT/"configs/tracking/tapip3d.local.json"
    if settings_path.exists() and not args.overwrite_settings:
        raise SystemExit("Local settings already exist. Use --overwrite-settings only if you intend to replace them.")
    if not args.skip_base:
        run(sys.executable,"-m","pip","install","-r",ROOT/"requirements.txt")
    run(sys.executable,"-m","pip","install","-r",ROOT/"requirements-tracking.txt")
    # The application already supplies NumPy, Open3D, AV and contrib OpenCV.
    # Avoid installing the vendor's second OpenCV wheel and optional GUI demos.
    run(sys.executable,"-m","pip","install","--no-deps","pyorbbecsdk2==2.1.3")
    source=ROOT/"third_party/TAPIP3D"
    checkpoint=ROOT/"checkpoint/tapip3d_final.pth"
    if not args.skip_downloads:
        if not (source/".git").exists():
            source.parent.mkdir(exist_ok=True)
            run("git","clone","https://github.com/zbw001/TAPIP3D.git",source)
        current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
        if current!=REVISION:
            changed=subprocess.check_output(["git","status","--porcelain"],cwd=source,text=True).strip()
            if changed:
                raise SystemExit("The TAPIP3D checkout has local changes; keep them before changing its revision.")
            run("git","-C",source,"fetch","origin",REVISION)
            run("git","-C",source,"checkout","--detach",REVISION)
        checkpoint.parent.mkdir(exist_ok=True)
        if not checkpoint.exists():
            temporary=checkpoint.with_suffix(".download")
            print("Downloading the authors' TAPIP3D weights (about 309 MB)…",flush=True)
            urllib.request.urlretrieve("https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth?download=true",temporary)
            temporary.replace(checkpoint)
        with checkpoint.open("rb") as handle:
            digest=hashlib.file_digest(handle,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(handle.read()).hexdigest()
        if digest!=CHECKPOINT_SHA256:
            raise SystemExit("Checkpoint checksum differs from the tested release. Keep the file and verify the download before proceeding.")
    settings=json.loads((ROOT/"configs/tracking/tapip3d.json").read_text())
    settings.update(upstream_repo=str(source),checkpoint=str(checkpoint),python=sys.executable)
    settings_path.parent.mkdir(exist_ok=True)
    settings_path.write_text(json.dumps(settings,indent=2),encoding="utf-8")
    print("Setup complete. Add checkpoint/breastseg.pth and configure your rig with tools/configure_tracking_capture.py.")


if __name__=="__main__":
    main()
