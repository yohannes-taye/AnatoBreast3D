"""Set up the optional recorded TAPIP3D tracker in the current Python environment."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT/"third_party/TAPIP3D")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoint/tapip3d_final.pth")
    parser.add_argument("--skip-deps", action="store_true", help="Reuse an already configured application environment")
    parser.add_argument("--skip-downloads", action="store_true", help="Verify existing source and checkpoint only")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if sys.version_info[:2] != (3,11):
        raise SystemExit("Use Python 3.11 for the pinned and tested application dependencies.")
    if not args.skip_deps:
        subprocess.run([sys.executable,"-m","pip","install","-r",str(ROOT/"requirements.txt")],check=True)
        subprocess.run([sys.executable,"-m","pip","install","-r",str(ROOT/"requirements-tracking.txt"),"pyorbbecsdk2==2.1.3"],check=True)
    source = args.upstream.resolve()
    if not source.exists() and not args.skip_downloads:
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["git","clone","--filter=blob:none","--no-checkout","https://github.com/zbw001/TAPIP3D.git",str(source)],check=True)
        subprocess.run(["git","checkout","--detach",REVISION],cwd=source,check=True)
    if not (source/"models/point_tracker_3d.py").is_file():
        raise SystemExit("TAPIP3D source is missing; rerun without --skip-downloads.")
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Use a dedicated TAPIP3D checkout at {REVISION}; the supplied checkout was left unchanged.")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists() and not args.skip_downloads:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temporary = checkpoint.with_suffix(".download")
        print("Downloading the authors' TAPIP3D checkpoint (309 MB)…",flush=True)
        urllib.request.urlretrieve("https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth?download=true",temporary)
        with temporary.open("rb") as handle:
            if hashlib.file_digest(handle,"sha256").hexdigest() != CHECKPOINT_SHA256:
                raise SystemExit("Checkpoint hash did not match the tested release; the existing checkpoint was not replaced.")
        temporary.replace(checkpoint)
    if not checkpoint.is_file():
        raise SystemExit("The checkpoint is missing; rerun without --skip-downloads.")
    with checkpoint.open("rb") as handle:
        if hashlib.file_digest(handle,"sha256").hexdigest() != CHECKPOINT_SHA256:
            raise SystemExit("This is not the tested released checkpoint. Existing files were left unchanged.")
    import torch
    import pyorbbecsdk
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the NVIDIA driver, or use --device cpu for debugging.")
    settings=json.loads((ROOT/"configs/tracking/tapip3d.json").read_text())
    settings.update(upstream_repo=str(source),checkpoint=str(checkpoint),python=sys.executable,
                    device=args.device,encoder_chunk_size=8)
    destination=ROOT/"configs/tracking/tapip3d.local.json"
    destination.write_text(json.dumps(settings,indent=2),encoding="utf-8")
    print("Configured:",destination)
    print("Torch:",torch.__version__,"; device:",args.device)
    print("Place your trained breastseg.pth in checkpoint/, or supply a first-frame --mask when tracking.")


if __name__ == "__main__":
    main()
