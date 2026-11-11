"""Check recording/tracking prerequisites without opening camera streams."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from anatomy_guided_recon3d.tracking.runtime import capture_config, read_settings
    problems=[]
    for package in ["numpy","opencv-contrib-python","torch","torchvision","open3d","pyorbbecsdk2","hydra-core","python-box","av"]:
        try:print(f"OK {package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:problems.append(f"Install {package}")
    try:
        import torch
        if torch.cuda.is_available():print("OK GPU:",torch.cuda.get_device_name())
        else:problems.append("CUDA is not available. Check the NVIDIA driver and the PyTorch installation")
    except ImportError:pass
    import yaml
    profile=capture_config()
    if not profile.exists():problems.append("Create a local camera profile with configure_tracking_capture.py")
    else:
        config=yaml.safe_load(profile.read_text())
        print("Camera profile:",profile)
        calibration=ROOT/config["extrinsics_path"]
        if not calibration.exists():problems.append("Supply your own calibrated extrinsics and run configure_tracking_capture.py")
        else:
            data=json.loads(calibration.read_text())
            missing={item["id"] for item in config["hardware"]["cameras"]}-set(data["cameras"])
            if missing:problems.append("Calibration is missing camera IDs: "+", ".join(sorted(missing)))
            else:print("OK saved extrinsics:",calibration)
        segmentation=ROOT/config["perception"]["checkpoint_path"]
        if not segmentation.is_file():problems.append("Place your trained breastseg.pth at "+str(segmentation))
        else:print("OK breast segmentation checkpoint")
    settings=read_settings()
    for name in ["upstream_repo","checkpoint","python"]:
        if not Path(settings[name]).exists():problems.append("Missing "+name+": "+settings[name])
        else:print("OK "+name)
    for problem in problems:print("NEEDS SETUP:",problem)
    if problems:raise SystemExit(1)
    print("Ready. Run python tools/serve_recon3d_viewer.py --port 8766 --no-clean-existing")


if __name__=="__main__":main()
