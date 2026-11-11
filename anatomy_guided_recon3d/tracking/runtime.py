from __future__ import annotations

import json
import os
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tracking" / "tapip3d.json"


def read_settings(path: Path | None = None) -> dict:
    local = DEFAULT_CONFIG.with_name("tapip3d.local.json")
    settings = json.loads((path or (local if local.exists() else DEFAULT_CONFIG)).read_text(encoding="utf-8"))
    for key in ("upstream_repo", "checkpoint", "python"):
        target = Path(settings[key])
        settings[key] = str((PROJECT_ROOT / target).resolve() if not target.is_absolute() else target)
    if os.environ.get("ANATOBREAST3D_TRACKING_PYTHON"):
        settings["python"] = os.environ["ANATOBREAST3D_TRACKING_PYTHON"]
    return settings


def capture_config() -> Path:
    """Prefer the user's rig profile while retaining the workstation default."""
    if os.environ.get("ANATOBREAST3D_CAPTURE_CONFIG"):
        return Path(os.environ["ANATOBREAST3D_CAPTURE_CONFIG"]).resolve()
    local = PROJECT_ROOT / "configs/recon3d/tracking_capture.local.yaml"
    return local if local.exists() else PROJECT_ROOT / "configs/recon3d/tracking_capture.yaml"


def write_json(path: Path, payload: dict) -> None:
    """Publish a complete JSON document even while the viewer polls it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    replace_file(temporary, path)


def replace_file(source: Path, destination: Path) -> None:
    """Windows readers briefly prevent replacement while serving a response."""
    for attempt in range(40):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.01)


def status(output: Path, state: str, message: str, **fields) -> None:
    write_json(output / "status.json", dict(state=state, message=message, updated_at=time.time(), **fields))


def check_paths(settings: dict) -> None:
    for key, label in (("upstream_repo", "TAPIP3D source"), ("checkpoint", "TAPIP3D checkpoint"), ("python", "Tracking Python")):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f"{label} is missing: {settings[key]}. See docs/POINT_TRACKING.md for setup.")
    if settings["knn_backend"] not in ("torch", "native"):
        raise ValueError("knn_backend must be torch or native")
