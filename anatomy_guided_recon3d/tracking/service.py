"""Small process manager used by the existing local viewer server."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from anatomy_guided_recon3d.tracking.runtime import check_paths, read_settings, write_json


_lock = threading.RLock()
_preview_process = None
_tracking_process = None
_playback_process = None


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _paths(root: Path) -> tuple[Path | None, Path | None]:
    current = _load(root / "runs/point_tracking/current.json")
    return tuple(root / current[key] if current.get(key) else None for key in ("preview", "tracking"))


def _remember(root: Path, key: str, output: Path) -> None:
    path = root / "runs/point_tracking/current.json"
    current = _load(path)
    current[key] = output.relative_to(root).as_posix()
    write_json(path, current)


def _spawn(root: Path, command: list[str], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (output / "worker.log").open("ab") as log:
        return subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)


def preview_running(root: Path) -> bool:
    path, _ = _paths(root)
    if path is None:
        return False
    payload = _load(path / "status.json")
    if _preview_process is not None:
        return _preview_process.poll() is None
    return payload.get("state") in ("starting", "preview", "recording") and time.time() - payload.get("updated_at", 0) < 30


def get_status(root: Path) -> dict:
    with _lock:
        preview, tracking = _paths(root)
        playback_name = _load(root / "runs/point_tracking/current.json").get("playback")
        playback = root / playback_name if playback_name else None
        payload = {"ok": True, "preview": {}, "tracking": {}, "playback": {}, "features": {"foreground_control": True, "rgbd_playback": True}}
        for key, output, process in (("preview", preview, _preview_process), ("tracking", tracking, _tracking_process),
                                     ("playback", playback, _playback_process)):
            if output is None:
                continue
            value = _load(output / "status.json")
            value["base"] = "/" + output.relative_to(root).as_posix() + "/"
            value["running"] = process.poll() is None if process is not None else key == "preview" and preview_running(root)
            if process is not None and process.poll() not in (None, 0) and value.get("state") != "failed":
                value.update(state="failed", message="The worker stopped; see its log for details")
            payload[key] = value
        return payload


def start_preview(root: Path) -> dict:
    global _preview_process
    with _lock:
        if preview_running(root):
            return get_status(root)
        settings = read_settings()
        if not Path(settings["python"]).is_file():
            raise FileNotFoundError("The recording Python environment is missing. See docs/POINT_TRACKING.md")
        output = root / "runs/point_tracking" / ("preview-" + uuid.uuid4().hex[:10])
        _remember(root, "preview", output)
        write_json(output / "status.json", dict(state="starting", message="Starting both calibrated cameras", updated_at=time.time()))
        _preview_process = _spawn(root, [settings["python"], "-m", "anatomy_guided_recon3d.tracking.recording", "--out", str(output)], output)
        return get_status(root)


def preview_command(root: Path, payload: dict) -> dict:
    with _lock:
        output, _ = _paths(root)
        if output is None or not preview_running(root):
            raise ValueError("Start the camera preview first")
        action = payload.get("action")
        if action not in ("record", "stop", "range"):
            raise ValueError("Unsupported recording action")
        if action == "record" and _load(output / "status.json").get("state") != "preview":
            raise ValueError("Wait for the calibrated preview before recording")
        command = _load(output / "command.json")
        # Display changes must not replace an unconsumed record/stop request.
        # The worker reads display fields continuously; no new action is needed.
        if action != "range":
            command.update(id=uuid.uuid4().hex, action=action)
        if action == "record":
            duration = float(payload.get("duration_s", 6))
            if not 1 <= duration <= 60:
                raise ValueError("Record between 1 and 60 seconds")
            command["duration_s"] = duration
        if action == "range":
            near, far = float(payload.get("min_depth_m", .3)), float(payload.get("max_depth_m", 1.0))
            if not .1 <= near < far <= 3:
                raise ValueError("Choose a depth range between 0.1 and 3 metres")
            command.update(min_depth_m=near, max_depth_m=far)
            if "foreground_only" in payload:
                command["foreground_only"] = bool(payload["foreground_only"])
                command["scope"] = "breast" if payload["foreground_only"] else "scene"
            if "scope" in payload:
                if payload["scope"] not in ("breast_area", "breast", "scene"):
                    raise ValueError("Choose full breast area, detected breast, or whole scene")
                command["scope"] = payload["scope"]
                command["foreground_only"] = payload["scope"] != "scene"
            if "detail" in payload:
                if payload["detail"] not in ("fast", "balanced", "fine"):
                    raise ValueError("Choose fast, balanced, or fine 3D detail")
                command["detail"] = payload["detail"]
            if "mask_overlay" in payload:
                command["mask_overlay"] = bool(payload["mask_overlay"])
        write_json(output / "command.json", command)
        return get_status(root)


def start_tracking(root: Path, session: Path, camera: str | None = None) -> dict:
    global _tracking_process
    with _lock:
        if _tracking_process is not None and _tracking_process.poll() is None:
            raise ValueError("A tracking job is already running")
        settings = read_settings()
        check_paths(settings)
        output = root / "runs/point_tracking" / ("tracks-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        _remember(root, "tracking", output)
        write_json(output / "status.json", dict(state="preparing", message="Reading the recorded RGB-D frames", updated_at=time.time()))
        command = [settings["python"], "-m", "anatomy_guided_recon3d.tracking.cli", "--session", str(session), "--out", str(output)]
        if camera:
            command += ["--camera", str(camera)]
        _tracking_process = _spawn(root, command, output)
        return get_status(root)


def start_playback(root: Path, session: Path) -> dict:
    global _playback_process
    with _lock:
        if _playback_process is not None and _playback_process.poll() is None:
            raise ValueError("A recording is already being prepared for playback")
        settings = read_settings()
        if not Path(settings["python"]).is_file():
            raise FileNotFoundError("The RGB-D Python environment is missing")
        output = root / "runs/point_tracking" / ("playback-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        _remember(root, "playback", output)
        write_json(output / "status.json", dict(state="preparing", message="Preparing recorded RGB-D playback", updated_at=time.time()))
        _playback_process = _spawn(root, [settings["python"], "-m", "anatomy_guided_recon3d.tracking.playback",
                                  "--session", str(session), "--out", str(output)], output)
        return get_status(root)
