from __future__ import annotations

import argparse
import atexit
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib import request as urlrequest
from urllib.parse import urlsplit
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.annotation_store import dataset_summary, save_annotation  # noqa: E402
from active_learning.training import latest_status, launch_finetune  # noqa: E402
from anatomy_guided_recon3d.tracking import service as point_tracking  # noqa: E402


DEFAULT_REPLAY_CONFIG = Path("configs") / "recon3d" / "default.yaml"
DEFAULT_LIVE_WEB_PORT = 9090
DEFAULT_LIVE_GRPC_PORT = 9876
DEFAULT_REPLAY_DATA_ROOT = Path("data") / "Replay"
DEFAULT_REPLAY_WEB_PORT = 9111
DEFAULT_REPLAY_GRPC_PORT = 9899
DEFAULT_REPLAY_PROBE_FRAMES = 180
DEFAULT_CLEAN_PORTS = (8765, 9090, 9876, 9091, 9877, DEFAULT_REPLAY_WEB_PORT, DEFAULT_REPLAY_GRPC_PORT)
REPLAY_PIPELINES = {
    "anatomy_guided_recon3d": {
        "label": "Anatomy-guided Recon3D",
        "module": "anatomy_guided_recon3d.cli",
        "out_suffix": "anatomy_guided_replay",
        "live_output": "anatomy_guided_recon3d_live",
    },
    "recon3d": {
        "label": "Recon3D",
        "module": "recon3d.cli",
        "out_suffix": "replay",
        "live_output": "recon3d_live",
    },
}
DEFAULT_REPLAY_PIPELINE = "anatomy_guided_recon3d"

_replay_lock = threading.Lock()
_replay_proc: subprocess.Popen[bytes] | None = None
_replay_log = None
_replay_info: dict[str, object] = {}
_live_lock = threading.Lock()
_live_proc: subprocess.Popen[bytes] | None = None
_live_log = None
_live_info: dict[str, object] = {}


def _stop_viewer_managed_replay(host: str, ports: list[int]) -> None:
    for api_path in ("/api/replay/stop", "/api/live/stop"):
        for port in ports:
            url = f"http://{host}:{port}{api_path}"
            try:
                req = urlrequest.Request(
                    url,
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=0.25):
                    pass
            except Exception:
                pass


def _stop_process(proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)


def _pid_matches_any_port(line: str, ports: set[int]) -> int | None:
    parts = line.split()
    if len(parts) < 5 or parts[0].upper() != "TCP":
        return None
    state = parts[3].upper()
    if state not in {"LISTENING", "ESTABLISHED"}:
        return None
    try:
        local_port = int(parts[1].rsplit(":", 1)[1])
        pid = int(parts[-1])
    except (IndexError, ValueError):
        return None
    if local_port not in ports or pid in {0, os.getpid()}:
        return None
    return pid


def clean_existing_processes(host: str, port: int) -> list[int]:
    ports = set(DEFAULT_CLEAN_PORTS)
    ports.update(range(int(port), int(port) + 20))
    ordered_ports = sorted(ports)
    viewer_api_ports = sorted({int(port), 8765, 8766, 8767, 8768, 8769})
    _stop_viewer_managed_replay(host, viewer_api_ports)
    if os.name != "nt":
        return []

    try:
        result = subprocess.run(  # noqa: S603 - netstat is a fixed diagnostic command
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    pids = {
        pid
        for line in result.stdout.splitlines()
        for pid in [_pid_matches_any_port(line, ports)]
        if pid is not None
    }
    stopped: list[int] = []
    for pid in sorted(pids):
        try:
            subprocess.run(  # noqa: S603 - PID comes from local netstat for known dev ports
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            stopped.append(pid)
        except (OSError, subprocess.SubprocessError):
            continue
    if stopped:
        time.sleep(0.5)
    return stopped


def _relative_id(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _viewer_base_for_output(output_dir: Path, root: Path = ROOT) -> str:
    return f"../{_relative_id(output_dir, root)}/"


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _load_replay_session(session_dir: Path, project_root: Path = ROOT) -> dict[str, object] | None:
    manifest_path = session_dir / "session.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != "recon3d_orbbec_bag_session_v1":
        return None

    cameras = manifest.get("cameras") or []
    bag_bytes = 0
    for camera in cameras:
        if not isinstance(camera, dict):
            continue
        bag_name = camera.get("bag") or camera.get("bag_path")
        if not bag_name:
            continue
        bag_path = Path(str(bag_name))
        if not bag_path.is_absolute():
            bag_path = session_dir / bag_path
        try:
            bag_bytes += bag_path.stat().st_size
        except OSError:
            pass

    return {
        "id": _relative_id(session_dir, project_root),
        "name": session_dir.name,
        "path": _relative_id(session_dir, project_root),
        "created_at_utc": manifest.get("created_at_utc"),
        "duration_s": manifest.get("duration_s"),
        "requested_duration_s": manifest.get("requested_duration_s"),
        "completed": bool(manifest.get("completed", False)),
        "camera_count": len(cameras),
        "cameras": [str(camera.get("id") or camera.get("cam_id")) for camera in cameras if isinstance(camera, dict)],
        "bag_bytes": bag_bytes,
        "bag_size": _format_bytes(bag_bytes),
    }


def list_replay_sessions(project_root: Path = ROOT) -> list[dict[str, object]]:
    data_root = project_root / DEFAULT_REPLAY_DATA_ROOT
    if not data_root.is_dir():
        return []
    sessions: list[dict[str, object]] = []
    for manifest_path in sorted(data_root.glob("*/session.json")):
        session = _load_replay_session(manifest_path.parent, project_root)
        if session is not None:
            sessions.append(session)
    sessions.sort(key=lambda item: str(item.get("created_at_utc") or ""), reverse=True)
    return sessions


def _resolve_replay_session(project_root: Path, session: str) -> Path:
    if not session:
        raise ValueError("Missing replay session")
    raw = Path(session)
    session_dir = raw if raw.is_absolute() else project_root / raw
    session_dir = session_dir.resolve()
    data_root = (project_root / DEFAULT_REPLAY_DATA_ROOT).resolve()
    try:
        session_dir.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Replay session must be under {_relative_id(data_root, project_root)}") from exc
    if not (session_dir / "session.json").is_file():
        raise FileNotFoundError(f"Replay session has no session.json: {_relative_id(session_dir, project_root)}")
    return session_dir


def _resolve_replay_pipeline(payload: dict[str, object]) -> tuple[str, dict[str, str]]:
    pipeline = str(payload.get("pipeline") or DEFAULT_REPLAY_PIPELINE)
    if pipeline not in REPLAY_PIPELINES:
        valid = ", ".join(sorted(REPLAY_PIPELINES))
        raise ValueError(f"Unsupported replay pipeline '{pipeline}'. Expected one of: {valid}")
    return pipeline, REPLAY_PIPELINES[pipeline]


def _replay_status_locked(project_root: Path = ROOT) -> dict[str, object]:
    global _replay_log, _replay_proc  # noqa: PLW0603
    running = _replay_proc is not None and _replay_proc.poll() is None
    returncode = None
    if _replay_proc is not None and not running:
        returncode = _replay_proc.poll()
        if _replay_log is not None:
            _replay_log.close()
            _replay_log = None
    payload = dict(_replay_info)
    payload.update(
        {
            "ok": True,
            "running": running,
            "returncode": returncode,
            "sessions": list_replay_sessions(project_root),
            "pipelines": [
                {"id": key, "label": value["label"]}
                for key, value in REPLAY_PIPELINES.items()
            ],
            "default_pipeline": DEFAULT_REPLAY_PIPELINE,
        }
    )
    return payload


def replay_status(project_root: Path = ROOT) -> dict[str, object]:
    with _replay_lock:
        return _replay_status_locked(project_root)


def _live_running_locked() -> bool:
    return _live_proc is not None and _live_proc.poll() is None


def _replay_running_locked() -> bool:
    return _replay_proc is not None and _replay_proc.poll() is None


def stop_replay_process(project_root: Path = ROOT, timeout_s: float = 5.0) -> dict[str, object]:
    global _replay_info, _replay_log, _replay_proc  # noqa: PLW0603
    with _replay_lock:
        if _replay_proc is None:
            return _replay_status_locked(project_root)
        _stop_process(_replay_proc, timeout_s)
        if _replay_log is not None:
            _replay_log.close()
            _replay_log = None
        _replay_info = {**_replay_info, "stopped_at": time.time()}
        return _replay_status_locked(project_root)


def start_replay_process(project_root: Path, payload: dict[str, object]) -> dict[str, object]:
    global _replay_info, _replay_log, _replay_proc  # noqa: PLW0603
    with _live_lock:
        if _live_running_locked():
            raise RuntimeError("A viewer-managed live process is already running; stop live before starting replay")
    session_dir = _resolve_replay_session(project_root, str(payload.get("session") or ""))
    pipeline, pipeline_info = _resolve_replay_pipeline(payload)
    config = (project_root / DEFAULT_REPLAY_CONFIG).resolve()
    out_root = project_root / "runs"
    out_dir = out_root / f"{session_dir.name}_{pipeline_info['out_suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_frames = max(int(payload.get("probe_frames") or DEFAULT_REPLAY_PROBE_FRAMES), DEFAULT_REPLAY_PROBE_FRAMES)

    command = [
        sys.executable,
        "-m",
        pipeline_info["module"],
        "replay",
        "--config",
        str(config),
        "--bag-session",
        str(session_dir),
        "--out",
        str(out_dir),
        "--visualize",
        "--rerun-web-port",
        str(int(payload.get("rerun_web_port") or DEFAULT_REPLAY_WEB_PORT)),
        "--rerun-grpc-port",
        str(int(payload.get("rerun_grpc_port") or DEFAULT_REPLAY_GRPC_PORT)),
        "--probe-frames",
        str(probe_frames),
        "--wait-timeout-ms",
        str(int(payload.get("wait_timeout_ms") or 5000)),
    ]
    if bool(payload.get("sync_diagnostics", False)):
        command.append("--sync-diagnostics")

    with _replay_lock:
        if _replay_running_locked():
            raise RuntimeError("A viewer-managed replay process is already running")
        if _replay_log is not None:
            _replay_log.close()
        log_path = out_dir / "viewer_replay.log"
        _replay_log = log_path.open("ab")
        _replay_proc = subprocess.Popen(  # noqa: S603 - command is fixed, user selects only a validated session path
            command,
            cwd=project_root,
            stdout=_replay_log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        _replay_info = {
            "session": _relative_id(session_dir, project_root),
            "session_name": session_dir.name,
            "pipeline": pipeline,
            "pipeline_label": pipeline_info["label"],
            "output": _relative_id(out_dir, project_root),
            "base": _viewer_base_for_output(out_dir, project_root),
            "log": _relative_id(log_path, project_root),
            "pid": _replay_proc.pid,
            "probe_frames": probe_frames,
            "started_at": time.time(),
            "command": " ".join(command),
        }
        return _replay_status_locked(project_root)


def _live_status_locked(project_root: Path = ROOT) -> dict[str, object]:
    global _live_log, _live_proc  # noqa: PLW0603
    running = _live_running_locked()
    returncode = None
    if _live_proc is not None and not running:
        returncode = _live_proc.poll()
        if _live_log is not None:
            _live_log.close()
            _live_log = None
    payload = dict(_live_info)
    payload.update(
        {
            "ok": True,
            "running": running,
            "returncode": returncode,
            "pipelines": [
                {"id": key, "label": value["label"]}
                for key, value in REPLAY_PIPELINES.items()
            ],
            "default_pipeline": DEFAULT_REPLAY_PIPELINE,
        }
    )
    payload.setdefault("pipeline", DEFAULT_REPLAY_PIPELINE)
    payload.setdefault("pipeline_label", REPLAY_PIPELINES[DEFAULT_REPLAY_PIPELINE]["label"])
    return payload


def live_status(project_root: Path = ROOT) -> dict[str, object]:
    with _live_lock:
        return _live_status_locked(project_root)


def stop_live_process(project_root: Path = ROOT, timeout_s: float = 5.0) -> dict[str, object]:
    global _live_info, _live_log, _live_proc  # noqa: PLW0603
    with _live_lock:
        if _live_proc is None:
            return _live_status_locked(project_root)
        _stop_process(_live_proc, timeout_s)
        if _live_log is not None:
            _live_log.close()
            _live_log = None
        _live_info = {**_live_info, "stopped_at": time.time()}
        return _live_status_locked(project_root)


def start_live_process(project_root: Path, payload: dict[str, object]) -> dict[str, object]:
    global _live_info, _live_log, _live_proc  # noqa: PLW0603
    with _replay_lock:
        if _replay_running_locked():
            raise RuntimeError("A viewer-managed replay process is already running; stop replay before starting live")

    pipeline, pipeline_info = _resolve_replay_pipeline(payload)
    config = (project_root / DEFAULT_REPLAY_CONFIG).resolve()
    out_dir = (project_root / "runs" / pipeline_info["live_output"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        pipeline_info["module"],
        "run",
        "--config",
        str(config),
        "--out",
        str(out_dir),
        "--visualize",
        "--rerun-web-port",
        str(int(payload.get("rerun_web_port") or DEFAULT_LIVE_WEB_PORT)),
        "--rerun-grpc-port",
        str(int(payload.get("rerun_grpc_port") or DEFAULT_LIVE_GRPC_PORT)),
    ]
    if bool(payload.get("sync_diagnostics", True)):
        command.append("--sync-diagnostics")

    with _live_lock:
        if _live_running_locked():
            raise RuntimeError("A viewer-managed live process is already running")
        if _live_log is not None:
            _live_log.close()
        log_path = out_dir / "viewer_live.log"
        _live_log = log_path.open("ab")
        _live_proc = subprocess.Popen(  # noqa: S603 - command is fixed; browser toggles only a known live pipeline
            command,
            cwd=project_root,
            stdout=_live_log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        _live_info = {
            "pipeline": pipeline,
            "pipeline_label": pipeline_info["label"],
            "output": _relative_id(out_dir, project_root),
            "base": _viewer_base_for_output(out_dir, project_root),
            "log": _relative_id(log_path, project_root),
            "pid": _live_proc.pid,
            "started_at": time.time(),
            "command": " ".join(command),
        }
        return _live_status_locked(project_root)


atexit.register(stop_replay_process)
atexit.register(stop_live_process)


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve the project tree with MIME types needed by the static viewer."""

    project_root = ROOT

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".ply": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _project_root(self) -> Path:
        return Path(self.project_root)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _redirect_to_live_viewer(self, query: str) -> None:
        location = "/viewer/live.html"
        if query:
            location = f"{location}?{query}"
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        parsed = urlsplit(self.path)
        if parsed.path in {"/viewer", "/viewer/"}:
            self._redirect_to_live_viewer(parsed.query)
            return
        if self.path == "/api/active-learning/summary":
            self._send_json(dataset_summary(self._project_root()))
            return
        if self.path == "/api/active-learning/status":
            self._send_json(latest_status(self._project_root()))
            return
        if self.path == "/api/replay/sessions":
            self._send_json({"ok": True, "sessions": list_replay_sessions(self._project_root())})
            return
        if self.path == "/api/replay/status":
            self._send_json(replay_status(self._project_root()))
            return
        if self.path == "/api/live/status":
            self._send_json(live_status(self._project_root()))
            return
        if self.path == "/api/point-tracking/status":
            self._send_json(point_tracking.get_status(self._project_root()))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        try:
            if self.path in {"/api/live/start", "/api/replay/start"} and point_tracking.preview_running(self._project_root()):
                raise ValueError("Stop the recording preview before starting live reconstruction or bag replay")
            if self.path == "/api/point-tracking/preview":
                if _live_running_locked() or _replay_running_locked():
                    raise ValueError("Stop live reconstruction and replay before opening the recording preview")
                self._send_json(point_tracking.start_preview(self._project_root()))
                return
            if self.path == "/api/point-tracking/command":
                self._send_json(point_tracking.preview_command(self._project_root(), self._read_json()))
                return
            if self.path == "/api/point-tracking/start":
                payload = self._read_json()
                session = _resolve_replay_session(self._project_root(), str(payload.get("session") or ""))
                self._send_json(point_tracking.start_tracking(self._project_root(), session, payload.get("camera")))
                return
            if self.path == "/api/point-tracking/playback":
                payload = self._read_json()
                session = _resolve_replay_session(self._project_root(), str(payload.get("session") or ""))
                self._send_json(point_tracking.start_playback(self._project_root(), session))
                return
            if self.path == "/api/active-learning/annotation":
                self._send_json(save_annotation(self._project_root(), self._read_json()))
                return
            if self.path == "/api/active-learning/fine-tune":
                self._send_json(launch_finetune(self._project_root(), self._read_json()))
                return
            if self.path == "/api/replay/start":
                self._send_json(start_replay_process(self._project_root(), self._read_json()))
                return
            if self.path == "/api/replay/stop":
                self._send_json(stop_replay_process(self._project_root()))
                return
            if self.path == "/api/live/start":
                self._send_json(start_live_process(self._project_root(), self._read_json()))
                return
            if self.path == "/api/live/stop":
                self._send_json(stop_live_process(self._project_root()))
                return
            self._send_json({"ok": False, "error": f"Unknown endpoint: {self.path}"}, status=404)
        except Exception as exc:  # noqa: BLE001 - return useful UI errors
            self._send_json({"ok": False, "error": str(exc)}, status=500)


def make_server(host: str, port: int, root: Path) -> ThreadingHTTPServer:
    handler_cls = type("ProjectViewerHandler", (ViewerHandler,), {"project_root": root.resolve()})
    handler = partial(handler_cls, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AnatoBreast3D browser viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    parser.add_argument(
        "--clean-existing",
        dest="clean_existing",
        action="store_true",
        default=True,
        help="Stop existing viewer/replay/Rerun processes on the development ports before binding.",
    )
    parser.add_argument(
        "--no-clean-existing",
        dest="clean_existing",
        action="store_false",
        help="Do not stop existing viewer/replay/Rerun processes before binding.",
    )
    args = parser.parse_args()

    root = ROOT
    if args.clean_existing:
        stopped = clean_existing_processes(args.host, args.port)
        if stopped:
            print(f"Stopped existing viewer/replay processes: {', '.join(str(pid) for pid in stopped)}")

    server = None
    chosen_port = args.port
    for port in range(args.port, args.port + 20):
        try:
            server = make_server(args.host, port, root)
            chosen_port = port
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port found from {args.port} to {args.port + 19}")

    url = f"http://{args.host}:{chosen_port}/viewer/live.html"
    print(f"Serving AnatoBreast3D Viewer at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
