from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlsplit
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.annotation_store import dataset_summary, save_annotation  # noqa: E402
from active_learning.training import latest_status, launch_finetune  # noqa: E402


# ---- rerun replay (viewer/replay.html) ----
RECORDINGS_GLOB = "runs/**/recordings/*.rrd"
REPLAY_WEB_PORT = 9091
REPLAY_GRPC_PORT = 9877
REPLAY_URL = f"http://localhost:{REPLAY_WEB_PORT}?url=rerun%2Bhttp://localhost:{REPLAY_GRPC_PORT}/proxy"
_replay_lock = threading.Lock()
_replay_proc: subprocess.Popen | None = None


def list_recordings(root: Path) -> list[dict]:
    """Return saved .rrd recordings under any runs/**/recordings/, newest first."""
    items: list[dict] = []
    for path in root.glob(RECORDINGS_GLOB):
        if not path.is_file():
            continue
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": path.relative_to(root).as_posix(),
                "size_mb": round(stat.st_size / 1e6, 1),
                "mtime": stat.st_mtime,
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items


def _resolve_recording(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    runs_dir = (root / "runs").resolve()
    if runs_dir not in candidate.parents or candidate.suffix != ".rrd" or not candidate.is_file():
        raise ValueError(f"Invalid recording path: {rel!r}")
    return candidate


def start_replay(root: Path, rel: str) -> dict:
    """Spawn (or restart) the rerun replay server for one recording on a fixed port."""
    global _replay_proc
    rrd = _resolve_recording(root, rel)
    with _replay_lock:
        if _replay_proc is not None and _replay_proc.poll() is None:
            _replay_proc.terminate()
            try:
                _replay_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _replay_proc.kill()
        command = [
            sys.executable,
            "tools/replay_recon.py",
            "--file",
            str(rrd),
            "--web-port",
            str(REPLAY_WEB_PORT),
            "--grpc-port",
            str(REPLAY_GRPC_PORT),
        ]
        _replay_proc = subprocess.Popen(
            command,
            cwd=str(root),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    return {"ok": True, "url": REPLAY_URL, "file": rrd.name, "pid": _replay_proc.pid}


@atexit.register
def _stop_replay() -> None:
    if _replay_proc is not None and _replay_proc.poll() is None:
        _replay_proc.terminate()


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve the project tree with MIME types needed by the static viewer."""

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
        return ROOT

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
        if self.path == "/api/recordings":
            self._send_json({"recordings": list_recordings(self._project_root())})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        try:
            if self.path == "/api/active-learning/annotation":
                self._send_json(save_annotation(self._project_root(), self._read_json()))
                return
            if self.path == "/api/active-learning/fine-tune":
                self._send_json(launch_finetune(self._project_root(), self._read_json()))
                return
            if self.path == "/api/replay":
                self._send_json(start_replay(self._project_root(), self._read_json().get("path", "")))
                return
            self._send_json({"ok": False, "error": f"Unknown endpoint: {self.path}"}, status=404)
        except Exception as exc:  # noqa: BLE001 - return useful UI errors
            self._send_json({"ok": False, "error": str(exc)}, status=500)


def make_server(host: str, port: int, root: Path) -> ThreadingHTTPServer:
    handler = partial(ViewerHandler, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Recon3D browser viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    args = parser.parse_args()

    root = ROOT
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
    print(f"Serving Recon3D viewer at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
