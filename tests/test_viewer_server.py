from __future__ import annotations

import http.client
import json
import threading

import pytest

from tools import serve_recon3d_viewer
from tools.serve_recon3d_viewer import (
    list_replay_sessions,
    make_server,
    start_live_process,
    start_replay_process,
    stop_live_process,
    stop_replay_process,
)


class _DummyProcess:
    pid = 12345

    def __init__(self) -> None:
        self._running = True

    def poll(self):
        return None if self._running else 0

    def terminate(self):
        self._running = False

    def wait(self, timeout=None):
        self._running = False
        return 0

    def kill(self):
        self._running = False


def test_viewer_directory_redirects_to_live_html(tmp_path) -> None:
    server = make_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/viewer/?base=..%2Fruns%2Flive%2F")
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 302
    assert response.getheader("Location") == "/viewer/live.html?base=..%2Fruns%2Flive%2F"


def test_list_replay_sessions_finds_sdk_bag_manifest(tmp_path) -> None:
    session_dir = tmp_path / "data" / "Replay" / "orbbec_bag_test"
    session_dir.mkdir(parents=True)
    (session_dir / "cam0.bag").write_bytes(b"bag0")
    (session_dir / "cam1.bag").write_bytes(b"bag1")
    (session_dir / "cam2.bag").write_bytes(b"bag2")
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "schema": "recon3d_orbbec_bag_session_v1",
                "created_at_utc": "2026-06-19T00:00:00+00:00",
                "duration_s": 12.5,
                "completed": True,
                "cameras": [
                    {"id": "cam0", "bag": "cam0.bag"},
                    {"id": "cam1", "bag": "cam1.bag"},
                    {"id": "cam2", "bag": "cam2.bag"},
                ],
            }
        ),
        encoding="utf-8",
    )

    sessions = list_replay_sessions(tmp_path)

    assert sessions == [
        {
            "id": "data/Replay/orbbec_bag_test",
            "name": "orbbec_bag_test",
            "path": "data/Replay/orbbec_bag_test",
            "created_at_utc": "2026-06-19T00:00:00+00:00",
            "duration_s": 12.5,
            "requested_duration_s": None,
            "completed": True,
            "camera_count": 3,
            "cameras": ["cam0", "cam1", "cam2"],
            "bag_bytes": 12,
            "bag_size": "12 B",
        }
    ]


def _write_replay_session(root) -> None:
    session_dir = root / "data" / "Replay" / "orbbec_bag_test"
    session_dir.mkdir(parents=True)
    (session_dir / "cam0.bag").write_bytes(b"bag0")
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "schema": "recon3d_orbbec_bag_session_v1",
                "duration_s": 1.0,
                "completed": True,
                "cameras": [{"id": "cam0", "bag": "cam0.bag"}],
            }
        ),
        encoding="utf-8",
    )


def test_start_replay_defaults_to_anatomy_guided_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_replay_session(tmp_path)
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(serve_recon3d_viewer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serve_recon3d_viewer.sys, "executable", "python")
    serve_recon3d_viewer._replay_proc = None
    serve_recon3d_viewer._replay_log = None
    serve_recon3d_viewer._replay_info = {}

    try:
        status = start_replay_process(tmp_path, {"session": "data/Replay/orbbec_bag_test"})
    finally:
        stop_replay_process(tmp_path)

    command = captured["command"]
    assert command[:4] == ["python", "-m", "anatomy_guided_recon3d.cli", "replay"]
    assert command[command.index("--probe-frames") + 1] == "180"
    assert status["pipeline"] == "anatomy_guided_recon3d"
    assert status["pipeline_label"] == "Anatomy-guided Recon3D"
    assert status["output"] == "runs/orbbec_bag_test_anatomy_guided_replay"
    assert status["base"] == "../runs/orbbec_bag_test_anatomy_guided_replay/"
    assert status["probe_frames"] == 180


def test_start_replay_runs_selected_recon3d_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_replay_session(tmp_path)
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(serve_recon3d_viewer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serve_recon3d_viewer.sys, "executable", "python")
    serve_recon3d_viewer._replay_proc = None
    serve_recon3d_viewer._replay_log = None
    serve_recon3d_viewer._replay_info = {}

    try:
        status = start_replay_process(
            tmp_path,
            {"session": "data/Replay/orbbec_bag_test", "pipeline": "recon3d"},
        )
    finally:
        stop_replay_process(tmp_path)

    command = captured["command"]
    assert command[:4] == ["python", "-m", "recon3d.cli", "replay"]
    assert command[command.index("--probe-frames") + 1] == "180"
    assert status["pipeline"] == "recon3d"
    assert status["pipeline_label"] == "Recon3D"
    assert status["output"] == "runs/orbbec_bag_test_replay"
    assert status["base"] == "../runs/orbbec_bag_test_replay/"


def test_start_replay_clamps_low_probe_frames(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_replay_session(tmp_path)
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(serve_recon3d_viewer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serve_recon3d_viewer.sys, "executable", "python")
    serve_recon3d_viewer._replay_proc = None
    serve_recon3d_viewer._replay_log = None
    serve_recon3d_viewer._replay_info = {}

    try:
        status = start_replay_process(
            tmp_path,
            {"session": "data/Replay/orbbec_bag_test", "probe_frames": 30},
        )
    finally:
        stop_replay_process(tmp_path)

    command = captured["command"]
    assert command[command.index("--probe-frames") + 1] == "180"
    assert status["probe_frames"] == 180


def test_start_live_runs_anatomy_guided_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(serve_recon3d_viewer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serve_recon3d_viewer.sys, "executable", "python")
    serve_recon3d_viewer._live_proc = None
    serve_recon3d_viewer._live_log = None
    serve_recon3d_viewer._live_info = {}
    serve_recon3d_viewer._replay_proc = None
    serve_recon3d_viewer._replay_log = None
    serve_recon3d_viewer._replay_info = {}

    try:
        status = start_live_process(tmp_path, {})
    finally:
        stop_live_process(tmp_path)

    command = captured["command"]
    assert command[:4] == ["python", "-m", "anatomy_guided_recon3d.cli", "run"]
    assert "--visualize" in command
    assert "--sync-diagnostics" in command
    assert status["pipeline"] == "anatomy_guided_recon3d"
    assert status["output"] == "runs/anatomy_guided_recon3d_live"
    assert status["base"] == "../runs/anatomy_guided_recon3d_live/"


def test_start_live_runs_selected_recon3d_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(serve_recon3d_viewer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serve_recon3d_viewer.sys, "executable", "python")
    serve_recon3d_viewer._live_proc = None
    serve_recon3d_viewer._live_log = None
    serve_recon3d_viewer._live_info = {}
    serve_recon3d_viewer._replay_proc = None
    serve_recon3d_viewer._replay_log = None
    serve_recon3d_viewer._replay_info = {}

    try:
        status = start_live_process(tmp_path, {"pipeline": "recon3d"})
    finally:
        stop_live_process(tmp_path)

    command = captured["command"]
    assert command[:4] == ["python", "-m", "recon3d.cli", "run"]
    assert status["pipeline"] == "recon3d"
    assert status["pipeline_label"] == "Recon3D"
    assert status["output"] == "runs/recon3d_live"
    assert status["base"] == "../runs/recon3d_live/"


def test_start_replay_rejects_unknown_pipeline(tmp_path) -> None:
    _write_replay_session(tmp_path)

    with pytest.raises(ValueError, match="Unsupported replay pipeline"):
        start_replay_process(
            tmp_path,
            {
                "session": "data/Replay/orbbec_bag_test",
                "pipeline": "not_a_pipeline",
            },
        )
