from __future__ import annotations

import http.client
import json
import threading

from tools.serve_recon3d_viewer import list_replay_sessions, make_server


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
