from __future__ import annotations

import http.client
import threading

from tools.serve_recon3d_viewer import make_server


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
