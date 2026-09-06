import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from web_app import MasarHandler


def test_health_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MasarHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health"
        ) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()