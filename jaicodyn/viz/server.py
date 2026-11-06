"""Local viewer server. Binds to this machine only, because match data must stay private."""

import json
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jaicodyn.data import available, load

from .api import View

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("viewer.html", "text/html"), "/viewer.js": ("viewer.js", "text/javascript"),
         "/viewer.css": ("viewer.css", "text/css")}
BLOCK = 1 << 20
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _view(key):
    return View(load(key))


def view(key):
    with _lock:
        return _view(key)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            self._route(self.path.split("?")[0])
        except (ConnectionError, TimeoutError):
            pass  # browsers drop video requests whenever they seek
        except (KeyError, FileNotFoundError, ValueError) as error:
            self._send(404, json.dumps({"error": str(error)}).encode(), "application/json")

    def _route(self, path):
        if path in FILES:
            name, kind = FILES[path]
            self._send(200, (STATIC / name).read_bytes(), kind)
        elif path == "/api/matches":
            self._json([{"key": m.key, "home": m.home, "away": m.away} for m in available()])
        elif re.fullmatch(r"/api/info/[\w-]+/[\w-]+", path):
            self._json(view(path.split("/", 3)[3]).info())
        elif re.fullmatch(r"/api/chunk/[\w-]+/[\w-]+/\d+", path):
            _, _, _, source, match_id, index = path.split("/")
            self._json(view("{}/{}".format(source, match_id)).chunk(int(index)))
        elif re.fullmatch(r"/video/[\w-]+/[\w-]+", path):
            video = view(path.split("/", 2)[2]).match.video
            if video is None:
                raise FileNotFoundError("no video")
            self._video(video)
        else:
            raise FileNotFoundError(path)

    def _json(self, value):
        self._send(200, json.dumps(value, separators=(",", ":")).encode(), "application/json")

    def _send(self, status, body, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _video(self, path):
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        start = int(match.group(1)) if match else 0
        stop = int(match.group(2)) + 1 if match and match.group(2) else size
        stop = min(stop, size)

        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(stop - start))
        if match:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, stop - 1, size))
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = stop - start
            while remaining:
                block = handle.read(min(BLOCK, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)


def serve(port):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
