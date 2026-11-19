"""Read-only REST API bound to the local computer."""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import unquote, urlsplit

from autobiz.storage import ProjectStore


def _handler_for(store: ProjectStore) -> type[BaseHTTPRequestHandler]:
    class ApiHandler(BaseHTTPRequestHandler):
        server_version = "AutoBizLocalAPI/1.0"

        def do_GET(self) -> None:
            try:
                self._route_get()
            except sqlite3.Error:
                self._send_json(
                    500, {"error": "Database lokal tidak dapat dibaca."}
                )

        def _route_get(self) -> None:
            path = urlsplit(self.path).path
            parts = [unquote(part) for part in path.strip("/").split("/")]

            if path == "/health":
                self._send_json(200, {"status": "ok", "service": "AutoBiz Data"})
                return

            if path == "/api/v1/projects":
                projects = store.list_projects()
                self._send_json(
                    200,
                    {"projects": [self._public_project(p) for p in projects]},
                )
                return

            if (
                len(parts) in (4, 5)
                and parts[:3] == ["api", "v1", "projects"]
            ):
                project = store.get_project(parts[3])
                if project is None:
                    self._send_json(404, {"error": "Proyek tidak ditemukan."})
                    return
                if len(parts) == 4:
                    self._send_json(
                        200, {"project": self._public_project(project)}
                    )
                    return
                if parts[4] == "analyses":
                    self._send_json(
                        200,
                        {"analyses": store.list_analyses(project["id"])},
                    )
                    return

            self._send_json(404, {"error": "Endpoint tidak ditemukan."})

        @staticmethod
        def _public_project(project: dict) -> dict:
            return {
                key: value for key, value in project.items()
                if key != "dataset_path"
            }

        def do_POST(self) -> None:
            self._send_json(405, {"error": "API hanya mendukung GET."})

        def do_PUT(self) -> None:
            self._send_json(405, {"error": "API hanya mendukung GET."})

        def do_DELETE(self) -> None:
            self._send_json(405, {"error": "API hanya mendukung GET."})

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"AutoBiz API: {format % args}")

    return ApiHandler


class LocalApiServer:
    def __init__(self, store: ProjectStore):
        self.store = store
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def base_url(self) -> str | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self, port: int = 8765) -> str:
        if self.is_running:
            if self.base_url is None:
                raise RuntimeError("Alamat API lokal belum tersedia.")
            return self.base_url

        self._server = ThreadingHTTPServer(
            ("127.0.0.1", port), _handler_for(self.store)
        )
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="autobiz-local-api",
            daemon=True,
        )
        self._thread.start()
        if self.base_url is None:
            raise RuntimeError("API lokal gagal mendapatkan alamat.")
        return self.base_url

    def stop(self) -> None:
        if self._server is None:
            return
        server = self._server
        thread = self._thread
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=3)
            if thread.is_alive():
                raise RuntimeError("Server API lokal belum berhenti.")
        self._server = None
        self._thread = None
