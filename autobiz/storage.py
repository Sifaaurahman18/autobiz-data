"""SQLite persistence for project metadata and analysis history."""

from __future__ import annotations

import os
import platform
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


def application_data_dir() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "AutoBizData"


class ProjectStore:
    def __init__(self, database_path: str | Path | None = None):
        self.database_path = (
            Path(database_path) if database_path is not None
            else application_data_dir() / "autobiz.sqlite3"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_path TEXT,
                    source TEXT NOT NULL DEFAULT 'file',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(id)
                        ON DELETE CASCADE,
                    method TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_project_created
                    ON analysis_history(project_id, created_at DESC);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save_project(
        self,
        name: str,
        dataset_path: str | None,
        source: str = "file",
        project_id: str | None = None,
    ) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Nama proyek tidak boleh kosong.")

        project_id = project_id or str(uuid.uuid4())
        now = self._now()
        normalized_path = (
            str(Path(dataset_path).expanduser().resolve())
            if dataset_path else None
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects
                    (id, name, dataset_path, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    dataset_path = excluded.dataset_path,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (project_id, clean_name, normalized_path, source, now, now),
            )
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("Proyek tersimpan tetapi tidak dapat dibaca.")
        return project

    def get_project(self, project_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC, name"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_analysis(
        self, project_id: str, method: str, result: str
    ) -> int:
        if not method.strip() or not result.strip():
            raise ValueError("Analisis dan hasilnya wajib diisi.")
        with self._connect() as connection:
            project = connection.execute(
                "SELECT 1 FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ValueError("Proyek tidak ditemukan.")
            cursor = connection.execute(
                """
                INSERT INTO analysis_history
                    (project_id, method, result, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, method.strip(), result, self._now()),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (self._now(), project_id),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Riwayat analisis gagal disimpan.")
            return cursor.lastrowid

    def list_analyses(self, project_id: str, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 500:
            raise ValueError("Batas riwayat harus antara 1 dan 500.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, project_id, method, result, created_at
                FROM analysis_history
                WHERE project_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
