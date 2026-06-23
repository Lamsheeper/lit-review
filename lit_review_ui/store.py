from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "review"


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            root
            or Path(os.getenv("LIT_REVIEW_UI_PROJECTS_DIR", "ui_projects"))
        ).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "projects.sqlite"
        self._init_db()
        self.interrupt_running_jobs()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    error TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS jobs_project_id ON jobs(project_id, created_at);
                """
            )

    def interrupt_running_jobs(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status='interrupted', finished_at=?, error=?
                WHERE status IN ('queued', 'running', 'cancelling')
                """,
                (utc_now(), "Application restarted while the job was active."),
            )

    def project_dir(self, project_id: str) -> Path:
        project = self.get_project(project_id)
        return self.root / project["slug"]

    def ensure_layout(self, path: Path) -> None:
        for relative in ["configs", "papers/logs", "review_run", "feature_extract", "jobs"]:
            (path / relative).mkdir(parents=True, exist_ok=True)

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Project name is required.")
        project_id = uuid.uuid4().hex
        base = slugify(name)
        slug = base
        counter = 2
        while (self.root / slug).exists():
            slug = f"{base}-{counter}"
            counter += 1
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, slug, name, description.strip(), now, now),
            )
        path = self.root / slug
        self.ensure_layout(path)
        self.write_json(
            path / "project.json",
            {
                "id": project_id,
                "slug": slug,
                "name": name,
                "description": description.strip(),
                "created_at": now,
                "updated_at": now,
            },
        )
        (path / "draft.md").write_text("", encoding="utf-8")
        (path / "goal.md").write_text("", encoding="utf-8")
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self.enrich_project(dict(row)) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return self.enrich_project(dict(row))

    def update_project(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        current = self.get_project(project_id)
        name = str(values.get("name", current["name"])).strip()
        description = str(values.get("description", current["description"])).strip()
        if not name:
            raise ValueError("Project name is required.")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET name=?, description=?, updated_at=? WHERE id=?",
                (name, description, now, project_id),
            )
        project = self.get_project(project_id)
        self.write_json(self.project_dir(project_id) / "project.json", project)
        return project

    def delete_project(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        active = self.active_job(project_id)
        if active:
            raise ValueError(f"Project has an active {active['kind']} job.")

        path = (self.root / project["slug"]).resolve()
        if path == self.root or self.root not in path.parents:
            raise ValueError("Refusing to delete a project outside the projects directory.")

        with self.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))

        if path.exists():
            shutil.rmtree(path)
        return project

    def touch_project(self, project_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (utc_now(), project_id))

    def enrich_project(self, project: dict[str, Any]) -> dict[str, Any]:
        path = self.root / project["slug"]
        progress = {
            "defined": bool((path / "goal.md").exists() and (path / "goal.md").stat().st_size),
            "taxonomy": bool((path / "taxonomy.json").exists()),
            "searched": bool((path / "papers/logs/candidates.json").exists()),
            "downloaded": len(list((path / "papers").glob("*.pdf"))),
            "extracted": bool((path / "feature_extract/features.jsonl").exists()),
        }
        project["progress"] = progress
        return project

    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)

    def create_job(self, project_id: str, kind: str, log_path: Path) -> dict[str, Any]:
        active = self.active_job(project_id)
        if active:
            raise ValueError(f"Project already has an active {active['kind']} job.")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                (id, project_id, kind, status, created_at, log_path)
                VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (job_id, project_id, kind, now, str(log_path)),
            )
        return self.get_job(job_id)

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status", "started_at", "finished_at", "return_code", "error", "log_path"
        }
        fields = [(key, value) for key, value in values.items() if key in allowed]
        if fields:
            assignments = ", ".join(f"{key}=?" for key, _ in fields)
            with self.connect() as conn:
                conn.execute(
                    f"UPDATE jobs SET {assignments} WHERE id=?",
                    tuple(value for _, value in fields) + (job_id,),
                )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def active_job(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE project_id=? AND status IN ('queued', 'running', 'cancelling')
                ORDER BY created_at DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]
