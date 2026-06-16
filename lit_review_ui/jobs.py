from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from .store import ProjectStore, utc_now


class JobManager:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.lock = threading.Lock()

    def start(
        self,
        project_id: str,
        kind: str,
        commands: list[list[str]],
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        project_dir = self.store.project_dir(project_id)
        provisional = project_dir / "jobs" / "pending.log"
        job = self.store.create_job(project_id, kind, provisional)
        log_path = project_dir / "jobs" / f"{job['id']}.log"
        job = self.store.update_job(job["id"], log_path=str(log_path))
        thread = threading.Thread(
            target=self._run,
            args=(job["id"], project_id, commands, log_path, env_overrides or {}),
            daemon=True,
        )
        thread.start()
        return job

    def _run(
        self,
        job_id: str,
        project_id: str,
        commands: list[list[str]],
        log_path: Path,
        env_overrides: dict[str, str],
    ) -> None:
        if self.store.get_job(job_id)["status"] == "cancelling":
            self.store.update_job(job_id, status="cancelled", finished_at=utc_now())
            return
        self.store.update_job(job_id, status="running", started_at=utc_now())
        env = os.environ.copy()
        env.update({key: value for key, value in env_overrides.items() if value})
        try:
            with log_path.open("a", encoding="utf-8") as log:
                for command in commands:
                    log.write("$ " + " ".join(self._redact_command(command)) + "\n")
                    log.flush()
                    process = subprocess.Popen(
                        command,
                        cwd=Path.cwd(),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    with self.lock:
                        self.processes[job_id] = process
                    return_code = process.wait()
                    with self.lock:
                        self.processes.pop(job_id, None)
                    if return_code != 0:
                        raise RuntimeError(
                            f"Command exited with status {return_code}: {command[1] if len(command) > 1 else command[0]}"
                        )
            self.store.update_job(
                job_id,
                status="completed",
                finished_at=utc_now(),
                return_code=0,
            )
            self.store.touch_project(project_id)
        except Exception as exc:
            current = self.store.get_job(job_id)
            status = "cancelled" if current["status"] == "cancelling" else "failed"
            self.store.update_job(
                job_id,
                status=status,
                finished_at=utc_now(),
                return_code=current.get("return_code") or 1,
                error=str(exc),
            )
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        self.store.update_job(job_id, status="cancelling")
        with self.lock:
            process = self.processes.get(job_id)
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return self.store.get_job(job_id)

    @staticmethod
    def _redact_command(command: list[str]) -> list[str]:
        redacted: list[str] = []
        hide_next = False
        for value in command:
            if hide_next:
                redacted.append("***redacted***")
                hide_next = False
            else:
                redacted.append(value)
                hide_next = value in {"--api-key", "--llm-api-key"}
        return redacted
