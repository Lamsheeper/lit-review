"""App defaults and an interactive gcloud ADC login, without exposing tokens."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from lit_review.core.google_auth import vertex_model_url


class CloudSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    project: str
    location: str = "us-central1"

    @field_validator("project", "location")
    @classmethod
    def clean(cls, value):
        return value.strip()

    def validate_endpoint(self):
        vertex_model_url(self.project, self.location, "gemini-2.5-flash")
        if not self.project or not self.location:
            raise ValueError("Enter a Google Cloud project and region.")


ACTIVE = {"starting", "awaiting_code", "verifying"}
ENV_KEYS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI")


class GoogleCloudSetup:
    def __init__(self, root: Path, login_timeout: float = 600):
        self.path = root / "google_cloud.json"
        self.lock = threading.RLock()
        self.base_env = {key: os.environ.get(key) for key in ENV_KEYS}
        self.settings: CloudSettings | None = None
        self.configuration_error = ""
        self.login: dict | None = None
        self.process = None
        self.timer = None
        self.login_timeout = login_timeout
        if self.path.exists():
            try:
                settings = CloudSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
                settings.validate_endpoint()
                self.settings = settings
                self.apply()
            except (OSError, ValueError):
                self.configuration_error = "Saved Google Cloud settings could not be loaded. Save a valid project and region to replace them."

    def apply(self):
        with self.lock:
            values = self.base_env
            if self.settings and self.settings.enabled:
                values = dict(zip(ENV_KEYS, (self.settings.project, self.settings.location, "true")))
            for key, value in values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def save(self, settings: CloudSettings):
        settings.validate_endpoint()
        with self.lock:
            if self.login and self.login["status"] in ACTIVE:
                raise ValueError("Finish or cancel Google sign-in before changing settings.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(settings.model_dump_json(indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
            self.settings = settings
            self.configuration_error = ""
            self.apply()
            return self.status()

    def status(self):
        with self.lock:
            explicit_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if explicit_file:
                credential_file = Path(explicit_file).expanduser()
            else:
                from google.auth import _cloud_sdk

                credential_file = Path(_cloud_sdk.get_application_default_credentials_path())
            return {
                "configured": self.settings is not None,
                "configuration_error": self.configuration_error,
                "enabled": bool(self.settings and self.settings.enabled),
                "project": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                "location": os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
                "saved_project": self.settings.project if self.settings else "",
                "saved_location": self.settings.location if self.settings else "",
                "gcloud_installed": shutil.which("gcloud") is not None,
                "adc_file_detected": credential_file.is_file(),
                "external_credentials": bool(explicit_file),
                "login": dict(self.login) if self.login else None,
            }

    def environment(self):
        with self.lock:
            if not self.settings or not self.settings.enabled:
                return {}
            return dict(zip(ENV_KEYS, (self.settings.project, self.settings.location, "true")))

    def start_login(self):
        with self.lock:
            if self.login and self.login["status"] in ACTIVE:
                raise ValueError("Google sign-in is already in progress.")
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS selects an external credential file. "
                    "Use Check credentials, or unset that variable before starting a user sign-in."
                )
            executable = shutil.which("gcloud")
            if not executable:
                raise ValueError(
                    "Install the Google Cloud CLI on the machine running this app, then try again."
                )
            if not self.settings or not self.settings.enabled:
                raise ValueError("Save your Google Cloud project and region before signing in.")
            # No shell, no access tokens, and no credentials in process arguments.
            try:
                process = subprocess.Popen(
                    [
                        executable,
                        "auth",
                        "application-default",
                        "login",
                        "--no-launch-browser",
                        "--quiet",
                        f"--project={self.settings.project}",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "CLOUDSDK_CORE_DISABLE_COLOR": "1"},
                )
            except OSError as exc:
                raise ValueError("The Google Cloud CLI could not start. Check its installation.") from exc
            self.process = process
            self.login = {
                "id": uuid.uuid4().hex,
                "status": "starting",
                "authorization_url": None,
                "message": "Preparing Google sign-in…",
            }
            login_id = self.login["id"]
            self.timer = threading.Timer(self.login_timeout, self._expire, args=(login_id,))
            self.timer.daemon = True
            self.timer.start()
            threading.Thread(target=self._read_login, args=(process, login_id), daemon=True).start()
            return dict(self.login)

    def _read_login(self, process, login_id):
        # Parse only Google's authorization URL. Never return or persist gcloud output.
        line = ""
        overflow = False
        try:
            while True:
                char = process.stdout.read(1)
                if not char:
                    break
                if char in "\r\n":
                    if not overflow:
                        self._authorization_line(line, login_id)
                    line, overflow = "", False
                elif len(line) < 32768:
                    line += char
                else:
                    overflow = True
            if line and not overflow:
                self._authorization_line(line, login_id)
            result = process.wait()
            with self.lock:
                if self.login and self.login["id"] == login_id and self.login["status"] in ACTIVE:
                    self.login.update(
                        status="succeeded" if result == 0 else "failed",
                        authorization_url=None,
                        message="Google sign-in complete. Check credentials to verify your setup."
                        if result == 0
                        else "Google sign-in failed. Start again and use the new authorization code.",
                    )
        except (OSError, ValueError):
            with self.lock:
                if self.login and self.login["id"] == login_id and self.login["status"] in ACTIVE:
                    self.login.update(
                        status="failed", authorization_url=None, message="Google sign-in stopped. Try again."
                    )
        finally:
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
            with self.lock:
                if self.login and self.login["id"] == login_id:
                    self.process = None
                    if self.timer:
                        self.timer.cancel()

    def _authorization_line(self, line, login_id):
        for candidate in re.findall(r"https://[^\s\x1b]+", line):
            parsed = urlsplit(candidate)
            if parsed.netloc != "accounts.google.com" or parsed.path not in {
                "/o/oauth2/auth",
                "/o/oauth2/v2/auth",
            }:
                continue
            with self.lock:
                if self.login and self.login["id"] == login_id and self.login["status"] == "starting":
                    self.login.update(
                        status="awaiting_code",
                        authorization_url=candidate,
                        message="Open Google sign-in, then paste the authorization code here.",
                    )

    def submit_code(self, login_id: str, code: str):
        code = code.strip()
        if not code or len(code) > 4096 or any(char.isspace() or ord(char) < 32 for char in code):
            raise ValueError("Enter the single authorization code shown by Google.")
        with self.lock:
            self._require_login(login_id)
            if self.login["status"] != "awaiting_code" or not self.process or self.process.poll() is not None:
                raise ValueError("This sign-in is not waiting for a code. Start a new sign-in.")
            try:
                self.process.stdin.write(code + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise ValueError("Google sign-in stopped. Start a new sign-in.") from exc
            self.login.update(
                status="verifying", authorization_url=None, message="Completing Google sign-in…"
            )
            return dict(self.login)

    def _require_login(self, login_id):
        if not self.login or self.login["id"] != login_id:
            raise ValueError("This Google sign-in has expired. Start a new sign-in.")

    def cancel(self, login_id: str, expired=False):
        with self.lock:
            self._require_login(login_id)
            if self.login["status"] not in ACTIVE:
                return dict(self.login)
            self.login.update(
                status="expired" if expired else "cancelled",
                authorization_url=None,
                message="Google sign-in expired. Start again." if expired else "Google sign-in cancelled.",
            )
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                timer = threading.Timer(2, self._kill, args=(self.process,))
                timer.daemon = True
                timer.start()
            if self.timer:
                self.timer.cancel()
            return dict(self.login)

    @staticmethod
    def _kill(process):
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def _expire(self, login_id):
        with self.lock:
            if self.login and self.login["id"] == login_id:
                self.cancel(login_id, expired=True)

    def close(self):
        with self.lock:
            if self.login:
                self.cancel(self.login["id"])

    def check_credentials(self):
        import google.auth
        from google.auth.transport.requests import Request

        class TimedRequest(Request):
            def __call__(self, *args, **kwargs):
                kwargs["timeout"] = 15
                return super().__call__(*args, **kwargs)

        with self.lock:
            project = self.settings.project if self.settings else os.getenv("GOOGLE_CLOUD_PROJECT")
            location = (
                self.settings.location
                if self.settings
                else os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
            )
        vertex_model_url(project, location, "gemini-2.5-flash")
        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            with TimedRequest().session as session:
                credentials.refresh(TimedRequest(session=session))
        except Exception as exc:
            raise ValueError(
                "Could not verify Google credentials. Sign in again or check the configured credential file and network connection."
            ) from exc
        return {
            "verified": True,
            "message": "Google credentials verified.",
            "project": project,
            "location": location,
        }
