"""Shared Vertex AI routing and refreshable Application Default Credentials."""

from __future__ import annotations

import os
import re
import threading


def use_vertex(api_key: str | None = None, *, auth_mode: str = "auto", project: str | None = None) -> bool:
    if auth_mode not in {"auto", "api_key", "adc"}:
        raise ValueError("auth_mode must be 'auto', 'api_key', or 'adc'.")
    if auth_mode != "auto":
        return auth_mode == "adc"
    enabled = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    return bool(project or enabled or (not api_key and os.getenv("GOOGLE_CLOUD_PROJECT")))


def vertex_model_url(project: str | None, location: str | None, model: str) -> str:
    project = (project or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1").strip()
    model = model.removeprefix("models/")
    if not project:
        raise ValueError("Vertex AI requires vertex_project or GOOGLE_CLOUD_PROJECT.")
    for name, value in (("project", project), ("location", location), ("model", model)):
        if not re.fullmatch(r"[A-Za-z0-9._@-]+", value):
            raise ValueError(f"Invalid Vertex AI {name}.")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}"


class ADCAuth:
    """Discover credentials lazily; refresh tokens before requests as needed."""

    def __init__(self) -> None:
        self._credentials = None
        self._lock = threading.Lock()

    def headers(self, url: str) -> dict[str, str]:
        import google.auth
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request

        headers: dict[str, str] = {}
        try:
            with self._lock:
                if self._credentials is None:
                    self._credentials, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                self._credentials.before_request(Request(), "POST", url, headers)
        except GoogleAuthError as exc:
            raise RuntimeError(
                "Vertex AI authentication failed. Run `gcloud auth application-default login` "
                "or configure Application Default Credentials for this environment."
            ) from exc
        return headers
