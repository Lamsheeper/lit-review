"""Core http."""

from __future__ import annotations


import gzip


import json


import os


import ssl


import tempfile

import threading

import time

import urllib.error

import urllib.parse

import urllib.request

import zlib


from pathlib import Path

from typing import Any

try:
    import certifi
except ImportError:  # pragma: no cover - system Python may provide a usable CA store
    certifi = None


VERSION = "0.2.0"


USER_AGENT_NAME = f"LitHarvest/{VERSION}"


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class NonRetryableDownloadError(RuntimeError):
    """Raised when another attempt at the same URL cannot turn it into a PDF."""


class HttpClient:
    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 2,
        backoff: float = 1.5,
        rate_limit_delay: float = 0.1,
        email: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.rate_limit_delay = rate_limit_delay
        self.email = email
        self.ssl_context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
        self._last_request_at = 0.0
        self._rate_lock = threading.Lock()

    @property
    def user_agent(self) -> str:
        if self.email:
            return f"{USER_AGENT_NAME} (mailto:{self.email})"
        return USER_AGENT_NAME

    def _build_url(self, url: str, params: dict[str, Any] | None = None) -> str:
        if not params:
            return url
        clean_params = {key: value for key, value in params.items() if value is not None}
        query = urllib.parse.urlencode(clean_params, doseq=True)
        separator = "&" if urllib.parse.urlparse(url).query else "?"
        return f"{url}{separator}{query}"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, application/pdf;q=0.8, */*;q=0.5",
            "User-Agent": self.user_agent,
        }
        if extra:
            headers.update(extra)
        return headers

    def _wait_for_rate_limit(self) -> None:
        with self._rate_lock:
            if self.rate_limit_delay <= 0:
                self._last_request_at = time.monotonic()
                return
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.rate_limit_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def _retry_sleep(self, attempt: int, headers: dict[str, Any] | None = None) -> None:
        retry_after = None
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                delay = min(float(retry_after), 30.0)
            except ValueError:
                delay = min(self.backoff * (attempt + 1), 30.0)
        else:
            delay = min(self.backoff * (attempt + 1), 30.0)
        time.sleep(delay)

    def request_bytes(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, dict[str, Any], str]:
        full_url = self._build_url(url, params)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(full_url, headers=self._headers(headers))
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self.ssl_context
                ) as response:
                    self._last_request_at = time.monotonic()
                    body = response.read()
                    response_headers = dict(response.headers.items())
                    body = decode_http_body(body, response_headers)
                    return body, response_headers, response.geturl()
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    self._retry_sleep(attempt, dict(exc.headers.items()))
                    continue
                detail = exc.read(500).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} for {full_url}: {detail}") from exc
            except urllib.error.URLError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if attempt < self.retries:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(f"Request failed for {full_url}: {exc}") from exc
        raise RuntimeError(f"Request failed for {full_url}: {last_error}")

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body, _, final_url = self.request_bytes(url, params=params, headers=headers)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            preview = body[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"Expected JSON from {final_url}, received: {preview}") from exc

    def download_pdf(
        self,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                url,
                headers=self._headers({"Accept": "application/pdf, */*;q=0.2"}),
            )
            temp_path: Path | None = None
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self.ssl_context
                ) as response:
                    self._last_request_at = time.monotonic()
                    content_type = response.headers.get("Content-Type", "")
                    content_length = safe_int(response.headers.get("Content-Length"))
                    if content_length is not None and content_length > max_bytes:
                        raise NonRetryableDownloadError(
                            f"PDF is larger than limit: {content_length} bytes > {max_bytes} bytes"
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        "wb", delete=False, dir=str(destination.parent), suffix=".tmp"
                    ) as handle:
                        temp_path = Path(handle.name)
                        first_chunk = response.read(8192)
                        if not is_pdf_response(content_type, first_chunk):
                            preview = first_chunk[:80].decode("utf-8", errors="replace")
                            raise NonRetryableDownloadError(
                                f"URL did not return a PDF; content-type={content_type!r}; "
                                f"preview={preview!r}"
                            )
                        handle.write(first_chunk)
                        total = len(first_chunk)
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                raise NonRetryableDownloadError(
                                    f"PDF exceeded limit while downloading: {total} bytes"
                                )
                            handle.write(chunk)

                    os.replace(temp_path, destination)
                    return {
                        "status": "downloaded",
                        "url": url,
                        "final_url": response.geturl(),
                        "path": str(destination),
                        "bytes": total,
                        "content_type": content_type,
                    }
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    self._retry_sleep(attempt, dict(exc.headers.items()))
                    continue
                detail = exc.read(200).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} while downloading PDF: {detail}") from exc
            except NonRetryableDownloadError as exc:
                self._last_request_at = time.monotonic()
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"PDF download failed for {url}: {exc}") from exc
            except (urllib.error.URLError, RuntimeError) as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                if attempt < self.retries:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(f"PDF download failed for {url}: {exc}") from exc
        raise RuntimeError(f"PDF download failed for {url}: {last_error}")


def decode_http_body(body: bytes, headers: dict[str, Any]) -> bytes:
    encoding = str(headers.get("Content-Encoding") or headers.get("content-encoding") or "").lower()
    if not body:
        return body
    if "gzip" in encoding:
        try:
            return gzip.decompress(body)
        except OSError:
            return body
    if "deflate" in encoding:
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except zlib.error:
                return body
    return body


def is_pdf_response(content_type: str | None, first_chunk: bytes) -> bool:
    content_type = (content_type or "").lower()
    if not first_chunk:
        return False
    if first_chunk.startswith(b"%PDF-"):
        return True
    if "pdf" in content_type and not looks_like_html(first_chunk):
        return True
    return False


def looks_like_html(content: bytes) -> bool:
    sample = content[:200].lstrip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html")
