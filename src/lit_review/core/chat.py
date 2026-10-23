"""Core chat."""

from __future__ import annotations


import dataclasses

import json

import logging


import urllib.error

import urllib.request


from typing import Any


LOGGER = logging.getLogger("litsynth")


def clean_api_key(api_key: str) -> str:
    """Remove copy/pasted quote wrappers that break HTTP header encoding."""
    return api_key.strip().strip("\"'“”‘’")


@dataclasses.dataclass
class ChatResult:
    content: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class ChatCompletionClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        reasoning_effort: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = clean_api_key(api_key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return self.chat_result(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        ).content

    def chat_result(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format

        def send_request(request_payload: dict[str, Any]) -> dict[str, Any]:
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            data = send_request(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if response_format is not None and exc.code in {400, 422}:
                LOGGER.warning("LLM API rejected response_format; retrying without JSON mode.")
                payload.pop("response_format", None)
                try:
                    data = send_request(payload)
                except urllib.error.HTTPError as retry_exc:
                    retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"LLM API returned HTTP {retry_exc.code}: {retry_detail}"
                    ) from retry_exc
                except urllib.error.URLError as retry_exc:
                    raise RuntimeError(f"LLM API request failed: {retry_exc}") from retry_exc
            else:
                raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        try:
            choice = data["choices"][0]
            content = str(choice["message"]["content"]).strip()
            usage = data.get("usage")
            if isinstance(usage, dict):
                usage_payload = usage
            else:
                usage_payload = None
            return ChatResult(
                content=content,
                finish_reason=choice.get("finish_reason"),
                usage=usage_payload,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response: {data}") from exc
