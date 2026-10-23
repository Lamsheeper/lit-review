"""Core llm."""

from __future__ import annotations


import json


import urllib.error

import urllib.request


from typing import Any


from lit_review.core.chat import ChatCompletionClient


def clean_api_key(api_key: str) -> str:
    return api_key.strip().strip("\"'“”‘’")


def native_gemini_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/openai"):
        base = base[: -len("/openai")]
    return base or "https://generativelanguage.googleapis.com/v1beta"


def is_gemini_base_url(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in base_url.lower()


def should_use_gemini_native_client(
    *,
    api_provider: str,
    base_url: str,
    input_mode: str,
) -> bool:
    if api_provider == "gemini":
        return True
    if api_provider == "openai":
        return False
    return input_mode == "pdf" and is_gemini_base_url(base_url)


def strip_data_url(value: str) -> str:
    if "," in value and value.lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def gemini_thinking_config(model: str, reasoning_effort: str | None) -> dict[str, Any] | None:
    if not reasoning_effort:
        return None
    effort = reasoning_effort.strip().lower()
    model_name = model.lower()
    if "gemini-3" in model_name:
        if effort in {"minimal", "low", "medium", "high"}:
            return {"thinkingLevel": effort}
        return None
    budgets = {
        "none": 0,
        "minimal": 1024,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }
    if effort in budgets:
        return {"thinkingBudget": budgets[effort]}
    return None


def gemini_parts_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"text": str(item)})
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"text", "input_text"}:
            parts.append({"text": str(item.get("text") or "")})
            continue
        if item_type in {"file", "input_file"}:
            file_payload = item.get("file") if isinstance(item.get("file"), dict) else item
            file_data = str(file_payload.get("file_data") or "")
            file_uri = str(file_payload.get("file_uri") or file_payload.get("file_url") or "")
            mime_type = str(file_payload.get("mime_type") or "application/pdf")
            if file_data:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": strip_data_url(file_data),
                        }
                    }
                )
            elif file_uri:
                parts.append(
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri,
                        }
                    }
                )
            else:
                raise ValueError("Gemini file input requires file_data or file_uri.")
            continue
        raise ValueError(f"Unsupported Gemini content part type: {item_type!r}.")
    return parts


def build_gemini_generate_content_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        parts = gemini_parts_from_content(message.get("content") or "")
        if role in {"system", "developer"}:
            system_parts.extend(parts)
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )

    generation_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if response_format and response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"
    thinking_config = gemini_thinking_config(model, reasoning_effort)
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config

    payload: dict[str, Any] = {
        "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


class GeminiGenerateContentClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = clean_api_key(api_key)
        self.model = model
        self.base_url = native_gemini_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = build_gemini_generate_content_payload(
            messages=messages,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        request = urllib.request.Request(
            f"{self.base_url}/models/{self.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(str(part.get("text") or "") for part in parts)
            text = text.strip()
            if not text:
                finish_reason = str(candidate.get("finishReason") or "unknown")
                raise RuntimeError(f"Gemini returned an empty response (finish_reason={finish_reason}).")
            return text
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini API response: {data}") from exc


def make_llm_client(
    *,
    api_key: str,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    api_provider: str = "auto",
    input_mode: str = "text",
    timeout: float | None = None,
) -> Any:
    client_timeout = 120.0 if timeout is None else max(1.0, float(timeout))
    if should_use_gemini_native_client(
        api_provider=api_provider,
        base_url=base_url,
        input_mode=input_mode,
    ):
        return GeminiGenerateContentClient(
            api_key=clean_api_key(api_key),
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout=client_timeout,
        )
    return ChatCompletionClient(
        api_key=clean_api_key(api_key),
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        timeout=client_timeout,
    )
