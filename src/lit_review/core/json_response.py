"""Parse model JSON and perform one bounded repair attempt."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

LOGGER = logging.getLogger("lit_review.responses")


def parse_json_content(content: str) -> Any:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as initial:
        candidates = []
        for opening, closing in (("{", "}"), ("[", "]")):
            start, end = text.find(opening), text.rfind(closing)
            if start >= 0 and end > start:
                candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise initial


def parse_json_object(content: str) -> dict:
    payload = parse_json_content(content)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def parse_rows(content: str, key: str, *, allow_array: bool = False) -> list[dict]:
    payload = parse_json_content(content)
    if allow_array and isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and key in payload:
        rows = payload[key]
    else:
        raise ValueError(f"Expected a JSON object with {key}")
    if not isinstance(rows, list):
        raise ValueError(f"Expected {key} to be a list")
    return [row for row in rows if isinstance(row, dict)]


def repair_rows(raw_content: str, client: Any, key: str) -> str:
    return client.chat(
        [
            {
                "role": "system",
                "content": f"You repair malformed JSON. Return only one valid JSON object with a {key} array. Do not add Markdown fences or explanation.",
            },
            {
                "role": "user",
                "content": f'Repair this response into valid JSON with exactly one top-level key, {key}. Preserve recoverable objects and fields, escaping quotes and newlines in strings. If none can be recovered, return {{"{key}": []}}.\n\nMALFORMED RESPONSE:\n{raw_content[:20000]}',
            },
        ],
        temperature=0.0,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )


def parse_with_repair(content: str, parser: Callable, repair: Callable, client: Any = None):
    try:
        return parser(content), False
    except ValueError as original:
        if client is None:
            raise
        LOGGER.warning("Malformed model JSON; requesting one repair: %s", original)
        repaired = repair(content, client)
        try:
            return parser(repaired), True
        except ValueError as failure:
            raise ValueError(f"Model JSON parse failed: {original}; repair failed: {failure}") from failure
