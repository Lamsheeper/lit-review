"""Feature-stage response contracts over the shared JSON parser."""

from __future__ import annotations
import logging
from typing import Any
from lit_review.core.json_response import parse_rows, repair_rows, parse_with_repair

LOGGER = logging.getLogger("litextract")


def parse_features_json(content: str) -> list[dict[str, Any]]:
    return parse_rows(content, "features", allow_array=True)


def repair_features_json(raw_content: str, client: Any) -> str:
    return repair_rows(raw_content, client, "features")


def parse_features_json_with_repair(content: str, client: Any = None) -> tuple[list[dict[str, Any]], bool]:
    return parse_with_repair(content, parse_features_json, repair_features_json, client)


def parse_curation_json(content: str) -> list[dict[str, Any]]:
    return parse_rows(content, "decisions", allow_array=False)


def repair_curation_json(raw_content: str, client: Any) -> str:
    return repair_rows(raw_content, client, "decisions")


def parse_curation_json_with_repair(content: str, client: Any = None) -> tuple[list[dict[str, Any]], bool]:
    return parse_with_repair(content, parse_curation_json, repair_curation_json, client)


def parse_annotation_json(content: str) -> list[dict[str, Any]]:
    return parse_rows(content, "annotations", allow_array=False)


def repair_annotation_json(raw_content: str, client: Any) -> str:
    return repair_rows(raw_content, client, "annotations")


def parse_annotation_json_with_repair(content: str, client: Any = None) -> tuple[list[dict[str, Any]], bool]:
    return parse_with_repair(content, parse_annotation_json, repair_annotation_json, client)


def parse_scoring_json(content: str) -> list[dict[str, Any]]:
    return parse_rows(content, "scores", allow_array=False)


def repair_scoring_json(raw_content: str, client: Any) -> str:
    return repair_rows(raw_content, client, "scores")


def parse_scoring_json_with_repair(content: str, client: Any = None) -> tuple[list[dict[str, Any]], bool]:
    return parse_with_repair(content, parse_scoring_json, repair_scoring_json, client)


def parse_llm_merge_json(content: str) -> list[dict[str, Any]]:
    return parse_rows(content, "decisions", allow_array=False)


def repair_llm_merge_json(raw_content: str, client: Any) -> str:
    return repair_rows(raw_content, client, "decisions")


def parse_llm_merge_json_with_repair(content: str, client: Any = None) -> tuple[list[dict[str, Any]], bool]:
    return parse_with_repair(content, parse_llm_merge_json, repair_llm_merge_json, client)
