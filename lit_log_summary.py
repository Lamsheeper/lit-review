#!/usr/bin/env python3
"""Summarize LitHarvest collection logs by source service and outcome."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FULL_DOCUMENT_STATUSES = {"downloaded", "already_exists"}
SERVICE_ALIASES = {
    "semantic_scholar": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "semantic-scholar": "semantic_scholar",
    "semantic scholar": "semantic_scholar",
    "open_alex": "openalex",
    "open-alex": "openalex",
    "open alex": "openalex",
    "europepmc": "europe_pmc",
    "europe-pmc": "europe_pmc",
    "europe pmc": "europe_pmc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_doi(raw: Any) -> str:
    if raw is None:
        return ""
    doi = str(raw).strip()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi, flags=re.I)
    return doi.strip().rstrip(".").lower()


def normalize_service(raw: Any) -> str:
    service = str(raw or "").strip()
    if not service:
        return ""
    if ":" in service:
        service = service.split(":", 1)[0]
    service = re.sub(r"[\s-]+", "_", service.lower())
    return SERVICE_ALIASES.get(service, service)


def candidate_identity(candidate: dict[str, Any]) -> str:
    doi = normalize_doi(candidate.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = re.sub(r"\s+", " ", str(candidate.get("title") or "").lower()).strip()
    if title:
        return f"title:{title}|year:{candidate.get('year') or ''}"
    return ""


def candidate_filename(candidate: dict[str, Any]) -> str:
    download = candidate.get("download") if isinstance(candidate.get("download"), dict) else {}
    return str(download.get("filename") or candidate.get("filename") or "")


def record_indexes(download_log: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    indexes: dict[str, dict[str, dict[str, Any]]] = {
        "identity": {},
        "doi": {},
        "filename": {},
    }
    records = download_log.get("records") or []
    if not isinstance(records, list):
        records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = str(record.get("identity") or "")
        doi = normalize_doi(record.get("doi"))
        filename = str(record.get("filename") or "")
        download = record.get("download") if isinstance(record.get("download"), dict) else {}
        filename = filename or str(download.get("filename") or "")
        if identity:
            indexes["identity"][identity] = record
        if doi:
            indexes["doi"][doi] = record
        if filename:
            indexes["filename"][filename] = record
    return indexes


def matching_record(
    candidate: dict[str, Any],
    indexes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    identity = candidate_identity(candidate)
    if identity and identity in indexes["identity"]:
        return indexes["identity"][identity]
    doi = normalize_doi(candidate.get("doi"))
    if doi and doi in indexes["doi"]:
        return indexes["doi"][doi]
    filename = candidate_filename(candidate)
    if filename and filename in indexes["filename"]:
        return indexes["filename"][filename]
    return None


def apply_download_log_statuses(
    candidates: list[dict[str, Any]],
    download_log: dict[str, Any] | None,
) -> None:
    if not download_log:
        return
    indexes = record_indexes(download_log)
    for candidate in candidates:
        download = candidate.get("download")
        if not isinstance(download, dict):
            download = {}
            candidate["download"] = download
        record = matching_record(candidate, indexes)
        if not record:
            continue
        record_download = record.get("download") if isinstance(record.get("download"), dict) else {}
        download.update(record_download)
        if record.get("status"):
            download["status"] = record["status"]
        if record.get("filename") and not download.get("filename"):
            download["filename"] = record["filename"]


def load_candidates_from_logs(logs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    logs_dir = logs_dir.expanduser()
    manifest_path = logs_dir / "manifest.json"
    candidates_path = logs_dir / "candidates.json"
    download_log_path = logs_dir / "download_log.json"
    source_files: dict[str, str] = {}

    candidates: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        source_files["manifest"] = str(manifest_path)
        raw_candidates = manifest.get("candidates") or []
        if isinstance(raw_candidates, list):
            candidates = [dict(item) for item in raw_candidates if isinstance(item, dict)]
    elif candidates_path.exists():
        candidate_cache = load_json(candidates_path)
        source_files["candidates"] = str(candidates_path)
        raw_candidates = candidate_cache.get("candidates") or []
        if isinstance(raw_candidates, list):
            candidates = [dict(item) for item in raw_candidates if isinstance(item, dict)]
    else:
        raise FileNotFoundError(
            f"No manifest.json or candidates.json found in logs directory: {logs_dir}"
        )

    download_log = None
    if download_log_path.exists():
        download_log = load_json(download_log_path)
        source_files["download_log"] = str(download_log_path)
    apply_download_log_statuses(candidates, download_log)
    return candidates, source_files


def services_for_candidate(candidate: dict[str, Any]) -> list[str]:
    services: set[str] = set()
    for raw in candidate.get("source_apis") or []:
        service = normalize_service(raw)
        if service:
            services.add(service)
    for item in candidate.get("discovered_via") or []:
        if isinstance(item, dict):
            service = normalize_service(item.get("source"))
            if service:
                services.add(service)
    source_ids = candidate.get("source_ids") or {}
    if isinstance(source_ids, dict):
        for raw in source_ids:
            service = normalize_service(raw)
            if service:
                services.add(service)
    return sorted(services) or ["unknown"]


def download_status(candidate: dict[str, Any]) -> str:
    download = candidate.get("download") if isinstance(candidate.get("download"), dict) else {}
    return str(download.get("status") or candidate.get("download_status") or "pending")


def increment_outcomes(target: dict[str, int], status: str) -> None:
    if status in FULL_DOCUMENT_STATUSES:
        target["downloaded"] += 1
    elif status == "failed":
        target["failed"] += 1
    elif status == "abstract_only":
        target["abstract_only"] += 1
    else:
        target["other"] += 1


def empty_service_summary() -> dict[str, Any]:
    return {
        "candidates": 0,
        "downloaded": 0,
        "failed": 0,
        "abstract_only": 0,
        "other": 0,
        "with_abstract": 0,
        "status_counts": {},
    }


def summarize_logs(logs_dir: Path) -> dict[str, Any]:
    candidates, source_files = load_candidates_from_logs(logs_dir)
    status_counts: Counter[str] = Counter()
    service_counts: dict[str, dict[str, Any]] = {}
    service_combo_counts: Counter[str] = Counter()
    totals = {
        "candidates": len(candidates),
        "downloaded": 0,
        "failed": 0,
        "abstract_only": 0,
        "other": 0,
        "with_abstract": 0,
    }

    for candidate in candidates:
        status = download_status(candidate)
        status_counts[status] += 1
        increment_outcomes(totals, status)
        if candidate.get("abstract"):
            totals["with_abstract"] += 1

        services = services_for_candidate(candidate)
        service_combo_counts["+".join(services)] += 1
        for service in services:
            summary = service_counts.setdefault(service, empty_service_summary())
            summary["candidates"] += 1
            increment_outcomes(summary, status)
            if candidate.get("abstract"):
                summary["with_abstract"] += 1
            service_status_counts = Counter(summary["status_counts"])
            service_status_counts[status] += 1
            summary["status_counts"] = dict(sorted(service_status_counts.items()))

    return {
        "logs_dir": str(logs_dir),
        "generated_at": utc_now(),
        "source_files": source_files,
        "notes": [
            "downloaded counts full-document outcomes: downloaded + already_exists.",
            "Candidates found through multiple services are counted once in each service bucket.",
        ],
        "total_candidates": len(candidates),
        "totals": totals,
        "status_counts": dict(sorted(status_counts.items())),
        "by_service": {
            service: service_counts[service]
            for service in sorted(service_counts)
        },
        "service_combinations": dict(sorted(service_combo_counts.items())),
    }


def discover_default_logs_dir(cwd: Path) -> Path:
    log_dirs = [
        path
        for path in cwd.glob("collected*/logs")
        if path.is_dir()
        and ((path / "manifest.json").exists() or (path / "candidates.json").exists())
    ]
    if not log_dirs:
        raise FileNotFoundError(
            "No default collected*/logs directory found. Pass a logs directory explicitly."
        )

    def newest_log_mtime(path: Path) -> float:
        candidates = [
            file_path.stat().st_mtime
            for file_path in [
                path / "manifest.json",
                path / "candidates.json",
                path / "download_log.json",
            ]
            if file_path.exists()
        ]
        return max(candidates) if candidates else path.stat().st_mtime

    return max(log_dirs, key=newest_log_mtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a simplified JSON summary of LitHarvest candidate/download logs.",
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        help="Logs directory containing manifest.json and/or candidates.json. Defaults to newest collected*/logs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        logs_dir = (
            Path(args.logs_dir).expanduser()
            if args.logs_dir
            else discover_default_logs_dir(Path.cwd())
        )
        summary = summarize_logs(logs_dir)
        if args.output:
            write_json(Path(args.output).expanduser(), summary)
        else:
            if args.compact:
                print(json.dumps(summary, sort_keys=True))
            else:
                print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"lit-log-summary: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
