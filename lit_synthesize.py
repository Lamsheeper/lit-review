#!/usr/bin/env python3
"""LitSynth: convert collected PDFs into an evidence-bound review report.

This is the downstream companion to ``lit_harvest.py``. It expects a folder of
collected PDFs plus, when available, the LitHarvest manifest. The pipeline is:

1. convert PDFs to basic text-only Markdown,
2. chunk and index those Markdown papers with SQLite FTS/BM25,
3. retrieve evidence for a goal file, and
4. write a Markdown report with either an LLM API call or an offline fallback.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from lit_harvest import content_tokens, dedupe_preserve, normalize_title, slugify
except ImportError:  # pragma: no cover - defensive standalone fallback
    WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }

    def content_tokens(text: str | None) -> list[str]:
        if not text:
            return []
        return [
            match.group(0).lower()
            for match in WORD_RE.finditer(text)
            if match.group(0).lower() not in STOPWORDS
        ]

    def dedupe_preserve(values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                output.append(value)
        return output

    def normalize_title(value: str | None) -> str:
        if not value:
            return ""
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def slugify(value: str | None, max_length: int = 80) -> str:
        slug = normalize_title(value).replace(" ", "_")
        return slug[:max_length].strip("_")


VERSION = "0.1.0"
LOGGER = logging.getLogger("litsynth")
DEFAULT_CHUNK_WORDS = 700
DEFAULT_OVERLAP_WORDS = 80
DEFAULT_MIN_CHUNK_WORDS = 40
DEFAULT_MODEL = "gpt-4o-mini"
PAGE_HEADING_RE = re.compile(r"^## Page\s+(\d+)\s*$", re.MULTILINE)
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
CONFIG_ARG_FIELDS: dict[str, list[tuple[str, str]]] = {
    "convert": [
        ("papers", "--papers"),
        ("manifest", "--manifest"),
        ("out", "--out"),
        ("extractor", "--extractor"),
        ("force", "--force"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "index": [
        ("run", "--run"),
        ("chunk_words", "--chunk-words"),
        ("overlap_words", "--overlap-words"),
        ("min_chunk_words", "--min-chunk-words"),
        ("force", "--force"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "search": [
        ("run", "--run"),
        ("query", "--query"),
        ("limit", "--limit"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "write": [
        ("run", "--run"),
        ("goal", "--goal"),
        ("output", "--output"),
        ("api_key", "--api-key"),
        ("model", "--model"),
        ("base_url", "--base-url"),
        ("max_questions", "--max-questions"),
        ("per_question", "--per-question"),
        ("max_evidence", "--max-evidence"),
        ("evidence_chars", "--evidence-chars"),
        ("temperature", "--temperature"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "run": [
        ("papers", "--papers"),
        ("manifest", "--manifest"),
        ("goal", "--goal"),
        ("out", "--out"),
        ("report", "--report"),
        ("extractor", "--extractor"),
        ("chunk_words", "--chunk-words"),
        ("overlap_words", "--overlap-words"),
        ("min_chunk_words", "--min-chunk-words"),
        ("api_key", "--api-key"),
        ("model", "--model"),
        ("base_url", "--base-url"),
        ("max_questions", "--max-questions"),
        ("per_question", "--per-question"),
        ("max_evidence", "--max-evidence"),
        ("evidence_chars", "--evidence-chars"),
        ("temperature", "--temperature"),
        ("force", "--force"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
}


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    pdf_path: str = ""
    md_path: str | None = None
    conversion: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    paper_id: str
    title: str
    year: int | None
    pages: list[int]
    text: str
    token_count: int


class MissingExtractor(RuntimeError):
    """Raised when a requested PDF text extractor is not installed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n[truncated]"


def load_manifest(manifest_path: Path | None) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return read_json(manifest_path)


def resolve_download_path(raw_path: str, papers_dir: Path, manifest_path: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        papers_dir / path.name,
    ]
    if manifest_path is not None:
        candidates.extend(
            [
                manifest_path.parent / path,
                manifest_path.parent.parent / path.name,
                manifest_path.parent.parent / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return papers_dir / path.name


def build_paper_id(candidate: dict[str, Any] | None, pdf_path: Path) -> str:
    candidate = candidate or {}
    title = candidate.get("title") or title_from_pdf_filename(pdf_path)
    digest_source = candidate.get("doi") or title or pdf_path.stem
    digest = hashlib.sha1(str(digest_source).encode("utf-8")).hexdigest()[:8]
    prefix = slugify(str(title), max_length=48) or "paper"
    return f"{prefix}_{digest}"


def title_from_pdf_filename(pdf_path: Path) -> str:
    stem = pdf_path.stem
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0].isdigit():
        title_parts = parts[2:-1] or parts[1:-1]
        return " ".join(title_parts).replace("-", " ").title()
    return stem.replace("_", " ").replace("-", " ").title()


def year_from_pdf_filename(pdf_path: Path) -> int | None:
    match = re.match(r"^(\d{4})[_-]", pdf_path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def unique_paper_id(base_id: str, used: set[str]) -> str:
    if base_id not in used:
        used.add(base_id)
        return base_id
    counter = 2
    while f"{base_id}_{counter}" in used:
        counter += 1
    value = f"{base_id}_{counter}"
    used.add(value)
    return value


def collect_paper_records(
    papers_dir: Path,
    manifest: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> list[PaperRecord]:
    papers_dir = papers_dir.expanduser()
    records: list[PaperRecord] = []
    seen_paths: set[Path] = set()
    used_ids: set[str] = set()

    for candidate in (manifest or {}).get("candidates", []):
        download = candidate.get("download") or {}
        status = download.get("status")
        if status not in {"downloaded", "already_exists"}:
            continue
        raw_path = download.get("path") or download.get("filename")
        if not raw_path:
            continue
        pdf_path = resolve_download_path(str(raw_path), papers_dir, manifest_path)
        if pdf_path.suffix.lower() != ".pdf":
            continue
        seen_paths.add(pdf_path.resolve() if pdf_path.exists() else pdf_path)
        paper_id = unique_paper_id(build_paper_id(candidate, pdf_path), used_ids)
        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=str(candidate.get("title") or title_from_pdf_filename(pdf_path)),
                authors=[str(author) for author in candidate.get("authors") or []],
                year=candidate.get("year"),
                doi=candidate.get("doi"),
                venue=candidate.get("venue"),
                pdf_path=str(pdf_path),
            )
        )

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        resolved = pdf_path.resolve()
        if resolved in seen_paths:
            continue
        title = title_from_pdf_filename(pdf_path)
        paper_id = unique_paper_id(build_paper_id({"title": title}, pdf_path), used_ids)
        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                year=year_from_pdf_filename(pdf_path),
                pdf_path=str(pdf_path),
            )
        )

    return sorted(records, key=lambda item: (item.year or 0, item.title.lower()))


def extract_pdf_pages(pdf_path: Path, extractor: str = "auto", timeout: int = 120) -> list[str]:
    extractor = extractor.lower()
    extractors = ["pymupdf", "pypdf", "pdftotext"] if extractor == "auto" else [extractor]
    errors: list[str] = []
    for name in extractors:
        try:
            if name == "pymupdf":
                return extract_with_pymupdf(pdf_path)
            if name == "pypdf":
                return extract_with_pypdf(pdf_path)
            if name == "pdftotext":
                return extract_with_pdftotext(pdf_path, timeout=timeout)
            raise ValueError(f"Unknown extractor: {name}")
        except MissingExtractor as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("No PDF extractor succeeded. " + " | ".join(errors))


def extract_with_pymupdf(pdf_path: Path) -> list[str]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MissingExtractor("PyMuPDF is not installed.") from exc

    pages: list[str] = []
    with fitz.open(str(pdf_path)) as document:  # pragma: no cover - optional package path
        for page in document:
            pages.append(normalize_text(page.get_text("text") or ""))
    return pages


def extract_with_pypdf(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MissingExtractor("pypdf is not installed.") from exc

    reader = PdfReader(str(pdf_path))  # pragma: no cover - optional package path
    return [normalize_text(page.extract_text() or "") for page in reader.pages]


def extract_with_pdftotext(pdf_path: Path, timeout: int = 120) -> list[str]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise MissingExtractor("pdftotext is not installed.")
    result = subprocess.run(
        [executable, "-layout", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(stderr or f"pdftotext exited with {result.returncode}")
    text = result.stdout.decode("utf-8", errors="replace")
    return [normalize_text(page) for page in text.split("\f")]


def render_paper_markdown(record: PaperRecord, pages: list[str]) -> str:
    metadata = {
        "paper_id": record.paper_id,
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "doi": record.doi,
        "venue": record.venue,
        "pdf_path": record.pdf_path,
    }
    lines = ["---"]
    for key, value in metadata.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["---", "", f"# {record.title}", ""])
    if not pages:
        lines.extend(["No extractable text recovered.", ""])
    for idx, page_text in enumerate(pages, start=1):
        lines.extend([f"## Page {idx}", "", normalize_text(page_text), ""])
    return "\n".join(lines).rstrip() + "\n"


def convert_pdfs_to_markdown(
    papers_dir: Path,
    manifest_path: Path | None,
    out_dir: Path,
    extractor: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    records = collect_paper_records(papers_dir, manifest=manifest, manifest_path=manifest_path)
    papers_md_dir = out_dir / "papers_md"
    papers_md_dir.mkdir(parents=True, exist_ok=True)

    converted: list[PaperRecord] = []
    for record in records:
        pdf_path = Path(record.pdf_path)
        md_path = papers_md_dir / f"{record.paper_id}.md"
        record.md_path = str(md_path)
        if md_path.exists() and not force:
            record.conversion = {
                "status": "already_exists",
                "extractor": extractor,
                "md_path": str(md_path),
            }
            converted.append(record)
            continue

        try:
            LOGGER.info("Converting %s", pdf_path)
            pages = extract_pdf_pages(pdf_path, extractor=extractor)
            md_path.write_text(render_paper_markdown(record, pages), encoding="utf-8")
            text_pages = sum(1 for page in pages if page.strip())
            record.conversion = {
                "status": "converted" if text_pages else "no_text",
                "extractor": extractor,
                "md_path": str(md_path),
                "page_count": len(pages),
                "text_page_count": text_pages,
            }
        except Exception as exc:
            LOGGER.warning("Could not convert %s: %s", pdf_path, exc)
            record.md_path = None
            record.conversion = {
                "status": "failed",
                "extractor": extractor,
                "reason": str(exc),
            }
        converted.append(record)

    payload = {
        "project": "LitSynth",
        "version": VERSION,
        "generated_at": utc_now(),
        "papers_dir": str(papers_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "papers_md_dir": str(papers_md_dir),
        "paper_count": len(converted),
        "converted_count": sum(
            1
            for record in converted
            if record.conversion.get("status") in {"converted", "already_exists", "no_text"}
        ),
        "failed_count": sum(
            1 for record in converted if record.conversion.get("status") == "failed"
        ),
        "papers": [dataclasses.asdict(record) for record in converted],
    }
    write_json(out_dir / "paper_index.json", payload)
    return payload


def strip_front_matter(markdown: str) -> str:
    if not markdown.startswith("---"):
        return markdown
    match = re.match(r"^---\s*\n.*?\n---\s*\n", markdown, flags=re.DOTALL)
    if match:
        return markdown[match.end() :]
    return markdown


def split_markdown_pages(markdown: str) -> list[tuple[int | None, str]]:
    body = strip_front_matter(markdown)
    matches = list(PAGE_HEADING_RE.finditer(body))
    if not matches:
        return [(None, normalize_text(body))]

    pages: list[tuple[int | None, str]] = []
    for idx, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        pages.append((page_number, normalize_text(body[start:end])))
    return pages


def chunk_words(text: str, chunk_words_count: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", normalize_text(text))
    if not words:
        return []
    if len(words) <= chunk_words_count:
        return [" ".join(words)]
    step = max(1, chunk_words_count - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words_count]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words_count >= len(words):
            break
    return chunks


def resolve_md_path(md_path: str, run_dir: Path) -> Path:
    path = Path(md_path).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        Path.cwd() / path,
        run_dir / path,
        run_dir / "papers_md" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def chunk_markdown_paper(
    record: dict[str, Any],
    run_dir: Path,
    chunk_words_count: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[ChunkRecord]:
    md_path_value = record.get("md_path")
    if not md_path_value:
        return []
    md_path = resolve_md_path(str(md_path_value), run_dir)
    if not md_path.exists():
        LOGGER.warning("Markdown file missing for %s: %s", record.get("paper_id"), md_path)
        return []

    markdown = md_path.read_text(encoding="utf-8", errors="replace")
    chunks: list[ChunkRecord] = []
    for page_number, page_text in split_markdown_pages(markdown):
        page = page_number or 0
        for idx, text in enumerate(
            chunk_words(page_text, chunk_words_count=chunk_words_count, overlap_words=overlap_words),
            start=1,
        ):
            token_count = len(re.findall(r"\S+", text))
            if token_count < min_chunk_words:
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{record['paper_id']}:p{page}:c{idx}",
                    paper_id=record["paper_id"],
                    title=record.get("title") or record["paper_id"],
                    year=record.get("year"),
                    pages=[page] if page else [],
                    text=text,
                    token_count=token_count,
                )
            )
    return chunks


def build_index(
    run_dir: Path,
    db_path: Path | None = None,
    chunk_words_count: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    force: bool = False,
) -> dict[str, Any]:
    paper_index_path = run_dir / "paper_index.json"
    if not paper_index_path.exists():
        raise FileNotFoundError(f"paper_index.json does not exist: {paper_index_path}")

    db_path = db_path or run_dir / "index.sqlite"
    if db_path.exists() and force:
        db_path.unlink()

    paper_index = read_json(paper_index_path)
    papers = [
        paper
        for paper in paper_index.get("papers", [])
        if paper.get("md_path")
        and paper.get("conversion", {}).get("status") in {"converted", "already_exists", "no_text"}
    ]
    chunks: list[ChunkRecord] = []
    for paper in papers:
        chunks.extend(
            chunk_markdown_paper(
                paper,
                run_dir=run_dir,
                chunk_words_count=chunk_words_count,
                overlap_words=overlap_words,
                min_chunk_words=min_chunk_words,
            )
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DROP TABLE IF EXISTS papers")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(
            """
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                year INTEGER,
                doi TEXT,
                venue TEXT,
                pdf_path TEXT,
                md_path TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                title TEXT NOT NULL,
                year INTEGER,
                pages_json TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, text)"
        )
        for paper in papers:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers (
                    paper_id, title, authors_json, year, doi, venue, pdf_path, md_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper["paper_id"],
                    paper.get("title") or paper["paper_id"],
                    json.dumps(paper.get("authors") or []),
                    paper.get("year"),
                    paper.get("doi"),
                    paper.get("venue"),
                    paper.get("pdf_path"),
                    paper.get("md_path"),
                ),
            )
        for chunk in chunks:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, paper_id, title, year, pages_json, text, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.paper_id,
                    chunk.title,
                    chunk.year,
                    json.dumps(chunk.pages),
                    chunk.text,
                    chunk.token_count,
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, title, text) VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.title, chunk.text),
            )
        conn.commit()

    chunk_rows = [dataclasses.asdict(chunk) for chunk in chunks]
    write_jsonl(run_dir / "chunks.jsonl", chunk_rows)
    payload = {
        "project": "LitSynth",
        "version": VERSION,
        "generated_at": utc_now(),
        "db_path": str(db_path),
        "paper_count": len(papers),
        "chunk_count": len(chunks),
        "chunk_words": chunk_words_count,
        "overlap_words": overlap_words,
        "min_chunk_words": min_chunk_words,
    }
    write_json(run_dir / "index_meta.json", payload)
    return payload


def build_fts_query(query: str, max_terms: int = 14) -> str:
    tokens = dedupe_preserve(content_tokens(query))[:max_terms]
    parts = []
    for token in tokens:
        escaped = token.replace('"', '""')
        parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def row_to_search_result(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    chunk_id, paper_id, title, year, pages_json, text, score = row
    try:
        pages = json.loads(pages_json)
    except json.JSONDecodeError:
        pages = []
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "pages": pages,
        "text": text,
        "score": score,
    }


def fallback_search(conn: sqlite3.Connection, query: str, limit: int) -> list[dict[str, Any]]:
    query_terms = set(content_tokens(query))
    if not query_terms:
        return []
    rows = conn.execute(
        "SELECT chunk_id, paper_id, title, year, pages_json, text FROM chunks"
    ).fetchall()
    scored: list[tuple[float, tuple[Any, ...]]] = []
    for row in rows:
        chunk_id, paper_id, title, year, pages_json, text = row
        text_terms = Counter(content_tokens(f"{title} {text}"))
        score = sum(text_terms[term] for term in query_terms)
        if score:
            scored.append((-float(score), (chunk_id, paper_id, title, year, pages_json, text, -score)))
    scored.sort(key=lambda item: item[0])
    return [row_to_search_result(row) for _, row in scored[:limit]]


def search_index(db_path: Path, query: str, limit: int = 10) -> list[dict[str, Any]]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.paper_id,
                    c.title,
                    c.year,
                    c.pages_json,
                    c.text,
                    bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            return [row_to_search_result(row) for row in rows]
        except sqlite3.OperationalError:
            LOGGER.debug("FTS query failed; falling back to token overlap.", exc_info=True)
            return fallback_search(conn, query, limit)


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "unknown page"
    if len(pages) == 1:
        return f"page {pages[0]}"
    return "pages " + ", ".join(str(page) for page in pages)


def derive_goal_title(goal_text: str) -> str:
    for line in goal_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                return title
        if stripped:
            return stripped[:120]
    return "Literature Review Report"


def markdown_headings(text: str) -> list[str]:
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading:
                headings.append(heading)
    return headings


def offline_plan_goal(goal_text: str, max_questions: int = 6) -> dict[str, Any]:
    headings = markdown_headings(goal_text)
    title = derive_goal_title(goal_text)
    questions: list[str] = []
    for heading in headings[1:]:
        if len(questions) >= max_questions:
            break
        questions.append(f"What does the literature establish about {heading}?")

    if len(questions) < max_questions:
        top_terms = dedupe_preserve(content_tokens(goal_text))[:24]
        for idx in range(0, len(top_terms), 4):
            phrase = " ".join(top_terms[idx : idx + 4])
            if phrase and len(questions) < max_questions:
                questions.append(f"What evidence is available for {phrase}?")

    defaults = [
        "What concepts and taxonomies are most relevant to the goal?",
        "What methods, datasets, and validation criteria does the literature use?",
        "Where do papers agree, disagree, or leave important gaps?",
        "What practical synthesis or taxonomy follows from the evidence?",
    ]
    for question in defaults:
        if len(questions) >= max_questions:
            break
        questions.append(question)

    return {
        "title": title,
        "questions": dedupe_preserve(questions)[:max_questions],
        "outline": [
            "Executive Summary",
            "Research Questions",
            "Evidence Map",
            "Synthesis",
            "Recommended Answer",
            "Limitations",
        ],
    }


class ChatCompletionClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response: {data}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def llm_plan_goal(
    goal_text: str,
    client: ChatCompletionClient,
    max_questions: int,
    temperature: float = 0.1,
) -> dict[str, Any]:
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a literature-review planning agent. Return strict JSON only. "
                    "Do not include Markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Plan an evidence-bound literature review for this goal. "
                    f"Return JSON with keys title, questions, and outline. "
                    f"Use at most {max_questions} focused research questions.\n\n"
                    f"GOAL:\n{goal_text}"
                ),
            },
        ],
        temperature=temperature,
        max_tokens=1200,
    )
    plan = extract_json_object(content)
    questions = [str(item) for item in plan.get("questions", []) if str(item).strip()]
    outline = [str(item) for item in plan.get("outline", []) if str(item).strip()]
    if not questions:
        return offline_plan_goal(goal_text, max_questions=max_questions)
    return {
        "title": str(plan.get("title") or derive_goal_title(goal_text)),
        "questions": questions[:max_questions],
        "outline": outline
        or ["Executive Summary", "Evidence Map", "Synthesis", "Recommendations"],
    }


def select_evidence(
    db_path: Path,
    questions: list[str],
    per_question: int = 6,
    max_total: int = 30,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in questions:
        for result in search_index(db_path, question, limit=per_question):
            chunk_id = result["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            result["question"] = question
            evidence.append(result)
            if len(evidence) >= max_total:
                return evidence
    return evidence


def format_evidence_for_prompt(
    evidence: list[dict[str, Any]],
    max_chars_per_chunk: int = 1400,
) -> str:
    blocks: list[str] = []
    for item in evidence:
        pages = format_pages(item.get("pages") or [])
        year = item.get("year") or "n.d."
        blocks.append(
            "\n".join(
                [
                    f"[{item['chunk_id']}] {item['title']} ({year}), {pages}",
                    f"Retrieved for: {item.get('question', '')}",
                    truncate_text(item["text"], max_chars_per_chunk),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def write_llm_report(
    goal_text: str,
    plan: dict[str, Any],
    evidence: list[dict[str, Any]],
    client: ChatCompletionClient,
    output_path: Path,
    temperature: float = 0.2,
    evidence_chars: int = 1400,
) -> str:
    evidence_text = format_evidence_for_prompt(evidence, max_chars_per_chunk=evidence_chars)
    prompt = f"""GOAL
{goal_text}

RESEARCH QUESTIONS
{json.dumps(plan.get("questions", []), indent=2)}

REQUESTED OUTLINE
{json.dumps(plan.get("outline", []), indent=2)}

EVIDENCE CHUNKS
{evidence_text}

Write a literature-review report in Markdown that directly fulfills the goal.
Use only the evidence chunks above. Cite claims with chunk IDs exactly like
[paper_id:p3:c1]. If the evidence is insufficient, state the limitation instead
of filling the gap from memory. End with a short "Evidence Gaps" section.
"""
    report = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are an evidence-bound literature-review synthesis agent. "
                    "You may reason across sources, but every factual claim must be "
                    "grounded in the provided evidence chunk IDs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=5000,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.rstrip() + "\n", encoding="utf-8")
    return report


def top_terms_from_evidence(evidence: list[dict[str, Any]], limit: int = 12) -> list[str]:
    counter: Counter[str] = Counter()
    for item in evidence:
        counter.update(content_tokens(item.get("text", "")))
    return [term for term, _ in counter.most_common(limit)]


def write_offline_report(
    goal_text: str,
    plan: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_path: Path,
) -> str:
    title = plan.get("title") or derive_goal_title(goal_text)
    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        (
            "This report was generated in offline mode, so it uses deterministic "
            "BM25 retrieval and extractive evidence notes instead of an LLM "
            "synthesis pass."
        ),
        "",
    ]
    terms = top_terms_from_evidence(evidence)
    if terms:
        lines.extend(
            [
                "The strongest retrieved terminology centers on: "
                + ", ".join(f"`{term}`" for term in terms[:10])
                + ".",
                "",
            ]
        )

    lines.extend(["## Research Questions", ""])
    for question in plan.get("questions", []):
        lines.append(f"- {question}")
    lines.append("")

    lines.extend(["## Evidence Map", ""])
    by_question: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        by_question.setdefault(str(item.get("question") or "General"), []).append(item)
    for question, items in by_question.items():
        lines.extend([f"### {question}", ""])
        for item in items[:5]:
            pages = format_pages(item.get("pages") or [])
            snippet = truncate_text(item["text"], 550).replace("\n", " ")
            lines.append(
                f"- [{item['chunk_id']}] {item['title']} ({item.get('year') or 'n.d.'}, "
                f"{pages}): {snippet}"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommended Answer",
            "",
            (
                "Use the evidence map above as the first-pass review scaffold. "
                "Run the same command with an API key to turn this scaffold into a "
                "narrative synthesis while preserving the chunk citations."
            ),
            "",
            "## Limitations",
            "",
            (
                "This offline report does not infer beyond retrieved passages. It is "
                "best used to inspect coverage, identify missing papers, and verify "
                "that the index is retrieving useful evidence for the goal."
            ),
            "",
        ]
    )
    report = "\n".join(lines).rstrip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def verify_report_citations(report: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_ids = {item["chunk_id"] for item in evidence}
    cited_ids = set(CITATION_RE.findall(report))
    invalid = sorted(cited_id for cited_id in cited_ids if cited_id not in evidence_ids)
    long_uncited: list[str] = []
    for paragraph in re.split(r"\n\s*\n", report):
        stripped = paragraph.strip()
        if len(stripped) > 280 and not CITATION_RE.search(stripped):
            long_uncited.append(truncate_text(stripped, 180))
    return {
        "evidence_count": len(evidence),
        "citation_count": len(cited_ids),
        "invalid_citations": invalid,
        "long_uncited_paragraph_count": len(long_uncited),
        "long_uncited_paragraphs": long_uncited[:10],
    }


def write_report(
    run_dir: Path,
    goal_path: Path,
    output_path: Path,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = "https://api.openai.com/v1",
    max_questions: int = 6,
    per_question: int = 6,
    max_evidence: int = 30,
    evidence_chars: int = 1400,
    temperature: float = 0.2,
) -> dict[str, Any]:
    db_path = run_dir / "index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Index database does not exist: {db_path}")
    goal_text = goal_path.read_text(encoding="utf-8", errors="replace")
    client = (
        ChatCompletionClient(api_key=api_key, model=model, base_url=base_url)
        if api_key
        else None
    )

    if client:
        LOGGER.info("Planning review with LLM model %s", model)
        try:
            plan = llm_plan_goal(
                goal_text, client=client, max_questions=max_questions, temperature=temperature
            )
        except Exception as exc:
            LOGGER.warning("LLM planning failed; using offline planner: %s", exc)
            plan = offline_plan_goal(goal_text, max_questions=max_questions)
    else:
        plan = offline_plan_goal(goal_text, max_questions=max_questions)

    evidence = select_evidence(
        db_path,
        questions=[str(question) for question in plan.get("questions", [])],
        per_question=per_question,
        max_total=max_evidence,
    )
    if client:
        LOGGER.info("Writing report with %d evidence chunks.", len(evidence))
        report = write_llm_report(
            goal_text=goal_text,
            plan=plan,
            evidence=evidence,
            client=client,
            output_path=output_path,
            temperature=temperature,
            evidence_chars=evidence_chars,
        )
        llm_used = True
    else:
        report = write_offline_report(
            goal_text=goal_text,
            plan=plan,
            evidence=evidence,
            output_path=output_path,
        )
        llm_used = False

    verification = verify_report_citations(report, evidence)
    trace = {
        "project": "LitSynth",
        "version": VERSION,
        "generated_at": utc_now(),
        "goal_path": str(goal_path),
        "output_path": str(output_path),
        "llm_used": llm_used,
        "model": model if llm_used else None,
        "base_url": base_url if llm_used else None,
        "plan": plan,
        "evidence": evidence,
        "verification": verification,
    }
    write_json(run_dir / "agent_trace.json", trace)
    return trace


def print_search_results(results: list[dict[str, Any]], query: str) -> None:
    print(f"# Search results for: {query}\n")
    if not results:
        print("No matching chunks found.")
        return
    for idx, item in enumerate(results, start=1):
        pages = format_pages(item.get("pages") or [])
        print(
            f"{idx}. [{item['chunk_id']}] {item['title']} "
            f"({item.get('year') or 'n.d.'}, {pages}) score={item['score']}"
        )
        print(truncate_text(item["text"], 450).replace("\n", " "))
        print()


def env_api_key() -> str | None:
    return os.getenv("LIT_SYNTH_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def env_base_url() -> str:
    return os.getenv("LIT_SYNTH_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or (
        "https://api.openai.com/v1"
    )


def env_model() -> str:
    return os.getenv("LIT_SYNTH_LLM_MODEL") or DEFAULT_MODEL


def load_synth_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Synthesis config does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Synthesis config must contain a JSON object.")
    return config


def synth_config_to_argv(config: dict[str, Any]) -> list[str]:
    command = str(config.get("command") or "").strip()
    if not command:
        raise ValueError("Synthesis config must include a command.")
    if command not in CONFIG_ARG_FIELDS:
        allowed = ", ".join(sorted(CONFIG_ARG_FIELDS))
        raise ValueError(f"Unknown synthesis config command {command!r}; expected one of {allowed}.")

    config = dict(config)
    api_key_env = config.get("api_key_env")
    if api_key_env and not config.get("api_key"):
        api_key = os.getenv(str(api_key_env))
        if api_key:
            config["api_key"] = api_key

    argv = [command]
    for key, flag in CONFIG_ARG_FIELDS[command]:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


def expand_config_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if "--config" not in argv:
        return argv

    config_index = argv.index("--config")
    try:
        config_path = Path(argv[config_index + 1]).expanduser()
    except IndexError as exc:
        raise ValueError("--config requires a JSON config path.") from exc

    remaining = argv[:config_index] + argv[config_index + 2 :]
    config_argv = synth_config_to_argv(load_synth_config(config_path))
    return config_argv + remaining


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress logs.")
    parser.add_argument("--quiet", action="store_true", help="Show warnings and errors only.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = expand_config_argv(argv)
    parser = argparse.ArgumentParser(
        description="Convert LitHarvest PDFs to Markdown, index them, and write a review report."
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a JSON synthesis config. Put this before the command, for example: "
            "lit-synthesize --config synthesize_configs/04_run_offline_smoke.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert collected PDFs to text Markdown.")
    add_common_args(convert)
    convert.add_argument("--papers", required=True, help="Folder containing collected PDFs.")
    convert.add_argument("--manifest", help="Optional LitHarvest manifest.json path.")
    convert.add_argument("--out", required=True, help="Run folder for Markdown and indexes.")
    convert.add_argument(
        "--extractor",
        default="auto",
        choices=["auto", "pymupdf", "pypdf", "pdftotext"],
        help="PDF text extractor to use.",
    )
    convert.add_argument("--force", action="store_true", help="Reconvert existing Markdown files.")

    index = subparsers.add_parser("index", help="Build the local chunk search index.")
    add_common_args(index)
    index.add_argument("--run", required=True, help="Run folder containing paper_index.json.")
    index.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    index.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    index.add_argument("--min-chunk-words", type=int, default=DEFAULT_MIN_CHUNK_WORDS)
    index.add_argument("--force", action="store_true", help="Overwrite an existing index.")

    search = subparsers.add_parser("search", help="Search the indexed paper chunks.")
    add_common_args(search)
    search.add_argument("--run", required=True, help="Run folder containing index.sqlite.")
    search.add_argument("--query", required=True, help="Search query.")
    search.add_argument("--limit", type=int, default=10, help="Maximum chunks to print.")

    write = subparsers.add_parser("write", help="Write a goal-driven literature report.")
    add_common_args(write)
    write.add_argument("--run", required=True, help="Run folder containing index.sqlite.")
    write.add_argument("--goal", required=True, help="Goal Markdown file.")
    write.add_argument("--output", required=True, help="Output report Markdown path.")
    write.add_argument("--api-key", help="LLM API key. Defaults to env vars.")
    write.add_argument("--model", default=env_model(), help="LLM model name.")
    write.add_argument("--base-url", default=env_base_url(), help="OpenAI-compatible base URL.")
    write.add_argument("--max-questions", type=int, default=6)
    write.add_argument("--per-question", type=int, default=6)
    write.add_argument("--max-evidence", type=int, default=30)
    write.add_argument("--evidence-chars", type=int, default=1400)
    write.add_argument("--temperature", type=float, default=0.2)

    run = subparsers.add_parser("run", help="Run convert, index, and write in one command.")
    add_common_args(run)
    run.add_argument("--papers", required=True, help="Folder containing collected PDFs.")
    run.add_argument("--manifest", help="Optional LitHarvest manifest.json path.")
    run.add_argument("--goal", required=True, help="Goal Markdown file.")
    run.add_argument("--out", required=True, help="Run folder for generated artifacts.")
    run.add_argument("--report", required=True, help="Output report Markdown path.")
    run.add_argument(
        "--extractor",
        default="auto",
        choices=["auto", "pymupdf", "pypdf", "pdftotext"],
        help="PDF text extractor to use.",
    )
    run.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    run.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    run.add_argument("--min-chunk-words", type=int, default=DEFAULT_MIN_CHUNK_WORDS)
    run.add_argument("--api-key", help="LLM API key. Defaults to env vars.")
    run.add_argument("--model", default=env_model(), help="LLM model name.")
    run.add_argument("--base-url", default=env_base_url(), help="OpenAI-compatible base URL.")
    run.add_argument("--max-questions", type=int, default=6)
    run.add_argument("--per-question", type=int, default=6)
    run.add_argument("--max-evidence", type=int, default=30)
    run.add_argument("--evidence-chars", type=int, default=1400)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--force", action="store_true", help="Overwrite conversion/index outputs.")

    return parser.parse_args(argv)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    try:
        if args.command == "convert":
            payload = convert_pdfs_to_markdown(
                papers_dir=Path(args.papers),
                manifest_path=Path(args.manifest) if args.manifest else None,
                out_dir=Path(args.out),
                extractor=args.extractor,
                force=args.force,
            )
            print(Path(args.out) / "paper_index.json")
            LOGGER.info(
                "Converted %d/%d papers.",
                payload["converted_count"],
                payload["paper_count"],
            )
        elif args.command == "index":
            payload = build_index(
                run_dir=Path(args.run),
                chunk_words_count=args.chunk_words,
                overlap_words=args.overlap_words,
                min_chunk_words=args.min_chunk_words,
                force=args.force,
            )
            print(payload["db_path"])
            LOGGER.info("Indexed %d chunks.", payload["chunk_count"])
        elif args.command == "search":
            results = search_index(Path(args.run) / "index.sqlite", args.query, limit=args.limit)
            print_search_results(results, args.query)
        elif args.command == "write":
            api_key = args.api_key or env_api_key()
            trace = write_report(
                run_dir=Path(args.run),
                goal_path=Path(args.goal),
                output_path=Path(args.output),
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                max_questions=args.max_questions,
                per_question=args.per_question,
                max_evidence=args.max_evidence,
                evidence_chars=args.evidence_chars,
                temperature=args.temperature,
            )
            print(args.output)
            LOGGER.info("Report written with %d evidence chunks.", len(trace["evidence"]))
        elif args.command == "run":
            convert_pdfs_to_markdown(
                papers_dir=Path(args.papers),
                manifest_path=Path(args.manifest) if args.manifest else None,
                out_dir=Path(args.out),
                extractor=args.extractor,
                force=args.force,
            )
            build_index(
                run_dir=Path(args.out),
                chunk_words_count=args.chunk_words,
                overlap_words=args.overlap_words,
                min_chunk_words=args.min_chunk_words,
                force=args.force,
            )
            trace = write_report(
                run_dir=Path(args.out),
                goal_path=Path(args.goal),
                output_path=Path(args.report),
                api_key=args.api_key or env_api_key(),
                model=args.model,
                base_url=args.base_url,
                max_questions=args.max_questions,
                per_question=args.per_question,
                max_evidence=args.max_evidence,
                evidence_chars=args.evidence_chars,
                temperature=args.temperature,
            )
            print(args.report)
            LOGGER.info("Run complete with %d evidence chunks.", len(trace["evidence"]))
        else:  # pragma: no cover - argparse enforces commands
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    time.sleep(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
