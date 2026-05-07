#!/usr/bin/env python3
"""PDF conversion and Markdown chunking service for LitSynth."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from lit_harvest import slugify
except ImportError:  # pragma: no cover - defensive standalone fallback

    def normalize_title(value: str | None) -> str:
        if not value:
            return ""
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def slugify(value: str | None, max_length: int = 80) -> str:
        slug = normalize_title(value).replace(" ", "_")
        return slug[:max_length].strip("_")


LOGGER = logging.getLogger("litsynth")
DEFAULT_PROJECT = "LitSynth"
DEFAULT_VERSION = "0.1.0"
DEFAULT_CHUNK_WORDS = 700
DEFAULT_OVERLAP_WORDS = 80
DEFAULT_MIN_CHUNK_WORDS = 40
PAGE_HEADING_RE = re.compile(r"^## Page\s+(\d+)\s*$", re.MULTILINE)


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


def chunk_paper_index(
    papers: list[dict[str, Any]],
    run_dir: Path,
    chunk_words_count: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
) -> list[ChunkRecord]:
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
    return chunks


class PdfConversionChunkingService:
    """Owns PDF-to-Markdown conversion and Markdown-to-chunk preparation."""

    def __init__(
        self,
        project: str = DEFAULT_PROJECT,
        version: str = DEFAULT_VERSION,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.project = project
        self.version = version
        self.logger = logger

    def collect_paper_records(
        self,
        papers_dir: Path,
        manifest: dict[str, Any] | None = None,
        manifest_path: Path | None = None,
    ) -> list[PaperRecord]:
        return collect_paper_records(papers_dir, manifest=manifest, manifest_path=manifest_path)

    def extract_pdf_pages(
        self,
        pdf_path: Path,
        extractor: str = "auto",
        timeout: int = 120,
    ) -> list[str]:
        return extract_pdf_pages(pdf_path, extractor=extractor, timeout=timeout)

    def render_paper_markdown(self, record: PaperRecord, pages: list[str]) -> str:
        return render_paper_markdown(record, pages)

    def convert_pdfs_to_markdown(
        self,
        papers_dir: Path,
        manifest_path: Path | None,
        out_dir: Path,
        extractor: str = "auto",
        force: bool = False,
    ) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        records = self.collect_paper_records(
            papers_dir, manifest=manifest, manifest_path=manifest_path
        )
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
                self.logger.info("Converting %s", pdf_path)
                pages = self.extract_pdf_pages(pdf_path, extractor=extractor)
                md_path.write_text(self.render_paper_markdown(record, pages), encoding="utf-8")
                text_pages = sum(1 for page in pages if page.strip())
                record.conversion = {
                    "status": "converted" if text_pages else "no_text",
                    "extractor": extractor,
                    "md_path": str(md_path),
                    "page_count": len(pages),
                    "text_page_count": text_pages,
                }
            except Exception as exc:
                self.logger.warning("Could not convert %s: %s", pdf_path, exc)
                record.md_path = None
                record.conversion = {
                    "status": "failed",
                    "extractor": extractor,
                    "reason": str(exc),
                }
            converted.append(record)

        payload = {
            "project": self.project,
            "version": self.version,
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

    def split_markdown_pages(self, markdown: str) -> list[tuple[int | None, str]]:
        return split_markdown_pages(markdown)

    def chunk_words(self, text: str, chunk_words_count: int, overlap_words: int) -> list[str]:
        return chunk_words(text, chunk_words_count=chunk_words_count, overlap_words=overlap_words)

    def chunk_markdown_paper(
        self,
        record: dict[str, Any],
        run_dir: Path,
        chunk_words_count: int,
        overlap_words: int,
        min_chunk_words: int,
    ) -> list[ChunkRecord]:
        return chunk_markdown_paper(
            record,
            run_dir=run_dir,
            chunk_words_count=chunk_words_count,
            overlap_words=overlap_words,
            min_chunk_words=min_chunk_words,
        )

    def chunk_paper_index(
        self,
        papers: list[dict[str, Any]],
        run_dir: Path,
        chunk_words_count: int = DEFAULT_CHUNK_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
        min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    ) -> list[ChunkRecord]:
        return chunk_paper_index(
            papers,
            run_dir=run_dir,
            chunk_words_count=chunk_words_count,
            overlap_words=overlap_words,
            min_chunk_words=min_chunk_words,
        )


DEFAULT_SERVICE = PdfConversionChunkingService()


def convert_pdfs_to_markdown(
    papers_dir: Path,
    manifest_path: Path | None,
    out_dir: Path,
    extractor: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    return DEFAULT_SERVICE.convert_pdfs_to_markdown(
        papers_dir=papers_dir,
        manifest_path=manifest_path,
        out_dir=out_dir,
        extractor=extractor,
        force=force,
    )
