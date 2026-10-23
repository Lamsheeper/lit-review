#!/usr/bin/env python3
"""Corpus-wide extraction of evidence-backed taxonomy feature mentions.

LitExtract is the database-building companion to LitSynth. It scans an existing
LitSynth run folder, sends Markdown-derived chunks or source PDFs to an
OpenAI-compatible or Gemini-native LLM endpoint, and writes normalized JSONL/CSV
artifacts for downstream taxonomy review.
"""

from __future__ import annotations
from lit_review.core.context import PROFILE

import base64
import dataclasses
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lit_review.core.io import read_json, utc_now, write_json
from lit_review.core.chat import ChatCompletionClient
from lit_review.core.taxonomy import load_taxonomy, taxonomy_category_lookup


from lit_review.core.llm import make_llm_client

from lit_review.features.records import load_goal_text, normalize_feature_item

from lit_review.features.prompts import build_extraction_prompt, build_pdf_extraction_prompt, pdf_chunk_id

from lit_review.features.responses import LOGGER, parse_features_json_with_repair

from lit_review.features.filtering import PROMPT_VERSION, VERSION
from lit_review.core.io import append_jsonl, read_jsonl
from lit_review.core.progress import progress_bar

from lit_review.profiles.media_ranking import title_relevance_score

from lit_review.features.extraction_artifacts import write_artifacts


PDF_EXTRACTION_SYSTEM_PROMPT = (
    "You are an evidence-bound taxonomy extraction agent. Return only "
    "valid JSON. Extract named feature mentions from academic PDFs."
)
PDF_INPUT_MAX_BYTES = 50 * 1024 * 1024

EVIDENCE_TYPES = {"definition", "taxonomy_row", "example", "named_mention"}


CURATED_MEMBER_FIELDNAMES = [
    "feature_key",
    "category",
    "original_feature_name",
    "original_normalized_feature_name",
    "action",
    "canonical_feature_key",
    "canonical_feature_name",
    "canonical_normalized_feature_name",
    "rejection_reason",
    "curation_confidence",
    "curation_notes",
    "paper_count",
    "mention_count",
    "model",
    "run_id",
    "curated_at",
]


@dataclass
class BatchResult:
    paper_id: str
    batch_index: int
    chunk_ids: list[str]
    features: list[dict[str, Any]]
    error: str | None = None
    raw_response: str | None = None


@dataclass
class ExtractionStats:
    paper_count: int = 0
    paper_completed_count: int = 0
    paper_skipped_count: int = 0
    pdf_count: int = 0
    chunk_count: int = 0
    batch_count: int = 0
    failed_batch_count: int = 0
    raw_feature_count: int = 0
    kept_feature_count: int = 0
    filtered_feature_count: int = 0
    repaired_json_count: int = 0


def chunk_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Extracting chunks", unit="chunk", show=show)


def extraction_progress(total: int, input_mode: str, show: bool) -> Any:
    if input_mode == "pdf":
        return progress_bar(total=total, desc="Extracting PDFs", unit="paper", show=show)
    return chunk_progress(total=total, show=show)


def title_score_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Scoring titles", unit="paper", show=show)


def rank_paper_records(
    records: list[dict[str, Any]],
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    with title_score_progress(total=len(records), show=show_progress) as progress:
        for idx, record in enumerate(records):
            paper = dict(record)
            paper["title_relevance_score"] = title_relevance_score(str(paper.get("title") or ""))
            ranked.append((-int(paper["title_relevance_score"]), idx, paper))
            progress.update(1)
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [paper for _, _, paper in ranked]


def load_papers(
    run_dir: Path,
    paper_limit: int | None = None,
    rank_by_title: bool = True,
    show_progress: bool = False,
    include_pdf_failures: bool = False,
) -> dict[str, dict[str, Any]]:
    paper_index_path = run_dir / "paper_index.json"
    if not paper_index_path.exists():
        raise FileNotFoundError(f"paper_index.json does not exist: {paper_index_path}")
    payload = read_json(paper_index_path)
    records: list[dict[str, Any]] = []
    for paper in payload.get("papers", []):
        if paper.get("conversion", {}).get("status") not in {
            "converted",
            "already_exists",
            "no_text",
        }:
            if not (include_pdf_failures and paper.get("pdf_path")):
                continue
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        records.append(dict(paper))
    if rank_by_title:
        records = rank_paper_records(records, show_progress=show_progress)
    if paper_limit is not None:
        records = records[:paper_limit]
    return {str(paper["paper_id"]): paper for paper in records}


def iter_chunks_by_paper(
    run_dir: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    db_path = run_dir / "index.sqlite"
    if db_path.exists():
        yield from iter_chunks_by_paper_sqlite(db_path, papers)
        return
    yield from iter_chunks_by_paper_jsonl(run_dir, papers)


def iter_chunks_by_paper_sqlite(
    db_path: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    with sqlite3.connect(db_path) as conn:
        for paper_id in papers:
            rows = conn.execute(
                """
                SELECT chunk_id, paper_id, title, year, pages_json, text, token_count
                FROM chunks
                WHERE paper_id = ?
                ORDER BY rowid
                """,
                (paper_id,),
            ).fetchall()
            chunks: list[dict[str, Any]] = []
            for chunk_id, row_paper_id, title, year, pages_json, text, token_count in rows:
                try:
                    pages = json.loads(pages_json)
                except json.JSONDecodeError:
                    pages = []
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_id": row_paper_id,
                        "title": title,
                        "year": year,
                        "pages": pages,
                        "text": text,
                        "token_count": token_count,
                    }
                )
            yield paper_id, chunks


def iter_chunks_by_paper_jsonl(
    run_dir: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl does not exist: {chunks_path}")

    chunks_by_paper = {paper_id: [] for paper_id in papers}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            paper_id = str(chunk.get("paper_id") or "")
            if paper_id in chunks_by_paper:
                chunks_by_paper[paper_id].append(chunk)
    for paper_id in papers:
        yield paper_id, chunks_by_paper.get(paper_id, [])


def count_chunks_for_papers(run_dir: Path, paper_ids: set[str]) -> int:
    if not paper_ids:
        return 0
    db_path = run_dir / "index.sqlite"
    if db_path.exists():
        total = 0
        paper_id_list = sorted(paper_ids)
        with sqlite3.connect(db_path) as conn:
            for idx in range(0, len(paper_id_list), 900):
                batch = paper_id_list[idx : idx + 900]
                placeholders = ",".join("?" for _ in batch)
                total += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM chunks WHERE paper_id IN ({placeholders})",
                        batch,
                    ).fetchone()[0]
                )
        return total

    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl does not exist: {chunks_path}")
    total = 0
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(chunk.get("paper_id") or "") in paper_ids:
                total += 1
    return total


def batch_chunks(chunks: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, batch_size)
    for idx in range(0, len(chunks), size):
        yield chunks[idx : idx + size]


def pdf_prompt_chunk(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": pdf_chunk_id(paper),
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "pages": [],
        "text": "",
    }


def resolve_pdf_path(paper: dict[str, Any], run_dir: Path) -> Path:
    raw_path = str(paper.get("pdf_path") or "").strip()
    if not raw_path:
        raise FileNotFoundError(f"Paper {paper.get('paper_id') or ''} has no pdf_path.")

    path = Path(raw_path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    run_relative = run_dir / path
    if run_relative.exists():
        return run_relative
    return path


def pdf_file_data_url(pdf_path: Path, max_bytes: int = PDF_INPUT_MAX_BYTES) -> str:
    size = pdf_path.stat().st_size
    if size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        size_mb = size / (1024 * 1024)
        raise ValueError(f"PDF is {size_mb:.1f} MB, above the {max_mb:.0f} MB direct-input limit.")
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def build_pdf_extraction_messages(
    *,
    paper: dict[str, Any],
    pdf_path: Path,
    goal_text: str = "",
    max_features_per_pdf: int = 40,
    taxonomy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": PDF_EXTRACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": pdf_path.name,
                        "file_data": pdf_file_data_url(pdf_path),
                    },
                },
                {
                    "type": "text",
                    "text": build_pdf_extraction_prompt(
                        paper,
                        goal_text=goal_text,
                        max_features_per_pdf=max_features_per_pdf,
                        taxonomy=taxonomy,
                    ),
                },
            ],
        },
    ]


def extract_batch_features(
    *,
    client: ChatCompletionClient,
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    model: str,
    run_id: str,
    max_chunk_chars: int,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
    temperature: float,
    extracted_at: str,
) -> BatchResult:
    prompt = build_extraction_prompt(
        paper,
        chunks,
        max_chunk_chars=max_chunk_chars,
        goal_text=goal_text,
        max_features_per_chunk=max_features_per_chunk,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-bound taxonomy extraction agent. Return only "
                "valid JSON. Extract named feature mentions from academic-paper chunks."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw_response = client.chat(
        messages,
        temperature=temperature,
        max_tokens=3500,
        response_format={"type": "json_object"},
    )
    parsed_items, _ = parse_features_json_with_repair(raw_response, client=client)
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    fallback_chunk = chunks[0] if chunks else {}
    normalized: list[dict[str, Any]] = []
    for item in parsed_items:
        chunk_id = str(item.get("chunk_id") or "")
        chunk = chunks_by_id.get(chunk_id) or fallback_chunk
        row = normalize_feature_item(
            item,
            paper=paper,
            chunk=chunk,
            run_id=run_id,
            model=model,
            extracted_at=extracted_at,
        )
        if row is not None:
            normalized.append(row)
    return BatchResult(
        paper_id=str(paper.get("paper_id") or ""),
        batch_index=0,
        chunk_ids=[str(chunk.get("chunk_id") or "") for chunk in chunks],
        features=normalized,
        raw_response=raw_response,
    )


def completed_paper_ids(output_dir: Path) -> set[str]:
    path = output_dir / "completed_papers.jsonl"
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(payload.get("paper_id") or "")
            if paper_id:
                completed.add(paper_id)
    return completed


def load_existing_mentions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "feature_mentions.jsonl"
    try:
        return read_jsonl(path)
    except json.JSONDecodeError:
        LOGGER.warning("Skipping malformed existing mention rows in %s", path)
        return []


def remove_existing_outputs(output_dir: Path) -> None:
    for name in [
        "feature_mentions.jsonl",
        "feature_mentions.csv",
        "paper_features.jsonl",
        "paper_features.csv",
        "features.jsonl",
        "features.csv",
        "completed_papers.jsonl",
        "extraction_errors.jsonl",
        "extraction_trace.json",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def unique_items(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item is None or item == "":
                continue
            key = json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else str(item)
            if key not in seen:
                seen.add(key)
                output.append(item)
    return output


def avg_float(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field)))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def max_float(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field)))
        except (TypeError, ValueError):
            continue
    return round(max(values), 4) if values else 0.0


def ranked_paper_preview(
    papers: dict[str, dict[str, Any]],
    limit: int = 25,
) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for paper in list(papers.values())[:limit]:
        preview.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "title_relevance_score": paper.get("title_relevance_score", 0),
            }
        )
    return preview


def dry_run_trace(
    *,
    run_dir: Path,
    output_dir: Path,
    papers: dict[str, dict[str, Any]],
    batch_chunks_count: int,
    input_mode: str,
    api_provider: str,
    model: str,
    run_id: str,
    goal_path: str | None = None,
    taxonomy_path: str | None = None,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
    max_features_per_pdf: int = 40,
    dynamic_taxonomy: bool = False,
) -> dict[str, Any]:
    stats = ExtractionStats(paper_count=len(papers))
    if input_mode == "pdf":
        stats.pdf_count = len(papers)
        stats.paper_completed_count = len(papers)
        stats.batch_count = len(papers)
    else:
        for _, chunks in iter_chunks_by_paper(run_dir, papers):
            stats.paper_completed_count += 1
            stats.chunk_count += len(chunks)
            stats.batch_count += sum(1 for _ in batch_chunks(chunks, batch_chunks_count))
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "taxonomy_path": taxonomy_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "input_mode": input_mode,
            "api_provider": api_provider,
            "batch_chunks": batch_chunks_count,
            "paper_ranking": "input_order" if dynamic_taxonomy else "title_relevance",
            "max_features_per_chunk": max_features_per_chunk,
            "max_features_per_pdf": max_features_per_pdf,
        },
        "top_ranked_papers": ranked_paper_preview(papers),
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "extraction_trace.json", trace)
    return trace


def run_extraction(
    *,
    run_dir: Path,
    output_dir: Path,
    api_key: str | None,
    model: str,
    base_url: str,
    api_provider: str = "auto",
    reasoning_effort: str | None,
    batch_chunks_count: int,
    paper_limit: int | None,
    goal_path: Path | None = None,
    taxonomy_path: Path | None = None,
    force: bool,
    dry_run: bool,
    max_chunk_chars: int,
    max_features_per_chunk: int = 12,
    max_features_per_pdf: int = 40,
    input_mode: str = "chunks",
    temperature: float,
    max_tokens: int = 6000,
    timeout: float | None = None,
    show_progress: bool = True,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    output_dir = output_dir.expanduser()
    if input_mode not in {"chunks", "pdf"}:
        raise ValueError("input_mode must be 'chunks' or 'pdf'.")
    if api_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("api_provider must be 'auto', 'openai', or 'gemini'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_existing_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    taxonomy = load_taxonomy(taxonomy_path.expanduser()) if taxonomy_path else None
    taxonomy_lookup = taxonomy_category_lookup(taxonomy) if taxonomy else None
    papers = load_papers(
        run_dir,
        paper_limit=paper_limit,
        rank_by_title=PROFILE.get() == "media_attributes",
        show_progress=show_progress,
        include_pdf_failures=input_mode == "pdf",
    )
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    if dry_run:
        return dry_run_trace(
            run_dir=run_dir,
            output_dir=output_dir,
            papers=papers,
            batch_chunks_count=batch_chunks_count,
            input_mode=input_mode,
            api_provider=api_provider,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            taxonomy_path=str(taxonomy_path) if taxonomy_path else None,
            goal_text=goal_text,
            max_features_per_chunk=max_features_per_chunk,
            max_features_per_pdf=max_features_per_pdf,
            dynamic_taxonomy=bool(taxonomy),
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = make_llm_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
            input_mode=input_mode,
            timeout=timeout,
        )

    stats = ExtractionStats(paper_count=len(papers))
    completed = completed_paper_ids(output_dir)
    mentions = load_existing_mentions(output_dir)
    extracted_at = utc_now()
    failures: list[dict[str, Any]] = []
    pending_paper_ids = {paper_id for paper_id in papers if paper_id not in completed}
    total_pending_units = (
        len(pending_paper_ids) if input_mode == "pdf" else count_chunks_for_papers(run_dir, pending_paper_ids)
    )
    output_counts = write_artifacts(output_dir, mentions)

    with extraction_progress(
        total=total_pending_units,
        input_mode=input_mode,
        show=show_progress,
    ) as progress:
        if input_mode == "pdf":
            for paper_id, paper in papers.items():
                if paper_id in completed:
                    stats.paper_skipped_count += 1
                    continue

                stats.pdf_count += 1
                stats.batch_count += 1
                pseudo_chunk = pdf_prompt_chunk(paper)
                paper_mentions: list[dict[str, Any]] = []
                paper_failed = False
                raw_response = ""
                pdf_path_text = ""
                try:
                    pdf_path = resolve_pdf_path(paper, run_dir)
                    pdf_path_text = str(pdf_path)
                    if not pdf_path.exists():
                        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
                    LOGGER.info("Extracting %s from PDF: %s", paper_id, pdf_path)
                    raw_response = client.chat(
                        build_pdf_extraction_messages(
                            paper=paper,
                            pdf_path=pdf_path,
                            goal_text=goal_text,
                            max_features_per_pdf=max_features_per_pdf,
                            taxonomy=taxonomy,
                        ),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    parsed, repaired_json = parse_features_json_with_repair(
                        raw_response,
                        client=client,
                    )
                    if repaired_json:
                        stats.repaired_json_count += 1
                    stats.raw_feature_count += len(parsed)
                    for item in parsed:
                        row = normalize_feature_item(
                            item,
                            paper=paper,
                            chunk=pseudo_chunk,
                            run_id=run_id,
                            model=model,
                            extracted_at=extracted_at,
                            taxonomy_lookup=taxonomy_lookup,
                        )
                        if row is None:
                            stats.filtered_feature_count += 1
                            continue
                        stats.kept_feature_count += 1
                        paper_mentions.append(row)
                except Exception as exc:
                    paper_failed = True
                    stats.failed_batch_count += 1
                    error_row = {
                        "paper_id": paper_id,
                        "batch_index": 1,
                        "chunk_ids": [str(pseudo_chunk.get("chunk_id") or "")],
                        "pdf_path": pdf_path_text,
                        "error": str(exc),
                        "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                        "run_id": run_id,
                        "created_at": utc_now(),
                    }
                    failures.append(error_row)
                    append_jsonl(output_dir / "extraction_errors.jsonl", error_row)
                    LOGGER.warning(
                        "PDF extraction failed for %s: %s",
                        paper_id,
                        exc,
                    )
                finally:
                    progress.update(1)

                mentions.extend(paper_mentions)
                output_counts = write_artifacts(output_dir, mentions)
                if paper_failed:
                    LOGGER.info("Leaving %s incomplete so failed PDF extraction can be retried.", paper_id)
                else:
                    append_jsonl(
                        output_dir / "completed_papers.jsonl",
                        {
                            "paper_id": paper_id,
                            "input_mode": input_mode,
                            "chunk_count": 0,
                            "pdf_path": pdf_path_text,
                            "feature_count": len(paper_mentions),
                            "run_id": run_id,
                            "completed_at": utc_now(),
                        },
                    )
                    completed.add(paper_id)
                    stats.paper_completed_count += 1
        else:
            for paper_id, chunks in iter_chunks_by_paper(run_dir, papers):
                if paper_id in completed:
                    stats.paper_skipped_count += 1
                    continue

                paper = papers[paper_id]
                stats.chunk_count += len(chunks)
                paper_mentions: list[dict[str, Any]] = []
                paper_failed = False
                LOGGER.info("Extracting %s with %d chunks.", paper_id, len(chunks))
                for batch_index, chunk_batch in enumerate(
                    batch_chunks(chunks, batch_chunks_count),
                    start=1,
                ):
                    stats.batch_count += 1
                    raw_response = ""
                    try:
                        raw_response = client.chat(
                            [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an evidence-bound taxonomy extraction agent. "
                                        "Return only valid JSON."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": build_extraction_prompt(
                                        paper,
                                        chunk_batch,
                                        max_chunk_chars=max_chunk_chars,
                                        goal_text=goal_text,
                                        max_features_per_chunk=max_features_per_chunk,
                                        taxonomy=taxonomy,
                                    ),
                                },
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                        )
                        parsed, repaired_json = parse_features_json_with_repair(
                            raw_response,
                            client=client,
                        )
                        if repaired_json:
                            stats.repaired_json_count += 1
                        stats.raw_feature_count += len(parsed)
                        chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunk_batch}
                        fallback_chunk = chunk_batch[0] if chunk_batch else {}
                        for item in parsed:
                            chunk_id = str(item.get("chunk_id") or "")
                            row = normalize_feature_item(
                                item,
                                paper=paper,
                                chunk=chunks_by_id.get(chunk_id) or fallback_chunk,
                                run_id=run_id,
                                model=model,
                                extracted_at=extracted_at,
                                taxonomy_lookup=taxonomy_lookup,
                            )
                            if row is None:
                                stats.filtered_feature_count += 1
                                continue
                            stats.kept_feature_count += 1
                            paper_mentions.append(row)
                    except Exception as exc:
                        paper_failed = True
                        stats.failed_batch_count += 1
                        error_row = {
                            "paper_id": paper_id,
                            "batch_index": batch_index,
                            "chunk_ids": [str(chunk.get("chunk_id") or "") for chunk in chunk_batch],
                            "error": str(exc),
                            "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                            "run_id": run_id,
                            "created_at": utc_now(),
                        }
                        failures.append(error_row)
                        append_jsonl(output_dir / "extraction_errors.jsonl", error_row)
                        LOGGER.warning(
                            "Extraction failed for %s batch %d: %s",
                            paper_id,
                            batch_index,
                            exc,
                        )
                    finally:
                        progress.update(len(chunk_batch))

                mentions.extend(paper_mentions)
                output_counts = write_artifacts(output_dir, mentions)
                if paper_failed:
                    LOGGER.info("Leaving %s incomplete so failed batches can be retried.", paper_id)
                else:
                    append_jsonl(
                        output_dir / "completed_papers.jsonl",
                        {
                            "paper_id": paper_id,
                            "input_mode": input_mode,
                            "chunk_count": len(chunks),
                            "feature_count": len(paper_mentions),
                            "run_id": run_id,
                            "completed_at": utc_now(),
                        },
                    )
                    completed.add(paper_id)
                    stats.paper_completed_count += 1

    errors_path = output_dir / "extraction_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "goal_path": resolved_goal_path,
        "taxonomy_path": str(taxonomy_path) if taxonomy_path else None,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "input_mode": input_mode,
            "api_provider": api_provider,
            "batch_chunks": batch_chunks_count,
            "paper_limit": paper_limit,
            "max_chunk_chars": max_chunk_chars,
            "max_features_per_chunk": max_features_per_chunk,
            "max_features_per_pdf": max_features_per_pdf,
            "temperature": temperature,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
            "paper_ranking": "title_relevance" if taxonomy is None else "input_order",
            "dynamic_taxonomy": bool(taxonomy),
        },
        "top_ranked_papers": ranked_paper_preview(papers),
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "extraction_trace.json", trace)
    return trace
