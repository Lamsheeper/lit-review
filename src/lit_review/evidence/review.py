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

from lit_review.core.llm import make_llm_client
from lit_review.core.google_auth import use_vertex
from lit_review.core.context import PROFILE
from lit_review.profiles.media_review import taxonomy_plan_goal, llm_plan_taxonomy_goal
from lit_review.core.json_response import parse_json_object

import dataclasses
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from lit_review.evidence.pdf import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_MIN_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    PdfConversionChunkingService,
    normalize_text,
)
from lit_review.core.io import read_json, utc_now, write_json, write_jsonl

from lit_review.core.text import content_tokens
from lit_review.collection.sources import dedupe_preserve

from lit_review.core.chat import ChatCompletionClient, LOGGER


VERSION = "0.2.0"
DEFAULT_MODEL = "gpt-4o-mini"
PDF_SERVICE = PdfConversionChunkingService(version=VERSION, logger=LOGGER)


CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n[truncated]"


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
    chunks = PDF_SERVICE.chunk_paper_index(
        papers,
        run_dir=run_dir,
        chunk_words_count=chunk_words_count,
        overlap_words=overlap_words,
        min_chunk_words=min_chunk_words,
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
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, text)")
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
    rows = conn.execute("SELECT chunk_id, paper_id, title, year, pages_json, text FROM chunks").fetchall()
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


def is_taxonomy_goal(goal_text: str) -> bool:
    lowered = goal_text.lower()
    return (
        "taxonomy" in lowered
        and "persuasion" in lowered
        and ("moral framing" in lowered or "emotional" in lowered)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    return parse_json_object(text)


def parse_plan_json(content: str) -> dict[str, Any]:
    plan = extract_json_object(content)
    questions = [str(item) for item in plan.get("questions", []) if str(item).strip()]
    outline = [str(item) for item in plan.get("outline", []) if str(item).strip()]
    if not questions:
        raise ValueError("Planner JSON did not include any questions.")
    return {
        "title": str(plan.get("title") or "Literature Review Report"),
        "questions": questions,
        "outline": outline or ["Executive Summary", "Evidence Map", "Synthesis", "Recommendations"],
    }


def repair_plan_json(
    raw_content: str,
    client: ChatCompletionClient,
    max_questions: int,
) -> dict[str, Any]:
    repaired = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return only one valid JSON object. "
                    "Do not include Markdown fences or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair this literature-review plan into valid JSON with exactly these keys: "
                    "title as a string, questions as an array of strings, and outline as an array "
                    f"of strings. Keep at most {max_questions} questions.\n\n"
                    f"MALFORMED JSON OR TEXT:\n{raw_content}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=max(2000, min(8000, 800 + max_questions * 180)),
        response_format={"type": "json_object"},
    )
    return parse_plan_json(repaired)


def parse_questions_json(content: str) -> list[str]:
    payload = extract_json_object(content)
    questions = [str(item) for item in payload.get("questions", []) if str(item).strip()]
    if not questions:
        raise ValueError("Planner JSON did not include any questions.")
    return questions


def repair_questions_json(
    raw_content: str,
    client: ChatCompletionClient,
    max_questions: int,
) -> list[str]:
    repaired = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return only one valid JSON object "
                    "with a questions array. Do not include Markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair this into valid JSON with exactly one key, questions, "
                    f"as an array of at most {max_questions} search questions.\n\n"
                    f"MALFORMED JSON OR TEXT:\n{raw_content}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=max(2000, min(8000, 800 + max_questions * 160)),
        response_format={"type": "json_object"},
    )
    return parse_questions_json(repaired)


def llm_plan_goal(
    goal_text: str,
    client: ChatCompletionClient,
    max_questions: int,
    temperature: float = 0.1,
) -> dict[str, Any]:
    planner_max_tokens = max(2000, min(8000, 800 + max_questions * 180))
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
        max_tokens=planner_max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        plan = parse_plan_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        LOGGER.warning("Planner returned invalid JSON; asking model to repair it: %s", exc)
        plan = repair_plan_json(content, client=client, max_questions=max_questions)

    return {
        "title": plan.get("title") or derive_goal_title(goal_text),
        "questions": plan.get("questions", [])[:max_questions],
        "outline": plan.get("outline")
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


TOP_LEVEL_OUTLINE_RE = re.compile(r"^\s*\d+\.\s+\S")


def group_outline_sections(outline: list[Any]) -> list[list[str]]:
    headings = [str(item).strip() for item in outline if str(item).strip()]
    if not headings:
        return [["Synthesis"]]
    if not any(TOP_LEVEL_OUTLINE_RE.match(heading) for heading in headings):
        return [[heading] for heading in headings]

    groups: list[list[str]] = []
    current: list[str] = []
    for heading in headings:
        if TOP_LEVEL_OUTLINE_RE.match(heading):
            if current:
                groups.append(current)
            current = [heading]
        elif current:
            current.append(heading)
        else:
            groups.append([heading])
    if current:
        groups.append(current)
    return groups


def ensure_evidence_gaps_section(section_groups: list[list[str]]) -> list[list[str]]:
    if any("evidence gap" in heading.lower() for group in section_groups for heading in group):
        return section_groups
    return [*section_groups, ["Evidence Gaps"]]


def select_section_evidence(
    section_headings: list[str],
    evidence: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(evidence) <= limit:
        return evidence

    query_terms = set(content_tokens(" ".join(section_headings)))
    if not query_terms:
        return evidence[:limit]

    scored: list[tuple[int, int, dict[str, Any]]] = []
    unscored: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(evidence):
        haystack = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("question") or ""),
                str(item.get("text") or ""),
            ]
        )
        term_counts = Counter(content_tokens(haystack))
        score = sum(term_counts[term] for term in query_terms)
        if score:
            scored.append((-score, idx, item))
        else:
            unscored.append((idx, item))

    scored.sort()
    selected = [item for _, _, item in scored[:limit]]
    seen = {item["chunk_id"] for item in selected}
    for _, item in unscored:
        if len(selected) >= limit:
            break
        if item["chunk_id"] not in seen:
            selected.append(item)
            seen.add(item["chunk_id"])
    return selected[:limit]


def build_section_prompt(
    goal_text: str,
    plan: dict[str, Any],
    section_headings: list[str],
    evidence_text: str,
    section_number: int,
    section_total: int,
) -> str:
    section_title = section_headings[0].lower() if section_headings else ""
    section_task = "Synthesize the supplied evidence to answer the goal for this section. Distinguish supported conclusions from evidence gaps."
    if PROFILE.get() == "media_attributes":
        from lit_review.profiles.media_review import section_task as media_task

        section_task = media_task(section_title)

    return f"""GOAL
{goal_text}

OVERALL RESEARCH QUESTIONS
{json.dumps(plan.get("questions", []), indent=2)}

FULL REQUESTED OUTLINE
{json.dumps(plan.get("outline", []), indent=2)}

CURRENT SECTION OUTLINE
{json.dumps(section_headings, indent=2)}

SECTION TASK
{section_task}

EVIDENCE CHUNKS FOR THIS SECTION
{evidence_text}

Write only section {section_number} of {section_total} in Markdown.
Start with the first current section heading as a level-2 heading. Use level-3
headings for any current subsection headings. Do not write the document title,
front matter, or sections outside the current section outline.

Output requirements:
- For taxonomy category sections, prioritize the requested taxonomy table over prose.
- Give concrete feature/category names, not just summaries of papers.
- Use evidence citations in every table row or bullet.
- Add a short "Operational boundary" note when a feature could be confused with another category.
- Keep prose short, but do not omit taxonomy entries merely to be concise.
- Do not write transitions, background filler, or broad literature-review prose.

    Use only the evidence chunks above. Cite factual claims with chunk IDs exactly
like [paper_id:p3:c1]. If the evidence is insufficient, write one concise
"Evidence gap:" bullet instead of filling the gap from memory.
"""


def finish_reason_indicates_truncation(finish_reason: str | None) -> bool:
    if not finish_reason:
        return False
    normalized = finish_reason.lower()
    return any(marker in normalized for marker in ["length", "token", "max"])


def continue_section_generation(
    client: ChatCompletionClient,
    messages: list[dict[str, str]],
    partial_text: str,
    temperature: float,
    max_continuations: int = 3,
) -> tuple[str, list[str | None]]:
    parts = [partial_text.strip()]
    finish_reasons: list[str | None] = []
    current_messages = [*messages]
    current_text = partial_text
    for _ in range(max_continuations):
        current_messages = [
            *current_messages,
            {"role": "assistant", "content": current_text},
            {
                "role": "user",
                "content": (
                    "Continue exactly where the previous answer stopped. Stay within "
                    "the same section and do not repeat text already written."
                ),
            },
        ]
        result = client.chat_result(
            current_messages,
            temperature=temperature,
            max_tokens=5000,
        )
        finish_reasons.append(result.finish_reason)
        if result.content:
            parts.append(result.content.strip())
            current_text = result.content
        if not finish_reason_indicates_truncation(result.finish_reason):
            break
    return "\n".join(part for part in parts if part), finish_reasons


def write_llm_report(
    goal_text: str,
    plan: dict[str, Any],
    evidence: list[dict[str, Any]],
    client: ChatCompletionClient,
    output_path: Path,
    temperature: float = 0.2,
    evidence_chars: int = 1400,
    section_evidence: int = 24,
) -> str:
    section_groups = ensure_evidence_gaps_section(group_outline_sections(plan.get("outline", [])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {plan.get('title') or derive_goal_title(goal_text)}", ""]
    output_path.write_text("This report is being generated section by section.\n\n", encoding="utf-8")
    for idx, section_headings in enumerate(section_groups, start=1):
        section_items = select_section_evidence(
            section_headings,
            evidence=evidence,
            limit=section_evidence,
        )
        evidence_text = format_evidence_for_prompt(
            section_items,
            max_chars_per_chunk=evidence_chars,
        )
        prompt = build_section_prompt(
            goal_text=goal_text,
            plan=plan,
            section_headings=section_headings,
            evidence_text=evidence_text,
            section_number=idx,
            section_total=len(section_groups),
        )
        LOGGER.info(
            "Writing section %d/%d with %d evidence chunks: %s",
            idx,
            len(section_groups),
            len(section_items),
            section_headings[0],
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-bound literature-review synthesis agent. "
                    "Write one compact requested report section at a time. Prefer "
                    "dense bullets and tables over prose. Every factual claim must "
                    "be grounded in the provided evidence chunk IDs."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        result = client.chat_result(
            messages,
            temperature=temperature,
            max_tokens=5000,
        )
        finish_reasons = [result.finish_reason]
        section_text = result.content
        if finish_reason_indicates_truncation(result.finish_reason):
            LOGGER.warning(
                "Section %d stopped early with finish_reason=%s; continuing.",
                idx,
                result.finish_reason,
            )
            section_text, continuation_reasons = continue_section_generation(
                client=client,
                messages=messages,
                partial_text=section_text,
                temperature=temperature,
            )
            finish_reasons.extend(continuation_reasons)
        LOGGER.info(
            "Finished section %d/%d with finish reasons: %s",
            idx,
            len(section_groups),
            ", ".join(str(reason) for reason in finish_reasons),
        )
        lines.extend([section_text.strip(), ""])
        partial_report = "\n".join(lines).rstrip() + "\n\n"
        if idx < len(section_groups):
            partial_report += f"<!-- LitSynth: generated {idx}/{len(section_groups)} sections. -->\n"
        output_path.write_text(partial_report, encoding="utf-8")

    report = "\n".join(lines).rstrip()
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
                f"- [{item['chunk_id']}] {item['title']} ({item.get('year') or 'n.d.'}, {pages}): {snippet}"
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
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    max_questions: int = 6,
    per_question: int = 6,
    max_evidence: int = 30,
    evidence_chars: int = 1400,
    section_evidence: int = 24,
    temperature: float = 0.2,
    api_provider: str = "openai",
    auth_mode: str = "auto",
    vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> dict[str, Any]:
    db_path = run_dir / "index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Index database does not exist: {db_path}")
    goal_text = goal_path.read_text(encoding="utf-8", errors="replace")
    base_url = base_url or (
        "https://generativelanguage.googleapis.com/v1beta" if api_provider == "gemini"
        else "https://api.openai.com/v1"
    )
    if api_provider != "gemini" and (auth_mode == "adc" or vertex_project):
        raise ValueError("Vertex AI ADC authentication requires the Gemini provider.")
    client = None
    if api_key or (api_provider == "gemini" and use_vertex(
        api_key, auth_mode=auth_mode, project=vertex_project
    )):
        client = make_llm_client(
            api_key=api_key, model=model, base_url=base_url,
            reasoning_effort=reasoning_effort, api_provider=api_provider,
            auth_mode=auth_mode, vertex_project=vertex_project, vertex_location=vertex_location,
        )

    if PROFILE.get() == "media_attributes" and is_taxonomy_goal(goal_text):
        if client:
            LOGGER.info("Planning taxonomy retrieval questions with LLM model %s", model)
            try:
                plan = llm_plan_taxonomy_goal(
                    goal_text,
                    client=client,
                    max_questions=max_questions,
                    temperature=temperature,
                )
            except Exception as exc:
                LOGGER.warning(
                    "LLM taxonomy question planning failed; using deterministic questions: %s",
                    exc,
                )
                plan = taxonomy_plan_goal(goal_text, max_questions=max_questions)
        else:
            LOGGER.info("Using taxonomy-specific deterministic plan.")
            plan = taxonomy_plan_goal(goal_text, max_questions=max_questions)
    elif client:
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
            section_evidence=section_evidence,
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
        "reasoning_effort": reasoning_effort if llm_used else None,
        "section_evidence": section_evidence if llm_used else None,
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
