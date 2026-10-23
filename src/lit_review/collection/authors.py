#!/usr/bin/env python3
"""Author-roster publication inventory and open-PDF collection.

This is the author-centric companion to ``lit_harvest``.  It resolves roster
entries to OpenAlex authors using the supplied institution as disambiguation,
then writes a resumable folder tree and spreadsheet-friendly CSV reports.
Downloads are opt-in and reuse LitHarvest's PDF resolver/downloader.
"""

from __future__ import annotations

from lit_review.core.io import write_json

import csv
import dataclasses
import difflib
import json
import logging
import re
import unicodedata
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from lit_review.collection.sources import Candidate, OpenAlexClient, SearchQuery
from lit_review.core.http import HttpClient
from lit_review.collection.downloads import PdfDownloader, PdfResolver, build_pdf_filename
from lit_review.collection.harvest import ensure_candidate_ids, resolve_and_download_candidate, slugify
from lit_review.core.io import utc_now


LOGGER = logging.getLogger("litauthorharvest")
OPENALEX_API = "https://api.openalex.org"
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

INSTITUTION_ALIASES = {
    "city college": "city college of new york",
    "cuny gradsch ofpub h and h pol": ("cuny graduate school of public health and health policy"),
    "nyc college of technology": "new york city college of technology",
    "john jay college": "john jay college of criminal justice",
}

CSV_FIELDS = [
    "roster_name",
    "college",
    "department",
    "openalex_author_name",
    "openalex_author_id",
    "author_match_status",
    "author_match_score",
    "title",
    "authors",
    "year",
    "work_type",
    "venue",
    "doi",
    "openalex_work_id",
    "citation_count",
    "oa_status",
    "is_oa",
    "pdf_availability",
    "pdf_url_count",
    "pdf_urls",
    "landing_page_url",
    "download_status",
    "download_path",
]


@dataclass
class FacultyRecord:
    name: str
    title: str = ""
    college: str = ""
    department: str = ""
    source_rows: list[int] = field(default_factory=list)


@dataclass
class AuthorMatch:
    faculty: FacultyRecord
    status: str
    score: float = 0.0
    author_id: str | None = None
    display_name: str | None = None
    works_count: int | None = None
    institutions: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


@dataclass
class Publication:
    candidate: Candidate
    openalex_work_id: str | None = None
    work_type: str | None = None
    oa_status: str | None = None
    is_oa: bool | None = None


def normalize(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalized_institution(value: str | None) -> str:
    normalized = normalize(value)
    return INSTITUTION_ALIASES.get(normalized, normalized)


def column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", cell_reference)
    if not letters:
        return 0
    result = 0
    for character in letters.group(0).upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook.find(f".//{{{XLSX_NS}}}sheet")
    if first_sheet is None:
        raise ValueError("Workbook does not contain a worksheet.")
    relationship_id = first_sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
    if not relationship_id:
        return "xl/worksheets/sheet1.xml"
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relation in relationships.findall(f"{{{REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relationship_id:
            target = relation.attrib.get("Target", "worksheets/sheet1.xml")
            return "xl/" + target.lstrip("/")
    return "xl/worksheets/sheet1.xml"


def read_xlsx_rows(path: Path) -> list[list[str]]:
    """Read scalar values from the first XLSX worksheet without a new dependency."""
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{XLSX_NS}}}si"):
                text = "".join(node.text or "" for node in item.iter(f"{{{XLSX_NS}}}t"))
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read(_xlsx_sheet_path(archive)))
        rows: list[list[str]] = []
        for row in sheet.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{{{XLSX_NS}}}c"):
                index = column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{XLSX_NS}}}v")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{{{XLSX_NS}}}t"))
                else:
                    value = "" if value_node is None else value_node.text or ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                values[index] = value.strip()
            width = max(values, default=-1) + 1
            rows.append([values.get(index, "") for index in range(width)])
        return rows


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [[value.strip() for value in row] for row in csv.reader(handle)]


def load_faculty(path: Path) -> list[FacultyRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Roster does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".xlsx":
        rows = read_xlsx_rows(path)
    elif suffix == ".csv":
        rows = read_csv_rows(path)
    else:
        raise ValueError("Roster must be an .xlsx or .csv file.")
    if not rows:
        raise ValueError("Roster is empty.")

    headers = {normalize(value): index for index, value in enumerate(rows[0]) if value}
    if "name" not in headers:
        raise ValueError("Roster must contain a Name column.")

    def get(row: list[str], header: str) -> str:
        index = headers.get(header)
        return row[index].strip() if index is not None and index < len(row) else ""

    deduped: dict[tuple[str, str, str], FacultyRecord] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        name = get(row, "name")
        if not name:
            continue
        record = FacultyRecord(
            name=name,
            title=get(row, "title"),
            college=get(row, "college"),
            department=get(row, "department"),
            source_rows=[row_number],
        )
        key = (normalize(record.name), normalize(record.college), normalize(record.department))
        if key in deduped:
            deduped[key].source_rows.append(row_number)
        else:
            deduped[key] = record
    return list(deduped.values())


def author_institutions(item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for institution in item.get("last_known_institutions") or []:
        if isinstance(institution, dict) and institution.get("display_name"):
            names.append(str(institution["display_name"]))
    legacy = item.get("last_known_institution")
    if isinstance(legacy, dict) and legacy.get("display_name"):
        names.append(str(legacy["display_name"]))
    for affiliation in item.get("affiliations") or []:
        institution = affiliation.get("institution") if isinstance(affiliation, dict) else None
        if isinstance(institution, dict) and institution.get("display_name"):
            names.append(str(institution["display_name"]))
    return list(dict.fromkeys(names))


def institution_similarity(expected: str, actual_names: Iterable[str]) -> float:
    expected_normalized = normalized_institution(expected)
    if not expected_normalized:
        return 0.0
    best = 0.0
    generic_tokens = {
        "city",
        "college",
        "institute",
        "new",
        "of",
        "school",
        "state",
        "the",
        "university",
        "york",
    }
    expected_tokens = set(expected_normalized.split()) - generic_tokens
    for actual in actual_names:
        actual_normalized = normalized_institution(actual)
        if not actual_normalized:
            continue
        if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
            best = max(best, 1.0)
            continue
        actual_tokens = set(actual_normalized.split()) - generic_tokens
        union = expected_tokens | actual_tokens
        if union:
            best = max(best, len(expected_tokens & actual_tokens) / len(union))
    return best


def score_author(faculty: FacultyRecord, item: dict[str, Any]) -> tuple[float, float, float]:
    expected_name = normalize(faculty.name)
    actual_name = normalize(str(item.get("display_name") or ""))
    expected_tokens = expected_name.split()
    actual_tokens = actual_name.split()
    same_outer_name = bool(
        expected_tokens
        and actual_tokens
        and expected_tokens[0] == actual_tokens[0]
        and expected_tokens[-1] == actual_tokens[-1]
    )
    if same_outer_name and set(expected_tokens).issubset(set(actual_tokens)):
        # Middle names/initials and accents should not turn an otherwise exact
        # identity match into a low-confidence result.
        name_score = 1.0
    else:
        name_score = difflib.SequenceMatcher(None, expected_name, actual_name).ratio()
    institution_score = institution_similarity(faculty.college, author_institutions(item))
    # A matching institution is the strongest protection against namesakes.
    total = 0.72 * name_score + 0.28 * institution_score
    return round(total, 5), name_score, institution_score


def resolve_author(http: HttpClient, faculty: FacultyRecord, per_page: int = 50) -> AuthorMatch:
    try:
        data = http.get_json(
            f"{OPENALEX_API}/authors",
            params={"search": faculty.name, "per-page": min(max(per_page, 1), 50)},
        )
    except Exception as exc:
        return AuthorMatch(faculty=faculty, status="error", reason=str(exc))

    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        score, name_score, institution_score = score_author(faculty, item)
        ranked.append((score, name_score, institution_score, item))
    ranked.sort(key=lambda value: (value[0], int(value[3].get("works_count") or 0)), reverse=True)
    if not ranked:
        return AuthorMatch(faculty=faculty, status="not_found", reason="No OpenAlex author result.")

    score, name_score, institution_score, selected = ranked[0]
    alternatives = []
    for alternative_score, alternative_name_score, alternative_inst_score, item in ranked[:5]:
        alternatives.append(
            {
                "id": item.get("id"),
                "display_name": item.get("display_name"),
                "works_count": item.get("works_count"),
                "institutions": author_institutions(item),
                "score": alternative_score,
                "name_score": round(alternative_name_score, 5),
                "institution_score": round(alternative_inst_score, 5),
            }
        )

    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    exact_name = name_score >= 0.98
    strong_institution = institution_score >= 0.65
    confident = exact_name and strong_institution and (score - runner_up >= 0.02 or score >= 0.98)
    status = "matched" if confident else "needs_review"
    reason = None
    if status == "needs_review":
        reason = "Best author match was not uniquely supported by both exact name and institution."
    return AuthorMatch(
        faculty=faculty,
        status=status,
        score=score,
        author_id=str(selected.get("id")) if selected.get("id") else None,
        display_name=selected.get("display_name"),
        works_count=int(selected.get("works_count") or 0),
        institutions=author_institutions(selected),
        alternatives=alternatives,
        reason=reason,
    )


def short_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def collect_publications(
    http: HttpClient,
    match: AuthorMatch,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    max_publications: int | None = None,
) -> list[Publication]:
    author_id = short_openalex_id(match.author_id)
    if not author_id:
        return []
    parser = OpenAlexClient(http)
    query = SearchQuery(bucket="author", text=match.faculty.name, terms=[])
    page = 1
    publications: list[Publication] = []
    while True:
        filters = [f"author.id:{author_id}"]
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        page_size = 200
        if max_publications is not None:
            remaining = max_publications - len(publications)
            if remaining <= 0:
                break
            page_size = min(page_size, remaining)
        try:
            data = http.get_json(
                f"{OPENALEX_API}/works",
                params={
                    "filter": ",".join(filters),
                    "per-page": page_size,
                    "page": page,
                    "sort": "publication_date:desc",
                },
            )
        except Exception:
            if publications:
                LOGGER.warning(
                    "Stopped publication paging for %s after %d records.",
                    match.faculty.name,
                    len(publications),
                )
                break
            raise
        results = data.get("results") or []
        for item in results:
            if not isinstance(item, dict):
                continue
            candidate = parser._candidate_from_work(item, query)
            if candidate is None:
                continue
            open_access = item.get("open_access") or {}
            publications.append(
                Publication(
                    candidate=candidate,
                    openalex_work_id=item.get("id"),
                    work_type=item.get("type"),
                    oa_status=open_access.get("oa_status"),
                    is_oa=open_access.get("is_oa"),
                )
            )
        if not results:
            break
        total = int((data.get("meta") or {}).get("count") or 0)
        if len(results) < page_size or (total and len(publications) >= total):
            break
        page += 1
        if max_publications is not None and len(publications) >= max_publications:
            publications = publications[:max_publications]
            break
    ensure_candidate_ids([publication.candidate for publication in publications])
    return publications


def availability(publication: Publication) -> str:
    if publication.candidate.candidate_pdf_urls:
        return "pdf_url_listed"
    if publication.is_oa:
        return "open_access_no_pdf_url"
    return "no_pdf_url_listed"


def publication_row(match: AuthorMatch, publication: Publication) -> dict[str, Any]:
    candidate = publication.candidate
    return {
        "roster_name": match.faculty.name,
        "college": match.faculty.college,
        "department": match.faculty.department,
        "openalex_author_name": match.display_name or "",
        "openalex_author_id": match.author_id or "",
        "author_match_status": match.status,
        "author_match_score": match.score,
        "title": candidate.title,
        "authors": "; ".join(candidate.authors),
        "year": candidate.year or "",
        "work_type": publication.work_type or "",
        "venue": candidate.venue or "",
        "doi": candidate.doi or "",
        "openalex_work_id": publication.openalex_work_id or "",
        "citation_count": candidate.citation_count if candidate.citation_count is not None else "",
        "oa_status": publication.oa_status or "",
        "is_oa": publication.is_oa if publication.is_oa is not None else "",
        "pdf_availability": availability(publication),
        "pdf_url_count": len(candidate.candidate_pdf_urls),
        "pdf_urls": "; ".join(candidate.candidate_pdf_urls),
        "landing_page_url": candidate.landing_page_url or "",
        "download_status": candidate.download.get("status", "not_attempted"),
        "download_path": candidate.download.get("path", ""),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def html_json(payload: Any) -> str:
    """Serialize inline page data without allowing an embedded script close."""
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def local_pdf_path(viewer_path: Path, raw_path: Any) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists() or candidate.suffix.casefold() != ".pdf":
        return ""
    try:
        with candidate.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                return ""
    except OSError:
        return ""
    try:
        relative = candidate.resolve().relative_to(viewer_path.parent.resolve())
    except ValueError:
        return ""
    return urllib.parse.quote(relative.as_posix(), safe="/._-+")


def publication_local_pdf(viewer_path: Path, row: dict[str, Any]) -> str:
    recorded = local_pdf_path(viewer_path, row.get("download_path"))
    if recorded:
        return recorded
    try:
        year = int(row["year"]) if str(row.get("year") or "").strip() else None
    except ValueError:
        year = None
    candidate = Candidate(
        title=str(row.get("title") or ""),
        authors=[value.strip() for value in str(row.get("authors") or "").split(";") if value.strip()],
        year=year,
        doi=str(row.get("doi") or "") or None,
        landing_page_url=str(row.get("landing_page_url") or "") or None,
    )
    inferred = (
        viewer_path.parent
        / "authors"
        / (slugify(str(row.get("roster_name") or ""), 60) or "unknown")
        / "pdfs"
        / build_pdf_filename(candidate)
    )
    return local_pdf_path(viewer_path, inferred)


def write_html_viewer(
    path: Path,
    author_rows: list[dict[str, Any]],
    publication_rows: list[dict[str, Any]],
    *,
    generated_at: str,
) -> None:
    """Write a standalone, buildless publication browser with embedded data."""
    authors = [
        {
            "name": row["roster_name"],
            "college": row["college"],
            "department": row["department"],
            "status": row["match_status"],
            "count": int(row["collected_publication_count"] or 0),
        }
        for row in author_rows
    ]
    publications = [
        {
            "name": row["roster_name"],
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "type": row["work_type"],
            "venue": row["venue"],
            "doi": row["doi"],
            "citations": row["citation_count"],
            "pdfs": [value.strip() for value in str(row["pdf_urls"]).split(";") if value.strip()],
            "landing": row["landing_page_url"],
            "local_pdf": publication_local_pdf(path, row),
        }
        for row in publication_rows
    ]
    counts = {
        "authors": len(authors),
        "publications": len(publications),
        "downloads": 0,
    }
    counts["downloads"] = len({row["local_pdf"] for row in publications if row["local_pdf"]})
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Faculty publication inventory</title>
  <style>
    :root { color-scheme: light; --ink:#172033; --muted:#667085; --line:#d9dee8; --paper:#fff; --wash:#f4f6fa; --accent:#2457d6; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--wash); font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    a { color:var(--accent); text-underline-offset:2px; }
    .shell { width:min(1180px,calc(100% - 32px)); margin:0 auto; }
    header { padding:38px 0 22px; }
    h1 { margin:0 0 7px; font-size:clamp(28px,4vw,42px); letter-spacing:-.035em; }
    header p { max-width:780px; margin:0; color:var(--muted); }
    .stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:22px 0; }
    .stat { padding:14px 16px; background:var(--paper); border:1px solid var(--line); border-radius:10px; }
    .stat strong { display:block; font-size:24px; }
    .stat span { color:var(--muted); font-size:13px; }
    .controls { display:grid; grid-template-columns:minmax(260px,1.6fr) minmax(210px,1fr) minmax(170px,.7fr); gap:10px; padding:14px; background:var(--paper); border:1px solid var(--line); border-radius:12px; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; font-weight:650; }
    input,select { width:100%; min-height:42px; padding:8px 10px; color:var(--ink); background:#fff; border:1px solid #bfc7d5; border-radius:7px; font:inherit; }
    .result-bar { display:flex; justify-content:space-between; align-items:center; gap:10px; margin:18px 2px 10px; color:var(--muted); }
    #reset { padding:7px 10px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--accent); cursor:pointer; }
    .list { display:grid; gap:9px; }
    article { padding:16px 18px; background:var(--paper); border:1px solid var(--line); border-radius:10px; }
    article h2 { margin:4px 0 5px; font-size:17px; line-height:1.3; }
    .eyebrow,.meta { color:var(--muted); font-size:13px; }
    .eyebrow { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    .authors { margin:0 0 7px; color:#344054; }
    .links { display:flex; flex-wrap:wrap; gap:8px 14px; margin-top:10px; }
    .pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#eef2f8; color:#465267; font-size:12px; }
    .empty { padding:34px; text-align:center; background:#fff; border:1px dashed #bfc7d5; border-radius:10px; color:var(--muted); }
    #more { display:none; margin:18px auto 36px; padding:10px 18px; border:1px solid var(--accent); border-radius:8px; background:#fff; color:var(--accent); cursor:pointer; }
    footer { padding:4px 0 34px; color:var(--muted); font-size:12px; }
    @media (max-width:800px) { .stats { grid-template-columns:repeat(2,1fr); } .controls { grid-template-columns:1fr 1fr; } }
    @media (max-width:520px) { .shell { width:min(100% - 20px,1180px); } header { padding-top:25px; } .controls { grid-template-columns:1fr; } .stats { gap:7px; } .stat { padding:11px; } article { padding:14px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>Faculty publications</h1>
      <p>Browse the author-resolved publication inventory. Links come from scholarly metadata and open in a new tab.</p>
    </header>
    <section class="stats" aria-label="Inventory summary">
      <div class="stat"><strong id="authorCount"></strong><span>faculty entries</span></div>
      <div class="stat"><strong id="publicationCount"></strong><span>publication records</span></div>
      <div class="stat"><strong id="downloadCount"></strong><span>downloaded PDF files</span></div>
    </section>
    <section class="controls" aria-label="Publication filters">
      <label>Search <input id="search" type="search" placeholder="Title, author, venue, DOI…"></label>
      <label>Faculty member <select id="author"><option value="">All faculty</option></select></label>
      <label>Sort <select id="sort"><option value="citations">Most cited</option><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="title">Title A–Z</option></select></label>
    </section>
    <div class="result-bar"><span id="resultCount" aria-live="polite"></span><button id="reset" type="button">Reset filters</button></div>
    <section id="list" class="list" aria-label="Publications"></section>
    <button id="more" type="button">Show more</button>
    <footer>Generated <span id="generated"></span>. Source: OpenAlex metadata collected by LitAuthorHarvest.</footer>
  </main>
  <noscript><p class="shell empty">JavaScript is required to filter and display this embedded publication inventory.</p></noscript>
  <script>
    const AUTHORS = /*__AUTHORS__*/;
    const PUBLICATIONS = /*__PUBLICATIONS__*/;
    const COUNTS = /*__COUNTS__*/;
    const GENERATED_AT = /*__GENERATED_AT__*/;
    const PAGE_SIZE = 100;
    const $ = id => document.getElementById(id);
    const search = $('search'), author = $('author'), sort = $('sort');
    let visible = PAGE_SIZE;

    const safeHref = value => /^https?:\/\//i.test(value || '') || /^authors\/[a-z0-9_./%+-]+\.pdf$/i.test(value || '') ? value : '';
    const number = value => Number.parseInt(value || '0', 10) || 0;
    const make = (tag, className, text) => { const node=document.createElement(tag); if(className) node.className=className; if(text!==undefined) node.textContent=text; return node; };
    const link = (label, url) => { const a=make('a','',label); a.href=safeHref(url) || '#'; a.target='_blank'; a.rel='noopener noreferrer'; return a; };

    AUTHORS.forEach(item => {
      const option=document.createElement('option'); option.value=item.name;
      option.textContent=`${item.name} (${item.count})${item.status==='needs_review' ? ' — needs identity review' : ''}`;
      author.appendChild(option);
    });
    $('authorCount').textContent=COUNTS.authors.toLocaleString();
    $('publicationCount').textContent=COUNTS.publications.toLocaleString();
    $('downloadCount').textContent=COUNTS.downloads.toLocaleString();
    $('generated').textContent=new Date(GENERATED_AT).toLocaleString();

    function filteredRows() {
      const query=search.value.trim().toLocaleLowerCase();
      const selectedAuthor=author.value;
      const rows=PUBLICATIONS.filter(row => {
        if(selectedAuthor && row.name!==selectedAuthor) return false;
        if(!query) return true;
        return [row.title,row.authors,row.venue,row.doi,row.name].join(' ').toLocaleLowerCase().includes(query);
      });
      rows.sort((a,b) => {
        if(sort.value==='oldest') return number(a.year)-number(b.year) || a.title.localeCompare(b.title);
        if(sort.value==='citations') return number(b.citations)-number(a.citations) || number(b.year)-number(a.year);
        if(sort.value==='title') return a.title.localeCompare(b.title);
        return number(b.year)-number(a.year) || a.title.localeCompare(b.title);
      });
      return rows;
    }

    function publicationCard(row) {
      const card=make('article');
      const eyebrow=make('div','eyebrow');
      eyebrow.append(make('span','pill',row.year || 'Undated'),make('span','',row.name));
      if(row.type) eyebrow.appendChild(make('span','pill',row.type.replaceAll('-', ' ')));
      card.appendChild(eyebrow);
      const heading=make('h2');
      if(safeHref(row.landing)) heading.appendChild(link(row.title,row.landing)); else heading.textContent=row.title;
      card.appendChild(heading);
      if(row.authors) card.appendChild(make('p','authors',row.authors));
      const metadata=[row.venue,row.citations ? `${number(row.citations).toLocaleString()} citations` : ''].filter(Boolean).join(' · ');
      if(metadata) card.appendChild(make('div','meta',metadata));
      const links=make('div','links');
      if(safeHref(row.local_pdf)) links.appendChild(link('Downloaded PDF',row.local_pdf));
      if(safeHref(row.landing)) links.appendChild(link('Publication page',row.landing));
      if(row.doi) links.appendChild(link('DOI',`https://doi.org/${row.doi}`));
      row.pdfs.forEach((url,index) => { if(safeHref(url)) links.appendChild(link(row.pdfs.length>1 ? `PDF source ${index+1}` : 'PDF source',url)); });
      if(links.childNodes.length) card.appendChild(links);
      return card;
    }

    function render(resetPage=true) {
      if(resetPage) visible=PAGE_SIZE;
      const rows=filteredRows(), list=$('list'); list.replaceChildren();
      $('resultCount').textContent=`${rows.length.toLocaleString()} publication${rows.length===1 ? '' : 's'}`;
      rows.slice(0,visible).forEach(row => list.appendChild(publicationCard(row)));
      if(!rows.length) {
        const selected=AUTHORS.find(item => item.name===author.value);
        const message=selected?.status==='needs_review' ? `${selected.name} is awaiting identity review, so no publications were automatically assigned.` : 'No publications match these filters.';
        list.appendChild(make('div','empty',message));
      }
      $('more').style.display=visible<rows.length ? 'block' : 'none';
    }
    [search,author,sort].forEach(control => control.addEventListener(control===search ? 'input' : 'change',() => render(true)));
    $('more').addEventListener('click',() => { visible+=PAGE_SIZE; render(false); });
    $('reset').addEventListener('click',() => { search.value=''; author.value=''; sort.value='citations'; render(true); search.focus(); });
    render();
  </script>
</body>
</html>
"""
    html = (
        template.replace("/*__AUTHORS__*/", html_json(authors))
        .replace("/*__PUBLICATIONS__*/", html_json(publications))
        .replace("/*__COUNTS__*/", html_json(counts))
        .replace("/*__GENERATED_AT__*/", html_json(generated_at))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def author_summary_row(match: AuthorMatch, publication_count: int) -> dict[str, Any]:
    return {
        "roster_name": match.faculty.name,
        "title": match.faculty.title,
        "college": match.faculty.college,
        "department": match.faculty.department,
        "source_rows": ";".join(str(row) for row in match.faculty.source_rows),
        "match_status": match.status,
        "match_score": match.score,
        "openalex_author_name": match.display_name or "",
        "openalex_author_id": match.author_id or "",
        "openalex_works_count": match.works_count if match.works_count is not None else "",
        "collected_publication_count": publication_count,
        "institutions": "; ".join(match.institutions),
        "reason": match.reason or "",
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    roster_path = Path(config["roster"]).expanduser()
    output_dir = Path(config["output"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    authors_dir = output_dir / "authors"
    authors_dir.mkdir(parents=True, exist_ok=True)
    faculty = load_faculty(roster_path)
    http = HttpClient(
        timeout=float(config.get("timeout", 30.0)),
        retries=int(config.get("retries", 2)),
        backoff=float(config.get("backoff", 1.5)),
        rate_limit_delay=float(config.get("rate_limit_delay", 0.12)),
        email=config.get("email"),
    )

    matches: list[AuthorMatch] = []
    all_rows: list[dict[str, Any]] = []
    author_rows: list[dict[str, Any]] = []
    author_manifests: list[dict[str, Any]] = []
    for index, record in enumerate(faculty, start=1):
        LOGGER.info("[%d/%d] Resolving %s (%s)", index, len(faculty), record.name, record.college)
        match = resolve_author(http, record)
        matches.append(match)
        publications: list[Publication] = []
        if match.status == "matched" or (
            match.status == "needs_review" and config.get("include_needs_review")
        ):
            try:
                publications = collect_publications(
                    http,
                    match,
                    year_from=config.get("year_from"),
                    year_to=config.get("year_to"),
                    max_publications=config.get("max_publications_per_author"),
                )
            except Exception as exc:
                match.status = "error"
                match.reason = f"Publication lookup failed: {exc}"
                LOGGER.warning("Publication lookup failed for %s: %s", record.name, exc)

        author_slug = slugify(record.name, 60) or f"author_{index}"
        author_dir = authors_dir / author_slug
        pdf_dir = author_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        if config.get("download") and publications:
            resolver = PdfResolver(http, unpaywall_email=config.get("unpaywall_email"))
            downloader = PdfDownloader(
                http,
                output_dir=pdf_dir,
                max_pdf_mb=int(config.get("max_pdf_mb", 80)),
                allow_abstract_fallback=False,
            )
            limit = int(config.get("max_downloads_per_author", 20))
            attempted = 0
            eligible_publications = (
                [publication for publication in publications if publication.candidate.candidate_pdf_urls]
                if config.get("listed_pdfs_only")
                else publications
            )
            download_queue = sorted(
                eligible_publications,
                key=lambda publication: (
                    bool(publication.candidate.candidate_pdf_urls),
                    publication.candidate.citation_count or 0,
                    publication.candidate.year or 0,
                ),
                reverse=True,
            )
            for publication in download_queue:
                candidate = publication.candidate
                if attempted >= limit:
                    candidate.download = {
                        "status": "not_attempted",
                        "reason": "max_downloads_per_author reached",
                        "filename": build_pdf_filename(candidate),
                    }
                    continue
                if candidate.candidate_pdf_urls:
                    result = downloader.download(candidate, candidate.candidate_pdf_urls)
                else:
                    result = resolve_and_download_candidate(
                        candidate, resolver, downloader, use_web_search=False
                    )
                attempted += 1
                LOGGER.info("%s: %s", result.get("status"), candidate.title)

        rows = [publication_row(match, publication) for publication in publications]
        all_rows.extend(rows)
        author_rows.append(author_summary_row(match, len(publications)))
        match_payload = dataclasses.asdict(match)
        author_manifest = {
            "generated_at": utc_now(),
            "author_match": match_payload,
            "publication_count": len(publications),
            "pdf_url_listed_count": sum(
                availability(publication) == "pdf_url_listed" for publication in publications
            ),
            "publications": [
                {
                    "metadata": dataclasses.asdict(publication),
                    "report_row": publication_row(match, publication),
                }
                for publication in publications
            ],
        }
        write_json(author_dir / "manifest.json", author_manifest)
        write_csv(author_dir / "publications.csv", rows, CSV_FIELDS)
        author_manifests.append(
            {
                "name": record.name,
                "folder": str(author_dir.relative_to(output_dir)),
                "match_status": match.status,
                "publication_count": len(publications),
                "pdf_url_listed_count": author_manifest["pdf_url_listed_count"],
            }
        )

    author_fields = list(author_rows[0]) if author_rows else ["roster_name"]
    write_csv(output_dir / "authors.csv", author_rows, author_fields)
    write_csv(output_dir / "publications.csv", all_rows, CSV_FIELDS)
    generated_at = utc_now()
    write_html_viewer(
        output_dir / "index.html",
        author_rows,
        all_rows,
        generated_at=generated_at,
    )
    manifest = {
        "project": "LitAuthorHarvest",
        "generated_at": generated_at,
        "roster": str(roster_path),
        "roster_row_count": sum(len(record.source_rows) for record in faculty),
        "unique_author_count": len(faculty),
        "matched_author_count": sum(match.status == "matched" for match in matches),
        "needs_review_count": sum(match.status == "needs_review" for match in matches),
        "publication_count": len(all_rows),
        "pdf_url_listed_count": sum(row["pdf_availability"] == "pdf_url_listed" for row in all_rows),
        "downloaded_count": len(
            {
                local_path
                for row in all_rows
                if (local_path := publication_local_pdf(output_dir / "index.html", row))
            }
        ),
        "html_viewer": "index.html",
        "config": {
            key: value
            for key, value in config.items()
            if key not in {"email", "unpaywall_email"} or not value
        },
        "authors": author_manifests,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
