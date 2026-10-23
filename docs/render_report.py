#!/usr/bin/env python3
"""Render literature_review.md to self-contained HTML for PDF printing.

Uses only the Python standard library. The intentionally small Markdown renderer
supports the constructs used by this report: headings, paragraphs, lists,
blockquotes, pipe tables, images, links, emphasis, code, and horizontal rules.
"""

from __future__ import annotations

import base64
import html
import mimetypes
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "literature_review.md"
OUTPUT = HERE / "literature_review.html"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def inline(text: str) -> str:
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = html.escape(text, quote=False)

    def image(match: re.Match[str]) -> str:
        alt, target = match.group(1), html.unescape(match.group(2))
        path = (HERE / target).resolve()
        src = data_uri(path) if path.is_file() else target
        return stash(f'<figure><img src="{src}" alt="{html.escape(alt, quote=True)}"><figcaption>{html.escape(alt)}</figcaption></figure>')

    text = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", image, text)

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), html.unescape(match.group(2))
        if not re.match(r"^[a-z]+://", target):
            target = (HERE / target).resolve().as_uri()
        return stash(f'<a href="{html.escape(target, quote=True)}">{html.escape(label)}</a>')

    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\[@([^]]+)\]", lambda m: f'<span class="citation">[{html.escape(m.group(1).replace("; @", "; "))}]</span>', text)
    text = text.replace("  ", " ")
    for i, value in enumerate(placeholders):
        text = text.replace(f"\x00{i}\x00", value)
    return text


def is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def render_markdown(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + inline(" ".join(x.strip() for x in paragraph)) + "</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2)
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quote.append(lines[i].strip()[2:])
                i += 1
            out.append("<blockquote><p>" + inline(" ".join(quote)) + "</p></blockquote>")
            continue
        if "|" in line and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            flush_paragraph()
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            table = ["<table><thead><tr>"]
            table.extend(f"<th>{inline(c)}</th>" for c in headers)
            table.append("</tr></thead><tbody>")
            for row in rows:
                table.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            table.append("</tbody></table>")
            out.append("".join(table))
            continue
        list_match = re.match(r"^\s*(?:(\d+)\.|-)\s+(.+)$", line)
        if list_match:
            flush_paragraph()
            ordered = list_match.group(1) is not None
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^\s*(?:(\d+)\.|-)\s+(.+)$", lines[i])
                if not match or (match.group(1) is not None) != ordered:
                    break
                items.append(match.group(2))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(item)}</li>" for item in items) + f"</{tag}>")
            continue
        if stripped in {"---", "***"}:
            flush_paragraph()
            out.append("<hr>")
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return "\n".join(out)


CSS = r"""
@page { size: Letter; margin: 0.72in 0.72in 0.7in; @bottom-right { content: counter(page); } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 8.2in; color: #17212b; font: 10.25pt/1.48 Georgia, 'Times New Roman', serif; }
h1, h2, h3 { font-family: Arial, Helvetica, sans-serif; color: #173b62; page-break-after: avoid; }
h1 { margin: 0.3in 0 0.08in; font-size: 25pt; line-height: 1.12; border-bottom: 3px solid #3568a8; padding-bottom: 0.12in; }
h2 { margin: 0.24in 0 0.08in; font-size: 16pt; border-bottom: 1px solid #b8c7d8; padding-bottom: 3px; }
h3 { margin: 0.18in 0 0.04in; font-size: 12.5pt; }
p { margin: 0.06in 0 0.11in; text-align: justify; orphans: 3; widows: 3; }
ul, ol { margin: 0.05in 0 0.12in 0.25in; padding-left: 0.18in; }
li { margin: 0.025in 0; }
blockquote { margin: 0.15in 0; padding: 0.08in 0.15in; border-left: 4px solid #5684b3; background: #f1f6fb; }
blockquote p { margin: 0; text-align: left; }
table { width: 100%; margin: 0.12in 0 0.18in; border-collapse: collapse; font: 8.6pt/1.3 Arial, sans-serif; page-break-inside: avoid; }
th { background: #dfeaf5; color: #173b62; text-align: left; }
th, td { border: 1px solid #aebdca; padding: 5px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fb; }
figure { margin: 0.18in auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 6.4in; }
figcaption { margin-top: 4px; font: italic 8.5pt Arial, sans-serif; color: #52606d; }
code { padding: 1px 3px; border-radius: 3px; background: #eef1f4; font: 8.8pt Menlo, Consolas, monospace; }
a { color: #245f98; text-decoration: none; }
.citation { color: #4d5f71; }
@media print {
  body { max-width: none; }
  h1 { page-break-before: auto; }
  h2 { break-before: auto; }
  a { color: inherit; }
}
"""


def main() -> None:
    body = render_markdown(SOURCE.read_text(encoding="utf-8"))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Building an Evidence-Backed Taxonomy for Persuasion in Media</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
