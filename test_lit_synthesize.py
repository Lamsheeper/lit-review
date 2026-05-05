import json
import tempfile
import unittest
from pathlib import Path

from lit_synthesize import (
    build_index,
    collect_paper_records,
    convert_pdfs_to_markdown,
    search_index,
    write_report,
)


class LitSynthCoreTests(unittest.TestCase):
    def test_collect_paper_records_uses_litharvest_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            papers_dir = root / "papers"
            papers_dir.mkdir()
            pdf_path = papers_dir / "2024_lovelace_moral_framing_test_abc123.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            manifest = {
                "candidates": [
                    {
                        "title": "Moral Framing in News",
                        "authors": ["Ada Lovelace"],
                        "year": 2024,
                        "doi": "10.1000/test",
                        "venue": "Journal of Tests",
                        "download": {
                            "status": "downloaded",
                            "path": str(pdf_path),
                        },
                    }
                ]
            }

            records = collect_paper_records(papers_dir, manifest=manifest)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].title, "Moral Framing in News")
            self.assertEqual(records[0].authors, ["Ada Lovelace"])
            self.assertEqual(records[0].year, 2024)
            self.assertTrue(records[0].paper_id.startswith("moral_framing_in_news_"))

    def test_index_and_search_markdown_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            md_dir = run_dir / "papers_md"
            md_dir.mkdir()
            md_path = md_dir / "paper1.md"
            md_path.write_text(
                """---
paper_id: "paper1"
title: "Moral Framing and Stance"
authors: ["Ada Lovelace"]
year: 2024
doi: null
venue: null
pdf_path: "paper1.pdf"
---

# Moral Framing and Stance

## Page 1

Moral framing theory studies care harm fairness authority loyalty and sanctity.
News media stance detection connects target specific positions to political
communication and institution legitimacy.

## Page 2

Emotion classification and propaganda technique taxonomies often require
multi label annotation, construct validity, and inter annotator agreement.
""",
                encoding="utf-8",
            )
            (run_dir / "paper_index.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "paper1",
                                "title": "Moral Framing and Stance",
                                "authors": ["Ada Lovelace"],
                                "year": 2024,
                                "doi": None,
                                "venue": None,
                                "pdf_path": "paper1.pdf",
                                "md_path": str(md_path),
                                "conversion": {"status": "converted"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            meta = build_index(
                run_dir,
                chunk_words_count=40,
                overlap_words=5,
                min_chunk_words=5,
                force=True,
            )
            results = search_index(run_dir / "index.sqlite", "moral framing authority", limit=3)

            self.assertGreaterEqual(meta["chunk_count"], 2)
            self.assertTrue(results)
            self.assertEqual(results[0]["paper_id"], "paper1")
            self.assertIn("moral", results[0]["text"].lower())

    def test_write_report_offline_creates_report_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            md_dir = run_dir / "papers_md"
            md_dir.mkdir()
            md_path = md_dir / "paper1.md"
            md_path.write_text(
                """---
paper_id: "paper1"
title: "Propaganda Taxonomies"
authors: []
year: 2023
doi: null
venue: null
pdf_path: "paper1.pdf"
---

# Propaganda Taxonomies

## Page 1

Propaganda technique classification includes loaded language, name calling,
appeal to fear, and causal oversimplification. Annotation schemas need
construct validity and clear non redundant categories.
""",
                encoding="utf-8",
            )
            (run_dir / "paper_index.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "paper1",
                                "title": "Propaganda Taxonomies",
                                "authors": [],
                                "year": 2023,
                                "doi": None,
                                "venue": None,
                                "pdf_path": "paper1.pdf",
                                "md_path": str(md_path),
                                "conversion": {"status": "converted"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            goal_path = run_dir / "goal.md"
            goal_path.write_text(
                "# Build a propaganda persuasion taxonomy\n\n"
                "Find evidence about propaganda technique classification and "
                "annotation validity.",
                encoding="utf-8",
            )
            build_index(
                run_dir,
                chunk_words_count=80,
                overlap_words=5,
                min_chunk_words=5,
                force=True,
            )

            trace = write_report(
                run_dir=run_dir,
                goal_path=goal_path,
                output_path=run_dir / "report.md",
                api_key=None,
                per_question=3,
                max_evidence=5,
            )
            report = (run_dir / "report.md").read_text(encoding="utf-8")

            self.assertFalse(trace["llm_used"])
            self.assertTrue(trace["evidence"])
            self.assertIn("Evidence Map", report)
            self.assertIn("[paper1:p1:c1]", report)
            self.assertTrue((run_dir / "agent_trace.json").exists())

    def test_convert_records_failed_when_no_extractor_can_read_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            papers_dir = root / "papers"
            papers_dir.mkdir()
            (papers_dir / "2024_unknown_test_deadbeef.pdf").write_bytes(b"%PDF-1.4\n")

            with self.assertLogs("litsynth", level="WARNING"):
                payload = convert_pdfs_to_markdown(
                    papers_dir=papers_dir,
                    manifest_path=None,
                    out_dir=root / "run",
                    extractor="pypdf",
                    force=True,
                )

            self.assertEqual(payload["paper_count"], 1)
            self.assertEqual(payload["failed_count"], 1)
            self.assertIn("pypdf", payload["papers"][0]["conversion"]["reason"])


if __name__ == "__main__":
    unittest.main()
