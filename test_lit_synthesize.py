import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lit_synthesize import (
    build_index,
    ChatResult,
    collect_paper_records,
    convert_pdfs_to_markdown,
    llm_plan_goal,
    llm_plan_taxonomy_goal,
    main,
    parse_args,
    search_index,
    taxonomy_plan_goal,
    write_llm_report,
    write_report,
)
from lit_pdf_service import prepare_direct_pdf_index


class LitSynthCoreTests(unittest.TestCase):
    def test_prepare_direct_pdf_index_does_not_create_markdown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            papers_dir = root / "papers"
            papers_dir.mkdir()
            pdf_path = papers_dir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            out_dir = root / "run"

            payload = prepare_direct_pdf_index(papers_dir, None, out_dir)

            self.assertEqual(payload["mode"], "direct_pdf")
            self.assertEqual(payload["paper_count"], 1)
            self.assertTrue((out_dir / "paper_index.json").exists())
            self.assertFalse((out_dir / "papers_md").exists())
            self.assertEqual(payload["papers"][0]["conversion"]["status"], "direct_pdf_ready")

    def test_prepare_direct_pdf_index_excludes_unselected_existing_pdfs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            papers_dir = root / "papers"
            papers_dir.mkdir()
            selected = papers_dir / "selected.pdf"
            unselected = papers_dir / "unselected.pdf"
            selected.write_bytes(b"%PDF-1.4\n")
            unselected.write_bytes(b"%PDF-1.4\n")
            manifest_path = papers_dir / "logs" / "manifest.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Selected",
                                "download": {"status": "downloaded", "path": str(selected)},
                            },
                            {
                                "title": "Unselected",
                                "download": {"status": "not_selected", "path": str(unselected)},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = prepare_direct_pdf_index(papers_dir, manifest_path, root / "run")

            self.assertEqual(payload["paper_count"], 1)
            self.assertEqual(Path(payload["papers"][0]["pdf_path"]), selected)

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

    def test_config_file_can_drive_index_command(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            md_dir = run_dir / "papers_md"
            md_dir.mkdir()
            md_path = md_dir / "paper1.md"
            md_path.write_text(
                """---
paper_id: "paper1"
title: "Emotion and Stance"
authors: []
year: 2024
doi: null
venue: null
pdf_path: "paper1.pdf"
---

# Emotion and Stance

## Page 1

Emotion classification, stance detection, and media framing are useful for
propaganda analysis and persuasion taxonomy validation.
""",
                encoding="utf-8",
            )
            (run_dir / "paper_index.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "paper1",
                                "title": "Emotion and Stance",
                                "authors": [],
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
            config_path = run_dir / "index_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "command": "index",
                        "run": str(run_dir),
                        "chunk_words": 60,
                        "overlap_words": 5,
                        "min_chunk_words": 5,
                        "force": True,
                        "quiet": True,
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(["--config", str(config_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "index.sqlite").exists())

    def test_config_file_supports_api_key_env_without_storing_secret(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "write_gemini.json"
            config_path.write_text(
                json.dumps(
                    {
                        "command": "write",
                        "run": "review_runs/demo",
                        "goal": "draft.md",
                        "output": "review_runs/demo/report.md",
                        "api_key_env": "GEMINI_API_KEY",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "model": "gemini-3-flash-preview",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}):
                args = parse_args(["--config", str(config_path)])

            self.assertEqual(args.command, "write")
            self.assertEqual(args.api_key, "test-secret")
            self.assertEqual(args.model, "gemini-3-flash-preview")

    def test_llm_plan_goal_repairs_malformed_json(self):
        class FakeClient:
            def __init__(self):
                self.calls = []
                self.responses = [
                    '{"title": "Broken Plan", "questions": ["What is framing?", "unterminated]',
                    json.dumps(
                        {
                            "title": "Repaired Plan",
                            "questions": ["What is framing?", "What is stance?"],
                            "outline": ["Evidence Map"],
                        }
                    ),
                ]

            def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                self.calls.append(
                    {
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "response_format": response_format,
                    }
                )
                return self.responses.pop(0)

        client = FakeClient()

        with self.assertLogs("litsynth", level="WARNING"):
            plan = llm_plan_goal(
                "Build a persuasion taxonomy.",
                client=client,
                max_questions=1,
            )

        self.assertEqual(plan["title"], "Repaired Plan")
        self.assertEqual(plan["questions"], ["What is framing?"])
        self.assertEqual(plan["outline"], ["Evidence Map"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[1]["response_format"], {"type": "json_object"})

    def test_write_llm_report_generates_sections_separately(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                self.calls.append(
                    {
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "response_format": response_format,
                    }
                )
                return f"## Generated Section {len(self.calls)}\n\nClaim [paper1:p1:c1]."

            def chat_result(
                self,
                messages,
                temperature=0.2,
                max_tokens=None,
                response_format=None,
            ):
                return ChatResult(
                    content=self.chat(messages, temperature, max_tokens, response_format),
                    finish_reason="stop",
                )

        with tempfile.TemporaryDirectory() as temp:
            output_path = Path(temp) / "report.md"
            client = FakeClient()
            report = write_llm_report(
                goal_text="Build a persuasion taxonomy.",
                plan={
                    "title": "Taxonomy Report",
                    "questions": ["What is persuasion?"],
                    "outline": [
                        "1. Introduction",
                        "1.1 Scope",
                        "2. Moral Framing",
                    ],
                },
                evidence=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "title": "Persuasion Paper",
                        "year": 2024,
                        "pages": [1],
                        "question": "What is persuasion?",
                        "text": "Persuasion techniques include fear appeals.",
                        "score": -1.0,
                    },
                    {
                        "chunk_id": "paper1:p2:c1",
                        "title": "Moral Paper",
                        "year": 2024,
                        "pages": [2],
                        "question": "What is moral framing?",
                        "text": "Moral framing includes care and fairness.",
                        "score": -2.0,
                    },
                ],
                client=client,
                output_path=output_path,
                section_evidence=1,
            )

        self.assertTrue(report.startswith("# Taxonomy Report"))
        self.assertEqual(len(client.calls), 3)
        self.assertIn("Generated Section 1", report)
        self.assertIn("Generated Section 2", report)
        self.assertIn("Generated Section 3", report)
        self.assertIn("Evidence Gaps", client.calls[2]["messages"][1]["content"])

    def test_taxonomy_plan_uses_taxonomy_specific_outline(self):
        plan = taxonomy_plan_goal(
            "Extract a persuasion taxonomy with moral framing and emotional targeting.",
            max_questions=4,
        )

        self.assertEqual(plan["mode"], "taxonomy")
        self.assertEqual(len(plan["questions"]), 4)
        outline = "\n".join(plan["outline"])
        self.assertIn("Category 1: Persuasion Techniques", outline)
        self.assertIn("Category 2: Moral Framing", outline)
        self.assertIn("Category 3: Emotional and Affective Targeting", outline)

    def test_llm_taxonomy_plan_uses_model_questions_with_fixed_outline(self):
        class FakeClient:
            def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                return json.dumps(
                    {
                        "questions": [
                            "What named propaganda techniques recur across datasets?",
                            "How are moral frames operationalized?",
                        ]
                    }
                )

        plan = llm_plan_taxonomy_goal(
            "Extract a persuasion taxonomy with moral framing and emotional targeting.",
            client=FakeClient(),
            max_questions=3,
        )

        self.assertEqual(plan["question_source"], "llm")
        self.assertEqual(
            plan["questions"][0],
            "What named propaganda techniques recur across datasets?",
        )
        outline = "\n".join(plan["outline"])
        self.assertIn("Category 1: Persuasion Techniques", outline)

    def test_taxonomy_section_prompt_requests_feature_table(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                self.calls.append(messages[1]["content"])
                return "## Category 1: Persuasion Techniques\n\n| Feature name | Evidence |\n|---|---|\n"

            def chat_result(
                self,
                messages,
                temperature=0.2,
                max_tokens=None,
                response_format=None,
            ):
                return ChatResult(
                    content=self.chat(messages, temperature, max_tokens, response_format),
                    finish_reason="stop",
                )

        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            write_llm_report(
                goal_text="Extract a persuasion taxonomy.",
                plan={
                    "title": "Taxonomy",
                    "questions": ["What persuasion techniques are used?"],
                    "outline": ["2. Category 1: Persuasion Techniques"],
                },
                evidence=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "title": "Persuasion Paper",
                        "year": 2024,
                        "pages": [1],
                        "question": "What persuasion techniques are used?",
                        "text": "Loaded language is a propaganda technique.",
                        "score": -1.0,
                    }
                ],
                client=client,
                output_path=Path(temp) / "report.md",
            )

        self.assertIn("Build a taxonomy table for persuasion techniques", client.calls[0])
        self.assertIn("Parent group", client.calls[0])
        self.assertIn("Feature name", client.calls[0])

    def test_write_llm_report_continues_truncated_sections(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def chat_result(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="## 1. Taxonomy Design Principles\n\n- Partial",
                        finish_reason="length",
                    )
                return ChatResult(content="- continued [paper1:p1:c1].", finish_reason="stop")

        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            report = write_llm_report(
                goal_text="Extract a persuasion taxonomy.",
                plan={
                    "title": "Taxonomy",
                    "questions": ["What evidence exists?"],
                    "outline": ["1. Taxonomy Design Principles"],
                },
                evidence=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "title": "Persuasion Paper",
                        "year": 2024,
                        "pages": [1],
                        "question": "What evidence exists?",
                        "text": "Taxonomies require operational boundaries.",
                        "score": -1.0,
                    }
                ],
                client=client,
                output_path=Path(temp) / "report.md",
            )

        self.assertEqual(client.calls, 3)
        self.assertIn("Partial", report)
        self.assertIn("continued", report)


if __name__ == "__main__":
    unittest.main()
