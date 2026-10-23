import csv
import json
import tempfile
import unittest
from lit_review.core.context import PROFILE, TAXONOMY
from pathlib import Path

from lit_review.features.records import (
    aggregate_features,
    aggregate_paper_features,
    canonical_category,
    normalize_feature_item,
)
from lit_review.features.prompts import (
    build_annotation_prompt,
    build_extraction_prompt,
    build_scoring_prompt,
    literature_support_score,
)
from lit_review.features.scoring_artifacts import build_scored_features, total_feature_score
from lit_review.features.scoring import run_scoring
from lit_review.core.llm import build_gemini_generate_content_payload
from lit_review.features.extraction import build_pdf_extraction_messages, load_papers, run_extraction
from lit_review.core.text import normalize_feature_name
from lit_review.profiles.media_ranking import title_relevance_score
from lit_review.features.extraction_artifacts import write_artifacts
from lit_review.profiles.media_attributes import annotation_exclusion_reason
from lit_review.features.annotation_artifacts import (
    normalize_annotation_decision,
    normalize_annotation_synonyms,
)
from lit_review.features.annotation import run_annotation
from lit_review.features.responses import (
    parse_annotation_json_with_repair,
    parse_curation_json_with_repair,
    parse_features_json,
    parse_features_json_with_repair,
    parse_scoring_json_with_repair,
)
from lit_review.features.filtering import run_curation
from lit_review.features.merging import run_llm_merge


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_run_dir(root: Path, papers, chunks) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    (run_dir / "paper_index.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        **paper,
                        "authors": paper.get("authors", []),
                        "doi": paper.get("doi"),
                        "venue": paper.get("venue"),
                        "conversion": {"status": "converted"},
                    }
                    for paper in papers
                ]
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "chunks.jsonl", chunks)
    return run_dir


def feature_row(
    category,
    name,
    *,
    parents=None,
    definitions=None,
    synonyms=None,
    examples=None,
    paper_ids=None,
    titles=None,
    mention_count=1,
    confidence=0.8,
    notes="",
):
    normalized = normalize_feature_name(name)
    paper_ids = paper_ids or ["paper1"]
    titles = titles or ["Paper One"]
    return {
        "category": category,
        "feature_name": name,
        "normalized_feature_name": normalized,
        "feature_key": f"{category}::{normalized}",
        "parent_categories": parents or [],
        "definitions": definitions or [],
        "synonyms": synonyms or [],
        "examples": examples or [],
        "paper_count": len(set(paper_ids)),
        "mention_count": mention_count,
        "source_paper_ids": paper_ids,
        "source_titles": titles,
        "evidence_types": ["definition"],
        "confidence": confidence,
        "notes": notes,
        "model": "extract-model",
        "run_id": "extract-run",
        "extracted_at": "2026-05-08T00:00:00+00:00",
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
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
        if not self.responses:
            raise AssertionError("Unexpected chat call.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeEmbeddingClient:
    def __init__(self, embeddings):
        self.embeddings = list(embeddings)
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if len(texts) > len(self.embeddings):
            raise AssertionError("FakeEmbeddingClient ran out of embeddings.")
        output = self.embeddings[: len(texts)]
        self.embeddings = self.embeddings[len(texts) :]
        return output

    @property
    def all_texts(self):
        return [text for call in self.calls for text in call]


class LitExtractTests(unittest.TestCase):
    def setUp(self):
        token = PROFILE.set("media_attributes")
        self.addCleanup(PROFILE.reset, token)
        taxonomy = {
            "version": 1,
            "title": "Media attributes",
            "families": [
                {"id": key, "label": key, "aliases": []}
                for key in ("persuasion", "moral_framing", "sentiment_affect")
            ],
        }
        tax_token = TAXONOMY.set(taxonomy)
        self.addCleanup(TAXONOMY.reset, tax_token)

    def test_parse_features_json_recovers_fenced_json(self):
        content = """```json
        {
          "features": [
            {
              "category": "Persuasion Techniques",
              "feature_name": "Loaded language"
            }
          ]
        }
        ```"""

        features = parse_features_json(content)

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["feature_name"], "Loaded language")

    def test_normalize_feature_name_and_category_aliases(self):
        self.assertEqual(normalize_feature_name("Care/Harm"), "care harm")
        self.assertEqual(normalize_feature_name("Name-calling"), "name calling")
        self.assertEqual(canonical_category("Moral Foundations"), "moral_framing")
        self.assertEqual(canonical_category("Emotional / Affective Targeting"), "sentiment_affect")
        self.assertEqual(canonical_category("Propaganda Techniques"), "persuasion")

    def test_extraction_prompt_includes_goal_and_feature_cap(self):
        prompt = build_extraction_prompt(
            {
                "paper_id": "paper1",
                "title": "Propaganda Taxonomies",
                "authors": [],
                "year": 2024,
            },
            [
                {
                    "chunk_id": "paper1:p1:c1",
                    "pages": [1],
                    "text": "Loaded language is a propaganda technique.",
                }
            ],
            max_chunk_chars=1000,
            goal_text="Preserve distinctions between rhetoric, morality, and emotion.",
            max_features_per_chunk=3,
        )

        self.assertIn("GOAL / TAXONOMY CONTEXT", prompt)
        self.assertIn("Preserve distinctions", prompt)
        self.assertIn("Return at most 3 features per chunk", prompt)

    def test_parse_features_json_repairs_malformed_model_output(self):
        malformed = """{
          "features": [
            {
              "category": "persuasion",
              "feature_name": "Loaded language",
              "source_quote": "calls this "loaded language" in the taxonomy"
            }
          ]
        }"""
        repaired = json.dumps(
            {
                "features": [
                    {
                        "category": "persuasion",
                        "feature_name": "Loaded language",
                        "source_quote": 'calls this "loaded language" in the taxonomy',
                    }
                ]
            }
        )
        client = FakeClient([repaired])

        features, was_repaired = parse_features_json_with_repair(malformed, client=client)

        self.assertTrue(was_repaired)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["feature_name"], "Loaded language")
        self.assertEqual(len(client.calls), 1)

    def test_normalize_feature_item_filters_irrelevant_ml_features(self):
        paper = {"paper_id": "paper1", "title": "Point Cloud Compression"}
        chunk = {
            "chunk_id": "paper1:p1:c1",
            "paper_id": "paper1",
            "pages": [1],
            "text": "We propose a geometric feature extraction method for point cloud compression.",
        }
        item = {
            "category": "persuasion",
            "feature_name": "geometric feature",
            "source_quote": "geometric feature extraction method",
        }

        row = normalize_feature_item(
            item,
            paper=paper,
            chunk=chunk,
            run_id="test",
            model="fake-model",
            extracted_at="2026-05-08T00:00:00+00:00",
        )

        self.assertIsNone(row)

    def test_aggregation_deduplicates_and_csv_serializes_lists(self):
        rows = [
            {
                "paper_id": "paper1",
                "title": "Propaganda Taxonomies",
                "authors": ["Ada Lovelace"],
                "year": 2024,
                "doi": None,
                "venue": "Journal",
                "category": "persuasion",
                "feature_name": "Loaded Language",
                "normalized_feature_name": "loaded language",
                "parent_category": "Manipulative wording",
                "definition": "",
                "synonyms": ["emotive language"],
                "examples": [],
                "evidence_type": "named_mention",
                "chunk_id": "paper1:p1:c1",
                "pages": [1],
                "snippet": "Loaded language is listed.",
                "source_quote": "Loaded language",
                "confidence": 0.6,
                "notes": "",
                "model": "fake-model",
                "run_id": "run",
                "extracted_at": "2026-05-08T00:00:00+00:00",
            },
            {
                "paper_id": "paper1",
                "title": "Propaganda Taxonomies",
                "authors": ["Ada Lovelace"],
                "year": 2024,
                "doi": None,
                "venue": "Journal",
                "category": "persuasion",
                "feature_name": "loaded language",
                "normalized_feature_name": "loaded language",
                "parent_category": "Manipulative wording",
                "definition": "Emotionally charged wording.",
                "synonyms": [],
                "examples": ["crooked"],
                "evidence_type": "definition",
                "chunk_id": "paper1:p2:c1",
                "pages": [2],
                "snippet": "Loaded language is emotionally charged wording.",
                "source_quote": "emotionally charged wording",
                "confidence": 0.9,
                "notes": "",
                "model": "fake-model",
                "run_id": "run",
                "extracted_at": "2026-05-08T00:00:00+00:00",
            },
        ]

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            counts = write_artifacts(output_dir, rows)
            paper_features = aggregate_paper_features(rows)
            features = aggregate_features(rows)

            self.assertEqual(counts["feature_mentions"], 2)
            self.assertEqual(len(paper_features), 1)
            self.assertEqual(paper_features[0]["mention_count"], 2)
            self.assertEqual(len(features), 1)

            with (output_dir / "feature_mentions.csv").open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(
                json.loads(csv_rows[0]["authors"]),
                ["Ada Lovelace"],
            )

    def test_parse_curation_json_repairs_malformed_model_output(self):
        malformed = """{
          "decisions": [
            {
              "feature_key": "moral_framing::authority",
              "action": "keep_as_canonical",
              "canonical_feature_name": "Authority",
              "notes": "called "Authority" in MFT"
            }
          ]
        }"""
        repaired = json.dumps(
            {
                "decisions": [
                    {
                        "feature_key": "moral_framing::authority",
                        "action": "keep_as_canonical",
                        "canonical_feature_name": "Authority",
                        "notes": 'called "Authority" in MFT',
                    }
                ]
            }
        )
        client = FakeClient([repaired])

        decisions, was_repaired = parse_curation_json_with_repair(malformed, client=client)

        self.assertTrue(was_repaired)
        self.assertEqual(decisions[0]["feature_key"], "moral_framing::authority")
        self.assertEqual(len(client.calls), 1)

    def test_run_curation_writes_curated_outputs_with_mock_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "extract"
            input_dir.mkdir()
            write_jsonl(
                input_dir / "features.jsonl",
                [
                    feature_row(
                        "moral_framing",
                        "Authority",
                        paper_ids=["p1"],
                        mention_count=5,
                    ),
                    feature_row(
                        "moral_framing",
                        "Authority Foundation",
                        paper_ids=["p2"],
                        mention_count=2,
                    ),
                ],
            )
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "decisions": [
                                {
                                    "feature_key": "moral_framing::authority",
                                    "action": "keep_as_canonical",
                                    "canonical_feature_name": "Authority",
                                    "canonical_normalized_feature_name": "authority",
                                    "curation_confidence": 0.95,
                                    "notes": "Canonical.",
                                },
                                {
                                    "feature_key": "moral_framing::authority foundation",
                                    "action": "merge_into_canonical",
                                    "canonical_feature_name": "Authority",
                                    "canonical_normalized_feature_name": "authority",
                                    "curation_confidence": 0.9,
                                    "notes": "Alias.",
                                },
                            ]
                        }
                    )
                ]
            )

            trace = run_curation(
                input_dir=input_dir,
                output_dir=input_dir,
                api_key=None,
                model="curator",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_features=40,
                force=True,
                dry_run=False,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(trace["outputs"]["filtered_features"], 2)
            self.assertEqual(trace["outputs"]["curation_decisions"], 2)
            self.assertEqual(trace["outputs"]["rejected_features"], 0)
            for name in [
                "filtered_features.jsonl",
                "filtered_features.csv",
                "filtered_features.jsonl",
                "rejected_features.jsonl",
                "rejected_features.csv",
                "curation_decisions.jsonl",
                "curation_trace.json",
            ]:
                self.assertTrue((input_dir / name).exists(), name)
            filtered_rows = [
                json.loads(line)
                for line in (input_dir / "filtered_features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["feature_name"] for row in filtered_rows],
                ["Authority", "Authority Foundation"],
            )
            decision_rows = [
                json.loads(line)
                for line in (input_dir / "curation_decisions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["action"] for row in decision_rows], ["keep", "keep"])
            self.assertFalse((input_dir / "final_curated_features_top100_by_category.jsonl").exists())

    def test_run_curation_resume_skips_completed_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "extract"
            input_dir.mkdir()
            write_jsonl(
                input_dir / "features.jsonl",
                [
                    feature_row("moral_framing", "Authority"),
                    feature_row("moral_framing", "Authority Foundation"),
                ],
            )
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "decisions": [
                                {
                                    "feature_key": "moral_framing::authority",
                                    "action": "keep_as_canonical",
                                    "canonical_feature_name": "Authority",
                                    "canonical_normalized_feature_name": "authority",
                                },
                                {
                                    "feature_key": "moral_framing::authority foundation",
                                    "action": "merge_into_canonical",
                                    "canonical_feature_name": "Authority",
                                    "canonical_normalized_feature_name": "authority",
                                },
                            ]
                        }
                    )
                ]
            )
            run_curation(
                input_dir=input_dir,
                output_dir=input_dir,
                api_key=None,
                model="curator",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_features=40,
                force=True,
                dry_run=False,
                temperature=0.0,
                show_progress=False,
                client=first_client,
            )

            second_client = FakeClient([])
            trace = run_curation(
                input_dir=input_dir,
                output_dir=input_dir,
                api_key=None,
                model="curator",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_features=40,
                force=False,
                dry_run=False,
                temperature=0.0,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 1)
            self.assertEqual(trace["outputs"]["filtered_features"], 2)

    def test_run_extraction_writes_normalized_outputs_with_mock_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = make_run_dir(
                root,
                papers=[
                    {
                        "paper_id": "paper1",
                        "title": "Propaganda Taxonomies",
                        "authors": ["Ada Lovelace"],
                        "year": 2024,
                    },
                    {
                        "paper_id": "paper2",
                        "title": "Moral Framing",
                        "authors": ["Grace Hopper"],
                        "year": 2023,
                    },
                ],
                chunks=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "paper_id": "paper1",
                        "title": "Propaganda Taxonomies",
                        "year": 2024,
                        "pages": [1],
                        "text": "Propaganda technique taxonomies include loaded language.",
                        "token_count": 7,
                    },
                    {
                        "chunk_id": "paper2:p1:c1",
                        "paper_id": "paper2",
                        "title": "Moral Framing",
                        "year": 2023,
                        "pages": [1],
                        "text": "Moral Foundations Theory includes Care/Harm as a moral frame.",
                        "token_count": 9,
                    },
                ],
            )
            output_dir = root / "extract"
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "features": [
                                {
                                    "category": "persuasion",
                                    "feature_name": "Loaded language",
                                    "evidence_type": "named_mention",
                                    "chunk_id": "paper1:p1:c1",
                                    "source_quote": "loaded language",
                                    "confidence": 0.7,
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "features": [
                                {
                                    "category": "moral_framing",
                                    "feature_name": "Care/Harm",
                                    "definition": "a moral frame",
                                    "evidence_type": "definition",
                                    "chunk_id": "paper2:p1:c1",
                                    "source_quote": "Care/Harm as a moral frame",
                                    "confidence": 0.8,
                                }
                            ]
                        }
                    ),
                ]
            )

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=output_dir,
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                client=client,
            )

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(trace["outputs"]["feature_mentions"], 2)
            for name in [
                "feature_mentions.jsonl",
                "feature_mentions.csv",
                "paper_features.jsonl",
                "paper_features.csv",
                "features.jsonl",
                "features.csv",
                "extraction_errors.jsonl",
                "extraction_trace.json",
            ]:
                self.assertTrue((output_dir / name).exists(), name)

    def test_run_extraction_can_send_source_pdf_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "paper_index.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "paper1",
                                "title": "Propaganda Technique Taxonomies",
                                "authors": ["Ada Lovelace"],
                                "year": 2024,
                                "pdf_path": str(pdf_path),
                                "conversion": {
                                    "status": "failed",
                                    "reason": "local extractor unavailable",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "extract"
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "features": [
                                {
                                    "category": "persuasion",
                                    "feature_name": "Loaded language",
                                    "evidence_type": "named_mention",
                                    "chunk_id": "paper1:pdf",
                                    "pages": [2],
                                    "source_quote": "loaded language",
                                    "confidence": 0.7,
                                }
                            ]
                        }
                    )
                ]
            )

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=output_dir,
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                input_mode="pdf",
                batch_chunks_count=99,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                max_features_per_pdf=5,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 1)
            content = client.calls[0]["messages"][1]["content"]
            self.assertEqual(content[0]["type"], "file")
            self.assertEqual(content[0]["file"]["filename"], "paper.pdf")
            self.assertTrue(content[0]["file"]["file_data"].startswith("data:application/pdf;base64,"))
            self.assertIn("Return at most 5 features from this PDF", content[1]["text"])
            self.assertEqual(trace["config"]["input_mode"], "pdf")
            self.assertEqual(trace["counts"]["pdf_count"], 1)
            self.assertEqual(trace["counts"]["chunk_count"], 0)
            self.assertEqual(trace["outputs"]["feature_mentions"], 1)

            rows = [
                json.loads(line)
                for line in (output_dir / "feature_mentions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["chunk_id"], "paper1:pdf")
            self.assertEqual(rows[0]["pages"], [2])

    def test_gemini_payload_maps_pdf_file_parts_to_inline_data(self):
        with tempfile.TemporaryDirectory() as temp:
            pdf_path = Path(temp) / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            messages = build_pdf_extraction_messages(
                paper={
                    "paper_id": "paper1",
                    "title": "Propaganda Technique Taxonomies",
                    "authors": [],
                    "year": 2024,
                },
                pdf_path=pdf_path,
                goal_text="Extract persuasion techniques.",
                max_features_per_pdf=5,
            )

            payload = build_gemini_generate_content_payload(
                messages=messages,
                model="gemini-3.1-flash-lite",
                reasoning_effort="low",
                temperature=0.1,
                max_tokens=5000,
                response_format={"type": "json_object"},
            )

            self.assertEqual(
                payload["generationConfig"]["responseMimeType"],
                "application/json",
            )
            self.assertEqual(
                payload["generationConfig"]["thinkingConfig"],
                {"thinkingLevel": "low"},
            )
            parts = payload["contents"][0]["parts"]
            self.assertEqual(parts[0]["inline_data"]["mime_type"], "application/pdf")
            self.assertNotIn(",", parts[0]["inline_data"]["data"])
            self.assertIn("attached academic PDF", parts[1]["text"])
            self.assertIn("taxonomy extraction agent", payload["systemInstruction"]["parts"][0]["text"])

    def test_title_ranking_prioritizes_likely_feature_papers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = make_run_dir(
                root,
                papers=[
                    {
                        "paper_id": "irrelevant",
                        "title": "3D Point Cloud Attribute Compression",
                        "authors": [],
                        "year": 2024,
                    },
                    {
                        "paper_id": "relevant",
                        "title": "Propaganda Technique Taxonomies in News",
                        "authors": [],
                        "year": 2024,
                    },
                ],
                chunks=[
                    {
                        "chunk_id": "irrelevant:p1:c1",
                        "paper_id": "irrelevant",
                        "title": "3D Point Cloud Attribute Compression",
                        "year": 2024,
                        "pages": [1],
                        "text": "Point cloud feature extraction.",
                        "token_count": 4,
                    },
                    {
                        "chunk_id": "relevant:p1:c1",
                        "paper_id": "relevant",
                        "title": "Propaganda Technique Taxonomies in News",
                        "year": 2024,
                        "pages": [1],
                        "text": "Propaganda technique taxonomies include loaded language.",
                        "token_count": 7,
                    },
                ],
            )
            client = FakeClient([json.dumps({"features": []}), json.dumps({"features": []})])

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=root / "extract",
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertGreater(
                title_relevance_score("Propaganda Technique Taxonomies in News"),
                title_relevance_score("3D Point Cloud Attribute Compression"),
            )
            ranked = list(load_papers(run_dir).keys())
            self.assertEqual(ranked[0], "relevant")
            self.assertIn('"paper_id": "relevant"', client.calls[0]["messages"][1]["content"])
            self.assertEqual(trace["top_ranked_papers"][0]["paper_id"], "relevant")

    def test_title_ranking_skews_toward_media_persuasion_and_propaganda(self):
        propaganda_score = title_relevance_score(
            "Propaganda Techniques and Persuasive Rhetoric in News Media"
        )
        moral_score = title_relevance_score("Moral Foundations and Framing in Political Communication")

        self.assertGreater(propaganda_score, moral_score)
        self.assertGreater(
            title_relevance_score("Media Bias and Misinformation Persuasion Analysis"),
            title_relevance_score("Moral Framing and Sentiment in Politics"),
        )

    def test_run_extraction_repairs_malformed_json_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = make_run_dir(
                root,
                papers=[
                    {
                        "paper_id": "paper1",
                        "title": "Propaganda Technique Taxonomies",
                        "authors": [],
                        "year": 2024,
                    }
                ],
                chunks=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "paper_id": "paper1",
                        "title": "Propaganda Technique Taxonomies",
                        "year": 2024,
                        "pages": [1],
                        "text": "Propaganda taxonomies include loaded language.",
                        "token_count": 5,
                    }
                ],
            )
            malformed = """{
              "features": [
                {
                  "category": "persuasion",
                  "feature_name": "Loaded language",
                  "chunk_id": "paper1:p1:c1",
                  "source_quote": "called "loaded language" here"
                }
              ]
            }"""
            repaired = json.dumps(
                {
                    "features": [
                        {
                            "category": "persuasion",
                            "feature_name": "Loaded language",
                            "chunk_id": "paper1:p1:c1",
                            "source_quote": 'called "loaded language" here',
                        }
                    ]
                }
            )
            client = FakeClient([malformed, repaired])

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=root / "extract",
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(trace["counts"]["repaired_json_count"], 1)
            self.assertEqual(trace["outputs"]["feature_mentions"], 1)

    def test_outputs_are_refreshed_after_each_paper(self):
        class CheckingClient:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
                self.calls = []

            def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
                self.calls.append(messages)
                if len(self.calls) == 1:
                    return json.dumps(
                        {
                            "features": [
                                {
                                    "category": "persuasion",
                                    "feature_name": "Loaded language",
                                    "chunk_id": "paper1:p1:c1",
                                    "source_quote": "loaded language",
                                }
                            ]
                        }
                    )

                rows = [
                    json.loads(line)
                    for line in (self.output_dir / "feature_mentions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                if len(rows) != 1:
                    raise AssertionError("feature_mentions.jsonl was not refreshed after paper1")
                return json.dumps(
                    {
                        "features": [
                            {
                                "category": "moral_framing",
                                "feature_name": "Care/Harm",
                                "chunk_id": "paper2:p1:c1",
                                "source_quote": "Care/Harm",
                            }
                        ]
                    }
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = make_run_dir(
                root,
                papers=[
                    {
                        "paper_id": "paper1",
                        "title": "Propaganda Persuasion Rhetorical Technique Taxonomies A",
                        "authors": [],
                        "year": 2024,
                    },
                    {
                        "paper_id": "paper2",
                        "title": "Moral Framing Taxonomies B",
                        "authors": [],
                        "year": 2024,
                    },
                ],
                chunks=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "paper_id": "paper1",
                        "title": "Propaganda Persuasion Rhetorical Technique Taxonomies A",
                        "year": 2024,
                        "pages": [1],
                        "text": "Propaganda taxonomies include loaded language.",
                        "token_count": 5,
                    },
                    {
                        "chunk_id": "paper2:p1:c1",
                        "paper_id": "paper2",
                        "title": "Moral Framing Taxonomies B",
                        "year": 2024,
                        "pages": [1],
                        "text": "Moral framing taxonomies include Care/Harm.",
                        "token_count": 5,
                    },
                ],
            )
            output_dir = root / "extract"
            client = CheckingClient(output_dir)

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=output_dir,
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(trace["outputs"]["feature_mentions"], 2)

    def test_resume_skips_completed_papers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = make_run_dir(
                root,
                papers=[
                    {
                        "paper_id": "paper1",
                        "title": "Propaganda Taxonomies",
                        "authors": [],
                        "year": 2024,
                    }
                ],
                chunks=[
                    {
                        "chunk_id": "paper1:p1:c1",
                        "paper_id": "paper1",
                        "title": "Propaganda Taxonomies",
                        "year": 2024,
                        "pages": [1],
                        "text": "Propaganda technique taxonomies include loaded language.",
                        "token_count": 7,
                    }
                ],
            )
            output_dir = root / "extract"
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "features": [
                                {
                                    "category": "persuasion",
                                    "feature_name": "Loaded language",
                                    "chunk_id": "paper1:p1:c1",
                                    "source_quote": "loaded language",
                                }
                            ]
                        }
                    )
                ]
            )
            run_extraction(
                run_dir=run_dir,
                output_dir=output_dir,
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=True,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                client=first_client,
            )

            second_client = FakeClient([])
            trace = run_extraction(
                run_dir=run_dir,
                output_dir=output_dir,
                api_key=None,
                model="fake-model",
                base_url="https://example.test/v1",
                reasoning_effort=None,
                batch_chunks_count=1,
                paper_limit=None,
                force=False,
                dry_run=False,
                max_chunk_chars=1000,
                temperature=0.0,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(trace["counts"]["paper_skipped_count"], 1)
            self.assertEqual(trace["outputs"]["feature_mentions"], 1)

    def test_annotation_prompt_includes_goal_counts_and_evidence(self):
        features = [
            feature_row(
                "persuasion",
                "Loaded Language",
                definitions=["Emotionally charged wording used to influence judgment."],
                synonyms=["emotive language"],
                examples=["crooked elite"],
                paper_ids=["p1", "p2"],
                mention_count=7,
            )
        ]
        features[0]["source_feature_names"] = [
            "Loaded Language",
            "Emotive Language",
            "Affective Language",
        ]

        prompt = build_annotation_prompt(
            features=features,
            goal_text="Keep persuasion techniques distinct from moral framing.",
        )

        self.assertIn("GOAL / TAXONOMY CONTEXT", prompt)
        self.assertIn("Keep persuasion techniques distinct", prompt)
        self.assertIn("Loaded Language", prompt)
        self.assertIn("Emotionally charged wording", prompt)
        self.assertIn('"paper_count": 2', prompt)
        self.assertIn('"mention_count": 7', prompt)
        self.assertIn("Affective Language", prompt)
        self.assertIn("Generate zero or more synonyms", prompt)
        self.assertIn("Return [] when there are no truly close", prompt)
        self.assertIn("definitionally close and substitutable", prompt)
        self.assertIn("Do not include parent categories", prompt)
        self.assertIn('status="excluded"', prompt)
        self.assertIn("sentiment polarity", prompt)
        self.assertIn("valence", prompt)
        self.assertIn("framing", prompt)
        self.assertIn("specific emotions or affective states", prompt)
        self.assertIn("arousal", prompt)
        self.assertIn("confusion", prompt)
        self.assertIn("uncertainty", prompt)
        self.assertIn("subjectivity", prompt)
        self.assertIn("high-level moral-frame", prompt)
        self.assertIn("loyalty/betrayal", prompt)
        self.assertIn("loyalty/ingroup", prompt)
        self.assertIn("Moral Foundations Theory", prompt)
        self.assertIn("Treat the status field as your final judgment", prompt)

    def test_annotation_synonym_cleanup_keeps_only_close_alias_candidates(self):
        feature = feature_row(
            "persuasion",
            "Loaded Language",
            parents=["Manipulative Wording"],
            examples=["crooked elite"],
            titles=["Loaded Language Studies"],
        )

        synonyms = normalize_annotation_synonyms(
            [
                "Emotive language",
                "emotive language",
                "Loaded Language",
                "Persuasion",
                "Manipulative Wording",
                "crooked elite",
                "Loaded Language Studies",
                "A technique that uses emotionally charged wording to influence audiences.",
                "Affective language",
            ],
            feature,
        )

        self.assertEqual(synonyms, ["Emotive language", "Affective language"])

    def test_annotation_exclusion_reason_flags_broad_descriptors(self):
        sentiment_polarity = feature_row("sentiment_affect", "Sentiment Polarity")
        valence = feature_row("sentiment_affect", "Valence")
        arousal = feature_row("sentiment_affect", "Arousal")
        activation = feature_row("sentiment_affect", "Activation")
        anger = feature_row("sentiment_affect", "Anger")
        hope = feature_row("sentiment_affect", "Hope")
        confusion = feature_row("sentiment_affect", "Confusion")
        uncertainty = feature_row("sentiment_affect", "Uncertainty")
        framing = feature_row("persuasion", "Framing")
        loaded_language = feature_row("persuasion", "Loaded Language")
        care_harm = feature_row("moral_framing", "Care/Harm")
        loyalty_ingroup = feature_row("moral_framing", "Loyalty/Ingroup")
        moral_foundations_theory = feature_row("moral_framing", "Moral Foundations Theory")
        bundled_persuasion = feature_row("persuasion", "Whataboutism, Straw Men, Red Herring")
        bundled_emotions = feature_row("sentiment_affect", "Anger, disgust, fear, and intolerance")

        self.assertIn(
            "broad descriptor",
            annotation_exclusion_reason(
                sentiment_polarity,
                "Sentiment polarity is the positive, negative, or neutral orientation of text.",
                "Overall sentiment orientation.",
            ),
        )
        self.assertIn(
            "broad descriptor",
            annotation_exclusion_reason(
                valence,
                "Valence measures how positive or negative an affective expression is.",
                "Positive or negative affective dimension.",
            ),
        )
        self.assertIn(
            "descriptor",
            annotation_exclusion_reason(
                arousal,
                "Arousal measures the activation level or intensity of affective expression.",
                "Affective activation level.",
            ),
        )
        self.assertIn(
            "descriptor",
            annotation_exclusion_reason(
                activation,
                "Activation describes the general intensity level of affect.",
                "Affective intensity.",
            ),
        )
        self.assertEqual(
            annotation_exclusion_reason(
                anger,
                "Anger is an emotion of irritation, hostility, or outrage.",
                "Feeling anger.",
            ),
            "",
        )
        self.assertEqual(
            annotation_exclusion_reason(
                hope,
                "Hope is an affective state of expectation for a positive outcome.",
                "Feeling hopeful.",
            ),
            "",
        )
        self.assertEqual(
            annotation_exclusion_reason(
                confusion,
                "Confusion is an affective state of uncertainty and lack of clarity.",
                "Feeling confused or uncertain.",
            ),
            "",
        )
        self.assertEqual(
            annotation_exclusion_reason(
                uncertainty,
                "Uncertainty is an affective state of doubt, hesitation, or lack of clarity.",
                "Feeling uncertain.",
            ),
            "",
        )
        self.assertIn(
            "broad descriptor",
            annotation_exclusion_reason(
                framing,
                "Framing is the broad process of selecting and emphasizing aspects of an issue.",
                "General framing process.",
            ),
        )
        self.assertEqual(
            annotation_exclusion_reason(
                loaded_language,
                "Loaded language uses emotionally charged wording to influence judgment.",
                "Emotionally charged persuasive wording.",
            ),
            "",
        )
        self.assertEqual(
            annotation_exclusion_reason(
                care_harm,
                "Care/Harm is a core moral foundation centered on suffering, protection, compassion, or harm prevention.",
                "Moral concern for suffering or protection.",
            ),
            "",
        )
        self.assertEqual(
            annotation_exclusion_reason(
                loyalty_ingroup,
                "Loyalty/Ingroup is a core Moral Foundations Theory frame about group cohesion and betrayal.",
                "Moral concern for allegiance and betrayal.",
            ),
            "",
        )
        self.assertIn(
            "framework",
            annotation_exclusion_reason(
                moral_foundations_theory,
                "Moral Foundations Theory is a theoretical framework for classifying moral intuitions.",
                "A moral psychology theory.",
            ),
        )
        self.assertIn(
            "multiple distinct feature labels",
            annotation_exclusion_reason(
                bundled_persuasion,
                "A bundled row listing several separate propaganda techniques.",
                "Multiple persuasion techniques in one label.",
            ),
        )
        self.assertIn(
            "multiple distinct feature labels",
            annotation_exclusion_reason(
                bundled_emotions,
                "A bundled row listing several affective states.",
                "Multiple emotions in one label.",
            ),
        )

    def test_annotation_normalization_lets_agent_keep_borderline_features(self):
        confusion = normalize_annotation_decision(
            {
                "feature_key": "sentiment_affect::confusion",
                "status": "annotated",
                "formal_definition": "Confusion is an affective state of uncertainty and lack of clarity.",
                "short_definition": "Feeling confused or uncertain.",
                "synonyms": ["uncertainty"],
                "confidence": 0.85,
            },
            feature_row("sentiment_affect", "Confusion"),
            model="annotator",
            run_id="run",
            annotated_at="2026-06-05T00:00:00+00:00",
        )
        self.assertEqual(confusion["status"], "annotated")
        self.assertEqual(confusion["definition_basis"], "evidence_synthesized")
        self.assertEqual(confusion["synonyms"], ["uncertainty"])

        sentiment_polarity = normalize_annotation_decision(
            {
                "feature_key": "sentiment_affect::sentiment polarity",
                "status": "annotated",
                "formal_definition": "Sentiment polarity is the positive, negative, or neutral orientation of text.",
                "short_definition": "Overall sentiment orientation.",
                "synonyms": ["polarity"],
                "confidence": 0.9,
            },
            feature_row("sentiment_affect", "Sentiment Polarity"),
            model="annotator",
            run_id="run",
            annotated_at="2026-06-05T00:00:00+00:00",
        )
        self.assertEqual(sentiment_polarity["status"], "excluded")
        self.assertEqual(sentiment_polarity["definition_basis"], "excluded_descriptor")

        bundled = normalize_annotation_decision(
            {
                "feature_key": "persuasion::whataboutism straw men red herring",
                "status": "annotated",
                "formal_definition": "A bundled row listing several separate propaganda techniques.",
                "short_definition": "Multiple persuasion techniques.",
                "synonyms": [],
                "confidence": 0.9,
            },
            feature_row("persuasion", "Whataboutism, Straw Men, Red Herring"),
            model="annotator",
            run_id="run",
            annotated_at="2026-06-05T00:00:00+00:00",
        )
        self.assertEqual(bundled["status"], "excluded")
        self.assertEqual(bundled["definition_basis"], "excluded_descriptor")
        self.assertIn("multiple distinct feature labels", bundled["notes"])

    def test_annotation_normalization_adds_moral_frame_aliases(self):
        loyalty = normalize_annotation_decision(
            {
                "feature_key": "moral_framing::loyalty ingroup",
                "status": "annotated",
                "formal_definition": "Loyalty/Ingroup frames an issue around allegiance, group cohesion, and betrayal.",
                "short_definition": "Moral concern for allegiance and betrayal.",
                "synonyms": ["ingroup loyalty"],
                "confidence": 0.9,
            },
            feature_row("moral_framing", "Loyalty/Ingroup"),
            model="annotator",
            run_id="run",
            annotated_at="2026-06-05T00:00:00+00:00",
        )
        self.assertEqual(loyalty["status"], "annotated")
        self.assertEqual(
            loyalty["synonyms"],
            ["ingroup loyalty", "Loyalty/Betrayal", "Loyalty foundation"],
        )

    def test_parse_annotation_json_repairs_malformed_model_output(self):
        malformed = """{
          "annotations": [
            {
              "feature_key": "persuasion::loaded language",
              "formal_definition": "Uses "charged" wording",
              "status": "annotated"
            }
          ]
        }"""
        repaired = json.dumps(
            {
                "annotations": [
                    {
                        "feature_key": "persuasion::loaded language",
                        "formal_definition": 'Uses "charged" wording',
                        "status": "annotated",
                    }
                ]
            }
        )
        client = FakeClient([repaired])

        annotations, was_repaired = parse_annotation_json_with_repair(malformed, client=client)

        self.assertTrue(was_repaired)
        self.assertEqual(annotations[0]["feature_key"], "persuasion::loaded language")
        self.assertEqual(len(client.calls), 1)

    def test_run_annotation_writes_outputs_with_mock_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(
                input_file,
                [
                    feature_row(
                        "persuasion",
                        "Loaded Language",
                        definitions=["Emotionally charged wording."],
                        synonyms=["emotive language"],
                        examples=["crooked"],
                        paper_ids=["p1", "p2"],
                        mention_count=5,
                    ),
                    feature_row(
                        "moral_framing",
                        "Care/Harm",
                        definitions=["A moral foundation about suffering and protection."],
                        paper_ids=["p3"],
                        mention_count=2,
                    ),
                    feature_row(
                        "sentiment_affect",
                        "Sentiment Polarity",
                        definitions=["The positive, negative, or neutral orientation of text."],
                        paper_ids=["p4"],
                        mention_count=3,
                    ),
                ],
            )
            output_dir = root / "annotations"
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "status": "annotated",
                                    "formal_definition": "Loaded Language is wording that uses emotionally charged terms to influence audience judgment.",
                                    "short_definition": "Emotionally charged persuasive wording.",
                                    "synonyms": [
                                        "emotive language",
                                        "Loaded Language",
                                        "affective language",
                                        "persuasion",
                                    ],
                                    "examples": ["crooked"],
                                    "paper_count": 2,
                                    "mention_count": 5,
                                    "confidence": 0.9,
                                    "definition_basis": "evidence_synthesized",
                                    "notes": "",
                                },
                                {
                                    "feature_key": "moral_framing::care harm",
                                    "status": "annotated",
                                    "formal_definition": "Care/Harm frames an issue around suffering, protection, compassion, or harm prevention.",
                                    "short_definition": "Moral concern for suffering or protection.",
                                    "synonyms": ["care foundation"],
                                    "examples": ["protecting victims"],
                                    "paper_count": 1,
                                    "mention_count": 2,
                                    "confidence": 0.85,
                                    "definition_basis": "evidence_synthesized",
                                    "notes": "",
                                },
                                {
                                    "feature_key": "sentiment_affect::sentiment polarity",
                                    "status": "annotated",
                                    "formal_definition": "Sentiment polarity is the positive, negative, or neutral orientation of text.",
                                    "short_definition": "Overall positive, negative, or neutral sentiment orientation.",
                                    "synonyms": ["polarity", "valence"],
                                    "examples": ["positive review"],
                                    "paper_count": 1,
                                    "mention_count": 3,
                                    "confidence": 0.9,
                                    "definition_basis": "evidence_synthesized",
                                    "notes": "",
                                },
                            ]
                        }
                    )
                ]
            )

            trace = run_annotation(
                input_file=input_file,
                output_dir=output_dir,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(trace["outputs"]["annotated_features"], 2)
            self.assertEqual(trace["outputs"]["needs_review_annotations"], 0)
            self.assertEqual(trace["outputs"]["excluded_annotations"], 1)
            for name in [
                "annotated_features.json",
                "annotated_features.jsonl",
                "annotated_features.csv",
                "excluded_annotations.json",
                "excluded_annotations.jsonl",
                "excluded_annotations.csv",
                "excluded_annotations_by_category.md",
                "annotation_trace.json",
                "annotation_decisions.jsonl",
                "annotation_errors.jsonl",
                "completed_annotation_batches.jsonl",
            ]:
                self.assertTrue((output_dir / name).exists(), name)
            for name in [
                "annotation_status_counts.png",
                "annotation_excluded_by_category.png",
                "annotation_status_by_category.png",
            ]:
                self.assertTrue((output_dir / "figures" / name).exists(), name)

            rows = [
                json.loads(line)
                for line in (output_dir / "annotated_features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["definitions"], ["Emotionally charged wording."])
            self.assertEqual(rows[0]["annotation"]["paper_count"], 2)
            self.assertEqual(rows[0]["annotation"]["mention_count"], 5)
            self.assertIn("emotionally charged", rows[0]["annotation"]["formal_definition"].lower())
            self.assertEqual(
                rows[0]["annotation"]["synonyms"],
                ["emotive language", "affective language"],
            )
            self.assertEqual(rows[0]["synonyms"], ["emotive language"])

            with (output_dir / "annotated_features.csv").open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["annotation_paper_count"], "2")
            self.assertIn("Loaded Language", csv_rows[0]["annotation_formal_definition"])
            self.assertEqual(
                json.loads(csv_rows[0]["annotation_synonyms"]),
                ["emotive language", "affective language"],
            )
            excluded_rows = [
                json.loads(line)
                for line in (output_dir / "excluded_annotations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(excluded_rows), 1)
            self.assertEqual(excluded_rows[0]["feature_name"], "Sentiment Polarity")
            self.assertEqual(excluded_rows[0]["annotation"]["status"], "excluded")
            self.assertEqual(
                excluded_rows[0]["annotation"]["definition_basis"],
                "excluded_descriptor",
            )
            excluded_markdown = (output_dir / "excluded_annotations_by_category.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("## sentiment_affect", excluded_markdown)
            self.assertIn("| 1 | Sentiment Polarity | 3 | 1 |", excluded_markdown)

    def test_run_annotation_resume_skips_completed_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(
                input_file,
                [
                    feature_row("persuasion", "Loaded Language"),
                    feature_row("moral_framing", "Care/Harm"),
                ],
            )
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "status": "annotated",
                                    "formal_definition": "Loaded Language uses emotionally charged wording.",
                                    "short_definition": "Emotionally charged wording.",
                                    "confidence": 0.8,
                                },
                                {
                                    "feature_key": "moral_framing::care harm",
                                    "status": "needs_review",
                                    "formal_definition": "",
                                    "short_definition": "",
                                    "confidence": 0.0,
                                    "definition_basis": "needs_review",
                                    "notes": "Insufficient evidence.",
                                },
                            ]
                        }
                    )
                ]
            )
            run_annotation(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=first_client,
            )

            second_client = FakeClient([])
            trace = run_annotation(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=False,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 1)
            self.assertEqual(trace["outputs"]["annotated_features"], 2)
            self.assertEqual(trace["outputs"]["needs_review_annotations"], 1)
            completed_rows = [
                json.loads(line)
                for line in (root / "completed_annotation_batches.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(completed_rows[0]["prompt_version"], "lit_extract_annotation_v5")

    def test_run_annotation_regenerates_batches_completed_with_old_prompt_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(input_file, [feature_row("persuasion", "Loaded Language")])
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "status": "annotated",
                                    "formal_definition": "Loaded Language uses emotionally charged wording.",
                                    "short_definition": "Emotionally charged wording.",
                                    "synonyms": ["emotive language"],
                                    "confidence": 0.8,
                                }
                            ]
                        }
                    )
                ]
            )
            run_annotation(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=first_client,
            )
            completed_rows = [
                json.loads(line)
                for line in (root / "completed_annotation_batches.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            completed_rows[0]["prompt_version"] = "lit_extract_annotation_v1"
            write_jsonl(root / "completed_annotation_batches.jsonl", completed_rows)

            second_client = FakeClient(
                [
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "status": "annotated",
                                    "formal_definition": "Loaded Language uses affective wording.",
                                    "short_definition": "Affective persuasive wording.",
                                    "synonyms": ["affective language"],
                                    "confidence": 0.9,
                                }
                            ]
                        }
                    )
                ]
            )
            trace = run_annotation(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=False,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 1)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 0)
            rows = [
                json.loads(line)
                for line in (root / "annotated_features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["annotation"]["synonyms"], ["affective language"])

    def test_annotation_force_removes_only_annotation_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(input_file, [feature_row("persuasion", "Loaded Language")])
            preserved = root / "filtered_features.jsonl"
            preserved.write_text("do not touch", encoding="utf-8")
            (root / "annotation_decisions.jsonl").write_text("stale\n", encoding="utf-8")
            (root / "completed_annotation_batches.jsonl").write_text("stale\n", encoding="utf-8")
            (root / "excluded_annotations.jsonl").write_text("stale\n", encoding="utf-8")
            (root / "excluded_annotations_by_category.md").write_text("stale\n", encoding="utf-8")
            stale_figure = root / "figures" / "annotation_status_counts.png"
            stale_figure.parent.mkdir(parents=True)
            stale_figure.write_bytes(b"stale")
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "annotations": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "status": "annotated",
                                    "formal_definition": "Loaded Language uses emotionally charged wording.",
                                    "short_definition": "Emotionally charged wording.",
                                    "confidence": 0.8,
                                }
                            ]
                        }
                    )
                ]
            )

            run_annotation(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="annotator",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=client,
            )

            self.assertEqual(preserved.read_text(encoding="utf-8"), "do not touch")
            decisions_text = (root / "annotation_decisions.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("stale", decisions_text)
            self.assertIn("Loaded Language uses emotionally charged wording", decisions_text)
            self.assertNotIn(
                "stale",
                (root / "excluded_annotations.jsonl").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "stale",
                (root / "excluded_annotations_by_category.md").read_text(encoding="utf-8"),
            )
            self.assertNotEqual(stale_figure.read_bytes(), b"stale")

    def test_scoring_formula_uses_literature_buckets_and_weighted_total(self):
        self.assertEqual(
            [literature_support_score(value) for value in range(0, 17)],
            [
                0,
                0,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
            ],
        )
        self.assertEqual(total_feature_score(5, 5, 4), 4.7)
        self.assertEqual(total_feature_score(0, 3, 2), 1.5)

    def test_build_scored_features_recomputes_deterministic_scores(self):
        feature = feature_row(
            "persuasion",
            "Loaded Language",
            paper_ids=[f"p{idx}" for idx in range(1, 11)],
        )
        scored = build_scored_features(
            [feature],
            [
                {
                    "feature_key": "persuasion::loaded language",
                    "literature_support": 5,
                    "literature_support_rationale": "Old rubric.",
                    "annotation_reliability": 5,
                    "distinctiveness": 4,
                    "total": 4.7,
                    "confidence": 0.9,
                }
            ],
        )

        self.assertEqual(scored[0]["score"]["literature_support"], 4)
        self.assertEqual(scored[0]["score"]["total"], 4.3)
        self.assertEqual(
            scored[0]["score"]["literature_support_rationale"],
            "paper_count=10 maps to literature_support=4.",
        )

    def test_scoring_prompt_includes_rubric_annotation_and_merge_context(self):
        loaded = feature_row(
            "persuasion",
            "Loaded Language",
            definitions=["Emotionally charged wording."],
            synonyms=["emotive language"],
            examples=["crooked elite"],
            paper_ids=["p1", "p2"],
            mention_count=7,
        )
        loaded["annotation"] = {
            "status": "annotated",
            "formal_definition": "Loaded Language uses emotionally charged terms to influence judgment.",
            "short_definition": "Emotionally charged persuasive wording.",
            "synonyms": ["affective language"],
            "examples": ["crooked elite"],
        }
        loaded["embedding_merge"] = {
            "group_id": "merge_group_0001",
            "canonical_feature_key": "persuasion::loaded language",
            "canonical_feature_name": "Loaded Language",
            "member_count": 2,
            "member_feature_names": ["Loaded Language", "Emotive Language"],
            "similarity_summary": {"max": 0.98},
        }
        neighbor = feature_row(
            "persuasion",
            "Emotive Language",
            definitions=["Emotionally evaluative wording."],
            paper_ids=["p3"],
        )

        prompt = build_scoring_prompt(
            features=[loaded],
            all_features=[loaded, neighbor],
            goal_text="Prefer reliable media annotation features.",
        )

        self.assertIn("GOAL / TAXONOMY CONTEXT", prompt)
        self.assertIn("Prefer reliable media annotation features", prompt)
        self.assertIn("Annotation Reliability", prompt)
        self.assertIn("Distinctiveness", prompt)
        self.assertIn('"literature_support_score": 1', prompt)
        self.assertIn("Loaded Language uses emotionally charged", prompt)
        self.assertIn("merge_group_0001", prompt)
        self.assertIn("Emotive Language", prompt)
        self.assertIn("Do not penalize a canonical merged row", prompt)

    def test_parse_scoring_json_repairs_malformed_model_output(self):
        malformed = """{
          "scores": [
            {
              "feature_key": "persuasion::loaded language",
              "annotation_reliability": 5,
              "annotation_reliability_rationale": "Uses "charged" wording",
              "distinctiveness": 4
            }
          ]
        }"""
        repaired = json.dumps(
            {
                "scores": [
                    {
                        "feature_key": "persuasion::loaded language",
                        "annotation_reliability": 5,
                        "annotation_reliability_rationale": 'Uses "charged" wording',
                        "distinctiveness": 4,
                    }
                ]
            }
        )
        client = FakeClient([repaired])

        scores, was_repaired = parse_scoring_json_with_repair(malformed, client=client)

        self.assertTrue(was_repaired)
        self.assertEqual(scores[0]["feature_key"], "persuasion::loaded language")
        self.assertEqual(len(client.calls), 1)

    def test_run_scoring_writes_outputs_with_mock_llm_and_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            rows = [
                feature_row(
                    "persuasion",
                    "Loaded Language",
                    definitions=["Emotionally charged wording."],
                    paper_ids=[f"p{idx}" for idx in range(1, 11)],
                    mention_count=12,
                ),
                feature_row(
                    "moral_framing",
                    "Care/Harm",
                    definitions=["A moral frame about suffering and protection."],
                    paper_ids=["p11", "p12"],
                    mention_count=3,
                ),
                feature_row(
                    "sentiment_affect",
                    "Sentiment Polarity",
                    definitions=["The positive, negative, or neutral orientation of text."],
                    paper_ids=["p13"],
                    mention_count=1,
                ),
            ]
            rows[0]["annotation"] = {
                "status": "annotated",
                "formal_definition": "Loaded Language uses emotionally charged wording.",
                "short_definition": "Emotionally charged wording.",
            }
            rows[1]["annotation"] = {
                "status": "annotated",
                "formal_definition": "Care/Harm frames suffering and protection.",
                "short_definition": "Moral concern for suffering.",
            }
            rows[2]["annotation"] = {
                "status": "excluded",
                "formal_definition": "Sentiment polarity is an overall orientation.",
                "short_definition": "Overall sentiment orientation.",
            }
            write_jsonl(input_file, rows)
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "scores": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "annotation_reliability": 5,
                                    "annotation_reliability_rationale": "Clear lexical cue and examples.",
                                    "distinctiveness": 4,
                                    "distinctiveness_rationale": "Some overlap with emotive language.",
                                    "confidence": 0.9,
                                    "notes": "",
                                },
                                {
                                    "feature_key": "moral_framing::care harm",
                                    "annotation_reliability": 4,
                                    "annotation_reliability_rationale": "Canonical frame but context-sensitive.",
                                    "distinctiveness": 5,
                                    "distinctiveness_rationale": "Distinct moral foundation.",
                                    "confidence": 0.85,
                                    "notes": "",
                                },
                            ]
                        }
                    )
                ]
            )

            trace = run_scoring(
                input_file=input_file,
                output_dir=root / "scores",
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=client,
            )

            output_dir = root / "scores"
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(trace["outputs"]["scored_features"], 3)
            for name in [
                "scored_features.json",
                "scored_features.jsonl",
                "scored_features.csv",
                "scored_features_ranked.csv",
                "scoring_summary_by_category.json",
                "scoring_percentiles.json",
                "scoring_percentiles.csv",
                "scoring_trace.json",
                "scoring_decisions.jsonl",
                "scoring_errors.jsonl",
                "completed_scoring_batches.jsonl",
            ]:
                self.assertTrue((output_dir / name).exists(), name)
            for name in [
                "scoring_total_distribution.png",
                "scoring_criteria_distributions.png",
                "scoring_percentiles.png",
                "scoring_total_percentiles_by_category.png",
            ]:
                self.assertTrue((output_dir / "figures" / name).exists(), name)

            scored_rows = [
                json.loads(line)
                for line in (output_dir / "scored_features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(scored_rows[0]["score"]["literature_support"], 4)
            self.assertEqual(scored_rows[0]["score"]["total"], 4.3)
            self.assertEqual(scored_rows[2]["score"]["annotation_reliability"], 0)
            self.assertIn("No model score returned", scored_rows[2]["score"]["notes"])

            with (output_dir / "scored_features.csv").open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["score_literature_support"], "4")
            self.assertEqual(csv_rows[0]["score_total"], "4.3")
            self.assertIn("Clear lexical cue", csv_rows[0]["score_annotation_reliability_rationale"])
            percentile_payload = json.loads(
                (output_dir / "scoring_percentiles.json").read_text(encoding="utf-8")
            )
            self.assertIn(50, percentile_payload["percentiles"])
            self.assertTrue(
                any(
                    row["scope"] == "all" and row["metric"] == "total_score" and row["percentile"] == 50
                    for row in percentile_payload["rows"]
                )
            )

    def test_run_scoring_resume_skips_completed_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(
                input_file,
                [
                    feature_row("persuasion", "Loaded Language"),
                    feature_row("moral_framing", "Care/Harm"),
                ],
            )
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "scores": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "annotation_reliability": 5,
                                    "distinctiveness": 4,
                                },
                                {
                                    "feature_key": "moral_framing::care harm",
                                    "annotation_reliability": 4,
                                    "distinctiveness": 5,
                                },
                            ]
                        }
                    )
                ]
            )
            run_scoring(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=first_client,
            )

            second_client = FakeClient([])
            trace = run_scoring(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=False,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 1)
            completed_rows = [
                json.loads(line)
                for line in (root / "completed_scoring_batches.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(completed_rows[0]["prompt_version"], "lit_extract_scoring_v1")

    def test_run_scoring_regenerates_batches_completed_with_old_prompt_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.jsonl"
            write_jsonl(input_file, [feature_row("persuasion", "Loaded Language")])
            first_client = FakeClient(
                [
                    json.dumps(
                        {
                            "scores": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "annotation_reliability": 3,
                                    "distinctiveness": 3,
                                }
                            ]
                        }
                    )
                ]
            )
            run_scoring(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=first_client,
            )
            completed_rows = [
                json.loads(line)
                for line in (root / "completed_scoring_batches.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            completed_rows[0]["prompt_version"] = "lit_extract_scoring_v0"
            write_jsonl(root / "completed_scoring_batches.jsonl", completed_rows)

            second_client = FakeClient(
                [
                    json.dumps(
                        {
                            "scores": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "annotation_reliability": 5,
                                    "distinctiveness": 4,
                                }
                            ]
                        }
                    )
                ]
            )
            trace = run_scoring(
                input_file=input_file,
                output_dir=root,
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=False,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 1)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 0)
            scored_rows = [
                json.loads(line)
                for line in (root / "scored_features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(scored_rows[0]["score"]["annotation_reliability"], 5)

    def test_run_scoring_accepts_csv_inventory_with_jsonish_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "features.csv"
            with input_file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "feature_key",
                        "category",
                        "feature_name",
                        "normalized_feature_name",
                        "paper_count",
                        "mention_count",
                        "annotation",
                        "embedding_merge",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "feature_key": "persuasion::loaded language",
                        "category": "persuasion",
                        "feature_name": "Loaded Language",
                        "normalized_feature_name": "loaded language",
                        "paper_count": "3",
                        "mention_count": "5",
                        "annotation": json.dumps(
                            {
                                "status": "annotated",
                                "formal_definition": "Loaded Language uses emotionally charged wording.",
                                "short_definition": "Emotionally charged wording.",
                            }
                        ),
                        "embedding_merge": json.dumps(
                            {
                                "group_id": "merge_group_0001",
                                "member_count": 2,
                                "member_feature_names": ["Loaded Language", "Emotive Language"],
                            }
                        ),
                    }
                )
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "scores": [
                                {
                                    "feature_key": "persuasion::loaded language",
                                    "annotation_reliability": 5,
                                    "distinctiveness": 4,
                                    "confidence": 0.9,
                                }
                            ]
                        }
                    )
                ]
            )

            run_scoring(
                input_file=input_file,
                output_dir=root / "scores",
                api_key=None,
                model="scorer",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                batch_features=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=client,
            )

            prompt = client.calls[0]["messages"][1]["content"]
            self.assertIn("Loaded Language uses emotionally charged wording", prompt)
            self.assertIn("merge_group_0001", prompt)
            rows = [
                json.loads(line)
                for line in (root / "scores" / "scored_features.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertIsInstance(rows[0]["annotation"], dict)
            self.assertEqual(rows[0]["score"]["literature_support"], 2)

    def test_run_llm_merge_reviews_embedding_top_k_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "annotated_features.json"
            rows = [
                {
                    **feature_row("persuasion", "Loaded Language", paper_ids=["p1", "p2"], mention_count=5),
                    "annotation": {
                        "status": "annotated",
                        "short_definition": "Emotionally charged persuasive wording.",
                        "formal_definition": "A technique that uses emotionally loaded words.",
                    },
                },
                {
                    **feature_row("persuasion", "Loaded Languages", paper_ids=["p3"], mention_count=2),
                    "annotation": {
                        "status": "annotated",
                        "short_definition": "Emotionally charged persuasive wording.",
                        "formal_definition": "Use of loaded wording to persuade.",
                    },
                },
                {
                    **feature_row("persuasion", "Bandwagon", paper_ids=["p4"], mention_count=1),
                    "annotation": {
                        "status": "annotated",
                        "short_definition": "Appeal to popularity.",
                    },
                },
                {
                    **feature_row("moral_framing", "Care/Harm", paper_ids=["p5"], mention_count=1),
                    "annotation": {
                        "status": "annotated",
                        "short_definition": "A moral frame about suffering and protection.",
                    },
                },
            ]
            input_file.write_text(json.dumps({"features": rows}), encoding="utf-8")
            category_config = root / "merge_config.json"
            category_config.write_text(
                json.dumps(
                    {
                        "persuasion": {
                            "text_fields": ["feature_name", "annotation.short_definition"],
                        },
                        "moral_framing": {
                            "text_fields": ["feature_name", "annotation.short_definition"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "merge"
            embedding_client = FakeEmbeddingClient(
                [
                    [1.0, 0.0],
                    [0.999, 0.01],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ]
            )
            client = FakeClient(
                [
                    json.dumps(
                        {
                            "decisions": [
                                {
                                    "pair_id": "merge_pair_000001",
                                    "decision": "merge",
                                    "confidence": 0.96,
                                    "rationale": "Plural variant with same definition.",
                                },
                                {
                                    "pair_id": "merge_pair_000002",
                                    "decision": "separate",
                                    "confidence": 0.8,
                                    "rationale": "Different persuasive role.",
                                },
                            ]
                        }
                    )
                ]
            )

            trace = run_llm_merge(
                input_file=input_file,
                output_dir=output_dir,
                api_key=None,
                model="merge-model",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                category_config=category_config,
                top_k=1,
                candidate_min_similarity=None,
                batch_pairs=10,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                cache_path=root / "cache.jsonl",
                embedding_client=embedding_client,
                client=client,
            )

            self.assertEqual(len(client.calls), 1)
            self.assertIn("Annotation Short Definition", embedding_client.all_texts[0])
            self.assertEqual(trace["counts"]["candidate_pair_count"], 2)
            self.assertEqual(trace["outputs"]["accepted_pair_count"], 1)
            self.assertEqual(trace["outputs"]["merge_group_count"], 1)
            self.assertEqual(trace["outputs"]["llm_merged_features"], 3)
            self.assertEqual(trace["ignored_category_config_fields"], {})
            for name in [
                "llm_merge_candidates.jsonl",
                "llm_merge_candidates.csv",
                "llm_merge_decisions.jsonl",
                "llm_merge_decisions.csv",
                "llm_merged_features.json",
                "llm_merged_features.jsonl",
                "llm_merged_features.csv",
                "llm_merge_qualitative_examples.md",
                "llm_merge_retention_by_category.png",
                "llm_merge_errors.jsonl",
                "llm_merge_trace.json",
            ]:
                self.assertTrue((output_dir / name).exists(), name)
            self.assertEqual(
                (output_dir / "llm_merge_retention_by_category.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            qualitative_markdown = (output_dir / "llm_merge_qualitative_examples.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("# LLM Top-K Merge Qualitative Examples", qualitative_markdown)
            self.assertIn("Most Similar Candidate Pairs", qualitative_markdown)
            self.assertIn("Accepted Merge Examples", qualitative_markdown)
            self.assertIn("Loaded Language", qualitative_markdown)
            self.assertIn("Loaded Languages", qualitative_markdown)
            self.assertIn("merge", qualitative_markdown)
            self.assertIn("separate", qualitative_markdown)
            self.assertIn("0.9999", qualitative_markdown)
            merged_payload = json.loads((output_dir / "llm_merged_features.json").read_text(encoding="utf-8"))
            merged_names = [row["feature_name"] for row in merged_payload["features"]]
            self.assertEqual(merged_names, ["Loaded Language", "Bandwagon", "Care/Harm"])
            loaded = merged_payload["features"][0]
            self.assertEqual(loaded["paper_count"], 3)
            self.assertEqual(loaded["mention_count"], 7)
            self.assertEqual(loaded["embedding_merge"]["mode"], "llm_top_k")
            self.assertEqual(loaded["embedding_merge"]["accepted_pair_ids"], ["merge_pair_000001"])

            second_client = FakeClient([])
            second_trace = run_llm_merge(
                input_file=input_file,
                output_dir=output_dir,
                api_key=None,
                model="merge-model",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                category_config=category_config,
                top_k=1,
                batch_pairs=10,
                force=False,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                cache_path=root / "cache.jsonl",
                embedding_client=FakeEmbeddingClient([]),
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(second_trace["counts"]["skipped_batch_count"], 1)
            self.assertEqual(second_trace["outputs"]["llm_merged_features"], 3)

    def test_run_llm_merge_dry_run_does_not_embed_or_call_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "annotated_features.jsonl"
            write_jsonl(input_file, [feature_row("persuasion", "Loaded Language")])
            trace = run_llm_merge(
                input_file=input_file,
                output_dir=root / "merge",
                api_key=None,
                model="merge-model",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                top_k=10,
                batch_pairs=10,
                force=True,
                dry_run=True,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                embedding_client=FakeEmbeddingClient([[1.0, 0.0]]),
                client=FakeClient([]),
            )

            self.assertTrue(trace["dry_run"])
            self.assertEqual(trace["counts"]["feature_count"], 1)
            self.assertFalse((root / "merge" / "llm_merge_candidates.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
