import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lit_extract import (
    aggregate_features,
    aggregate_paper_features,
    build_extraction_prompt,
    build_gemini_generate_content_payload,
    build_curation_outputs,
    build_pdf_extraction_messages,
    canonical_category,
    load_papers,
    normalize_feature_item,
    normalize_feature_name,
    parse_curation_json_with_repair,
    parse_features_json,
    parse_features_json_with_repair,
    parse_args,
    run_curation,
    run_extraction,
    run_final_curation,
    select_top_curated_features_by_category,
    title_relevance_score,
    write_artifacts,
)


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


class LitExtractTests(unittest.TestCase):
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

    def test_curation_aggregation_merges_and_rejects(self):
        features = [
            feature_row(
                "moral_framing",
                "Authority",
                parents=["Moral Foundations"],
                definitions=["Respect for hierarchy and legitimate authority."],
                synonyms=["Authority/Subversion"],
                paper_ids=["p1", "p2"],
                titles=["Paper 1", "Paper 2"],
                mention_count=10,
                confidence=0.9,
            ),
            feature_row(
                "moral_framing",
                "Authority/Subversion",
                parents=["Moral Foundations Theory"],
                definitions=["A moral foundation concerning authority and subversion."],
                paper_ids=["p3"],
                titles=["Paper 3"],
                mention_count=4,
                confidence=0.8,
            ),
            feature_row(
                "moral_framing",
                "Authority Foundation",
                parents=["Binding foundations"],
                examples=["refusing to stand for a judge"],
                paper_ids=["p2", "p4"],
                titles=["Paper 2", "Paper 4"],
                mention_count=3,
                confidence=0.7,
            ),
            feature_row(
                "moral_framing",
                "Attestation",
                parents=["Moral Selfhood"],
                definitions=["An epistemic value linked to selfhood."],
                paper_ids=["p5"],
                titles=["Paper 5"],
                mention_count=1,
            ),
        ]
        decisions = [
            {
                "feature_key": "moral_framing::authority",
                "category": "moral_framing",
                "original_feature_name": "Authority",
                "original_normalized_feature_name": "authority",
                "action": "keep_as_canonical",
                "canonical_feature_key": "moral_framing::authority",
                "canonical_feature_name": "Authority",
                "canonical_normalized_feature_name": "authority",
                "rejection_reason": "",
                "curation_confidence": 0.95,
                "curation_notes": "Canonical MFT label.",
                "paper_count": 2,
                "mention_count": 10,
                "model": "curator",
                "run_id": "curate-run",
                "curated_at": "2026-05-08T00:00:00+00:00",
            },
            {
                "feature_key": "moral_framing::authority subversion",
                "category": "moral_framing",
                "original_feature_name": "Authority/Subversion",
                "original_normalized_feature_name": "authority subversion",
                "action": "merge_into_canonical",
                "canonical_feature_key": "moral_framing::authority",
                "canonical_feature_name": "Authority",
                "canonical_normalized_feature_name": "authority",
                "rejection_reason": "",
                "curation_confidence": 0.9,
                "curation_notes": "Alias pair.",
                "paper_count": 1,
                "mention_count": 4,
                "model": "curator",
                "run_id": "curate-run",
                "curated_at": "2026-05-08T00:00:00+00:00",
            },
            {
                "feature_key": "moral_framing::authority foundation",
                "category": "moral_framing",
                "original_feature_name": "Authority Foundation",
                "original_normalized_feature_name": "authority foundation",
                "action": "merge_into_canonical",
                "canonical_feature_key": "moral_framing::authority",
                "canonical_feature_name": "Authority",
                "canonical_normalized_feature_name": "authority",
                "rejection_reason": "",
                "curation_confidence": 0.85,
                "curation_notes": "Near synonym.",
                "paper_count": 2,
                "mention_count": 3,
                "model": "curator",
                "run_id": "curate-run",
                "curated_at": "2026-05-08T00:00:00+00:00",
            },
            {
                "feature_key": "moral_framing::attestation",
                "category": "moral_framing",
                "original_feature_name": "Attestation",
                "original_normalized_feature_name": "attestation",
                "action": "reject",
                "canonical_feature_key": "",
                "canonical_feature_name": "",
                "canonical_normalized_feature_name": "",
                "rejection_reason": "Too far from the target taxonomy.",
                "curation_confidence": 0.8,
                "curation_notes": "Off-goal.",
                "paper_count": 1,
                "mention_count": 1,
                "model": "curator",
                "run_id": "curate-run",
                "curated_at": "2026-05-08T00:00:00+00:00",
            },
        ]

        curated, members, rejected = build_curation_outputs(features, decisions)

        self.assertEqual(len(curated), 1)
        self.assertEqual(curated[0]["feature_name"], "Authority")
        self.assertEqual(curated[0]["mention_count"], 17)
        self.assertEqual(curated[0]["paper_count"], 4)
        self.assertIn("Authority Foundation", curated[0]["source_feature_names"])
        self.assertIn("Binding foundations", curated[0]["parent_categories"])
        self.assertEqual(len(members), 4)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["feature_name"], "Attestation")
        self.assertEqual(rejected[0]["rejection_reason"], "Too far from the target taxonomy.")

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
                merge_style="moderate",
                force=True,
                dry_run=False,
                temperature=0.0,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(trace["outputs"]["curated_features"], 1)
            self.assertEqual(trace["outputs"]["final_curated_features_top100_by_category"], 1)
            self.assertEqual(trace["outputs"]["curated_feature_members"], 2)
            self.assertEqual(trace["outputs"]["rejected_features"], 0)
            for name in [
                "curated_features.jsonl",
                "curated_features.csv",
                "final_curated_features_top100_by_category.jsonl",
                "final_curated_features_top100_by_category.csv",
                "curated_feature_members.jsonl",
                "curated_feature_members.csv",
                "rejected_features.jsonl",
                "rejected_features.csv",
                "curation_trace.json",
            ]:
                self.assertTrue((input_dir / name).exists(), name)

    def test_final_curated_export_keeps_top_100_per_category(self):
        curated = [
            {
                "category": "persuasion",
                "feature_name": f"Persuasion {idx:03d}",
                "paper_count": idx,
                "mention_count": idx,
            }
            for idx in range(1, 121)
        ]
        curated.extend(
            [
                {
                    "category": "moral_framing",
                    "feature_name": "Moral top",
                    "paper_count": 5,
                    "mention_count": 6,
                },
                {
                    "category": "moral_framing",
                    "feature_name": "Moral tie winner",
                    "paper_count": 4,
                    "mention_count": 10,
                },
                {
                    "category": "moral_framing",
                    "feature_name": "Moral tie loser",
                    "paper_count": 4,
                    "mention_count": 2,
                },
            ]
        )

        final_rows = select_top_curated_features_by_category(curated)
        persuasion = [row for row in final_rows if row["category"] == "persuasion"]
        moral = [row for row in final_rows if row["category"] == "moral_framing"]

        self.assertEqual(len(persuasion), 100)
        self.assertEqual(persuasion[0]["paper_count"], 120)
        self.assertEqual(persuasion[-1]["paper_count"], 21)
        self.assertEqual([row["feature_name"] for row in moral], [
            "Moral top",
            "Moral tie winner",
            "Moral tie loser",
        ])

    def test_run_final_curation_writes_llm_top_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "extract"
            input_dir.mkdir()
            write_jsonl(
                input_dir / "curated_features.jsonl",
                [
                    feature_row(
                        "persuasion",
                        "Loaded Language",
                        paper_ids=["p1", "p2", "p3"],
                        mention_count=3,
                    ),
                    feature_row(
                        "persuasion",
                        "Loaded Languages",
                        paper_ids=["p4"],
                        mention_count=1,
                    ),
                    feature_row(
                        "moral_framing",
                        "Care Harm",
                        paper_ids=["p5", "p6"],
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
                                    "feature_key": "persuasion::loaded language",
                                    "action": "keep_as_canonical",
                                    "canonical_feature_name": "Loaded Language",
                                    "canonical_normalized_feature_name": "loaded language",
                                    "curation_confidence": 0.95,
                                },
                                {
                                    "feature_key": "persuasion::loaded languages",
                                    "action": "merge_into_canonical",
                                    "canonical_feature_name": "Loaded Language",
                                    "canonical_normalized_feature_name": "loaded language",
                                    "curation_confidence": 0.9,
                                },
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "decisions": [
                                {
                                    "feature_key": "moral_framing::care harm",
                                    "action": "keep_as_canonical",
                                    "canonical_feature_name": "Care/Harm",
                                    "canonical_normalized_feature_name": "care harm",
                                    "curation_confidence": 0.9,
                                }
                            ]
                        }
                    ),
                ]
            )

            trace = run_final_curation(
                input_dir=input_dir,
                output_dir=input_dir,
                api_key=None,
                model="final-model",
                base_url="https://example.test/v1",
                api_provider="openai",
                reasoning_effort=None,
                merge_style="moderate",
                target_per_category=1,
                candidate_pool_per_category=0,
                force=True,
                dry_run=False,
                temperature=0.0,
                max_tokens=1000,
                show_progress=False,
                client=client,
            )

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(trace["outputs"]["final_curated_features_llm_top100_by_category"], 2)
            self.assertEqual(trace["outputs"]["final_curated_feature_members_llm"], 3)
            for name in [
                "final_curated_features_llm_top100_by_category.jsonl",
                "final_curated_features_llm_top100_by_category.csv",
                "final_curated_feature_members_llm.jsonl",
                "final_rejected_features_llm.jsonl",
                "final_curation_trace.json",
            ]:
                self.assertTrue((input_dir / name).exists(), name)

            final_rows = [
                json.loads(line)
                for line in (input_dir / "final_curated_features_llm_top100_by_category.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(final_rows[0]["category"], "persuasion")
            self.assertEqual(final_rows[0]["feature_name"], "Loaded Language")
            self.assertEqual(final_rows[0]["paper_count"], 4)

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
                merge_style="moderate",
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
                merge_style="moderate",
                force=False,
                dry_run=False,
                temperature=0.0,
                show_progress=False,
                client=second_client,
            )

            self.assertEqual(len(second_client.calls), 0)
            self.assertEqual(trace["counts"]["skipped_batch_count"], 1)
            self.assertEqual(trace["outputs"]["curated_features"], 1)

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
            self.assertTrue(
                content[0]["file"]["file_data"].startswith("data:application/pdf;base64,")
            )
            self.assertIn("Return at most 5 features from this PDF", content[1]["text"])
            self.assertEqual(trace["config"]["input_mode"], "pdf")
            self.assertEqual(trace["counts"]["pdf_count"], 1)
            self.assertEqual(trace["counts"]["chunk_count"], 0)
            self.assertEqual(trace["outputs"]["feature_mentions"], 1)

            rows = [
                json.loads(line)
                for line in (output_dir / "feature_mentions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
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
        moral_score = title_relevance_score(
            "Moral Foundations and Framing in Political Communication"
        )

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
                    for line in (self.output_dir / "feature_mentions.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
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

    def test_config_file_supports_gemini_api_key_env(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "extract_gemini.json"
            config_path.write_text(
                json.dumps(
                    {
                        "command": "run",
                        "run": "review_runs/media_attributes_v2",
                        "output_dir": "review_runs/media_attributes_v2/feature_extract",
                        "goal": "goal.md",
                        "api_key_env": "GEMINI_API_KEY",
                        "model": "gemini-3.1-flash-lite",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "api_provider": "gemini",
                        "reasoning_effort": "low",
                        "input_mode": "pdf",
                        "batch_chunks": 4,
                        "max_features_per_chunk": 12,
                        "max_features_per_pdf": 30,
                        "temperature": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}):
                args = parse_args(["--config", str(config_path)])

            self.assertEqual(args.command, "run")
            self.assertEqual(args.api_key, "test-secret")
            self.assertEqual(args.model, "gemini-3.1-flash-lite")
            self.assertEqual(args.api_provider, "gemini")
            self.assertEqual(args.reasoning_effort, "low")
            self.assertEqual(args.goal, "goal.md")
            self.assertEqual(args.input_mode, "pdf")
            self.assertEqual(args.max_features_per_chunk, 12)
            self.assertEqual(args.max_features_per_pdf, 30)

    def test_config_file_supports_curate_command(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "curate_gemini.json"
            config_path.write_text(
                json.dumps(
                    {
                        "command": "curate",
                        "input_dir": "review_runs/media_attributes_v2/feature_extract",
                        "output_dir": "review_runs/media_attributes_v2/feature_extract",
                        "goal": "goal.md",
                        "api_key_env": "GEMINI_API_KEY",
                        "model": "gemini-3-pro-preview",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "reasoning_effort": "high",
                        "batch_features": 40,
                        "merge_style": "moderate",
                        "temperature": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}):
                args = parse_args(["--config", str(config_path)])

            self.assertEqual(args.command, "curate")
            self.assertEqual(args.api_key, "test-secret")
            self.assertEqual(args.model, "gemini-3-pro-preview")
            self.assertEqual(args.batch_features, 40)
            self.assertEqual(args.merge_style, "moderate")

    def test_config_file_supports_finalize_command(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "finalize_gemini.json"
            config_path.write_text(
                json.dumps(
                    {
                        "command": "finalize",
                        "input_dir": "review_runs/media_attributes_v3/feature_extract_pdf",
                        "output_dir": "review_runs/media_attributes_v3/feature_extract_pdf",
                        "goal": "goal.md",
                        "api_key_env": "GEMINI_API_KEY",
                        "model": "gemini-3-pro-preview",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta",
                        "api_provider": "gemini",
                        "reasoning_effort": "high",
                        "target_per_category": 100,
                        "candidate_pool_per_category": 0,
                        "merge_style": "moderate",
                        "temperature": 0.0,
                        "max_tokens": 30000,
                        "timeout": 600,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}):
                args = parse_args(["--config", str(config_path)])

            self.assertEqual(args.command, "finalize")
            self.assertEqual(args.api_key, "test-secret")
            self.assertEqual(args.model, "gemini-3-pro-preview")
            self.assertEqual(args.api_provider, "gemini")
            self.assertEqual(args.target_per_category, 100)
            self.assertEqual(args.candidate_pool_per_category, 0)
            self.assertEqual(args.max_tokens, 30000)
            self.assertEqual(args.timeout, 600.0)


if __name__ == "__main__":
    unittest.main()
