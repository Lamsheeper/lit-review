import csv
import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from feature_embedding_tsne import FeatureItem

from embedding_merge import (
    SimilarityPair,
    build_merge_groups,
    build_parser,
    choose_canonical_index,
    find_similarity_pairs,
    main,
    run_embedding_merge,
    run_embedding_merge_sweep,
    threshold_range,
)


class FakeEmbeddingClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = list(embeddings)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(texts) > len(self.embeddings):
            raise AssertionError("FakeEmbeddingClient ran out of embeddings.")
        output = self.embeddings[: len(texts)]
        self.embeddings = self.embeddings[len(texts) :]
        return output

    @property
    def all_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def make_item(
    label: str,
    *,
    category: str = "category",
    row_index: int = 1,
    paper_count: int = 0,
    mention_count: int = 0,
) -> FeatureItem:
    feature_key = f"{category}::{label.lower().replace(' ', '_')}"
    return FeatureItem(
        row_index=row_index,
        feature_key=feature_key,
        label=label,
        category=category,
        paper_count=paper_count,
        mention_count=mention_count,
        embedding_text=f"Feature Name: {label}",
        raw_row={
            "feature_key": feature_key,
            "feature_name": label,
            "category": category,
            "paper_count": paper_count,
            "mention_count": mention_count,
        },
    )


class EmbeddingMergeTests(unittest.TestCase):
    def test_threshold_range_is_inclusive_and_defaults_to_centisteps(self):
        self.assertEqual(threshold_range(0.91, 0.93), [0.91, 0.92, 0.93])
        self.assertEqual(threshold_range(0.91, 0.91), [0.91])
        self.assertEqual(threshold_range(0.91, 0.94, 0.02), [0.91, 0.93, 0.94])

    def test_threshold_range_rejects_invalid_sweep_args(self):
        with self.assertRaisesRegex(ValueError, "start"):
            threshold_range(0.95, 0.90)
        with self.assertRaisesRegex(ValueError, "step"):
            threshold_range(0.90, 0.95, 0.0)

    def test_selected_text_fields_use_nested_annotation_short_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "annotated_features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_name": "Loaded Language",
                                "category": "persuasion",
                                "annotation": {
                                    "short_definition": "Emotionally charged persuasive wording.",
                                    "formal_definition": "This should not be embedded for this test.",
                                },
                            },
                            {
                                "feature_name": "Care/Harm",
                                "category": "moral_framing",
                                "annotation": {
                                    "short_definition": "Concern with suffering and protection.",
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            client = FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])

            run_embedding_merge(
                features_path=features_path,
                threshold=0.95,
                text_fields=["feature_name", "annotation.short_definition"],
                scope="all",
                output_dir=root / "out",
                prefix="nested",
                cache_path=root / "cache.jsonl",
                client=client,
            )

        joined_texts = "\n".join(client.all_texts)
        self.assertIn("Annotation Short Definition: Emotionally charged persuasive wording.", joined_texts)
        self.assertIn("Annotation Short Definition: Concern with suffering and protection.", joined_texts)
        self.assertNotIn("This should not be embedded for this test.", joined_texts)

    def test_selected_text_fields_fall_back_to_flattened_annotation_csv_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "annotated_features.csv"
            with features_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["feature_name", "category", "annotation_short_definition"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "feature_name": "Visual Framing",
                        "category": "media_attributes",
                        "annotation_short_definition": "Image composition that influences interpretation.",
                    }
                )
                writer.writerow(
                    {
                        "feature_name": "Source Credibility",
                        "category": "media_attributes",
                        "annotation_short_definition": "Signals that affect perceived trustworthiness.",
                    }
                )
            client = FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])

            run_embedding_merge(
                features_path=features_path,
                threshold=0.95,
                text_fields=["feature_name", "annotation.short_definition"],
                scope="all",
                output_dir=root / "out",
                prefix="flat",
                cache_path=root / "cache.jsonl",
                client=client,
            )

        joined_texts = "\n".join(client.all_texts)
        self.assertIn("Annotation Short Definition: Image composition that influences interpretation.", joined_texts)
        self.assertIn("Annotation Short Definition: Signals that affect perceived trustworthiness.", joined_texts)

    def test_find_similarity_pairs_defaults_to_cross_category_merging(self):
        items = [
            make_item("Tone", category="style", row_index=1),
            make_item("Tone of Voice", category="audio", row_index=2),
        ]

        pairs = find_similarity_pairs(
            items,
            [[1.0, 0.0], [0.999, 0.01]],
            threshold=0.99,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0].left_index, pairs[0].right_index), (0, 1))

    def test_category_scope_filters_cross_category_pairs(self):
        items = [
            make_item("Tone", category="style", row_index=1),
            make_item("Tone of Voice", category="audio", row_index=2),
        ]

        pairs = find_similarity_pairs(
            items,
            [[1.0, 0.0], [0.999, 0.01]],
            threshold=0.99,
            scope="category",
        )

        self.assertEqual(pairs, [])

    def test_connected_components_group_transitive_pairs(self):
        items = [
            make_item("A", row_index=1),
            make_item("B", row_index=2),
            make_item("C", row_index=3),
        ]

        groups = build_merge_groups(
            items,
            [
                SimilarityPair(0, 1, 0.96),
                SimilarityPair(1, 2, 0.95),
            ],
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].member_indices, [0, 1, 2])
        self.assertEqual([(pair.left_index, pair.right_index) for pair in groups[0].pairs], [(0, 1), (1, 2)])

    def test_canonical_representative_prefers_support_then_shorter_name_then_row_order(self):
        items = [
            make_item("Longer Name", row_index=1, paper_count=4, mention_count=7),
            make_item("Short", row_index=2, paper_count=4, mention_count=7),
            make_item("Small", row_index=3, paper_count=3, mention_count=100),
        ]

        canonical = choose_canonical_index(items, [0, 1, 2])

        self.assertEqual(canonical, 1)

    def test_run_embedding_merge_writes_outputs_and_merged_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_key": "style::tone",
                                "feature_name": "Tone",
                                "category": "style",
                                "paper_count": 2,
                                "mention_count": 3,
                                "definitions": ["Vocal or written attitude."],
                                "synonyms": ["style tone"],
                                "examples": ["warm tone"],
                                "source_paper_ids": ["paper-1", "paper-2"],
                                "source_titles": ["Paper One"],
                                "annotation": {
                                    "short_definition": "The attitude conveyed by wording.",
                                    "synonyms": ["tonality"],
                                    "examples": ["sarcastic tone"],
                                },
                            },
                            {
                                "feature_key": "audio::tone_of_voice",
                                "feature_name": "Tone of Voice",
                                "category": "audio",
                                "paper_count": 10,
                                "mention_count": 5,
                                "definitions": ["How speech sounds emotionally."],
                                "synonyms": ["voice tone"],
                                "examples": ["calm voice"],
                                "source_paper_ids": ["paper-2", "paper-3"],
                                "source_titles": ["Paper Two"],
                                "annotation": {
                                    "short_definition": "The emotional quality of spoken delivery.",
                                    "synonyms": ["prosody"],
                                    "examples": ["excited delivery"],
                                },
                            },
                            {
                                "feature_key": "style::topic",
                                "feature_name": "Topic",
                                "category": "style",
                                "paper_count": 1,
                                "mention_count": 1,
                                "source_paper_ids": ["paper-4"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "merge_outputs"
            client = FakeEmbeddingClient([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]])

            trace = run_embedding_merge(
                features_path=features_path,
                threshold=0.99,
                text_fields=["feature_name", "annotation.short_definition"],
                scope="all",
                output_dir=output_dir,
                prefix="features",
                cache_path=root / "cache.jsonl",
                client=client,
            )

            expected_paths = {
                "features_embedding_merge_pairs.jsonl",
                "features_embedding_merge_pairs.csv",
                "features_embedding_merge_groups.json",
                "features_embedding_merge_groups.csv",
                "features_embedding_merge_qualitative.json",
                "features_embedding_merge_qualitative.csv",
                "features_embedding_merge_qualitative.md",
                "features_embedding_merged_features.json",
                "features_embedding_merged_features.jsonl",
                "features_embedding_merged_features.csv",
                "features_embedding_similarity_distribution.png",
                "features_embedding_similarity_boxplot.png",
                "features_embedding_similarity_percentiles.png",
                "features_embedding_similarity_distribution_by_category.png",
                "features_embedding_similarity_boxplot_by_category.png",
                "features_embedding_similarity_percentiles_by_category.png",
                "features_embedding_merge_trace.json",
            }
            actual_paths = {path.name for path in output_dir.iterdir()}
            self.assertTrue(expected_paths.issubset(actual_paths))

            merged_payload = json.loads((output_dir / "features_embedding_merged_features.json").read_text())
            qualitative_payload = json.loads((output_dir / "features_embedding_merge_qualitative.json").read_text())
            qualitative_markdown = (output_dir / "features_embedding_merge_qualitative.md").read_text()
            pair_lines = (output_dir / "features_embedding_merge_pairs.jsonl").read_text().strip().splitlines()
            merged_lines = (output_dir / "features_embedding_merged_features.jsonl").read_text().strip().splitlines()
            with (output_dir / "features_embedding_merge_qualitative.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                qualitative_csv_rows = list(csv.DictReader(handle))
            similarity_distribution_png_header = (
                output_dir / "features_embedding_similarity_distribution.png"
            ).read_bytes()[:8]
            similarity_boxplot_png_header = (
                output_dir / "features_embedding_similarity_boxplot.png"
            ).read_bytes()[:8]
            similarity_percentiles_png_header = (
                output_dir / "features_embedding_similarity_percentiles.png"
            ).read_bytes()[:8]
            similarity_distribution_by_category_png_header = (
                output_dir / "features_embedding_similarity_distribution_by_category.png"
            ).read_bytes()[:8]
            similarity_boxplot_by_category_png_header = (
                output_dir / "features_embedding_similarity_boxplot_by_category.png"
            ).read_bytes()[:8]
            similarity_percentiles_by_category_png_header = (
                output_dir / "features_embedding_similarity_percentiles_by_category.png"
            ).read_bytes()[:8]

        self.assertEqual(trace["counts"]["input_features"], 3)
        self.assertEqual(trace["counts"]["similarity_pairs"], 1)
        self.assertEqual(trace["counts"]["merge_groups"], 1)
        self.assertEqual(trace["counts"]["merged_features"], 2)
        self.assertEqual(trace["counts"]["qualitative_rows"], 1)
        self.assertEqual(trace["tsne"]["status"], "skipped")
        self.assertEqual(trace["similarity_distribution"]["pair_count"], 3)
        self.assertEqual(trace["similarity_distribution"]["bin_count"], 80)
        self.assertEqual(trace["similarity_distributions_by_category"]["style"]["pair_count"], 1)
        self.assertEqual(similarity_distribution_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_boxplot_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_percentiles_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_distribution_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_boxplot_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_percentiles_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(pair_lines), 1)
        self.assertEqual(len(merged_lines), 2)
        self.assertEqual(merged_payload["merged_feature_count"], 2)
        self.assertEqual(qualitative_payload["top_n_per_category"], 10)
        self.assertEqual(len(qualitative_payload["results"]), 1)
        self.assertEqual(len(qualitative_csv_rows), 1)
        self.assertEqual(qualitative_payload["results"][0]["merged_feature_name"], "Tone")
        self.assertEqual(qualitative_payload["results"][0]["merged_into_feature_name"], "Tone of Voice")
        self.assertEqual(qualitative_payload["results"][0]["category"], "style")
        self.assertIn("| style | 1 | Tone | Tone of Voice | audio |", qualitative_markdown)

        merged_features = merged_payload["features"]
        merged_group = merged_features[0]
        singleton = merged_features[1]
        self.assertEqual(merged_group["feature_name"], "Tone of Voice")
        self.assertEqual(merged_group["paper_count"], 3)
        self.assertEqual(merged_group["mention_count"], 8)
        self.assertCountEqual(merged_group["source_paper_ids"], ["paper-1", "paper-2", "paper-3"])
        self.assertCountEqual(merged_group["annotation"]["synonyms"], ["tonality", "prosody"])
        self.assertCountEqual(merged_group["annotation"]["examples"], ["sarcastic tone", "excited delivery"])
        self.assertEqual(merged_group["embedding_merge"]["member_feature_keys"], ["audio::tone_of_voice", "style::tone"])
        self.assertEqual(merged_group["embedding_merge"]["member_categories"], ["audio", "style"])
        self.assertEqual(singleton["feature_name"], "Topic")
        self.assertNotIn("embedding_merge", singleton)

    def test_run_embedding_merge_generates_tsne_html_for_real_merged_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_name": "Frame",
                                "category": "visual",
                                "paper_count": 5,
                                "mention_count": 10,
                            },
                            {
                                "feature_name": "Framing",
                                "category": "visual",
                                "paper_count": 3,
                                "mention_count": 8,
                            },
                            {
                                "feature_name": "Tone",
                                "category": "text",
                                "paper_count": 2,
                                "mention_count": 4,
                            },
                            {
                                "feature_name": "Source",
                                "category": "metadata",
                                "paper_count": 1,
                                "mention_count": 2,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "merge_outputs"
            client = FakeEmbeddingClient(
                [
                    [1.0, 0.0],
                    [0.999, 0.01],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )

            with mock.patch(
                "embedding_merge.tsne_coordinates",
                return_value=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            ) as tsne_mock:
                trace = run_embedding_merge(
                    features_path=features_path,
                    threshold=0.99,
                    text_fields=["feature_name"],
                    scope="all",
                    output_dir=output_dir,
                    prefix="features",
                    cache_path=root / "cache.jsonl",
                    client=client,
                )

            html_path = output_dir / "features_embedding_merged_features_gemini_embedding_tsne.html"
            html = html_path.read_text(encoding="utf-8")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(trace["counts"]["merged_features"], 3)
        self.assertEqual(trace["tsne"]["status"], "written")
        self.assertEqual(trace["tsne"]["usable_features"], 3)
        self.assertEqual(trace["outputs"]["merged_tsne_html"], str(html_path))
        self.assertTrue(tsne_mock.called)
        self.assertIn("Frame", html)
        self.assertIn("Gemini embedding t-SNE", html)

    def test_run_embedding_merge_sweep_embeds_once_and_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {"feature_name": "Alpha", "category": "one"},
                            {"feature_name": "Alpha Variant", "category": "two"},
                            {"feature_name": "Alpha Neighbor", "category": "three"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "sweep_outputs"
            client = FakeEmbeddingClient(
                [
                    [1.0, 0.0],
                    [0.999, 0.01],
                    [0.97, 0.24310488],
                ]
            )

            trace = run_embedding_merge_sweep(
                features_path=features_path,
                threshold_start=0.96,
                threshold_end=0.98,
                threshold_step=0.01,
                text_fields=["feature_name"],
                scope="all",
                output_dir=output_dir,
                prefix="features",
                cache_path=root / "cache.jsonl",
                client=client,
            )

            expected_paths = {
                "features_embedding_merge_sweep.csv",
                "features_embedding_merge_sweep.json",
                "features_embedding_merge_sweep_trace.json",
                "features_embedding_merge_sweep_metrics.png",
                "features_embedding_merge_sweep_by_category.csv",
                "features_embedding_merge_sweep_by_category.json",
                "features_embedding_merge_sweep_by_category.png",
                "features_embedding_similarity_distribution.png",
                "features_embedding_similarity_boxplot.png",
                "features_embedding_similarity_percentiles.png",
                "features_embedding_similarity_distribution_by_category.png",
                "features_embedding_similarity_boxplot_by_category.png",
                "features_embedding_similarity_percentiles_by_category.png",
            }
            actual_paths = {path.name for path in output_dir.iterdir()}
            self.assertTrue(expected_paths.issubset(actual_paths))
            sweep_payload = json.loads((output_dir / "features_embedding_merge_sweep.json").read_text())
            category_payload = json.loads((output_dir / "features_embedding_merge_sweep_by_category.json").read_text())
            with (output_dir / "features_embedding_merge_sweep.csv").open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            with (output_dir / "features_embedding_merge_sweep_by_category.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                category_csv_rows = list(csv.DictReader(handle))
            png_header = (output_dir / "features_embedding_merge_sweep_metrics.png").read_bytes()[:8]
            category_png_header = (output_dir / "features_embedding_merge_sweep_by_category.png").read_bytes()[:8]
            similarity_distribution_png_header = (
                output_dir / "features_embedding_similarity_distribution.png"
            ).read_bytes()[:8]
            similarity_boxplot_png_header = (
                output_dir / "features_embedding_similarity_boxplot.png"
            ).read_bytes()[:8]
            similarity_percentiles_png_header = (
                output_dir / "features_embedding_similarity_percentiles.png"
            ).read_bytes()[:8]
            similarity_distribution_by_category_png_header = (
                output_dir / "features_embedding_similarity_distribution_by_category.png"
            ).read_bytes()[:8]
            similarity_boxplot_by_category_png_header = (
                output_dir / "features_embedding_similarity_boxplot_by_category.png"
            ).read_bytes()[:8]
            similarity_percentiles_by_category_png_header = (
                output_dir / "features_embedding_similarity_percentiles_by_category.png"
            ).read_bytes()[:8]

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]), 3)
        self.assertEqual(trace["thresholds"], [0.96, 0.97, 0.98])
        self.assertEqual(trace["counts"]["thresholds"], 3)
        self.assertEqual(png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_distribution_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_boxplot_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_percentiles_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_distribution_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_boxplot_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(similarity_percentiles_by_category_png_header, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(trace["similarity_distribution"]["pair_count"], 3)
        self.assertEqual(trace["similarity_distribution"]["bin_count"], 80)
        self.assertEqual(trace["similarity_distributions_by_category"]["one"]["feature_count"], 1)
        self.assertEqual([row["threshold"] for row in trace["results"]], [0.96, 0.97, 0.98])
        self.assertEqual([row["similarity_pairs"] for row in trace["results"]], [3, 3, 1])
        self.assertEqual([row["merge_groups"] for row in trace["results"]], [1, 1, 1])
        self.assertEqual([row["merged_features"] for row in trace["results"]], [1, 1, 2])
        self.assertEqual([row["merged_away_features"] for row in trace["results"]], [2, 2, 1])
        self.assertEqual(sweep_payload["threshold_count"], 3)
        self.assertEqual(len(sweep_payload["category_results"]), 9)
        self.assertEqual(category_payload["threshold_count"], 3)
        self.assertEqual(len(category_payload["results"]), 9)
        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(len(category_csv_rows), 9)
        self.assertEqual(category_png_header, b"\x89PNG\r\n\x1a\n")

        category_rows = {
            (row["threshold"], row["category"]): row
            for row in trace["category_results"]
        }
        self.assertEqual(category_rows[(0.96, "one")]["resulting_features"], 1)
        self.assertEqual(category_rows[(0.96, "two")]["resulting_features"], 0)
        self.assertEqual(category_rows[(0.96, "three")]["resulting_features"], 0)
        self.assertEqual(category_rows[(0.98, "one")]["merge_groups"], 1)
        self.assertEqual(category_rows[(0.98, "two")]["merge_groups"], 1)
        self.assertEqual(category_rows[(0.98, "three")]["merge_groups"], 0)

    def test_main_single_threshold_still_writes_full_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {"feature_name": "Tone", "category": "style"},
                            {"feature_name": "Tone of Voice", "category": "audio"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cache_path = root / "cache.jsonl"
            run_embedding_merge(
                features_path=features_path,
                threshold=0.99,
                text_fields=["feature_name"],
                scope="all",
                output_dir=root / "seed",
                prefix="features",
                cache_path=cache_path,
                client=FakeEmbeddingClient([[1.0, 0.0], [0.999, 0.01]]),
            )
            output_dir = root / "cli_single"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(features_path),
                        "--threshold",
                        "0.99",
                        "--text-fields",
                        "feature_name",
                        "--cache",
                        str(cache_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            actual_paths = {path.name for path in output_dir.iterdir()}

        self.assertEqual(exit_code, 0)
        self.assertIn("features_embedding_merge_pairs.jsonl", actual_paths)
        self.assertIn("features_embedding_merged_features.json", actual_paths)
        self.assertNotIn("features_embedding_merge_sweep.csv", actual_paths)

    def test_main_sweep_rejects_mixing_threshold_modes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "features.json",
                        "--threshold",
                        "0.90",
                        "--threshold-start",
                        "0.80",
                        "--threshold-end",
                        "0.90",
                    ]
                )

        self.assertNotEqual(raised.exception.code, 0)

    def test_run_embedding_merge_reads_jsonl_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.jsonl"
            features_path.write_text(
                "\n".join(
                    [
                        json.dumps({"feature_name": "Name Calling", "category": "persuasion"}),
                        json.dumps({"feature_name": "Labeling", "category": "persuasion"}),
                    ]
                ),
                encoding="utf-8",
            )
            client = FakeEmbeddingClient([[1.0, 0.0], [0.999, 0.01]])

            trace = run_embedding_merge(
                features_path=features_path,
                threshold=0.99,
                text_fields=["feature_name"],
                scope="category",
                output_dir=root / "out",
                prefix="jsonl",
                cache_path=root / "cache.jsonl",
                client=client,
            )

        self.assertEqual(trace["counts"]["input_features"], 2)
        self.assertEqual(trace["counts"]["merge_groups"], 1)

    def test_cli_argument_validation_rejects_invalid_threshold(self):
        parser = build_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(["features.json", "--threshold", "0"])

        self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
