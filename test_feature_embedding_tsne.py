import csv
import json
import tempfile
import unittest
from pathlib import Path

from feature_embedding_tsne import (
    FeatureItem,
    build_embedding_text,
    choose_perplexity,
    embed_texts_with_cache,
    load_feature_items,
    parse_text_fields,
    read_feature_rows,
    write_interactive_html,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]


class FeatureEmbeddingTsneTests(unittest.TestCase):
    def test_loads_csv_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["feature_name", "category", "definitions", "synonyms", "paper_count"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "feature_name": "Loaded Language",
                        "category": "persuasion",
                        "definitions": json.dumps(["Words with strong emotional implications."]),
                        "synonyms": json.dumps(["emotive language"]),
                        "paper_count": "5",
                    }
                )

            rows = read_feature_rows(path)
            items = load_feature_items(path, ["feature_name", "category", "definitions", "synonyms"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].label, "Loaded Language")
        self.assertEqual(items[0].paper_count, 5)
        self.assertIn("Definitions: Words with strong emotional implications.", items[0].embedding_text)

    def test_loads_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"feature_name": "Appeal", "category": "persuasion"}),
                        json.dumps({"feature_name": "Care", "category": "moral_framing"}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = read_feature_rows(path)

        self.assertEqual([row["feature_name"] for row in rows], ["Appeal", "Care"])

    def test_loads_json_rows_from_features_key(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.json"
            path.write_text(
                json.dumps(
                    {
                        "features": [
                            {"feature_name": "Anger", "category": "sentiment_affect"},
                            {"feature_name": "Joy", "category": "sentiment_affect"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = read_feature_rows(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["feature_name"], "Joy")

    def test_build_embedding_text_uses_scalar_and_jsonish_fields(self):
        row = {
            "feature_name": "Name Calling",
            "category": "persuasion",
            "definitions": json.dumps(["Derogatory labeling.", "Attacking a person or group."]),
            "synonyms": "labeling|labelling",
        }

        text = build_embedding_text(row, parse_text_fields("feature_name,category,definitions,synonyms"))

        self.assertIn("Feature Name: Name Calling", text)
        self.assertIn("Category: persuasion", text)
        self.assertIn("Definitions: Derogatory labeling.; Attacking a person or group.", text)
        self.assertIn("Synonyms: labeling; labelling", text)

    def test_embed_texts_with_cache_reuses_cached_vectors(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "embedding_cache.jsonl"
            first_client = FakeEmbeddingClient()

            embeddings, stats = embed_texts_with_cache(
                ["alpha", "beta", "alpha"],
                cache_path=cache_path,
                client=first_client,
                model="gemini-embedding-001",
                task_type="CLUSTERING",
                output_dimensionality=768,
                batch_size=10,
            )

            second_client = FakeEmbeddingClient()
            cached_embeddings, cached_stats = embed_texts_with_cache(
                ["alpha", "beta"],
                cache_path=cache_path,
                client=second_client,
                model="gemini-embedding-001",
                task_type="CLUSTERING",
                output_dimensionality=768,
                batch_size=10,
            )

        self.assertEqual(first_client.calls, [["alpha", "beta"]])
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.api_misses, 2)
        self.assertEqual(stats.api_batches, 1)
        self.assertEqual(embeddings[0], embeddings[2])
        self.assertEqual(second_client.calls, [])
        self.assertEqual(cached_stats.cache_hits, 2)
        self.assertEqual(cached_stats.api_misses, 0)
        self.assertEqual(cached_embeddings, embeddings[:2])

    def test_choose_perplexity_requires_three_samples_and_valid_requested_value(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            choose_perplexity(2)
        with self.assertRaisesRegex(ValueError, "less than"):
            choose_perplexity(5, requested=5)

        self.assertLess(choose_perplexity(3), 3)
        self.assertEqual(choose_perplexity(10, requested=4.5), 4.5)

    def test_write_interactive_html_embeds_points_and_controls(self):
        items = [
            FeatureItem(
                row_index=1,
                feature_key="persuasion::loaded language",
                label="Loaded Language",
                category="persuasion",
                paper_count=5,
                mention_count=8,
                embedding_text="Feature Name: Loaded Language\nCategory: persuasion",
                raw_row={},
            ),
            FeatureItem(
                row_index=2,
                feature_key="moral::care",
                label="Care",
                category="moral_framing",
                paper_count=2,
                mention_count=3,
                embedding_text="Feature Name: Care\nCategory: moral_framing",
                raw_row={},
            ),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plot.html"

            write_interactive_html(
                path,
                items,
                [(0.0, 1.0), (1.0, 0.0)],
                title="Feature Map",
                annotate_top=1,
            )
            html = path.read_text(encoding="utf-8")

        self.assertIn('id="feature-search"', html)
        self.assertIn("Loaded Language", html)
        self.assertIn('"paper_count": 5', html)
        self.assertIn("Wheel zoom, drag pan", html)


if __name__ == "__main__":
    unittest.main()
