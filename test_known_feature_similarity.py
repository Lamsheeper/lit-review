import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from known_feature_similarity import (
    ATTRIBUTE_COMPARISON_FIELDS,
    HEATMAP_COLORMAP,
    HEATMAP_MAX_SIMILARITY,
    HEATMAP_MIN_SIMILARITY,
    SYNONYM_GENERATION_MAX_TOKENS,
    apply_generated_attributes,
    attribute_combination_slug,
    attribute_combinations,
    best_threshold_f1_row,
    build_generated_attribute_prompt,
    build_synonym_review_prompt,
    build_synonym_control_rows,
    ensure_synonym_review_for_run,
    feature_identity,
    generate_attribute_metadata,
    generate_synonym_proposal,
    load_synonym_review,
    main,
    pairwise_similarity_data,
    parse_synonym_proposal,
    rank_attribute_comparisons,
    reconcile_synonym_review,
    resolve_known_rows,
    run_attribute_comparison,
    run_known_feature_similarity,
    run_synonym_review,
    separation_metrics,
    similarity_summary,
    synonym_control_pair_data,
    threshold_feature_set_metrics_sweep,
    unicode_normalize_key,
    validate_review_synonyms,
    write_most_similar_distinct_features_markdown,
    write_synonym_review,
    write_synonym_controls_markdown,
    write_attribute_ranking_plot,
    write_feature_set_metrics_plot,
)
from feature_embedding_tsne import EmbeddingTextCleanup, FeatureItem, load_feature_items


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


class DeterministicEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([1.0 + digest[index] / 255.0 for index in range(8)])
        return vectors


class FakeAttributeClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
        self.prompts.append(str(messages[-1]["content"]))
        if not self.responses:
            raise AssertionError("FakeAttributeClient ran out of responses.")
        return json.dumps({"annotations": [self.responses.pop(0)]})


class FakeSynonymClient:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
        self.prompts.append(str(messages[-1]["content"]))
        self.calls.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("FakeSynonymClient ran out of responses.")
        return json.dumps({"synonyms": self.responses.pop(0)})


def write_complete_review(
    path: Path,
    known_path: Path,
    pairs: dict[str, list[str]],
) -> Path:
    with known_path.open(newline="", encoding="utf-8") as handle:
        known_rows = list(csv.DictReader(handle))
    review = reconcile_synonym_review(
        known_rows,
        None,
        known_set_path=known_path,
        model="fake-synonym-model",
    )
    for entry in review["entries"]:
        entry["status"] = "approved"
        entry["approved_synonyms"] = pairs[entry["feature_name"]]
        entry["current_proposal"] = list(entry["approved_synonyms"])
    write_synonym_review(path, review)
    return path


class KnownFeatureSimilarityTests(unittest.TestCase):
    def test_heatmap_uses_focused_green_similarity_scale(self):
        self.assertEqual(HEATMAP_COLORMAP, "Greens")
        self.assertEqual(HEATMAP_MIN_SIMILARITY, 0.8)
        self.assertEqual(HEATMAP_MAX_SIMILARITY, 1.0)

    def test_attribute_combinations_include_all_fifteen_non_empty_subsets(self):
        combinations = attribute_combinations()

        self.assertEqual(len(combinations), 15)
        self.assertEqual(combinations[0], ("feature_name",))
        self.assertEqual(combinations[-1], ATTRIBUTE_COMPARISON_FIELDS)
        self.assertEqual(
            [attribute_combination_slug(fields) for fields in combinations[:5]],
            ["name", "synonyms", "definitions", "examples", "name_synonyms"],
        )

    def test_generated_attribute_prompt_contains_only_label_and_category_input(self):
        prompt = build_generated_attribute_prompt("Anger", "sentiment_affect")

        input_payload = json.loads(prompt.split("INPUT\n", 1)[1])
        self.assertEqual(input_payload, {"feature_name": "Anger", "category": "sentiment_affect"})
        self.assertNotIn("authoritative", prompt.casefold())
        self.assertNotIn("definitions\": [", prompt)
        self.assertIn("1 to 5 extremely close substitutable alternate labels", prompt)
        self.assertIn("merely associated concepts", prompt)
        self.assertNotIn("strict substitution test", prompt)
        self.assertNotIn('"irritation" is not', prompt)

    def test_semeval_2018_emotion_inventory_has_official_eleven_atomic_labels(self):
        inventory_path = Path("sanity_checks/semeval_2018_task1_emotion/expected_features.csv")
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(
            [row["canonical_feature_name"] for row in rows],
            [
                "Anger",
                "Anticipation",
                "Disgust",
                "Fear",
                "Joy",
                "Love",
                "Optimism",
                "Pessimism",
                "Sadness",
                "Surprise",
                "Trust",
            ],
        )
        self.assertTrue(all(not row["matching_aliases"] for row in rows))
        self.assertTrue(all(row["related_terms"] for row in rows))

    def test_geneva_emotion_wheel_2_inventory_has_forty_atomic_terms(self):
        inventory_path = Path("sanity_checks/geneva_emotion_wheel_2_0/expected_features.csv")
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 40)
        self.assertEqual(
            [row["canonical_feature_name"] for row in rows[:3]],
            ["Irritation", "Anger", "Involvement"],
        )
        self.assertEqual(len({row["source_family"] for row in rows}), 20)
        self.assertEqual(
            Counter((row["valence"], row["control"]) for row in rows),
            {
                ("negative", "high"): 10,
                ("positive", "high"): 10,
                ("positive", "low"): 10,
                ("negative", "low"): 10,
            },
        )

    def test_geneva_emotion_wheel_1_inventory_has_sixteen_atomic_terms(self):
        inventory_path = Path("sanity_checks/geneva_emotion_wheel_1_0/expected_features.csv")
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[4]["canonical_feature_name"], "Guilt")
        self.assertEqual(rows[11]["canonical_feature_name"], "Relief")
        self.assertTrue(all("," not in row["canonical_feature_name"] for row in rows))

    def test_atomic_inventory_control_populations_are_exactly_three_rows_per_feature(self):
        expected = {
            Path("sanity_checks/semeval_2020_task11/expected_features.csv"): (18, 54),
            Path("sanity_checks/geneva_emotion_wheel_1_0/expected_features.csv"): (16, 48),
            Path("sanity_checks/geneva_emotion_wheel_2_0/expected_features.csv"): (40, 120),
        }
        for path, (feature_count, control_count) in expected.items():
            with self.subTest(path=path):
                with path.open(newline="", encoding="utf-8") as handle:
                    known_rows = list(csv.DictReader(handle))
                resolved = resolve_known_rows(known_rows)
                approved = {
                    feature_identity(row["feature_name"], row["category"]): [
                        f"{row['feature_name']} control alpha",
                        f"{row['feature_name']} control beta",
                    ]
                    for row in resolved
                }
                controls = build_synonym_control_rows(resolved, approved)

                self.assertEqual(len(resolved), feature_count)
                self.assertEqual(len(controls), control_count)
                self.assertEqual(control_count, feature_count * 3)

    def test_semeval_retains_name_calling_and_exaggeration_combined_rows(self):
        inventory_path = Path("sanity_checks/semeval_2020_task11/expected_features.csv")
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        labels = [row["canonical_feature_name"] for row in rows]
        self.assertEqual(len(rows), 18)
        self.assertIn("Name calling, labeling", labels)
        self.assertIn("Exaggeration or minimization", labels)
        self.assertNotIn("Name calling", labels)
        self.assertNotIn("Labeling", labels)
        self.assertNotIn("Exaggeration", labels)
        self.assertNotIn("Minimization", labels)
        self.assertNotIn("Dictatorship", labels)
        combined = {row["canonical_feature_name"]: row for row in rows}
        self.assertIn("labeling", combined["Name calling, labeling"]["matching_aliases"])
        self.assertIn("Exaggeration", combined["Exaggeration or minimization"]["matching_aliases"])

    def test_exact_canonical_match_takes_priority_over_alias_matches(self):
        known = [
            {
                "canonical_feature_name": "Doubt",
                "aliases": "Questioning|Skepticism",
                "category": "persuasion",
            }
        ]
        annotated = [
            {
                "feature_name": "Questioning",
                "category": "persuasion",
                "definitions": ["Alias definition"],
            },
            {
                "feature_name": "Doubt",
                "category": "persuasion",
                "definitions": ["Canonical definition"],
            },
        ]

        row = resolve_known_rows(known, annotated)[0]

        self.assertEqual(row["resolution"]["status"], "exact_canonical")
        self.assertEqual(row["resolution"]["matched_row_count"], 1)
        self.assertEqual(row["resolution"]["matched_rows"][0]["feature_name"], "Doubt")
        self.assertEqual(row["definitions"], ["Canonical definition"])
        self.assertNotIn("Questioning", row["synonyms"])

    def test_atomic_known_rows_resolve_components_separately(self):
        known = [
            {
                "canonical_feature_name": "Whataboutism",
                "source_merged_label": "Whataboutism, straw man, red herring",
                "category": "persuasion",
            },
            {
                "canonical_feature_name": "Straw man",
                "source_merged_label": "Whataboutism, straw man, red herring",
                "category": "persuasion",
            },
            {
                "canonical_feature_name": "Red herring",
                "source_merged_label": "Whataboutism, straw man, red herring",
                "category": "persuasion",
            },
        ]
        annotated = [
            {
                "feature_name": "Whataboutism",
                "category": "persuasion",
                "annotation": {
                    "short_definition": "Answers criticism with a counter-accusation.",
                    "synonyms": ["whataboutery"],
                },
            },
            {
                "feature_name": "Straw Man",
                "category": "persuasion",
                "definitions": ["Misrepresents an opposing position."],
            },
            {
                "feature_name": "Red Herring",
                "category": "persuasion",
                "examples": ["Changing the subject"],
            },
        ]

        rows = resolve_known_rows(known, annotated)

        self.assertEqual([row["feature_name"] for row in rows], ["Whataboutism", "Straw man", "Red herring"])
        self.assertTrue(all(row["resolution"]["matched_row_count"] == 1 for row in rows))
        self.assertTrue(all(row["resolution"]["status"] == "exact_canonical" for row in rows))

    def test_missing_label_uses_deterministic_known_metadata_fallback(self):
        known = [
            {
                "feature_id": "known-1",
                "canonical_feature_name": "Unmatched Technique",
                "aliases": "Unmatched label",
                "category": "persuasion",
                "source_note": "Authoritative inventory",
            }
        ]

        first = resolve_known_rows(known, [])[0]
        second = resolve_known_rows(known, [])[0]

        self.assertEqual(first, second)
        self.assertEqual(first["feature_key"], "known-1")
        self.assertEqual(first["resolution"]["status"], "known_metadata_fallback")
        self.assertEqual(first["known_set_metadata"]["source_note"], "Authoritative inventory")

    def test_unicode_normalization_matches_cliche_and_cliche_with_accent(self):
        self.assertEqual(unicode_normalize_key("Thought-Terminating Cliché"), "thought terminating cliche")
        known = [
            {
                "canonical_feature_name": "Thought-terminating cliche",
                "category": "persuasion",
            }
        ]
        annotated = [
            {
                "feature_name": "Thought-Terminating Cliché",
                "category": "persuasion",
            }
        ]

        row = resolve_known_rows(known, annotated)[0]

        self.assertEqual(row["resolution"]["status"], "exact_canonical")

    def test_category_aware_matching_avoids_unrelated_same_name(self):
        known = [{"canonical_feature_name": "Authority", "category": "persuasion"}]
        annotated = [
            {
                "feature_name": "Authority",
                "category": "moral_framing",
                "definitions": ["Moral authority frame."],
            }
        ]

        row = resolve_known_rows(known, annotated)[0]

        self.assertEqual(row["resolution"]["status"], "known_metadata_fallback")
        self.assertEqual(row["definitions"], [])

    def test_pairwise_data_excludes_diagonal_and_summarizes_all_unique_pairs(self):
        labels = ["One", "Two", "Three"]
        normalized = [
            [1.0, 0.0],
            [0.0, 1.0],
            [2**-0.5, 2**-0.5],
        ]

        pairs, matrix = pairwise_similarity_data(labels, normalized)
        summary = similarity_summary(pairs)

        self.assertEqual(len(pairs), 3)
        self.assertEqual(summary["pair_count"], 3)
        self.assertEqual(len(matrix), 3)
        self.assertEqual([matrix[index][index] for index in range(3)], [1.0, 1.0, 1.0])
        self.assertFalse(any(row["left_index"] == row["right_index"] for row in pairs))

    def test_threshold_feature_set_metrics_scores_produced_inventory(self):
        items = [
            FeatureItem(
                row_index=index + 1,
                feature_key=f"feature-{index}",
                label=f"Feature {index}",
                category="test",
                paper_count=0,
                mention_count=0,
                embedding_text=f"Feature {index}",
                raw_row={"synonym_control": {"group_feature_key": "actual-a" if index < 3 else "actual-b"}},
            )
            for index in range(6)
        ]
        matrix = [[1.0, 0.0] if index < 3 else [0.0, 1.0] for index in range(6)]
        rows = threshold_feature_set_metrics_sweep(items, matrix)
        by_threshold = {row["threshold"]: row for row in rows}
        exact_set = by_threshold[1.0]
        collapsed_set = by_threshold[0.0]
        best = best_threshold_f1_row(rows)

        self.assertEqual(exact_set["actual_feature_count"], 2)
        self.assertEqual(exact_set["produced_feature_count"], 2)
        self.assertEqual(exact_set["matched_feature_count"], 2)
        self.assertEqual(exact_set["precision"], 1.0)
        self.assertEqual(exact_set["recall"], 1.0)
        self.assertEqual(exact_set["f1"], 1.0)
        self.assertEqual(collapsed_set["produced_feature_count"], 1)
        self.assertEqual(collapsed_set["precision"], 1.0)
        self.assertEqual(collapsed_set["recall"], 0.5)
        self.assertEqual(best["threshold"], 1.0)
        self.assertEqual(best["f1"], 1.0)

    def test_threshold_feature_set_metrics_penalizes_duplicate_outputs(self):
        items = [
            FeatureItem(
                row_index=index + 1,
                feature_key=f"feature-{index}",
                label=f"Feature {index}",
                category="test",
                paper_count=0,
                mention_count=0,
                embedding_text=f"Feature {index}",
                raw_row={"synonym_control": {"group_feature_key": "actual-a" if index < 3 else "actual-b"}},
            )
            for index in range(6)
        ]
        matrix = [[1.0, 0.0] if index < 3 else [0.0, 1.0] for index in range(6)]
        rows = threshold_feature_set_metrics_sweep(items, matrix)
        strict_no_merge = max(rows, key=lambda row: row["threshold"])

        self.assertEqual(strict_no_merge["produced_feature_count"], 6)
        self.assertEqual(strict_no_merge["matched_feature_count"], 2)
        self.assertEqual(strict_no_merge["precision"], 0.33333333)
        self.assertEqual(strict_no_merge["recall"], 1.0)

    def test_feature_set_metrics_plot_marks_only_best_f1_without_vertical_line(self):
        import matplotlib.axes

        labels: list[str] = []
        vertical_lines: list[tuple[object, ...]] = []
        original_scatter = matplotlib.axes.Axes.scatter
        original_axvline = matplotlib.axes.Axes.axvline

        def capture_scatter(axis, *args, **kwargs):
            labels.append(str(kwargs.get("label") or ""))
            return original_scatter(axis, *args, **kwargs)

        def capture_axvline(axis, *args, **kwargs):
            vertical_lines.append(args)
            return original_axvline(axis, *args, **kwargs)

        rows = [
            {"threshold": 0.7, "precision": 0.5, "recall": 1.0, "f1": 0.66666667},
            {"threshold": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8},
        ]
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(matplotlib.axes.Axes, "scatter", capture_scatter):
                with mock.patch.object(matplotlib.axes.Axes, "axvline", capture_axvline):
                    write_feature_set_metrics_plot(
                        Path(temp) / "metrics.png",
                        rows,
                        title="Metrics",
                    )

        self.assertTrue(any(label.startswith("Best F1") for label in labels))
        self.assertFalse(any(label.startswith("Full recall") for label in labels))
        self.assertEqual(vertical_lines, [])

    def test_most_similar_distinct_markdown_sorts_limits_and_escapes_rows(self):
        pairs = [
            {
                "similarity": 0.5 + index / 1000,
                "left_index": 1,
                "left_feature_name": "Left | Feature",
                "right_index": 2,
                "right_feature_name": f"Pair {index}",
            }
            for index in range(25)
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pairs.md"
            write_most_similar_distinct_features_markdown(
                path,
                known_set_path=Path("known.csv"),
                text_fields=["feature_name"],
                pairs=pairs,
                categories=["cat | one", "cat two"],
            )
            markdown = path.read_text()

        self.assertIn("| 1 | Left \\| Feature | cat \\| one | Pair 24 | cat two | 0.52400000 | 0.52400000 |", markdown)
        self.assertIn("| 20 | Left \\| Feature | cat \\| one | Pair 5 | cat two | 0.50500000 | 0.50500000 |", markdown)
        self.assertNotIn("Pair 4 |", markdown)

    def test_synonym_controls_use_only_human_approved_terms_and_resolve_independently(self):
        resolved = [
            {
                "feature_key": "known-one",
                "feature_name": "Known One",
                "category": "test",
                "definitions": ["A stable definition."],
                "synonyms": ["annotation-only synonym"],
                "known_set_metadata": {
                    "canonical_feature_name": "Known One",
                    "matching_aliases": "matching-only alias",
                },
            }
        ]
        approved = {feature_identity("Known One", "test"): ["First", "Primary"]}
        annotated = [
            {"feature_name": "First", "category": "test", "definitions": ["First definition."]}
        ]

        controls = build_synonym_control_rows(resolved, approved, annotated)

        self.assertEqual([row["feature_name"] for row in controls], ["Known One", "First", "Primary"])
        self.assertTrue(all(row["synonym_control"]["source"] == "human_approved_synonym" for row in controls))
        self.assertEqual(controls[1]["definitions"], ["First definition."])
        self.assertEqual(controls[2]["resolution"]["status"], "known_metadata_fallback")
        self.assertNotIn("matching-only alias", [row["feature_name"] for row in controls])

    def test_synonym_control_pairs_compare_only_rows_from_the_same_feature(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "controls.json"
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "feature_name": "One",
                                "synonym_control": {
                                    "group_feature_key": "one",
                                    "group_feature_name": "One",
                                    "variant_type": "canonical",
                                },
                            },
                            {
                                "feature_name": "First",
                                "synonym_control": {
                                    "group_feature_key": "one",
                                    "group_feature_name": "One",
                                    "variant_type": "synonym",
                                },
                            },
                            {
                                "feature_name": "Primary",
                                "synonym_control": {
                                    "group_feature_key": "one",
                                    "group_feature_name": "One",
                                    "variant_type": "synonym",
                                },
                            },
                            {
                                "feature_name": "Two",
                                "synonym_control": {
                                    "group_feature_key": "two",
                                    "group_feature_name": "Two",
                                    "variant_type": "canonical",
                                },
                            },
                            {
                                "feature_name": "Second",
                                "synonym_control": {
                                    "group_feature_key": "two",
                                    "group_feature_name": "Two",
                                    "variant_type": "synonym",
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            items = load_feature_items(path, ["feature_name"])

        pairs = synonym_control_pair_data(
            items,
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
        )

        self.assertEqual(len(pairs), 4)
        self.assertEqual({row["group_feature_name"] for row in pairs}, {"One", "Two"})
        self.assertEqual(
            [row["pair_type"] for row in pairs].count("canonical_synonym"),
            3,
        )
        self.assertEqual(
            [row["pair_type"] for row in pairs].count("synonym_synonym"),
            1,
        )

    def test_review_validation_requires_exactly_two_and_rejects_collisions(self):
        entries = [
            {
                "identity_key": "one",
                "feature_name": "One",
                "category": "test",
                "status": "pending",
                "approved_synonyms": [],
            },
            {
                "identity_key": "two",
                "feature_name": "Two",
                "category": "test",
                "status": "approved",
                "approved_synonyms": ["Second", "Other"],
            },
        ]

        with self.assertRaisesRegex(ValueError, "Exactly two"):
            validate_review_synonyms(["First"], entry=entries[0], entries=entries)
        with self.assertRaisesRegex(ValueError, "collide"):
            validate_review_synonyms(["Two", "Primary"], entry=entries[0], entries=entries)
        with self.assertRaisesRegex(ValueError, "collide"):
            validate_review_synonyms(["Second", "Primary"], entry=entries[0], entries=entries)
        self.assertEqual(
            validate_review_synonyms(["First", "Primary"], entry=entries[0], entries=entries),
            ["First", "Primary"],
        )

    def test_interactive_synonym_review_approves_edits_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\ntwo,Two,test\n",
                encoding="utf-8",
            )
            review_path = root / "review.json"
            client = FakeSynonymClient([["First", "Primary"], ["Second", "Secondary"]])
            first_answers = iter(["a", "q"])
            first = run_synonym_review(
                known_set_path=known_path,
                review_path=review_path,
                client=client,
                model="fake",
                input_fn=lambda _: next(first_answers),
                output_fn=lambda _: None,
            )
            resumed_answers = iter(["e", "Second", "Alternate", "a"])
            resumed = run_synonym_review(
                known_set_path=known_path,
                review_path=review_path,
                client=None,
                model="fake",
                input_fn=lambda _: next(resumed_answers),
                output_fn=lambda _: None,
            )
            review = load_synonym_review(review_path)

        self.assertFalse(first["complete"])
        self.assertTrue(resumed["complete"])
        self.assertEqual(review["entries"][0]["approved_synonyms"], ["First", "Primary"])
        self.assertEqual(review["entries"][1]["approved_synonyms"], ["Second", "Alternate"])
        self.assertEqual(len(client.prompts), 2)

    def test_interactive_review_records_generation_failure_and_manual_terms(self):
        class FailingClient:
            def __init__(self):
                self.calls = 0

            def chat(self, *args, **kwargs):
                self.calls += 1
                raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\n",
                encoding="utf-8",
            )
            review_path = root / "review.json"
            answers = iter(["e", "First", "Primary", "a"])
            client = FailingClient()
            trace = run_synonym_review(
                known_set_path=known_path,
                review_path=review_path,
                client=client,
                model="fake",
                input_fn=lambda _: next(answers),
                output_fn=lambda _: None,
            )
            review = load_synonym_review(review_path)

        self.assertTrue(trace["complete"])
        self.assertEqual(client.calls, 1)
        self.assertEqual(review["entries"][0]["proposal_history"][0]["action"], "generation_failed")
        self.assertEqual(review["entries"][0]["proposal_history"][0]["model"], "fake")
        self.assertEqual(review["entries"][0]["approved_synonyms"], ["First", "Primary"])

    def test_interactive_review_regenerates_with_user_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\n",
                encoding="utf-8",
            )
            client = FakeSynonymClient([["First", "Primary"], ["Foremost", "Principal"]])
            answers = iter(["r", "Prefer formal terms", "a"])
            trace = run_synonym_review(
                known_set_path=known_path,
                review_path=root / "review.json",
                client=client,
                model="fake",
                input_fn=lambda _: next(answers),
                output_fn=lambda _: None,
            )

        self.assertTrue(trace["complete"])
        self.assertIn("Prefer formal terms", client.prompts[1])

    def test_ensure_review_runs_interactively_then_reuses_complete_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            review_path = root / "review.json"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\ntwo,Two,test\n",
                encoding="utf-8",
            )
            client = FakeSynonymClient([["First", "Primary"], ["Second", "Secondary"]])
            answers = iter(["a", "a"])
            first = ensure_synonym_review_for_run(
                known_set_path=known_path,
                review_path=review_path,
                client=client,
                model="fake",
                interactive=True,
                input_fn=lambda _: next(answers),
                output_fn=lambda _: None,
            )
            second = ensure_synonym_review_for_run(
                known_set_path=known_path,
                review_path=review_path,
                client=None,
                model="fake",
                interactive=False,
                output_fn=lambda _: None,
            )

        self.assertTrue(first["complete"])
        self.assertIsNone(second)

    def test_ensure_review_fails_cleanly_when_noninteractive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\ntwo,Two,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be completed non-interactively"):
                ensure_synonym_review_for_run(
                    known_set_path=known_path,
                    review_path=root / "review.json",
                    client=None,
                    model="fake",
                    interactive=False,
                    output_fn=lambda _: None,
                )

    def test_review_reconciliation_preserves_unchanged_approval_and_reopens_collision(self):
        original = [
            {"feature_id": "one", "canonical_feature_name": "One", "category": "test"},
        ]
        review = reconcile_synonym_review(
            original,
            None,
            known_set_path=Path("known.csv"),
            model="fake",
        )
        review["entries"][0]["status"] = "approved"
        review["entries"][0]["approved_synonyms"] = ["First", "Primary"]

        reconciled = reconcile_synonym_review(
            [
                *original,
                {"feature_id": "first", "canonical_feature_name": "First", "category": "test"},
            ],
            review,
            known_set_path=Path("known.csv"),
            model="fake",
        )

        self.assertEqual(reconciled["entries"][0]["status"], "pending")
        self.assertEqual(reconciled["entries"][1]["status"], "pending")
        self.assertEqual(reconciled["entries"][0]["proposal_history"][-1]["action"], "approval_reopened")

    def test_synonym_review_prompt_contains_only_label_category_and_feedback(self):
        prompt = build_synonym_review_prompt("Anger", "sentiment_affect", "Prefer same intensity")
        payload = json.loads(prompt.split("INPUT\n", 1)[1])

        self.assertEqual(
            payload,
            {
                "feature_name": "Anger",
                "category": "sentiment_affect",
                "user_feedback": "Prefer same intensity",
            },
        )
        self.assertIn("exactly two", prompt.casefold())

    def test_synonym_proposal_parser_recovers_wrappers_arrays_and_truncated_json(self):
        self.assertEqual(
            parse_synonym_proposal(
                'Here you go: {"synonyms": ["Wrath", "Fury"]}',
                "Anger",
                "sentiment_affect",
            ),
            ["Wrath", "Fury"],
        )
        self.assertEqual(
            parse_synonym_proposal('["Wrath", "Fury"]', "Anger", "sentiment_affect"),
            ["Wrath", "Fury"],
        )
        self.assertEqual(
            parse_synonym_proposal(
                '{\n  "synonyms": ["Wrath", "Fury"',
                "Anger",
                "sentiment_affect",
            ),
            ["Wrath", "Fury"],
        )
        with self.assertRaisesRegex(ValueError, "response preview=''"):
            parse_synonym_proposal("", "Anger", "sentiment_affect")

    def test_synonym_generation_uses_sufficient_output_budget(self):
        client = FakeSynonymClient([["Wrath", "Fury"]])
        entry = {"feature_name": "Anger", "category": "sentiment_affect"}

        self.assertEqual(generate_synonym_proposal(entry, client=client), ["Wrath", "Fury"])
        self.assertEqual(client.calls[0]["max_tokens"], SYNONYM_GENERATION_MAX_TOKENS)
        self.assertGreaterEqual(SYNONYM_GENERATION_MAX_TOKENS, 1000)

    def test_generated_attributes_replace_shared_metadata_and_reuse_cache(self):
        rows = [
            {
                "feature_name": "Known One",
                "category": "sentiment_affect",
                "definitions": ["Shared definition"],
                "synonyms": [],
                "examples": ["Shared example"],
            },
            {
                "feature_name": "First",
                "category": "sentiment_affect",
                "definitions": ["Shared definition"],
                "synonyms": [],
                "examples": ["Shared example"],
            },
        ]
        responses = [
            {
                "definition": "A precise definition of Known One.",
                "synonyms": ["Primary"],
                "examples": ["Known One example A", "Known One example B"],
            },
            {
                "definition": "A precise definition of First.",
                "synonyms": ["Foremost"],
                "examples": ["First example A", "First example B"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "attributes.jsonl"
            client = FakeAttributeClient(responses)
            generated, trace = generate_attribute_metadata(rows, cache_path=cache_path, client=client)
            cached, cached_trace = generate_attribute_metadata(rows, cache_path=cache_path, client=None)
            replaced = apply_generated_attributes(rows, generated, model="gemini-3.5-flash")

        self.assertEqual(trace["generated_rows"], 2)
        self.assertEqual(cached_trace["cache_hits"], 2)
        self.assertEqual(generated, cached)
        self.assertEqual(replaced[0]["definitions"], ["A precise definition of Known One."])
        self.assertEqual(replaced[1]["definitions"], ["A precise definition of First."])
        self.assertNotEqual(replaced[0]["examples"], replaced[1]["examples"])
        self.assertEqual(replaced[0]["generated_attributes"]["input_fields"], ["feature_name", "category"])

    def test_synonym_control_markdown_groups_families_and_shows_examples(self):
        rows = [
            {
                "feature_name": "One",
                "definitions": ["Definition one."],
                "synonyms": ["Primary"],
                "examples": ["Example one."],
                "synonym_control": {
                    "group_feature_key": "one",
                    "group_feature_name": "One family",
                    "variant_type": "canonical",
                },
            },
            {
                "feature_name": "First",
                "definitions": ["Definition first."],
                "synonyms": ["Foremost"],
                "examples": ["Example first."],
                "synonym_control": {
                    "group_feature_key": "one",
                    "group_feature_name": "One family",
                    "variant_type": "synonym",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "controls.md"
            write_synonym_controls_markdown(
                path,
                known_set_path=Path("sanity_checks/test_set/known.csv"),
                rows=rows,
                known_rows=[
                    {"feature_key": "one", "feature_name": "One family"},
                    {"feature_key": "two", "feature_name": "Two family"},
                ],
            )
            markdown = path.read_text()

        self.assertIn("2 synonym-control rows across 1 of 2 atomic features", markdown)
        self.assertIn("`One`, `First`", markdown)
        self.assertIn("`Two family`", markdown)
        self.assertIn("Generated close synonyms", markdown)
        self.assertIn("Example first.", markdown)

    def test_generated_attributes_retry_after_strict_synonym_filtering(self):
        rows = [{"feature_name": "Anger", "category": "sentiment_affect"}]
        client = FakeAttributeClient(
            [
                {
                    "definition": "A strong negative emotion.",
                    "synonyms": ["Anger", "emotion", "A related concept."],
                    "examples": ["Example A", "Example B"],
                },
                {
                    "definition": "A strong feeling of displeasure or hostility.",
                    "synonyms": ["Wrath", "Rage"],
                    "examples": ["Fuming at an insult", "Raging after an injustice"],
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            generated, trace = generate_attribute_metadata(
                rows,
                cache_path=Path(temp) / "attributes.jsonl",
                client=client,
            )

        attributes = generated[("anger", "sentiment affect")]
        self.assertEqual(attributes["synonyms"], ["Wrath", "Rage"])
        self.assertEqual(trace["generation_attempts"], 2)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("RETRY CORRECTION", client.prompts[1])
        self.assertIn('paired contrast such as "X vs. Y"', client.prompts[1])

    def test_generated_attributes_fail_after_exhausted_retries(self):
        rows = [{"feature_name": "Anger", "category": "sentiment_affect"}]
        invalid = {
            "definition": "A strong negative emotion.",
            "synonyms": ["Anger", "emotion"],
            "examples": ["Only one example"],
        }
        client = FakeAttributeClient([invalid, invalid, invalid])

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Generated attributes are incomplete"):
                generate_attribute_metadata(
                    rows,
                    cache_path=Path(temp) / "attributes.jsonl",
                    client=client,
                )
        self.assertEqual(len(client.prompts), 3)

    def test_separation_metrics_and_ranking_tie_breakers(self):
        metrics = separation_metrics([0.1, 0.2, 0.3], [0.8, 0.9, 1.0])
        rankings = rank_attribute_comparisons(
            [
                {"combination": "z", "field_count": 2, "roc_auc": 1.0, "mean_gap": 0.5},
                {"combination": "a", "field_count": 1, "roc_auc": 1.0, "mean_gap": 0.5},
                {"combination": "b", "field_count": 1, "roc_auc": 1.0, "mean_gap": 0.5},
                {"combination": "higher_gap", "field_count": 4, "roc_auc": 1.0, "mean_gap": 0.6},
                {"combination": "lower_auc", "field_count": 1, "roc_auc": 0.9, "mean_gap": 0.9},
            ]
        )

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["cliffs_delta"], 1.0)
        self.assertNotIn("best_f1", metrics)
        self.assertEqual(
            [row["combination"] for row in rankings],
            ["higher_gap", "a", "b", "z", "lower_auc"],
        )

    def test_attribute_ranking_plot_annotates_roc_auc_not_gap(self):
        import matplotlib.axes

        seen: list[str] = []
        original = matplotlib.axes.Axes.text

        def capture_text(axis, x, y, text, *args, **kwargs):
            seen.append(str(text))
            return original(axis, x, y, text, *args, **kwargs)

        rankings = [
            {"combination": "name", "roc_auc": 0.875, "mean_gap": 0.12},
            {"combination": "definitions", "roc_auc": 0.75, "mean_gap": 0.08},
        ]
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(matplotlib.axes.Axes, "text", capture_text):
                write_attribute_ranking_plot(
                    Path(temp) / "ranking.png",
                    rankings,
                    title="Ranking",
                )

        self.assertIn("ROC AUC 0.875", seen)
        self.assertIn("ROC AUC 0.750", seen)
        self.assertFalse(any(text.startswith("gap ") for text in seen))

    def test_attribute_comparison_writes_fifteen_full_runs_and_rankings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            with known_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["feature_id", "canonical_feature_name", "aliases", "category"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"feature_id": "one", "canonical_feature_name": "One", "aliases": "First", "category": "test"},
                        {"feature_id": "two", "canonical_feature_name": "Two", "aliases": "Second", "category": "test"},
                        {"feature_id": "three", "canonical_feature_name": "Three", "aliases": "Third", "category": "test"},
                    ]
                )
            output_dir = root / "out"
            review_path = write_complete_review(
                root / "review.json",
                known_path,
                {
                    "One": ["First", "Primary"],
                    "Two": ["Second", "Secondary"],
                    "Three": ["Third", "Tertiary"],
                },
            )
            labels = ["One", "Two", "Three", "First", "Primary", "Second", "Secondary", "Third", "Tertiary"]
            attribute_client = FakeAttributeClient(
                [
                    {
                        "definition": f"Definition unique to {label}.",
                        "synonyms": [f"{label} equivalent"],
                        "examples": [f"{label} example A", f"{label} example B"],
                    }
                    for label in labels
                ]
            )
            trace = run_attribute_comparison(
                known_set_path=known_path,
                annotated_features_path=None,
                output_dir=output_dir,
                prefix="known",
                embedding_cache_path=root / "embeddings.jsonl",
                attribute_cache_path=root / "attributes.jsonl",
                synonym_review_path=review_path,
                embedding_client=DeterministicEmbeddingClient(),
                attribute_client=attribute_client,
                output_dimensionality=8,
            )
            combination_dirs = sorted(path.name for path in (output_dir / "combinations").iterdir())
            per_run_traces = [
                json.loads((output_dir / "combinations" / slug / "known_similarity_trace.json").read_text())
                for slug in combination_dirs
            ]
            required_run_artifacts = {
                "known_resolved_features.json",
                "known_pairwise_similarities.csv",
                "known_most_similar_distinct_features.md",
                "known_similarity_matrix.csv",
                "known_similarity_heatmap.png",
                "known_similarity_histogram.png",
                "known_similarity_boxplot.png",
                "known_synonym_control_rows.json",
                "known_synonym_pairwise_similarities.csv",
                "known_synonym_similarity_histogram.png",
                "known_synonym_similarity_boxplot.png",
                "known_similarity_distribution_comparison.png",
                "known_feature_set_precision_recall_f1.png",
                "known_similarity_trace.json",
            }
            artifact_sets = [
                {path.name for path in (output_dir / "combinations" / slug).iterdir()}
                for slug in combination_dirs
            ]
            with (output_dir / "known_attribute_comparison_rankings.csv").open(newline="", encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            rankings_payload = json.loads((output_dir / "known_attribute_comparison_rankings.json").read_text())
            synonym_controls_markdown = (output_dir / "known_synonym_controls.md").read_text()
            ranking_png_header = (output_dir / "known_attribute_comparison_ranking.png").read_bytes()[:8]

        self.assertEqual(trace["combination_count"], 15)
        self.assertEqual(len(combination_dirs), 15)
        self.assertEqual(len(ranking_rows), 15)
        self.assertEqual(rankings_payload["combination_count"], 15)
        self.assertTrue(all(required_run_artifacts.issubset(artifacts) for artifacts in artifact_sets))
        self.assertEqual({row["counts"]["known_features"] for row in per_run_traces}, {3})
        self.assertEqual({row["counts"]["synonym_control_rows"] for row in per_run_traces}, {9})
        self.assertTrue(all(row["experiment_context"]["mode"] == "embed_attribute_comparison" for row in per_run_traces))
        self.assertTrue(all(row["experiment_context"]["synonym_review"]["path"] == str(review_path) for row in per_run_traces))
        self.assertTrue(all("threshold_set_metrics" in row for row in per_run_traces))
        self.assertEqual(len(attribute_client.prompts), 9)
        self.assertIn("3 of 3 atomic features", synonym_controls_markdown)
        self.assertIn("known_feature_generated_attributes_v1", synonym_controls_markdown)
        self.assertEqual(ranking_png_header, b"\x89PNG\r\n\x1a\n")

    def test_attribute_comparison_requires_complete_synonym_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "canonical_feature_name,category\nOne,test\nTwo,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                run_attribute_comparison(
                    known_set_path=known_path,
                    annotated_features_path=None,
                    output_dir=root / "out",
                    prefix="known",
                    embedding_cache_path=root / "embeddings.jsonl",
                    attribute_cache_path=root / "attributes.jsonl",
                    synonym_review_path=root / "missing-review.json",
                    embedding_client=DeterministicEmbeddingClient(),
                    attribute_client=None,
                )

    def test_cli_rejects_text_fields_with_attribute_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            known_path = Path(temp) / "known.csv"
            known_path.write_text(
                "canonical_feature_name,aliases,category\nOne,First,test\nTwo,Second,test\n",
                encoding="utf-8",
            )
            result = main(
                [
                    str(known_path),
                    "--embed-attribute-comparison",
                    "--text-fields",
                    "feature_name",
                ]
            )
        self.assertEqual(result, 1)

    def test_cli_rejects_conflicting_review_and_distinct_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            known_path = Path(temp) / "known.csv"
            known_path.write_text(
                "canonical_feature_name,category\nOne,test\nTwo,test\n",
                encoding="utf-8",
            )
            review_conflict = main([str(known_path), "--review-synonyms", "--distinct-only"])
            comparison_conflict = main([str(known_path), "--embed-attribute-comparison", "--distinct-only"])

        self.assertEqual(review_conflict, 1)
        self.assertEqual(comparison_conflict, 1)

    def test_cli_attribute_comparison_ensures_review_before_running(self):
        with tempfile.TemporaryDirectory() as temp:
            known_path = Path(temp) / "known.csv"
            known_path.write_text(
                "canonical_feature_name,category\nOne,test\nTwo,test\n",
                encoding="utf-8",
            )
            comparison_trace = {
                "combination_count": 15,
                "rankings": [{"combination": "name", "roc_auc": 1.0}],
                "outputs": {},
            }
            with mock.patch("known_feature_similarity.ensure_synonym_review_for_run") as ensure:
                with mock.patch(
                    "known_feature_similarity.run_attribute_comparison",
                    return_value=comparison_trace,
                ) as run_comparison:
                    result = main([str(known_path), "--embed-attribute-comparison"])

        self.assertEqual(result, 0)
        ensure.assert_called_once()
        run_comparison.assert_called_once()

    def test_distinct_only_run_does_not_require_review_or_write_synonym_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\ntwo,Two,test\n",
                encoding="utf-8",
            )
            trace = run_known_feature_similarity(
                known_set_path=known_path,
                annotated_features_path=None,
                text_fields=["feature_name"],
                output_dir=root / "out",
                prefix="known",
                cache_path=root / "cache.jsonl",
                client=DeterministicEmbeddingClient(),
                distinct_only=True,
            )
            artifact_names = {path.name for path in (root / "out").iterdir()}

        self.assertTrue(trace["distinct_only"])
        self.assertEqual(trace["counts"]["synonym_control_rows"], 0)
        self.assertFalse(any("synonym" in name for name in artifact_names))
        self.assertIn("known_most_similar_distinct_features.md", artifact_names)
        self.assertNotIn("known_feature_set_precision_recall_f1.png", artifact_names)
        self.assertNotIn("threshold_set_metrics", trace)
        self.assertNotIn("feature_set_metrics_png", trace["outputs"])
        self.assertNotIn("known_similarity_distribution_comparison.png", artifact_names)

    def test_standard_run_fails_before_embedding_when_review_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,test\ntwo,Two,test\n",
                encoding="utf-8",
            )
            client = DeterministicEmbeddingClient()
            with self.assertRaisesRegex(ValueError, "Synonym analysis requires"):
                run_known_feature_similarity(
                    known_set_path=known_path,
                    annotated_features_path=None,
                    text_fields=["feature_name"],
                    output_dir=root / "out",
                    prefix="known",
                    cache_path=root / "cache.jsonl",
                    client=client,
                )

            self.assertEqual(client.calls, [])
            self.assertFalse((root / "out").exists())

    def test_run_known_feature_similarity_scopes_cleanup_to_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            known_path.write_text(
                "feature_id,canonical_feature_name,category\none,One,persuasion\ntwo,Two,persuasion\n",
                encoding="utf-8",
            )
            client = DeterministicEmbeddingClient()
            cleanup = EmbeddingTextCleanup(
                strip_builtins=("persuasion_high_confidence_boilerplate",),
                min_chars=24,
                fallback="original",
            )

            trace = run_known_feature_similarity(
                known_set_path=known_path,
                annotated_features_path=None,
                text_fields=["feature_name", "definitions", "synonyms"],
                output_dir=root / "out",
                prefix="known",
                cache_path=root / "cache.jsonl",
                client=client,
                distinct_only=True,
                resolved_rows_override=[
                    {
                        "feature_key": "persuasion::one",
                        "feature_name": "A persuasive technique label",
                        "category": "persuasion",
                        "resolution": {"status": "test_override"},
                        "definitions": [
                            "A persuasive technique that repeats a slogan to increase salience."
                        ],
                        "synonyms": ["A persuasive technique synonym"],
                    },
                    {
                        "feature_key": "persuasion::two",
                        "feature_name": "Two",
                        "category": "persuasion",
                        "resolution": {"status": "test_override"},
                        "definitions": ["A persuasive tactic that uses urgent scarcity language."],
                        "synonyms": ["urgent appeal"],
                    },
                ],
                annotated_candidate_count_override=0,
                embedding_text_cleanup=cleanup,
                embedding_text_cleanup_fields=["definitions"],
            )

        joined_texts = "\n".join(text for call in client.calls for text in call)
        self.assertIn("Feature Name: A persuasive technique label", joined_texts)
        self.assertIn("Definitions: repeats a slogan to increase salience.", joined_texts)
        self.assertIn("Synonyms: A persuasive technique synonym", joined_texts)
        self.assertEqual(
            trace["embedding_text_cleanup"],
            {
                "strip_builtins": ["persuasion_high_confidence_boilerplate"],
                "min_chars": 24,
                "fallback": "original",
                "fields": ["definitions"],
                "applied_to_selected_fields": ["definitions"],
            },
        )

    def test_run_writes_artifacts_and_reuses_embedding_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known_path = root / "known.csv"
            annotated_path = root / "annotated.json"
            output_dir = root / "out"
            cache_path = root / "cache.jsonl"
            with known_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["feature_id", "canonical_feature_name", "aliases", "category"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "feature_id": "one",
                            "canonical_feature_name": "One",
                            "aliases": "First",
                            "category": "test",
                        },
                        {
                            "feature_id": "two",
                            "canonical_feature_name": "Two and Second",
                            "aliases": "Two|Second",
                            "category": "test",
                        },
                        {
                            "feature_id": "three",
                            "canonical_feature_name": "Three",
                            "aliases": "",
                            "category": "test",
                        },
                    ]
                )
            annotated_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_name": "One",
                                "category": "test",
                                "definitions": ["First definition"],
                            },
                            {
                                "feature_name": "Two",
                                "category": "test",
                                "definitions": ["Second definition"],
                            },
                            {
                                "feature_name": "Second",
                                "category": "test",
                                "definitions": ["Another definition"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            review_path = write_complete_review(
                root / "review.json",
                known_path,
                {
                    "One": ["First control", "Primary"],
                    "Two and Second": ["Dual", "Pair"],
                    "Three": ["Third", "Tertiary"],
                },
            )
            client = DeterministicEmbeddingClient()

            trace = run_known_feature_similarity(
                known_set_path=known_path,
                annotated_features_path=annotated_path,
                text_fields=["feature_name", "definitions", "synonyms"],
                output_dir=output_dir,
                prefix="known",
                cache_path=cache_path,
                client=client,
                synonym_review_path=review_path,
            )
            cached_trace = run_known_feature_similarity(
                known_set_path=known_path,
                annotated_features_path=annotated_path,
                text_fields=["feature_name", "definitions", "synonyms"],
                output_dir=output_dir,
                prefix="known",
                cache_path=cache_path,
                client=None,
                synonym_review_path=review_path,
            )

            expected_paths = {
                "known_resolved_features.json",
                "known_resolved_features.jsonl",
                "known_pairwise_similarities.csv",
                "known_pairwise_similarities.jsonl",
                "known_most_similar_distinct_features.md",
                "known_similarity_matrix.csv",
                "known_similarity_heatmap.png",
                "known_similarity_histogram.png",
                "known_similarity_boxplot.png",
                "known_synonym_control_rows.json",
                "known_synonym_control_rows.jsonl",
                "known_synonym_pairwise_similarities.csv",
                "known_synonym_pairwise_similarities.jsonl",
                "known_synonym_similarity_histogram.png",
                "known_synonym_similarity_boxplot.png",
                "known_similarity_distribution_comparison.png",
                "known_feature_set_precision_recall_f1.png",
                "known_similarity_trace.json",
            }
            actual_paths = {path.name for path in output_dir.iterdir()}
            resolved_payload = json.loads((output_dir / "known_resolved_features.json").read_text())
            pair_lines = (output_dir / "known_pairwise_similarities.jsonl").read_text().strip().splitlines()
            with (output_dir / "known_similarity_matrix.csv").open(newline="", encoding="utf-8") as handle:
                matrix_rows = list(csv.reader(handle))
            png_headers = [
                (output_dir / filename).read_bytes()[:8]
                for filename in (
                    "known_similarity_heatmap.png",
                    "known_similarity_histogram.png",
                    "known_similarity_boxplot.png",
                    "known_synonym_similarity_histogram.png",
                    "known_synonym_similarity_boxplot.png",
                    "known_similarity_distribution_comparison.png",
                    "known_feature_set_precision_recall_f1.png",
                )
            ]

        self.assertTrue(expected_paths.issubset(actual_paths))
        self.assertEqual(trace["counts"]["known_features"], 3)
        self.assertEqual(trace["counts"]["pairwise_similarities"], 3)
        self.assertEqual(trace["counts"]["synonym_control_groups"], 3)
        self.assertEqual(trace["counts"]["synonym_control_rows"], 9)
        self.assertEqual(trace["counts"]["synonym_pairwise_similarities"], 9)
        self.assertEqual(trace["synonym_similarity_summary"]["pair_count"], 9)
        self.assertIn("threshold_set_metrics", trace)
        self.assertEqual(trace["threshold_set_metrics"]["evaluation_unit"], "produced feature inventory")
        self.assertIn("feature_set_metrics_png", trace["outputs"])
        self.assertIn("most_similar_distinct_features_markdown", trace["outputs"])
        self.assertEqual(
            trace["synonym_similarity_summaries_by_pair_type"]["canonical_synonym"]["pair_count"],
            6,
        )
        self.assertEqual(
            trace["synonym_similarity_summaries_by_pair_type"]["synonym_synonym"]["pair_count"],
            3,
        )
        self.assertEqual(trace["resolution_counts"]["exact_canonical"], 1)
        self.assertEqual(trace["resolution_counts"]["alias_aggregate"], 1)
        self.assertEqual(trace["resolution_counts"]["known_metadata_fallback"], 1)
        self.assertEqual(trace["cache"]["misses"], 9)
        self.assertEqual(cached_trace["cache"]["hits"], 12)
        self.assertEqual(cached_trace["cache"]["misses"], 0)
        self.assertEqual(len(resolved_payload["features"]), 3)
        self.assertEqual(len(pair_lines), 3)
        self.assertEqual(len(matrix_rows), 4)
        self.assertTrue(all(header == b"\x89PNG\r\n\x1a\n" for header in png_headers))


if __name__ == "__main__":
    unittest.main()
