import csv
import json
import tempfile
import unittest
from pathlib import Path

from comparison_plots import (
    apply_semantic_matches,
    build_parser,
    final_feature_candidates,
    feature_union_rows,
    filter_feature_set_by_category,
    jaccard_series,
    linear_fit,
    load_feature_set,
    parse_final_k_overrides,
    parse_semantic_match_response,
    parse_category_specs,
    semantic_candidate_groups,
    scatter_fit_line,
    scatter_points,
    top_keys_by_category,
    unique_features_markdown,
    venn_counts,
    write_final_feature_set,
)


class ComparisonPlotsTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_jaccard_series_uses_top_k_by_paper_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.jsonl"
            second_path = tmp_path / "second.jsonl"
            self.write_jsonl(
                first_path,
                [
                    {"category": "persuasion", "feature_name": "Appeal", "paper_count": 10},
                    {"category": "persuasion", "feature_name": "Bandwagon", "paper_count": 5},
                    {"category": "persuasion", "feature_name": "Card stacking", "paper_count": 2},
                ],
            )
            self.write_jsonl(
                second_path,
                [
                    {"category": "persuasion", "feature_name": "Appeal", "paper_count": 8},
                    {"category": "persuasion", "feature_name": "Card stacking", "paper_count": 7},
                    {"category": "persuasion", "feature_name": "Distraction", "paper_count": 1},
                ],
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        self.assertEqual(
            jaccard_series(first, second, 1, 3),
            [
                (1, 1.0),
                (2, 1 / 3),
                (3, 0.5),
            ],
        )
        self.assertEqual(venn_counts(first, second, 2), (1, 1, 1))

    def test_top_k_is_balanced_per_category_for_overall_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.jsonl"
            second_path = tmp_path / "second.jsonl"
            self.write_jsonl(
                first_path,
                [
                    {"category": "persuasion", "feature_name": "Shared persuasion", "paper_count": 10},
                    {"category": "persuasion", "feature_name": "Set one persuasion", "paper_count": 9},
                    {"category": "moral_framing", "feature_name": "Shared moral", "paper_count": 1},
                ],
            )
            self.write_jsonl(
                second_path,
                [
                    {"category": "persuasion", "feature_name": "Shared persuasion", "paper_count": 8},
                    {"category": "moral_framing", "feature_name": "Shared moral", "paper_count": 7},
                    {"category": "moral_framing", "feature_name": "Set two moral", "paper_count": 6},
                ],
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        self.assertEqual(len(top_keys_by_category(first, 1)), 2)
        self.assertEqual(jaccard_series(first, second, 1, 1), [(1, 1.0)])
        self.assertEqual(venn_counts(first, second, 1), (0, 2, 0))

    def test_scatter_points_use_top_k_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "sentiment_affect", "feature_name": "Anger", "paper_count": 9},
                        {"category": "sentiment_affect", "feature_name": "Joy", "paper_count": 4},
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {"category": "sentiment_affect", "feature_name": "Anger", "paper_count": 7},
                            {"category": "sentiment_affect", "feature_name": "Fear", "paper_count": 3},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        points = {point.label: point for point in scatter_points(first, second, 2)}
        anger = points["Anger (sentiment_affect)"]
        self.assertEqual((anger.count_1, anger.count_2), (9, 7))
        self.assertEqual(set(points), {"Anger (sentiment_affect)"})

    def test_match_scope_name_ignores_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {
                            "category": "persuasion",
                            "feature_name": "Loaded Language",
                            "paper_count": 5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {
                            "category": "sentiment_affect",
                            "feature_name": "loaded-language",
                            "paper_count": 4,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "name")
            second = load_feature_set(second_path, "second", "name")

        self.assertEqual(venn_counts(first, second, 1), (0, 1, 0))

    def test_name_aliases_set_feature_set_labels(self):
        args = build_parser().parse_args(
            [
                "first.jsonl",
                "second.jsonl",
                "--name-1",
                "core set",
                "--name-2",
                "media attributes",
            ]
        )

        self.assertEqual(args.label_1, "core set")
        self.assertEqual(args.label_2, "media attributes")

    def test_scatter_labels_are_off_by_default(self):
        args = build_parser().parse_args(["first.jsonl", "second.jsonl"])

        self.assertEqual(args.annotate_top, 0)

    def test_linear_fit_returns_slope_intercept_and_r_value(self):
        fit = linear_fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

        self.assertIsNotNone(fit)
        slope, intercept, r_value = fit or (None, None, None)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 0.0)
        self.assertAlmostEqual(r_value or 0.0, 1.0)

    def test_scatter_fit_line_uses_log_transformed_values_for_log_scale(self):
        r_value, fit_xs, fit_ys = scatter_fit_line([0, 9, 99], [0, 99, 9999], "log")

        self.assertAlmostEqual(r_value or 0.0, 1.0)
        self.assertEqual(len(fit_xs), 2)
        self.assertEqual(len(fit_ys), 2)
        self.assertAlmostEqual(fit_xs[0], 1.0)
        self.assertAlmostEqual(fit_xs[1], 100.0)
        self.assertAlmostEqual(fit_ys[0], 1.0)
        self.assertAlmostEqual(fit_ys[1], 10000.0)

    def test_category_filter_uses_category_local_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Appeal", "paper_count": 9},
                        {"category": "moral_framing", "feature_name": "Care", "paper_count": 7},
                        {"category": "sentiment_affect", "feature_name": "Anger", "paper_count": 5},
                    ]
                ),
                encoding="utf-8",
            )

            feature_set = load_feature_set(first_path, "first", "category")
            specs = parse_category_specs("moral framing,sentiment")

        moral = filter_feature_set_by_category(feature_set, specs[0].key)
        sentiment = filter_feature_set_by_category(feature_set, specs[1].key)
        self.assertEqual([record.display_name for record in moral.records], ["Care"])
        self.assertEqual([record.display_name for record in sentiment.records], ["Anger"])

    def test_feature_union_rows_report_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Appeal", "paper_count": 9},
                        {"category": "persuasion", "feature_name": "Bandwagon", "paper_count": 5},
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Appeal", "paper_count": 7},
                        {"category": "persuasion", "feature_name": "Distraction", "paper_count": 3},
                    ]
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        rows = {row["feature_name"]: row for row in feature_union_rows(first, second)}
        self.assertEqual(rows["Appeal"]["membership"], "both")
        self.assertEqual(rows["Appeal"]["belongs_to"], ["first", "second"])
        self.assertEqual(rows["Bandwagon"]["membership"], "set_1_only")
        self.assertTrue(rows["Bandwagon"]["in_set_1"])
        self.assertFalse(rows["Bandwagon"]["in_set_2"])
        self.assertIsNone(rows["Bandwagon"]["set_2"])
        self.assertEqual(rows["Distraction"]["membership"], "set_2_only")
        self.assertIsNone(rows["Distraction"]["set_1"])

    def test_final_feature_candidates_use_category_peak_jaccard_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Only first high", "paper_count": 10},
                        {"category": "persuasion", "feature_name": "Shared", "paper_count": 9},
                        {"category": "persuasion", "feature_name": "Only first low", "paper_count": 1},
                        {"category": "moral_framing", "feature_name": "Care", "paper_count": 8},
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Only second high", "paper_count": 10},
                        {"category": "persuasion", "feature_name": "Shared", "paper_count": 9},
                        {"category": "persuasion", "feature_name": "Only second low", "paper_count": 1},
                        {"category": "moral_framing", "feature_name": "Care", "paper_count": 7},
                    ]
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        rows, summaries = final_feature_candidates(
            first,
            second,
            top_a=1,
            top_b=3,
            category_specs=parse_category_specs("persuasion,moral_framing"),
        )
        summaries_by_category = {summary["category_key"]: summary for summary in summaries}
        persuasion_summary = summaries_by_category["persuasion"]
        self.assertEqual(persuasion_summary["peak_k"], 2)
        self.assertAlmostEqual(persuasion_summary["peak_jaccard"], 1 / 3)
        self.assertEqual(persuasion_summary["top_k_union_count"], 3)
        self.assertEqual(summaries_by_category["moral framing"]["peak_k"], 1)

        persuasion_rows = {
            row["feature_name"]: row
            for row in rows
            if row["category_key"] == "persuasion"
        }
        self.assertEqual(set(persuasion_rows), {"Only first high", "Shared", "Only second high"})
        self.assertEqual(persuasion_rows["Shared"]["membership"], "both")
        self.assertTrue(persuasion_rows["Shared"]["selected_from_set_1_top_k"])
        self.assertTrue(persuasion_rows["Shared"]["selected_from_set_2_top_k"])
        self.assertEqual(persuasion_rows["Only first high"]["membership"], "set_1_only")
        self.assertFalse(persuasion_rows["Only first high"]["selected_from_set_2_top_k"])

    def test_final_feature_candidates_can_override_category_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "sentiment_affect", "feature_name": "Shared", "paper_count": 10},
                        {"category": "sentiment_affect", "feature_name": "Only first", "paper_count": 9},
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {"category": "sentiment_affect", "feature_name": "Shared", "paper_count": 10},
                        {"category": "sentiment_affect", "feature_name": "Only second", "paper_count": 9},
                    ]
                ),
                encoding="utf-8",
            )
            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        rows, summaries = final_feature_candidates(
            first,
            second,
            top_a=1,
            top_b=2,
            category_specs=parse_category_specs("sentiment"),
            final_k_overrides={"sentiment affect": 2},
        )

        self.assertEqual(summaries[0]["peak_k"], 1)
        self.assertEqual(summaries[0]["selection_k"], 2)
        self.assertEqual(summaries[0]["selection_k_source"], "manual_override")
        self.assertEqual({row["feature_name"] for row in rows}, {"Shared", "Only first", "Only second"})
        self.assertTrue(all(row["selection_k"] == 2 for row in rows))

    def test_parse_final_k_overrides_normalizes_category_aliases(self):
        self.assertEqual(
            parse_final_k_overrides(["sentiment=75", "moral_framing:40,persuasion=60"]),
            {
                "sentiment affect": 75,
                "moral framing": 40,
                "persuasion": 60,
            },
        )

    def test_write_final_feature_set_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            json_path = tmp_path / "final.json"
            csv_path = tmp_path / "final.csv"
            first_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Appeal", "paper_count": 5}]),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Appeal", "paper_count": 4}]),
                encoding="utf-8",
            )
            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

            written = write_final_feature_set(
                json_path,
                csv_path,
                first,
                second,
                "category",
                "singular",
                1,
                2,
                parse_category_specs("persuasion"),
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(written, [json_path, csv_path])
        self.assertEqual(payload["feature_count"], 1)
        self.assertEqual(payload["categories"][0]["peak_k"], 1)
        self.assertEqual(csv_rows[0]["feature_name"], "Appeal")
        self.assertIn("manual_decision", csv_rows[0])

    def test_singular_match_normalization_merges_simple_plurals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Slogan", "paper_count": 4}]),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Slogans", "paper_count": 3}]),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category", "singular")
            second = load_feature_set(second_path, "second", "category", "singular")

        rows = feature_union_rows(first, second)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["membership"], "both")
        self.assertEqual(rows[0]["set_1"]["feature_name"], "Slogan")
        self.assertEqual(rows[0]["set_2"]["feature_name"], "Slogans")

    def test_exact_match_normalization_keeps_simple_plurals_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Slogan", "paper_count": 4}]),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Slogans", "paper_count": 3}]),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category", "exact")
            second = load_feature_set(second_path, "second", "category", "exact")

        rows = feature_union_rows(first, second)
        self.assertEqual([row["membership"] for row in rows], ["set_1_only", "set_2_only"])

    def test_singular_match_normalization_merges_slash_and_and_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {
                            "category": "persuasion",
                            "feature_name": "Exaggeration/Minimization",
                            "paper_count": 4,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {
                            "category": "persuasion",
                            "feature_name": "Exaggeration and Minimization",
                            "paper_count": 3,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category", "singular")
            second = load_feature_set(second_path, "second", "category", "singular")

        rows = feature_union_rows(first, second)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["membership"], "both")

    def test_unique_features_markdown_reports_top_unique_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Shared", "paper_count": 10},
                        {"category": "persuasion", "feature_name": "Only first high", "paper_count": 8},
                        {"category": "persuasion", "feature_name": "Only first low", "paper_count": 2},
                    ]
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    [
                        {"category": "persuasion", "feature_name": "Shared", "paper_count": 9},
                        {"category": "persuasion", "feature_name": "Only second high", "paper_count": 7},
                        {"category": "persuasion", "feature_name": "Only second low", "paper_count": 1},
                    ]
                ),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")

        markdown = unique_features_markdown(first, second, "category", "singular", 1)
        self.assertIn("## Top 1 Only In first", markdown)
        self.assertIn("## Top 1 Only In second", markdown)
        self.assertIn("| 1 | Only first high | persuasion | 8 |", markdown)
        self.assertIn("| 1 | Only second high | persuasion | 7 |", markdown)
        self.assertNotIn("Only first low", markdown)
        self.assertNotIn("Only second low", markdown)
        self.assertNotIn("| Shared |", markdown)

    def test_semantic_match_response_can_rekey_set_2_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Doubt", "paper_count": 8}]),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Sowing Doubt", "paper_count": 5}]),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")
            groups = semantic_candidate_groups(first, second, match_scope="category", candidate_limit=3)

        response = json.dumps(
            {
                "matches": [
                    {
                        "set_2_id": groups[0]["set_2_id"],
                        "set_1_id": groups[0]["candidates"][0]["set_1_id"],
                        "confidence": 0.92,
                        "reason": "Both are doubt-based persuasion features.",
                    }
                ]
            }
        )
        matches = parse_semantic_match_response(response, groups, confidence_threshold=0.78)
        remapped_second = apply_semantic_matches(second, matches)
        rows = feature_union_rows(first, remapped_second)

        self.assertEqual(len(matches), 1)
        self.assertEqual(rows[0]["membership"], "both")
        self.assertEqual(rows[0]["set_1"]["feature_name"], "Doubt")
        self.assertEqual(rows[0]["set_2"]["feature_name"], "Sowing Doubt")

    def test_semantic_match_response_respects_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.json"
            second_path = tmp_path / "second.json"
            first_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Doubt", "paper_count": 8}]),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps([{"category": "persuasion", "feature_name": "Sowing Doubt", "paper_count": 5}]),
                encoding="utf-8",
            )

            first = load_feature_set(first_path, "first", "category")
            second = load_feature_set(second_path, "second", "category")
            groups = semantic_candidate_groups(first, second, match_scope="category", candidate_limit=3)

        response = json.dumps(
            {
                "matches": [
                    {
                        "set_2_id": groups[0]["set_2_id"],
                        "set_1_id": groups[0]["candidates"][0]["set_1_id"],
                        "confidence": 0.5,
                        "reason": "Maybe related.",
                    }
                ]
            }
        )

        self.assertEqual(parse_semantic_match_response(response, groups, confidence_threshold=0.78), [])


if __name__ == "__main__":
    unittest.main()
