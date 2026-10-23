import csv
import json
import tempfile
import unittest
from pathlib import Path

from lit_review.embeddings.core import embedding_cache_key, model_resource_name
from lit_review.analysis.wikidata import main, run_wikidata_mapping_analysis


MAPPED_FIELDNAMES = [
    "input_record_id",
    "feature_name",
    "proposal_mode",
    "verified_qids",
    "verified_wikidata_labels",
    "mapping_types",
    "proposal_confidences",
    "proposal_rationales",
    "verification_warnings",
    "qid_suggestion_status",
]


class WikidataMappingAnalysisTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def write_mapped_csv(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MAPPED_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def fixture_paths(self, root: Path) -> tuple[Path, Path]:
        scored_path = root / "scored_features.jsonl"
        mapped_path = root / "wikidata_mapped_features.csv"
        self.write_jsonl(
            scored_path,
            [
                {
                    "feature_key": "persuasion::band-wagon",
                    "category": "persuasion",
                    "feature_name": "Band Wagon",
                    "normalized_feature_name": "band wagon",
                    "score": {"total": 5.0},
                    "paper_count": 10,
                    "mention_count": 11,
                },
                {
                    "feature_key": "persuasion::high-missing",
                    "category": "persuasion",
                    "feature_name": "High Missing",
                    "normalized_feature_name": "high missing",
                    "score": {"total": 4.0},
                    "paper_count": 9,
                    "mention_count": 9,
                },
                {
                    "feature_key": "persuasion::appeal-to-fear",
                    "category": "persuasion",
                    "feature_name": "Appeal to Fear",
                    "normalized_feature_name": "appeal to fear",
                    "score": {"total": 3.0},
                    "paper_count": 3,
                    "mention_count": 4,
                },
                {
                    "feature_key": "moral_framing::appeal-to-fear",
                    "category": "moral_framing",
                    "feature_name": "Appeal to Fear",
                    "normalized_feature_name": "appeal to fear",
                    "score": {"total": 6.0},
                    "paper_count": 12,
                    "mention_count": 12,
                },
            ],
        )
        self.write_mapped_csv(
            mapped_path,
            [
                {
                    "input_record_id": "input_appeal_to_fear",
                    "feature_name": "Appeal-To Fear",
                    "proposal_mode": "alternatives",
                    "verified_qids": "Q1\nQ2",
                    "verified_wikidata_labels": "fear appeal\nshared topic",
                    "mapping_types": "exact\nrelated",
                    "proposal_confidences": "0.95\n0.55",
                    "proposal_rationales": "direct match\nloose overlap",
                    "verification_warnings": "missing_official_description",
                    "qid_suggestion_status": "verified_suggestions",
                },
                {
                    "input_record_id": "input_band_wagon",
                    "feature_name": "Band Wagon",
                    "proposal_mode": "single",
                    "verified_qids": "Q2",
                    "verified_wikidata_labels": "shared topic",
                    "mapping_types": "close",
                    "proposal_confidences": "0.75",
                    "proposal_rationales": "related propaganda concept",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
                {
                    "input_record_id": "input_band_wagon_duplicate",
                    "feature_name": "Band-Wagon",
                    "proposal_mode": "single",
                    "verified_qids": "Q3",
                    "verified_wikidata_labels": "bandwagon effect",
                    "mapping_types": "exact",
                    "proposal_confidences": "0.9",
                    "proposal_rationales": "canonical label",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
                {
                    "input_record_id": "input_unselected",
                    "feature_name": "Unselected Mapping",
                    "proposal_mode": "single",
                    "verified_qids": "Q9",
                    "verified_wikidata_labels": "outside category",
                    "mapping_types": "exact",
                    "proposal_confidences": "1.0",
                    "proposal_rationales": "not in selected scored category",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
            ],
        )
        return mapped_path, scored_path

    def embedding_fixture_paths(self, root: Path) -> tuple[Path, Path, Path]:
        scored_path = root / "scored_features.jsonl"
        mapped_path = root / "wikidata_mapped_features.csv"
        features_path = root / "embedding_features.json"
        cache_path = root / "embedding_cache.jsonl"
        trace_path = root / "embedding_merge_trace.json"
        model = "gemini-embedding-001"
        task_type = "CLUSTERING"
        output_dimensionality = 2

        scored_rows = [
            {
                "feature_key": "persuasion::alpha",
                "category": "persuasion",
                "feature_name": "Alpha Appeal",
                "normalized_feature_name": "alpha appeal",
                "score": {"total": 5.0},
                "paper_count": 5,
                "mention_count": 5,
            },
            {
                "feature_key": "persuasion::beta",
                "category": "persuasion",
                "feature_name": "Beta Appeal",
                "normalized_feature_name": "beta appeal",
                "score": {"total": 4.0},
                "paper_count": 4,
                "mention_count": 4,
            },
            {
                "feature_key": "persuasion::gamma",
                "category": "persuasion",
                "feature_name": "Gamma Appeal",
                "normalized_feature_name": "gamma appeal",
                "score": {"total": 3.0},
                "paper_count": 3,
                "mention_count": 3,
            },
        ]
        self.write_jsonl(scored_path, scored_rows)
        features_path.write_text(json.dumps({"features": scored_rows}), encoding="utf-8")
        self.write_mapped_csv(
            mapped_path,
            [
                {
                    "input_record_id": "input_alpha",
                    "feature_name": "Alpha Appeal",
                    "proposal_mode": "components",
                    "verified_qids": "Q1\nQ2",
                    "verified_wikidata_labels": "Article One\nArticle Two",
                    "mapping_types": "exact\nclose",
                    "proposal_confidences": "1.0\n0.9",
                    "proposal_rationales": "direct\nnear",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
                {
                    "input_record_id": "input_beta",
                    "feature_name": "Beta Appeal",
                    "proposal_mode": "components",
                    "verified_qids": "Q1\nQ2",
                    "verified_wikidata_labels": "Article One\nArticle Two",
                    "mapping_types": "related\nclose",
                    "proposal_confidences": "0.7\n0.85",
                    "proposal_rationales": "loose\nnear",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
                {
                    "input_record_id": "input_gamma",
                    "feature_name": "Gamma Appeal",
                    "proposal_mode": "single",
                    "verified_qids": "Q3",
                    "verified_wikidata_labels": "Article Three",
                    "mapping_types": "exact",
                    "proposal_confidences": "1.0",
                    "proposal_rationales": "direct",
                    "verification_warnings": "",
                    "qid_suggestion_status": "verified_suggestions",
                },
            ],
        )

        embeddings = {
            "Feature Name: Alpha Appeal": [1.0, 0.0],
            "Feature Name: Beta Appeal": [0.8, 0.6],
            "Feature Name: Gamma Appeal": [0.0, 1.0],
        }
        cache_entries = []
        for text, embedding in embeddings.items():
            cache_entries.append(
                {
                    "cache_key": embedding_cache_key(
                        text,
                        model=model,
                        task_type=task_type,
                        output_dimensionality=output_dimensionality,
                    ),
                    "model": model_resource_name(model),
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality,
                    "text": text,
                    "embedding": embedding,
                }
            )
        cache_path.write_text(
            "\n".join(json.dumps(entry, sort_keys=True) for entry in cache_entries) + "\n",
            encoding="utf-8",
        )
        trace_path.write_text(
            json.dumps(
                {
                    "input": str(features_path),
                    "cache_path": str(cache_path),
                    "model": model,
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality,
                    "category_config": {
                        "persuasion": {
                            "text_fields": ["feature_name"],
                            "threshold": 0.9,
                            "linkage": "complete",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return mapped_path, scored_path, trace_path

    def test_analysis_outputs_coverage_candidates_and_audit_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapped_path, scored_path = self.fixture_paths(root)
            output_dir = root / "analysis"
            trace = run_wikidata_mapping_analysis(
                mapped_csv=mapped_path,
                scored_jsonl=scored_path,
                output_dir=output_dir,
                category="persuasion",
                top_k_values=(1, 2, 3),
                low_confidence_threshold=0.8,
            )

            self.assertEqual(trace["counts"]["selected_features"], 3)
            self.assertEqual(trace["counts"]["covered_features"], 2)
            self.assertEqual(trace["counts"]["matched_mapped_rows"], 3)
            self.assertEqual(trace["counts"]["candidate_mappings"], 4)
            self.assertEqual(trace["counts"]["duplicate_mapped_names"], 1)
            self.assertEqual(trace["counts"]["reused_qids"], 1)
            self.assertEqual(
                [
                    (row["top_k"], row["covered_feature_count"], row["feature_count"])
                    for row in trace["coverage"]["top_k"]
                ],
                [(1, 1, 1), (2, 1, 2), (3, 2, 3)],
            )

            coverage_rows = self.read_csv(output_dir / "persuasion_wikidata_mapping_feature_coverage.csv")
            by_name = {row["normalized_feature_name"]: row for row in coverage_rows}
            self.assertEqual(by_name["band wagon"]["matched_mapped_row_count"], "2")
            self.assertEqual(by_name["band wagon"]["candidate_count"], "2")
            self.assertEqual(by_name["high missing"]["mapped"], "no")

            mapping_payload = json.loads(
                (output_dir / "persuasion_wikidata_mapping_feature_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping_payload["category"], "persuasion")
            self.assertEqual(mapping_payload["summary"]["selected_scored_features"], 3)
            self.assertTrue(
                all(feature["category"] == "persuasion" for feature in mapping_payload["features"])
            )
            payload_by_name = {
                feature["normalized_feature_name"]: feature for feature in mapping_payload["features"]
            }
            self.assertTrue(payload_by_name["appeal to fear"]["mapping_status"]["mapped"])
            self.assertEqual(
                [
                    candidate["qid"]
                    for row in payload_by_name["appeal to fear"]["mapped_rows"]
                    for candidate in row["candidate_mappings"]
                ],
                ["Q1", "Q2"],
            )

            candidate_rows = self.read_csv(output_dir / "persuasion_wikidata_mapping_candidate_mappings.csv")
            appeal_candidates = [
                row for row in candidate_rows if row["normalized_feature_name"] == "appeal to fear"
            ]
            self.assertEqual([row["qid"] for row in appeal_candidates], ["Q1", "Q2"])
            self.assertEqual([row["mapping_type"] for row in appeal_candidates], ["exact", "related"])

            qid_rows = self.read_csv(output_dir / "persuasion_wikidata_mapping_qid_reuse.csv")
            qid_by_id = {row["qid"]: row for row in qid_rows}
            self.assertEqual(qid_by_id["Q2"]["feature_count"], "2")
            self.assertIn("Appeal-To Fear", qid_by_id["Q2"]["feature_names"])
            self.assertIn("Band Wagon", qid_by_id["Q2"]["feature_names"])

            audit_rows = self.read_csv(output_dir / "persuasion_wikidata_mapping_audit_flags.csv")
            flag_types = {row["flag_type"] for row in audit_rows}
            self.assertTrue(
                {
                    "low_confidence",
                    "non_exact",
                    "warning",
                    "duplicate_mapped_name",
                    "reused_qid",
                    "unmapped_top_ranked",
                }.issubset(flag_types)
            )
            self.assertTrue(
                any(row["flag_type"] == "low_confidence" and row["qid"] == "Q2" for row in audit_rows)
            )
            self.assertTrue(
                any(
                    row["flag_type"] == "unmapped_top_ranked" and row["feature_name"] == "High Missing"
                    for row in audit_rows
                )
            )

            report = (output_dir / "persuasion_wikidata_mapping_report.md").read_text(encoding="utf-8")
            self.assertIn("Wikidata Mapping Analysis: persuasion", report)
            self.assertIn("covered scored features", report)
            self.assertIn("High Missing", report)

    def test_cli_writes_report_csvs_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapped_path, scored_path = self.fixture_paths(root)
            output_dir = root / "cli-output"

            result = main(
                [
                    str(mapped_path),
                    str(scored_path),
                    "--category",
                    "persuasion",
                    "--output-dir",
                    str(output_dir),
                    "--top-k",
                    "1,2,3",
                    "--low-confidence-threshold",
                    "0.8",
                ]
            )

            self.assertEqual(result, 0)
            for name in [
                "persuasion_wikidata_mapping_report.md",
                "persuasion_wikidata_mapping_feature_coverage.csv",
                "persuasion_wikidata_mapping_feature_mapping.json",
                "persuasion_wikidata_mapping_candidate_mappings.csv",
                "persuasion_wikidata_mapping_qid_reuse.csv",
                "persuasion_wikidata_mapping_audit_flags.csv",
                "persuasion_wikidata_mapping_coverage_topk.png",
                "persuasion_wikidata_mapping_mapping_types.png",
                "persuasion_wikidata_mapping_confidence_histogram.png",
                "persuasion_wikidata_mapping_top_reused_qids.png",
            ]:
                path = output_dir / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 0, name)

    def test_embedding_similarity_outputs_use_cache_and_closeness_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapped_path, scored_path, trace_path = self.embedding_fixture_paths(root)
            output_dir = root / "analysis"

            trace = run_wikidata_mapping_analysis(
                mapped_csv=mapped_path,
                scored_jsonl=scored_path,
                output_dir=output_dir,
                category="persuasion",
                top_k_values=(1, 2, 3),
                embedding_merge_trace=trace_path,
            )

            self.assertEqual(trace["embedding_similarity"]["selected_embedded_features"], 3)
            self.assertEqual(trace["embedding_similarity"]["mapped_embedded_features"], 3)
            self.assertEqual(trace["embedding_similarity"]["same_article_pairs"], 1)
            self.assertEqual(trace["embedding_similarity"]["different_article_pairs"], 2)
            self.assertEqual(trace["embedding_similarity"]["embedding_text_fields"], ["feature_name"])
            self.assertEqual(trace["embedding_similarity"]["cache"]["misses"], 0)
            self.assertEqual(trace["embedding_similarity"]["cache"]["api_batches"], 0)

            pair_rows = self.read_csv(
                output_dir / "persuasion_wikidata_mapping_embedding_pair_similarities.csv"
            )
            self.assertEqual(len(pair_rows), 3)
            same_article = [row for row in pair_rows if row["relationship"] == "same_article"]
            self.assertEqual(len(same_article), 1)
            self.assertEqual(same_article[0]["shared_qids"], "Q1 | Q2")
            self.assertEqual(same_article[0]["shared_wikidata_labels"], "Article One | Article Two")
            self.assertEqual(same_article[0]["pair_mapping_closeness"], "close")
            self.assertAlmostEqual(float(same_article[0]["similarity"]), 0.8)

            summary_rows = self.read_csv(
                output_dir / "persuasion_wikidata_mapping_embedding_similarity_summary.csv"
            )
            summary = {(row["group_type"], row["group"]): row for row in summary_rows}
            self.assertEqual(summary[("relationship", "same_article")]["pair_count"], "1")
            self.assertEqual(summary[("relationship", "different_article")]["pair_count"], "2")
            self.assertAlmostEqual(float(summary[("relationship", "same_article")]["mean_similarity"]), 0.8)
            self.assertAlmostEqual(
                float(summary[("relationship", "different_article")]["mean_similarity"]), 0.3
            )
            self.assertEqual(summary[("pair_mapping_closeness", "close")]["pair_count"], "1")

            report = (output_dir / "persuasion_wikidata_mapping_report.md").read_text(encoding="utf-8")
            self.assertIn("## Embedding Similarity", report)
            self.assertIn("embedding text fields", report)
            self.assertIn("same-article pairs", report)

            for name in [
                "persuasion_wikidata_mapping_embedding_similarity_relationship_bar.png",
                "persuasion_wikidata_mapping_embedding_similarity_relationship_zscore_bar.png",
                "persuasion_wikidata_mapping_embedding_similarity_relationship_boxplot.png",
                "persuasion_wikidata_mapping_embedding_similarity_relationship_overlay_histogram.png",
                "persuasion_wikidata_mapping_embedding_similarity_closeness_bar.png",
                "persuasion_wikidata_mapping_embedding_similarity_closeness_zscore_bar.png",
            ]:
                path = output_dir / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
