import json
import tempfile
import unittest
from pathlib import Path

from lit_extract import canonical_category, normalize_feature_item, run_extraction
from lit_taxonomy import load_taxonomy, taxonomy_category_lookup, validate_taxonomy


TAXONOMY = {
    "version": 1,
    "title": "Aviation Human Factors",
    "families": [
        {
            "id": "crew_coordination",
            "label": "Crew coordination",
            "description": "Coordination concepts.",
            "aliases": ["team coordination"],
        },
        {
            "id": "workload",
            "label": "Workload",
            "description": "Cognitive and task workload.",
            "aliases": [],
        },
    ],
}


class TaxonomyTests(unittest.TestCase):
    def test_validate_and_load_taxonomy(self):
        validated = validate_taxonomy(TAXONOMY)
        self.assertEqual(validated["families"][0]["id"], "crew_coordination")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "taxonomy.json"
            path.write_text(json.dumps(TAXONOMY), encoding="utf-8")
            self.assertEqual(load_taxonomy(path), validated)

    def test_rejects_duplicate_aliases_across_families(self):
        payload = json.loads(json.dumps(TAXONOMY))
        payload["families"][1]["aliases"] = ["Crew coordination"]
        with self.assertRaises(ValueError):
            validate_taxonomy(payload)

    def test_dynamic_category_lookup_accepts_label_alias_and_id(self):
        lookup = taxonomy_category_lookup(TAXONOMY)
        self.assertEqual(canonical_category("Crew coordination", lookup), "crew_coordination")
        self.assertEqual(canonical_category("team coordination", lookup), "crew_coordination")
        self.assertEqual(canonical_category("workload", lookup), "workload")
        self.assertIsNone(canonical_category("unknown", lookup))

    def test_dynamic_normalization_skips_legacy_relevance_filter(self):
        lookup = taxonomy_category_lookup(TAXONOMY)
        row = normalize_feature_item(
            {"category": "Workload", "feature_name": "Task saturation", "confidence": 0.8},
            paper={"paper_id": "p1", "title": "Flight Study"},
            chunk={"chunk_id": "p1:pdf", "text": ""},
            run_id="run",
            model="fake",
            extracted_at="now",
            taxonomy_lookup=lookup,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["category"], "workload")

    def test_dynamic_taxonomy_dry_run_uses_input_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            run_dir.mkdir()
            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            (run_dir / "paper_index.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "p1",
                                "title": "A Paper",
                                "pdf_path": str(pdf_path),
                                "conversion": {"status": "direct_pdf_ready"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            taxonomy_path = root / "taxonomy.json"
            taxonomy_path.write_text(json.dumps(TAXONOMY), encoding="utf-8")

            trace = run_extraction(
                run_dir=run_dir,
                output_dir=root / "extract",
                api_key=None,
                model="fake",
                base_url="https://example.test",
                api_provider="openai",
                reasoning_effort=None,
                input_mode="pdf",
                batch_chunks_count=1,
                paper_limit=None,
                taxonomy_path=taxonomy_path,
                force=False,
                dry_run=True,
                max_chunk_chars=100,
                temperature=0.0,
                show_progress=False,
            )

            self.assertEqual(trace["config"]["paper_ranking"], "input_order")


if __name__ == "__main__":
    unittest.main()
