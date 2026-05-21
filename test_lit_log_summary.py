import json
import tempfile
import unittest
from pathlib import Path

import lit_log_summary


class LitLogSummaryTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_summarizes_manifest_by_service_and_outcome(self):
        logs_dir = Path(tempfile.mkdtemp()) / "logs"
        self.write_json(
            logs_dir / "manifest.json",
            {
                "candidates": [
                    {
                        "title": "OpenAlex Full Text",
                        "source_apis": ["openalex"],
                        "abstract": "An abstract.",
                        "download": {"status": "downloaded"},
                    },
                    {
                        "title": "Shared Abstract",
                        "source_apis": ["semantic_scholar", "openalex"],
                        "download": {"status": "abstract_only"},
                    },
                    {
                        "title": "Crossref Failed",
                        "source_apis": ["crossref"],
                        "download": {"status": "failed"},
                    },
                    {
                        "title": "Europe PMC Existing",
                        "discovered_via": [{"source": "europe-pmc"}],
                        "download": {"status": "already_exists"},
                    },
                    {
                        "title": "Unknown Pending",
                        "download": {},
                    },
                ],
            },
        )

        summary = lit_log_summary.summarize_logs(logs_dir)

        self.assertEqual(summary["total_candidates"], 5)
        self.assertEqual(summary["totals"]["downloaded"], 2)
        self.assertEqual(summary["totals"]["failed"], 1)
        self.assertEqual(summary["totals"]["abstract_only"], 1)
        self.assertEqual(summary["totals"]["other"], 1)
        self.assertEqual(summary["totals"]["with_abstract"], 1)
        self.assertEqual(summary["status_counts"]["already_exists"], 1)
        self.assertEqual(summary["by_service"]["openalex"]["candidates"], 2)
        self.assertEqual(summary["by_service"]["openalex"]["downloaded"], 1)
        self.assertEqual(summary["by_service"]["openalex"]["abstract_only"], 1)
        self.assertEqual(summary["by_service"]["semantic_scholar"]["abstract_only"], 1)
        self.assertEqual(summary["by_service"]["europe_pmc"]["downloaded"], 1)
        self.assertEqual(summary["by_service"]["unknown"]["other"], 1)
        self.assertEqual(summary["service_combinations"]["openalex+semantic_scholar"], 1)

    def test_falls_back_to_download_log_records_for_candidate_cache(self):
        logs_dir = Path(tempfile.mkdtemp()) / "logs"
        self.write_json(
            logs_dir / "candidates.json",
            {
                "candidates": [
                    {
                        "title": "Failed Candidate",
                        "doi": "https://doi.org/10.1000/ABC",
                        "source_apis": ["Semantic Scholar"],
                    }
                ],
            },
        )
        self.write_json(
            logs_dir / "download_log.json",
            {
                "records": [
                    {
                        "identity": "doi:10.1000/abc",
                        "doi": "10.1000/abc",
                        "filename": "failed.pdf",
                        "status": "failed",
                        "download": {"status": "failed", "filename": "failed.pdf"},
                    }
                ]
            },
        )

        summary = lit_log_summary.summarize_logs(logs_dir)

        self.assertEqual(summary["total_candidates"], 1)
        self.assertEqual(summary["totals"]["failed"], 1)
        self.assertEqual(summary["by_service"]["semantic_scholar"]["failed"], 1)

    def test_cli_writes_output_file(self):
        tmp_dir = Path(tempfile.mkdtemp())
        logs_dir = tmp_dir / "logs"
        output_path = tmp_dir / "summary.json"
        self.write_json(
            logs_dir / "manifest.json",
            {
                "candidates": [
                    {
                        "title": "Paper",
                        "source_apis": ["crossref"],
                        "download": {"status": "downloaded"},
                    }
                ],
            },
        )

        result = lit_log_summary.main([str(logs_dir), "--output", str(output_path)])

        self.assertEqual(result, 0)
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["by_service"]["crossref"]["downloaded"], 1)


if __name__ == "__main__":
    unittest.main()
