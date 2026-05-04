import unittest

from lit_harvest import (
    Candidate,
    build_pdf_filename,
    build_search_queries,
    extract_keywords,
    merge_candidates,
    normalize_doi,
)


class LitHarvestCoreTests(unittest.TestCase):
    def test_extract_keywords_and_queries_from_markdown(self):
        draft = """# Robust Sensor Fusion for Autonomous Flight

        Abstract
        Robust sensor fusion improves autonomous flight in degraded visual
        environments. Sensor fusion methods combine radar, inertial, and visual
        odometry for navigation.
        """
        terms = extract_keywords(draft)
        queries = build_search_queries(terms, max_queries=5)

        self.assertEqual(terms.title, "Robust Sensor Fusion for Autonomous Flight")
        self.assertTrue(any("sensor fusion" in phrase for phrase in terms.noun_phrases))
        self.assertLessEqual(len(queries), 5)
        self.assertTrue(any("sensor" in query.text for query in queries))

    def test_normalize_doi(self):
        self.assertEqual(
            normalize_doi("https://doi.org/10.1145/1234.5678"),
            "10.1145/1234.5678",
        )
        self.assertEqual(normalize_doi("doi:10.1000/ABC."), "10.1000/abc")

    def test_merge_candidates_by_doi(self):
        first = Candidate(
            title="A Study of Sensor Fusion",
            doi="10.1000/test",
            source_apis=["openalex"],
            candidate_pdf_urls=["https://example.org/a.pdf"],
        )
        second = Candidate(
            title="A Study of Sensor Fusion",
            doi="10.1000/test",
            source_apis=["crossref"],
            candidate_pdf_urls=["https://example.org/b.pdf"],
        )

        merged = merge_candidates([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_apis, ["openalex", "crossref"])
        self.assertEqual(
            merged[0].candidate_pdf_urls,
            ["https://example.org/a.pdf", "https://example.org/b.pdf"],
        )

    def test_pdf_filename_is_safe_and_stable(self):
        candidate = Candidate(
            title="Sensor Fusion: Radar/Visual Navigation?",
            authors=["Ada Lovelace"],
            year=2024,
            doi="10.1000/test",
        )
        filename = build_pdf_filename(candidate)

        self.assertTrue(filename.endswith(".pdf"))
        self.assertIn("2024_lovelace_sensor_fusion_radar_visual_navigation", filename)
        self.assertNotIn("/", filename)


if __name__ == "__main__":
    unittest.main()
