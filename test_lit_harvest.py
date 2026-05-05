import unittest

from pathlib import Path
import tempfile

from lit_harvest import (
    Candidate,
    HttpClient,
    PdfDownloader,
    build_pdf_filename,
    build_search_queries,
    candidate_matches_keywords,
    extract_pdf_urls_from_html,
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

    def test_extract_pdf_urls_from_html(self):
        html_text = '''
            <html>
              <head>
                <meta name="citation_pdf_url" content="/papers/1234.pdf">
                <link rel="alternate" type="application/pdf" href="https://example.org/full.pdf">
              </head>
              <body>
                <a href="/papers/5678.pdf">Download PDF</a>
                <a href="https://example.org/files/report.PDF">Full paper</a>
                <iframe src="https://example.org/embed.pdf"></iframe>
                <a href="javascript:void(0)">Not a pdf</a>
                <a href="/download?type=pdf">PDF Link</a>
              </body>
            </html>
        '''
        urls = extract_pdf_urls_from_html(html_text, "https://example.com/path/page.html")

        self.assertIn("https://example.com/papers/1234.pdf", urls)
        self.assertIn("https://example.org/full.pdf", urls)
        self.assertIn("https://example.com/papers/5678.pdf", urls)
        self.assertIn("https://example.org/files/report.PDF", urls)
        self.assertIn("https://example.org/embed.pdf", urls)
        self.assertIn("https://example.com/download?type=pdf", urls)

    def test_candidate_matches_keywords(self):
        candidate = Candidate(
            title="Autonomous Sensor Fusion for Spacecraft",
            abstract="This work shows robust sensor fusion techniques for aerospace.",
            venue="Journal of Autonomous Systems",
        )
        matches = candidate_matches_keywords(
            candidate,
            ["sensor", "fusion", "autonomous", "spacecraft", "robotics"],
        )

        self.assertCountEqual(matches, ["sensor", "fusion", "autonomous", "spacecraft"])

    def test_merge_candidates_preserves_matched_keywords(self):
        first = Candidate(
            title="A Study",
            matched_keywords=["sensor", "fusion"],
            source_apis=["openalex"],
        )
        second = Candidate(
            title="A Study",
            matched_keywords=["fusion", "autonomous"],
            source_apis=["crossref"],
        )
        merged = merge_candidates([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].matched_keywords, ["sensor", "fusion", "autonomous"])

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

    def test_download_generates_pdf_from_abstract_when_no_pdf_urls(self):
        candidate = Candidate(
            title="Abstract-Only Paper",
            authors=["Test Author"],
            abstract="This is the abstract text.",
        )
        downloader = PdfDownloader(HttpClient(), Path(tempfile.mkdtemp()))

        result = downloader.download(candidate, [])

        self.assertEqual(result["status"], "abstract_only")
        self.assertTrue(result["path"].endswith(".pdf"))
        self.assertTrue(Path(result["path"]).exists())
        self.assertEqual(candidate.download["status"], "abstract_only")


if __name__ == "__main__":
    unittest.main()
