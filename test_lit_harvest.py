import unittest

from pathlib import Path
import tempfile
from unittest import mock

import lit_harvest
from lit_harvest import (
    Candidate,
    DownloadLogLedger,
    HttpClient,
    PdfDownloader,
    PdfResolver,
    WebPdfSearcher,
    WebSearchProvider,
    WebSearchResult,
    build_pdf_filename,
    build_search_queries,
    build_web_pdf_queries,
    combine_search_queries,
    configured_search_queries,
    candidate_matches_keywords,
    apply_download_log_resume,
    download_candidates,
    extract_pdf_urls_from_html,
    extract_keywords,
    inferred_pdf_urls,
    is_saved_document_status,
    load_candidate_cache,
    merge_candidates,
    normalize_doi,
    write_candidate_cache,
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
                <meta content="/papers/1234.pdf" name="citation_pdf_url">
                <link href="https://example.org/full.pdf" type="application/pdf" rel="alternate">
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

    def test_extract_pdf_urls_from_embedded_json_and_pdf_paths(self):
        html_text = r'''
            <script>
              window.paper = {"pdfUrl":"https:\/\/repo.example.org\/paper\/full.pdf"};
            </script>
            <a href="/doi/pdf/10.1000/example">Publisher PDF</a>
            <div data-pdf-url="/article-pdf/12345">PDF</div>
        '''
        urls = extract_pdf_urls_from_html(html_text, "https://publisher.example.org/article")

        self.assertIn("https://repo.example.org/paper/full.pdf", urls)
        self.assertIn("https://publisher.example.org/doi/pdf/10.1000/example", urls)
        self.assertIn("https://publisher.example.org/article-pdf/12345", urls)

    def test_resolve_landing_page_uses_redirect_final_url_as_base(self):
        class FakeHttp:
            def request_bytes(self, url, params=None, headers=None):
                return (
                    b'<html><a href="/article/123/full.pdf">PDF</a></html>',
                    {"Content-Type": "text/html"},
                    "https://publisher.example.org/article/123",
                )

        resolver = PdfResolver(FakeHttp())
        urls = resolver._resolve_landing_page("https://doi.org/10.1000/example")

        self.assertEqual(urls, ["https://publisher.example.org/article/123/full.pdf"])

    def test_inferred_pdf_urls_from_common_open_identifiers(self):
        candidate = Candidate(
            title="Open Identifiers",
            doi="10.48550/arxiv.2301.12345",
            source_ids={
                "semantic_scholar:ACL": "P19-1021",
                "semantic_scholar:PubMedCentral": "PMC1234567",
            },
        )

        urls = inferred_pdf_urls(candidate)

        self.assertIn("https://arxiv.org/pdf/2301.12345.pdf", urls)
        self.assertIn("https://aclanthology.org/P19-1021.pdf", urls)
        self.assertIn("https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/", urls)

    def test_build_web_pdf_queries_uses_doi_and_exact_title(self):
        candidate = Candidate(
            title='A "Very" Specific Paper',
            doi="10.1000/example",
            authors=["Ada Lovelace"],
        )

        queries = build_web_pdf_queries(candidate, max_queries=3)

        self.assertEqual(queries[0], '"10.1000/example" pdf')
        self.assertIn('"A Very Specific Paper" filetype:pdf', queries)
        self.assertIn('"A Very Specific Paper" pdf', queries)

    def test_web_pdf_searcher_splits_pdf_and_landing_results(self):
        class FakeProvider(WebSearchProvider):
            name = "fake"

            def search(self, query, max_results):
                return [
                    WebSearchResult(
                        source=self.name,
                        query=query,
                        rank=1,
                        url="https://repo.example.org/very-specific-paper.pdf",
                        title="A Very Specific Paper",
                    ),
                    WebSearchResult(
                        source=self.name,
                        query=query,
                        rank=2,
                        url="https://authors.example.org/very-specific-paper",
                        title="A Very Specific Paper project page",
                    ),
                    WebSearchResult(
                        source=self.name,
                        query=query,
                        rank=3,
                        url="https://irrelevant.example.org/file.pdf",
                        title="An unrelated result",
                    ),
                ]

        candidate = Candidate(title="A Very Specific Paper")
        searcher = WebPdfSearcher([FakeProvider()], max_results=5, queries_per_candidate=1)

        pdf_urls, landing_urls, attempts = searcher.resolve(candidate)

        self.assertEqual(pdf_urls, ["https://repo.example.org/very-specific-paper.pdf"])
        self.assertEqual(landing_urls, ["https://authors.example.org/very-specific-paper"])
        self.assertEqual(attempts[0]["source"], "web_search:fake")
        self.assertEqual(attempts[0]["accepted_count"], 2)

    def test_candidate_cache_round_trips_post_search_data(self):
        tmp_dir = Path(tempfile.mkdtemp())
        draft_path = tmp_dir / "draft.md"
        draft_text = "# Cache Test\n\nsensor fusion methods"
        draft_path.write_text(draft_text, encoding="utf-8")
        cache_path = tmp_dir / "logs" / "candidates.json"
        extracted = extract_keywords(draft_text)
        queries = build_search_queries(extracted, max_queries=2)
        candidate = Candidate(
            title="A Study of Sensor Fusion",
            authors=["Ada Lovelace"],
            year=2024,
            doi="10.1000/test",
            source_apis=["openalex"],
            candidate_pdf_urls=["https://example.org/paper.pdf"],
            candidate_landing_page_urls=["https://example.org/paper"],
            relevance_score=0.42,
        )

        write_candidate_cache(
            cache_path,
            draft_path=draft_path,
            draft_text=draft_text,
            config={"draft": str(draft_path), "output": str(tmp_dir)},
            extracted=extracted,
            queries=queries,
            search_errors=[],
            keyword_search_counts={},
            candidates=[candidate],
        )
        loaded = load_candidate_cache(cache_path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["candidate_count"], 1)
        self.assertEqual(loaded["candidates"][0].title, "A Study of Sensor Fusion")
        self.assertEqual(loaded["candidates"][0].doi, "10.1000/test")
        self.assertEqual(loaded["candidates"][0].candidate_pdf_urls, ["https://example.org/paper.pdf"])
        self.assertEqual(loaded["queries"][0].text, queries[0].text)

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

    def test_download_pdf_does_not_retry_html_response(self):
        class FakeResponse:
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return b"<html>not a pdf</html>"

            def geturl(self):
                return "https://example.org/not-pdf"

        destination = Path(tempfile.mkdtemp()) / "paper.pdf"
        client = HttpClient(retries=4, backoff=0, rate_limit_delay=0)

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
            with self.assertRaises(RuntimeError):
                client.download_pdf(
                    "https://example.org/not-pdf",
                    destination,
                    max_bytes=1024 * 1024,
                )

        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(destination.exists())

    def test_download_candidates_runs_multiple_candidates(self):
        class FakeResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                return ["https://example.org/paper.pdf"]

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False

            def download(self, candidate, urls):
                result = {
                    "status": "downloaded",
                    "url": urls[0],
                    "filename": f"{candidate.title}.pdf",
                    "attempts": [{"url": urls[0], "status": "downloaded"}],
                }
                candidate.download = result
                return result

        candidates = [Candidate(title=f"Paper {idx}") for idx in range(4)]

        count = download_candidates(
            candidates,
            resolver=FakeResolver(),
            downloader=FakeDownloader(Path(tempfile.mkdtemp())),
            max_downloads=None,
            max_web_search_candidates=None,
            download_workers=2,
        )

        self.assertEqual(count, 4)
        self.assertTrue(all(candidate.download["status"] == "downloaded" for candidate in candidates))

    def test_download_candidates_preflights_existing_files(self):
        class FailingResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                raise AssertionError("existing files should not be resolved")

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False

            def download(self, candidate, urls):
                raise AssertionError("existing files should not be downloaded")

        output_dir = Path(tempfile.mkdtemp())
        candidate = Candidate(title="Already Saved", authors=["Ada Lovelace"], year=2024)
        existing_path = output_dir / build_pdf_filename(candidate)
        existing_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        count = download_candidates(
            [candidate],
            resolver=FailingResolver(),
            downloader=FakeDownloader(output_dir),
            max_downloads=None,
            max_web_search_candidates=None,
            download_workers=4,
        )

        self.assertEqual(count, 1)
        self.assertEqual(candidate.download["status"], "already_exists")
        self.assertEqual(candidate.download["path"], str(existing_path))

    def test_download_candidates_updates_download_log_after_each_candidate(self):
        class FakeResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                return ["https://example.org/paper.pdf"]

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False

            def download(self, candidate, urls):
                result = {
                    "status": "downloaded",
                    "url": urls[0],
                    "filename": build_pdf_filename(candidate),
                    "attempts": [{"url": urls[0], "status": "downloaded"}],
                }
                candidate.download = result
                return result

        tmp_dir = Path(tempfile.mkdtemp())
        log_path = tmp_dir / "logs" / "download_log.json"
        ledger = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        candidates = [Candidate(title="First Paper"), Candidate(title="Second Paper")]

        count = download_candidates(
            candidates,
            resolver=FakeResolver(),
            downloader=FakeDownloader(tmp_dir),
            max_downloads=None,
            max_web_search_candidates=None,
            download_workers=1,
            ledger=ledger,
        )

        self.assertEqual(count, 2)
        self.assertTrue(log_path.exists())
        loaded = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        loaded.load()
        self.assertIsNotNone(loaded.record_for_candidate(candidates[0]))
        self.assertIsNotNone(loaded.record_for_candidate(candidates[1]))

    def test_download_log_resume_skips_failed_candidates(self):
        tmp_dir = Path(tempfile.mkdtemp())
        log_path = tmp_dir / "logs" / "download_log.json"
        candidate = Candidate(title="Failed Paper", doi="10.1000/fail")
        ledger = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        candidate.download = {
            "status": "failed",
            "filename": build_pdf_filename(candidate),
            "reason": "previous failure",
            "attempts": [],
        }
        ledger.record(candidate, candidate.download)

        fresh = Candidate(title="Failed Paper", doi="10.1000/fail")
        loaded = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        loaded.load()
        apply_download_log_resume(
            [fresh],
            loaded,
            PdfDownloader(HttpClient(), tmp_dir),
            retry_failed_downloads=False,
        )

        self.assertEqual(fresh.download["status"], "failed")
        self.assertEqual(fresh.download["reason"], "previous failure")

    def test_progress_total_excludes_download_log_resumed_candidates(self):
        class FakeResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                return ["https://example.org/paper.pdf"]

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False

            def download(self, candidate, urls):
                result = {
                    "status": "downloaded",
                    "url": urls[0],
                    "filename": build_pdf_filename(candidate),
                    "attempts": [{"url": urls[0], "status": "downloaded"}],
                }
                candidate.download = result
                return result

        class FakeProgress:
            instances = []

            def __init__(self, total, **kwargs):
                self.total = total
                self.enabled = False
                self.current_values = []
                self.candidate_lengths = []
                self.advances = 0
                FakeProgress.instances.append(self)

            def set_current(self, current, candidates, force=False):
                self.current_values.append(current)
                self.candidate_lengths.append(len(candidates))

            def advance(self, candidates, step=1):
                self.advances += step
                self.candidate_lengths.append(len(candidates))

            def done(self, candidates):
                self.candidate_lengths.append(len(candidates))

        tmp_dir = Path(tempfile.mkdtemp())
        log_path = tmp_dir / "logs" / "download_log.json"
        previous = Candidate(title="Failed Paper", doi="10.1000/fail")
        ledger = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        previous.download = {
            "status": "failed",
            "filename": build_pdf_filename(previous),
            "reason": "previous failure",
            "attempts": [],
        }
        ledger.record(previous, previous.download)

        resumed = Candidate(title="Failed Paper", doi="10.1000/fail")
        fresh = Candidate(title="Fresh Paper", doi="10.1000/fresh")
        loaded = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        loaded.load()
        candidates = [resumed, fresh]
        apply_download_log_resume(
            candidates,
            loaded,
            PdfDownloader(HttpClient(), tmp_dir),
            retry_failed_downloads=False,
        )

        with mock.patch.object(lit_harvest, "ProgressReporter", FakeProgress):
            count = download_candidates(
                candidates,
                resolver=FakeResolver(),
                downloader=FakeDownloader(tmp_dir),
                max_downloads=None,
                max_web_search_candidates=None,
                download_workers=1,
                ledger=loaded,
                progress_enabled=True,
            )

        self.assertEqual(count, 1)
        self.assertEqual(FakeProgress.instances[0].total, 1)
        self.assertEqual(FakeProgress.instances[0].current_values[0], 0)
        self.assertTrue(
            all(length == 1 for length in FakeProgress.instances[0].candidate_lengths)
        )
        self.assertEqual(FakeProgress.instances[0].advances, 1)

    def test_saved_document_status_includes_abstract_fallbacks(self):
        self.assertTrue(is_saved_document_status("downloaded"))
        self.assertTrue(is_saved_document_status("already_exists"))
        self.assertTrue(is_saved_document_status("abstract_only"))
        self.assertFalse(is_saved_document_status("failed"))

    def test_configured_queries_precede_generated_queries(self):
        configured = configured_search_queries(
            [
                {
                    "bucket": "persuasion",
                    "text": "propaganda technique classification rhetorical strategy taxonomy",
                    "terms": ["propaganda", "rhetorical strategy"],
                }
            ]
        )
        generated_queries = build_search_queries(
            extract_keywords("# Moral Framing\n\nmoral framing detection media"),
            max_queries=2,
        )
        combined = combine_search_queries(configured, generated_queries, max_queries=3)

        self.assertEqual(combined[0].bucket, "persuasion")
        self.assertEqual(
            combined[0].text,
            "propaganda technique classification rhetorical strategy taxonomy",
        )
        self.assertLessEqual(len(combined), 3)


if __name__ == "__main__":
    unittest.main()
