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
    build_direct_pdf_citation_content,
    build_pymupdf_markdown_citation_content,
    build_search_queries,
    build_web_pdf_queries,
    combine_search_queries,
    configured_search_queries,
    candidate_matches_keywords,
    apply_download_log_resume,
    candidates_from_citation_extractions,
    citation_match_reason,
    download_candidates,
    extracted_reference_to_candidate,
    extract_pdf_urls_from_html,
    extract_keywords,
    inferred_pdf_urls,
    is_full_document_status,
    is_saved_document_status,
    load_candidate_cache,
    load_config,
    merge_candidates,
    normalize_doi,
    parse_args,
    parse_citation_llm_json,
    validate_citation_bibliography,
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

    def test_download_skips_abstract_fallback_when_disabled(self):
        candidate = Candidate(
            title="No Abstract Fallback Paper",
            authors=["Test Author"],
            abstract="This is the abstract text.",
        )
        downloader = PdfDownloader(
            HttpClient(),
            Path(tempfile.mkdtemp()),
            abstract_fallback=False,
        )

        result = downloader.download(candidate, [])

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "No candidate PDF URL available.")
        self.assertEqual(result["attempts"], [])
        self.assertEqual(candidate.download["status"], "failed")

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

    def test_download_log_resume_retries_abstract_only_candidates(self):
        tmp_dir = Path(tempfile.mkdtemp())
        log_path = tmp_dir / "logs" / "download_log.json"
        candidate = Candidate(title="Abstract Paper", doi="10.1000/abstract")
        existing_path = tmp_dir / build_pdf_filename(candidate)
        existing_path.write_bytes(b"%PDF-1.4\nabstract fallback\n%%EOF\n")
        ledger = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        candidate.download = {
            "status": "abstract_only",
            "filename": build_pdf_filename(candidate),
            "path": str(existing_path),
            "attempts": [],
        }
        ledger.record(candidate, candidate.download)

        fresh = Candidate(title="Abstract Paper", doi="10.1000/abstract")
        loaded = DownloadLogLedger(log_path, candidate_cache={}, search_errors=[])
        loaded.load()
        result = apply_download_log_resume(
            [fresh],
            loaded,
            PdfDownloader(HttpClient(), tmp_dir),
            retry_failed_downloads=False,
        )

        self.assertEqual(result["abstract_only_retry"], 1)
        self.assertEqual(fresh.download["status"], "retry_abstract_only")

    def test_download_candidates_retries_abstract_only_even_if_file_exists(self):
        class FakeResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                return ["https://example.org/full.pdf"]

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False
                self.calls = 0

            def download(self, candidate, urls):
                self.calls += 1
                result = {
                    "status": "downloaded",
                    "url": urls[0],
                    "filename": build_pdf_filename(candidate),
                    "attempts": [{"url": urls[0], "status": "downloaded"}],
                }
                candidate.download = result
                return result

        tmp_dir = Path(tempfile.mkdtemp())
        candidate = Candidate(title="Abstract Paper", doi="10.1000/abstract")
        existing_path = tmp_dir / build_pdf_filename(candidate)
        existing_path.write_bytes(b"%PDF-1.4\nabstract fallback\n%%EOF\n")
        candidate.download = {
            "status": "retry_abstract_only",
            "filename": build_pdf_filename(candidate),
            "path": str(existing_path),
        }
        downloader = FakeDownloader(tmp_dir)

        count = download_candidates(
            [candidate],
            resolver=FakeResolver(),
            downloader=downloader,
            max_downloads=None,
            max_web_search_candidates=None,
            download_workers=1,
        )

        self.assertEqual(count, 1)
        self.assertEqual(downloader.calls, 1)
        self.assertEqual(candidate.download["status"], "downloaded")

    def test_abstract_only_does_not_count_toward_max_downloads(self):
        class FakeResolver:
            web_searcher = None

            def resolve(self, candidate, use_web_search=True):
                return ["https://example.org/paper.pdf"]

        class FakeDownloader:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.force_download = False

            def download(self, candidate, urls):
                if candidate.title == "Abstract Only":
                    result = {
                        "status": "abstract_only",
                        "filename": build_pdf_filename(candidate),
                        "attempts": [{"source": "abstract_fallback", "status": "generated"}],
                    }
                else:
                    result = {
                        "status": "downloaded",
                        "url": urls[0],
                        "filename": build_pdf_filename(candidate),
                        "attempts": [{"url": urls[0], "status": "downloaded"}],
                    }
                candidate.download = result
                return result

        candidates = [
            Candidate(title="Abstract Only"),
            Candidate(title="Full Text"),
            Candidate(title="After Limit"),
        ]

        count = download_candidates(
            candidates,
            resolver=FakeResolver(),
            downloader=FakeDownloader(Path(tempfile.mkdtemp())),
            max_downloads=1,
            max_web_search_candidates=None,
            download_workers=1,
        )

        self.assertEqual(count, 1)
        self.assertEqual(candidates[0].download["status"], "abstract_only")
        self.assertEqual(candidates[1].download["status"], "downloaded")
        self.assertEqual(candidates[2].download["status"], "not_attempted")

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
        self.assertTrue(is_full_document_status("downloaded"))
        self.assertTrue(is_full_document_status("already_exists"))
        self.assertFalse(is_full_document_status("abstract_only"))

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

    def test_citation_schema_validation_and_malformed_json(self):
        valid = {
            "paper_id": "seed",
            "references": [
                {
                    "ref_id": "R001",
                    "raw_reference": "Ada Lovelace. A Study. 2024.",
                    "authors": ["Ada Lovelace"],
                    "title": "A Study",
                    "year": 2024,
                    "venue": "Journal",
                    "doi": "10.1000/test",
                    "arxiv_id": None,
                    "url": None,
                }
            ],
        }

        self.assertEqual(validate_citation_bibliography(valid), [])
        parsed, errors = parse_citation_llm_json("{not json", "seed")

        self.assertEqual(parsed, {"paper_id": "seed", "references": []})
        self.assertTrue(errors)

    def test_direct_pdf_citation_content_uses_pdf_file_input(self):
        tmp_dir = Path(tempfile.mkdtemp())
        pdf_path = tmp_dir / "core.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        content, metadata = build_direct_pdf_citation_content(pdf_path, "core")

        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_file")
        self.assertEqual(content[1]["filename"], "core.pdf")
        self.assertEqual(content[1]["file_data"], "JVBERi0xLjQgdGVzdA==")
        self.assertEqual(metadata["pdf_bytes"], len(b"%PDF-1.4 test"))

    def test_pymupdf_markdown_citation_content_truncates_text(self):
        pdf_path = Path(tempfile.mkdtemp()) / "core.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        with mock.patch(
            "lit_harvest.extract_pdf_text_for_citations",
            return_value=("Reference text that should be truncated", 3),
        ):
            content, metadata = build_pymupdf_markdown_citation_content(
                pdf_path,
                "core",
                max_text_chars=14,
            )

        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("truncated to 14 characters", content[0]["text"])
        self.assertIn("Reference text", content[0]["text"])
        self.assertEqual(metadata["num_pages"], 3)
        self.assertTrue(metadata["text_truncated"])

    def test_extracted_reference_to_candidate_handles_identifiers_and_url(self):
        reference = {
            "ref_id": "R001",
            "raw_reference": "Lovelace A. A Study. 2024.",
            "authors": ["Ada Lovelace"],
            "title": "A Study",
            "year": 2024,
            "venue": "Journal",
            "doi": "https://doi.org/10.1000/TEST.",
            "arxiv_id": "arXiv:2301.12345",
            "url": "https://example.org/paper.pdf",
        }

        candidate = extracted_reference_to_candidate(
            reference,
            core_pdf="core.pdf",
            paper_id="core",
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.title, "A Study")
        self.assertEqual(candidate.doi, "10.1000/test")
        self.assertEqual(candidate.source_ids["citation:arxiv"], "2301.12345")
        self.assertIn("https://example.org/paper.pdf", candidate.candidate_pdf_urls)
        self.assertEqual(candidate.discovered_via[0]["ref_id"], "R001")

    def test_citation_extractions_dedupe_by_arxiv(self):
        extraction = {
            "core_pdf": "core.pdf",
            "paper_id": "core",
            "references": [
                {
                    "ref_id": "R001",
                    "raw_reference": "First",
                    "authors": [],
                    "title": "One Title",
                    "year": None,
                    "venue": None,
                    "doi": None,
                    "arxiv_id": "2301.12345",
                    "url": None,
                },
                {
                    "ref_id": "R002",
                    "raw_reference": "Second",
                    "authors": [],
                    "title": "Different Title",
                    "year": None,
                    "venue": None,
                    "doi": None,
                    "arxiv_id": "https://arxiv.org/abs/2301.12345",
                    "url": None,
                },
            ],
        }

        candidates = merge_candidates(candidates_from_citation_extractions([extraction]))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].discovered_via), 2)

    def test_citation_match_reason_accepts_exact_and_rejects_weak_matches(self):
        seed = Candidate(
            title="Persuasion Techniques in Propaganda Detection",
            authors=["Ada Lovelace"],
            year=2024,
            doi="10.1000/test",
        )
        doi_match = Candidate(
            title="Other Title",
            authors=["Different Person"],
            year=1999,
            doi="https://doi.org/10.1000/TEST",
        )
        title_match = Candidate(
            title="Persuasion Techniques in Propaganda Detection",
            authors=[],
            year=None,
        )
        weak_match = Candidate(
            title="A General Survey of Media",
            authors=["Ada Lovelace"],
            year=2024,
        )

        self.assertEqual(citation_match_reason(seed, doi_match), "doi")
        self.assertEqual(citation_match_reason(seed, title_match), "title")
        self.assertIsNone(citation_match_reason(seed, weak_match))

    def test_citation_lookup_resume_skips_completed_sources(self):
        query = "Seed Paper 2020 Smith"
        candidate = Candidate(title="Seed Paper", authors=["Alice Smith"], year=2020)
        candidate.discovered_via.append(
            {
                "source": "citation_lookup",
                "query": query,
                "attempts": [
                    {
                        "source": "openalex",
                        "query": query,
                        "result_count": 1,
                        "accepted_count": 1,
                    }
                ],
            }
        )

        class FakeClient:
            def __init__(self, name, results=None):
                self.name = name
                self.results = results or []
                self.calls = 0

            def search(self, query, top_k, year_from, year_to):
                self.calls += 1
                return self.results

        openalex = FakeClient("openalex")
        crossref = FakeClient(
            "crossref",
            [
                Candidate(
                    title="Seed Paper",
                    authors=["Alice Smith"],
                    year=2020,
                    doi="10.1000/seed",
                    source_apis=["crossref"],
                )
            ],
        )
        checkpoints = []

        candidates, errors = lit_harvest.enrich_citation_candidates(
            [candidate],
            [openalex, crossref],
            {
                "top_k_per_query": 1,
                "progress": False,
                "_initial_citation_lookup_errors": [],
            },
            checkpoint=lambda current, lookup_errors, progress: checkpoints.append(progress),
        )

        self.assertEqual(errors, [])
        self.assertEqual(openalex.calls, 0)
        self.assertEqual(crossref.calls, 1)
        self.assertEqual(candidates[0].doi, "10.1000/seed")
        self.assertEqual(len(candidates[0].discovered_via), 2)
        self.assertEqual(checkpoints[-1]["completed_lookup_attempts"], 2)
        self.assertEqual(checkpoints[-1]["total_lookup_attempts"], 2)

    def test_citation_mode_config_validation(self):
        args = parse_args(
            [
                "--harvest-mode",
                "citations",
                "--core-pdf",
                "seed.pdf",
                "--output",
                "out",
                "--llm-api-key",
                "secret",
            ]
        )
        config = load_config(args)

        self.assertEqual(config["harvest_mode"], "citations")
        self.assertEqual(config["core_pdf"], ["seed.pdf"])
        self.assertFalse(config["allow_abstract_fallback"])

        with self.assertRaises(ValueError):
            load_config(parse_args(["--harvest-mode", "citations", "--output", "out"]))

        tmp_dir = Path(tempfile.mkdtemp())
        config_path = tmp_dir / "bad_config.json"
        config_path.write_text(
            """{
              "harvest_mode": "citations",
              "core_pdf": ["seed.pdf"],
              "output": "out",
              "citation_extractor": "pdf_images"
            }""",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_config(parse_args(["--config", str(config_path)]))

    def test_downloader_can_disable_abstract_fallback(self):
        tmp_dir = Path(tempfile.mkdtemp())
        candidate = Candidate(
            title="Abstract-Only Paper",
            authors=["Test Author"],
            abstract="This is the abstract text.",
        )
        downloader = PdfDownloader(
            HttpClient(),
            tmp_dir,
            allow_abstract_fallback=False,
        )

        result = downloader.download(candidate, [])

        self.assertEqual(result["status"], "failed")
        self.assertEqual(candidate.download["status"], "failed")
        self.assertEqual(list(tmp_dir.glob("*.pdf")), [])

    def test_no_abstract_cli_sets_config_flag(self):
        args = lit_harvest.parse_args(
            ["--draft", "draft.md", "--output", "papers", "--no-abstract"]
        )
        config = lit_harvest.load_config(args)

        self.assertTrue(config["no_abstract"])


if __name__ == "__main__":
    unittest.main()
