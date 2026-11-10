import unittest

from lit_review.collection.downloads import PdfResolver
from lit_review.collection.sources import Candidate, CrossrefClient, SearchQuery, add_doi_pdf_urls, add_identifier_pdf_urls, normalize_acl_id


class AclIdentifierTests(unittest.TestCase):
    def test_historical_identifiers_and_urls_use_the_required_uppercase_prefix(self):
        for raw in ("d19-1565", "D19-1565", "https://aclanthology.org/d19-1565/", "https://www.aclanthology.org/d19-1565.PDF"):
            with self.subTest(identifier=raw):
                self.assertEqual(normalize_acl_id(raw), "D19-1565")
                urls = []
                add_identifier_pdf_urls(urls, raw, kind="acl")
                self.assertEqual(urls, ["https://aclanthology.org/D19-1565.pdf"])

    def test_modern_identifiers_remain_lowercase(self):
        for raw in ("2021.semeval-1.7", "2021.SEMEVAL-1.7", "https://aclanthology.org/2021.SEMEVAL-1.7.pdf"):
            with self.subTest(identifier=raw):
                self.assertEqual(normalize_acl_id(raw), "2021.semeval-1.7")
                urls = []
                add_identifier_pdf_urls(urls, raw, kind="acl")
                self.assertEqual(urls, ["https://aclanthology.org/2021.semeval-1.7.pdf"])

    def test_lowercased_crossref_doi_produces_a_canonical_acl_pdf_url(self):
        candidate = CrossrefClient(None)._candidate_from_work(
            {"title": ["Fine-Grained Analysis of Propaganda in News Articles"], "DOI": "10.18653/v1/d19-1565"},
            SearchQuery(bucket="fixture", text="propaganda", terms=["propaganda"]),
        )
        self.assertEqual(candidate.doi, "10.18653/v1/d19-1565")
        self.assertIn("https://aclanthology.org/D19-1565.pdf", candidate.candidate_pdf_urls)
        urls = []
        add_doi_pdf_urls(urls, "10.18653/v1/d19-1565")
        self.assertEqual(urls, ["https://aclanthology.org/D19-1565.pdf"])

    def test_cached_lowercase_metadata_does_not_hide_the_correct_inferred_url(self):
        candidate = Candidate(
            title="Historical ACL paper", doi="10.18653/v1/d19-1565",
            candidate_pdf_urls=["https://aclanthology.org/d19-1565.pdf"],
        )
        resolver = PdfResolver(None)
        resolver._resolve_landing_page = lambda url: []
        urls = resolver.resolve(candidate, use_web_search=False)
        self.assertIn("https://aclanthology.org/D19-1565.pdf", urls)
        self.assertNotIn("https://aclanthology.org/d19-1565.pdf", urls)
