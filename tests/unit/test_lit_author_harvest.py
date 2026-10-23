import csv
import tempfile
import unittest
from pathlib import Path

from lit_review.collection.authors import (
    FacultyRecord,
    author_institutions,
    institution_similarity,
    load_faculty,
    local_pdf_path,
    score_author,
    write_html_viewer,
)


class AuthorHarvestTests(unittest.TestCase):
    def test_load_faculty_deduplicates_repeated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faculty.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["No", "Name", "Title", "College", "Department"])
                writer.writerow([1, "Hernan Makse", "Professor", "City College", "Physics"])
                writer.writerow([2, "Hernan Makse", "Professor", "City College", "Physics"])
                writer.writerow([3, "Susan Epstein", "Professor", "Hunter College", "CS"])

            records = load_faculty(path)

        self.assertEqual([record.name for record in records], ["Hernan Makse", "Susan Epstein"])
        self.assertEqual(records[0].source_rows, [2, 3])

    def test_institution_aliases_match_cuny_campus_names(self):
        self.assertEqual(
            institution_similarity("City College", ["The City College of New York"]),
            1.0,
        )
        self.assertEqual(
            institution_similarity("John Jay College", ["John Jay College of Criminal Justice"]),
            1.0,
        )

    def test_score_author_prefers_exact_name_and_institution(self):
        faculty = FacultyRecord(name="Bruce Lee", college="CUNY GradSch ofPub H and H Pol")
        correct = {
            "display_name": "Bruce Lee",
            "last_known_institutions": [
                {"display_name": "CUNY Graduate School of Public Health and Health Policy"}
            ],
        }
        namesake = {
            "display_name": "Bruce Lee",
            "last_known_institutions": [{"display_name": "University of Washington"}],
        }
        self.assertGreater(score_author(faculty, correct)[0], score_author(faculty, namesake)[0])
        self.assertEqual(
            author_institutions(correct), ["CUNY Graduate School of Public Health and Health Policy"]
        )

    def test_score_author_treats_middle_initials_and_accents_as_name_match(self):
        faculty = FacultyRecord(name="Hernan Makse", college="City College")
        candidate = {
            "display_name": "Hernán A. Makse",
            "last_known_institutions": [{"display_name": "City College of New York"}],
        }
        total, name_score, institution_score = score_author(faculty, candidate)
        self.assertEqual(name_score, 1.0)
        self.assertEqual(institution_score, 1.0)
        self.assertEqual(total, 1.0)

    def test_score_author_does_not_treat_prefixed_names_as_exact(self):
        faculty = FacultyRecord(name="Hui Chen", college="Brooklyn College")
        candidate = {
            "display_name": "Ming-Hui Chen",
            "last_known_institutions": [{"display_name": "Florida State University"}],
        }
        _, name_score, institution_score = score_author(faculty, candidate)
        self.assertLess(name_score, 1.0)
        self.assertEqual(institution_score, 0.0)

    def test_html_viewer_embeds_filters_links_and_safe_data(self):
        authors = [
            {
                "roster_name": "Ada Lovelace",
                "college": "Example College",
                "department": "Computing",
                "match_status": "matched",
                "collected_publication_count": 1,
            }
        ]
        publication = {
            field: ""
            for field in __import__("lit_review.collection.authors", fromlist=["CSV_FIELDS"]).CSV_FIELDS
        }
        publication.update(
            {
                "roster_name": "Ada Lovelace",
                "title": "A </script> safe title",
                "year": "1843",
                "pdf_availability": "pdf_url_listed",
                "pdf_urls": "https://example.org/paper.pdf",
                "landing_page_url": "https://example.org/paper",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.html"
            pdf_path = Path(directory) / "authors" / "ada_lovelace" / "pdfs" / "paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF-1.4\n")
            publication["download_status"] = "downloaded"
            publication["download_path"] = str(pdf_path)
            write_html_viewer(
                path,
                authors,
                [publication, publication.copy()],
                generated_at="2026-10-20T12:00:00Z",
            )
            page = path.read_text(encoding="utf-8")
        self.assertIn("Faculty publications", page)
        self.assertNotIn("PDF availability", page)
        self.assertIn("Most cited", page)
        self.assertIn("Downloaded PDF", page)
        self.assertIn('"downloads":1', page)
        self.assertIn("https://example.org/paper.pdf", page)
        self.assertNotIn('"A </script> safe title"', page)

    def test_local_pdf_path_rejects_empty_pdf_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_pdf = root / "empty.pdf"
            empty_pdf.touch()
            self.assertEqual(local_pdf_path(root / "index.html", empty_pdf), "")


if __name__ == "__main__":
    unittest.main()
