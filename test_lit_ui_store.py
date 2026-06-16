import tempfile
import unittest
from pathlib import Path

from lit_review_ui.store import ProjectStore


class ProjectStoreTests(unittest.TestCase):
    def test_project_lifecycle_and_active_job_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project("My Review", "Description")
            self.assertEqual(project["name"], "My Review")
            self.assertTrue((Path(temp) / project["slug"] / "draft.md").exists())
            updated = store.update_project(project["id"], {"name": "Renamed"})
            self.assertEqual(updated["name"], "Renamed")
            first = store.create_job(project["id"], "search", Path(temp) / "job.log")
            with self.assertRaises(ValueError):
                store.create_job(project["id"], "extract", Path(temp) / "job2.log")
            store.update_job(first["id"], status="completed")
            second = store.create_job(project["id"], "extract", Path(temp) / "job2.log")
            self.assertEqual(second["status"], "queued")

            restarted = ProjectStore(Path(temp))
            self.assertEqual(restarted.get_job(second["id"])["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
