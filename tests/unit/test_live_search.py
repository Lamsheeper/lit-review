import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lit_review.collection import harvest
from lit_review.collection.sources import Candidate


class LiveSearchTests(unittest.TestCase):
    def test_preview_is_published_between_sources_and_never_reused_as_completed_search(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = root / "draft.md"
            draft.write_text("# Universities\nResearch universities and teaching.")

            class First:
                name = "first"

                def search(self, *args, **kwargs):
                    return [Candidate(title="University study", doi="10.1000/test")]

            class Second:
                name = "second"

                def search(self, *args, **kwargs):
                    preview = json.loads((root / "logs/candidates.live.json").read_text())
                    self_test.assertEqual(len(preview["candidates"]), 1)
                    self_test.assertFalse(preview["complete"])
                    self_test.assertEqual(preview["progress"]["completed"], 1)
                    self_test.assertFalse((root / "logs/candidates.json").exists())
                    raise KeyboardInterrupt("cancel during search")

            self_test = self
            config = {
                "draft": str(draft),
                "output": str(root / "papers"),
                "logs": str(root / "logs"),
                "max_queries": 1,
                "top_k_per_query": 10,
                "timeout": 30,
                "retries": 0,
                "backoff": 0,
                "rate_limit_delay": 0,
                "refresh_candidates": True,
            }
            with patch.object(harvest, "build_clients", return_value=[First(), Second()]):
                with self.assertRaises(KeyboardInterrupt):
                    harvest.run_pipeline(config)
            self.assertFalse((root / "logs/candidates.json").exists())
            preview = root / "logs/candidates.live.json"
            self.assertIsNone(harvest.load_candidate_cache(preview))
