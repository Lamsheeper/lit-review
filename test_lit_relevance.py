import json
import tempfile
import unittest
from pathlib import Path

from lit_relevance import load_relevance_profile, validate_relevance_profile


def example_profile():
    return {
        "version": 1,
        "title": "Clinical intervention relevance",
        "description": "Prioritize papers that evaluate clinical interventions.",
        "criteria": [
            {
                "id": "clinical_intervention",
                "label": "Clinical intervention",
                "description": "The paper evaluates an intervention in clinical care.",
                "weight": 4,
                "terms": ["clinical intervention", "randomized trial"],
            }
        ],
        "field_weights": {"title": 3, "abstract": 2, "matched_keywords": 1},
        "metadata_weights": {
            "source_relevance": 0,
            "citation_impact": 0,
            "recency": 0,
            "open_access": 0,
            "doi": 0,
        },
        "exclusions": [
            {"term": "animal model", "penalty": 0.5, "reason": "Human studies only."}
        ],
    }


class RelevanceProfileTests(unittest.TestCase):
    def test_validates_and_loads_profile(self):
        profile = validate_relevance_profile(example_profile())
        self.assertEqual(profile["criteria"][0]["id"], "clinical_intervention")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "relevance_scoring.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            self.assertEqual(load_relevance_profile(path), profile)

    def test_rejects_unknown_metadata_signal(self):
        profile = example_profile()
        profile["metadata_weights"]["prestige"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            validate_relevance_profile(profile)

    def test_requires_a_positive_scoring_weight(self):
        profile = example_profile()
        profile["criteria"][0]["weight"] = 0
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            validate_relevance_profile(profile)


if __name__ == "__main__":
    unittest.main()
