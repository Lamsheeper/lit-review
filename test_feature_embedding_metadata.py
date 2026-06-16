import json
import tempfile
import unittest
from pathlib import Path

from feature_embedding_metadata import (
    load_embedding_metadata_overrides,
    parse_categories,
    run_embedding_metadata_generation,
)


class FakeAttributeClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)

    def chat(self, messages, temperature=0.2, max_tokens=None, response_format=None):
        if not self.responses:
            raise AssertionError("FakeAttributeClient ran out of responses.")
        return json.dumps({"annotations": [self.responses.pop(0)]})


class FeatureEmbeddingMetadataTests(unittest.TestCase):
    def test_parse_categories_accepts_comma_separated_values(self):
        self.assertEqual(parse_categories("sentiment_affect, persuasion"), {"sentiment_affect", "persuasion"})

    def test_generation_replaces_selected_embedding_fields_and_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_key": "sentiment_affect::joy",
                                "feature_name": "Joy",
                                "category": "sentiment_affect",
                                "synonyms": ["old synonym"],
                                "annotation": {"short_definition": "Evidence-grounded joy definition."},
                            },
                            {
                                "feature_key": "persuasion::slogan",
                                "feature_name": "Slogan",
                                "category": "persuasion",
                                "synonyms": ["catchphrase"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            trace = run_embedding_metadata_generation(
                features_path=features_path,
                output_dir=root / "out",
                prefix="features",
                cache_path=root / "cache.jsonl",
                client=FakeAttributeClient(
                    [
                        {
                            "feature_name": "Joy",
                            "definition": "A positive emotional state of delight.",
                            "synonyms": ["happiness", "delight"],
                            "examples": ["Celebrating good news.", "Smiling after a success."],
                        }
                    ]
                ),
                categories={"sentiment_affect"},
            )
            payload = json.loads(Path(trace["outputs"]["features_json"]).read_text())

        joy, slogan = payload["features"]
        self.assertEqual(joy["synonyms"], ["happiness", "delight"])
        self.assertEqual(joy["examples"], ["Celebrating good news.", "Smiling after a success."])
        self.assertEqual(joy["annotation"]["short_definition"], "Evidence-grounded joy definition.")
        self.assertEqual(joy["generated_attributes"]["input_fields"], ["feature_name", "category"])
        self.assertEqual(slogan["synonyms"], ["catchphrase"])
        self.assertEqual(trace["counts"]["generated_features"], 1)
        self.assertEqual(trace["counts"]["output_features"], 2)

    def test_manual_override_completes_uncached_row_without_client(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.json"
            features_path.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "feature_key": "moral_framing::contrast",
                                "feature_name": "A vs. B",
                                "category": "moral_framing",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            overrides_path = root / "overrides.json"
            overrides_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "feature_name": "A vs. B",
                                "category": "moral_framing",
                                "definition": "A contrast between A and B.",
                                "synonyms": ["A-B dichotomy"],
                                "examples": ["Favoring A over B.", "Favoring B over A."],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            trace = run_embedding_metadata_generation(
                features_path=features_path,
                output_dir=root / "out",
                prefix="features",
                cache_path=root / "cache.jsonl",
                client=None,
                categories={"moral_framing"},
                overrides=load_embedding_metadata_overrides(overrides_path),
                overrides_path=overrides_path,
            )
            payload = json.loads(Path(trace["outputs"]["features_json"]).read_text())

        self.assertEqual(payload["features"][0]["synonyms"], ["A-B dichotomy"])
        self.assertEqual(trace["counts"]["manual_overrides"], 1)


if __name__ == "__main__":
    unittest.main()
