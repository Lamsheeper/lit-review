import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from lit_review.core.io import read_json, read_jsonl, write_json, write_jsonl
from lit_review.core.settings import PipelineConfig, StageSpec
from lit_review.pipeline import compile_pipeline, load_pipeline, run_pipeline
from lit_review.services import execute_operation


class ResearchClient:
    def __init__(self):
        self.prompts = []
        self.fail_annotation_once = False

    def chat(self, messages, **kwargs):
        content = messages[-1]["content"]
        if isinstance(content, list):
            content = next(part["text"] for part in content if part.get("type") == "text")
        self.prompts.append(content)
        payload = json.loads(content.rsplit("INPUT\n", 1)[1])
        if content.startswith("Extract taxonomy"):
            return json.dumps(
                {
                    "features": [
                        {
                            "category": "ergonomics",
                            "feature_name": name,
                            "definition": "A spinal alignment observed during seated work.",
                            "source_quote": "Participants adopted a neutral spine posture.",
                            "evidence_type": "definition",
                            "confidence": 0.9,
                        }
                        for name in ("Neutral posture", "Neutral spine posture")
                    ]
                }
            )
        if content.startswith("Filter extracted"):
            return json.dumps(
                {
                    "decisions": [
                        {"feature_key": row["feature_key"], "action": "keep"} for row in payload["features"]
                    ]
                }
            )
        if content.startswith("Write polished"):
            if self.fail_annotation_once:
                self.fail_annotation_once = False
                raise RuntimeError("temporary provider failure")
            return json.dumps(
                {
                    "annotations": [
                        {
                            "feature_key": row["feature_key"],
                            "status": "annotated",
                            "formal_definition": "A neutral spinal alignment during seated work.",
                            "short_definition": "Neutral spinal alignment.",
                            "synonyms": ["Aligned posture"],
                            "examples": ["The participant sat with an aligned spine."],
                            "confidence": 0.9,
                        }
                        for row in payload["features"]
                    ]
                }
            )
        if "candidate pairs" in content or "merge decisions" in content or "pair_id" in content:
            return json.dumps(
                {
                    "decisions": [
                        {"pair_id": row["pair_id"], "decision": "merge", "confidence": 0.95}
                        for row in payload["candidate_pairs"]
                    ]
                }
            )
        if "scores" in content:
            return json.dumps(
                {
                    "scores": [
                        {
                            "feature_key": row["feature_key"],
                            "annotation_reliability": 4,
                            "distinctiveness": 4,
                            "confidence": 0.9,
                        }
                        for row in payload["features"]
                    ]
                }
            )
        raise AssertionError(content[:100])


class Embeddings:
    def embed_texts(self, texts):
        return [[1.0, 0.01 * (i + 1)] for i in range(len(texts))]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.papers = self.root / "papers"
        self.papers.mkdir()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(str(self.papers / "workplace.pdf"))
        self.taxonomy = self.root / "taxonomy.json"
        write_json(
            self.taxonomy,
            {
                "version": 1,
                "title": "Workplace factors",
                "families": [
                    {
                        "id": "ergonomics",
                        "label": "Ergonomic factors",
                        "description": "Posture and workstation factors",
                        "aliases": [],
                    }
                ],
            },
        )
        self.goal = self.root / "goal.md"
        self.goal.write_text("Extract workplace posture factors.")
        self.client = ResearchClient()
        self.config = PipelineConfig(
            output_root=self.root / "run",
            inputs={"papers": str(self.papers), "taxonomy": str(self.taxonomy), "goal": str(self.goal)},
            stages=[
                StageSpec(id=stage, operation=operation, options=options)
                for stage, operation, options in [
                    ("prepare", "evidence.prepare", {}),
                    ("extract", "features.extract", {"progress": False}),
                    ("filter", "features.filter", {"progress": False}),
                    ("annotate", "features.annotate", {"progress": False}),
                    (
                        "merge",
                        "features.merge",
                        {"progress": False, "cache": str(self.root / "embeddings.jsonl")},
                    ),
                    ("score", "features.score", {"progress": False}),
                ]
            ],
        )

    def executor(self, operation, options, **kwargs):
        runtime = (
            {"client": self.client, "embedding_client": Embeddings()}
            if operation.startswith("features.")
            else {}
        )
        return execute_operation(operation, options, **kwargs, **runtime)

    def test_arbitrary_taxonomy_end_to_end_optional_scoring_and_resume(self):
        result = run_pipeline(
            self.config, stages=["prepare", "extract", "filter", "annotate", "merge"], executor=self.executor
        )
        self.assertNotIn("score", result["stages"])
        merged = read_jsonl(self.config.output_root / "stages/merge/llm_merged_features.jsonl")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["category"], "ergonomics")
        self.assertTrue(merged[0]["source_paper_ids"])
        self.assertFalse((self.config.output_root / "stages/filter/curated_features.jsonl").exists())
        self.assertFalse(
            any(
                "For moral_framing" in prompt or "specific emotions" in prompt
                for prompt in self.client.prompts
            )
        )
        calls = len(self.client.prompts)
        run_pipeline(
            self.config, stages=["prepare", "extract", "filter", "annotate", "merge"], executor=self.executor
        )
        self.assertEqual(len(self.client.prompts), calls)
        result = run_pipeline(self.config, stages=["score"], executor=self.executor)
        self.assertEqual(result["stages"]["score"]["status"], "completed")
        self.assertEqual(len(read_jsonl(self.config.output_root / "stages/score/scored_features.jsonl")), 1)

    def test_preview_is_read_only_and_accepts_planned_intermediates(self):
        before = sorted(self.root.rglob("*"))
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            preview = run_pipeline(self.config, dry_run=True)
        self.assertEqual(len(preview["stages"]), 6)
        self.assertEqual(sorted(self.root.rglob("*")), before)
        self.assertFalse(self.config.output_root.exists())

    def test_feature_pipeline_exports_without_analysis_dependencies(self):
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None, "sklearn": None}):
            result = run_pipeline(self.config, executor=self.executor)
        self.assertTrue(all(record["status"] == "completed" for record in result["stages"].values()))
        merged = self.config.output_root / "stages/merge/llm_merged_features.jsonl"
        scored = self.config.output_root / "stages/score/scored_features.jsonl"
        self.assertEqual(len(read_jsonl(merged)), 1)
        self.assertEqual(len(read_jsonl(scored)), 1)
        self.assertFalse(list(self.config.output_root.rglob("*.png")))
        self.assertEqual(result["stages"]["merge"]["result"]["outputs"]["llm_merge_retention_by_category"], 0)

    def test_analysis_explains_optional_install(self):
        from lit_review.analysis.dispatch import analysis_arguments

        with patch("lit_review.analysis.dispatch.find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "uv sync --extra analysis"):
                analysis_arguments("cluster", {}, self.root / "analysis")

    def test_failed_stage_resumes_and_never_completes_downstream(self):
        self.client.fail_annotation_once = True
        with self.assertRaisesRegex(RuntimeError, "failed items"):
            run_pipeline(self.config, executor=self.executor)
        manifest = read_json(self.config.output_root / "manifest.json")
        self.assertEqual(manifest["stages"]["annotate"]["status"], "failed")
        self.assertNotIn("merge", manifest["stages"])
        extraction_calls = sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts)
        result = run_pipeline(self.config, executor=self.executor)
        self.assertEqual(result["stages"]["merge"]["status"], "completed")
        self.assertEqual(
            sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts), extraction_calls
        )

    def test_upstream_changes_require_dependencies_and_archive_old_results(self):
        run_pipeline(self.config, executor=self.executor)
        self.goal.write_text("A revised posture extraction goal.")
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            run_pipeline(self.config, stages=["merge"], executor=self.executor)
        result = run_pipeline(self.config, executor=self.executor)
        self.assertEqual(result["stages"]["merge"]["status"], "completed")
        self.assertTrue(list((self.config.output_root / "history").glob("extract-*")))

    def test_forced_preparation_rebuilds_downstream_checkpoints(self):
        run_pipeline(self.config, executor=self.executor)
        calls = sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts)
        run_pipeline(self.config, rerun=["prepare"], executor=self.executor)
        self.assertEqual(
            sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts), calls + 1
        )

    def test_changed_producer_is_detected_across_selected_stage_runs(self):
        run_pipeline(self.config, executor=self.executor)
        calls = sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts)
        with (self.papers / "workplace.pdf").open("ab") as handle:
            handle.write(b"\n% Changed source content\n")
        run_pipeline(self.config, stages=["prepare"], executor=self.executor)
        run_pipeline(
            self.config, stages=["extract", "filter", "annotate", "merge", "score"], executor=self.executor
        )
        self.assertEqual(
            sum(prompt.startswith("Extract taxonomy") for prompt in self.client.prompts), calls + 1
        )

    def test_empty_inventory_completes_without_credentials(self):
        source = self.root / "empty"
        source.mkdir()
        write_jsonl(source / "features.jsonl", [])
        config = PipelineConfig(
            output_root=self.root / "empty-run",
            inputs={"taxonomy": str(self.taxonomy), "input_dir": str(source)},
            stages=[
                StageSpec(id="filter", operation="features.filter"),
                StageSpec(id="annotate", operation="features.annotate"),
                StageSpec(id="merge", operation="features.merge"),
            ],
        )
        result = run_pipeline(config)
        self.assertTrue(all(record["status"] == "completed" for record in result["stages"].values()))
        self.assertEqual(read_jsonl(config.output_root / "stages/merge/llm_merged_features.jsonl"), [])

    def test_cancelled_stage_is_recorded_and_resumable(self):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            run_pipeline(self.config, executor=interrupt)
        record = read_json(self.config.output_root / "manifest.json")["stages"]["prepare"]
        self.assertEqual(record["status"], "cancelled")
        self.assertFalse((self.config.output_root / ".lock").exists())
        run_pipeline(self.config, executor=self.executor)

    def test_rejects_ignored_settings_missing_taxonomy_and_bad_ids(self):
        self.config.stages[2].options["merge_style"] = "moderate"
        with self.assertRaisesRegex(ValueError, "do not apply"):
            run_pipeline(self.config, dry_run=True)
        self.config.stages[2].options.clear()
        with self.assertRaisesRegex(ValueError, "Unknown stages"):
            run_pipeline(self.config, dry_run=True, stages=["missing"])
        self.config.inputs.pop("taxonomy")
        with self.assertRaises(ValueError):
            run_pipeline(self.config, dry_run=True)

    def test_collection_rerun_preserves_downloaded_pdf_inputs(self):
        draft = self.root / "draft.md"
        draft.write_text("# Workplace factors")
        config = PipelineConfig(
            output_root=self.root / "collection",
            stages=[StageSpec(id="collect", operation="collect.search", options={"draft": str(draft)})],
        )
        attempts = []

        def collect(operation, options, *, output_dir, force):
            if attempts:
                self.assertEqual((output_dir / "existing.pdf").read_bytes(), b"%PDF-example")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "existing.pdf").write_bytes(b"%PDF-example")
            write_json(output_dir / "logs/manifest.json", {"candidates": []})
            attempts.append(operation)
            return {}

        run_pipeline(config, executor=collect)
        config.stages[0].options["max_queries"] = 2
        run_pipeline(config, executor=collect)
        self.assertEqual(len(attempts), 2)

    def test_config_relative_paths_and_cli_overrides(self):
        path = self.root / "pipeline.json"
        write_json(
            path,
            {
                "version": 1,
                "output_root": "new-run",
                "inputs": {"papers": "papers"},
                "stages": [{"id": "prepare", "operation": "evidence.prepare"}],
            },
        )
        config = load_pipeline(path, overrides={"output_root": "override-run"})
        self.assertEqual(config.output_root, self.root / "override-run")
        self.assertEqual(compile_pipeline(config)[0]["options"]["papers"], str(self.papers))

    def test_output_cannot_be_nested_inside_input_corpus(self):
        self.config.output_root = self.papers / "nested-output"
        with self.assertRaisesRegex(ValueError, "separate"):
            run_pipeline(self.config, dry_run=True)
        self.assertFalse(self.config.output_root.exists())

    def test_literal_credentials_never_appear_in_validation_errors(self):
        from lit_review.core.settings import ProviderSettings

        with self.assertRaises(ValueError) as failure:
            ProviderSettings.model_validate({"api_key": "never-print-this"})
        self.assertNotIn("never-print-this", str(failure.exception))

    def test_artifact_tampering_invalidates_stage(self):
        run_pipeline(self.config, executor=self.executor)
        merged = self.config.output_root / "stages/merge/llm_merged_features.jsonl"
        merged.write_text("not JSON")
        result = run_pipeline(self.config, executor=self.executor)
        self.assertEqual(result["stages"]["merge"]["status"], "completed")
        self.assertEqual(len(read_jsonl(merged)), 1)
