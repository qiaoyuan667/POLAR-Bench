"""Offline tests for public configuration names and pre-release compatibility."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ab_eval", ROOT / "scripts/ab_eval.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class PublicConfigurationTests(unittest.TestCase):
    args = ["--dataset", "data.json", "--model-a", "model-a", "--domains", "medical"]

    def test_public_cli(self):
        args = evaluator.parse_args(self.args + ["--base-url", "https://example.test/v1"])
        self.assertEqual(args.base_url, "https://example.test/v1")
        self.assertEqual(args.model_a_provider, "openai-compatible")

    def test_legacy_cli(self):
        args = evaluator.parse_args(self.args + [
            "--Anonymous-base-url", "https://example.test/v1",
            "--model-a-provider", "Anonymous",
        ])
        self.assertEqual(args.base_url, "https://example.test/v1")
        self.assertEqual(args.model_a_provider, "openai-compatible")

    def test_base_url_environment_and_cli_override(self):
        with patch.dict(os.environ, {"POLAR_BASE_URL": "https://env.test/v1"}, clear=True):
            self.assertEqual(evaluator.parse_args(self.args).base_url, "https://env.test/v1")
            self.assertEqual(evaluator.parse_args(self.args + [
                "--base-url", "https://cli.test/v1"]).base_url, "https://cli.test/v1")

    def test_no_placeholder_default(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.parse_args(self.args)

    def test_help_uses_public_names(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            evaluator.parse_args(["--help"])
        self.assertNotIn("Anonymous", output.getvalue())
        self.assertIn("--base-url", output.getvalue())
        self.assertIn("openai-compatible", output.getvalue())

    def test_key_precedence_and_legacy_fallbacks(self):
        with patch.dict(os.environ, {"POLAR_API_KEY": "new", "CSCS_SERVING_API": "cscs",
                                     "ANonymous_SERVING_API": "old"}, clear=True):
            self.assertEqual(evaluator.get_compatible_api_key(), "new")
            del os.environ["POLAR_API_KEY"]
            self.assertEqual(evaluator.get_compatible_api_key(), "cscs")
            del os.environ["CSCS_SERVING_API"]
            self.assertEqual(evaluator.get_compatible_api_key(), "old")

    def config(self):
        return evaluator.build_run_config(
            dataset_path="data.json", model_a_names=["model-a"], domains=["medical"],
            samples_per_domain=2, max_rounds=6, seed=42,
            output_path="summary.json", output_details_path="details.json",
            base_url="https://example.test/v1", model_b_name="model-b",
            model_a_provider="openai-compatible", max_workers=5,
            deterministic_llm=True, defense="none",
        )

    def test_legacy_checkpoint_accepted(self):
        current = self.config()
        old = dict(current)
        old["Anonymous_models"] = old.pop("model_a_names")
        old["Anonymous_base_url"] = old.pop("base_url")
        old["model_a_provider"] = "Anonymous"
        evaluator.validate_checkpoint_config({"config": old}, current)
        self.assertEqual(evaluator.normalize_run_config(old), current)
        self.assertIn("Anonymous_models", old)  # Normalization does not mutate input.

    def test_checkpoint_mismatch_still_rejected(self):
        current = self.config()
        old = dict(current, base_url="https://different.test/v1")
        with self.assertRaises(RuntimeError):
            evaluator.validate_checkpoint_config({"config": old}, current)

    def test_new_output_config_uses_public_names(self):
        current = self.config()
        for maker in (evaluator.make_initial_summary_results, evaluator.make_initial_detail_results):
            config = maker(current, 1, 1, {}, 1)["config"]
            self.assertIn("model_a_names", config)
            self.assertIn("base_url", config)
            self.assertNotIn("Anonymous_models", config)
        checkpoint = evaluator.make_initial_checkpoint(current, [])
        self.assertEqual(set(checkpoint["models"]), {"model-a"})


if __name__ == "__main__":
    unittest.main()
