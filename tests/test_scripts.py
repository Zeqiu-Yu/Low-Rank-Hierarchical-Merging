"""Smoke tests for the experiment scripts (no model is ever loaded).

Importing each script module validates that the CLI plumbing, config-file
handling and helper functions stay healthy without requiring torch /
transformers models.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


class TestScriptImports(unittest.TestCase):
    def test_modules_import(self):
        import _common  # noqa: F401
        import build_lhm_model  # noqa: F401
        import collect_results  # noqa: F401
        import extract_hessian  # noqa: F401
        import merge_baselines  # noqa: F401
        import plot_figures  # noqa: F401
        import run_ablation  # noqa: F401
        import run_hyperparam_sweep  # noqa: F401
        import tune_lambda  # noqa: F401


class TestCommonHelpers(unittest.TestCase):
    def test_config_file_roundtrip_and_aliases(self):
        import _common

        payload = {
            "base_model": "Qwen/Qwen2.5-Math-1.5B",
            "reasoning_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "rank": 16,
            "alpha": 0.15,
            "lambda_bar": 0.4,
            "trace_cache_path": "traces/base.json",
            "exclude_param_names_regex": [r".*embed_tokens\.weight"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            loaded = _common.load_config_file(path)
            defaults = _common.config_to_cli_defaults(loaded)
        self.assertEqual(defaults["trace_cache"], "traces/base.json")
        self.assertEqual(defaults["exclude_param_names_regex"], r".*embed_tokens\.weight")
        self.assertEqual(defaults["lambda_bar"], 0.4)

    def test_build_config_from_args(self):
        import _common

        parser = _common.add_common_arguments(
            argparse.ArgumentParser(prog="test")
        )
        parser.add_argument("--output_model_dir", default="out")
        args = parser.parse_args(
            ["--scale", "7b", "--method", "aim", "--rank", "32", "--alpha", "0.2"]
        )
        config = _common.build_config(args)
        self.assertEqual(config.rank, 32)
        self.assertEqual(config.alpha, 0.2)
        self.assertEqual(config.method, "aim")
        self.assertIn("Qwen2.5-Math-7B", config.base_model)

    def test_unsupported_config_extension(self):
        import _common

        with self.assertRaises(SystemExit):
            _common.load_config_file("config.txt")


class TestAblationHelpers(unittest.TestCase):
    def test_simple_linear_coefficients(self):
        import numpy as np

        import run_ablation

        scores = {0: 1.0, 1: 3.0}
        coefficients = run_ablation.simple_linear_coefficients(scores, 0.5)
        self.assertAlmostEqual(float(np.mean(list(coefficients.values()))), 0.5, places=6)
        self.assertLess(coefficients[0], coefficients[1])

    def test_scalar_coefficients(self):
        import run_ablation

        coefficients = run_ablation.scalar_coefficients({0: 1.0, 1: 2.0}, 0.7)
        self.assertEqual(coefficients, {0: 0.7, 1: 0.7})


class TestSweepGrids(unittest.TestCase):
    def test_grid_for(self):
        import run_hyperparam_sweep as sweep

        self.assertEqual(sweep.grid_for("rank", None), [4, 8, 16, 32])
        self.assertEqual(sweep.grid_for("lambda", None), [0.3, 0.4, 0.5, 0.6])
        self.assertEqual(
            sweep.grid_for("alpha", "0.1,0.2"), [0.1, 0.2]
        )


if __name__ == "__main__":
    unittest.main()
