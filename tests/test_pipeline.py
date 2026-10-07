"""End-to-end tests: full LHM pipeline plus IO / reporting utilities.

These tests never download or load a real language model -- every component is
exercised on the toy modules from :mod:`tests._toy`.
"""

import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _toy import make_pair
from lhm.config import LHMConfig, LAMBDA_GRID, ALPHA_GRID
from lhm.io_utils import (
    load_config,
    load_dccm,
    load_hessian_traces,
    load_lsos,
    run_directory,
    save_config,
    save_dccm,
    save_hessian_traces,
    save_lsos,
)
from lhm.merging import LHMMerger
from lhm.report import average_over_benchmarks, render_markdown_table, summarize_eval_dir


def small_config(**overrides) -> LHMConfig:
    config = LHMConfig(
        rank=8,
        alpha=0.15,
        lambda_bar=0.5,
        num_sequences=2,
        seq_len=8,
        method="ta",
        exclude_param_names_regex=[
            r".*embed_tokens\.weight",
            r".*lm_head\.weight",
        ],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestFullPipeline(unittest.TestCase):
    def test_run_returns_merged_lsos_and_dccm(self):
        torch.manual_seed(0)
        base, reasoning = make_pair(seed=0)
        merger = LHMMerger(config=small_config())

        merged, lsos, dccm = merger.run(base, reasoning)

        self.assertEqual(sorted(lsos.scores), [0, 1, 2])
        self.assertEqual(sorted(dccm.coefficients), [0, 1, 2])
        for value in dccm.coefficients.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        # layer parameters move, embeddings stay frozen
        self.assertFalse(
            torch.allclose(
                merged["model.layers.0.fc1.weight"],
                base.state_dict()["model.layers.0.fc1.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                merged["model.embed_tokens.weight"],
                base.state_dict()["model.embed_tokens.weight"],
            )
        )

    def test_lsos_is_deterministic(self):
        torch.manual_seed(0)
        base, reasoning = make_pair(seed=0)
        merger = LHMMerger(config=small_config())
        merged_a, lsos_a, _ = merger.run(base, reasoning)
        merged_b, lsos_b, _ = merger.run(base, reasoning)
        for layer in lsos_a.scores:
            self.assertAlmostEqual(lsos_a.scores[layer], lsos_b.scores[layer], places=10)
        for name in merged_a:
            self.assertTrue(torch.allclose(merged_a[name], merged_b[name]))

    def test_cached_traces_reuse(self):
        torch.manual_seed(0)
        base, reasoning = make_pair(seed=0)
        merger = LHMMerger(config=small_config())
        _, lsos_a, _ = merger.run(base, reasoning)
        traces = lsos_a.hessian_traces
        merged_b, lsos_b, _ = merger.run(base, reasoning, hessian_traces=traces)
        self.assertEqual(lsos_b.hessian_traces, traces)

    def test_all_methods_run_end_to_end(self):
        for method in ("ta", "ties", "dare", "aim"):
            with self.subTest(method=method):
                torch.manual_seed(0)
                base, reasoning = make_pair(seed=0)
                merger = LHMMerger(config=small_config(method=method))
                merged, _, _ = merger.run(base, reasoning)
                for name, param in base.named_parameters():
                    self.assertEqual(merged[name].shape, param.shape)


class TestConfig(unittest.TestCase):
    def test_paper_defaults(self):
        config = LHMConfig()
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.alpha, 0.15)
        self.assertEqual(config.num_sequences, 32)
        self.assertEqual(config.seq_len, 512)
        self.assertIn(0.3, LAMBDA_GRID)
        self.assertIn(0.15, ALPHA_GRID)

    def test_json_roundtrip(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            save_config(config, path)
            restored = load_config(path)
        self.assertEqual(restored.to_dict(), config.to_dict())


class TestIoUtils(unittest.TestCase):
    def test_artefact_roundtrip(self):
        torch.manual_seed(0)
        base, reasoning = make_pair(seed=0)
        merger = LHMMerger(config=small_config())
        merged, lsos, dccm = merger.run(base, reasoning)
        traces = merger.estimate_hessian_traces(base)

        with tempfile.TemporaryDirectory() as tmp:
            lsos_path = save_lsos(lsos, os.path.join(tmp, "lsos.json"))
            dccm_path = save_dccm(dccm, os.path.join(tmp, "dccm.json"))
            trace_path = save_hessian_traces(traces, os.path.join(tmp, "traces.json"))

            self.assertEqual(load_lsos(lsos_path).scores, lsos.scores)
            self.assertEqual(load_dccm(dccm_path).coefficients, dccm.coefficients)
            self.assertEqual(
                load_hessian_traces(trace_path).traces, traces.traces
            )

    def test_run_directory_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = run_directory(tmp, "1.5B", "aim", 0.5, 0.15, 16)
            self.assertTrue(os.path.isdir(path))
            self.assertIn("1.5B_aim_lam0.50_alpha0.15_k16", path)


class TestReport(unittest.TestCase):
    def _write_jsonl(self, directory, name, rows):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_summarize_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_jsonl(
                os.path.join(tmp, "gsm8k"),
                "test.jsonl",
                [
                    {"code": ["a b c"], "score": [1]},
                    {"code": ["a b"], "score": [0]},
                ],
            )
            self._write_jsonl(
                os.path.join(tmp, "math500"),
                "test.jsonl",
                [{"code": ["a b c d"], "score": [1]}],
            )
            summary = summarize_eval_dir(tmp)
            self.assertAlmostEqual(summary["gsm8k"]["accuracy"], 50.0)
            self.assertAlmostEqual(summary["gsm8k"]["mean_length"], 2.5)
            avg = average_over_benchmarks(summary, benchmarks=["gsm8k", "math500"])
            self.assertAlmostEqual(avg["accuracy"], 75.0)

            table = render_markdown_table({"model": summary}, benchmarks=["gsm8k", "math500"])
            self.assertIn("model", table)
            self.assertIn("50.0", table)


if __name__ == "__main__":
    unittest.main()
