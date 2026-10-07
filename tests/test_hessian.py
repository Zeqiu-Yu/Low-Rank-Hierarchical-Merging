"""Unit tests for the low-rank Hessian trace estimator (Phase 2)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _toy import ToyLM, QuadraticModel
from lhm.hessian import (
    HessianTraceResult,
    LowRankHessianEstimator,
    exact_trace_of_matrix,
    hutchinson_trace,
    hutchinson_trace_numpy,
    hutchinson_trace_of_matrix,
)


def random_psd(dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    b = torch.randn(dim, dim, dtype=torch.float64)
    return b @ b.t() / dim


class TestHutchinsonCore(unittest.TestCase):
    def test_matrix_estimator_matches_exact_trace(self):
        A = random_psd(64, seed=1)
        exact = exact_trace_of_matrix(A)
        estimate = hutchinson_trace_of_matrix(A, k=1024, seed=42)
        self.assertGreater(exact, 0.0)
        self.assertLess(abs(estimate - exact) / exact, 0.15)

    def test_generic_matvec(self):
        A = random_psd(32, seed=2)
        exact = exact_trace_of_matrix(A)
        generator = torch.Generator().manual_seed(7)
        estimate = hutchinson_trace(
            matvec=lambda v: A @ v,
            dim=32,
            k=2048,
            dtype=torch.float64,
            generator=generator,
        )
        self.assertLess(abs(estimate - exact) / exact, 0.10)

    def test_hutchinson_is_unbiased_on_average(self):
        A = random_psd(24, seed=3)
        exact = exact_trace_of_matrix(A)
        estimates = [
            hutchinson_trace_of_matrix(A, k=4, seed=s) for s in range(200)
        ]
        self.assertLess(abs(np.mean(estimates) - exact) / exact, 0.10)

    def test_numpy_reference(self):
        rng = np.random.default_rng(0)
        b = rng.normal(size=(40, 40))
        A = b @ b.T
        exact = float(np.trace(A))
        estimate = hutchinson_trace_numpy(A, k=2048, seed=5)
        self.assertLess(abs(estimate - exact) / exact, 0.10)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            hutchinson_trace(lambda v: v, dim=0, k=4)
        with self.assertRaises(ValueError):
            hutchinson_trace(lambda v: v, dim=4, k=0)


class TestDoubleBackward(unittest.TestCase):
    """The Hessian-vector product must reproduce a closed-form Hessian trace."""

    def test_quadratic_model_trace_matches_closed_form(self):
        torch.manual_seed(0)
        model = QuadraticModel(vocab_size=16, in_features=6, out_features=5)
        input_ids = torch.randint(0, 16, (2, 7))
        with torch.no_grad():
            exact = model.exact_trace(input_ids)

        estimator = LowRankHessianEstimator(model=model)
        # Feeding the same token sequence makes the comparison exact-to-target.
        result = estimator.estimate(
            k=1024, input_ids=input_ids, seed=0
        )
        self.assertIn(0, result.traces)
        estimate = result.traces[0]
        self.assertGreater(exact, 0.0)
        self.assertLess(abs(estimate - exact) / exact, 0.05)

    def test_determinism_with_seed(self):
        torch.manual_seed(0)
        model = QuadraticModel()
        estimator = LowRankHessianEstimator(model=model)
        first = estimator.estimate(k=32, num_sequences=1, seq_len=5, vocab_size=16, seed=123)
        second = estimator.estimate(k=32, num_sequences=1, seq_len=5, vocab_size=16, seed=123)
        for layer in first.traces:
            self.assertAlmostEqual(first.traces[layer], second.traces[layer], places=8)

    def test_explicit_input_ids_are_honoured(self):
        torch.manual_seed(0)
        model = QuadraticModel()
        ids = torch.randint(0, 16, (1, 4))
        estimator = LowRankHessianEstimator(model=model)
        result = estimator.estimate(k=8, input_ids=ids, seed=0)
        self.assertEqual(result.num_sequences, 1)
        with torch.no_grad():
            exact = model.exact_trace(ids)
        # k=8 is a noisy estimate, so only require the same order of magnitude
        self.assertGreater(result.traces[0], 0.0)
        self.assertLess(abs(result.traces[0] - exact) / exact, 0.6)

    def test_accepts_a_list_of_sequences(self):
        torch.manual_seed(0)
        model = QuadraticModel()
        seqs = [torch.randint(0, 16, (1, 3)) for _ in range(3)]
        estimator = LowRankHessianEstimator(model=model)
        result = estimator.estimate(k=4, input_ids=seqs, seed=0)
        self.assertEqual(result.num_sequences, 3)


class TestModelEstimator(unittest.TestCase):
    def test_toy_lm_runs_and_returns_finite_traces(self):
        torch.manual_seed(0)
        model = ToyLM(vocab_size=32, hidden=8, num_layers=3)
        estimator = LowRankHessianEstimator(model=model)
        result = estimator.estimate(k=8, num_sequences=2, seq_len=12, vocab_size=32, seed=0)
        self.assertEqual(sorted(result.traces), [0, 1, 2])
        for value in result.traces.values():
            self.assertTrue(np.isfinite(value))
        self.assertEqual(result.num_probes, 8)
        self.assertEqual(result.num_sequences, 2)

    def test_include_non_layer(self):
        torch.manual_seed(0)
        model = ToyLM(vocab_size=32, hidden=8, num_layers=2)
        estimator = LowRankHessianEstimator(model=model)
        result = estimator.estimate(
            k=8, num_sequences=1, seq_len=8, vocab_size=32, seed=0, include_non_layer=True
        )
        self.assertIn(-1, result.traces)

    def test_result_json_roundtrip(self):
        result = HessianTraceResult(traces={0: 1.5, 1: 2.5}, num_probes=4, num_sequences=2, seq_len=8)
        restored = HessianTraceResult.from_dict(result.to_dict())
        self.assertEqual(restored.traces, result.traces)
        self.assertEqual(restored.num_probes, result.num_probes)

    def test_layer_params_grouping(self):
        torch.manual_seed(0)
        model = ToyLM(num_layers=3)
        estimator = LowRankHessianEstimator(model=model)
        # ``layer_params`` also contains the non-layer group (-1): the final
        # norm and the LM head; ``layer_indices()`` filters those out.
        self.assertEqual(sorted(estimator.layer_params), [-1, 0, 1, 2])
        self.assertEqual(estimator.layer_indices(), [0, 1, 2])
        self.assertEqual(len(estimator.layer_params[0]), 4)  # fc1.weight/bias, fc2.weight/bias


if __name__ == "__main__":
    unittest.main()
