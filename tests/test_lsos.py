"""Unit tests for the LSOS index (Phase 3)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _toy import make_pair
from lhm.lsos import (
    LSOSResult,
    aggregate_squared_norms_by_layer,
    compute_lsos,
    task_vector_squared_norms,
)


class TestAggregation(unittest.TestCase):
    def test_aggregate_by_layer(self):
        norms = {
            "model.layers.0.fc1.weight": 1.0,
            "model.layers.0.fc2.weight": 2.0,
            "model.layers.1.fc1.weight": 4.0,
            "model.norm.weight": 100.0,
        }
        grouped = aggregate_squared_norms_by_layer(norms)
        self.assertEqual(grouped, {0: 3.0, 1: 4.0})
        grouped_with_non_layer = aggregate_squared_norms_by_layer(
            norms, include_non_layer=True
        )
        self.assertEqual(grouped_with_non_layer[-1], 100.0)

    def test_task_vector_squared_norms(self):
        delta = {
            "model.layers.0.fc.weight": torch.ones(2, 3),   # 6
            "model.layers.1.fc.weight": 2.0 * torch.ones(2, 3),  # 24
        }
        norms = task_vector_squared_norms(delta)
        self.assertAlmostEqual(norms[0], 6.0, places=5)
        self.assertAlmostEqual(norms[1], 24.0, places=5)

    def test_task_vector_squared_norms_from_pair(self):
        base, reasoning = make_pair(seed=0)
        from lhm.merging import build_task_vector_dict

        delta = build_task_vector_dict(base, reasoning)
        norms = task_vector_squared_norms(delta)
        self.assertEqual(sorted(norms), [0, 1, 2])
        for value in norms.values():
            self.assertGreater(value, 0.0)


class TestLSOS(unittest.TestCase):
    def test_product_definition(self):
        traces = {0: 2.0, 1: 0.5, 2: 1.0}
        norms = {0: 3.0, 1: 4.0, 2: 0.0}
        result = compute_lsos(traces, norms)
        self.assertAlmostEqual(result.scores[0], 6.0)
        self.assertAlmostEqual(result.scores[1], 2.0)
        self.assertAlmostEqual(result.scores[2], 0.0)
        self.assertEqual(result.L, 3)

    def test_missing_layers_default_to_zero(self):
        result = compute_lsos({0: 1.0}, {1: 1.0})
        self.assertEqual(sorted(result.scores), [0, 1])
        self.assertEqual(result.scores[0], 0.0)
        self.assertEqual(result.scores[1], 0.0)

    def test_explicit_layer_set(self):
        result = compute_lsos({0: 2.0}, {0: 5.0}, layers=[0, 1, 2])
        self.assertEqual(sorted(result.scores), [0, 1, 2])
        self.assertAlmostEqual(result.scores[0], 10.0)
        self.assertEqual(result.scores[2], 0.0)

    def test_top_k(self):
        result = compute_lsos({0: 1.0, 1: 5.0, 2: 3.0}, {0: 1.0, 1: 1.0, 2: 1.0})
        self.assertEqual(list(result.top_k(2)), [1, 2])

    def test_loss_perturbation_proxy(self):
        result = compute_lsos({0: 2.0}, {0: 3.0})
        perturbation = result.approx_loss_perturbation(coefficient=1.0)
        self.assertAlmostEqual(perturbation[0], 0.5 * 6.0)
        perturbation_half = result.approx_loss_perturbation(coefficient=0.5)
        self.assertAlmostEqual(perturbation_half[0], 0.5 * 0.25 * 6.0)

    def test_as_vector(self):
        result = compute_lsos({0: 1.0, 2: 2.0}, {0: 1.0, 2: 1.0})
        self.assertEqual(result.as_vector(), [1.0, 0.0, 2.0])

    def test_json_roundtrip(self):
        result = compute_lsos({0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0})
        restored = LSOSResult.from_dict(result.to_dict())
        self.assertEqual(restored.scores, result.scores)
        self.assertEqual(restored.hessian_traces, result.hessian_traces)


if __name__ == "__main__":
    unittest.main()
