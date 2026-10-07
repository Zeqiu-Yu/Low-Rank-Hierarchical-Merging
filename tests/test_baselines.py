"""Unit tests for the baseline merging methods (Tables 1-2)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _toy import ToyLM, make_pair
from lhm.activations import collect_activation_norms
from lhm.baselines import (
    acm_layer_coefficients,
    aim_layer_coefficients,
    dare_merging,
    model_soup,
    orca_layer_coefficients,
    task_arithmetic,
    ties_merging,
    uniform_coefficients,
)


class TestCoefficientProviders(unittest.TestCase):
    def test_uniform(self):
        self.assertEqual(uniform_coefficients(3, 0.7), {0: 0.7, 1: 0.7, 2: 0.7})

    def test_aim_mean_is_omega(self):
        norms = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        coefficients = aim_layer_coefficients(norms, omega=0.4)
        self.assertAlmostEqual(float(np.mean(list(coefficients.values()))), 0.4, places=6)
        self.assertTrue(all(v > 0 for v in coefficients.values()))

    def test_orca_is_normalised(self):
        norms = {0: 1.0, 1: 2.0, 2: 4.0}
        coefficients = orca_layer_coefficients(norms)
        self.assertAlmostEqual(min(coefficients.values()), 0.0)
        self.assertAlmostEqual(max(coefficients.values()), 1.0)

    def test_acm_mean_is_one(self):
        base = {0: 1.0, 1: 2.0}
        reason = {0: 2.0, 1: 1.0}
        coefficients = acm_layer_coefficients(base, reason, temperature=1.0)
        self.assertAlmostEqual(float(np.mean(list(coefficients.values()))), 1.0, places=6)


class TestBaselineMerging(unittest.TestCase):
    def test_model_soup(self):
        a = {"w": torch.zeros(2)}
        b = {"w": torch.full((2,), 2.0)}
        merged = model_soup([a, b])
        self.assertTrue(torch.allclose(merged["w"], torch.full((2,), 1.0)))

    def test_weighted_model_soup(self):
        a = {"w": torch.zeros(2)}
        b = {"w": torch.full((2,), 4.0)}
        merged = model_soup([a, b], weights=[0.25, 0.75])
        self.assertTrue(torch.allclose(merged["w"], torch.full((2,), 3.0)))

    def test_task_arithmetic_default_coefficient(self):
        base, reason = make_pair(seed=0)
        merged = task_arithmetic(
            base, reason,
            exclude_param_names_regex=[r".*embed_tokens\.weight", r".*lm_head\.weight"],
        )
        name = "model.layers.0.fc1.weight"
        expected = (
            base.state_dict()[name]
            + 0.7 * (reason.state_dict()[name] - base.state_dict()[name])
        )
        self.assertTrue(torch.allclose(merged[name], expected, atol=1e-6))

    def test_ties_and_dare_run(self):
        base, reason = make_pair(seed=0)
        exclude = [r".*embed_tokens\.weight", r".*lm_head\.weight"]
        ties = ties_merging(base, reason, param_value_mask_rate=0.2, exclude_param_names_regex=exclude)
        dare = dare_merging(base, reason, mask_rate=0.2, exclude_param_names_regex=exclude)
        for name in base.state_dict():
            self.assertEqual(ties[name].shape, base.state_dict()[name].shape)
            self.assertEqual(dare[name].shape, base.state_dict()[name].shape)


class TestActivationCollector(unittest.TestCase):
    def test_collects_per_layer_norms(self):
        torch.manual_seed(0)
        model = ToyLM(vocab_size=32, hidden=8, num_layers=3)
        input_ids = torch.randint(0, 32, (1, 10))
        norms = collect_activation_norms(model, input_ids)
        self.assertEqual(sorted(norms), [0, 1, 2])
        for value in norms.values():
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_aggregate_modes(self):
        torch.manual_seed(0)
        model = ToyLM(vocab_size=32, hidden=8, num_layers=2)
        input_ids = torch.randint(0, 32, (1, 6))
        mean = collect_activation_norms(model, input_ids, aggregate="mean")
        rms = collect_activation_norms(model, input_ids, aggregate="rms")
        for layer in mean:
            self.assertAlmostEqual(rms[layer], mean[layer] ** 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
