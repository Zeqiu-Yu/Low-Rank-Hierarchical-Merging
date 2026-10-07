"""Unit tests for the Dual-Constraint Coefficient Mapping (Phase 4)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lhm.dccm import (
    DCCMResult,
    apply_fusion_error_bound,
    dccm_anchored_mean,
    dccm_coefficients,
    min_max_normalize,
)


SCORES = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}


class TestNormalization(unittest.TestCase):
    def test_min_max_range(self):
        normalized = min_max_normalize(SCORES)
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[4], 1.0)
        self.assertEqual(set(normalized), set(SCORES))

    def test_constant_scores_map_to_zero(self):
        normalized = min_max_normalize({0: 7.0, 1: 7.0, 2: 7.0})
        self.assertEqual(set(normalized.values()), {0.0})

    def test_empty(self):
        self.assertEqual(min_max_normalize({}), {})


class TestDCCMFormula(unittest.TestCase):
    def test_closed_form_matches_definition(self):
        alpha, lambda_bar = 0.15, 0.5
        result = dccm_coefficients(SCORES, lambda_bar=lambda_bar, alpha=alpha)
        normalized = min_max_normalize(SCORES)
        mean_s = float(np.mean(list(normalized.values())))
        for layer, score in SCORES.items():
            expected = lambda_bar + alpha * (2.0 * normalized[layer] - 1.0 - mean_s)
            self.assertAlmostEqual(result.coefficients[layer], expected, places=10)

    def test_mean_anchoring_property(self):
        alpha, lambda_bar = 0.15, 0.5
        result = dccm_coefficients(SCORES, lambda_bar=lambda_bar, alpha=alpha)
        normalized = min_max_normalize(SCORES)
        mean_s = float(np.mean(list(normalized.values())))
        expected_mean = dccm_anchored_mean(lambda_bar, alpha, mean_s)
        self.assertAlmostEqual(result.achieved_mean, expected_mean, places=10)
        # residual is bounded by alpha (Appendix B.4)
        self.assertLessEqual(abs(result.achieved_mean - lambda_bar), alpha + 1e-9)

    def test_alpha_zero_recovers_uniform_merging(self):
        result = dccm_coefficients(SCORES, lambda_bar=0.42, alpha=0.0)
        for value in result.coefficients.values():
            self.assertAlmostEqual(value, 0.42, places=10)

    def test_monotone_in_lsos(self):
        result = dccm_coefficients(SCORES, lambda_bar=0.5, alpha=0.2)
        coeffs = [result.coefficients[i] for i in sorted(SCORES)]
        self.assertEqual(coeffs, sorted(coeffs))
        self.assertEqual(len(set(round(c, 8) for c in coeffs)), len(coeffs))

    def test_degenerate_scores_fall_back_to_uniform(self):
        result = dccm_coefficients({0: 3.0, 1: 3.0, 2: 3.0}, lambda_bar=0.6, alpha=0.15)
        self.assertTrue(result.degenerate)
        for value in result.coefficients.values():
            self.assertAlmostEqual(value, 0.6)

    def test_clipping(self):
        result = dccm_coefficients({0: 0.0, 1: 1e9}, lambda_bar=0.9, alpha=0.5, clip=True)
        for value in result.coefficients.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_exact_mean_anchor(self):
        result = dccm_coefficients(
            SCORES, lambda_bar=0.45, alpha=0.15, mean_anchor_exact=True
        )
        self.assertAlmostEqual(result.achieved_mean, 0.45, places=10)

    def test_lambda_bar_scales_the_coefficients(self):
        low = dccm_coefficients(SCORES, lambda_bar=0.3, alpha=0.15)
        high = dccm_coefficients(SCORES, lambda_bar=0.6, alpha=0.15)
        for layer in SCORES:
            self.assertAlmostEqual(
                high.coefficients[layer] - low.coefficients[layer], 0.3, places=10
            )

    def test_empty_input(self):
        result = dccm_coefficients({}, lambda_bar=0.5, alpha=0.15)
        self.assertTrue(result.degenerate)
        self.assertEqual(result.coefficients, {})

    def test_as_vector(self):
        result = dccm_coefficients({0: 1.0, 2: 3.0}, lambda_bar=0.5, alpha=0.15)
        vector = result.as_vector()
        self.assertEqual(len(vector), 3)
        self.assertAlmostEqual(vector[0], result.coefficients[0])
        self.assertEqual(vector[1], 0.0)

    def test_json_roundtrip(self):
        result = dccm_coefficients(SCORES, lambda_bar=0.5, alpha=0.15)
        restored = DCCMResult.from_dict(result.to_dict())
        self.assertEqual(restored.coefficients, result.coefficients)
        self.assertAlmostEqual(restored.achieved_mean, result.achieved_mean)


class TestFusionErrorBound(unittest.TestCase):
    def test_bound_is_enforced(self):
        scores = {i: float(10 ** i) for i in range(4)}
        result = dccm_coefficients(scores, lambda_bar=0.9, alpha=0.4, epsilon=0.3)
        self.assertTrue(result.rescaled)
        self.assertLessEqual(result.fusion_error_ratio, 0.3 + 1e-9)

    def test_no_rescale_when_within_budget(self):
        scores = {0: 1.0, 1: 1.0}
        result = dccm_coefficients(scores, lambda_bar=0.1, alpha=0.15, epsilon=0.9)
        self.assertFalse(result.rescaled)

    def test_scaling_preserves_ordering(self):
        scores = {0: 1.0, 1: 4.0, 2: 16.0}
        coefficients = {0: 0.1, 1: 0.5, 2: 0.9}
        bounded, rescaled = apply_fusion_error_bound(coefficients, scores, epsilon=0.05)
        self.assertTrue(rescaled)
        order = [bounded[i] for i in (0, 1, 2)]
        self.assertEqual(order, sorted(order))

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            apply_fusion_error_bound({0: 0.5}, {0: 1.0}, epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
