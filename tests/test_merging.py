"""Unit tests for the layer-adaptive merging engine (Phases 1 & 5)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _toy import make_pair
from lhm.merging import (
    build_task_vector_dict,
    copy_params_to_model,
    dare_merge_layer,
    merge_with_coefficients,
    ties_merge_layer,
)


def toy_dicts(seed: int = 0):
    torch.manual_seed(seed)
    base = {
        "model.layers.0.fc.weight": torch.zeros(3, 3),
        "model.layers.1.fc.weight": torch.zeros(3, 3),
        "model.embed_tokens.weight": torch.zeros(4, 3),
        "model.norm.weight": torch.zeros(3),
    }
    reason = {
        "model.layers.0.fc.weight": torch.full((3, 3), 0.1),
        "model.layers.1.fc.weight": torch.full((3, 3), 0.2),
        "model.embed_tokens.weight": torch.full((4, 3), 9.0),
        "model.norm.weight": torch.full((3,), 0.3),
    }
    return base, reason


EXCLUDE = [r".*embed_tokens\.weight"]


class TestTaskVectorDecomposition(unittest.TestCase):
    def test_build_task_vector_dict(self):
        base, reason = toy_dicts()
        delta = build_task_vector_dict(base, reason, EXCLUDE)
        self.assertIn("model.layers.0.fc.weight", delta)
        self.assertNotIn("model.embed_tokens.weight", delta)
        self.assertTrue(
            torch.allclose(
                delta["model.layers.1.fc.weight"],
                reason["model.layers.1.fc.weight"] - base["model.layers.1.fc.weight"],
            )
        )


class TestTaskArithmeticMerging(unittest.TestCase):
    def test_uniform_coefficient(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 0.7, 1: 0.7}, method="ta",
            exclude_param_names_regex=EXCLUDE,
        )
        self.assertTrue(
            torch.allclose(
                merged["model.layers.0.fc.weight"], 0.7 * torch.full((3, 3), 0.1)
            )
        )
        self.assertTrue(
            torch.allclose(
                merged["model.layers.1.fc.weight"], 0.7 * torch.full((3, 3), 0.2)
            )
        )

    def test_per_layer_coefficients(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 0.5, 1: 0.25}, method="ta",
            exclude_param_names_regex=EXCLUDE,
        )
        self.assertTrue(
            torch.allclose(
                merged["model.layers.0.fc.weight"], 0.5 * torch.full((3, 3), 0.1)
            )
        )
        self.assertTrue(
            torch.allclose(
                merged["model.layers.1.fc.weight"], 0.25 * torch.full((3, 3), 0.2)
            )
        )

    def test_excluded_params_are_copied_from_base(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 1.0, 1: 1.0}, method="ta",
            exclude_param_names_regex=EXCLUDE,
        )
        self.assertTrue(
            torch.allclose(
                merged["model.embed_tokens.weight"], base["model.embed_tokens.weight"]
            )
        )

    def test_non_layer_coefficient(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 1.0, 1: 1.0}, method="ta",
            exclude_param_names_regex=[], non_layer_coefficient=0.5,
        )
        # norm has layer index -1 -> scaled by non_layer_coefficient
        self.assertTrue(
            torch.allclose(merged["model.norm.weight"], 0.5 * torch.full((3,), 0.3))
        )

    def test_unsupported_method(self):
        base, reason = toy_dicts()
        with self.assertRaises(ValueError):
            merge_with_coefficients(base, reason, {0: 0.5}, method="bogus")


class TestTiesLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.delta = {"w": torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])}

    def test_zero_trim_rate_is_identity(self):
        out = ties_merge_layer([self.delta], param_value_mask_rate=0.0)
        self.assertTrue(torch.allclose(out["w"], self.delta["w"]))

    def test_trimming_zeros_smallest_magnitudes(self):
        out = ties_merge_layer([self.delta], param_value_mask_rate=0.4)  # drop 2 of 5
        nonzero = int((out["w"] != 0).sum())
        self.assertEqual(nonzero, 3)

    def test_full_trim_rate_zeros_everything(self):
        out = ties_merge_layer([self.delta], param_value_mask_rate=1.0)
        self.assertTrue(torch.allclose(out["w"], torch.zeros_like(self.delta["w"])))

    def test_sign_conflict_resolution(self):
        tv1 = {"w": torch.tensor([0.5, -0.5, 0.5, -0.5])}
        tv2 = {"w": torch.tensor([0.5, 0.5, -0.5, -0.5])}
        out = ties_merge_layer([tv1, tv2], param_value_mask_rate=0.0)
        for i in range(4):
            value = float(out["w"][i])
            if abs(value) > 0:
                self.assertTrue(
                    (value > 0) == (float(tv1["w"][i]) > 0)
                    or (value > 0) == (float(tv2["w"][i]) > 0)
                )


class TestDareLayer(unittest.TestCase):
    def test_zero_mask_rate_is_identity(self):
        delta = {"w": torch.randn(10)}
        out = dare_merge_layer(
            delta, mask_rate=0.0, mask_strategy="random",
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue(torch.allclose(out["w"], delta["w"]))

    def test_full_mask_rate_is_zero(self):
        delta = {"w": torch.randn(10)}
        out = dare_merge_layer(delta, mask_rate=1.0, mask_strategy="random")
        self.assertTrue(torch.allclose(out["w"], torch.zeros_like(delta["w"])))

    def test_random_mask_drops_fraction(self):
        delta = {"w": torch.ones(1000)}
        out = dare_merge_layer(
            delta, mask_rate=0.3, mask_strategy="random",
            use_weight_rescale=False,
            generator=torch.Generator().manual_seed(0),
        )
        kept = int((out["w"] != 0).sum())
        self.assertGreater(kept, 600)
        self.assertLess(kept, 800)

    def test_invalid_mask_rate(self):
        with self.assertRaises(ValueError):
            dare_merge_layer({"w": torch.randn(3)}, mask_rate=1.5)


class TestAimMerging(unittest.TestCase):
    def test_zero_omega_reverts_to_base(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 0.5, 1: 0.5}, method="aim",
            exclude_param_names_regex=EXCLUDE, aim_omega=0.0,
        )
        self.assertTrue(
            torch.allclose(
                merged["model.layers.0.fc.weight"], base["model.layers.0.fc.weight"]
            )
        )

    def test_positive_omega_moves_towards_reasoning(self):
        base, reason = toy_dicts()
        merged = merge_with_coefficients(
            base, reason, {0: 0.5, 1: 0.5}, method="aim",
            exclude_param_names_regex=EXCLUDE, aim_omega=0.4,
        )
        delta = merged["model.layers.0.fc.weight"] - base["model.layers.0.fc.weight"]
        self.assertTrue(torch.all(delta > 0))


class TestCopyParams(unittest.TestCase):
    def test_copy_into_model(self):
        base, reasoning = make_pair(seed=0)
        merged = merge_with_coefficients(
            base, reasoning, {0: 0.5, 1: 0.5, 2: 0.5}, method="ta",
            exclude_param_names_regex=[r".*embed_tokens\.weight", r".*lm_head\.weight"],
        )
        copy_params_to_model(merged, base)
        for name, param in base.named_parameters():
            if name in merged:
                self.assertTrue(torch.allclose(param.data, merged[name]))


if __name__ == "__main__":
    unittest.main()
