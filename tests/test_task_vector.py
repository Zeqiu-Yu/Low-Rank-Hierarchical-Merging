"""Unit tests for :mod:`lhm.task_vector` (no model loading)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lhm.task_vector import TaskVector, sum_task_vectors, get_param_names_to_merge
from lhm.layer_utils import LayerKeyMapper


def state_dict(seed: int, names=("model.layers.0.fc.weight", "model.layers.1.fc.weight")):
    torch.manual_seed(seed)
    return {name: torch.randn(4, 4) for name in names}


class TestTaskVector(unittest.TestCase):
    def test_subtraction(self):
        base, ft = state_dict(0), state_dict(1)
        tv = TaskVector(pretrained_model=base, finetuned_model=ft)
        for name in base:
            self.assertTrue(torch.allclose(tv[name], ft[name] - base[name]))

    def test_exclusion(self):
        base = state_dict(0, names=("model.layers.0.fc.weight", "model.embed_tokens.weight"))
        ft = state_dict(1, names=("model.layers.0.fc.weight", "model.embed_tokens.weight"))
        tv = TaskVector(
            pretrained_model=base,
            finetuned_model=ft,
            exclude_param_names_regex=[r".*embed_tokens\.weight"],
        )
        self.assertEqual(list(tv.keys()), ["model.layers.0.fc.weight"])

    def test_addition_and_scaling(self):
        base, a, b = state_dict(0), state_dict(1), state_dict(2)
        tv_a = TaskVector(pretrained_model=base, finetuned_model=a)
        tv_b = TaskVector(pretrained_model=base, finetuned_model=b)
        summed = tv_a + tv_b
        for name in base:
            self.assertTrue(
                torch.allclose(summed[name], (a[name] - base[name]) + (b[name] - base[name]))
            )
        scaled = tv_a.scaled(0.5)
        for name in base:
            self.assertTrue(torch.allclose(scaled[name], 0.5 * (a[name] - base[name])))

    def test_sum_task_vectors(self):
        base, a, b = state_dict(0), state_dict(1), state_dict(2)
        total = sum_task_vectors(
            [
                TaskVector(pretrained_model=base, finetuned_model=a),
                TaskVector(pretrained_model=base, finetuned_model=b),
            ]
        )
        for name in base:
            self.assertTrue(
                torch.allclose(total[name], (a[name] - base[name]) + (b[name] - base[name]))
            )
        self.assertEqual(len(sum_task_vectors([]).task_vector_param_dict), 0)

    def test_apply_layer_coefficients(self):
        base, ft = state_dict(0), state_dict(1)
        tv = TaskVector(pretrained_model=base, finetuned_model=ft)
        mapper = LayerKeyMapper()
        scaled = tv.apply_layer_coefficients(
            {0: 0.5, 1: 0.25}, layer_index_fn=mapper.layer_index
        )
        self.assertTrue(
            torch.allclose(
                scaled["model.layers.0.fc.weight"],
                0.5 * (ft["model.layers.0.fc.weight"] - base["model.layers.0.fc.weight"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                scaled["model.layers.1.fc.weight"],
                0.25 * (ft["model.layers.1.fc.weight"] - base["model.layers.1.fc.weight"]),
            )
        )

    def test_combine_with_pretrained(self):
        base, ft = state_dict(0), state_dict(1)
        tv = TaskVector(pretrained_model=base, finetuned_model=ft)
        merged = tv.combine_with_pretrained_model(base, scaling_coefficient=0.7)
        for name in base:
            self.assertTrue(
                torch.allclose(merged[name], base[name] + 0.7 * (ft[name] - base[name]))
            )

    def test_squared_norms(self):
        base, ft = state_dict(0), state_dict(1)
        tv = TaskVector(pretrained_model=base, finetuned_model=ft)
        norms = tv.squared_norms()
        expected = float(torch.sum((ft["model.layers.0.fc.weight"] - base["model.layers.0.fc.weight"]) ** 2))
        self.assertAlmostEqual(norms["model.layers.0.fc.weight"], expected, places=5)

    def test_get_param_names_to_merge(self):
        names = ["a.b", "embed_tokens.weight", "c.d"]
        self.assertEqual(
            get_param_names_to_merge(names, [r".*embed_tokens\.weight"]), ["a.b", "c.d"]
        )


if __name__ == "__main__":
    unittest.main()
