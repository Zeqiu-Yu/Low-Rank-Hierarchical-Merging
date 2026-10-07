"""Unit tests for :mod:`lhm.layer_utils` (no torch / model required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lhm.layer_utils import (
    LayerKeyMapper,
    group_param_names_by_layer,
    num_layers,
    filter_layer_param_names,
)


class TestLayerKeyMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = LayerKeyMapper()

    def test_layer_indices(self):
        cases = {
            "model.layers.0.self_attn.q_proj.weight": 0,
            "model.layers.12.mlp.down_proj.weight": 12,
            "model.layers.27.input_layernorm.weight": 27,
            "transformer.h.3.attn.c_attn.weight": -1,
            "model.embed_tokens.weight": -1,
            "lm_head.weight": -1,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.mapper.layer_index(name), expected)

    def test_is_layer_param(self):
        self.assertTrue(self.mapper.is_layer_param("model.layers.1.fc.weight"))
        self.assertFalse(self.mapper.is_layer_param("model.norm.weight"))

    def test_grouping_is_ordered_and_complete(self):
        names = [
            "model.layers.1.fc.weight",
            "model.layers.0.fc.weight",
            "model.layers.1.fc.bias",
            "model.norm.weight",
        ]
        groups = self.mapper.group_param_names_by_layer(names)
        self.assertEqual(list(groups.keys()), [-1, 0, 1])
        self.assertEqual(
            groups[1], ["model.layers.1.fc.weight", "model.layers.1.fc.bias"]
        )

    def test_num_layers(self):
        names = ["model.layers.0.a", "model.layers.5.b", "model.norm.weight"]
        self.assertEqual(num_layers(names), 6)
        self.assertEqual(num_layers(["model.norm.weight"]), 0)


class TestHelpers(unittest.TestCase):
    def test_functional_grouping(self):
        groups = group_param_names_by_layer(
            ["model.layers.2.x", "model.layers.0.y"]
        )
        self.assertEqual(list(groups.keys()), [0, 2])

    def test_filter_excludes_and_layer_only(self):
        names = [
            "model.layers.0.fc.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ]
        kept = filter_layer_param_names(
            names,
            exclude_param_names_regex=[r".*embed_tokens\.weight", r".*lm_head\.weight"],
            only_layer_params=True,
        )
        self.assertEqual(kept, ["model.layers.0.fc.weight"])

    def test_filter_without_layer_restriction(self):
        names = ["model.layers.0.fc.weight", "model.norm.weight"]
        kept = filter_layer_param_names(names, only_layer_params=False)
        self.assertEqual(set(kept), set(names))


if __name__ == "__main__":
    unittest.main()
