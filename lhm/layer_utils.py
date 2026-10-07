"""Utilities for mapping parameter / module names onto transformer layers.

Transformer LLMs (Qwen2, Llama, ...) name their blocks
``model.layers.<idx>.<...>``.  LHM operates on *layers*, so every parameter has
to be attributed to exactly one layer index before the Hessian trace, the LSOS
index or the merge coefficient can be computed for it.

The helpers here are intentionally dependency-free (pure Python + regex) so
that they can be unit-tested without torch, numpy or any model.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_LAYER_PATTERN = r"(?:^|\.)layers\.(\d+)(?:\.|$)"


class LayerKeyMapper:
    """Map fully-qualified parameter names onto transformer layer indices.

    Parameters
    ----------
    layer_pattern:
        A regex with exactly one capture group that matches the layer index
        inside a parameter name.  The default matches ``layers.<idx>``.
    non_layer_key:
        Key stored for parameters that do not belong to a transformer block
        (embeddings, final norm, ``lm_head`` ...).  Defaults to ``-1``.
    """

    def __init__(
        self,
        layer_pattern: str = DEFAULT_LAYER_PATTERN,
        non_layer_key: int = -1,
    ) -> None:
        self.layer_pattern = re.compile(layer_pattern)
        self.non_layer_key = non_layer_key

    def layer_index(self, param_name: str) -> int:
        """Return the layer index of ``param_name`` (or ``non_layer_key``)."""
        match = self.layer_pattern.search(param_name)
        if match is None:
            return self.non_layer_key
        return int(match.group(1))

    def is_layer_param(self, param_name: str) -> bool:
        return self.layer_index(param_name) != self.non_layer_key

    def group_param_names_by_layer(
        self, param_names: Iterable[str]
    ) -> "OrderedDict[int, List[str]]":
        """Group ``param_names`` by layer index, preserving first-seen order."""
        groups: "OrderedDict[int, List[str]]" = OrderedDict()
        for name in param_names:
            groups.setdefault(self.layer_index(name), []).append(name)
        return OrderedDict(sorted(groups.items(), key=lambda kv: kv[0]))


def group_param_names_by_layer(
    param_names: Iterable[str],
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    non_layer_key: int = -1,
) -> "OrderedDict[int, List[str]]":
    """Functional wrapper around :meth:`LayerKeyMapper.group_param_names_by_layer`."""
    return LayerKeyMapper(layer_pattern, non_layer_key).group_param_names_by_layer(
        param_names
    )


def num_layers(
    param_names: Iterable[str], layer_pattern: str = DEFAULT_LAYER_PATTERN
) -> int:
    """Number of *contiguous* transformer layers discovered in ``param_names``.

    Returns the highest discovered layer index + 1, which is the natural ``L``
    used by DCCM.  Non-layer parameters are ignored.
    """
    mapper = LayerKeyMapper(layer_pattern)
    indices = [mapper.layer_index(n) for n in param_names]
    layer_indices = [i for i in indices if i != mapper.non_layer_key]
    if not layer_indices:
        return 0
    return max(layer_indices) + 1


def layer_of_module(module_name: str, layer_pattern: str = DEFAULT_LAYER_PATTERN) -> int:
    """Layer index for a module name such as ``model.layers.12.mlp.down_proj``."""
    return LayerKeyMapper(layer_pattern).layer_index(module_name)


def filter_layer_param_names(
    param_names: Sequence[str],
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    only_layer_params: bool = True,
) -> List[str]:
    """Apply the exclude-regex list and (optionally) keep only layer params."""
    patterns = list(exclude_param_names_regex or [])
    mapper = LayerKeyMapper()
    kept: List[str] = []
    for name in param_names:
        if any(re.match(p, name) for p in patterns):
            continue
        if only_layer_params and not mapper.is_layer_param(name):
            continue
        kept.append(name)
    return kept
