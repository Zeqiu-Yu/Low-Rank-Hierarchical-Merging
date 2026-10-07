"""Task vectors for long-to-short model merging.

This is a small, dependency-light re-implementation of the task-vector
formalism used in *Editing Models with Task Arithmetic* (Ilharco et al., 2022),
extended with the layer-wise coefficient machinery required by LHM:

    .. math::
        \\theta_{\\mathrm{merged}} = \\theta_0 + \\boldsymbol{\\Lambda} \\odot
        (\\theta_r - \\theta_0)

The class works with any tensor-like object that supports element-wise
``+``/``-``/``*`` (``torch.Tensor`` and ``numpy.ndarray`` both qualify), which
keeps it usable in model-free unit tests.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def _to_param_dict(model_or_dict) -> Dict[str, object]:
    """Accept an ``nn.Module``, a ``state_dict`` or a plain ``dict``."""
    if isinstance(model_or_dict, Mapping):
        return dict(model_or_dict)
    if hasattr(model_or_dict, "named_parameters"):
        return {name: p for name, p in model_or_dict.named_parameters()}
    if hasattr(model_or_dict, "state_dict"):
        return dict(model_or_dict.state_dict())
    raise TypeError(
        "Expected an nn.Module, a state_dict or a dict of named tensors, got "
        f"{type(model_or_dict)!r}."
    )


def get_param_names_to_merge(
    input_param_names: Sequence[str],
    exclude_param_names_regex: Optional[Sequence[str]] = None,
) -> List[str]:
    """Drop parameters whose names match any regex in ``exclude_param_names_regex``."""
    patterns = list(exclude_param_names_regex or [])
    return [
        name
        for name in input_param_names
        if not any(re.match(pattern, name) for pattern in patterns)
    ]


class TaskVector:
    """A per-parameter task vector ``delta = theta_finetuned - theta_pretrained``."""

    def __init__(
        self,
        pretrained_model=None,
        finetuned_model=None,
        exclude_param_names_regex: Optional[Sequence[str]] = None,
        task_vector_param_dict: Optional[Dict[str, object]] = None,
    ) -> None:
        if task_vector_param_dict is not None:
            self.task_vector_param_dict: Dict[str, object] = dict(
                task_vector_param_dict
            )
            return

        if pretrained_model is None or finetuned_model is None:
            raise ValueError(
                "Provide either task_vector_param_dict or both "
                "pretrained_model and finetuned_model."
            )

        pretrained = _to_param_dict(pretrained_model)
        finetuned = _to_param_dict(finetuned_model)
        names = get_param_names_to_merge(
            list(pretrained.keys()), exclude_param_names_regex
        )
        self.task_vector_param_dict = {}
        for name in names:
            if name not in finetuned:
                raise KeyError(
                    f"Parameter {name!r} missing from the finetuned model."
                )
            self.task_vector_param_dict[name] = finetuned[name] - pretrained[name]

    # ------------------------------------------------------------------ #
    # container protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.task_vector_param_dict)

    def __iter__(self):
        return iter(self.task_vector_param_dict)

    def __getitem__(self, key: str):
        return self.task_vector_param_dict[key]

    def items(self):
        return self.task_vector_param_dict.items()

    def keys(self):
        return self.task_vector_param_dict.keys()

    def __add__(self, other: "TaskVector") -> "TaskVector":
        if not isinstance(other, TaskVector):
            raise TypeError("TaskVector addition requires another TaskVector.")
        out = {}
        for name in self.task_vector_param_dict:
            if name not in other.task_vector_param_dict:
                raise KeyError(f"Parameter {name!r} not present in both task vectors.")
            out[name] = (
                self.task_vector_param_dict[name]
                + other.task_vector_param_dict[name]
            )
        return TaskVector(task_vector_param_dict=out)

    def __radd__(self, other):
        # allows ``sum([tv1, tv2], 0)``
        if other == 0:
            return self
        return self.__add__(other)

    # ------------------------------------------------------------------ #
    # coefficients & composition
    # ------------------------------------------------------------------ #
    def scaled(self, coefficient: float) -> "TaskVector":
        """Return a new task vector with every parameter scaled by ``coefficient``."""
        return TaskVector(
            task_vector_param_dict={
                name: coefficient * value
                for name, value in self.task_vector_param_dict.items()
            }
        )

    def apply_layer_coefficients(
        self,
        coefficients: Mapping[int, float],
        layer_index_fn,
        default_coefficient: float = 1.0,
    ) -> "TaskVector":
        """Scale each parameter by the coefficient of its layer.

        Parameters
        ----------
        coefficients:
            Mapping ``layer_index -> coefficient``.
        layer_index_fn:
            Callable ``param_name -> int`` returning the layer index (see
            :class:`lhm.layer_utils.LayerKeyMapper`).
        default_coefficient:
            Coefficient used for parameters whose layer index is not present in
            ``coefficients`` (e.g. embeddings).
        """
        out = {}
        for name, value in self.task_vector_param_dict.items():
            layer = layer_index_fn(name)
            coefficient = coefficients.get(layer, default_coefficient)
            out[name] = coefficient * value
        return TaskVector(task_vector_param_dict=out)

    def combine_with_pretrained_model(
        self, pretrained_model, scaling_coefficient: float = 1.0
    ) -> Dict[str, object]:
        """``theta_0 + scaling_coefficient * delta`` for every merged parameter."""
        pretrained = _to_param_dict(pretrained_model)
        merged: Dict[str, object] = {}
        for name, value in self.task_vector_param_dict.items():
            merged[name] = pretrained[name] + scaling_coefficient * value
        return merged

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def squared_norms(self) -> Dict[str, object]:
        """Per-parameter squared L2 norm.  Works for tensors and numpy arrays."""
        out = {}
        for name, value in self.task_vector_param_dict.items():
            try:
                import torch  # local import keeps the module importable w/o torch

                if isinstance(value, torch.Tensor):
                    out[name] = float(torch.sum(value.detach().to(torch.float64) ** 2))
                    continue
            except Exception:  # pragma: no cover - torch missing
                pass
            flat = value.reshape(-1) if hasattr(value, "reshape") else value
            total = 0.0
            for element in flat:
                total += float(element) ** 2
            out[name] = total
        return out

    def l2_norm(self) -> float:
        """Total L2 norm of the whole task vector."""
        return float(sum(self.squared_norms().values())) ** 0.5

    def sorted_items(self) -> "OrderedDict[str, object]":
        return OrderedDict(sorted(self.task_vector_param_dict.items()))

    def to_cpu_state_dict(self) -> Dict[str, object]:
        """Detach + move every delta to CPU (no-op for numpy arrays)."""
        out = {}
        for name, value in self.task_vector_param_dict.items():
            if hasattr(value, "detach"):
                out[name] = value.detach().cpu().clone()
            else:
                out[name] = value
        return out

    @classmethod
    def from_state_dicts(
        cls,
        pretrained_state_dict: Mapping[str, object],
        finetuned_state_dict: Mapping[str, object],
        exclude_param_names_regex: Optional[Sequence[str]] = None,
    ) -> "TaskVector":
        return cls(
            pretrained_model=dict(pretrained_state_dict),
            finetuned_model=dict(finetuned_state_dict),
            exclude_param_names_regex=exclude_param_names_regex,
        )


def sum_task_vectors(task_vectors: Iterable[TaskVector]) -> TaskVector:
    """Sum an iterable of task vectors (empty input -> empty task vector)."""
    iterator = iter(task_vectors)
    try:
        total = next(iterator)
    except StopIteration:
        return TaskVector(task_vector_param_dict={})
    for task_vector in iterator:
        total = total + task_vector
    return total
