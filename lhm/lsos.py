"""Layer Second-Order Sensitivity (LSOS) index -- Phase 3 of Algorithm 1.

The Hessian trace alone is a poor predictor of fusion difficulty: a layer can
be extremely curved yet barely moved by post-training, while a moderately
curved layer can be displaced a lot.  A second-order Taylor expansion of the
layer loss around ``theta_0`` gives

.. math::
    \\Delta L_l \\approx \\frac{1}{2} \\delta_l^\\top H_l \\delta_l,

and under an isotropy approximation
``delta_l^T H_l delta_l ~= (||delta_l||^2 / d_l) tr(H_l)``.  Dropping the
architectural constant ``d_l`` yields the LSOS index

.. math::
    \\mathrm{LSOS}_l = \\widehat{\\mathrm{tr}}(H_l) \\cdot \\|\\delta_l\\|_2^2 .

LSOS is therefore proportional to the expected loss perturbation induced by
merging layer ``l`` at full coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

from .layer_utils import LayerKeyMapper, DEFAULT_LAYER_PATTERN


def aggregate_squared_norms_by_layer(
    param_squared_norms: Mapping[str, float],
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    include_non_layer: bool = False,
) -> Dict[int, float]:
    """Sum per-parameter squared norms into per-layer squared norms."""
    mapper = LayerKeyMapper(layer_pattern)
    out: Dict[int, float] = {}
    for name, value in param_squared_norms.items():
        layer = mapper.layer_index(name)
        if layer == mapper.non_layer_key and not include_non_layer:
            continue
        out[layer] = out.get(layer, 0.0) + float(value)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def task_vector_squared_norms(
    task_vector_or_dict,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    include_non_layer: bool = False,
) -> Dict[int, float]:
    """Per-layer squared L2 norm ``||delta_l||_2^2`` of a task vector.

    Accepts either a :class:`lhm.task_vector.TaskVector` or a plain
    ``{param_name: delta_tensor}`` mapping.
    """
    if hasattr(task_vector_or_dict, "squared_norms"):
        sq = task_vector_or_dict.squared_norms()
    else:
        sq = {}
        for name, value in dict(task_vector_or_dict).items():
            sq[name] = _squared_norm(value)
    return aggregate_squared_norms_by_layer(
        sq, layer_pattern=layer_pattern, include_non_layer=include_non_layer
    )


def _squared_norm(value) -> float:
    """Squared L2 norm of a tensor / numpy array / sequence of floats."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(torch.sum(value.detach().to(torch.float64) ** 2))
    except Exception:  # pragma: no cover
        pass
    if hasattr(value, "shape") and hasattr(value, "reshape"):
        flat = value.reshape(-1)
    else:
        flat = value
    total = 0.0
    for element in flat:
        total += float(element) ** 2
    return total


@dataclass
class LSOSResult:
    """Per-layer LSOS scores and the ingredients they were built from."""

    scores: Dict[int, float] = field(default_factory=dict)
    hessian_traces: Dict[int, float] = field(default_factory=dict)
    squared_norms: Dict[int, float] = field(default_factory=dict)
    layer_indices: Sequence[int] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if not self.layer_indices:
            self.layer_indices = sorted(self.scores.keys())

    @property
    def L(self) -> int:
        return len(self.scores)

    def top_k(self, k: int) -> Sequence[int]:
        """The ``k`` layers with the largest LSOS score (descending)."""
        return [
            idx
            for idx, _ in sorted(
                self.scores.items(), key=lambda kv: kv[1], reverse=True
            )[:k]
        ]

    def approx_loss_perturbation(self, coefficient: float = 1.0) -> Dict[int, float]:
        """:math:`\\Delta L_l \\approx 0.5 \\lambda^2 LSOS_l` (Taylor, Eq. 8/9)."""
        factor = 0.5 * float(coefficient) ** 2
        return {idx: factor * score for idx, score in self.scores.items()}

    def as_vector(self, fill: float = 0.0) -> "list[float]":
        """Dense score vector indexed by layer id ``0..max(layer)+1``."""
        if not self.scores:
            return []
        n = max(self.scores) + 1
        vec = [fill] * n
        for idx, score in self.scores.items():
            vec[idx] = score
        return vec

    def to_dict(self) -> Dict:
        return {
            "scores": {str(k): v for k, v in self.scores.items()},
            "hessian_traces": {
                str(k): v for k, v in self.hessian_traces.items()
            },
            "squared_norms": {str(k): v for k, v in self.squared_norms.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "LSOSResult":
        return cls(
            scores={int(k): float(v) for k, v in payload["scores"].items()},
            hessian_traces={
                int(k): float(v) for k, v in payload.get("hessian_traces", {}).items()
            },
            squared_norms={
                int(k): float(v) for k, v in payload.get("squared_norms", {}).items()
            },
        )


def compute_lsos(
    hessian_traces: Mapping[int, float],
    squared_norms_by_layer: Mapping[int, float],
    layers: Optional[Sequence[int]] = None,
    missing_norm: float = 0.0,
    missing_trace: float = 0.0,
) -> LSOSResult:
    """Compute ``LSOS_l = tr_hat(H_l) * ||delta_l||^2`` for every layer.

    Parameters
    ----------
    hessian_traces:
        ``{layer_index: estimated Hessian trace}``.
    squared_norms_by_layer:
        ``{layer_index: ||delta_l||_2^2}``.
    layers:
        Optional explicit layer set.  Defaults to the union of the keys of the
        two dicts.  Useful when a layer has a zero task vector.
    """
    if layers is None:
        layers = sorted(set(hessian_traces) | set(squared_norms_by_layer))
    else:
        layers = sorted(layers)

    scores: Dict[int, float] = {}
    traces: Dict[int, float] = {}
    norms: Dict[int, float] = {}
    for idx in layers:
        trace = float(hessian_traces.get(idx, missing_trace))
        norm = float(squared_norms_by_layer.get(idx, missing_norm))
        traces[idx] = trace
        norms[idx] = norm
        scores[idx] = trace * norm
    return LSOSResult(
        scores=scores,
        hessian_traces=traces,
        squared_norms=norms,
        layer_indices=list(layers),
    )
