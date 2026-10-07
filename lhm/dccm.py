"""Dual-Constraint Coefficient Mapping (DCCM) -- Phase 4 of Algorithm 1.

DCCM turns the per-layer LSOS scores into the merge-coefficient vector
``Lambda = [lambda_1, ..., lambda_L]`` through a closed-form, two-step
transform that simultaneously respects two orthogonal constraints:

* **fusion-error constraint** -- larger LSOS must receive a larger coefficient
  (the coefficient is a strictly increasing function of the LSOS score);
* **length-reduction constraint** -- the global mean of the coefficients is
  anchored to a user-chosen target ``lambda_bar`` that trades accuracy for
  conciseness.

Step 1 -- min-max normalisation::

    S~_l = (LSOS_l - min LSOS) / (max LSOS - min LSOS)          in [0, 1]

Step 2 -- mean-anchored affine transform::

    lambda_l = lambda_bar + alpha * (2 S~_l - 1 - mean(S~))

whose empirical mean is exactly ``lambda_bar + alpha * (mean(S~) - 1)``,
bounded in magnitude by ``alpha`` and equal to ``lambda_bar`` when
``mean(S~) == 1`` (Appendix B.4).  ``alpha -> 0`` recovers uniform scalar
merging; the paper's default bandwidth is ``alpha = 0.15``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _as_ordered(mapping: Mapping[int, float]) -> "Tuple[list[int], np.ndarray]":
    layers = sorted(mapping.keys())
    values = np.asarray([float(mapping[l]) for l in layers], dtype=np.float64)
    return layers, values


def min_max_normalize(scores: Mapping[int, float]) -> Dict[int, float]:
    """Min-max normalise LSOS scores into ``[0, 1]`` (Eq. 10).

    If all scores are identical the function returns ``0.0`` for every layer;
    :func:`dccm_coefficients` detects this degenerate case and falls back to
    uniform scalar merging.
    """
    if not scores:
        return {}
    layers, values = _as_ordered(scores)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-12:
        return {layer: 0.0 for layer in layers}
    scaled = (values - lo) / (hi - lo)
    return {layer: float(v) for layer, v in zip(layers, scaled)}


def dccm_anchored_mean(lambda_bar: float, alpha: float, mean_normalized: float) -> float:
    """Closed-form mean of the DCCM coefficient vector.

    .. math::
        \\frac{1}{L}\\sum_l \\lambda_l
        = \\bar\\lambda + \\alpha\\,(\\overline{\\widetilde S} - 1).
    """
    return float(lambda_bar) + float(alpha) * (float(mean_normalized) - 1.0)


@dataclass
class DCCMResult:
    """Output of :func:`dccm_coefficients`."""

    coefficients: Dict[int, float] = field(default_factory=dict)
    normalized_scores: Dict[int, float] = field(default_factory=dict)
    target_lambda_bar: float = 0.5
    alpha: float = 0.15
    achieved_mean: float = 0.5
    fusion_error_ratio: float = 0.0
    rescaled: bool = False
    degenerate: bool = False

    @property
    def L(self) -> int:
        return len(self.coefficients)

    def as_vector(self, fill: float = 0.0) -> "list[float]":
        if not self.coefficients:
            return []
        vec = [fill] * (max(self.coefficients) + 1)
        for idx, value in self.coefficients.items():
            vec[idx] = value
        return vec

    def to_dict(self) -> Dict:
        return {
            "coefficients": {str(k): v for k, v in self.coefficients.items()},
            "normalized_scores": {
                str(k): v for k, v in self.normalized_scores.items()
            },
            "target_lambda_bar": self.target_lambda_bar,
            "alpha": self.alpha,
            "achieved_mean": self.achieved_mean,
            "fusion_error_ratio": self.fusion_error_ratio,
            "rescaled": self.rescaled,
            "degenerate": self.degenerate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DCCMResult":
        return cls(
            coefficients={int(k): float(v) for k, v in payload["coefficients"].items()},
            normalized_scores={
                int(k): float(v)
                for k, v in payload.get("normalized_scores", {}).items()
            },
            target_lambda_bar=float(payload.get("target_lambda_bar", 0.5)),
            alpha=float(payload.get("alpha", 0.15)),
            achieved_mean=float(payload.get("achieved_mean", 0.5)),
            fusion_error_ratio=float(payload.get("fusion_error_ratio", 0.0)),
            rescaled=bool(payload.get("rescaled", False)),
            degenerate=bool(payload.get("degenerate", False)),
        )


def _fusion_error_proxy(
    coefficients: Mapping[int, float], lsos: Mapping[int, float]
) -> float:
    """``E(Lambda) = sum_l lambda_l^2 * LSOS_l``  (proportional to total dL)."""
    total = 0.0
    for idx, coefficient in coefficients.items():
        total += float(coefficient) ** 2 * float(lsos.get(idx, 0.0))
    return total


def dccm_coefficients(
    lsos_scores: Mapping[int, float],
    lambda_bar: float = 0.5,
    alpha: float = 0.15,
    clip: bool = False,
    clip_range: Tuple[float, float] = (0.0, 1.0),
    mean_anchor_exact: bool = False,
    epsilon: Optional[float] = None,
) -> DCCMResult:
    """Map LSOS scores to layer-wise merge coefficients (Eq. 11).

    Parameters
    ----------
    lsos_scores:
        ``{layer_index: LSOS_l}``.
    lambda_bar:
        Global target mean coefficient ``lambda_bar`` (length-reduction knob).
    alpha:
        Bandwidth controlling how strongly layers are differentiated
        (paper default ``0.15``); ``alpha = 0`` -> uniform merging.
    clip:
        Clip coefficients to ``clip_range`` (default ``[0, 1]``).
    clip_range:
        Lower / upper clipping bounds.
    mean_anchor_exact:
        If ``True``, shift all coefficients by a constant so that the empirical
        mean equals ``lambda_bar`` exactly (the paper only requires it to be
        close, cf. Appendix B.4).  Ordering -- hence the fusion-error
        constraint -- is preserved.
    epsilon:
        Optional fusion-error budget.  When given, coefficients are rescaled by
        a single positive factor so that
        ``sum_l lambda_l^2 LSOS_l <= epsilon * sum_l LSOS_l``.  ``None``
        reproduces the paper's pure closed-form transform.
    """
    if not lsos_scores:
        return DCCMResult(
            coefficients={},
            normalized_scores={},
            target_lambda_bar=lambda_bar,
            alpha=alpha,
            achieved_mean=float("nan"),
            degenerate=True,
        )

    layers, values = _as_ordered(lsos_scores)
    normalized = min_max_normalize(lsos_scores)
    mean_normalized = float(np.mean([normalized[l] for l in layers]))

    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-12:
        # Degenerate case: no layer is more sensitive than any other, so DCCM
        # cannot differentiate and degrades gracefully to scalar merging.
        coefficients = {layer: float(lambda_bar) for layer in layers}
        return DCCMResult(
            coefficients=coefficients,
            normalized_scores=normalized,
            target_lambda_bar=lambda_bar,
            alpha=alpha,
            achieved_mean=float(lambda_bar),
            fusion_error_ratio=_safe_ratio(coefficients, lsos_scores),
            degenerate=True,
        )

    coefficients = {
        layer: float(lambda_bar)
        + float(alpha) * (2.0 * normalized[layer] - 1.0 - mean_normalized)
        for layer in layers
    }

    if mean_anchor_exact:
        current_mean = float(np.mean(list(coefficients.values())))
        shift = float(lambda_bar) - current_mean
        coefficients = {layer: value + shift for layer, value in coefficients.items()}

    if clip:
        low, high = clip_range
        coefficients = {
            layer: float(min(max(value, low), high))
            for layer, value in coefficients.items()
        }

    rescaled = False
    if epsilon is not None:
        coefficients, rescaled = apply_fusion_error_bound(
            coefficients, lsos_scores, epsilon=epsilon, clip=clip, clip_range=clip_range
        )

    return DCCMResult(
        coefficients=coefficients,
        normalized_scores=normalized,
        target_lambda_bar=lambda_bar,
        alpha=alpha,
        achieved_mean=float(np.mean(list(coefficients.values()))),
        fusion_error_ratio=_safe_ratio(coefficients, lsos_scores),
        rescaled=rescaled,
        degenerate=False,
    )


def _safe_ratio(
    coefficients: Mapping[int, float], lsos_scores: Mapping[int, float]
) -> float:
    reference = _fusion_error_proxy({l: 1.0 for l in lsos_scores}, lsos_scores)
    if reference <= 0.0:
        return 0.0
    return _fusion_error_proxy(coefficients, lsos_scores) / reference


def apply_fusion_error_bound(
    coefficients: Mapping[int, float],
    lsos_scores: Mapping[int, float],
    epsilon: float,
    clip: bool = False,
    clip_range: Tuple[float, float] = (0.0, 1.0),
) -> "Tuple[Dict[int, float], bool]":
    """Enforce ``E(Lambda) <= epsilon * E(1)`` by shrinking ``Lambda`` uniformly.

    Scaling all coefficients by the same positive factor ``s <= 1`` preserves
    their ordering, so the fusion-error constraint (monotone ``LSOS -> lambda``)
    remains satisfied while the quadratic loss perturbation is brought inside
    the budget.
    """
    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be a positive float.")
    reference = _fusion_error_proxy({l: 1.0 for l in lsos_scores}, lsos_scores)
    if reference <= 0.0:
        return dict(coefficients), False
    current = _fusion_error_proxy(coefficients, lsos_scores)
    budget = float(epsilon) * reference
    if current <= budget or current <= 0.0:
        return dict(coefficients), False
    scale = float(np.sqrt(budget / current))
    new = {layer: value * scale for layer, value in coefficients.items()}
    if clip:
        low, high = clip_range
        new = {layer: float(min(max(v, low), high)) for layer, v in new.items()}
    return new, True


def coefficients_to_dense_vector(
    coefficients: Mapping[int, float], num_layers: Optional[int] = None, fill: float = 0.0
) -> "list[float]":
    """Dense ``Lambda`` vector so that ``vector[i]`` is layer ``i``'s coefficient."""
    if num_layers is None:
        if not coefficients:
            return []
        num_layers = max(coefficients) + 1
    vector = [fill] * num_layers
    for idx, value in coefficients.items():
        if 0 <= idx < num_layers:
            vector[idx] = float(value)
    return vector
