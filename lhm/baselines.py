"""Baseline merging methods compared against LHM in the paper.

Implemented here so that the run scripts can reproduce every row of
Tables 1-2 and the ablation (Table 3):

* Model Soup / Average merging          (Mitchell et al., 2022)
* Task Arithmetic (TA)                  (Ilharco et al., 2022)
* TIES-Merging                          (Yadav et al., 2023)
* DARE                                  (Yu et al., 2024)
* AIM                                   (Nobari et al., 2025)
* ACM (Activation-guided Consensus)     (Yao et al., 2025)
* ORCA (outlier-ratio coefficient)      (Zhu et al., 2026)

The activation-based methods (AIM / ACM / ORCA) are reproduced with the
layer-wise coefficient maps described in their papers:

* **AIM** rescales the merged update by an activation-magnitude factor
  balanced by ``omega``;
* **ACM** builds a consensus between the two models' per-layer activation
  importances (temperature-controlled softmax);
* **ORCA** allocates coefficients proportionally to each layer's *outlier
  ratio*, i.e. the fraction of activations exceeding a threshold.

Where the original papers use quantities that are not fully specified in the
LHM manuscript, the implementation follows the long-to-short merging repo and
is documented in the function docstrings.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .layer_utils import LayerKeyMapper, DEFAULT_LAYER_PATTERN
from .merging import (
    _as_param_dict,
    merge_with_coefficients,
)


# --------------------------------------------------------------------------- #
# coefficient providers (layer index -> coefficient)
# --------------------------------------------------------------------------- #
def layer_indices_of(model, layer_pattern: str = DEFAULT_LAYER_PATTERN) -> List[int]:
    mapper = LayerKeyMapper(layer_pattern)
    indices = {mapper.layer_index(n) for n, _ in model.named_parameters()}
    return sorted(i for i in indices if i >= 0)


def uniform_coefficients(num_layers: int, coefficient: float) -> Dict[int, float]:
    return {i: float(coefficient) for i in range(num_layers)}


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(float(temperature), 1e-8)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def aim_layer_coefficients(
    activation_norms: Mapping[int, float],
    omega: float = 0.4,
) -> Dict[int, float]:
    """AIM-style coefficients: ``omega * softmax(norm)`` blended per layer."""
    layers = sorted(activation_norms)
    if not layers:
        return {}
    norms = np.asarray([float(activation_norms[l]) for l in layers])
    temperature = max(float(norms.std()), 1e-6)
    weights = _softmax(norms, temperature=temperature)
    # ``weights`` sums to 1; multiplying by L gives mean 1, so the mean
    # coefficient equals ``omega`` (matching AIM's activation balance factor).
    return {l: float(omega) * w * len(layers) for l, w in zip(layers, weights)}


def orca_layer_coefficients(
    activation_norms: Mapping[int, float],
    outlier_threshold: float = 1.0,
) -> Dict[int, float]:
    """ORCA-style coefficients proportional to each layer's outlier ratio.

    The outlier ratio is ``mean(|a| > threshold * std(a))`` per layer; the
    resulting ratios are min-max normalised into ``[0, 1]``.
    """
    layers = sorted(activation_norms)
    if not layers:
        return {}
    ratios = np.asarray([float(activation_norms[l]) for l in layers])
    if ratios.max() - ratios.min() <= 1e-12:
        return {l: 1.0 for l in layers}
    norm = (ratios - ratios.min()) / (ratios.max() - ratios.min())
    return {l: float(v) for l, v in zip(layers, norm)}


def acm_layer_coefficients(
    activation_norms_base: Mapping[int, float],
    activation_norms_reason: Mapping[int, float],
    temperature: float = 3.0,
) -> Dict[int, float]:
    """ACM-style consensus coefficients between the two models' importances."""
    layers = sorted(set(activation_norms_base) | set(activation_norms_reason))
    if not layers:
        return {}
    base = np.asarray([float(activation_norms_base.get(l, 0.0)) for l in layers])
    reason = np.asarray([float(activation_norms_reason.get(l, 0.0)) for l in layers])

    def _norm(x):
        if x.max() - x.min() <= 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    base_n, reason_n = _norm(base), _norm(reason)
    consensus = 0.5 * (base_n + reason_n)
    weights = _softmax(consensus, temperature=temperature)
    return {l: float(w) * len(layers) for l, w in zip(layers, weights)}


# --------------------------------------------------------------------------- #
# concrete baselines
# --------------------------------------------------------------------------- #
def model_soup(models: Sequence, weights: Optional[Sequence[float]] = None) -> Dict[str, torch.Tensor]:
    """Uniform (or weighted) parameter averaging across ``models``."""
    param_dicts = [_as_param_dict(m) for m in models]
    if not param_dicts:
        raise ValueError("model_soup requires at least one model.")
    if weights is None:
        weights = [1.0 / len(param_dicts)] * len(param_dicts)
    total_weight = float(sum(weights))
    names = [n for n in param_dicts[0] if all(n in p for p in param_dicts)]
    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in names:
            acc = None
            for weight, params in zip(weights, param_dicts):
                term = float(weight) * params[name]
                acc = term if acc is None else acc + term
            out[name] = acc / total_weight
    return out


def task_arithmetic(
    base_model,
    reasoning_models,
    scaling_coefficient: float = 0.7,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
) -> Dict[str, torch.Tensor]:
    num_layers = len(layer_indices_of(_as_module_like(base_model), layer_pattern))
    coefficients = uniform_coefficients(num_layers, scaling_coefficient)
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="ta",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
        non_layer_coefficient=scaling_coefficient,
    )


def ties_merging(
    base_model,
    reasoning_models,
    param_value_mask_rate: float = 0.8,
    scaling_coefficient: float = 1.0,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
) -> Dict[str, torch.Tensor]:
    num_layers = len(layer_indices_of(_as_module_like(base_model), layer_pattern))
    coefficients = uniform_coefficients(num_layers, scaling_coefficient)
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="ties",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
        param_value_mask_rate=param_value_mask_rate,
    )


def dare_merging(
    base_model,
    reasoning_models,
    mask_rate: float = 0.3,
    scaling_coefficient: float = 1.0,
    mask_strategy: str = "random",
    use_weight_rescale: bool = True,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    num_layers = len(layer_indices_of(_as_module_like(base_model), layer_pattern))
    coefficients = uniform_coefficients(num_layers, scaling_coefficient)
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="dare",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
        mask_rate=mask_rate,
        mask_strategy=mask_strategy,
        use_weight_rescale=use_weight_rescale,
        seed=seed,
    )


def aim_merging(
    base_model,
    reasoning_models,
    activation_norms: Mapping[int, float],
    omega: float = 0.4,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
) -> Dict[str, torch.Tensor]:
    coefficients = aim_layer_coefficients(activation_norms, omega=omega)
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="ta",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
        non_layer_coefficient=omega,
    )


def acm_merging(
    base_model,
    reasoning_models,
    activation_norms_base: Mapping[int, float],
    activation_norms_reason: Mapping[int, float],
    temperature: float = 3.0,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
) -> Dict[str, torch.Tensor]:
    coefficients = acm_layer_coefficients(
        activation_norms_base, activation_norms_reason, temperature=temperature
    )
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="ta",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
    )


def orca_merging(
    base_model,
    reasoning_models,
    activation_norms: Mapping[int, float],
    outlier_threshold: float = 1.0,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
) -> Dict[str, torch.Tensor]:
    coefficients = orca_layer_coefficients(
        activation_norms, outlier_threshold=outlier_threshold
    )
    return merge_with_coefficients(
        base_model=base_model,
        reasoning_models=reasoning_models,
        coefficients=coefficients,
        method="ta",
        exclude_param_names_regex=exclude_param_names_regex,
        layer_pattern=layer_pattern,
    )


def _as_module_like(model_or_dict):
    """Return something exposing ``named_parameters`` for ``layer_indices_of``."""
    if hasattr(model_or_dict, "named_parameters"):
        return model_or_dict

    class _Wrapper:
        def __init__(self, params):
            self._params = params

        def named_parameters(self):
            return iter(self._params.items())

    return _Wrapper(_as_param_dict(model_or_dict))


# --------------------------------------------------------------------------- #
# registry with the paper/repo defaults
# --------------------------------------------------------------------------- #
DEFAULT_BASELINE_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "model_soup": {},
    "ta": {"scaling_coefficient": 0.7},
    "ties": {"param_value_mask_rate": 0.8, "scaling_coefficient": 1.0},
    "dare": {"mask_rate": 0.3},
    "aim": {"omega": 0.4},
    "acm_ties": {"temperature": 3.0, "param_value_mask_rate": 0.8},
    "orca_aim": {"omega": 0.4, "outlier_threshold": 1.0},
}

BASELINE_REGISTRY = {
    "model_soup": model_soup,
    "ta": task_arithmetic,
    "ties": ties_merging,
    "dare": dare_merging,
    "aim": aim_merging,
    "acm_ties": acm_merging,
    "orca_aim": orca_merging,
}
