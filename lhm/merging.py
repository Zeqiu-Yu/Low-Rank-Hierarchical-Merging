"""Layer-adaptive model merging -- Phases 1 & 5 of Algorithm 1.

Phase 1 (task-vector decomposition) is handled by
:class:`lhm.task_vector.TaskVector`; Phase 5 applies the coefficient vector
produced by DCCM to the task vector:

.. math::
    \\theta_{\\mathrm{merged}} = \\theta_0 + \\boldsymbol{\\Lambda} \\odot
    (\\theta_r - \\theta_0).

The plug-and-play variants from Appendix A are implemented here as well: the
very same ``Lambda`` is dropped into Task Arithmetic (``ta``), TIES
(``ties``), DARE (``dare``) and AIM (``aim``) to obtain LHM-TA, LHM-TIES,
LHM-DARE and LHM-AIM.

All the routines operate on plain ``{name: tensor}`` dictionaries, so they can
be unit-tested on toy tensors without ever loading a language model.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import LHMConfig
from .dccm import DCCMResult, dccm_coefficients
from .hessian import LowRankHessianEstimator, HessianTraceResult
from .layer_utils import LayerKeyMapper, DEFAULT_LAYER_PATTERN
from .lsos import LSOSResult, compute_lsos, task_vector_squared_norms
from .task_vector import TaskVector, get_param_names_to_merge


SUPPORTED_METHODS = ("ta", "ties", "dare", "aim")


# --------------------------------------------------------------------------- #
# dictionary helpers
# --------------------------------------------------------------------------- #
def _as_param_dict(model_or_dict) -> Dict[str, torch.Tensor]:
    if isinstance(model_or_dict, Mapping):
        return dict(model_or_dict)
    if hasattr(model_or_dict, "named_parameters"):
        return {name: p.detach() for name, p in model_or_dict.named_parameters()}
    if hasattr(model_or_dict, "state_dict"):
        return dict(model_or_dict.state_dict())
    raise TypeError(f"Unsupported model argument: {type(model_or_dict)!r}")


def build_task_vector_dict(
    base_model,
    reasoning_model,
    exclude_param_names_regex: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """``delta = theta_reasoning - theta_base`` for every shared parameter."""
    base = _as_param_dict(base_model)
    reason = _as_param_dict(reasoning_model)
    names = get_param_names_to_merge(list(base.keys()), exclude_param_names_regex)
    delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    with torch.no_grad():
        for name in names:
            if name not in reason:
                continue
            delta[name] = (reason[name] - base[name]).detach().clone()
    return delta


def _group_by_layer(
    names: Sequence[str], layer_pattern: str = DEFAULT_LAYER_PATTERN
) -> "OrderedDict[int, List[str]]":
    mapper = LayerKeyMapper(layer_pattern)
    groups: "OrderedDict[int, List[str]]" = OrderedDict()
    for name in names:
        groups.setdefault(mapper.layer_index(name), []).append(name)
    return OrderedDict(sorted(groups.items(), key=lambda kv: kv[0]))


# --------------------------------------------------------------------------- #
# layer-wise strategies (operate on per-layer delta tensors)
# --------------------------------------------------------------------------- #
def _trim_layer_delta(
    layer_delta: Mapping[str, torch.Tensor], mask_rate: float
) -> Dict[str, torch.Tensor]:
    """TIES trimming: zero out the smallest-magnitude ``mask_rate`` fraction.

    Exactly ``floor(mask_rate * numel)`` entries are removed (the
    smallest-magnitude ones); the threshold is the ``keep_count``-th largest
    magnitude, so ties on the threshold are kept.
    """
    if not layer_delta:
        return {}
    flat = torch.cat([t.detach().reshape(-1) for t in layer_delta.values()])
    numel = flat.numel()
    num_mask = int(numel * float(mask_rate))
    if num_mask <= 0:
        return {name: t.clone() for name, t in layer_delta.items()}
    if num_mask >= numel:
        return {name: torch.zeros_like(t) for name, t in layer_delta.items()}
    keep_count = numel - num_mask
    threshold = torch.topk(flat.abs(), keep_count, largest=True).values[-1]
    return {
        name: torch.where(t.abs() >= threshold, t, torch.zeros_like(t))
        for name, t in layer_delta.items()
    }


def ties_merge_layer(
    layer_task_vectors: Sequence[Mapping[str, torch.Tensor]],
    param_value_mask_rate: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """TIES (trim + elect sign + disjoint merge) for one layer.

    ``layer_task_vectors`` is a list of per-task-vector deltas restricted to the
    parameters of a *single* layer.  With a single task vector this reduces to
    magnitude-based trimming, which is the L2S setting of the paper.
    """
    if not layer_task_vectors:
        return {}
    trimmed = [
        _trim_layer_delta(tv, param_value_mask_rate) for tv in layer_task_vectors
    ]
    names = list(trimmed[0].keys())
    summed = {name: sum(tv[name] for tv in trimmed) for name in names}
    majority_sign = torch.sign(
        sum(torch.sign(t).sum() for t in summed.values())
    )
    if float(majority_sign) == 0.0:
        majority_sign = torch.ones_like(majority_sign)
    aggregated_sign = {name: torch.sign(summed[name]) for name in names}
    # replace zero signs by the layer's majority sign
    out: Dict[str, torch.Tensor] = {}
    for name in names:
        sign = aggregated_sign[name].clone()
        sign[sign == 0] = majority_sign
        preserved = [
            torch.where(
                ((sign > 0) & (tv[name] > 0)) | ((sign < 0) & (tv[name] < 0)),
                tv[name],
                torch.zeros_like(tv[name]),
            )
            for tv in trimmed
        ]
        counts = sum((p != 0).to(p.dtype) for p in preserved)
        out[name] = sum(preserved) / torch.clamp(counts, min=1.0)
    return out


def dare_merge_layer(
    layer_delta: Mapping[str, torch.Tensor],
    mask_rate: float = 0.3,
    mask_strategy: str = "random",
    use_weight_rescale: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """DARE: drop a fraction of the task-vector entries and rescale the rest."""
    if mask_rate < 0.0 or mask_rate > 1.0:
        raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}.")
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in layer_delta.items():
        if mask_strategy == "random":
            keep_prob = 1.0 - float(mask_rate)
            keep = torch.bernoulli(
                torch.full_like(tensor, keep_prob), generator=generator
            )
            masked = tensor * keep
        elif mask_strategy == "magnitude":
            trimmed = _trim_layer_delta({name: tensor}, mask_rate)[name]
            masked = trimmed
        else:
            raise ValueError(
                f"Unsupported mask_strategy {mask_strategy!r}; use 'random' or 'magnitude'."
            )
        if use_weight_rescale and mask_rate != 1.0:
            masked = masked / (1.0 - float(mask_rate))
        out[name] = masked
    return out


def aim_layer_weight(
    activation_norm: float,
    omega: float = 0.4,
) -> float:
    """AIM-style activation-informed multiplicative weight for one layer.

    AIM (Nobari et al., 2025) rescales the merged update by a quantity derived
    from the layer's activation statistics, balanced by ``omega``.  Here the
    activation magnitude is normalised to a multiplicative factor
    ``omega * (1 + tanh(log(1 + norm)))`` so that ``omega = 0`` disables the
    activation term and the factor stays strictly positive.
    """
    import math

    return float(omega) * (1.0 + math.tanh(math.log1p(max(float(activation_norm), 0.0))))


# --------------------------------------------------------------------------- #
# main merge entry point
# --------------------------------------------------------------------------- #
def merge_with_coefficients(
    base_model,
    reasoning_models,
    coefficients: Mapping[int, float],
    method: str = "ta",
    exclude_param_names_regex: Optional[Sequence[str]] = None,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    param_value_mask_rate: float = 0.8,
    mask_rate: float = 0.3,
    mask_strategy: str = "random",
    use_weight_rescale: bool = True,
    activation_norms: Optional[Mapping[int, float]] = None,
    aim_omega: float = 0.4,
    non_layer_coefficient: float = 1.0,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Apply the LHM coefficient vector through a chosen merge strategy.

    Parameters
    ----------
    base_model:
        Base model (``theta_0``) as ``nn.Module``, ``state_dict`` or dict.
    reasoning_models:
        A single reasoning model or a list of them.  Each contributes one task
        vector ``theta_r - theta_0``; multiple task vectors enable genuine TIES
        sign-conflict resolution.
    coefficients:
        ``{layer_index: lambda_l}`` produced by :func:`lhm.dccm.dccm_coefficients`.
    method:
        One of ``"ta"``, ``"ties"``, ``"dare"``, ``"aim"``.
    """
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}; expected one of {SUPPORTED_METHODS}."
        )

    base = _as_param_dict(base_model)
    if isinstance(reasoning_models, (list, tuple)):
        reasoning_list = list(reasoning_models)
    else:
        reasoning_list = [reasoning_models]

    task_vectors: List["OrderedDict[str, torch.Tensor]"] = [
        _as_param_dict(reason) for reason in reasoning_list
    ]
    # convert raw models into deltas
    deltas: List["OrderedDict[str, torch.Tensor]"] = []
    for reason_params in task_vectors:
        delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, value in base.items():
            if name in reason_params:
                delta[name] = (reason_params[name] - value).detach()
        if exclude_param_names_regex:
            keep = set(
                get_param_names_to_merge(list(delta.keys()), exclude_param_names_regex)
            )
            delta = OrderedDict((n, t) for n, t in delta.items() if n in keep)
        deltas.append(delta)

    if not deltas:
        raise ValueError("No task vectors provided.")
    names = [n for n in deltas[0].keys() if all(n in d for d in deltas)]
    groups = _group_by_layer(names, layer_pattern)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    merged_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    with torch.no_grad():
        for layer, layer_names in groups.items():
            if layer < 0:
                # handled below with ``non_layer_coefficient``
                continue
            layer_deltas = [
                {n: delta[n] for n in layer_names} for delta in deltas
            ]
            if method == "ta":
                layer_out = layer_deltas[0] if len(layer_deltas) == 1 else {
                    n: sum(d[n] for d in layer_deltas) for n in layer_names
                }
            elif method == "ties":
                layer_out = ties_merge_layer(
                    layer_deltas, param_value_mask_rate=param_value_mask_rate
                )
            elif method == "dare":
                layer_out = {}
                for d in layer_deltas:
                    for n, t in dare_merge_layer(
                        d,
                        mask_rate=mask_rate,
                        mask_strategy=mask_strategy,
                        use_weight_rescale=use_weight_rescale,
                        generator=generator,
                    ).items():
                        layer_out[n] = layer_out.get(n, 0.0) + t
            elif method == "aim":
                layer_out = {}
                norm = (
                    float(activation_norms.get(layer, 0.0))
                    if activation_norms is not None
                    else 0.0
                )
                weight = aim_layer_weight(norm, omega=aim_omega)
                for d in layer_deltas:
                    for n, t in d.items():
                        layer_out[n] = layer_out.get(n, 0.0) + weight * t
            else:  # pragma: no cover - guarded above
                raise AssertionError(method)

            coefficient = float(coefficients.get(layer, non_layer_coefficient))
            for name in layer_names:
                merged_delta[name] = coefficient * layer_out[name]

        # parameters outside any transformer block (embedding, lm_head, ...)
        non_layer_names = [n for n in names if _layer_index(n, layer_pattern) < 0]
        for name in non_layer_names:
            merged_delta[name] = float(non_layer_coefficient) * sum(
                d[name] for d in deltas
            )

        merged: Dict[str, torch.Tensor] = {}
        for name, value in base.items():
            if name in merged_delta:
                merged[name] = value + merged_delta[name]
            else:
                merged[name] = value.clone()
    return merged


def _layer_index(name: str, layer_pattern: str) -> int:
    return LayerKeyMapper(layer_pattern).layer_index(name)


# --------------------------------------------------------------------------- #
# high-level orchestrator
# --------------------------------------------------------------------------- #
class LHMMerger:
    """End-to-end LHM pipeline (Algorithm 1).

    The class is deliberately model-agnostic: it accepts any ``nn.Module``
    (toy modules included), so the full pipeline can be exercised in unit tests
    without downloading a language model.
    """

    def __init__(
        self,
        config: Optional[LHMConfig] = None,
        layer_pattern: str = DEFAULT_LAYER_PATTERN,
        device: str = "cpu",
        hessian_estimator: Optional[LowRankHessianEstimator] = None,
    ) -> None:
        self.config = config or LHMConfig()
        self.layer_pattern = layer_pattern
        self.device = device
        self.hessian_estimator = hessian_estimator

    # ------------------------------------------------------------------ #
    def estimate_hessian_traces(
        self, model, vocab_size: Optional[int] = None
    ) -> HessianTraceResult:
        estimator = self.hessian_estimator or LowRankHessianEstimator(
            model=model, layer_pattern=self.layer_pattern, device=self.device
        )
        estimator.model = model
        estimator._layer_params = None  # invalidate cache for the new model
        return estimator.estimate(
            k=self.config.rank,
            num_sequences=self.config.num_sequences,
            seq_len=self.config.seq_len,
            vocab_size=vocab_size,
            seed=self.config.seed,
        )

    def compute_lsos(
        self,
        base_model,
        reasoning_model,
        hessian_traces: Optional[Mapping[int, float]] = None,
        vocab_size: Optional[int] = None,
        hessian_source: Optional[str] = None,
    ) -> LSOSResult:
        """Phase 2 + 3: Hessian traces -> LSOS scores."""
        hessian_source = hessian_source or self.config.hessian_source
        if hessian_traces is None:
            source_model = base_model if hessian_source == "base" else reasoning_model
            hessian_traces = self.estimate_hessian_traces(
                source_model, vocab_size=vocab_size
            ).traces
        delta = build_task_vector_dict(
            base_model,
            reasoning_model,
            exclude_param_names_regex=self.config.exclude_param_names_regex,
        )
        squared_norms = task_vector_squared_norms(
            delta, layer_pattern=self.layer_pattern
        )
        layers = sorted(set(hessian_traces) | set(squared_norms))
        return compute_lsos(hessian_traces, squared_norms, layers=layers)

    def compute_coefficients(self, lsos: LSOSResult) -> DCCMResult:
        """Phase 4: LSOS -> layer-wise coefficients (DCCM)."""
        return dccm_coefficients(
            lsos.scores,
            lambda_bar=self.config.lambda_bar,
            alpha=self.config.alpha,
            clip=True,
            mean_anchor_exact=self.config.mean_anchor_exact,
            epsilon=self.config.fusion_error_epsilon,
        )

    def merge(
        self,
        base_model,
        reasoning_model,
        coefficients: Mapping[int, float],
        activation_norms: Optional[Mapping[int, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Phase 5: apply ``Lambda`` and return the merged parameter dict."""
        return merge_with_coefficients(
            base_model=base_model,
            reasoning_models=reasoning_model,
            coefficients=coefficients,
            method=self.config.method,
            exclude_param_names_regex=self.config.exclude_param_names_regex,
            layer_pattern=self.layer_pattern,
            param_value_mask_rate=self.config.param_value_mask_rate,
            mask_rate=self.config.mask_rate,
            activation_norms=activation_norms,
            aim_omega=self.config.aim_omega,
            seed=self.config.seed,
        )

    def run(
        self,
        base_model,
        reasoning_model,
        hessian_traces: Optional[Mapping[int, float]] = None,
        activation_norms: Optional[Mapping[int, float]] = None,
        vocab_size: Optional[int] = None,
        hessian_source: Optional[str] = None,
    ) -> "Tuple[Dict[str, torch.Tensor], LSOSResult, DCCMResult]":
        """Run Phases 1-5 and return the merged parameters plus diagnostics."""
        lsos = self.compute_lsos(
            base_model,
            reasoning_model,
            hessian_traces=hessian_traces,
            vocab_size=vocab_size,
            hessian_source=hessian_source,
        )
        dccm = self.compute_coefficients(lsos)
        merged = self.merge(
            base_model,
            reasoning_model,
            dccm.coefficients,
            activation_norms=activation_norms,
        )
        return merged, lsos, dccm


def copy_params_to_model(params: Mapping[str, torch.Tensor], model) -> None:
    """In-place copy of ``params`` into ``model`` (matching by parameter name)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name].to(param.dtype).to(param.device))
