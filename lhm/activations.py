"""Per-layer activation statistics used by the activation-based baselines.

AIM, ACM and ORCA all derive their layer coefficients from *first-order*
activation statistics (outlier ratios / activation magnitudes).  LHM replaces
these with second-order curvature, but we still need activation statistics to
reproduce the baselines for comparison.

The collector below attaches forward hooks to every module whose name contains
``layers.<idx>`` and records the mean squared output norm per layer.  It works
with any ``nn.Module`` and, like the Hessian estimator, can be driven by
randomly generated token ids so that no calibration corpus is required.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .layer_utils import LayerKeyMapper, DEFAULT_LAYER_PATTERN


class ActivationNormCollector:
    """Collect per-layer mean activation (output) norms via forward hooks."""

    def __init__(
        self,
        model,
        layer_pattern: str = DEFAULT_LAYER_PATTERN,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.mapper = LayerKeyMapper(layer_pattern)
        self._sums: Dict[int, float] = {}
        self._counts: Dict[int, int] = {}
        self._handles: List = []

    # ------------------------------------------------------------------ #
    def _hook(self, module, inputs, output):
        name = None
        # ``module`` may not expose its qualified name, so we store it in a map.
        name = self._name_map.get(id(module))
        if name is None:
            return
        layer = self.mapper.layer_index(name)
        if layer < 0:
            return
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            return
        with torch.no_grad():
            norm_sq = float(torch.sum(tensor.detach().to(torch.float64) ** 2))
            count = int(tensor.numel())
        self._sums[layer] = self._sums.get(layer, 0.0) + norm_sq
        self._counts[layer] = self._counts.get(layer, 0) + max(count, 1)

    def __enter__(self) -> "ActivationNormCollector":
        self._name_map = {id(m): n for n, m in self.model.named_modules()}
        # Hook the transformer blocks themselves (one hook per layer).
        for name, module in self.model.named_modules():
            if self.mapper.is_layer_param(name) or self.mapper.layer_index(name) >= 0:
                # only hook top-level block containers, i.e. names ending in a
                # numeric index (``...layers.5``), to avoid double counting.
                if name.split(".")[-1].isdigit():
                    self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------ #
    def result(self, aggregate: str = "mean") -> Dict[int, float]:
        """Per-layer activation statistic.

        ``aggregate="mean"``  -> mean squared activation norm per element
        ``aggregate="sum"``   -> summed squared activation norm
        ``aggregate="rms"``   -> root-mean-square activation norm
        """
        out: Dict[int, float] = {}
        for layer, total in self._sums.items():
            count = max(self._counts.get(layer, 1), 1)
            if aggregate == "sum":
                out[layer] = total
            elif aggregate == "mean":
                out[layer] = total / count
            elif aggregate == "rms":
                out[layer] = (total / count) ** 0.5
            else:
                raise ValueError(
                    f"Unsupported aggregate {aggregate!r}; use mean/sum/rms."
                )
        return dict(sorted(out.items(), key=lambda kv: kv[0]))


def collect_activation_norms(
    model,
    input_ids: torch.Tensor,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    device: str = "cpu",
    aggregate: str = "mean",
) -> Dict[int, float]:
    """Run a single forward pass and return per-layer activation statistics."""
    was_training = model.training
    model.eval()
    with ActivationNormCollector(model, layer_pattern, device) as collector:
        with torch.no_grad():
            model(input_ids=input_ids.to(device))
    if was_training:
        model.train()
    return collector.result(aggregate=aggregate)
