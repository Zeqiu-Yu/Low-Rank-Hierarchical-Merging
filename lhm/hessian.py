"""Data-free low-rank Hessian trace estimation (Phase 2 of Algorithm 1).

The layer Hessian ``H_l = grad^2_{W_l} L`` is intractable to materialise, but
its *trace* is not.  Hutchinson's identity

.. math::
    \\mathrm{tr}(A) = \\mathbb{E}[v^\\top A v],
    \\qquad \\mathbb{E}[v v^\\top] = I

is estimated with ``k`` Rademacher probe vectors ``v_j in {-1, +1}^d``:

.. math::
    \\widehat{\\mathrm{tr}}(H_l) =
    \\frac{1}{k} \\sum_{j=1}^{k} v_j^\\top H_l v_j .

Each Hessian-vector product ``H_l v_j`` is obtained with two back-propagation
passes:

1. ``g_l = grad_{W_l} L``  (first-order gradient, ``create_graph=True``)
2. ``H_l v = grad_{W_l} <g_l, v>``  (second-order, "double backward")

The passes are driven by *random token sequences* sampled uniformly from the
vocabulary, so no task-specific calibration corpus is required.

Total cost: ``O(L d k)``, a factor ``d / k`` cheaper than the full Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .layer_utils import LayerKeyMapper, DEFAULT_LAYER_PATTERN


# --------------------------------------------------------------------------- #
# Generic Hutchinson estimator
# --------------------------------------------------------------------------- #
def hutchinson_trace(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    k: int = 16,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Estimate ``tr(A)`` for a symmetric operator given through ``matvec``.

    Parameters
    ----------
    matvec:
        Callable implementing ``v -> A v``.
    dim:
        Dimension ``d`` of the (flattened) operator.
    k:
        Number of Rademacher probe vectors (paper default 16).
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    total = torch.zeros((), dtype=dtype, device=device)
    for _ in range(k):
        # Rademacher probe: entries +/-1 with equal probability.
        v = (
            torch.randint(
                low=0,
                high=2,
                size=(dim,),
                device=device,
                generator=generator,
            ).to(dtype)
            * 2
            - 1
        )
        Av = matvec(v)
        total = total + torch.sum(v * Av)
    return float(total / k)


def hutchinson_trace_numpy(A, k: int = 16, seed: int = 0) -> float:
    """Reference Hutchinson estimator on a dense ``numpy`` matrix.

    Provided so that the estimator can be validated against ``numpy.trace``
    without torch, and so users without torch can still inspect the math.
    """
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    total = 0.0
    for _ in range(k):
        v = rng.integers(0, 2, size=d).astype(np.float64) * 2.0 - 1.0
        total += float(v @ (A @ v))
    return total / k


# --------------------------------------------------------------------------- #
# Model-level estimator
# --------------------------------------------------------------------------- #
@dataclass
class HessianTraceResult:
    """Per-layer Hessian trace estimates produced by :class:`LowRankHessianEstimator`."""

    traces: Dict[int, float] = field(default_factory=dict)
    num_probes: int = 0
    num_sequences: int = 0
    seq_len: int = 0

    def to_dict(self) -> Dict:
        return {
            "traces": {str(k): v for k, v in self.traces.items()},
            "num_probes": self.num_probes,
            "num_sequences": self.num_sequences,
            "seq_len": self.seq_len,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "HessianTraceResult":
        return cls(
            traces={int(k): float(v) for k, v in payload["traces"].items()},
            num_probes=int(payload.get("num_probes", 0)),
            num_sequences=int(payload.get("num_sequences", 0)),
            seq_len=int(payload.get("seq_len", 0)),
        )


class LowRankHessianEstimator:
    """Estimate per-layer Hessian traces of a language model, data-free.

    Parameters
    ----------
    model:
        An ``nn.Module`` (typically a HuggingFace causal LM).  Parameters that
        require no gradient are ignored.
    layer_pattern:
        Regex (one capture group) locating the layer index in parameter names.
    device:
        Torch device used for the random inputs / probes.
    loss_fn:
        ``callable(model, input_ids) -> scalar tensor``.  Defaults to the
        causal-LM loss ``model(input_ids, labels=input_ids).loss``.
    """

    def __init__(
        self,
        model,
        layer_pattern: str = DEFAULT_LAYER_PATTERN,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.mapper = LayerKeyMapper(layer_pattern)
        self.loss_fn = loss_fn or self._default_loss_fn
        self._layer_params: Optional[Dict[int, List[Tuple[str, torch.Tensor]]]] = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_loss_fn(model, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids, labels=input_ids)
        return outputs.loss

    @property
    def layer_params(self) -> Dict[int, List[Tuple[str, torch.Tensor]]]:
        """Trainable parameters grouped by transformer layer index."""
        if self._layer_params is None:
            groups: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                groups.setdefault(self.mapper.layer_index(name), []).append(
                    (name, param)
                )
            self._layer_params = dict(sorted(groups.items(), key=lambda kv: kv[0]))
        return self._layer_params

    def layer_indices(self, include_non_layer: bool = False) -> List[int]:
        return [
            idx
            for idx in self.layer_params
            if include_non_layer or idx != self.mapper.non_layer_key
        ]

    def _sample_random_tokens(
        self, vocab_size: int, seq_len: int, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=vocab_size,
            size=(1, seq_len),
            device=self.device,
            generator=generator,
        )

    def _probe_vectors(
        self,
        params: Sequence[Tuple[str, torch.Tensor]],
        generator: torch.Generator,
    ) -> List[torch.Tensor]:
        probes = []
        for _, param in params:
            v = (
                torch.randint(
                    low=0,
                    high=2,
                    size=param.shape,
                    device=self.device,
                    generator=generator,
                ).to(param.dtype)
                * 2
                - 1
            )
            probes.append(v)
        return probes

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def _infer_vocab_size(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None and getattr(config, "vocab_size", None):
            return int(config.vocab_size)
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name or "wte" in name:
                return int(param.shape[0])
        raise ValueError(
            "Could not infer vocab_size; pass it explicitly to estimate()."
        )

    def estimate(
        self,
        k: int = 16,
        num_sequences: int = 32,
        seq_len: int = 512,
        vocab_size: Optional[int] = None,
        seed: int = 0,
        include_non_layer: bool = False,
        progress: bool = False,
        input_ids=None,
    ) -> HessianTraceResult:
        """Run the Hutchinson trace estimator over all layers.

        Parameters mirror the paper's Appendix C.1: ``k = 16`` probes,
        ``num_sequences = 32`` random token sequences of ``seq_len = 512``.

        ``input_ids`` optionally supplies explicit token sequences (a single
        ``(1, T)`` tensor or a list of them).  When given, random sampling is
        bypassed -- this makes the estimator fully deterministic and lets unit
        tests compare against an exact, analytically known Hessian trace.
        """
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                sequences = [input_ids]
            else:
                sequences = [torch.as_tensor(ids) for ids in input_ids]
        else:
            sequences = None

        if sequences is None and vocab_size is None:
            vocab_size = self._infer_vocab_size()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        layer_indices = self.layer_indices(include_non_layer=include_non_layer)
        # ``k`` fixed Rademacher probe sets per layer -> the estimator is
        # unbiased for the sequence-averaged Hessian (Hutchinson, 1989).
        # ``probe_vectors[layer][j]`` is one probe vector per parameter of that
        # layer, for probe index ``j = 0..k-1``.
        probe_vectors: Dict[int, List[List[torch.Tensor]]] = {
            idx: [self._probe_vectors(self.layer_params[idx], generator) for _ in range(k)]
            for idx in layer_indices
        }
        acc: Dict[int, float] = {idx: 0.0 for idx in layer_indices}

        was_training = self.model.training
        self.model.eval()
        n_done = 0
        num_steps = len(sequences) if sequences is not None else num_sequences
        iterator = range(num_steps)
        if progress:  # pragma: no cover - purely cosmetic
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Hessian traces")
            except Exception:
                pass

        for step in iterator:
            if sequences is not None:
                batch_ids = sequences[step].to(self.device)
            else:
                batch_ids = self._sample_random_tokens(
                    vocab_size, seq_len, generator
                )
            self.model.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model, batch_ids)
            if not torch.isfinite(loss):
                continue

            params_all = [
                param
                for idx in layer_indices
                for _, param in self.layer_params[idx]
            ]
            if not params_all:
                raise RuntimeError("No trainable parameters found to differentiate.")

            # ---- pass 1: first-order gradient, kept in the graph ----
            grads = torch.autograd.grad(
                loss, params_all, create_graph=True, retain_graph=True
            )
            grad_lookup = {id(p): g for p, g in zip(params_all, grads)}

            for idx in layer_indices:
                params = [p for _, p in self.layer_params[idx]]
                for probes in probe_vectors[idx]:
                    # s = <g_l, v>
                    s = None
                    for probe, param in zip(probes, params):
                        term = torch.sum(probe * grad_lookup[id(param)])
                        s = term if s is None else s + term
                    # ---- pass 2: H_l v = grad_{W_l} <g_l, v> ----
                    hv = torch.autograd.grad(
                        s, params, retain_graph=True, allow_unused=True
                    )
                    q = 0.0
                    for probe, h in zip(probes, hv):
                        if h is None:
                            continue
                        q += float(torch.sum(probe * h))
                    acc[idx] += q
            n_done += 1
            del loss, grads, grad_lookup

        if was_training:
            self.model.train()

        denom = max(n_done, 1) * max(k, 1)
        traces = {
            idx: acc[idx] / denom
            for idx in layer_indices
            if include_non_layer or idx != self.mapper.non_layer_key
        }
        return HessianTraceResult(
            traces=traces,
            num_probes=k,
            num_sequences=n_done,
            seq_len=seq_len,
        )


# --------------------------------------------------------------------------- #
# Matrix-level helpers (used by the unit tests)
# --------------------------------------------------------------------------- #
def exact_trace_of_matrix(A: torch.Tensor) -> float:
    """Exact trace of a dense matrix (test helper)."""
    return float(torch.trace(A.to(torch.float64)))


def hutchinson_trace_of_matrix(A: torch.Tensor, k: int = 16, seed: int = 0) -> float:
    """Estimate ``tr(A)`` for a dense matrix ``A`` via Rademacher probes."""
    A = A.to(torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return hutchinson_trace(
        matvec=lambda v: A @ v,
        dim=A.shape[0],
        k=k,
        device=A.device,
        dtype=torch.float64,
        generator=generator,
    )


def layerwise_hessian_trace_estimate(
    model,
    k: int = 16,
    num_sequences: int = 32,
    seq_len: int = 512,
    vocab_size: Optional[int] = None,
    seed: int = 0,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    device: str = "cpu",
) -> Dict[int, float]:
    """Thin functional wrapper around :class:`LowRankHessianEstimator`."""
    estimator = LowRankHessianEstimator(
        model=model, layer_pattern=layer_pattern, device=device
    )
    return estimator.estimate(
        k=k,
        num_sequences=num_sequences,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    ).traces
