"""Central configuration for the LHM experiments.

Every default value here is copied verbatim from the Experiments / Appendix of
the paper so that the run scripts never hard-code hyper-parameters.

Paper anchors
-------------
* rank ``k = 16``                       (Section 4 / Appendix C.1)
* bandwidth ``alpha = 0.15``            (Section 4 / Appendix C.1)
* ``lambda_bar`` grid ``{0.3, 0.4, 0.5, 0.6}``  (Section 4 / Appendix C.1)
* ``alpha`` sweep ``{0.05, 0.10, 0.15, 0.20, 0.25}`` (Section 4.3)
* 32 random token sequences of length 512 for Hessian estimation (Appendix C.1)
* 64-sample GSM8K validation split for ``lambda_bar`` selection (Appendix C.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json


# --------------------------------------------------------------------------- #
# Grids used across the experimental section.
# --------------------------------------------------------------------------- #
LAMBDA_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
ALPHA_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
RANK_GRID: Tuple[int, ...] = (4, 8, 16, 32)


@dataclass
class ModelPair:
    """A (base, reasoning) model pair with its scale tag."""

    name: str
    base_model: str
    reasoning_model: str
    scale: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# Model pairs used in the paper (Section 4.1 / Appendix C.1).
MODEL_PAIRS: Dict[str, ModelPair] = {
    "1.5b": ModelPair(
        name="Qwen2.5-Math-1.5B__R1-Distill-Qwen-1.5B",
        base_model="Qwen/Qwen2.5-Math-1.5B",
        reasoning_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        scale="1.5B",
    ),
    "7b": ModelPair(
        name="Qwen2.5-Math-7B__R1-Distill-Qwen-7B",
        base_model="Qwen/Qwen2.5-Math-7B",
        reasoning_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        scale="7B",
    ),
}


# The six benchmarks evaluated in the paper (Section 4.1).
BENCHMARKS: Tuple[str, ...] = (
    "gsm8k",
    "math500",
    "minerva_math",
    "olympiadbench",
    "college_math",
    "aime24",
)

# Short names used in the results tables.
BENCHMARK_SHORT: Dict[str, str] = {
    "gsm8k": "GSM",
    "math500": "M500",
    "minerva_math": "Min.",
    "olympiadbench": "Oly.",
    "college_math": "Coll.",
    "aime24": "AIME",
}


@dataclass
class LHMConfig:
    """Full configuration of an LHM run.

    Attributes
    ----------
    base_model, reasoning_model:
        HF identifiers (or local paths) of the two models constituting the pair.
    scale:
        ``"1.5B"`` or ``"7B"`` (only used for bookkeeping / book-keeping of
        output directories).
    rank:
        Hutchinson probe rank ``k`` (paper default 16).
    alpha:
        DCCM bandwidth ``alpha`` (paper default 0.15).
    lambda_bar:
        Global mean merge coefficient ``lambda_bar`` (selected from
        :data:`LAMBDA_GRID` via grid search).
    num_sequences:
        Number of randomly generated token sequences for trace estimation.
    seq_len:
        Length of each random token sequence.
    seed:
        Seed used for probe-vector and random-token generation.
    fusion_error_epsilon:
        Optional safety margin ``eps`` used by the *clipped* DCCM variant.  When
        provided, coefficients are clipped so that the quadratic fusion-error
        bound ``eps * sum_l lambda_l <= eps`` cannot be exceeded (see
        :mod:`lhm.dccm`).  ``None`` disables clipping and reproduces the paper's
        closed-form transform exactly.
    mean_anchor_exact:
        Whether to re-centre the coefficient vector so that its empirical mean
        equals ``lambda_bar`` exactly (default ``False`` = paper's affine form).
    method:
        Which strategy consumes the LHM coefficients.  One of
        ``{"ta", "ties", "dare", "aim"}`` -- these correspond to LHM-TA,
        LHM-TIES, LHM-DARE and LHM-AIM in the paper.
    """

    base_model: str = MODEL_PAIRS["1.5b"].base_model
    reasoning_model: str = MODEL_PAIRS["1.5b"].reasoning_model
    scale: str = "1.5B"

    rank: int = 16
    alpha: float = 0.15
    lambda_bar: float = 0.5
    num_sequences: int = 32
    seq_len: int = 512
    seed: int = 0

    fusion_error_epsilon: Optional[float] = None
    mean_anchor_exact: bool = False

    # Which model the per-layer curvature is measured on.  The Taylor analysis
    # in the paper expands around theta_0 (the base model), hence "base".
    hessian_source: str = "base"

    method: str = "ta"

    # ---- baseline-specific knobs -------------------------------------------
    param_value_mask_rate: float = 0.8      # TIES trim ratio
    mask_rate: float = 0.3                  # DARE drop rate
    aim_omega: float = 0.4                  # AIM balance factor

    # ---- paths -------------------------------------------------------------
    output_dir: str = "outputs/lhm"
    trace_cache_path: Optional[str] = None

    # ---- excluded parameter regexes ---------------------------------------
    # By default the embedding / LM-head / norm parameters are *not* merged,
    # following the long-to-short merging literature.
    exclude_param_names_regex: List[str] = field(
        default_factory=lambda: [
            r".*embed_tokens\.weight",
            r".*lm_head\.weight",
        ]
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict) -> "LHMConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# A convenience singleton mirroring the paper's default 1.5B recipe.
DEFAULT_CONFIG = LHMConfig()
