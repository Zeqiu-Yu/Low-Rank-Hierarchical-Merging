"""Low-Rank Hierarchical Merging (LHM).

Reference implementation of
    *Low-Rank Hierarchical Merging for Efficient Long-to-Short Reasoning*
    (NeurIPS 2026).

The package is organised around the three components of LHM:

``lhm.hessian``
    Data-free low-rank Hessian trace estimation via Hutchinson's estimator
    (Phase 2 of Algorithm 1).
``lhm.lsos``
    Layer Second-Order Sensitivity index (Phase 3).
``lhm.dccm``
    Dual-Constraint Coefficient Mapping (Phase 4).

They are wired together by :class:`lhm.merging.LHMMerger` (Phase 1 and 5), and
the remaining baselines live in :mod:`lhm.baselines`.
"""

from .config import LHMConfig, DEFAULT_CONFIG, LAMBDA_GRID, ALPHA_GRID
from .task_vector import TaskVector
from .layer_utils import LayerKeyMapper, group_param_names_by_layer
from .hessian import LowRankHessianEstimator, hutchinson_trace
from .lsos import (
    LSOSResult,
    compute_lsos,
    task_vector_squared_norms,
    aggregate_squared_norms_by_layer,
)
from .dccm import (
    DCCMResult,
    min_max_normalize,
    dccm_coefficients,
    dccm_anchored_mean,
)
from .merging import LHMMerger, merge_with_coefficients

__all__ = [
    "LHMConfig",
    "DEFAULT_CONFIG",
    "LAMBDA_GRID",
    "ALPHA_GRID",
    "TaskVector",
    "LayerKeyMapper",
    "group_param_names_by_layer",
    "LowRankHessianEstimator",
    "hutchinson_trace",
    "LSOSResult",
    "compute_lsos",
    "task_vector_squared_norms",
    "aggregate_squared_norms_by_layer",
    "DCCMResult",
    "min_max_normalize",
    "dccm_coefficients",
    "dccm_anchored_mean",
    "LHMMerger",
    "merge_with_coefficients",
]

__version__ = "1.0.0"
