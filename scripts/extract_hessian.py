#!/usr/bin/env python
"""Phase 2 in isolation: estimate per-layer Hessian traces and cache them.

Usage
-----
python scripts/extract_hessian.py \
    --scale 1.5b --hessian_source base \
    --rank 16 --num_sequences 32 --seq_len 512 \
    --output outputs/traces/qwen2.5-math-1.5b.json

The cached JSON file can be fed to ``build_lhm_model.py`` through
``--trace_cache`` so that the (cheap) trace estimation is never repeated.
"""

from __future__ import annotations

import argparse
import sys

from _common import (
    add_common_arguments,
    build_config,
    finalize_args,
    load_model,
    print_config,
)

from lhm.hessian import LowRankHessianEstimator
from lhm.io_utils import save_hessian_traces


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Estimate per-layer Hessian traces (LHM Phase 2).")
    )
    parser.add_argument("--output", type=str, required=True,
                        help="destination JSON file for the Hessian traces")
    parser.add_argument("--include_non_layer", action="store_true",
                        help="also estimate the trace of non-block parameters")
    args = finalize_args(parser)

    config = build_config(args)
    print_config(config)

    source = (
        config.base_model if config.hessian_source == "base" else config.reasoning_model
    )
    print(f"[extract_hessian] curvature source model: {source}")
    model = load_model(source, device=args.device, dtype=args.dtype)

    estimator = LowRankHessianEstimator(model=model, device=args.device)
    result = estimator.estimate(
        k=config.rank,
        num_sequences=config.num_sequences,
        seq_len=config.seq_len,
        seed=config.seed,
        include_non_layer=args.include_non_layer,
        progress=True,
    )

    for layer, trace in result.traces.items():
        print(f"  layer {layer:>3}  tr(H) = {trace:.6e}")

    path = save_hessian_traces(result, args.output)
    print(f"[extract_hessian] wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
