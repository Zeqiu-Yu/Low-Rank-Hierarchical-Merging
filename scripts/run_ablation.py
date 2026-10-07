#!/usr/bin/env python
"""Incremental ablation study (Table 3 of the paper).

Configuration ladder, evaluated for both the TA and AIM base methods:

===============  ==========================================================
``ta``           uniform scalar coefficient (baseline)
``hessian``      coefficients from the Hessian trace only  (L)
``lsos``         coefficients from LSOS with a simple linear map (L+S)
``dccm``         full LHM: LSOS + dual-constraint DCCM   (L+S+D)
===============  ==========================================================

Usage
-----
python scripts/run_ablation.py --scale 1.5b --base_method ta --config dccm \
    --output_model_dir outputs/ablation/lhm_ta
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

from _common import (
    add_common_arguments,
    build_config,
    finalize_args,
    load_model,
    load_tokenizer,
    print_config,
    save_model,
)

from lhm.dccm import dccm_coefficients, min_max_normalize
from lhm.merging import LHMMerger, copy_params_to_model
from lhm.io_utils import save_dccm, save_lsos


ABLATION_CONFIGS = ("ta", "hessian", "lsos", "dccm")


def simple_linear_coefficients(lsos_scores: Dict[int, float], lambda_bar: float) -> Dict[int, float]:
    """Mean-anchored *linear* map used by the ``lsos`` ablation row.

    ``lambda_l = lambda_bar * S~_l / mean(S~)`` -- monotone in the LSOS score
    but without the DCCM mean-anchored affine transform / clipping.
    """
    normalized = min_max_normalize(lsos_scores)
    if not normalized:
        return {}
    values = np.asarray(list(normalized.values()), dtype=np.float64)
    mean = float(values.mean())
    if mean <= 1e-12:
        return {layer: float(lambda_bar) for layer in normalized}
    return {
        layer: float(lambda_bar) * value / mean
        for layer, value in normalized.items()
    }


def scalar_coefficients(lsos_scores: Dict[int, float], lambda_bar: float) -> Dict[int, float]:
    return {layer: float(lambda_bar) for layer in lsos_scores}


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="LHM incremental ablation.")
    )
    parser.add_argument("--base_method", type=str, default="ta", choices=["ta", "aim"])
    parser.add_argument("--rung", type=str, default="dccm", choices=list(ABLATION_CONFIGS),
                        help="which rung of the ablation ladder to build")
    parser.add_argument("--output_model_dir", type=str, required=True)
    args = finalize_args(parser)

    config = build_config(args)
    config.method = args.base_method
    print_config(config)
    print(f"[run_ablation] base_method={args.base_method} rung={args.rung}")

    base_model = load_model(config.base_model, device=args.device, dtype=args.dtype)
    reasoning_model = load_model(
        config.reasoning_model, device=args.device, dtype=args.dtype
    )
    tokenizer = load_tokenizer(config.base_model)

    merger = LHMMerger(config=config, device=args.device)
    lsos = merger.compute_lsos(
        base_model, reasoning_model,
        vocab_size=getattr(base_model.config, "vocab_size", None),
    )

    if args.rung == "ta":
        coefficients = scalar_coefficients(lsos.scores, config.lambda_bar)
    elif args.rung == "hessian":
        trace_only = {layer: lsos.hessian_traces[layer] for layer in lsos.scores}
        coefficients = dccm_coefficients(
            trace_only, lambda_bar=config.lambda_bar, alpha=config.alpha, clip=True
        ).coefficients
    elif args.rung == "lsos":
        coefficients = simple_linear_coefficients(lsos.scores, config.lambda_bar)
    else:  # dccm
        coefficients = merger.compute_coefficients(lsos).coefficients

    merged = merger.merge(base_model, reasoning_model, coefficients)
    copy_params_to_model(merged, base_model)
    save_model(base_model, tokenizer, args.output_model_dir)

    os.makedirs(args.output_model_dir, exist_ok=True)
    save_lsos(lsos, os.path.join(args.output_model_dir, "lsos.json"))
    print(f"[run_ablation] mean coefficient = "
          f"{sum(coefficients.values())/max(len(coefficients),1):.4f}")
    print(f"[run_ablation] saved to {args.output_model_dir}")


if __name__ == "__main__":
    sys.exit(main())
