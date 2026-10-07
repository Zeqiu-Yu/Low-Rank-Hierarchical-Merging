#!/usr/bin/env python
"""Build a merged model with Low-Rank Hierarchical Merging (full Algorithm 1).

This is the main entry point used by every experiment in the paper:

    python scripts/build_lhm_model.py \
        --scale 1.5b --method aim \
        --rank 16 --alpha 0.15 --lambda_bar 0.5 \
        --output_dir outputs/models/lhm_aim_qwen2.5-1.5b \
        --device cuda --dtype bfloat16

It performs all five phases:

1. task-vector decomposition  (base vs. reasoning weights)
2. data-free low-rank Hessian trace estimation (Hutchinson)
3. LSOS index computation
4. DCCM coefficient mapping
5. layer-adaptive merging + ``save_pretrained``

All intermediate artefacts are written next to the merged model so that every
number in the paper can be traced back to the numbers produced here.
"""

from __future__ import annotations

import argparse
import os
import sys

from _common import (
    add_common_arguments,
    build_config,
    finalize_args,
    load_model,
    load_tokenizer,
    print_config,
    save_model,
)

from lhm.io_utils import (
    load_hessian_traces,
    run_directory,
    save_config,
    save_dccm,
    save_hessian_traces,
    save_lsos,
)
from lhm.merging import LHMMerger, copy_params_to_model


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Build a merged model with LHM.")
    )
    parser.add_argument("--output_model_dir", type=str, default=None,
                        help="destination of the merged model (default: run directory)")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", dest="save_model", action="store_false")
    parser.add_argument("--tag", type=str, default=None)
    args = finalize_args(parser)

    config = build_config(args)
    print_config(config)

    work_dir = run_directory(
        config.output_dir,
        config.scale,
        config.method,
        config.lambda_bar,
        config.alpha,
        config.rank,
        tag=args.tag,
    )
    print(f"[build_lhm_model] artefacts -> {work_dir}")

    base_model = load_model(config.base_model, device=args.device, dtype=args.dtype)
    reasoning_model = load_model(
        config.reasoning_model, device=args.device, dtype=args.dtype
    )
    tokenizer = None
    if args.save_model:
        tokenizer = load_tokenizer(config.base_model)

    merger = LHMMerger(config=config, device=args.device)

    # ---- Phase 2: Hessian traces (cached or freshly estimated) -------------
    if config.trace_cache_path:
        print(f"[build_lhm_model] loading cached Hessian traces: {config.trace_cache_path}")
        traces = load_hessian_traces(config.trace_cache_path)
    else:
        print("[build_lhm_model] estimating Hessian traces (Phase 2) ...")
        source_model = (
            base_model if config.hessian_source == "base" else reasoning_model
        )
        traces = merger.estimate_hessian_traces(
            source_model, vocab_size=getattr(base_model.config, "vocab_size", None)
        )

    # ---- Phases 3-5 --------------------------------------------------------
    lsos = merger.compute_lsos(
        base_model, reasoning_model, hessian_traces=traces.traces
    )
    dccm = merger.compute_coefficients(lsos)
    merged_params = merger.merge(base_model, reasoning_model, dccm.coefficients)

    # ---- persist every intermediate artefact --------------------------------
    save_config(config, os.path.join(work_dir, "config.json"))
    save_hessian_traces(traces, os.path.join(work_dir, "hessian_traces.json"))
    save_lsos(lsos, os.path.join(work_dir, "lsos.json"))
    save_dccm(dccm, os.path.join(work_dir, "dccm.json"))

    print(f"[build_lhm_model] LSOS  mean={sum(lsos.scores.values())/max(lsos.L,1):.4e} "
          f"max={max(lsos.scores.values()) if lsos.scores else float('nan'):.4e}")
    print(f"[build_lhm_model] DCCM  mean={dccm.achieved_mean:.4f} "
          f"(target {dccm.target_lambda_bar}) fusion-error ratio="
          f"{dccm.fusion_error_ratio:.4f}")

    copy_params_to_model(merged_params, base_model)

    if args.save_model:
        output_model_dir = args.output_model_dir or os.path.join(work_dir, "model")
        save_model(base_model, tokenizer, output_model_dir)
        print(f"[build_lhm_model] merged model saved to {output_model_dir}")
    else:
        print("[build_lhm_model] --no_save_model set; parameters merged in memory only")


if __name__ == "__main__":
    sys.exit(main())
