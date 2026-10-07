#!/usr/bin/env python
"""Select ``lambda_bar`` via a lightweight grid search on a GSM8K split.

Appendix C.1 of the paper: *"we select lambda_bar via a lightweight grid search
over {0.3, 0.4, 0.5, 0.6} on a 64-sample validation split of GSM8K"*.

The Hessian traces and LSOS scores depend only on the model pair -- not on
``lambda_bar`` -- so they are computed **once** and reused for every grid point
(the DCCM step is a closed-form transform, i.e. essentially free).

Because scoring the split requires the Qwen2.5-Math evaluation toolkit, the
script can either

* run an external evaluation command template for each grid point, or
* run in ``--dry_run`` mode and simply emit the exact commands to run.

Usage
-----
python scripts/tune_lambda.py --scale 1.5b --method aim --num_samples 64 \
    --eval_command_template \
      'bash evaluation/l2s_eval.sh qwen25-math-cot {model_dir} 10240 0 gsm8k' \
    --output outputs/tuning/lhm_aim_lambda.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, Optional

from _common import (
    add_common_arguments,
    build_config,
    finalize_args,
    load_model,
    load_tokenizer,
    print_config,
    save_model,
)

from lhm.config import LAMBDA_GRID
from lhm.io_utils import ensure_dir, save_json, save_lsos
from lhm.merging import LHMMerger, copy_params_to_model


def parse_accuracy(text: str) -> Optional[float]:
    """Extract the last ``accuracy: <number>`` style figure from eval stdout."""
    matches = re.findall(r"(\d+\.\d+)\s*%?", text)
    if not matches:
        return None
    return float(matches[-1])


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Grid-search the LHM lambda_bar.")
    )
    parser.add_argument("--lambda_grid", type=str,
                        default=",".join(str(v) for v in LAMBDA_GRID))
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--eval_command_template", type=str, default=None,
                        help="command template with {model_dir}, {num_samples}, "
                             "{output_dir} placeholders")
    parser.add_argument("--output", type=str,
                        default="outputs/tuning/lambda_selection.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="only emit the commands, do not evaluate")
    parser.add_argument("--save_candidates", action="store_true",
                        help="persist every candidate merged model")
    args = finalize_args(parser)

    grid = [float(v) for v in args.lambda_grid.split(",") if v]
    config = build_config(args)
    # The grid search itself sets ``lambda_bar`` per point.
    print_config(config)

    base_model = load_model(config.base_model, device=args.device, dtype=args.dtype)
    reasoning_model = load_model(
        config.reasoning_model, device=args.device, dtype=args.dtype
    )
    tokenizer = load_tokenizer(config.base_model) if not args.dry_run else None

    merger = LHMMerger(config=config, device=args.device)
    print("[tune_lambda] computing Hessian traces + LSOS once (shared across grid) ...")
    lsos = merger.compute_lsos(
        base_model,
        reasoning_model,
        vocab_size=getattr(base_model.config, "vocab_size", None),
    )

    tuning_dir = ensure_dir(os.path.join(config.output_dir, "tuning"))
    save_lsos(lsos, os.path.join(tuning_dir, "lsos.json"))

    results: Dict[str, Dict] = {}
    commands = []
    for lambda_bar in grid:
        config.lambda_bar = lambda_bar
        dccm = merger.compute_coefficients(lsos)
        merged = merger.merge(base_model, reasoning_model, dccm.coefficients)
        candidate_dir = os.path.join(
            tuning_dir, f"lambda_{lambda_bar:.2f}", "model"
        )
        command = None
        if args.eval_command_template:
            command = args.eval_command_template.format(
                model_dir=candidate_dir,
                num_samples=args.num_samples,
                output_dir=os.path.join(tuning_dir, f"lambda_{lambda_bar:.2f}"),
            )
            commands.append(command)

        accuracy = None
        if not args.dry_run and args.eval_command_template:
            copy_params_to_model(merged, base_model)
            save_model(base_model, tokenizer, candidate_dir)
            print(f"[tune_lambda] evaluating lambda_bar={lambda_bar:.2f} ...")
            proc = subprocess.run(command, shell=True, capture_output=True, text=True)
            accuracy = parse_accuracy(proc.stdout + proc.stderr)
        elif args.save_candidates:
            copy_params_to_model(merged, base_model)
            save_model(base_model, tokenizer, candidate_dir)

        results[f"{lambda_bar:.2f}"] = {
            "lambda_bar": lambda_bar,
            "accuracy": accuracy,
            "mean_coefficient": dccm.achieved_mean,
            "fusion_error_ratio": dccm.fusion_error_ratio,
            "command": command,
        }
        print(
            f"[tune_lambda] lambda_bar={lambda_bar:.2f}  "
            f"mean(lambda)={dccm.achieved_mean:.4f}  "
            f"accuracy={accuracy}"
        )

    scored = [r for r in results.values() if r["accuracy"] is not None]
    if scored:
        best = max(scored, key=lambda r: r["accuracy"])
    else:
        # No evaluation available -> fall back to the balanced operating point.
        best = results.get("0.50") or list(results.values())[0]

    payload = {
        "grid": grid,
        "num_samples": args.num_samples,
        "results": results,
        "best_lambda_bar": best["lambda_bar"],
        "selection_rule": (
            "max accuracy on the GSM8K split"
            if scored
            else "fallback: balanced operating point (no evaluation run)"
        ),
        "commands": commands,
    }
    save_json(payload, args.output)
    print(f"[tune_lambda] best lambda_bar = {best['lambda_bar']}")
    print(f"[tune_lambda] wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
