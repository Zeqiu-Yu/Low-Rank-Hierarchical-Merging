#!/usr/bin/env python
"""Hyper-parameter sweeps for Figure 4 of the paper.

Three independent sweeps are supported:

``--sweep rank``     k in {4, 8, 16, 32}          (effect of the probe rank)
``--sweep lambda``   lambda_bar in {0.3, 0.4, 0.5, 0.6}   (Pareto frontier)
``--sweep alpha``    alpha in {0.05, 0.10, 0.15, 0.20, 0.25} (bandwidth)

For each grid point the script either builds the merged model (Slow, but fully
reproducible) or, with ``--dry_run``, just emits the exact
``build_lhm_model.py`` commands plus the corresponding evaluation commands.

Usage
-----
python scripts/run_hyperparam_sweep.py --sweep alpha --scale 1.5b --method aim \
    --emit_manifest outputs/sweeps/alpha.json --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from _common import add_common_arguments, build_config, finalize_args, print_config
from lhm.config import ALPHA_GRID, LAMBDA_GRID, RANK_GRID


SWEEP_GRIDS = {
    "rank": RANK_GRID,
    "lambda": LAMBDA_GRID,
    "alpha": ALPHA_GRID,
}


def grid_for(sweep: str, custom: str | None):
    if custom:
        return [float(v) for v in custom.split(",") if v]
    return list(SWEEP_GRIDS[sweep])


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="LHM hyper-parameter sweep.")
    )
    parser.add_argument("--sweep", type=str, required=True,
                        choices=list(SWEEP_GRIDS))
    parser.add_argument("--grid", type=str, default=None,
                        help="comma-separated override for the sweep grid")
    parser.add_argument("--emit_manifest", type=str,
                        default="outputs/sweeps/manifest.json")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--eval_template", type=str,
                        default="bash evaluation/l2s_eval.sh qwen25-math-cot "
                                "{model_dir} 10240 0 gsm8k,math500,minerva_math,"
                                "olympiadbench,college_math,aime24")
    args = finalize_args(parser)

    config = build_config(args)
    print_config(config)
    grid = grid_for(args.sweep, args.grid)
    print(f"[run_hyperparam_sweep] sweep={args.sweep} grid={grid}")

    entries = []
    for value in grid:
        rank, alpha, lambda_bar = config.rank, config.alpha, config.lambda_bar
        if args.sweep == "rank":
            rank = int(value)
        elif args.sweep == "alpha":
            alpha = float(value)
        else:
            lambda_bar = float(value)

        run_dir = os.path.join(
            config.output_dir,
            f"sweep_{args.sweep}",
            f"value_{value}",
        )
        model_dir = os.path.join(run_dir, "model")
        build_cmd = (
            f"python scripts/build_lhm_model.py "
            f"--scale {args.scale} --method {config.method} "
            f"--rank {rank} --alpha {alpha} --lambda_bar {lambda_bar} "
            f"--output_dir {run_dir} --output_model_dir {model_dir} "
            f"--device {args.device} --dtype {args.dtype}"
        )
        eval_cmd = args.eval_template.format(model_dir=model_dir)
        entry = {
            "sweep": args.sweep,
            "value": value,
            "rank": rank,
            "alpha": alpha,
            "lambda_bar": lambda_bar,
            "run_dir": run_dir,
            "model_dir": model_dir,
            "build_command": build_cmd,
            "eval_command": eval_cmd,
        }
        entries.append(entry)

        if not args.dry_run:
            os.makedirs(run_dir, exist_ok=True)
            print(f"[run_hyperparam_sweep] building {run_dir} ...")
            subprocess.run(build_cmd, shell=True, check=True)

    os.makedirs(os.path.dirname(args.emit_manifest) or ".", exist_ok=True)
    with open(args.emit_manifest, "w", encoding="utf-8") as handle:
        json.dump(
            {"sweep": args.sweep, "grid": grid, "entries": entries},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"[run_hyperparam_sweep] manifest -> {args.emit_manifest}")
    if args.dry_run:
        for entry in entries:
            print("  " + entry["build_command"])


if __name__ == "__main__":
    sys.exit(main())
