#!/usr/bin/env python
"""Reproduce the baseline merging methods (Tables 1-2 of the paper).

Supported ``--baseline`` values and their default hyper-parameters follow the
long-to-short merging repo (and the LHM paper's Table/Configurations section):

    model_soup | ta (alpha=0.7) | ties (k=0.8, alpha=1.0)
    dare (p=0.3) | aim (omega=0.4) | acm_ties (T=3.0) | orca_aim (omega=0.4)

Usage
-----
python scripts/merge_baselines.py --baseline ta --scale 1.5b \
    --output_model_dir outputs/models/ta_qwen2.5-1.5b --device cuda --dtype bfloat16
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

from lhm.baselines import (
    BASELINE_REGISTRY,
    DEFAULT_BASELINE_HYPERPARAMS,
    acm_merging,
    aim_merging,
    dare_merging,
    model_soup,
    orca_merging,
    task_arithmetic,
    ties_merging,
)
from lhm.activations import collect_activation_norms
from lhm.merging import copy_params_to_model


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Build a baseline-merged model.")
    )
    parser.add_argument("--baseline", type=str, default="ta",
                        choices=list(BASELINE_REGISTRY))
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--scaling_coefficient", type=float, default=0.7)
    parser.add_argument("--outlier_threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--calib_seq_len", type=int, default=512)
    args = finalize_args(parser)

    config = build_config(args)
    print_config(config)
    print(f"[merge_baselines] baseline = {args.baseline} "
          f"(defaults: {DEFAULT_BASELINE_HYPERPARAMS.get(args.baseline, {})})")

    base_model = load_model(config.base_model, device=args.device, dtype=args.dtype)
    reasoning_model = load_model(
        config.reasoning_model, device=args.device, dtype=args.dtype
    )
    tokenizer = load_tokenizer(config.base_model)
    common = dict(
        exclude_param_names_regex=config.exclude_param_names_regex,
    )

    if args.baseline == "model_soup":
        merged = model_soup([base_model, reasoning_model])
    elif args.baseline == "ta":
        merged = task_arithmetic(
            base_model, reasoning_model,
            scaling_coefficient=args.scaling_coefficient, **common
        )
    elif args.baseline == "ties":
        merged = ties_merging(
            base_model, reasoning_model,
            param_value_mask_rate=config.param_value_mask_rate,
            scaling_coefficient=args.scaling_coefficient, **common
        )
    elif args.baseline == "dare":
        merged = dare_merging(
            base_model, reasoning_model,
            mask_rate=config.mask_rate, **common
        )
    elif args.baseline in ("aim", "orca_aim"):
        print("[merge_baselines] collecting per-layer activation norms ...")
        import torch

        vocab_size = getattr(base_model.config, "vocab_size", 32000)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        input_ids = torch.randint(
            0, vocab_size, (1, args.calib_seq_len), generator=gen
        )
        activation_norms = collect_activation_norms(
            reasoning_model, input_ids, device=args.device
        )
        if args.baseline == "orca_aim":
            merged = orca_merging(
                base_model, reasoning_model,
                activation_norms=activation_norms,
                outlier_threshold=args.outlier_threshold, **common
            )
        else:
            merged = aim_merging(
                base_model, reasoning_model,
                activation_norms=activation_norms,
                omega=config.aim_omega, **common
            )
    elif args.baseline == "acm_ties":
        print("[merge_baselines] collecting activation norms for both models ...")
        import torch

        vocab_size = getattr(base_model.config, "vocab_size", 32000)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        input_ids = torch.randint(
            0, vocab_size, (1, args.calib_seq_len), generator=gen
        )
        norms_base = collect_activation_norms(base_model, input_ids, device=args.device)
        norms_reason = collect_activation_norms(
            reasoning_model, input_ids, device=args.device
        )
        merged = acm_merging(
            base_model, reasoning_model,
            activation_norms_base=norms_base,
            activation_norms_reason=norms_reason,
            temperature=args.temperature, **common
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise SystemExit(f"Unknown baseline {args.baseline!r}")

    copy_params_to_model(merged, base_model)
    save_model(base_model, tokenizer, args.output_model_dir)
    print(f"[merge_baselines] merged model saved to {args.output_model_dir}")


if __name__ == "__main__":
    sys.exit(main())
