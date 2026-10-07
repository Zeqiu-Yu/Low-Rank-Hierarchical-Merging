"""Shared helpers for the LHM command-line scripts.

Keeping model loading / config plumbing in one place means every experiment
script stays a thin, reviewable wrapper around the library in ``lhm``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Make ``import lhm`` work when the scripts are executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lhm.config import LHMConfig, MODEL_PAIRS  # noqa: E402  (path juggling above)


DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--base_model", type=str, help="base / pretrained model")
    parser.add_argument("--reasoning_model", type=str, help="reasoning model to merge in")
    parser.add_argument("--scale", type=str, default="1.5b", choices=list(MODEL_PAIRS))
    parser.add_argument("--output_dir", type=str, default="outputs/lhm")
    parser.add_argument("--method", type=str, default="ta",
                        choices=["ta", "ties", "dare", "aim"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--lambda_bar", type=float, default=0.5)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    parser.add_argument("--param_value_mask_rate", type=float, default=0.8)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--aim_omega", type=float, default=0.4)
    parser.add_argument("--hessian_source", type=str, default="base",
                        choices=["base", "reasoning"])
    parser.add_argument("--trace_cache", type=str, default=None,
                        help="JSON file with pre-computed Hessian traces")
    parser.add_argument("--exclude_param_names_regex", type=str, default=None,
                        help="comma-separated regexes (default: embeddings/lm_head)")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON/YAML config file with defaults (CLI flags win)")
    return parser


# Config-file keys that do not match the CLI dest name one-to-one.
_CONFIG_KEY_ALIASES = {
    "trace_cache_path": "trace_cache",
    "traces_path": "trace_cache",
}


def load_config_file(path: str) -> Dict:
    """Load a JSON (or YAML, if PyYAML is installed) config file."""
    if path.endswith((".json",)):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"Reading {path} needs PyYAML, which is not installed. "
                "Use a .json config instead."
            ) from exc
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    raise SystemExit(f"Unsupported config extension for {path!r} (use .json/.yaml).")


def config_to_cli_defaults(payload: Dict) -> Dict:
    """Translate a serialized :class:`LHMConfig` into argparse defaults."""
    defaults: Dict = {}
    for key, value in payload.items():
        dest = _CONFIG_KEY_ALIASES.get(key, key)
        if dest == "exclude_param_names_regex" and isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        if dest == "scale" and isinstance(value, str):
            value = value.lower()
        defaults[dest] = value
    return defaults


def finalize_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args, letting a ``--config`` file supply defaults for known flags.

    Explicit command-line flags always win over the config file, so a config
    only needs to specify what differs from the paper defaults.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()
    if known.config:
        payload = load_config_file(known.config)
        defaults = config_to_cli_defaults(payload)
        valid = set(parser._option_string_actions.keys())  # type: ignore[attr-defined]
        filtered = {
            key: value
            for key, value in defaults.items()
            if f"--{key}" in valid or key in {a.dest for a in parser._actions}
        }
        parser.set_defaults(**filtered)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LHMConfig:
    pair = MODEL_PAIRS.get(args.scale)
    base_model = args.base_model or (pair.base_model if pair else None)
    reasoning_model = args.reasoning_model or (pair.reasoning_model if pair else None)
    if base_model is None or reasoning_model is None:
        raise SystemExit(
            "Please provide --base_model and --reasoning_model (or a valid --scale)."
        )
    exclude = None
    if args.exclude_param_names_regex is not None:
        exclude = [p for p in args.exclude_param_names_regex.split(",") if p]
    config = LHMConfig(
        base_model=base_model,
        reasoning_model=reasoning_model,
        scale=pair.scale if pair else args.scale,
        rank=args.rank,
        alpha=args.alpha,
        lambda_bar=args.lambda_bar,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
        method=args.method,
        param_value_mask_rate=args.param_value_mask_rate,
        mask_rate=args.mask_rate,
        aim_omega=args.aim_omega,
        hessian_source=args.hessian_source,
        output_dir=args.output_dir,
        trace_cache_path=args.trace_cache,
        exclude_param_names_regex=(
            exclude if exclude is not None else LHMConfig().exclude_param_names_regex
        ),
    )
    return config


def load_model(path: str, device: str = "cpu", dtype: str = "float32"):
    """Load a HuggingFace causal LM (imported lazily so tests need no transformers)."""
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, DTYPES.get(dtype, "float32"))
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model.to(device)


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_model(model, tokenizer, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir


def print_config(config: LHMConfig) -> None:
    print("=" * 78)
    for key, value in config.to_dict().items():
        print(f"{key:>28}: {value}")
    print("=" * 78)
