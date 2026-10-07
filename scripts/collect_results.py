#!/usr/bin/env python
"""Aggregate evaluation outputs into Markdown tables (Tables 1-3).

Example
-------
python scripts/collect_results.py \
    --entry "Qwen2.5-Math-1.5B=outputs/eval/qwen2.5-math-1.5b" \
    --entry "R1-Distill-1.5B=outputs/eval/r1-distill-1.5b" \
    --entry "LHM-AIM (Ours)=outputs/eval/lhm_aim" \
    --output_md outputs/tables/main_1p5b.md \
    --output_json outputs/tables/main_1p5b.json

Each ``--entry`` points at a directory containing one sub-directory per
benchmark with ``.jsonl`` evaluation outputs (the layout produced by
``evaluation/l2s_eval.sh``).
"""

from __future__ import annotations

import argparse
import os
import sys

from _common import load_tokenizer

from lhm.config import BENCHMARKS
from lhm.io_utils import save_json
from lhm.report import (
    average_over_benchmarks,
    render_markdown_table,
    summarize_eval_dir,
)


def parse_entries(values):
    entries = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--entry must be LABEL=DIR, got {item!r}")
        label, path = item.split("=", 1)
        entries[label] = path
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LHM evaluation results.")
    parser.add_argument("--entry", action="append", default=[],
                        help="LABEL=EVAL_DIR (repeatable)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="tokenizer path used to count response tokens")
    parser.add_argument("--benchmarks", type=str,
                        default=",".join(BENCHMARKS))
    parser.add_argument("--output_md", type=str, default="outputs/tables/results.md")
    parser.add_argument("--output_json", type=str, default="outputs/tables/results.json")
    args = parser.parse_args()

    entries = parse_entries(args.entry)
    if not entries:
        raise SystemExit("Provide at least one --entry LABEL=EVAL_DIR.")

    length_fn = None
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        length_fn = lambda text: len(tokenizer(text)["input_ids"])  # noqa: E731

    benchmarks = [b for b in args.benchmarks.split(",") if b]
    summaries = {}
    for label, path in entries.items():
        if not os.path.isdir(path):
            print(f"[collect_results] WARNING: {path} does not exist; skipping {label}")
            continue
        summaries[label] = summarize_eval_dir(path, length_fn=length_fn)

    markdown = render_markdown_table(summaries, benchmarks=benchmarks)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    payload = {
        label: {
            "per_benchmark": summary,
            "average": average_over_benchmarks(summary, benchmarks),
        }
        for label, summary in summaries.items()
    }
    save_json(payload, args.output_json)

    print(markdown)
    print(f"\n[collect_results] wrote {args.output_md} and {args.output_json}")


if __name__ == "__main__":
    sys.exit(main())
