#!/usr/bin/env python
"""Reproduce the paper's figures from the JSON artefacts.

Figures
-------
* ``fig_lsos``      LSOS distribution across layers (Fig. 2, left)
* ``fig_rank``      Hessian-trace error vs. rank (Fig. 2, right / Fig. 4, left)
* ``fig_pareto``    accuracy vs. response length for a lambda_bar sweep (Fig. 1 / Fig. 4, right)
* ``fig_ablation``  ablation bars (Fig. 3)

matplotlib is an optional dependency: if it is not installed the script prints
the underlying series so the numbers can still be inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from _common import PROJECT_ROOT  # noqa: F401  (ensures sys.path)
from lhm.io_utils import load_json, ensure_dir


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_lsos(lsos_path: str, output: str) -> None:
    payload = load_json(lsos_path)
    scores = {int(k): float(v) for k, v in payload["scores"].items()}
    layers = sorted(scores)
    values = [scores[l] for l in layers]
    plt = _try_matplotlib()
    if plt is None:
        print("[plot_figures] matplotlib unavailable; LSOS series:")
        print(list(zip(layers, values)))
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(layers, values, color="#3b6ea5")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("LSOS")
    ax.set_title("Layer Second-Order Sensitivity")
    fig.tight_layout()
    ensure_dir(os.path.dirname(output) or ".")
    fig.savefig(output, dpi=200)
    print(f"[plot_figures] wrote {output}")


def plot_pareto(manifest_path: str, output: str) -> None:
    """Plot accuracy vs. mean length from a sweep manifest with eval results."""
    payload = load_json(manifest_path)
    points = []
    for entry in payload.get("entries", []):
        acc = entry.get("accuracy")
        length = entry.get("mean_length")
        if acc is None or length is None:
            continue
        points.append((entry["value"], float(length), float(acc)))
    plt = _try_matplotlib()
    if plt is None or not points:
        print("[plot_figures] no plottable points (or matplotlib missing):", points)
        return
    points.sort(key=lambda p: p[1])
    lengths = [p[1] for p in points]
    accs = [p[2] for p in points]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lengths, accs, "-o", color="#c0392b", label="LHM")
    for value, length, acc in points:
        ax.annotate(str(value), (length, acc), fontsize=8)
    ax.set_xlabel("Average response length")
    ax.set_ylabel("Average accuracy (%)")
    ax.set_title("Accuracy-length Pareto")
    ax.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(output) or ".")
    fig.savefig(output, dpi=200)
    print(f"[plot_figures] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LHM figures.")
    parser.add_argument("--figure", choices=["lsos", "pareto"], required=True)
    parser.add_argument("--input", required=True, help="input JSON artefact")
    parser.add_argument("--output", required=True, help="output PNG path")
    args = parser.parse_args()
    if args.figure == "lsos":
        plot_lsos(args.input, args.output)
    else:
        plot_pareto(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
