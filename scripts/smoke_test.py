#!/usr/bin/env python
"""Local smoke test: run all five LHM phases on a tiny model.

No language model is downloaded or loaded -- a 3-layer toy transformer with
Qwen-style parameter names is used instead.  Every phase is executed and the
resulting artefacts are written to a temporary directory:

    python scripts/smoke_test.py

This is the quickest way to verify that a fresh checkout is functional.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lhm.config import LHMConfig
from lhm.io_utils import (
    load_dccm,
    load_lsos,
    save_config,
    save_dccm,
    save_lsos,
)
from lhm.merging import LHMMerger


# --------------------------------------------------------------------------- #
# tiny model
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class _ToyLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, layers: int = 3):
        super().__init__()
        self.config = type("Cfg", (), {"vocab_size": vocab, "hidden_size": hidden})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._vocab = vocab

    def forward(self, input_ids, labels=None):
        x = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            x = block(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self._vocab), labels.reshape(-1)
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


def build_pair(seed: int = 0):
    torch.manual_seed(seed)
    base = _ToyLM()
    torch.manual_seed(seed + 1)
    reasoning = _ToyLM()
    with torch.no_grad():
        for idx, (b, r) in enumerate(zip(base.model.layers, reasoning.model.layers)):
            r.fc1.weight.add_(0.1 * (idx + 1) * torch.randn_like(r.fc1.weight))
            r.fc2.weight.add_(0.05 * (idx + 1) * torch.randn_like(r.fc2.weight))
    return base, reasoning


def main() -> int:
    parser = argparse.ArgumentParser(description="LHM local smoke test.")
    parser.add_argument("--method", default="aim", choices=["ta", "ties", "dare", "aim"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num_sequences", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    args = parser.parse_args()

    print("=" * 70)
    print(" LHM smoke test (toy model -- no external model is loaded)")
    print("=" * 70)

    base, reasoning = build_pair()
    config = LHMConfig(
        rank=args.rank,
        alpha=0.15,
        lambda_bar=0.5,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        method=args.method,
        exclude_param_names_regex=[
            r".*embed_tokens\.weight",
            r".*lm_head\.weight",
        ],
    )
    print(f"config: method={config.method} k={config.rank} alpha={config.alpha} "
          f"lambda_bar={config.lambda_bar}")

    # ---- Phase 2: Hessian traces -----------------------------------------
    merger = LHMMerger(config=config)
    traces = merger.estimate_hessian_traces(base)
    print(f"\n[Phase 2] Hessian traces ({len(traces.traces)} layers):")
    for layer, value in traces.traces.items():
        print(f"    layer {layer}: tr(H) = {value:+.6e}")

    # ---- Phase 3: LSOS ----------------------------------------------------
    lsos = merger.compute_lsos(base, reasoning, hessian_traces=traces.traces)
    print("\n[Phase 3] LSOS index (trace x ||delta||^2):")
    for layer, value in lsos.scores.items():
        print(f"    layer {layer}: LSOS = {value:.6e}")

    # ---- Phase 4: DCCM ----------------------------------------------------
    dccm = merger.compute_coefficients(lsos)
    print("\n[Phase 4] DCCM coefficients:")
    for layer, value in dccm.coefficients.items():
        print(f"    layer {layer}: lambda = {value:.4f}")
    print(f"    mean(lambda) = {dccm.achieved_mean:.4f} (target {dccm.target_lambda_bar})")

    # ---- Phase 5: merge ---------------------------------------------------
    merged = merger.merge(base, reasoning, dccm.coefficients)
    moved = 0
    for name, param in base.named_parameters():
        if name in merged and not torch.allclose(merged[name], param):
            moved += 1
    print(f"\n[Phase 5] merged parameters changed: {moved}")
    print(f"    embeddings untouched: "
          f"{torch.allclose(merged['model.embed_tokens.weight'], base.model.embed_tokens.weight)}")

    # ---- artefact IO ------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = save_config(config, os.path.join(tmp, "config.json"))
        lsos_path = save_lsos(lsos, os.path.join(tmp, "lsos.json"))
        dccm_path = save_dccm(dccm, os.path.join(tmp, "dccm.json"))
        assert load_lsos(lsos_path).scores == lsos.scores
        assert load_dccm(dccm_path).coefficients == dccm.coefficients
        print(f"\n[IO] wrote {cfg_path}, {lsos_path}, {dccm_path}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
