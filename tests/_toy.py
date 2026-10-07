"""Tiny, dependency-light ``nn.Module`` fixtures for the LHM unit tests.

The models mirror the *naming* of a HuggingFace causal LM
(``model.layers.<i>.…``) so that every layer-mapping / Hessian / LSOS / merge
routine can be exercised exactly as in the real experiments -- but with a few
thousand parameters instead of billions, and without any model download.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToyConfig:
    def __init__(self, vocab_size: int = 32, hidden: int = 8):
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.max_position_embeddings = 64


class ToyBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


class ToyLM(nn.Module):
    """Minimal causal LM with Qwen-style parameter names."""

    def __init__(self, vocab_size: int = 32, hidden: int = 8, num_layers: int = 3):
        super().__init__()
        self.config = ToyConfig(vocab_size=vocab_size, hidden=hidden)

        # ``model.*`` container so parameter names look like
        #   model.embed_tokens.weight
        #   model.layers.0.fc1.weight
        #   model.norm.weight
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.model.layers = nn.ModuleList(
            [ToyBlock(hidden) for _ in range(num_layers)]
        )
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        self._vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            x = block(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self._vocab_size),
                labels.reshape(-1),
            )
        return _Output(logits=logits, loss=loss)


class _Output:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss


class QuadraticModel(nn.Module):
    """Model whose per-layer Hessian trace has a closed form.

    Forward: ``x = embed(input_ids)`` (embedding frozen), ``y = W x``,
    ``loss = 0.5 * sum(y^2)``.  For a single linear layer without bias the
    Hessian of the loss w.r.t. ``W`` is ``H = sum_b (x_b x_b^T) ⊗ I_out``, so

        tr(H) = out_features * sum_b ||x_b||_2^2 .

    This lets the tests check the double-backward Hutchinson estimator against
    an exact analytic value.
    """

    def __init__(self, vocab_size: int = 16, in_features: int = 6, out_features: int = 5):
        super().__init__()
        self.config = ToyConfig(vocab_size=vocab_size, hidden=in_features)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, in_features)
        self.model.embed_tokens.weight.requires_grad_(False)
        block = nn.Module()
        block.fc = nn.Linear(in_features, out_features, bias=False)
        self.model.layers = nn.ModuleList([block])
        self.out_features = out_features

    def forward(self, input_ids: torch.Tensor, labels=None):
        x = self.model.embed_tokens(input_ids)
        y = self.model.layers[0].fc(x)
        loss = 0.5 * (y ** 2).sum()
        return _Output(logits=y, loss=loss)

    def exact_trace(self, input_ids: torch.Tensor) -> float:
        with torch.no_grad():
            x = self.model.embed_tokens(input_ids)
            return float(self.out_features * (x ** 2).sum())


def make_pair(seed: int = 0, num_layers: int = 3, hidden: int = 8, vocab: int = 32):
    """Return a ``(base, reasoning)`` toy model pair with different weights."""
    torch.manual_seed(seed)
    base = ToyLM(vocab_size=vocab, hidden=hidden, num_layers=num_layers)
    torch.manual_seed(seed + 1)
    reasoning = ToyLM(vocab_size=vocab, hidden=hidden, num_layers=num_layers)
    # make the task vector non-trivial and layer-dependent
    with torch.no_grad():
        for idx, (b, r) in enumerate(zip(base.model.layers, reasoning.model.layers)):
            r.fc1.weight.add_(0.1 * (idx + 1) * torch.randn_like(r.fc1.weight))
            r.fc2.weight.add_(0.05 * (idx + 1) * torch.randn_like(r.fc2.weight))
    return base, reasoning
