"""Aggregate evaluation outputs into the paper's results tables.

The Qwen2.5-Math evaluation toolkit writes one ``.jsonl`` file per benchmark,
with each line containing ``{"code": [...], "score": [...], ...}``.  This
module turns those files into accuracy / response-length summaries and renders
them as Markdown tables matching Tables 1-3 of the paper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import BENCHMARKS, BENCHMARK_SHORT


def _default_length(text: str) -> int:
    """Fallback length measure: number of whitespace-separated tokens."""
    return len(text.split())


def summarize_jsonl(path: str | os.PathLike, length_fn: Callable[[str], int]) -> Dict:
    """Accuracy and mean response length of a single ``.jsonl`` file."""
    scores: List[float] = []
    lengths: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            code = record.get("code", [""])
            text = code[0] if isinstance(code, (list, tuple)) and code else str(code)
            score = record.get("score", [0])
            value = score[0] if isinstance(score, (list, tuple)) and score else score
            scores.append(float(value))
            lengths.append(int(length_fn(text)))
    n = len(scores)
    return {
        "n": n,
        "accuracy": (100.0 * sum(scores) / n) if n else float("nan"),
        "mean_length": (sum(lengths) / n) if n else float("nan"),
    }


def summarize_eval_dir(
    eval_dir: str | os.PathLike,
    length_fn: Optional[Callable[[str], int]] = None,
) -> Dict[str, Dict]:
    """Summarise every benchmark sub-directory under ``eval_dir``."""
    length_fn = length_fn or _default_length
    root = Path(eval_dir)
    summary: Dict[str, Dict] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(dataset_dir.glob("*.jsonl"))
        if not files:
            continue
        merged_scores: List[float] = []
        merged_lengths: List[int] = []
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    code = record.get("code", [""])
                    text = code[0] if code else ""
                    score = record.get("score", [0])
                    value = score[0] if isinstance(score, (list, tuple)) else score
                    merged_scores.append(float(value))
                    merged_lengths.append(int(length_fn(text)))
        n = len(merged_scores)
        summary[dataset_dir.name] = {
            "n": n,
            "accuracy": (100.0 * sum(merged_scores) / n) if n else float("nan"),
            "mean_length": (sum(merged_lengths) / n) if n else float("nan"),
        }
    return summary


def average_over_benchmarks(
    summary: Mapping[str, Mapping[str, float]],
    benchmarks: Sequence[str] = BENCHMARKS,
) -> Dict[str, float]:
    """Mean accuracy / length across benchmarks (missing ones are skipped)."""
    accs = [summary[b]["accuracy"] for b in benchmarks if b in summary]
    lengths = [summary[b]["mean_length"] for b in benchmarks if b in summary]
    return {
        "accuracy": sum(accs) / len(accs) if accs else float("nan"),
        "mean_length": sum(lengths) / len(lengths) if lengths else float("nan"),
        "num_benchmarks": len(accs),
    }


def render_markdown_row(
    label: str,
    summary: Mapping[str, Mapping[str, float]],
    benchmarks: Sequence[str] = BENCHMARKS,
) -> str:
    """Render one results-table row: accuracy then length per benchmark."""
    acc_cells = [
        f"{summary[b]['accuracy']:.1f}" if b in summary else "-" for b in benchmarks
    ]
    len_cells = [
        f"{int(round(summary[b]['mean_length']))}" if b in summary else "-"
        for b in benchmarks
    ]
    avg = average_over_benchmarks(summary, benchmarks)
    acc_avg = f"{avg['accuracy']:.1f}" if avg["num_benchmarks"] else "-"
    len_avg = f"{int(round(avg['mean_length']))}" if avg["num_benchmarks"] else "-"
    return " | ".join([label, *acc_cells, acc_avg, *len_cells, len_avg])


def render_markdown_table(
    rows: Mapping[str, Mapping[str, Mapping[str, float]]],
    benchmarks: Sequence[str] = BENCHMARKS,
) -> str:
    """Render a full Markdown table given ``{label: summary}``."""
    short = [BENCHMARK_SHORT.get(b, b) for b in benchmarks]
    header_acc = " | ".join(["Model", *short, "Avg."])
    header_len = " | ".join(["", *short, "Avg."])
    lines = [
        "| " + header_acc + " | " + header_len + " |",
        "|" + "---|" * (1 + len(short) + 1 + len(short) + 1),
    ]
    for label, summary in rows.items():
        lines.append("| " + render_markdown_row(label, summary, benchmarks) + " |")
    return "\n".join(lines)


def save_summary(
    summary: Mapping[str, Mapping[str, float]],
    path: str | os.PathLike,
) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
    return path
