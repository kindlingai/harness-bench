"""Markdown comparison of several models' runs, from the JSONL files `run --out` writes."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .runner import REL_WARN, RunResult, summarise
from .stats import mean_ci


def load(files: list[str]) -> list[RunResult]:
    rows = []
    for f in files:
        rows += [RunResult(**json.loads(line)) for line in Path(f).read_text().splitlines() if line.strip()]
    return rows


def _pct(g: dict) -> str:
    if not g or not g["runs"]:
        return "–"
    lo, hi = g["pass_ci95"]
    return f"{g['pass_rate']:.0%} [{lo:.0%}, {hi:.0%}]"


def _mean(v: dict | None, digits: int = 1) -> str:
    if not v or v.get("mean") is None:
        return "–"
    if v.get("ci95") is None:
        return f"{v['mean']:.{digits}f} (n={v['n']})"
    flag = "*" if v.get("rel") is not None and v["rel"] > REL_WARN else ""
    return f"{v['mean']:.{digits}f} ± {v['ci95']:.{digits}f}{flag}"


def markdown(rows: list[RunResult]) -> str:
    by_model: dict[str, list[RunResult]] = collections.defaultdict(list)
    for r in rows:
        by_model[r.model or r.harness].append(r)
    sums = {m: summarise(rs) for m, rs in sorted(by_model.items())}
    tasks = sorted({r.task for r in rows})
    out = [
        "### Pass rate (Wilson 95%)",
        "",
        "| model | runs | errors | all | code | prose | mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, s in sums.items():
        a = s.get("all", {})
        out.append(
            f"| {m} | {a.get('runs', 0)} | {a.get('errors_excluded', 0)} | {_pct(a)} | "
            f"{_pct(s.get('kind:code'))} | {_pct(s.get('kind:prose'))} | {_mean(a.get('score'), 2)} |"
        )
    out += ["", "### Pass rate by task", "", "| model | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
    for m, s in sums.items():
        out.append(f"| {m} | " + " | ".join(_pct(s.get(f"task:{t}")) for t in tasks) + " |")
    out += [
        "",
        "### Throughput, mean over runs (t 95%)",
        "",
        "| model | provider | TTFT s | decode tok/s | prose tok/s | code tok/s | reasoning tok/s | flushes/turn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, s in sums.items():
        a = s.get("all", {})
        providers = ", ".join(p for p, _ in collections.Counter(a.get("providers", {})).most_common(2)) or "–"
        flushes = mean_ci(
            [r.throughput.get("flushes_per_turn_median") for r in by_model[m] if not r.stop.startswith("error")]
        )
        out.append(
            f"| {m} | {providers} | {_mean(a.get('ttft_s'), 2)} | {_mean(a.get('decode_tok_s'))} | "
            f"{_mean(a.get('prose_tok_s'))} | {_mean(a.get('code_tok_s'))} | "
            f"{_mean(a.get('reasoning_tok_s'))} | {_mean(flushes, 0)} |"
        )
    out += [
        "",
        f"`*`: the 95% interval is wider than ±{REL_WARN:.0%} of the mean; more seeds would narrow it. Runs that "
        + "ended on an API error are excluded and counted under errors.",
    ]
    return "\n".join(out) + "\n"
