"""Run tasks x seeds through one harness and summarise them with intervals."""

from __future__ import annotations

import collections
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .harnesses.base import Harness
from .meter import CLASSES, Meter
from .stats import mean_ci, wilson
from .task import Task

# A mean whose 95% half width is above this fraction of the mean is flagged as not yet trustworthy.
REL_WARN = 0.10


@dataclass
class RunResult:
    task: str
    kind: str
    seed: int
    harness: str
    model: str
    passed: bool
    score: float
    checks: dict
    notes: str
    stop: str
    wall_s: float
    tool_calls: int
    throughput: dict
    turns: list[dict]


def run_one(
    task_cls: type[Task], seed: int, harness: Harness, *, max_turns: int, keep: Path | None = None
) -> RunResult:
    task = task_cls(seed)
    with tempfile.TemporaryDirectory(prefix=f"hb-{task.name}-{seed}-") as tmp:
        ws = Path(tmp) / "workspace"
        setup = task.setup(ws)
        meter = Meter()
        t0 = time.perf_counter()
        tr = harness.run(task, setup, meter, max_turns=max_turns)
        wall = time.perf_counter() - t0
        v = task.verify(ws, setup.tools)
        if keep:
            import shutil

            shutil.copytree(ws, keep / f"{task.name}-{seed}", dirs_exist_ok=True)
    return RunResult(
        task=task.name,
        kind=task.kind,
        seed=seed,
        harness=harness.name,
        model=getattr(harness, "model", ""),
        passed=v.passed,
        score=round(v.score, 4),
        checks=v.checks,
        notes=v.notes,
        stop=tr.stop,
        wall_s=round(wall, 3),
        tool_calls=len(setup.tools.calls),
        throughput=meter.summary(),
        turns=meter.turns_json(),
    )


def summarise(results: list[RunResult]) -> dict:
    """Per task and per kind (code / prose): pass rate (Wilson 95%), score and throughput means (t 95%)."""
    groups: dict[str, list[RunResult]] = collections.defaultdict(list)
    for r in results:
        groups[f"task:{r.task}"].append(r)
        groups[f"kind:{r.kind}"].append(r)
        groups["all"].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        p, lo, hi = wilson(sum(r.passed for r in rs), len(rs))
        g = {
            "runs": len(rs),
            "pass_rate": round(p, 3),
            "pass_ci95": [round(lo, 3), round(hi, 3)],
            "score": mean_ci([r.score for r in rs]),
            "prefill_tok_s": mean_ci([r.throughput.get("prefill_tok_s") for r in rs]),
            "decode_tok_s": mean_ci([r.throughput.get("decode_tok_s") for r in rs]),
            "ttft_s": mean_ci([r.throughput.get("ttft_s_median") for r in rs]),
            "turns": mean_ci([r.throughput.get("turns") for r in rs]),
        }
        for cls in CLASSES:
            g[f"{cls}_tok_s"] = mean_ci(
                [r.throughput["by_class"][cls]["decode_tok_s"] for r in rs if r.throughput.get("by_class")]
            )
        g["warnings"] = [
            f"{k}: ±{v['rel']:.0%} of the mean with n={v['n']}; add seeds"
            for k, v in g.items()
            if isinstance(v, dict) and v.get("rel") is not None and v["rel"] > REL_WARN
        ]
        out[key] = g
    return out


def result_dict(r: RunResult) -> dict:
    return asdict(r)
