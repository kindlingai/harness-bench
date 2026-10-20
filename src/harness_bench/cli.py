"""harness-bench: list, run and summarise."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from . import registry
from .runner import result_dict, run_one, summarise


def _seeds(text: str) -> list[int]:
    """'20' -> 0..19, '5-9' -> 5..9, '1,4,9' -> those."""
    if "," in text:
        return [int(x) for x in text.split(",")]
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return list(range(int(text)))


def _options(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"-o expects key=value, got {p!r}")
        out[k] = v
    return out


def cmd_list(_args) -> None:
    print("tasks:")
    for name, t in registry.tasks().items():
        print(f"  {name:22s} {t.kind:6s} {t.description}")
    print("harnesses:")
    for name, h in registry.harnesses().items():
        opts = ", ".join(h.options) or "no options"
        print(f"  {name:22s} {opts}")


def cmd_run(args) -> None:
    tasks = registry.tasks()
    names = list(tasks) if args.tasks == "all" else args.tasks.split(",")
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}; see `harness-bench list`")
    harness = registry.harnesses()[args.harness](**_options(args.option))
    seeds = _seeds(args.seeds)
    keep = Path(args.keep) if args.keep else None
    results = []
    with open(args.out, "a") if args.out else contextlib.nullcontext() as out:
        _run_all(names, tasks, seeds, harness, args, keep, results, out)
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))


def _run_all(names, tasks, seeds, harness, args, keep, results, out) -> None:
    for name in names:
        for seed in seeds:
            r = run_one(tasks[name], seed, harness, max_turns=args.max_turns, keep=keep)
            results.append(r)
            t = r.throughput
            print(
                f"{name:20s} seed {seed:3d}  {'PASS' if r.passed else 'fail'}  score {r.score:.2f}  "
                f"turns {t['turns']:2d}  decode {t['decode_tok_s'] or 0:7.1f} tok/s  stop {r.stop}",
                file=sys.stderr,
                flush=True,
            )
            if out:
                out.write(json.dumps(result_dict(r)) + "\n")
                out.flush()


def cmd_summarise(args) -> None:
    from .runner import RunResult

    rows = []
    for f in args.files:
        rows += [RunResult(**json.loads(line)) for line in Path(f).read_text().splitlines() if line.strip()]
    print(json.dumps(summarise(rows), indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="harness-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list tasks and harnesses").set_defaults(fn=cmd_list)
    r = sub.add_parser("run", help="run tasks x seeds through a harness")
    r.add_argument("--harness", default="openai-chat")
    r.add_argument("-o", "--option", action="append", default=[], help="harness option key=value (repeatable)")
    r.add_argument("--tasks", default="all", help="comma list or 'all'")
    r.add_argument("--seeds", default="20", help="N (0..N-1), A-B, or a comma list; default 20")
    r.add_argument("--max-turns", type=int, default=30)
    r.add_argument("--out", help="append one JSON line per run")
    r.add_argument("--summary", help="write the summary JSON here too")
    r.add_argument("--keep", help="copy each final workspace under this directory")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summarise", help="summarise JSONL files written by run --out")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_summarise)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
