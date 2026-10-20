"""Throughput of a run, split by what the model was writing: prose, code, reasoning or tool arguments.

A harness opens one turn per model request and reports what streamed back. Tokens come from the server's usage
counts and are split across classes by character share; a class's decode rate is measured only over turns where it
makes up at least PURE of the output, so a turn that mixes code and prose does not blur either number.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

CLASSES = ("prose", "code", "reasoning", "tool")
PURE = 0.8
CODE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".go",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".java",
    ".kt",
    ".rb",
    ".sh",
    ".zig",
    ".swift",
    ".cs",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".css",
    ".html",
    ".cu",
    ".lua",
}
FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|$)", re.DOTALL)


def file_class(path: str) -> str:
    """'code' for a source or config file, 'prose' for anything else (Markdown, text, ...)."""
    return "code" if PurePosixPath(path).suffix.lower() in CODE_SUFFIXES else "prose"


def split_content(text: str) -> dict[str, int]:
    """Characters of assistant text inside fenced code blocks (code) and outside them (prose)."""
    code = sum(len(m.group(1)) for m in FENCE.finditer(text))
    return {"code": code, "prose": max(len(text) - code, 0)}


@dataclass
class Turn:
    t_start: float
    t_first: float | None = None
    t_end: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False  # True when the server sent no usage and tokens are chars / 4
    chars: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def decode_s(self) -> float | None:
        if self.t_first is None or self.t_end is None:
            return None
        return self.t_end - self.t_first

    def share(self, cls: str) -> float:
        total = sum(self.chars.values())
        return self.chars[cls] / total if total else 0.0


class Meter:
    """Collects turns. Use: `t = meter.begin()`, `meter.first_token(t)` on the first streamed delta,
    `meter.text(t, cls, s)` / `meter.content(t, s)` / `meter.tool_call(t, name, args)` as output arrives, then
    `meter.end(t, usage)`."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.turns: list[Turn] = []

    def begin(self) -> Turn:
        t = Turn(t_start=self.clock())
        self.turns.append(t)
        return t

    def first_token(self, t: Turn) -> None:
        if t.t_first is None:
            t.t_first = self.clock()

    def text(self, t: Turn, cls: str, s: str) -> None:
        t.chars[cls] += len(s)

    def content(self, t: Turn, s: str) -> None:
        """Assistant text: fenced blocks count as code, the rest as prose."""
        for cls, n in split_content(s).items():
            t.chars[cls] += n

    def tool_call(self, t: Turn, name: str, arguments: str) -> None:
        """A complete tool call. A file write counts its content as code or prose by extension; the rest of the
        arguments count as tool."""
        try:
            args = json.loads(arguments) if arguments else {}
        except ValueError:
            args = None
        if isinstance(args, dict) and isinstance(args.get("content"), str) and isinstance(args.get("path"), str):
            body = args["content"]
            t.chars[file_class(args["path"])] += len(body)
            t.chars["tool"] += max(len(arguments) - len(body), 0)
        else:
            t.chars["tool"] += len(arguments)

    def end(self, t: Turn, usage: dict | None) -> None:
        t.t_end = self.clock()
        if usage:
            t.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            t.completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            t.cached_tokens = int(details.get("cached_tokens") or 0)
        else:
            t.completion_tokens = max(sum(t.chars.values()) // 4, 1)
            t.estimated = True

    def summary(self) -> dict:
        done = [t for t in self.turns if t.decode_s is not None]
        out: dict = {
            "turns": len(self.turns),
            "completion_tokens": sum(t.completion_tokens for t in done),
            "prompt_tokens": sum(t.prompt_tokens for t in done),
            "tokens_estimated": any(t.estimated for t in done),
        }
        ttfts = [t.ttft for t in done if t.ttft is not None]
        out["ttft_s_median"] = _median(ttfts)
        fresh = [((t.prompt_tokens - t.cached_tokens), t.ttft) for t in done if t.ttft and t.prompt_tokens]
        out["prefill_tok_s"] = _rate(sum(n for n, _ in fresh), sum(s for _, s in fresh))
        out["decode_tok_s"] = _rate(sum(max(t.completion_tokens - 1, 0) for t in done), sum(t.decode_s for t in done))
        per = {}
        for cls in CLASSES:
            tokens = sum(t.completion_tokens * t.share(cls) for t in done)
            pure = [t for t in done if t.share(cls) >= PURE and t.decode_s]
            per[cls] = {
                "tokens": round(tokens),
                "pure_turns": len(pure),
                "decode_tok_s": _rate(
                    sum(max(t.completion_tokens - 1, 0) for t in pure), sum(t.decode_s for t in pure)
                ),
            }
        out["by_class"] = per
        return out

    def turns_json(self) -> list[dict]:
        return [asdict(t) | {"ttft": t.ttft, "decode_s": t.decode_s} for t in self.turns]


def _rate(n: float, s: float) -> float | None:
    return round(n / s, 2) if s > 0 and n > 0 else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None
