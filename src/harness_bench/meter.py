"""Throughput of a run, split by what the model was writing: prose, code, reasoning or tool arguments.

A harness opens one turn per model request and reports each streamed piece as it arrives. Pieces that arrive
together form a flush; each flush is charged the time since the previous flush, split across its characters by
class. A turn that mixes reasoning, code and prose therefore splits correctly, and a provider that holds a tool call
back and sends it in one burst is still charged the time it spent generating it. The first flush of a turn closes
its prefill (time to first token) and is not charged decode time. Characters convert to tokens with
each turn's own ratio from the server's usage counts.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

CLASSES = ("prose", "code", "reasoning", "tool")
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
# Pieces that arrive within this many seconds of the previous one were sent together (one flush). Servers and
# routers batch output, so time is charged per flush and split across its characters.
FLUSH_S = 0.005
FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|$)", re.DOTALL)


def file_class(path: str) -> str:
    """'code' for a source or config file, 'prose' for anything else (Markdown, text, ...)."""
    return "code" if PurePosixPath(path).suffix.lower() in CODE_SUFFIXES else "prose"


def fenced_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of text inside fenced code blocks."""
    return [m.span(1) for m in FENCE.finditer(text)]


def split_content(text: str) -> dict[str, int]:
    """Characters of assistant text inside fenced code blocks (code) and outside them (prose)."""
    code = sum(b - a for a, b in fenced_spans(text))
    return {"code": code, "prose": len(text) - code}


def _empty() -> dict[str, float]:
    return dict.fromkeys(CLASSES, 0.0)


@dataclass
class Piece:
    at: float
    kind: str  # "content", "reasoning" or "tool"
    n: int
    offset: int = 0  # content: position in the turn's text
    index: int = 0  # tool: the call's index


@dataclass
class Turn:
    t_start: float
    t_first: float | None = None
    t_end: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False  # True when the server sent no usage and tokens are chars / 4
    provider: str = ""  # the upstream that served it, when a router reports one
    flushes: int = 0  # separate arrivals; few flushes means coarse per-class timing
    chars: dict[str, float] = field(default_factory=_empty)  # every streamed character, by class
    timed_chars: dict[str, float] = field(default_factory=_empty)  # characters after the first piece
    seconds: dict[str, float] = field(default_factory=_empty)  # decode time charged to each class
    pieces: list[Piece] = field(default_factory=list, repr=False)
    text: list[str] = field(default_factory=list, repr=False)
    calls: dict[int, str] = field(default_factory=dict, repr=False)  # tool call index -> class

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def decode_s(self) -> float | None:
        if self.t_first is None or self.t_end is None:
            return None
        return self.t_end - self.t_first

    @property
    def tokens_per_char(self) -> float:
        total = sum(self.chars.values())
        return self.completion_tokens / total if total else 0.0


class Meter:
    """Per request: `t = meter.begin()`; for each streamed delta `meter.content(t, s)`, `meter.reasoning(t, s)` or
    `meter.tool_args(t, index, s)`; for each completed call `meter.tool_call(t, index, name, arguments)`; then
    `meter.end(t, usage)` with the server's usage dict (or None to estimate tokens from characters)."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.turns: list[Turn] = []

    def begin(self) -> Turn:
        t = Turn(t_start=self.clock())
        self.turns.append(t)
        return t

    def _piece(self, t: Turn, piece: Piece) -> None:
        if t.t_first is None and piece.n:
            t.t_first = piece.at
        t.pieces.append(piece)

    def content(self, t: Turn, s: str) -> None:
        if s:
            self._piece(t, Piece(self.clock(), "content", len(s), offset=sum(map(len, t.text))))
            t.text.append(s)

    def reasoning(self, t: Turn, s: str) -> None:
        if s:
            self._piece(t, Piece(self.clock(), "reasoning", len(s)))

    def tool_args(self, t: Turn, index: int, s: str) -> None:
        """A streamed piece of a tool call; an empty `s` still marks the time the call started."""
        self._piece(t, Piece(self.clock(), "tool", len(s), index=index))

    def tool_call(self, t: Turn, index: int, name: str, arguments: str) -> None:
        """Classify a completed call: a file write is code or prose by the file's extension, anything else tool."""
        try:
            args = json.loads(arguments) if arguments else {}
        except ValueError:
            args = None
        is_write = isinstance(args, dict) and isinstance(args.get("content"), str) and isinstance(args.get("path"), str)
        t.calls[index] = file_class(args["path"]) if is_write else "tool"

    def end(self, t: Turn, usage: dict | None, provider: str = "") -> None:
        t.t_end = self.clock()
        t.provider = provider
        code = fenced_spans("".join(t.text))
        flushes: list[list[tuple[float, dict[str, int]]]] = []
        for p in t.pieces:
            if not p.n:  # a call-start marker carries no text
                continue
            split = self._classes(t, p, code)
            for cls, n in split.items():
                t.chars[cls] += n
            if flushes and p.at - flushes[-1][-1][0] <= FLUSH_S:
                flushes[-1].append((p.at, split))
            else:
                flushes.append([(p.at, split)])
        t.flushes = len(flushes)
        for before, flush in itertools.pairwise(flushes):
            dt = flush[0][0] - before[-1][0]
            n = sum(sum(split.values()) for _, split in flush)
            for _, split in flush:
                for cls, k in split.items():
                    t.timed_chars[cls] += k
                    t.seconds[cls] += dt * k / n
        if usage:
            t.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            t.completion_tokens = int(usage.get("completion_tokens") or 0)
            t.cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        else:
            t.completion_tokens = max(int(sum(t.chars.values()) // 4), 1)
            t.estimated = True
        t.pieces, t.text = [], []

    @staticmethod
    def _classes(t: Turn, p: Piece, code: list[tuple[int, int]]) -> dict[str, int]:
        if p.kind == "reasoning":
            return {"reasoning": p.n}
        if p.kind == "tool":
            return {t.calls.get(p.index, "tool"): p.n}
        a, b = p.offset, p.offset + p.n
        in_code = sum(max(0, min(b, y) - max(a, x)) for x, y in code)
        return {"code": in_code, "prose": p.n - in_code}

    def summary(self) -> dict:
        done = [t for t in self.turns if t.decode_s is not None]
        out: dict = {
            "turns": len(self.turns),
            "completion_tokens": sum(t.completion_tokens for t in done),
            "prompt_tokens": sum(t.prompt_tokens for t in done),
            "tokens_estimated": any(t.estimated for t in done),
            "ttft_s_median": _median([t.ttft for t in done if t.ttft is not None]),
            "flushes_per_turn_median": _median([t.flushes for t in done]),
        }
        fresh = [(t.prompt_tokens - t.cached_tokens, t.ttft) for t in done if t.ttft and t.prompt_tokens]
        out["prefill_tok_s"] = _rate(sum(n for n, _ in fresh), sum(s for _, s in fresh))
        out["decode_tok_s"] = _rate(sum(max(t.completion_tokens - 1, 0) for t in done), sum(t.decode_s for t in done))
        out["by_class"] = {
            cls: {
                "tokens": round(sum(t.chars[cls] * t.tokens_per_char for t in done)),
                "seconds": round(sum(t.seconds[cls] for t in done), 3),
                "decode_tok_s": _rate(
                    sum(t.timed_chars[cls] * t.tokens_per_char for t in done), sum(t.seconds[cls] for t in done)
                ),
            }
            for cls in CLASSES
        }
        return out

    def turns_json(self) -> list[dict]:
        return [
            {k: v for k, v in asdict(t).items() if k not in ("pieces", "text", "calls")}
            | {"ttft": t.ttft, "decode_s": t.decode_s}
            for t in self.turns
        ]


def _rate(n: float, s: float) -> float | None:
    return round(n / s, 2) if s > 0 and n > 0 else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None
