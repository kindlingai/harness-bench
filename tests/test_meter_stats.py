import json
import math

from harness_bench.meter import Meter, split_content
from harness_bench.stats import mean_ci, wilson


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_split_content():
    text = "intro\n```py\nx = 1\n```\noutro"
    assert split_content(text) == {"code": 6, "prose": len(text) - 6}


def test_mixed_turn_splits_time_by_piece():
    """Reasoning for 1 s, then a code write for 2 s, in one request."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 0.5
    m.reasoning(t, "r" * 10)  # first piece: closes prefill, charged no decode time
    for _ in range(10):
        clock.t += 0.1
        m.reasoning(t, "r" * 10)
    args = json.dumps({"path": "a.py", "content": "x" * 380})
    m.tool_args(t, 0, "")
    for i in range(0, len(args), 20):
        clock.t += 2.0 / (len(args) / 20)
        m.tool_args(t, 0, args[i : i + 20])
    m.tool_call(t, 0, "write_file", args)
    m.end(t, {"prompt_tokens": 1000, "completion_tokens": (110 + len(args)) // 2})
    s = m.summary()
    assert math.isclose(s["by_class"]["reasoning"]["seconds"], 1.0, rel_tol=0.05)
    assert math.isclose(s["by_class"]["code"]["seconds"], 2.0, rel_tol=0.05)
    assert s["by_class"]["prose"]["decode_tok_s"] is None
    assert s["prefill_tok_s"] == 2000.0


def test_buffered_tool_call_is_charged_its_generation_time():
    """A provider that sends the whole call after 3 s of silence: 3 s of tool time, not ~0."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 0.2
    m.content(t, "Let me look.")
    clock.t = 3.2
    m.tool_args(t, 0, json.dumps({"path": "logs"}))
    m.tool_call(t, 0, "list_files", json.dumps({"path": "logs"}))
    m.end(t, {"prompt_tokens": 100, "completion_tokens": 10})
    assert math.isclose(t.seconds["tool"], 3.0)
    assert m.summary()["decode_tok_s"] is None  # two arrivals: too coarse to time, so not counted


def test_call_start_marker_does_not_swallow_the_gap():
    """Reasoning, 2 s of silence, an empty call-start piece, then the arguments in a burst."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 0.1
    m.reasoning(t, "thinking")
    clock.t = 2.1
    m.tool_args(t, 0, "")
    args = json.dumps({"path": "a.py", "content": "x" * 100})
    for i in range(0, len(args), 10):
        clock.t += 1e-6
        m.tool_args(t, 0, args[i : i + 10])
    m.tool_call(t, 0, "write_file", args)
    m.end(t, {"prompt_tokens": 10, "completion_tokens": 40})
    assert math.isclose(t.seconds["code"], 2.0, rel_tol=0.01)


def test_one_flush_splits_its_time_by_characters():
    """The end of the reasoning and a whole tool call arrive together after a 2 s gap."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 1.0
    m.reasoning(t, "r" * 50)
    clock.t = 3.0
    m.reasoning(t, "r" * 100)
    args = json.dumps({"path": "a.py", "content": "x" * 280})
    m.tool_args(t, 0, args)
    m.tool_call(t, 0, "write_file", args)
    m.end(t, {"prompt_tokens": 10, "completion_tokens": 100})
    assert t.flushes == 2
    assert math.isclose(t.seconds["reasoning"], 2.0 * 100 / (100 + len(args)))
    assert math.isclose(t.seconds["reasoning"] + t.seconds["code"], 2.0)


def test_a_burst_is_not_streamed():
    """Three arrivals, but the last carries nearly everything."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    for at, text in ((1.0, "a"), (1.5, "b"), (5.0, "c" * 500)):
        clock.t = at
        m.content(t, text)
    m.end(t, {"prompt_tokens": 10, "completion_tokens": 120})
    assert t.flushes == 3 and t.max_flush_share > 0.9
    assert m.summary()["streamed_turns"] == 0


def test_a_class_sent_in_one_burst_has_no_rate():
    """Reasoning streams in many pieces; the code write then lands in one flush."""
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    for i in range(10):
        clock.t = 1.0 + 0.1 * i
        m.reasoning(t, "r" * 80)
    clock.t = 2.5
    args = json.dumps({"path": "a.py", "content": "x" * 300})
    m.tool_args(t, 0, args)
    m.tool_call(t, 0, "write_file", args)
    m.end(t, {"prompt_tokens": 10, "completion_tokens": 150})
    by = m.summary()["by_class"]
    assert by["reasoning"]["decode_tok_s"] and by["code"]["decode_tok_s"] is None and by["code"]["tokens"] > 0


def test_output_rate_needs_no_streaming():
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 4.0
    m.content(t, "all at once")
    m.end(t, {"prompt_tokens": 10, "completion_tokens": 200})
    s = m.summary()
    assert s["output_tok_s"] == 50.0 and s["streamed_turns"] == 0 and s["decode_tok_s"] is None


def test_markdown_write_is_prose_and_fences_are_code():
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    for piece in ["Here is the fix:\n", "```py\n", "x = 1\n", "```\n", "Done."]:
        clock.t += 0.1
        m.content(t, piece)
    args = json.dumps({"path": "NOTES.md", "content": "words " * 20})
    clock.t += 0.1
    m.tool_args(t, 0, args)
    m.tool_call(t, 0, "write_file", args)
    m.end(t, None)
    assert t.chars["code"] == len("x = 1\n")
    assert t.chars["prose"] > len("words " * 20)
    assert t.estimated


def test_wilson_bounds():
    p, lo, hi = wilson(0, 20)
    assert p == 0 and lo == 0 and 0 < hi < 0.2
    p, lo, hi = wilson(20, 20)
    assert p == 1 and hi == 1 and 0.8 < lo < 1


def test_mean_ci():
    r = mean_ci([10, 12, 11, 13, 9])
    assert r["mean"] == 11 and r["n"] == 5
    assert math.isclose(r["ci95"], 2.776 * math.sqrt(2.5) / math.sqrt(5), rel_tol=1e-3)
    assert mean_ci([None])["n"] == 0
