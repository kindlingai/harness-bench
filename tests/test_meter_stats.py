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
    assert split_content("intro\n```py\nx = 1\n```\noutro") == {
        "code": 6,
        "prose": len("intro\n```py\nx = 1\n```\noutro") - 6,
    }


def test_classes_and_rates():
    clock = Clock()
    m = Meter(clock=clock)
    t = m.begin()
    clock.t = 0.5
    m.first_token(t)
    m.tool_call(t, "write_file", json.dumps({"path": "a.py", "content": "x" * 900}))
    clock.t = 2.5
    m.end(t, {"prompt_tokens": 1000, "completion_tokens": 201})
    s = m.summary()
    assert s["prefill_tok_s"] == 2000.0
    assert s["decode_tok_s"] == 100.0
    assert s["by_class"]["code"]["pure_turns"] == 1
    assert s["by_class"]["code"]["decode_tok_s"] == 100.0
    assert s["by_class"]["prose"]["pure_turns"] == 0


def test_markdown_write_is_prose():
    m = Meter(clock=Clock())
    t = m.begin()
    m.tool_call(t, "write_file", json.dumps({"path": "NOTES.md", "content": "words " * 50}))
    assert t.share("prose") > 0.8


def test_wilson_bounds():
    p, lo, hi = wilson(0, 20)
    assert p == 0 and lo == 0 and 0 < hi < 0.2
    p, lo, hi = wilson(20, 20)
    assert p == 1 and hi == 1 and 0.8 < lo < 1


def test_mean_ci():
    r = mean_ci([10, 12, 11, 13, 9])
    assert (
        r["mean"] == 11 and r["n"] == 5 and math.isclose(r["ci95"], 2.776 * math.sqrt(2.5) / math.sqrt(5), rel_tol=1e-3)
    )
    assert mean_ci([None])["n"] == 0
