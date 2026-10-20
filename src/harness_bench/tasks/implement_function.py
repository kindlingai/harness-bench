"""Code: implement a function from its docstring. A few visible tests ship with the stub; the verifier also runs
seeded hidden cases, so the score measures correctness beyond the examples."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from ..task import Task, Verdict, verdict
from ..tools import ToolBox, python_unittest


def _duration(s: str) -> int:
    total, num = 0, ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            total += int(num) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[ch]
            num = ""
    return total


def _gen_duration(rng: Random) -> str:
    parts = [(u, rng.randint(1, 59)) for u in "dhms" if rng.random() < 0.6] or [("m", rng.randint(1, 59))]
    return "".join(f"{n}{u}" for u, n in parts)


ROMAN = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _to_roman(n: int) -> str:
    out = ""
    for v, s in ROMAN:
        while n >= v:
            out, n = out + s, n - v
    return out


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        total += -vals[a] if b != " " and vals[a] < vals[b] else vals[a]
    return total


def _rle(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def _gen_rle(rng: Random) -> str:
    return "".join(rng.choice("abcxyz") * rng.randint(1, 5) for _ in range(rng.randint(1, 6)))


@dataclass(frozen=True)
class Spec:
    name: str
    signature: str
    doc: str
    solution: str
    oracle: Callable
    gen: Callable[[Random], object]


SPECS = [
    Spec(
        "parse_duration",
        "def parse_duration(text: str) -> int:",
        'Return the number of seconds in a duration such as "1h30m" or "2d4h10s".\n\n'
        "    Units are d (days), h, m and s, each at most once, largest first. Every unit has a number.",
        "def parse_duration(text: str) -> int:\n    total, num = 0, ''\n    for ch in text:\n"
        "        if ch.isdigit():\n            num += ch\n        else:\n"
        "            total += int(num) * {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[ch]\n            num = ''\n"
        "    return total\n",
        _duration,
        _gen_duration,
    ),
    Spec(
        "roman_to_int",
        "def roman_to_int(numeral: str) -> int:",
        "Convert a Roman numeral (I, V, X, L, C, D, M; subtractive forms such as IV and CM) to an integer.",
        "def roman_to_int(numeral: str) -> int:\n"
        "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n"
        "    for a, b in zip(numeral, numeral[1:] + ' '):\n"
        "        total += -vals[a] if b != ' ' and vals[a] < vals[b] else vals[a]\n    return total\n",
        _roman_to_int,
        lambda rng: _to_roman(rng.randint(1, 3999)),
    ),
    Spec(
        "run_length_encode",
        "def run_length_encode(text: str) -> str:",
        'Encode runs of the same character as the character and the run length: "aaabcc" -> "a3b1c2".',
        "def run_length_encode(text: str) -> str:\n    out, i = [], 0\n    while i < len(text):\n        j = i\n"
        "        while j < len(text) and text[j] == text[i]:\n            j += 1\n"
        "        out.append(f'{text[i]}{j - i}')\n        i = j\n    return ''.join(out)\n",
        _rle,
        _gen_rle,
    ),
]

HIDDEN = 40


class ImplementFunction(Task):
    name = "implement-function"
    kind = "code"
    description = "Implement a seeded function from its docstring; scored on visible plus seeded hidden cases."

    def test_command(self) -> list[str]:
        return python_unittest()

    def _plan(self):
        rng = self.rng
        spec = rng.choice(SPECS)
        visible = [spec.gen(rng) for _ in range(3)]
        hidden = [spec.gen(rng) for _ in range(HIDDEN)]
        return spec, visible, hidden

    def build(self, workspace: Path) -> str:
        spec, visible, _ = self._plan()
        (workspace / "solution.py").write_text(
            f'{spec.signature}\n    """{spec.doc}\n    """\n    raise NotImplementedError\n'
        )
        cases = "".join(f"        self.assertEqual({spec.name}({x!r}), {spec.oracle(x)!r})\n" for x in visible)
        (workspace / "test_solution.py").write_text(
            f"import unittest\n\nfrom solution import {spec.name}\n\n\nclass SolutionTest(unittest.TestCase):\n"
            f"    def test_examples(self):\n{cases}"
        )
        return (
            f"Implement `{spec.name}` in solution.py as its docstring describes. The tests in "
            f"test_solution.py show a few examples; the implementation must be correct in general, not only "
            f"for those. Run the tests, then call finish."
        )

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        spec, visible, hidden = self._plan()
        cases = [(x, spec.oracle(x)) for x in visible + hidden]
        script = (
            f"import json, sys\nsys.path.insert(0, {str(workspace)!r})\nfrom solution import {spec.name} as f\n"
            f"cases = json.loads(sys.stdin.read())\nok = 0\nfor x, want in cases:\n    try:\n"
            f"        ok += f(x) == want\n    except Exception:\n        pass\nprint(ok)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "check.py").write_text(script)
            try:
                proc = subprocess.run(
                    [sys.executable, str(Path(tmp, "check.py"))],
                    input=json.dumps(cases),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    cwd=tmp,
                )
                ok = int(proc.stdout.strip() or 0)
            except (subprocess.TimeoutExpired, ValueError):
                ok = 0
        v = verdict(
            {
                "all_cases": ok == len(cases),
                "visible_cases": ok >= len(visible) and ok > 0,
                "ran_tests": any(c.name == "run_tests" for c in tools.calls),
            },
            notes=f"{ok}/{len(cases)} cases",
        )
        v.score = ok / len(cases)
        return v

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        spec, _, _ = self._plan()
        tools.call("read_file", {"path": "solution.py"})
        tools.call("write_file", {"path": "solution.py", "content": spec.solution})
        tools.call("run_tests", {})
        tools.call("finish", {"message": f"implemented {spec.name}"})
