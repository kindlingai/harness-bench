"""Code: a small utility module has one seeded bug that its test suite catches. Fix it without touching the tests."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..task import Task, Verdict, verdict
from ..tools import ToolBox, python_unittest


@dataclass(frozen=True)
class Func:
    name: str
    correct: str
    bugs: tuple[str, ...]  # drop-in replacements, each failing `tests`
    tests: str


FUNCS = [
    Func(
        "clamp",
        "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
        (
            "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
            "def clamp(x, lo, hi):\n    return max(lo, min(x, lo))\n",
        ),
        "        self.assertEqual(clamp(5, 0, 10), 5)\n        self.assertEqual(clamp(-3, 0, 10), 0)\n"
        "        self.assertEqual(clamp(42, 0, 10), 10)\n",
    ),
    Func(
        "mean",
        "def mean(xs):\n    return sum(xs) / len(xs)\n",
        ("def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n", "def mean(xs):\n    return sum(xs) // len(xs)\n"),
        "        self.assertEqual(mean([1, 2, 3, 4]), 2.5)\n        self.assertEqual(mean([7]), 7)\n",
    ),
    Func(
        "median",
        "def median(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n"
        "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n",
        (
            "def median(xs):\n    m = len(xs) // 2\n    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2\n",
            "def median(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n    return s[m]\n",
        ),
        "        self.assertEqual(median([3, 1, 2]), 2)\n        self.assertEqual(median([4, 1, 3, 2]), 2.5)\n",
    ),
    Func(
        "slugify",
        "import re\n\n\ndef slugify(s):\n    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')\n",
        (
            "import re\n\n\ndef slugify(s):\n    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')\n",
            "import re\n\n\ndef slugify(s):\n    return re.sub(r'[^a-z0-9]+', '-', s.lower())\n",
        ),
        "        self.assertEqual(slugify('Hello, World!'), 'hello-world')\n"
        "        self.assertEqual(slugify('  a  b '), 'a-b')\n",
    ),
    Func(
        "chunk",
        "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
        (
            "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - 1, n)]\n",
            "def chunk(xs, n):\n    return [xs[i:i + n - 1] for i in range(0, len(xs), n)]\n",
        ),
        "        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])\n"
        "        self.assertEqual(chunk([1, 2, 3], 1), [[1], [2], [3]])\n",
    ),
    Func(
        "count_words",
        "def count_words(s):\n    return len(s.split())\n",
        ("def count_words(s):\n    return len(s.split(' '))\n", "def count_words(s):\n    return s.count(' ') + 1\n"),
        "        self.assertEqual(count_words('one two  three'), 3)\n        self.assertEqual(count_words(''), 0)\n",
    ),
    Func(
        "is_palindrome",
        "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
        (
            "def is_palindrome(s):\n    return s == s[::-1]\n",
            "def is_palindrome(s):\n    t = [c for c in s if c.isalnum()]\n    return t == t[::-1]\n",
        ),
        "        self.assertTrue(is_palindrome('A man, a plan, a canal: Panama'))\n"
        "        self.assertFalse(is_palindrome('harness'))\n",
    ),
]

MODULE = "textkit.py"
TESTS = "test_textkit.py"


def _module(funcs: list[Func], bodies: dict[str, str]) -> str:
    imports = sorted({line for f in funcs for line in bodies[f.name].splitlines() if line.startswith("import ")})
    parts = ['"""Small text and number helpers."""\n'] + ["\n".join(imports) + "\n" if imports else ""]
    for f in funcs:
        body = "\n".join(line for line in bodies[f.name].splitlines() if not line.startswith("import ")).strip()
        parts.append("\n\n" + body + "\n")
    return "".join(parts)


def _tests(funcs: list[Func]) -> str:
    names = ", ".join(f.name for f in funcs)
    cases = "".join(f"    def test_{f.name}(self):\n{f.tests}\n" for f in funcs)
    return f"import unittest\n\nfrom textkit import {names}\n\n\nclass TextkitTest(unittest.TestCase):\n{cases}"


class FixBug(Task):
    name = "fix-bug"
    kind = "code"
    description = "Find and fix one seeded bug in a utility module; its unittest suite must pass, tests unedited."

    def test_command(self) -> list[str]:
        return python_unittest()

    def _plan(self) -> tuple[list[Func], Func, str]:
        rng = self.rng
        funcs = rng.sample(FUNCS, 5)
        target = rng.choice(funcs)
        return funcs, target, rng.choice(target.bugs)

    def build(self, workspace: Path) -> str:
        funcs, target, bug = self._plan()
        bodies = {f.name: f.correct for f in funcs} | {target.name: bug}
        (workspace / MODULE).write_text(_module(funcs, bodies))
        (workspace / TESTS).write_text(_tests(funcs))
        (workspace / ".tests.sha256").write_text(hashlib.sha256((workspace / TESTS).read_bytes()).hexdigest())
        return (
            f"The unit tests in this project fail. Find the bug in {MODULE} and fix it. Do not change "
            f"{TESTS}. Run the tests to confirm they pass, then call finish."
        )

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        tests = workspace / TESTS
        unchanged = (
            tests.exists()
            and hashlib.sha256(tests.read_bytes()).hexdigest() == (workspace / ".tests.sha256").read_text()
        )
        try:
            proc = subprocess.run(
                python_unittest(), cwd=workspace, capture_output=True, text=True, timeout=60, check=False
            )
            passing = proc.returncode == 0
        except subprocess.TimeoutExpired:
            passing = False
        ran = any(c.name == "run_tests" for c in tools.calls)
        return verdict({"tests_pass": passing, "tests_unchanged": unchanged, "ran_tests": ran})

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        _, target, bug = self._plan()
        src = tools.call("read_file", {"path": MODULE})
        body = lambda s: "\n".join(x for x in s.splitlines() if not x.startswith("import ")).strip()
        tools.call("write_file", {"path": MODULE, "content": src.replace(body(bug), body(target.correct))})
        tools.call("run_tests", {})
        tools.call("finish", {"message": f"fixed {target.name}"})
