"""Tasks: seeded, self-checking pieces of legitimate agentic work.

A task builds its workspace from a seed, so the same (task, seed) pair always gives the same files and the same
expected answer. It then checks the workspace after the model has worked in it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from .tools import ToolBox, standard_tools

Kind = Literal["code", "prose"]

SYSTEM_PROMPT = (
    "You are working in a project directory through tools. Inspect files before changing them, make the change "
    "the user asks for, check your work where a tool lets you, and call finish when you are done."
)


@dataclass
class Setup:
    """What a task hands the harness: the instructions and the tools, already bound to the workspace."""

    prompt: str
    tools: ToolBox
    system: str = SYSTEM_PROMPT


@dataclass
class Verdict:
    passed: bool
    score: float  # 0..1, partial credit
    checks: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


class Task(ABC):
    """Subclass, set `name`, `kind` and `description`, and implement the three methods."""

    name: ClassVar[str]
    kind: ClassVar[Kind]
    description: ClassVar[str]

    def __init__(self, seed: int):
        self.seed = seed

    @property
    def rng(self) -> random.Random:
        """A new generator seeded by the task and seed on every access. Take it once per plan, so that `build`,
        `verify` and `reference` each re-derive the same plan."""
        return random.Random(f"{self.name}:{self.seed}")

    def setup(self, workspace: Path) -> Setup:
        """Write the task's files into `workspace` (an empty directory) and return its prompt and tools."""
        workspace.mkdir(parents=True, exist_ok=True)
        prompt = self.build(workspace)
        return Setup(prompt=prompt, tools=standard_tools(workspace, test_command=self.test_command()))

    def test_command(self) -> list[str] | None:
        """argv of the workspace's test suite, if the task has one (adds the `run_tests` tool)."""
        return None

    @abstractmethod
    def build(self, workspace: Path) -> str:
        """Write the seeded files and return the user prompt, drawing every choice from one `self.rng`."""

    @abstractmethod
    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        """Judge the workspace after the run."""

    @abstractmethod
    def reference(self, workspace: Path, tools: ToolBox) -> None:
        """Solve the task through `tools` as a model would; the scripted harness and the tests use it."""


def verdict(checks: dict[str, bool], notes: str = "") -> Verdict:
    score = sum(checks.values()) / len(checks) if checks else 0.0
    return Verdict(passed=all(checks.values()), score=score, checks=checks, notes=notes)
