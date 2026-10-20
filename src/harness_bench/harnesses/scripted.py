"""A harness with no model: it plays the task's reference solution. Checks tasks and the pipeline offline."""

from __future__ import annotations

from ..meter import Meter
from ..task import Setup, Task
from .base import Harness, Transcript


class ScriptedHarness(Harness):
    name = "scripted"

    def run(self, task: Task, setup: Setup, meter: Meter, *, max_turns: int) -> Transcript:
        task.reference(setup.tools.root, setup.tools)
        return Transcript(stop="finish" if setup.tools.finished else "no_tool_calls")
