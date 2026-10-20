"""A harness with no model: it plays the task's reference solution. Checks tasks and the pipeline offline."""

from __future__ import annotations

from ..meter import Meter
from ..task import Setup, Task
from .base import Harness, Transcript


class ScriptedHarness(Harness):
    name = "scripted"

    def run(self, task: Task, setup: Setup, meter: Meter, *, max_turns: int) -> Transcript:
        if setup.new_worker is not None:
            setup.tools.spawner = lambda prompt: self._worker(task, setup, prompt)
        task.reference(setup.tools.root, setup.tools)
        return Transcript(stop="finish" if setup.tools.finished else "no_tool_calls")

    @staticmethod
    def _worker(task: Task, setup: Setup, prompt: str) -> str:
        box = setup.new_worker()
        setup.workers.append(box)
        task.reference_worker(prompt, box)
        return box.final_message
