"""The harness interface: drive one model through one task's tools until it finishes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from ..meter import Meter
from ..task import Setup, Task


@dataclass
class Transcript:
    messages: list[dict] = field(default_factory=list)
    stop: str = ""  # "finish", "no_tool_calls", "max_turns" or "error: ..."


class Harness(ABC):
    """Subclass, set `name`, list the `options` it takes (name -> help, all strings from the CLI), and implement
    `run`. A harness owns the conversation with the model; tools and timing are shared (`setup.tools`, `meter`)."""

    name: ClassVar[str]
    options: ClassVar[dict[str, str]] = {}

    def __init__(self, **options: str):
        unknown = set(options) - set(self.options)
        if unknown:
            raise ValueError(f"{self.name}: unknown options {sorted(unknown)}; takes {sorted(self.options)}")
        self.opts = options

    @abstractmethod
    def run(self, task: Task, setup: Setup, meter: Meter, *, max_turns: int) -> Transcript:
        """Work the task through `setup.tools`, reporting every model request to `meter`."""
