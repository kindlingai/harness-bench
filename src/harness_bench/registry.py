"""Find tasks and harnesses through entry points, so another package can add its own without editing this one."""

from __future__ import annotations

from importlib.metadata import entry_points

from .harnesses.base import Harness
from .task import Task


def _load(group: str) -> dict[str, type]:
    return {ep.name: ep.load() for ep in entry_points(group=group)}


def tasks() -> dict[str, type[Task]]:
    return dict(sorted(_load("harness_bench.tasks").items()))


def harnesses() -> dict[str, type[Harness]]:
    return dict(sorted(_load("harness_bench.harnesses").items()))
