"""The tools a task offers the model, confined to the task's workspace.

Every harness exposes the same tools, so results compare across harnesses. A tool returns text for the model and
never raises: a bad call (an unknown file, a path outside the workspace) is an error message the model can read.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT = 8_000  # characters of tool output returned to the model


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]

    def schema(self) -> dict:
        """The OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class Call:
    """One executed tool call, kept for the verifier and the report."""

    name: str
    arguments: dict
    result: str


@dataclass
class ToolBox:
    """The workspace and the tools bound to it. `finished` is set by the `finish` tool."""

    root: Path
    tools: dict[str, Tool] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    finished: bool = False
    final_message: str = ""
    test_command: list[str] | None = None
    reads_left: int | None = None  # a per-agent read budget, when the task sets one
    # Runs one subagent on a task description and returns its report; the harness binds it (spawn_agent tool).
    spawner: Callable[[str], str] | None = None

    def path(self, rel: str) -> Path:
        """Resolve `rel` inside the workspace; raises ValueError for a path that escapes it."""
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path outside the workspace: {rel}")
        return p

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: str | dict) -> str:
        """Run one tool call from the model; malformed arguments come back as an error string."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            if not isinstance(args, dict):
                raise TypeError("arguments must be a JSON object")
        except (ValueError, TypeError) as e:
            result = f"error: bad arguments: {e}"
            self.calls.append(Call(name, {}, result))
            return result
        tool = self.tools.get(name)
        if tool is None:
            result = f"error: unknown tool {name!r}; available: {', '.join(self.tools)}"
        else:
            try:
                result = tool.fn(**args)
            except TypeError as e:
                result = f"error: {e}"
            except (OSError, ValueError) as e:
                result = f"error: {e}"
        result = result if len(result) <= MAX_OUTPUT else result[:MAX_OUTPUT] + "\n[output truncated]"
        self.calls.append(Call(name, args, result))
        return result


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def standard_tools(
    root: Path,
    *,
    test_command: list[str] | None = None,
    only: tuple[str, ...] | None = None,
    read_budget: int | None = None,
    spawn: bool = False,
) -> ToolBox:
    """The shared tool set. `test_command` adds `run_tests` (argv run in the workspace); `only` keeps just the named
    tools; `read_budget` caps `read_file` calls; `spawn` adds `spawn_agent`, which needs `box.spawner` bound."""
    box = ToolBox(root.resolve(), test_command=test_command, reads_left=read_budget)

    def list_files(path: str = ".") -> str:
        base = box.path(path)
        files = sorted(str(p.relative_to(box.root)) for p in base.rglob("*") if p.is_file())
        return "\n".join(files) or "(no files)"

    def read_file(path: str) -> str:
        if box.reads_left is not None:
            if box.reads_left <= 0:
                return "error: this agent's read budget is used up; finish and report what you found"
            box.reads_left -= 1
        return box.path(path).read_text()

    def write_file(path: str, content: str) -> str:
        p = box.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def search(pattern: str, path: str = ".") -> str:
        rx = re.compile(pattern)
        base = box.path(path)
        hits = []
        for p in sorted(base.rglob("*") if base.is_dir() else [base]):
            if not p.is_file():
                continue
            for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(box.root)}:{n}: {line}")
        return "\n".join(hits) or "(no matches)"

    def run_tests() -> str:
        proc = subprocess.run(box.test_command, cwd=box.root, capture_output=True, text=True, timeout=60, check=False)
        return f"exit code {proc.returncode}\n{proc.stdout}{proc.stderr}"

    def finish(message: str = "") -> str:
        box.finished, box.final_message = True, message
        return "finished"

    def spawn_agent(task: str) -> str:
        if box.spawner is None:
            return "error: this harness cannot run subagents"
        return box.spawner(task)

    path_arg = {"type": "string", "description": "path relative to the workspace root"}
    for tool in [
        Tool(
            "list_files",
            "List every file under a directory of the workspace.",
            _obj({"path": {**path_arg, "default": "."}}, []),
            list_files,
        ),
        Tool("read_file", "Read a text file.", _obj({"path": path_arg}, ["path"]), read_file),
        Tool(
            "write_file",
            "Create or overwrite a text file with the given content.",
            _obj({"path": path_arg, "content": {"type": "string"}}, ["path", "content"]),
            write_file,
        ),
        Tool(
            "search",
            "Search files for a regular expression; returns path:line: text for each match.",
            _obj({"pattern": {"type": "string"}, "path": {**path_arg, "default": "."}}, ["pattern"]),
            search,
        ),
        Tool(
            "finish",
            "Call when the task is complete, with a one-line summary of what you did.",
            _obj({"message": {"type": "string"}}, []),
            finish,
        ),
        Tool(
            "spawn_agent",
            "Start a subagent with its own fresh context and tools on a self-contained task. Describe the task fully; "
            + "the subagent sees nothing of this conversation. Returns the subagent's final report.",
            _obj({"task": {"type": "string", "description": "complete instructions for the subagent"}}, ["task"]),
            spawn_agent,
        ),
    ]:
        if (only is None or tool.name in only) and (spawn or tool.name != "spawn_agent"):
            box.tools[tool.name] = tool
    if test_command:
        box.tools["run_tests"] = Tool(
            "run_tests", "Run the project's test suite and return its output.", _obj({}, []), run_tests
        )
    return box


def python_unittest() -> list[str]:
    """argv for running a workspace's unittest suite with this interpreter."""
    return [sys.executable, "-m", "unittest", "-q"]
