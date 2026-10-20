# Contributing

Tasks and harnesses are plugins found through entry points, so they can live in this repository or in your own
package.

## Add a task

A task is a class with three methods. Keep it small, deterministic and honest: real work a coding agent would be
asked to do, checked by facts, not by matching the model's wording.

```python
from pathlib import Path
from harness_bench import Task, ToolBox, Verdict, verdict


class MyTask(Task):
    name = "my-task"
    kind = "code"  # or "prose": what the model mostly writes
    description = "one line for `harness-bench list`"

    def build(self, workspace: Path) -> str:
        # rng = self.rng (seeded by the task name and seed); write files and return the prompt
        ...

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        # inspect the workspace (and tools.calls if the process matters) and return named checks
        return verdict({"file_written": ..., "facts_correct": ...})

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        # solve it through tools.call(...) exactly as a model would; the tests require it to pass
        ...
```

A task that needs delegation overrides `lead_tools()` (for example `standard_tools(ws, only=(...), spawn=True)` to
drop reading and add `spawn_agent`) and `worker_tools()` (each subagent's tools, optionally with a `read_budget`), plus
`reference_worker()` for the scripted harness. See `tasks/audit_services.py`.

Override `test_command()` to return an argv (for example `python_unittest()`) and the task gains a `run_tests`
tool. Register it in `pyproject.toml`:

```toml
[project.entry-points."harness_bench.tasks"]
my-task = "harness_bench.tasks.my_task:MyTask"
```

Rules the test suite enforces for every registered task:

- the same seed gives byte-identical files and prompt; different seeds give different workspaces;
- the reference solution passes for seeds 0-7;
- an untouched workspace fails.

Draw every random choice from `self.rng`, never from `random` or the clock. Each access to `self.rng` returns a
new generator with the same seed, so take it once at the start of a planning method (`rng = self.rng`) and call that
method from `build`, `verify` and `reference` alike. If a seeded draw can produce an
ambiguous answer (a tie for "most errors"), redraw until it cannot, as `log-report` does.

## Add a harness

A harness drives one model through the tool loop. Subclass `Harness`, declare the options it takes, and report
every model request to the meter:

```python
from harness_bench import Harness, Transcript


class MyHarness(Harness):
    name = "my-api"
    options = {"base_url": "...", "model": "..."}

    def run(self, task, setup, meter, *, max_turns):
        # setup.system, setup.prompt: the conversation start
        # setup.tools.schemas(): OpenAI-style function definitions
        # setup.tools.call(name, arguments) -> str: execute one call
        # setup.tools.finished: True once the model called finish
        # setup.new_worker: set for delegation tasks; bind setup.tools.spawner to a function that takes the
        #   subagent's task text, runs it with setup.new_worker() and setup.worker_system, and returns its report
        #   (append each worker box to setup.workers and tag its turns with meter.begin(f"sub{n}"))
        for _ in range(max_turns):
            turn = meter.begin()
            ...  # send the request, stream the reply
            meter.first_token(turn)  # on the first streamed delta of any kind
            meter.content(turn, text)  # assistant text
            meter.text(turn, "reasoning", r)  # reasoning text
            meter.tool_call(turn, name, args)  # each complete tool call
            meter.end(turn, usage)  # server usage dict, or None to estimate
            ...
        return Transcript(messages=..., stop="finish")
```

`harnesses/openai_chat.py` is the worked example. Register it under
`[project.entry-points."harness_bench.harnesses"]`. Stream if the API can: without streaming there is no time to
first token and prefill and decode cannot be separated.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```
