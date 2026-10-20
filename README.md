# harness-bench

Benchmark LLM serving stacks on real agentic work. A model gets a small project and a set of tools, does a
legitimate task (fix a bug, implement a function, write an incident report, write release notes), and the task
checks the result. While it works, a shared meter times every request and splits the output into **prose**,
**code**, **reasoning** and **tool** arguments, so you get decode throughput for the kind of text a coding agent
actually writes, not for a synthetic prompt.

- **Tasks** are seeded: the same task and seed always produce the same files, prompt and expected answer. Run many
  seeds to get a distribution, not an anecdote.
- **Harnesses** are independent adapters that drive a model through the tool loop. Each one is a small class;
  adding one for another API or agent framework does not touch the tasks or the meter.
- **Numbers come with intervals.** Pass rates get Wilson 95% intervals; scores and throughput get Student-t 95%
  intervals over per-run values. The summary flags any mean whose interval is wider than ±10% of it.

## Quick start

```bash
uv sync
uv run harness-bench list

# offline: every task's reference solution through the scripted harness
uv run harness-bench run --harness scripted --seeds 5

# any OpenAI-compatible server (vLLM, SGLang, llama.cpp, OpenAI, OpenRouter, ...)
uv run harness-bench run --harness openai-chat \
  -o base_url=http://localhost:8000/v1 -o model=my-model \
  --seeds 20 --out runs.jsonl --summary summary.json
```

`--seeds 20` runs seeds 0-19 of every task (also `5-9` or `1,4,9`). `--out` appends one JSON line per run with the
verdict, every request's timings and token split; `harness-bench summarise runs.jsonl ...` rebuilds the summary from
any number of such files.

OpenRouter, for example:

```bash
OPENAI_API_KEY=$OPENROUTER_API_KEY uv run harness-bench run \
  -o base_url=https://openrouter.ai/api/v1 -o model=qwen/qwen3-coder --seeds 20
```

> Tasks run model-written code (the test suites) on your machine. Run untrusted models in a container or VM.

## Tasks

| task | kind | what the model does | how it is checked |
|---|---|---|---|
| `fix-bug` | code | find one seeded bug in a utility module and fix it | the unittest suite passes, the tests are unedited, the model ran them |
| `implement-function` | code | implement a function from its docstring and three examples | 3 visible + 40 seeded hidden cases; score is the fraction passed |
| `log-report` | prose | read three service logs, write an incident handoff | at least 120 words of prose stating four exact facts (count, service, timestamp, code) |
| `release-notes` | prose | turn a changelog into user-facing release notes | intro of at least 60 words, every feature and fix PR listed, internal changes left out |

## Tools

Every task offers the same tools, confined to its workspace: `list_files`, `read_file`, `write_file`, `search`,
`finish`, and `run_tests` where the task has a test suite. A bad call (unknown file, path outside the workspace,
malformed JSON) returns an error message to the model instead of failing the run.

## What the throughput numbers mean

Per request, the meter records the time to the first streamed token (prefill), the time from there to the last
token (decode), and the server's token counts. Output is classed by what it is:

| class | counted from |
|---|---|
| prose | assistant text outside code fences; content written to `.md`, `.txt` and other non-code files |
| code | assistant text inside code fences; content written to source and config files |
| reasoning | the reasoning channel (`reasoning_content` / `reasoning`) |
| tool | every other tool-call argument |

Decode time is charged per arrival, not per request. Servers and routers send output in batches: pieces that arrive
within 5 ms of each other form one flush. Each flush is charged the time since the previous flush, split across its
characters by class. So a request that reasons, writes code and then explains itself splits its time correctly, and
a provider that holds a tool call back and sends it in one burst is still charged the time it spent generating it.
Characters convert to tokens with each request's own ratio from the server's usage counts. The first flush closes
prefill; prefill rate uses prompt tokens minus any the server reports as cached.

Per-class rates are only as fine as the stream: `flushes_per_turn_median` in each run says how coarse it was. A
local vLLM or llama.cpp server streams every token; routed APIs send anywhere from a few dozen to a few hundred
flushes per request.

## Layout

```
src/harness_bench/
  task.py         Task base class, Setup, Verdict
  tools.py        the shared tool set and its workspace sandbox
  meter.py        per-request timing and the prose/code/reasoning/tool split
  stats.py        Wilson and t intervals
  runner.py       tasks x seeds through a harness; the summary
  cli.py          harness-bench list | run | summarise
  harnesses/      openai_chat.py (streaming tool calls), scripted.py (reference solutions, no model)
  tasks/          one module per task
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to add a task or a harness.

## License

MIT
