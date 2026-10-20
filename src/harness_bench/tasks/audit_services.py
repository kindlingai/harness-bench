"""Delegation: audit a set of service configs the lead agent cannot read itself.

The lead can list files, write its report and spawn subagents, but has no read tool. Each subagent can read at most
two files, so the lead has to split the work across several subagents and combine their reports. Every config has
exactly one setting that is wrong for production; the report must name it for every service.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..task import Task, Verdict, verdict
from ..tools import ToolBox, standard_tools

SERVICES = ["billing", "auth", "search", "ingest", "notifier", "gateway", "scheduler", "media", "ledger", "export"]
N_SERVICES = 6
READS_PER_WORKER = 2
REPORT = "AUDIT.md"

# issue key -> (what the report must mention, a regex any matching mention satisfies)
ISSUES = {
    "replicas": r"replica",
    "timeout": r"timeout",
    "tls": r"tls",
    "health": r"health",
    "memory": r"memory|limit",
    "debug": r"debug",
}


def _config(name: str, issue: str, rng) -> str:
    request = rng.choice([256, 512, 768, 1024, 2048])
    values = {
        "replicas": str(rng.randint(2, 6)),
        "timeout_s": str(rng.choice([10, 15, 30, 45, 60])),
        "tls": "true",
        "health_path": '"/healthz"',
        "memory_request_mb": str(request),
        "memory_limit_mb": str(request * 2),
        "debug": "false",
    }
    if issue == "replicas":
        values["replicas"] = "0"
    elif issue == "timeout":
        values["timeout_s"] = "0"
    elif issue == "tls":
        values["tls"] = "false"
    elif issue == "health":
        values["health_path"] = '""'
    elif issue == "memory":
        values["memory_limit_mb"] = str(request // 2)
    elif issue == "debug":
        values["debug"] = "true"
    body = "".join(f"{k} = {v}\n" for k, v in values.items())
    return f'name = "{name}"\nenvironment = "production"\n{body}'


def diagnose(text: str) -> tuple[str, str]:
    """(service name, issue key) of one config; the reference subagent's analysis."""
    kv = dict(re.findall(r"^(\w+) = (.*)$", text, re.MULTILINE))
    name = kv["name"].strip('"')
    if kv["replicas"] == "0":
        return name, "replicas"
    if kv["timeout_s"] == "0":
        return name, "timeout"
    if kv["tls"] == "false":
        return name, "tls"
    if kv["health_path"] == '""':
        return name, "health"
    if int(kv["memory_limit_mb"]) < int(kv["memory_request_mb"]):
        return name, "memory"
    return name, "debug"


EXPLAIN = {
    "replicas": "replicas is 0, so nothing serves traffic",
    "timeout": "timeout_s is 0, so every request times out at once or never does",
    "tls": "tls is false in production",
    "health": "health_path is empty, so the orchestrator cannot health-check it",
    "memory": "memory_limit_mb is below memory_request_mb, so it cannot be scheduled",
    "debug": "debug is true in production",
}


class AuditServices(Task):
    name = "audit-services"
    kind = "prose"
    description = "Audit service configs the lead cannot read; it must delegate to subagents that read two files each."

    def _plan(self):
        rng = self.rng
        names = rng.sample(SERVICES, N_SERVICES)
        issues = rng.sample(list(ISSUES), N_SERVICES)
        configs = {n: _config(n, i, rng) for n, i in zip(names, issues)}
        return names, dict(zip(names, issues)), configs

    def lead_tools(self, workspace: Path) -> ToolBox:
        return standard_tools(workspace, only=("list_files", "write_file", "finish", "spawn_agent"), spawn=True)

    def worker_tools(self, workspace: Path) -> ToolBox:
        return standard_tools(workspace, only=("list_files", "read_file", "finish"), read_budget=READS_PER_WORKER)

    def build(self, workspace: Path) -> str:
        _, _, configs = self._plan()
        (workspace / "services").mkdir()
        for n, text in configs.items():
            (workspace / "services" / f"{n}.toml").write_text(text)
        return (
            "Audit the production service configs in services/. Each file has exactly one setting that is wrong for "
            "production. You cannot read files yourself: delegate with spawn_agent, and note that each subagent can "
            f"read at most {READS_PER_WORKER} files. Then write {REPORT} with one line per service giving the service "
            "name, the setting at fault and why it is a problem, and call finish."
        )

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        names, issues, _ = self._plan()
        path = workspace / REPORT
        lines = path.read_text().lower().splitlines() if path.exists() else []
        found = {n: any(re.search(rf"\b{n}\b", ln) and re.search(ISSUES[issues[n]], ln) for ln in lines) for n in names}
        spawned = sum(c.name == "spawn_agent" for c in tools.calls)
        checks = {"report_written": bool(lines), "delegated": spawned >= math.ceil(N_SERVICES / READS_PER_WORKER)}
        checks |= {f"found_{n}": ok for n, ok in found.items()}
        return verdict(checks, notes=f"{spawned} subagents; expected {issues}")

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        files = tools.call("list_files", {"path": "services"}).splitlines()
        rows = []
        for i in range(0, len(files), READS_PER_WORKER):
            batch = files[i : i + READS_PER_WORKER]
            report = tools.call(
                "spawn_agent",
                {
                    "task": f"Read {' and '.join(batch)}. Each is a production service config with exactly one setting "
                    "that is wrong for production. Report, one line per file, the service name and the bad setting."
                },
            )
            rows += report.splitlines()
        tools.call("write_file", {"path": REPORT, "content": "# Service audit\n\n" + "".join(f"- {r}\n" for r in rows)})
        tools.call("finish", {"message": f"audited {len(files)} services"})

    def reference_worker(self, prompt: str, tools: ToolBox) -> None:
        lines = []
        for path in re.findall(r"services/[\w-]+\.toml", prompt):
            name, issue = diagnose(tools.call("read_file", {"path": path}))
            lines.append(f"{name}: {EXPLAIN[issue]}")
        tools.call("finish", {"message": "\n".join(lines)})
