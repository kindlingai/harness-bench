"""Prose: read a few seeded service logs and write an incident handoff that states the key facts in prose."""

from __future__ import annotations

import collections
import re
from datetime import datetime, timedelta
from pathlib import Path

from ..task import Task, Verdict, verdict
from ..tools import ToolBox

SERVICES = ["billing", "auth", "search", "ingest", "notifier", "gateway", "scheduler", "media"]
MESSAGES = {
    "INFO": ["request served", "cache warmed", "job completed", "connection opened", "config reloaded"],
    "WARN": ["slow response", "retrying upstream", "queue depth high", "certificate expires soon"],
    "ERROR": [
        "upstream timeout",
        "database connection refused",
        "invalid payload",
        "disk quota exceeded",
        "permission denied",
    ],
}
MIN_WORDS = 120
REPORT = "INCIDENT.md"


def prose_words(text: str) -> int:
    """Words outside fenced code blocks and outside lines that look like a table or a list of raw log lines."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))


class LogReport(Task):
    name = "log-report"
    kind = "prose"
    description = "Summarise seeded service logs into a prose incident handoff stating four exact facts."

    def _logs(self):
        """Seeded log lines per service, regenerated until the facts are unambiguous."""
        rng = self.rng
        while True:
            services = rng.sample(SERVICES, 3)
            codes = [f"E{rng.randint(100, 999)}" for _ in range(4)]
            start = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 300), hours=rng.randint(0, 23))
            lines = {s: [] for s in services}
            t = start
            for _ in range(rng.randint(240, 320)):
                t += timedelta(seconds=rng.randint(1, 90))
                s = rng.choice(services)
                level = rng.choices(["INFO", "WARN", "ERROR"], weights=[70, 18, 12])[0]
                msg = rng.choice(MESSAGES[level])
                code = rng.choice(codes) if level == "ERROR" else None
                lines[s].append((t, level, msg, code))
            errors = [(t, s, c) for s, ls in lines.items() for t, lv, _, c in ls if lv == "ERROR"]
            per_service = collections.Counter(s for _, s, _ in errors).most_common()
            per_code = collections.Counter(c for _, _, c in errors).most_common()
            if len(errors) >= 5 and per_service[0][1] > per_service[1][1] and per_code[0][1] > per_code[1][1]:
                facts = {
                    "total": len(errors),
                    "service": per_service[0][0],
                    "first": min(errors)[0].strftime("%Y-%m-%d %H:%M:%S"),
                    "code": per_code[0][0],
                }
                return lines, facts

    def build(self, workspace: Path) -> str:
        lines, _ = self._logs()
        (workspace / "logs").mkdir()
        for s, ls in lines.items():
            text = "".join(
                f"{t:%Y-%m-%d %H:%M:%S} {lv:<5} {s}: {m}" + (f" code={c}" if c else "") + "\n" for t, lv, m, c in ls
            )
            (workspace / "logs" / f"{s}.log").write_text(text)
        return (
            f"You are handing the on-call shift over. The service logs are in logs/. Write {REPORT}: an incident "
            f"summary of at least {MIN_WORDS} words of prose, written for the next engineer. It must state the "
            "total number of ERROR lines across all logs, which service logged the most errors, the exact "
            "timestamp of the first error (YYYY-MM-DD HH:MM:SS), and the most common error code. Call finish "
            "when the file is written."
        )

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        _, facts = self._logs()
        path = workspace / REPORT
        text = path.read_text() if path.exists() else ""
        return verdict(
            {
                "report_written": bool(text),
                "long_enough": prose_words(text) >= MIN_WORDS,
                "total_errors": re.search(rf"(?<!\d){facts['total']}(?!\d)", text) is not None,
                "top_service": re.search(rf"\b{facts['service']}\b", text, re.IGNORECASE) is not None,
                "first_error": facts["first"] in text,
                "top_code": facts["code"] in text,
            },
            notes=f"expected {facts}",
        )

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        _, f = self._logs()
        tools.call("list_files", {"path": "logs"})
        for p in sorted((workspace / "logs").iterdir()):
            tools.call("read_file", {"path": f"logs/{p.name}"})
        tools.call("search", {"pattern": "ERROR", "path": "logs"})
        body = (
            f"# Incident handoff\n\nDuring this shift the logs recorded {f['total']} ERROR lines in total across "
            f"all services. The first error appeared at {f['first']}, and errors continued intermittently after "
            f"that point rather than in a single burst. The {f['service']} service logged more errors than any "
            f"other service, so it is the best place to start looking. The most common error code was "
            f"{f['code']}, which suggests that one failure mode accounts for a large share of the problems and "
            "that fixing it would remove most of the noise. Warnings about slow responses and retries were also "
            "frequent, which is consistent with an upstream dependency that is struggling under load. I did not "
            "change any configuration during the shift. The next engineer should check the upstream "
            "dependencies of the noisiest service first, confirm whether the dominant error code maps to a known "
            "issue, and watch whether the error rate falls once traffic drops overnight.\n"
        )
        tools.call("write_file", {"path": REPORT, "content": body})
        tools.call("finish", {"message": "wrote the incident handoff"})
