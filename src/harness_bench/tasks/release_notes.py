"""Prose: turn a seeded changelog of commit subjects into release notes for users."""

from __future__ import annotations

import re
from pathlib import Path

from ..task import Task, Verdict, verdict
from ..tools import ToolBox
from .log_report import prose_words

SUBJECTS = {
    "feat": [
        "add CSV export to reports",
        "support dark mode in the dashboard",
        "add retry policy to webhooks",
        "allow API keys to expire",
        "add bulk delete for projects",
        "show build duration in the history view",
        "support SSO login with OIDC",
        "add keyboard shortcuts to the editor",
    ],
    "fix": [
        "handle empty search queries",
        "stop leaking file handles on upload",
        "correct time zones in reminders",
        "prevent double charges on retry",
        "fix crash when a project has no owner",
        "respect locale when formatting numbers",
        "repair pagination on the audit log",
    ],
    "docs": ["clarify the install guide", "document rate limits", "fix typos in the API reference"],
    "chore": ["bump dependencies", "reformat with the new linter", "update CI images"],
    "refactor": ["split the billing module", "simplify config loading"],
}
MIN_WORDS = 60
NOTES = "RELEASE_NOTES.md"


class ReleaseNotes(Task):
    name = "release-notes"
    kind = "prose"
    description = "Write user-facing release notes from a seeded changelog: intro prose, features and fixes by PR."

    def _plan(self):
        rng = self.rng
        version = f"{rng.randint(1, 4)}.{rng.randint(0, 20)}.0"
        entries, pr = [], rng.randint(100, 900)
        for kind, n in (
            ("feat", rng.randint(3, 5)),
            ("fix", rng.randint(3, 5)),
            ("docs", 2),
            ("chore", 2),
            ("refactor", 1),
        ):
            for subject in rng.sample(SUBJECTS[kind], n):
                pr += rng.randint(1, 9)
                entries.append((kind, subject, pr))
        rng.shuffle(entries)
        return version, entries

    def build(self, workspace: Path) -> str:
        version, entries = self._plan()
        (workspace / "CHANGES.txt").write_text("".join(f"{k}: {s} (#{n})\n" for k, s, n in entries))
        return (
            f"Write {NOTES} for version {version} from CHANGES.txt. Start with a paragraph of at least "
            f"{MIN_WORDS} words that tells users what this release means for them. Then add a '## Features' "
            "section and a '## Fixes' section, describing each user-visible change in plain language with its "
            "PR number (#123). Leave out docs, chore and refactor entries entirely. Call finish when done."
        )

    def verify(self, workspace: Path, tools: ToolBox) -> Verdict:
        version, entries = self._plan()
        path = workspace / NOTES
        text = path.read_text() if path.exists() else ""
        sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
        intro = sections[0]
        features = next((s for s in sections[1:] if s.lower().startswith("feature")), "")
        fixes = next((s for s in sections[1:] if s.lower().startswith("fix")), "")
        has = lambda body, n: re.search(rf"#{n}(?!\d)", body) is not None
        return verdict(
            {
                "notes_written": bool(text),
                "intro_long_enough": prose_words(intro) >= MIN_WORDS,
                "features_listed": bool(features) and all(has(features, n) for k, _, n in entries if k == "feat"),
                "fixes_listed": bool(fixes) and all(has(fixes, n) for k, _, n in entries if k == "fix"),
                "internal_omitted": not any(has(text, n) for k, _, n in entries if k not in ("feat", "fix")),
            },
            notes=f"version {version}",
        )

    def reference(self, workspace: Path, tools: ToolBox) -> None:
        version, entries = self._plan()
        tools.call("read_file", {"path": "CHANGES.txt"})
        intro = (
            f"# Release {version}\n\nThis release focuses on making everyday work smoother and more reliable. "
            "It brings several new capabilities that people have asked for, and it fixes a number of problems "
            "that could interrupt your work or produce confusing results. Upgrading is recommended for "
            "everyone, and no manual migration steps are required. The sections below describe each change "
            "and link the pull request where you can read more.\n"
        )
        feats = "".join(f"- {s[0].upper()}{s[1:]} (#{n})\n" for k, s, n in entries if k == "feat")
        fixes = "".join(f"- {s[0].upper()}{s[1:]} (#{n})\n" for k, s, n in entries if k == "fix")
        tools.call("write_file", {"path": NOTES, "content": f"{intro}\n## Features\n\n{feats}\n## Fixes\n\n{fixes}"})
        tools.call("finish", {"message": f"wrote release notes for {version}"})
