import hashlib
from pathlib import Path

import pytest

from harness_bench import registry
from harness_bench.harnesses.scripted import ScriptedHarness
from harness_bench.runner import run_one

TASKS = registry.tasks()
SEEDS = range(8)


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", TASKS)
def test_same_seed_same_workspace(name, tmp_path):
    a, b = TASKS[name](3).setup(tmp_path / "a"), TASKS[name](3).setup(tmp_path / "b")
    assert a.prompt == b.prompt
    assert tree_digest(tmp_path / "a") == tree_digest(tmp_path / "b")


@pytest.mark.parametrize("name", TASKS)
def test_seeds_vary(name, tmp_path):
    digests = {tree_digest(TASKS[name](s).setup(tmp_path / str(s)).tools.root) for s in SEEDS}
    assert len(digests) > 1


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_reference_passes(name, seed):
    r = run_one(TASKS[name], seed, ScriptedHarness(), max_turns=1)
    assert r.passed, r.checks


@pytest.mark.parametrize("name", TASKS)
def test_doing_nothing_fails(name, tmp_path):
    task = TASKS[name](0)
    setup = task.setup(tmp_path)
    assert not task.verify(tmp_path, setup.tools).passed


def test_tools_stay_in_workspace(tmp_path):
    setup = TASKS["fix-bug"](0).setup(tmp_path / "ws")
    assert setup.tools.call("read_file", {"path": "../../etc/passwd"}).startswith("error:")
    assert setup.tools.call("nope", {}).startswith("error: unknown tool")
    assert setup.tools.call("read_file", "{not json").startswith("error: bad arguments")


def test_audit_needs_delegation(tmp_path):
    """The lead cannot read; a worker reads at most two files."""
    task = TASKS["audit-services"](0)
    setup = task.setup(tmp_path)
    assert "read_file" not in setup.tools.tools and "spawn_agent" in setup.tools.tools
    worker = setup.new_worker()
    files = sorted(p.name for p in (tmp_path / "services").iterdir())
    assert not worker.call("read_file", {"path": f"services/{files[0]}"}).startswith("error")
    worker.call("read_file", {"path": f"services/{files[1]}"})
    assert worker.call("read_file", {"path": f"services/{files[2]}"}).startswith("error: this agent's read budget")


def test_audit_reference_delegates():
    r = run_one(TASKS["audit-services"], 0, ScriptedHarness(), max_turns=1)
    assert r.checks["delegated"] and r.passed
