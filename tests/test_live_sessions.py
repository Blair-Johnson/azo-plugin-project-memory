"""Observational registry contracts; private stores, no production registry writes."""
from __future__ import annotations

import pickle
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_utils import State
from agent_utils.files.buffer_manager import BufferManager
from agent_zoo import live_session_registry as registry, session_load
from tmux_pilot.fs_store import RecordStore
from tmux_pilot.process_identity import ObservationState, ProcessIdentity

import project_memory as pm

FIRST = "344655c1-1111-1111-1111-111111111111"
CURRENT = "aaaaaaaa-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def private_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "home"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))


def wait_for(fn, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(.005)
    pytest.fail("registry did not settle")


def state(root):
    observer = State(token_budget=10000)
    observer.step = 1
    observer._session_id = CURRENT
    observer.buffer_manager = BufferManager()
    observer._agent_zoo_context = dict(project_name="demo", project_dir=str(root))
    return observer


def record(instance="one", sid=FIRST, *, health="alive", observed=99, status="running", **extra):
    identity = ProcessIdentity("host-id", "boot-id", "pid-ns", 123, "start-token")
    observation = registry.LiveProcessObservation(
        ObservationState(health), "fixture-observation", observed, 1
    ) if health is not None else None
    return registry.LiveSessionRecord(
        session_id=sid, instance_id=instance, hostname="remote-host", pid=123,
        websocket_url="ws://127.0.0.1:8123/agent", status=status,
        title="Fixture title", description="Fixture description", updated_at=100,
        process_identity=identity, process_observation=observation, **extra,
    )


class HistorySpy:
    def __init__(self):
        self.calls = []

    def issue_sessions(self, observer, rows):
        self.calls.append(rows)
        return {row["session_id"]: "ses_" + row["session_id"][:8] for row in rows}


def opened(tmp_path, monkeypatch, rows=(), *, history=None):
    root = tmp_path / "project"
    store = root / "coordination"
    store.mkdir(parents=True)
    calls = []

    def read(path, *, include_stale=False):
        calls.append(path)
        assert include_stale is True
        assert threading.current_thread() is not threading.main_thread()
        return tuple(rows)

    monkeypatch.setattr(registry, "list_live_sessions", read)
    now = [100.0]
    component = pm.ProjectLiveSessionBuffer(history=history, clock=lambda: now[0])
    observer = state(root)
    component(observer)
    text = wait_for(lambda: (v if "Registry Read (UTC; last successful): 1970" in
                            (v := component.render(observer).text) else None))
    return component, observer, now, calls, text


def test_per_instance_current_rlm_liveness_and_allowlisted_metadata(tmp_path, monkeypatch):
    history = HistorySpy()
    rows = [record("two"), record("one"), record("current", CURRENT, kind="rlm"),
            record("old", observed=1), record("future", observed=101),
            record("unverified", health=None), record("dead", health="dead", observed=1),
            record("retired", status="stopped"),
            replace(record("safe"), metadata={"password": "NEVER-DISPLAY-THIS"})]
    _, _, _, calls, text = opened(tmp_path, monkeypatch, rows, history=history)
    assert "Registry Instances: 9" in text
    assert "Active candidates: 7; confirmed alive: 4; unknown: 3" in text
    assert "Excluded: 1 terminal lifecycle; 1 other confirmed-stopped instances" in text
    assert "## Instance: one" in text and "## Instance: two" in text
    assert "## Instance: dead" not in text and "## Instance: retired" not in text
    assert "Current session: yes" in text and "Child/RLM: yes" in text
    assert text.index("## Instance: one") < text.index("## Instance: two")
    assert "process state: unknown" in text and "age: -1.0s" in text
    assert "NEVER-DISPLAY-THIS" not in text
    assert "ses_344655c1 | ses_344655c1_full" in text
    assert "NOT necessarily this runtime instance" in text
    assert "availability checked on open" in text
    assert {r["session_id"] for r in history.calls[-1]} == {FIRST, CURRENT}
    assert len(history.calls[-1]) == 2 and len(calls) == 1


@pytest.mark.parametrize(("endpoint", "expected"), [
    ("ws://127.0.0.1:8123/agent", "ws://127.0.0.1:8123/agent"),
    ("wss://user:SECRET@example.test:8123/agent?token=SECRET#SECRET",
     "wss://example.test:8123/agent [sanitized]"),
    ("ws://[::1]:8123/agent?secret=value", "ws://[::1]:8123/agent [sanitized]"),
    ("ws://user:SECRET@example.test/agent", "ws://example.test/agent [sanitized]"),
    ("ws://[broken/SECRET", "(unavailable: malformed advertised endpoint)"),
    ("ws://example.test:bad/SECRET", "(unavailable: malformed advertised endpoint)"),
    ("https://example.test/SECRET", "(unavailable: malformed advertised endpoint)"),
    ("ws://example.test/SECRET\n", "(unavailable: malformed advertised endpoint)"),
])
def test_endpoint_sanitization(endpoint, expected):
    assert pm.ProjectLiveSessionBuffer._endpoint(endpoint) == expected


def test_authoritative_context_override_and_attached_store_roots(tmp_path, monkeypatch):
    component, observer, now, calls, _ = opened(tmp_path, monkeypatch)
    override = tmp_path / "override"
    override.mkdir()
    observer._agent_zoo_context["project_store_root"] = str(override)
    observer.agent_db = SimpleNamespace(store=SimpleNamespace(root=tmp_path / "wrong"))
    wait_for(lambda: "Registry: empty" in component.render(observer).text)
    assert calls[-1] == override
    assert tmp_path / "wrong" not in calls
    del observer._agent_zoo_context["project_store_root"]
    attached = tmp_path / "attached"
    attached.mkdir()
    observer.agent_db.store.root = attached
    wait_for(lambda: "Registry: empty" in component.render(observer).text)
    assert calls[-1] == attached
    observer._agent_zoo_context["project_store_root"] = "relative/path"
    text = component.render(observer).text
    assert "must be absolute" in text
    assert not (tmp_path / "wrong").exists()


def test_missing_context_absent_empty_unavailable_and_unsupported_are_distinct(tmp_path, monkeypatch):
    component = pm.ProjectLiveSessionBuffer()
    observer = state(tmp_path / "project")
    observer._agent_zoo_context = {}
    assert "Registry: missing-context" in component.render(observer).text
    observer._agent_zoo_context = dict(project_name="absent", project_dir=str(tmp_path / "missing"))
    wait_for(lambda: "Registry: absent" in component.render(observer).text)
    assert not (tmp_path / "missing").exists()
    store = RecordStore(tmp_path / "empty")
    observer._agent_zoo_context = dict(project_name="empty", project_store_root=str(store.root))
    wait_for(lambda: "Registry: empty (successful read; zero instances)" in component.render(observer).text)
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "format.json").write_text("{bad JSON")
    observer._agent_zoo_context = dict(project_name="malformed", project_store_root=str(malformed))
    text = wait_for(lambda: (v if "Registry: unavailable" in (v := component.render(observer).text) else None))
    assert "No successful registry snapshot" in text
    monkeypatch.setattr(registry, "list_live_sessions", None)
    observer._agent_zoo_context = dict(project_name="unsupported", project_store_root=str(store.root))
    wait_for(lambda: "Registry: unsupported" in component.render(observer).text)


def test_failed_refresh_preserves_snapshot_time_recomputes_health_and_recovers(tmp_path, monkeypatch):
    component, observer, now, calls, text = opened(tmp_path, monkeypatch, [record()])
    old_read_time = next(line for line in text.splitlines() if line.startswith("Registry Read"))
    failed = []

    def failure(*args, **kwargs):
        failed.append(True)
        raise OSError("SECRET error payload must not leak")

    monkeypatch.setattr(registry, "list_live_sessions", failure)
    now[0] = 130
    text = wait_for(lambda: (v if "Registry: unavailable" in (v := component.render(observer).text) else None))
    assert "Last-good snapshot retained (stale)" in text
    assert old_read_time in text
    assert "confirmed alive: 0; unknown: 1" in text
    assert "SECRET" not in text
    for _ in range(5):
        component.render(observer)
    assert len(failed) == 1  # Retry cooldown is respected.
    now[0] = 136
    monkeypatch.setattr(registry, "list_live_sessions", lambda *a, **k: (record(observed=136),))
    text = wait_for(lambda: (v if "Registry: ready" in (v := component.render(observer).text) else None))
    assert "confirmed alive: 1" in text and old_read_time not in text


def test_buffer_manager_same_step_cache_readonly_and_pickle_safe_provider(tmp_path, monkeypatch):
    component, observer, now, _, _ = opened(tmp_path, monkeypatch, [record()])
    manager = observer.buffer_manager
    first = manager.resolve_for_read(observer, "live_sessions")
    assert first.readonly and first.path == "project-memory://live-sessions"
    now[0] += 30
    assert manager.resolve_for_read(observer, "live_sessions") is first
    assert "confirmed alive: 1" in first.text
    observer.step += 1
    later = manager.resolve_for_read(observer, "live_sessions")
    assert "confirmed alive: 0" in later.text
    with pytest.raises(ValueError, match="readonly|read-only|special"):
        manager.require_managed("live_sessions")
    restored = pickle.loads(pickle.dumps(manager))
    observer.step += 1
    observer.buffer_manager = restored
    component(observer)  # Rebind state after restoration, as the pipeline does.
    assert restored.resolve_for_read(observer, "live_sessions").readonly


def test_delayed_read_is_nonblocking_one_pending_and_worker_exits(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "coordination").mkdir(parents=True)
    release, entered = threading.Event(), threading.Event()
    calls = []

    def delayed(*args, **kwargs):
        calls.append(True)
        entered.set()
        assert release.wait(3)
        return (record(),)

    monkeypatch.setattr(registry, "list_live_sessions", delayed)
    component = pm.ProjectLiveSessionBuffer(clock=lambda: 100)
    observer = state(root)
    try:
        started = time.monotonic()
        assert "Registry: pending" in component.render(observer).text
        assert time.monotonic() - started < .2
        assert entered.wait(2)
        for _ in range(20):
            assert "Registry: pending" in component.render(observer).text
        assert calls == [True]
    finally:
        release.set()
    wait_for(lambda: "Registry: ready" in component.render(observer).text)
    wait_for(lambda: component._worker is None)


def test_private_registry_read_never_loads_history_probes_or_mutates(tmp_path, monkeypatch):
    import socket
    import tmux_pilot.process_identity as process_identity

    store = RecordStore(tmp_path / "private-store")
    row = record()
    store.put("live", row.instance_id, row.to_value())

    def contents():
        return {str(p.relative_to(store.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in store.root.rglob("*") if p.is_file()}

    before = contents()

    def forbidden(*args, **kwargs):
        pytest.fail("observational read performed a forbidden action")

    monkeypatch.setattr(session_load, "load_session", forbidden)
    monkeypatch.setattr(registry, "register_live_session", forbidden)
    monkeypatch.setattr(registry, "publish_live_session", forbidden)
    monkeypatch.setattr(registry, "capture_process_identity", forbidden)
    monkeypatch.setattr(process_identity, "observe_process", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(RecordStore, "put", forbidden)
    monkeypatch.setattr(RecordStore, "update", forbidden)
    observer = state(tmp_path / "project-without-catalog")
    observer._agent_zoo_context["project_store_root"] = str(store.root)
    component = pm.ProjectLiveSessionBuffer(history=HistorySpy(), clock=lambda: 100)
    text = wait_for(lambda: (v if "Registry: ready" in (v := component.render(observer).text) else None))
    assert "ses_344655c1_full" in text
    assert contents() == before
    assert not (tmp_path / "project-without-catalog").exists()


def test_row_limit_and_disabled_history(tmp_path, monkeypatch):
    component, observer, _, _, _ = opened(tmp_path, monkeypatch, [record("one"), record("two")])
    component.ROW_LIMIT = 1
    text = component.render(observer).text
    assert "Rows shown: 1 of 2 active candidates; truncated: 1" in text
    assert "History: disabled" in text
    assert "ses_344655c1" not in text
    assert "## Instance: one" in text and "## Instance: two" not in text


def test_refresh_queue_is_bounded_and_projects_do_not_share_snapshots(tmp_path, monkeypatch):
    release, entered = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return ()

    monkeypatch.setattr(registry, "list_live_sessions", blocked)
    component = pm.ProjectLiveSessionBuffer(clock=lambda: 100)
    observer = state(tmp_path / "project")
    try:
        first = tmp_path / "store-0"
        first.mkdir()
        observer._agent_zoo_context["project_store_root"] = str(first)
        component.render(observer)
        assert entered.wait(2)
        for i in range(1, component.QUEUE_LIMIT + 2):
            root = tmp_path / f"store-{i}"
            root.mkdir()
            observer._agent_zoo_context["project_store_root"] = str(root)
            text = component.render(observer).text
            assert "No successful registry snapshot" in text or "Registry: pending" in text
        assert "Registry: busy" in text
        assert component._work.qsize() == component.QUEUE_LIMIT
        assert sum(s["pending"] for s in component._snapshots.values()) == component.QUEUE_LIMIT + 1
    finally:
        release.set()
    wait_for(lambda: component._worker is None)


def test_corrupt_live_record_is_not_an_empty_registry(tmp_path):
    store = RecordStore(tmp_path / "private-store")
    store.put("live", "broken", {"session_id": FIRST, "websocket_url": "ws://host/path"})
    observer = state(tmp_path / "project")
    observer._agent_zoo_context["project_store_root"] = str(store.root)
    component = pm.ProjectLiveSessionBuffer()
    text = wait_for(lambda: (v if "Registry: unavailable" in (v := component.render(observer).text) else None))
    assert "No successful registry snapshot" in text
    assert "Registry: empty" not in text


def test_future_dead_evidence_is_unknown_not_excluded(tmp_path, monkeypatch):
    _, _, _, _, text = opened(tmp_path, monkeypatch, [record("future-dead", health="dead", observed=101)])
    assert "Active candidates: 1; confirmed alive: 0; unknown: 1" in text
    assert "## Instance: future-dead" in text
    assert "Excluded: 0 terminal lifecycle; 0 other confirmed-stopped instances" in text


def test_real_history_alias_is_issued_without_catalog_or_transcript_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registry discovery eagerly loaded a transcript")

    monkeypatch.setattr(session_load, "load_session", forbidden)
    history = pm.ProjectSessionBuffers(clock=lambda: 100)
    _, observer, _, _, text = opened(tmp_path, monkeypatch, [record()], history=history)
    assert "ses_344655c1 | ses_344655c1_full" in text
    assert not (Path(observer._agent_zoo_context["project_dir"]) / "sessions").exists()
