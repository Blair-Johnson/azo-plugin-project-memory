"""Observational registry contracts via AgentZooSessionDriver; private stores only."""
from __future__ import annotations

import pickle
import queue
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_utils.core.events import IoCompleted, IoFailed, RenderRequested, TurnSettled
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import Pipeline
from agent_utils.core.session import Session
from agent_utils.core.state import State
from agent_utils.features.buffers import BufferManager
from agent_utils.features.core import IoFence
from agent_utils.runtime.simulation import SimulationRuntime
from agent_zoo.drivers.sessions import (
    AgentZooSessionDriver,
    SimSessionDriver,
    live_record_to_info,
)
from agent_zoo.events import (
    DiscoverSessions,
    DiscoverSessionsResult,
    LiveSessionInfo,
    SessionReady,
)
from agent_zoo.host import registry
from agent_zoo.storage import load as session_load
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


def info_from_record(rec, *, now=100.0):
    return live_record_to_info(rec, now=now, probe_local_processes=False, sanitize_url=True)


class HistorySpy:
    def __init__(self):
        self.calls = []

    def issue_sessions(self, observer, rows):
        self.calls.append(rows)
        return {row["session_id"]: "ses_" + row["session_id"][:8] for row in rows}


def dispatch_via_driver(component, observer, event, driver, *, ts=None):
    """Step ``event``, execute submitted ``DiscoverSessions`` on ``driver``, feed result back."""
    if ts is not None:
        event.ts = ts
    component.handle(event, observer, lambda _ev: None)
    effects = observer.io.take_outbox()
    for effect in effects:
        assert isinstance(effect, SubmitOp)
        assert isinstance(effect.request, DiscoverSessions)
        assert effect.request.include_stale is True
        assert effect.request.probe_local_processes is False
        inbox: queue.Queue = queue.Queue()
        driver.start(inbox.put)
        try:
            driver.submit(effect.op_id, effect.request)
            io_ev = inbox.get(timeout=4.0)
        finally:
            driver.close()
        observer.io.complete(effect.op_id)
        if ts is not None:
            io_ev.ts = ts
        component.handle(io_ev, observer, lambda _ev: None)
    return effects


def opened(tmp_path, rows=(), *, history=None, now_val=100.0):
    """Step ``ProjectLiveSessionBuffer`` through ``state.io`` + ``IoCompleted(DiscoverSessionsResult)``."""
    root = tmp_path / "project"
    now = [float(now_val)]
    component = pm.ProjectLiveSessionBuffer(history=history, clock=lambda: now[0])
    observer = state(root)
    requests = []

    def refresh(ev=None, *, rows_override=None, result_override=None, error=None):
        event = ev if ev is not None else RenderRequested(ts=now[0])
        component.handle(event, observer, lambda _e: None)
        effects = observer.io.take_outbox()
        for eff in effects:
            assert isinstance(eff, SubmitOp)
            assert isinstance(eff.request, DiscoverSessions)
            assert eff.request.include_stale is True
            assert eff.request.probe_local_processes is False
            requests.append(eff.request)
            observer.io.complete(eff.op_id)
            if error is not None:
                component.handle(IoFailed(op_id=eff.op_id, error=error, ts=now[0]), observer, lambda _e: None)
            elif result_override is not None:
                component.handle(IoCompleted(op_id=eff.op_id, value=result_override, ts=now[0]), observer, lambda _e: None)
            else:
                src = rows if rows_override is None else rows_override
                infos = tuple(
                    item if isinstance(item, LiveSessionInfo) else info_from_record(item, now=now[0])
                    for item in src
                )
                component.handle(
                    IoCompleted(
                        op_id=eff.op_id,
                        value=DiscoverSessionsResult(status="ready", rows=infos, observed_at=now[0]),
                        ts=now[0],
                    ),
                    observer,
                    lambda _e: None,
                )
        return effects

    refresh(SessionReady(ts=now[0]))
    text = component.render(observer).text
    return component, observer, now, requests, refresh, text


def test_per_instance_current_rlm_liveness_and_allowlisted_metadata(tmp_path):
    history = HistorySpy()
    rows = [record("two"), record("one"), record("current", CURRENT, kind="rlm"),
            record("old", observed=1), record("future", observed=101),
            record("unverified", health=None), record("dead", health="dead", observed=1),
            record("retired", status="stopped"),
            replace(record("safe"), metadata={"password": "NEVER-DISPLAY-THIS"})]
    _, _, _, requests, _, text = opened(tmp_path, rows, history=history)
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
    assert len(history.calls[-1]) == 2 and len(requests) == 1


@pytest.mark.parametrize("supervisor_id", ["", "runtime-supervisor"])
@pytest.mark.parametrize(("kind", "expected"), [
    ("session", "not indicated"),
    ("default", "not indicated"),
    ("rlm", "yes"),
])
def test_child_label_uses_kind_not_process_supervision(tmp_path, kind, expected, supervisor_id):
    rows = [record(kind=kind, supervisor_id=supervisor_id)]
    _, _, _, _, _, text = opened(tmp_path, rows)
    assert f"Kind: {kind}\nChild/RLM: {expected}\n" in text


@pytest.mark.parametrize(("endpoint", "expected"), [
    ("ws://127.0.0.1:8123/agent", "ws://127.0.0.1:8123/agent"),
    ("wss://user:SECRET@example.test:8123/agent?token=SECRET#SECRET",
     "wss://example.test:8123/agent"),
    ("ws://[::1]:8123/agent?secret=value", "ws://[::1]:8123/agent"),
    ("ws://user:SECRET@example.test/agent", "ws://example.test/agent"),
    ("ws://[broken/SECRET", "(unavailable: malformed advertised endpoint)"),
    ("ws://example.test:bad/SECRET", "(unavailable: malformed advertised endpoint)"),
    ("https://example.test/SECRET", "(unavailable: malformed advertised endpoint)"),
])
def test_endpoint_sanitization(tmp_path, endpoint, expected):
    rec = replace(record("ep"), websocket_url=endpoint)
    _, _, _, _, _, text = opened(tmp_path, [rec])
    assert f"Websocket endpoint: {expected}\n" in text
    assert "SECRET" not in text


def test_authoritative_context_override_and_attached_store_roots(tmp_path):
    component, observer, now, requests, refresh, _ = opened(tmp_path)
    override = tmp_path / "override"
    observer._agent_zoo_context["project_store_root"] = str(override)
    observer._agent_zoo_context["local_state_root"] = str(tmp_path / "local-root")
    observer.agent_db = SimpleNamespace(store=SimpleNamespace(root=tmp_path / "wrong"))
    refresh()
    assert requests[-1].store_root == str(override)
    assert requests[-1].local_state_root == str(tmp_path / "local-root")
    assert all(req.store_root != str(tmp_path / "wrong") for req in requests)
    assert "Registry: empty" in component.render(observer).text

    del observer._agent_zoo_context["project_store_root"]
    attached = tmp_path / "attached"
    observer.agent_db.store.root = attached
    refresh()
    assert requests[-1].store_root == str(attached)
    assert "Registry: empty" in component.render(observer).text

    observer._agent_zoo_context["project_store_root"] = "relative/path"
    text = component.render(observer).text
    assert "must be absolute" in text
    assert not (tmp_path / "wrong").exists()
    assert not override.exists()
    assert not attached.exists()


def test_missing_context_absent_empty_unavailable_and_unsupported_are_distinct(tmp_path):
    component = pm.ProjectLiveSessionBuffer(clock=lambda: 100.0)
    observer = state(tmp_path / "project")
    observer._agent_zoo_context = {}
    component.handle(SessionReady(ts=100.0), observer, lambda _ev: None)
    assert observer.io.take_outbox() == []
    assert "Registry: missing-context" in component.render(observer).text

    # Real AgentZooSessionDriver never creates missing directories.
    missing = tmp_path / "missing"
    observer._agent_zoo_context = dict(project_name="absent", project_dir=str(missing))
    dispatch_via_driver(component, observer, RenderRequested(), AgentZooSessionDriver(), ts=100.0)
    assert not missing.exists()
    assert "Registry: empty (successful read; zero instances)" in component.render(observer).text

    # Driver-reported absent status is rendered distinctly.
    observer._agent_zoo_context = dict(project_name="driver-absent", project_store_root=str(tmp_path / "gone"))
    component.handle(RenderRequested(ts=100.0), observer, lambda _ev: None)
    eff = observer.io.take_outbox()[0]
    observer.io.complete(eff.op_id)
    component.handle(
        IoCompleted(
            op_id=eff.op_id,
            value=DiscoverSessionsResult(status="absent", error="coordination store does not exist"),
            ts=100.0,
        ),
        observer,
        lambda _ev: None,
    )
    assert "Registry: absent" in component.render(observer).text
    assert "No successful registry snapshot" in component.render(observer).text

    # Existing empty store via real AgentZooSessionDriver.
    store = RecordStore(tmp_path / "empty")
    observer._agent_zoo_context = dict(project_name="empty", project_store_root=str(store.root))
    dispatch_via_driver(component, observer, RenderRequested(), AgentZooSessionDriver(), ts=100.0)
    assert "Registry: empty (successful read; zero instances)" in component.render(observer).text

    # Malformed store -> IoFailed -> unavailable, never empty.
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "format.json").write_text("{bad JSON")
    observer._agent_zoo_context = dict(project_name="malformed", project_store_root=str(malformed))
    dispatch_via_driver(component, observer, RenderRequested(), AgentZooSessionDriver(), ts=100.0)
    text = component.render(observer).text
    assert "Registry: unavailable" in text
    assert "No successful registry snapshot" in text

    # Unsupported when runtime IO is unavailable.
    no_io = SimpleNamespace(
        _agent_zoo_context=dict(project_name="unsupported", project_store_root=str(store.root)),
        buffer_manager=BufferManager(),
    )
    component.handle(RenderRequested(ts=100.0), no_io, lambda _ev: None)
    assert "Registry: unsupported" in component.render(no_io).text


def test_failed_refresh_preserves_snapshot_time_recomputes_health_and_recovers(tmp_path):
    component, observer, now, requests, refresh, text = opened(tmp_path, [record()])
    old_read_time = next(line for line in text.splitlines() if line.startswith("Registry Read"))

    # Within 5s throttle window -> no new DiscoverSessions submitted.
    now[0] = 104.0
    assert refresh(RenderRequested(ts=104.0)) == []
    assert len(requests) == 1

    # Past 5s throttle -> refresh fires and fails with secret-bearing exception.
    now[0] = 130.0
    refresh(RenderRequested(ts=130.0), error=OSError("SECRET error payload must not leak"))
    text = component.render(observer).text
    assert "Registry: unavailable" in text
    assert "Last-good snapshot retained (stale)" in text
    assert old_read_time in text
    assert "confirmed alive: 0; unknown: 1" in text
    assert "SECRET" not in text
    assert len(requests) == 2

    # Retry cooldown (5s) is respected via event timestamps.
    for delta in (0.5, 1.0, 2.0, 4.9):
        now[0] = 130.0 + delta
        assert refresh(RenderRequested(ts=now[0])) == []
    assert len(requests) == 2

    # Past cooldown -> recovers with fresh observation.
    now[0] = 136.0
    refresh(RenderRequested(ts=136.0), rows_override=[record(observed=136)])
    text = component.render(observer).text
    assert "Registry: ready." in text
    assert "confirmed alive: 1" in text and old_read_time not in text


def test_buffer_manager_same_step_cache_readonly_and_pickle_safe_provider(tmp_path):
    component, observer, now, _, _, _ = opened(tmp_path, [record()])
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
    component.handle(SessionReady(ts=now[0]), observer, lambda _ev: None)
    assert restored.resolve_for_read(observer, "live_sessions").readonly


def test_in_flight_discovery_is_nonblocking_deduplicated_and_uses_state_io(tmp_path):
    root = tmp_path / "project"
    component = pm.ProjectLiveSessionBuffer(clock=lambda: 100.0)
    assert not hasattr(component, "_worker")
    assert not hasattr(component, "_work")
    observer = state(root)

    # First event submits exactly one DiscoverSessions op through state.io.
    component.handle(SessionReady(ts=100.0), observer, lambda _ev: None)
    effects = observer.io.take_outbox()
    assert len(effects) == 1
    eff = effects[0]
    assert isinstance(eff, SubmitOp)
    assert isinstance(eff.request, DiscoverSessions)
    assert eff.request.include_stale is True
    assert eff.request.probe_local_processes is False
    assert "Registry: pending" in component.render(observer).text

    # Subsequent events while in-flight do NOT duplicate the request.
    for ts in (100.5, 102.0, 110.0):
        component.handle(RenderRequested(ts=ts), observer, lambda _ev: None)
        assert observer.io.take_outbox() == []
        assert "Registry: pending" in component.render(observer).text

    # Completing via IoCompleted populates the snapshot.
    observer.io.complete(eff.op_id)
    component.handle(
        IoCompleted(
            op_id=eff.op_id,
            value=DiscoverSessionsResult(
                status="ready",
                rows=(info_from_record(record(), now=100.0),),
                observed_at=100.0,
            ),
            ts=100.5,
        ),
        observer,
        lambda _ev: None,
    )
    assert "Registry: ready." in component.render(observer).text

    # External cancellation of a pending op recovers cleanly on the next event.
    component.handle(TurnSettled(ts=106.0), observer, lambda _ev: None)
    eff2 = observer.io.take_outbox()[0]
    observer.io.cancel(eff2.op_id)
    observer.io.take_outbox()
    component.handle(RenderRequested(ts=106.1), observer, lambda _ev: None)
    eff3 = observer.io.take_outbox()
    assert len(eff3) == 1 and eff3[0].op_id != eff2.op_id


def test_private_registry_read_never_loads_history_probes_or_mutates(tmp_path, monkeypatch):
    import socket
    import tmux_pilot.process_identity as process_identity
    import agent_zoo.drivers.sessions as sessions_mod

    store = RecordStore(tmp_path / "private-store")
    now = time.time()
    row = record(observed=now - 1.0)
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
    monkeypatch.setattr(sessions_mod, "observe_process", forbidden)
    monkeypatch.setattr(sessions_mod, "capture_process_identity", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(RecordStore, "put", forbidden)
    monkeypatch.setattr(RecordStore, "update", forbidden)

    observer = state(tmp_path / "project-without-catalog")
    observer._agent_zoo_context["project_store_root"] = str(store.root)
    component = pm.ProjectLiveSessionBuffer(history=HistorySpy(), clock=lambda: now)
    dispatch_via_driver(component, observer, SessionReady(), AgentZooSessionDriver(), ts=now)
    text = component.render(observer).text
    assert "Registry: ready." in text
    assert "confirmed alive: 1" in text
    assert "ses_344655c1_full" in text
    assert contents() == before
    assert not (tmp_path / "project-without-catalog").exists()


def test_row_limit_and_disabled_history(tmp_path):
    component, observer, _, _, _, _ = opened(tmp_path, [record("one"), record("two")])
    component.ROW_LIMIT = 1
    text = component.render(observer).text
    assert "Rows shown: 1 of 2 active candidates; truncated: 1" in text
    assert "History: disabled" in text
    assert "ses_344655c1" not in text
    assert "## Instance: one" in text and "## Instance: two" not in text


def test_in_flight_limit_is_bounded_and_projects_do_not_share_snapshots(tmp_path):
    component = pm.ProjectLiveSessionBuffer(clock=lambda: 100.0)
    observer = state(tmp_path / "project")
    for i in range(component.MAX_IN_FLIGHT + 1):
        root = tmp_path / f"store-{i}"
        observer._agent_zoo_context["project_store_root"] = str(root)
        component.handle(RenderRequested(ts=100.0), observer, lambda _ev: None)
        text = component.render(observer).text
        assert "No successful registry snapshot" in text or "Registry: pending" in text
    assert "Registry: busy" in text
    assert len(component._pending_ops) == component.MAX_IN_FLIGHT
    assert sum(s["pending"] for s in component._snapshots.values()) == component.MAX_IN_FLIGHT


def test_corrupt_live_record_is_not_an_empty_registry(tmp_path):
    store = RecordStore(tmp_path / "private-store")
    store.put("live", "broken", {"session_id": FIRST, "websocket_url": "ws://host/path"})
    observer = state(tmp_path / "project")
    observer._agent_zoo_context["project_store_root"] = str(store.root)
    component = pm.ProjectLiveSessionBuffer()
    dispatch_via_driver(component, observer, SessionReady(), AgentZooSessionDriver(), ts=100.0)
    text = component.render(observer).text
    assert "Registry: unavailable" in text
    assert "No successful registry snapshot" in text
    assert "Registry: empty" not in text


def test_future_dead_evidence_is_unknown_not_excluded(tmp_path):
    _, _, _, _, _, text = opened(tmp_path, [record("future-dead", health="dead", observed=101)])
    assert "Active candidates: 1; confirmed alive: 0; unknown: 1" in text
    assert "## Instance: future-dead" in text
    assert "Excluded: 0 terminal lifecycle; 0 other confirmed-stopped instances" in text


def test_real_history_alias_is_issued_without_catalog_or_transcript_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registry discovery eagerly loaded a transcript")

    monkeypatch.setattr(session_load, "load_session", forbidden)
    history = pm.ProjectSessionBuffers(clock=lambda: 100)
    _, observer, _, _, _, text = opened(tmp_path, [record()], history=history)
    assert "ses_344655c1 | ses_344655c1_full" in text
    assert not (Path(observer._agent_zoo_context["project_dir"]) / "sessions").exists()


def test_simulation_runtime_end_to_end_with_sim_session_driver(tmp_path):
    sim_driver = SimSessionDriver(default_delay=0.1)
    sim_ref = [None]
    component = pm.ProjectLiveSessionBuffer(clock=lambda: sim_ref[0].now)
    pipeline = Pipeline([IoFence(), component])
    session = Session(pipeline=pipeline)
    session.state.step = 1
    session.state._session_id = CURRENT
    session.state._agent_zoo_context = dict(
        project_name="sim-demo",
        project_store_root=str(tmp_path / "sim-store"),
    )
    sim = SimulationRuntime(session, seed=42)
    sim_ref[0] = sim
    sim_driver.register(sim)

    # Seed one alive instance observed at t=100.0 with a credentialed WS URL.
    sim.now = 100.0
    from agent_zoo.drivers.sessions import _SimSession

    sim_driver.sessions["inst-sim-1"] = _SimSession(
        session_id=FIRST,
        instance_id="inst-sim-1",
        pid=4321,
        owned=False,
        process_identity=ProcessIdentity("sim-host", "sim-boot", "sim-ns", 4321, "tok-1"),
        websocket_url="ws://user:TOPSECRET@127.0.0.1:41000/session?token=TOPSECRET",
        status="running",
        title="Simulated worker",
        kind="rlm",
        started_at=95.0,
        updated_at=99.5,
        observation_state=ObservationState.ALIVE,
        observation_reason="sim-heartbeat",
    )

    sim.inject(SessionReady())
    sim.run_until_idle()
    text = session.state.buffer_manager.resolve_for_read(session.state, "live_sessions").text
    assert "Registry: ready." in text
    assert "confirmed alive: 1; unknown: 0" in text
    assert "ws://127.0.0.1:41000/session" in text
    assert "TOPSECRET" not in text
    assert "Child/RLM: yes" in text

    # Advance 20s without a new observation -> health degrades honestly to unknown.
    sim.now = 120.0
    session.state.step += 1
    sim.inject(RenderRequested())
    sim.run_until_idle()
    stale_text = session.state.buffer_manager.resolve_for_read(session.state, "live_sessions").text
    assert "confirmed alive: 0; unknown: 1" in stale_text
    assert "process state: unknown" in stale_text
