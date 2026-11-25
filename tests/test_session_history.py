"""Canonical Sans-I/O history contracts via ProjectHistoryDriver + ProjectSessionBuffers."""
from __future__ import annotations

import functools
import json
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from agent_utils.core.events import IoCompleted, IoFailed, IoInterrupted, RenderRequested, SessionRestored
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import Pipeline, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, State, ToolCall
from agent_utils.features.buffers import BufferManager, buffer_editing_feature
from agent_utils.features.core import core_feature
from agent_utils.storage.repository import StoredState
from agent_utils.storage.serialization import restore_session_document
from agent_zoo.core.feature_state import session_state
from agent_zoo.events import SessionReady
from agent_zoo.storage import load as session_load, projects
from agent_zoo.storage.load import LoadedSession, PendingSessionLoad, SessionLoader
from agent_zoo.storage.serialization import SESSION_PERSISTENCE_EXCLUDED_ATTRS
from agent_zoo.storage.store import SessionPersistence
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io

import project_memory as pm

FIRST = "344655c1-1111-1111-1111-111111111111"
SECOND = "97568885-2222-2222-2222-222222222222"
CURRENT = "aaaaaaaa-0000-0000-0000-000000000000"
ALIAS = "ses_344655c1"


@pytest.fixture(autouse=True)
def private_environment(tmp_path, monkeypatch):
    for name, path in (
        ("AGENT_ZOO_HOME", tmp_path / "home"),
        ("AGENT_ZOO_STATE_ROOT", tmp_path / "home"),
        ("AGENT_ZOO_LOCAL_STATE_ROOT", tmp_path / "local"),
        ("AZO_PROJECT_MEMORY_CONFIG", tmp_path / "missing.yaml"),
    ):
        monkeypatch.setenv(name, str(path))
    monkeypatch.setattr(projects, "_STATE_ROOT", tmp_path / "home")
    loader = SessionLoader()
    monkeypatch.setattr(session_load, "_loader", loader)
    yield
    assert loader.close(2)


def row(sid=FIRST, **extra):
    return dict(
        session_id=sid,
        title="Known title",
        description="Known description",
        created_at=1,
        updated_at=2,
        kind="",
        **extra,
    )


def catalog(root, rows):
    path = root / "sessions" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def state(root):
    result = State(token_budget=10000)
    result.step = 1
    session_state(result).session_id = CURRENT
    session_state(result).project_name = "demo"
    result.buffer_manager = BufferManager()
    result._agent_zoo_context = dict(
        project_name="demo",
        project_dir=str(root),
        local_state_root=str(root / "local"),
    )
    return result


def pump_history_io(component, observer, *, clock=time.time, driver=None):
    """Execute all pending ProjectHistoryDriver requests from observer.io and deliver results."""
    effects = observer.io.take_outbox()
    if not effects:
        return []
    own_driver = driver is None
    drv = driver or pm.ProjectHistoryDriver(clock=clock)
    inbox: queue.Queue = queue.Queue()
    if own_driver:
        drv.start(inbox.put)
    try:
        handled = []
        for eff in effects:
            if not isinstance(eff, SubmitOp) or not drv.handles(eff.request):
                continue
            handled.append(eff)
            drv.submit(eff.op_id, eff.request)
            io_ev = inbox.get(timeout=5.0)
            observer.io.complete(eff.op_id)
            component.handle(io_ev, observer, lambda _ev: None)
        return handled
    finally:
        if own_driver:
            drv.close()


def opened(tmp_path, rows=None):
    root = tmp_path / "project"
    catalog(root, rows if rows is not None else [row()])
    now = [100.0]
    component = pm.ProjectSessionBuffers(clock=lambda: now[0])
    observer = state(root)
    component.handle(SessionReady(), observer, lambda _ev: None)
    pump_history_io(component, observer, clock=lambda: now[0])
    assert "Inventory: ready" in component.render_index(observer).text
    return component, observer, root, now


def loaded(sid=FIRST, *, commit="first", source="local"):
    document = {
        "entries": [
            {"index": 4, "messages": [{"role": "user", "content": "Old question"}]},
            {"index": 5, "messages": [{"role": "assistant", "content": "Old answer"}]},
            {"index": 8, "messages": [{"role": "user", "content": [{"text": "New question"}]}]},
            {
                "index": 9,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "tool preamble",
                        "tool_calls": [{"id": "tool"}],
                    }
                ],
            },
            {
                "index": 10,
                "system_generated": True,
                "messages": [{"role": "user", "content": "hidden system entry"}],
            },
            {"index": 11, "messages": [{"role": "assistant", "content": "New answer"}]},
        ]
    }
    saved = StoredState(
        Path("/private/repository"),
        Path("/private/journal"),
        sid,
        "instance",
        commit,
        None,
        10.0,
        document,
    )
    return LoadedSession(saved, source, "local-unchecked", "shared selection is unchecked")


def pending(tmp_path, sid=FIRST, value=None):
    handle = PendingSessionLoad(tmp_path / sid / "session.json", sid, None, None, tmp_path / "local")
    if value is not None:
        handle._value = value
        handle._done.set()
    return handle


def load_view(component, observer, alias, *, clock=time.time):
    view = component.render_session(observer, alias)
    if view is not None and view.path.endswith("/pending"):
        pump_history_io(component, observer, clock=clock)
        view = component.render_session(observer, alias)
    return view


def test_sans_io_audit_zero_partials_and_driver_contract(tmp_path):
    assert audit_module_sans_io(Path(__file__).resolve().parents[1] / "src/project_memory.py") == []
    assert pm.ProjectHistoryDriver.durability == "ephemeral"
    assert pm.ProjectHistoryDriver.request_types == (
        pm.LoadProjectCatalogRequest,
        pm.LoadProjectSessionRequest,
    )
    assert not hasattr(pm.ProjectSessionBuffers, "bind_session")

    session = Session()
    builder = PipelineBuilder().add(core_feature()).add(buffer_editing_feature())
    pm.register_features(builder, session=session, config={"project_memory": {"project_sessions": True}})
    session.pipeline = builder.build(suppress_warnings=True)
    session.ensure_initialized()
    session_state(session.state).session_id = CURRENT
    session.state._agent_zoo_context = dict(project_name="demo", project_dir=str(tmp_path / "project"))
    assert_pipeline_sans_io(
        session.pipeline,
        session=session,
        events=[SessionReady(), RenderRequested(), SessionRestored()],
    )
    # Zero functools.partial closures attached to state.__dict__!
    for k, v in session.state.__dict__.items():
        assert not isinstance(v, functools.partial), f"Unexpected partial on state.{k}"


def test_fast_first_view_projects_complete_entries_and_cached_view_is_immediate(tmp_path, monkeypatch):
    component, observer, _, now = opened(tmp_path)
    handle = pending(tmp_path)
    calls = []

    def request(locator, **kwargs):
        calls.append(kwargs)
        assert kwargs["timeout_s"] == 0
        handle._value = loaded()
        handle._done.set()
        return handle

    monkeypatch.setattr(session_load, "load_session", request)
    first_pending = component.render_session(observer, ALIAS)
    assert first_pending.path.endswith("/pending")
    pump_history_io(component, observer, clock=lambda: now[0])
    view = component.render_session(observer, ALIAS)
    assert view.path.endswith("/commits/first")
    for text in (
        "USER [entry 8]",
        "ASSISTANT [entry 11]",
        "Source Commit: first",
        "Source Instance: instance",
        "Source Format: journal",
        "Freshness: local-unchecked",
        "not journal file lines",
        "shared selection is unchecked",
    ):
        assert text in view.text
    assert view.text.index("New question") < view.text.index("Old question")
    assert "tool preamble" not in view.text and "hidden system entry" not in view.text
    # Cached read emits zero IO!
    assert component.render_session(observer, ALIAS).text == view.text
    assert observer.io.take_outbox() == []
    assert len(calls) == 1


def test_slow_session_runs_in_driver_thread_and_quiesce_emits_io_interrupted(tmp_path, monkeypatch):
    component, observer, root, now = opened(tmp_path, [row(), row(SECOND)])
    entered, release, calls = threading.Event(), threading.Event(), []

    def read(job):
        calls.append(job.session_id)
        assert threading.current_thread() is not threading.main_thread()
        if job.session_id == FIRST:
            entered.set()
            release.wait(2.0)
        return loaded(job.session_id)

    monkeypatch.setattr(SessionLoader, "_read", staticmethod(read))
    drv = pm.ProjectHistoryDriver(clock=lambda: now[0])
    inbox: queue.Queue = queue.Queue()
    drv.start(inbox.put)
    try:
        # Opening FIRST submits LoadProjectSessionRequest without blocking pipeline thread!
        started = time.monotonic()
        view = component.render_session(observer, ALIAS)
        assert time.monotonic() - started < 0.1
        assert view.path.endswith("/pending")
        assert "Known title" in view.text and "Known description" in view.text
        eff1 = observer.io.take_outbox()[0]
        drv.submit(eff1.op_id, eff1.request)
        assert entered.wait(2.0)

        # Duplicate read while in-flight does not re-submit.
        assert component.render_session(observer, ALIAS).path.endswith("/pending")
        assert observer.io.take_outbox() == []

        # Opening SECOND succeeds concurrently without being blocked by FIRST.
        view2_pending = component.render_session(observer, "ses_97568885")
        assert view2_pending.path.endswith("/pending")
        eff2 = observer.io.take_outbox()[0]
        drv.submit(eff2.op_id, eff2.request)
        io_ev2 = inbox.get(timeout=4.0)
        observer.io.complete(eff2.op_id)
        component.handle(io_ev2, observer, lambda _ev: None)
        assert component.render_session(observer, "ses_97568885").path.endswith("/commits/first")

        # Quiescing interrupts the still-in-flight FIRST op cleanly.
        interrupted = drv.quiesce(policy="terminate")
        assert len(interrupted) == 1
        assert isinstance(interrupted[0], IoInterrupted)
        assert interrupted[0].op_id == eff1.op_id
        observer.io.complete(eff1.op_id)
        component.handle(interrupted[0], observer, lambda _ev: None)
    finally:
        release.set()
        drv.close()

    # After interruption, FIRST can be re-requested cleanly and finishes.
    final_view = load_view(component, observer, ALIAS, clock=lambda: now[0])
    assert final_view.path.endswith("/commits/first")


def test_unattached_context_is_pending_not_missing():
    observer = State(token_budget=1000)
    observer.buffer_manager = BufferManager()
    component = pm.ProjectSessionBuffers()
    component.handle(SessionReady(), observer, lambda _ev: None)
    assert observer.io.take_outbox() == []
    assert "Inventory: pending" in component.render_index(observer).text
    assert "context is not attached" in component.render_session(observer, ALIAS).text
    assert component.render_session(observer, "ses_not-valid") is None


def test_catalog_is_async_filtered_retriable_and_driver_owned(tmp_path, monkeypatch):
    root = tmp_path / "project"
    rows = [
        row(),
        row(SECOND),
        row(CURRENT),
        dict(row("bbbbbbbb-1234"), kind="rlm-child"),
        row("../invalid"),
        row(),
    ]
    path = catalog(root, rows)
    read_orig = pm.ProjectHistoryDriver._read_catalog_snapshot
    calls = []

    def tracked(*args):
        calls.append(threading.current_thread())
        return read_orig(*args)

    monkeypatch.setattr(pm.ProjectHistoryDriver, "_read_catalog_snapshot", staticmethod(tracked))
    monkeypatch.setattr(session_load, "load_session", lambda *a, **kw: pytest.fail("catalog read a document"))
    now = [100.0]
    component = pm.ProjectSessionBuffers(clock=lambda: now[0])
    observer = state(root)

    component.handle(SessionReady(), observer, lambda _ev: None)
    assert "Inventory: pending" in component.render_index(observer).text
    pump_history_io(component, observer, clock=lambda: now[0])

    index = component.render_index(observer).text
    assert "Inventory: ready" in index
    assert "Sessions: 2" in index and CURRENT not in index and "bbbbbbbb" not in index
    assert all(thread is not threading.current_thread() for thread in calls)

    path.write_text("{bad json")
    now[0] += 6
    component.render_index(observer)
    pump_history_io(component, observer, clock=lambda: now[0])
    unavail = component.render_index(observer).text
    assert "Inventory: unavailable" in unavail
    assert FIRST in unavail

    # Within retry window -> no new IO submitted.
    count = len(calls)
    for _ in range(10):
        component.render_index(observer)
    assert observer.io.take_outbox() == []
    assert len(calls) == count

    path.unlink()
    now[0] += 6
    component.render_index(observer)
    pump_history_io(component, observer, clock=lambda: now[0])
    assert "Inventory: absent" in component.render_index(observer).text


def test_over_32_pins_survive_removal_collision_and_byte_cache_eviction(tmp_path, monkeypatch):
    rows = [row()] + [row(f"{i:08x}-1234") for i in range(40)]
    component, observer, root, now = opened(tmp_path, rows)
    monkeypatch.setattr(pm, "SESSION_PROJECTION_CACHE_BYTES", 2500)
    requests = []

    def request(locator, **kwargs):
        requests.append(kwargs)
        return loaded(
            kwargs["session_id"],
            commit=kwargs["commit_id"] or "first",
            source="shared" if kwargs["commit_id"] else "local",
        )

    monkeypatch.setattr(session_load, "load_session", request)
    original = load_view(component, observer, ALIAS, clock=lambda: now[0])
    for alias in pm._session_buffer_ids(rows).values():
        assert "/commits/" in load_view(component, observer, alias, clock=lambda: now[0]).path
    rt = pm._runtime(observer)
    assert len(rt.pins) == 41 and rt.text_bytes <= 2500
    assert all(scope[1] != ALIAS for scope in rt.texts)

    collision = "344655c1-2222-2222-2222-222222222222"
    catalog(root, [dict(row(collision), title="New collision")])
    now[0] += 6
    component.render_index(observer)
    pump_history_io(component, observer, clock=lambda: now[0])
    index = component.render_index(observer)
    assert "New collision" in index.text
    assert "Buffer: ses_344655c12" in index.text

    again = load_view(component, observer, ALIAS, clock=lambda: now[0])
    assert (again.path, again.text) == (original.path, original.text)
    assert requests[-1]["commit_id"] == "first"
    assert rt.text_bytes == sum(size for _, size in rt.texts.values())


@pytest.mark.parametrize("budget", [0, 20])
def test_oversize_text_is_not_retained_and_first_receipt_is_stable(tmp_path, monkeypatch, budget):
    component, observer, _, now = opened(tmp_path)
    monkeypatch.setattr(pm, "SESSION_PROJECTION_CACHE_BYTES", budget)
    replies = [loaded(), replace(loaded(), source="shared", freshness="exact", warning=None)]
    commits = []

    def request(*args, **kwargs):
        commits.append(kwargs["commit_id"])
        return replies.pop(0)

    monkeypatch.setattr(session_load, "load_session", request)
    first = load_view(component, observer, ALIAS, clock=lambda: now[0])
    rt = pm._runtime(observer)
    assert not rt.texts and not rt.session_requests and not rt.one_shot_texts
    now[0] += 100
    second = load_view(component, observer, ALIAS, clock=lambda: now[0])
    assert first.text == second.text and commits == [None, "first"]
    assert not rt.texts and not rt.session_requests


def test_pending_and_error_requests_are_bounded_with_explicit_retry(tmp_path, monkeypatch):
    rows = [row(f"{i:08x}-1234") for i in range(40)]
    component, observer, _, now = opened(tmp_path, rows)
    aliases = list(pm._session_buffer_ids(rows).values())

    for alias in aliases[:8]:
        assert component.render_session(observer, alias).path.endswith("/pending")
    assert "capacity" in component.render_session(observer, aliases[8]).text
    rt = pm._runtime(observer)
    assert len(rt.session_requests) == 8

    # Complete first pending op -> frees slot for aliases[8].
    outbox = observer.io.take_outbox()
    first_eff = outbox[0]
    observer.io.complete(first_eff.op_id)
    req0 = first_eff.request
    component.handle(
        IoCompleted(
            op_id=first_eff.op_id,
            value=pm.ProjectSessionResult(
                project_name=req0.project_name,
                project_dir=req0.project_dir,
                alias=req0.alias,
                full=req0.full,
                commit_id="c1",
                instance_id="inst",
                source_format="journal",
                generated_at=now[0],
                receipt="receipt",
                text="ready text",
            ),
        ),
        observer,
        lambda _ev: None,
    )
    assert component.render_session(observer, aliases[8]).path.endswith("/pending")
    assert len(rt.session_requests) == 8

    # Fail remaining pending ops -> bounded error + retry cooldown.
    outbox.extend(observer.io.take_outbox())
    for eff in outbox[1:]:
        observer.io.complete(eff.op_id)
        component.handle(
            IoFailed(op_id=eff.op_id, error=OSError("pin capacity buffer is pinned " + "x" * 600)),
            observer,
            lambda _ev: None,
        )
    view_err = component.render_session(observer, aliases[1])
    assert view_err.path.endswith("/unavailable")
    assert len(view_err.text) < 400
    assert observer.io.take_outbox() == []

    # After cooldown -> retries cleanly.
    now[0] += 6
    assert component.render_session(observer, aliases[1]).path.endswith("/pending")
    assert len(observer.io.take_outbox()) == 1


def test_project_roots_isolate_identical_aliases(tmp_path, monkeypatch):
    component, observer, _, now = opened(tmp_path)
    monkeypatch.setattr(
        session_load,
        "load_session",
        lambda locator, **kw: loaded(commit=str(locator.parent.parent.parent.name)),
    )
    first = load_view(component, observer, ALIAS, clock=lambda: now[0])
    other = tmp_path / "other-project"
    catalog(other, [row()])
    observer._agent_zoo_context["project_dir"] = str(other)
    component.handle(SessionReady(), observer, lambda _ev: None)
    pump_history_io(component, observer, clock=lambda: now[0])
    assert "Inventory: ready" in component.render_index(observer).text
    second = load_view(component, observer, ALIAS, clock=lambda: now[0])
    assert first.path != second.path and len(pm._runtime(observer).pins) == 2


@pytest.mark.parametrize("full", [False, True])
def test_real_journal_registration_readonly_and_persistence_restore(tmp_path, full):
    projects.ensure_project("demo")
    root = tmp_path / "home" / "projects" / "demo"
    local = tmp_path / "local"
    source = Session(pipeline=Pipeline(()))
    session_state(source.state).session_id = FIRST
    source.state.entries = [
        Entry(index=i, step=i, messages=[dict(role=role, content=text)])
        for i, (role, text) in enumerate(
            (("user", "Journal question"), ("assistant", "Journal answer"))
        )
    ]
    locator = root / "sessions" / FIRST / "session.json"
    writer = SessionPersistence(
        locator, session_id=FIRST, instance_id="source", local_state_root=local
    )
    writers = [writer]

    def save(session, persistence):
        receipt = persistence.save(session, exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS)
        assert receipt.local_saved
        assert persistence.wait_for_shared(receipt.revision_id, timeout_s=4).shared_saved
        return receipt.revision_id

    def viewer(document=None):
        session = Session()
        session.state.buffer_manager = BufferManager()
        builder = PipelineBuilder().add(core_feature()).add(buffer_editing_feature())
        pm.register_features(
            builder, session=session, config={"project_memory": {"project_sessions": True}}
        )
        session.pipeline = builder.build(suppress_warnings=True)
        if document is not None:
            restore_session_document(document, session.pipeline, state=session.state)
        session.ensure_initialized()
        session_state(session.state).session_id = CURRENT
        session.state._agent_zoo_context = dict(
            project_name="demo", project_dir=str(root), local_state_root=str(local)
        )
        component = next(c for c in session.pipeline if isinstance(c, pm.ProjectSessionBuffers))
        effects = session.step(SessionRestored() if document is not None else SessionReady())
        assert session.state.buffer_manager.resolve_for_read(session.state, "live_sessions").readonly
        # Run ProjectHistoryDriver on submitted catalog request
        drv = pm.ProjectHistoryDriver()
        inbox: queue.Queue = queue.Queue()
        drv.start(inbox.put)
        try:
            for eff in effects:
                if isinstance(eff, SubmitOp) and drv.handles(eff.request):
                    drv.submit(eff.op_id, eff.request)
                    session.step(inbox.get(timeout=4.0))
        finally:
            drv.close()
        assert "Inventory: ready" in component.render_index(session.state).text
        return session

    try:
        commit = save(source, writer)
        projects.register_session("demo", FIRST, title="Journal history")
        session = viewer()
        alias = ALIAS + ("_full" if full else "")

        def read_and_pump(sess):
            sess.state.step += 1
            v = sess.state.buffer_manager.resolve_for_read(sess.state, alias)
            if v.path.endswith("/pending"):
                drv = pm.ProjectHistoryDriver()
                inbox: queue.Queue = queue.Queue()
                drv.start(inbox.put)
                try:
                    for eff in sess.state.io.take_outbox():
                        if isinstance(eff, SubmitOp) and drv.handles(eff.request):
                            drv.submit(eff.op_id, eff.request)
                            sess.step(inbox.get(timeout=4.0))
                finally:
                    drv.close()
                sess.state.step += 1
                v = sess.state.buffer_manager.resolve_for_read(sess.state, alias)
            return v

        view = read_and_pump(session)
        assert f"Source Commit: {commit}" in view.text and "Source Format: journal" in view.text
        assert "Journal answer" in view.text and not locator.exists()
        assert view.readonly
        with pytest.raises(ValueError, match="readonly"):
            session.state.buffer_manager.require_writable(alias)

        target = root / "sessions" / CURRENT / "session.json"
        persistence = SessionPersistence(
            target, session_id=CURRENT, instance_id="viewer", local_state_root=local
        )
        writers.append(persistence)
        save(session, persistence)
        restored = session_load.load_session(target, session_id=CURRENT, local_state_root=local, timeout_s=2)
        assert isinstance(restored, LoadedSession)
        reopened = viewer(restored.state.document)
        view2 = read_and_pump(reopened)
        assert "Journal answer" in view2.text and f"Source Commit: {commit}" in view2.text
    finally:
        for persistence in writers:
            persistence.close()


@pytest.mark.parametrize("full_first", [False, True])
def test_full_and_compact_share_pin_in_either_order(tmp_path, monkeypatch, full_first):
    component, observer, root, now = opened(tmp_path)
    commits = []

    def request(locator, **kwargs):
        commits.append(kwargs["commit_id"])
        return loaded(commit=kwargs["commit_id"] or f"revision-{len(commits)}")

    monkeypatch.setattr(session_load, "load_session", request)
    aliases = [ALIAS + "_full", ALIAS] if full_first else [ALIAS, ALIAS + "_full"]
    views = {alias: load_view(component, observer, alias, clock=lambda: now[0]) for alias in aliases}
    assert commits == [None, "revision-1"]
    compact, full = views[ALIAS], views[ALIAS + "_full"]
    assert compact.readonly and full.readonly
    assert compact.path.endswith("/commits/revision-1")
    assert full.path.endswith("/commits/revision-1/full")
    assert f"Full buffer: {ALIAS}_full" in compact.text
    assert f"Compact buffer: {ALIAS}" in full.text
    assert "tool preamble" not in compact.text and "tool preamble" in full.text
    assert full.text.index("New question") < full.text.index("Old question")
    assert compact.text.index("New question") < compact.text.index("Old question")
    rt = pm._runtime(observer)
    assert len(rt.pins) == 1 and not rt.session_requests
    assert rt.text_bytes == sum(size for _, size in rt.texts.values())
    now[0] += 10
    assert component.render_session(observer, ALIAS + "_full").text == full.text
    # Eviction is not refresh for either projection.
    rt.texts.clear()
    rt.text_bytes = 0
    monkeypatch.setattr(pm, "SESSION_PROJECTION_CACHE_BYTES", 1)
    for alias in aliases:
        assert load_view(component, observer, alias, clock=lambda: now[0]).text == views[alias].text
    assert commits[-2:] == ["revision-1", "revision-1"]
    assert not rt.texts and rt.text_bytes == 0


def test_registry_issued_aliases_do_not_need_catalog_membership(tmp_path, monkeypatch):
    component, observer, root, now = opened(tmp_path, rows=[])
    aliases = component.issue_sessions(
        observer,
        [row(), row(), dict(row(CURRENT), kind="rlm"), {"session_id": "../bad"}, {}],
    )
    assert set(aliases) == {FIRST, CURRENT}
    monkeypatch.setattr(session_load, "load_session", lambda locator, **kw: loaded(kw["session_id"]))
    for sid, alias in aliases.items():
        assert f"Session ID: {sid}" in load_view(component, observer, alias + "_full", clock=lambda: now[0]).text
    assert "Known title" not in component.render_index(observer).text
    collision = "344655c1-2222-2222-2222-222222222222"
    added = component.issue_sessions(observer, [row(collision)])
    assert added[collision] != aliases[FIRST]
    assert component.issue_sessions(observer, [row()])[FIRST] == aliases[FIRST]


@pytest.mark.parametrize("alias", ["ses_344655c1_full_full", "ses_344655c1/full", "ses_short_full"])
def test_full_namespace_rejects_malformed_ids(tmp_path, alias):
    component, observer, _, _ = opened(tmp_path)
    assert component.render_session(observer, alias) is None


def test_registry_navigation_to_full_saved_tool_history(tmp_path):
    from agent_zoo.drivers.sessions import AgentZooSessionDriver
    from agent_zoo.host.registry import LiveProcessObservation, LiveSessionRecord
    from tmux_pilot.fs_store import RecordStore
    from tmux_pilot.process_identity import ObservationState, ProcessIdentity

    projects.ensure_project("demo")
    root = tmp_path / "home" / "projects" / "demo"
    local = tmp_path / "local"
    paths = ["/fixtures/backend-worktrees/topic", "/fixtures/tui-worktrees/topic"]
    args = json.dumps({"command": f"git worktree add {paths[0]} topic; git worktree add {paths[1]} topic"})
    call = ToolCall("make", "terminal_run", args, json.loads(args), result="Fixture checkout prepared")
    source = Session(pipeline=Pipeline(()))
    session_state(source.state).session_id = FIRST
    source.state.entries = [
        Entry(index=0, step=0, messages=[{"role": "user", "content": "Implement the feature"}]),
        Entry(
            index=1,
            step=1,
            messages=[
                {
                    "role": "assistant",
                    "content": "Preparing checkouts",
                    "tool_calls": [
                        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": args}}
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": call.result},
            ],
            tool_calls=[call],
        ),
        Entry(index=2, step=2, messages=[{"role": "assistant", "content": "Implemented; final answer omits paths"}]),
    ]
    writer = SessionPersistence(
        root / "sessions" / FIRST / "session.json",
        session_id=FIRST,
        instance_id="saved-instance",
        local_state_root=local,
    )
    try:
        receipt = writer.save(source, exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS)
        assert receipt.local_saved
        assert writer.wait_for_shared(receipt.revision_id, timeout_s=4).shared_saved
        stamp = time.time()
        live_row = LiveSessionRecord(
            session_id=FIRST,
            instance_id="live-instance",
            pid=123,
            status="running",
            websocket_url="ws://127.0.0.1:8123",
            title="Feature work",
            process_identity=ProcessIdentity("fixture-host", "fixture-boot", "fixture-ns", 123, "fixture-start"),
            process_observation=LiveProcessObservation(ObservationState.ALIVE, "fixture", stamp, 1),
        )
        RecordStore(root / "coordination").put("live", live_row.instance_id, live_row.to_value())
        observer = state(root)
        observer._agent_zoo_context["local_state_root"] = str(local)
        history = pm.ProjectSessionBuffers()
        live = pm.ProjectLiveSessionBuffer(history=history)
        history.handle(SessionReady(), observer, lambda _ev: None)
        live.handle(SessionReady(), observer, lambda _ev: None)

        sess_driver = AgentZooSessionDriver()
        hist_driver = pm.ProjectHistoryDriver()
        inbox: queue.Queue = queue.Queue()
        sess_driver.start(inbox.put)
        hist_driver.start(inbox.put)
        try:
            def drain():
                for effect in observer.io.take_outbox():
                    if not isinstance(effect, SubmitOp):
                        continue
                    drv = sess_driver if sess_driver.handles(effect.request) else hist_driver
                    drv.submit(effect.op_id, effect.request)
                    io_ev = inbox.get(timeout=4.0)
                    observer.io.complete(effect.op_id)
                    history.handle(io_ev, observer, lambda _ev: None)
                    live.handle(io_ev, observer, lambda _ev: None)

            drain()

            def read(alias):
                observer.step += 1
                v = observer.buffer_manager.resolve_for_read(observer, alias)
                if v.path.endswith("/pending"):
                    drain()
                    observer.step += 1
                    v = observer.buffer_manager.resolve_for_read(observer, alias)
                return v

            text = read("live_sessions").text
            assert "Saved history:" in text
            assert f"{ALIAS} | {ALIAS}_full" in text
            compact = read(ALIAS)
            full = read(ALIAS + "_full")
            for path in paths:
                assert path not in compact.text and path in full.text
            assert full.text.count(args) == 1
            assert "Source Instance: saved-instance" in full.text and "live-instance" in text
            assert f"Source Commit: {receipt.revision_id}" in compact.text
            assert f"Source Commit: {receipt.revision_id}" in full.text
            lines = full.text.splitlines()
            line = next(i for i, value in enumerate(lines) if "git worktree add" in value)
            assert "LOCAL FUNCTION REQUEST" in "\n".join(lines[max(0, line - 2) : line + 2])
            assert read("live_sessions").readonly and full.readonly
        finally:
            sess_driver.close()
            hist_driver.close()
    finally:
        writer.close()


def test_registry_only_session_without_saved_history_reports_unavailable(tmp_path):
    component, observer, root, now = opened(tmp_path, rows=[])
    alias = component.issue_sessions(observer, [row()])[FIRST] + "_full"
    view = load_view(component, observer, alias, clock=lambda: now[0])
    assert view is not None and view.path.endswith("/unavailable")
    assert view.readonly
    assert "No saved entries recorded" not in view.text
    assert not (root / "sessions" / FIRST / "session.json").exists()
