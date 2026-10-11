"""Canonical history contracts; all filesystem fixtures and loader pools are private."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from agent_utils import Entry, Session, State
from agent_utils.builder import PipelineBuilder, buffer_editing, core
from agent_utils.files.buffer_manager import BufferManager
from agent_utils.session import flatten
from agent_utils.session_repository import StoredState
from agent_zoo import projects, session_load
from agent_zoo.runtime_support import SESSION_PERSISTENCE_EXCLUDED_ATTRS
from agent_zoo.session_load import LoadedSession, PendingSessionLoad, SessionLoader
from agent_zoo.session_store import SessionPersistence

import project_memory as pm

FIRST = "344655c1-1111-1111-1111-111111111111"
SECOND = "97568885-2222-2222-2222-222222222222"
CURRENT = "aaaaaaaa-0000-0000-0000-000000000000"
ALIAS = "ses_344655c1"


@pytest.fixture(autouse=True)
def private_environment(tmp_path, monkeypatch):
    for name, path in (("AGENT_ZOO_HOME", tmp_path / "home"),
                       ("AGENT_ZOO_STATE_ROOT", tmp_path / "home"),
                       ("AGENT_ZOO_LOCAL_STATE_ROOT", tmp_path / "local"),
                       ("AZO_PROJECT_MEMORY_CONFIG", tmp_path / "missing.yaml")):
        monkeypatch.setenv(name, str(path))
    monkeypatch.setattr(projects, "_STATE_ROOT", tmp_path / "home")
    loader = SessionLoader()
    monkeypatch.setattr(session_load, "_loader", loader)
    yield
    assert loader.close(2)


def wait_for(fn, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(.005)
    pytest.fail("condition did not settle")


def row(sid=FIRST, **extra):
    return dict(session_id=sid, title="Known title", description="Known description",
                created_at=1, updated_at=2, kind="", **extra)


def catalog(root, rows):
    path = root / "sessions" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def state(root):
    result = State(token_budget=10000)
    result.step = 1
    result._session_id = CURRENT
    result.buffer_manager = BufferManager()
    result._agent_zoo_context = dict(project_name="demo", project_dir=str(root),
                                   local_state_root=str(root / "local"))
    return result


def opened(tmp_path, rows=None):
    root = tmp_path / "project"
    catalog(root, rows if rows is not None else [row()])
    now = [100.0]
    component, observer = pm.ProjectSessionBuffers(clock=lambda: now[0]), state(root)
    component(observer)
    wait_for(lambda: "Inventory: ready" in component.render_index(observer).text)
    return component, observer, root, now


def loaded(sid=FIRST, *, commit="first", source="local"):
    document = {"entries": [
        {"index": 4, "messages": [{"role": "user", "content": "Old question"}]},
        {"index": 5, "messages": [{"role": "assistant", "content": "Old answer"}]},
        {"index": 8, "messages": [{"role": "user", "content": [{"text": "New question"}]}]},
        {"index": 9, "messages": [{"role": "assistant", "content": "tool preamble",
                                     "tool_calls": [{"id": "tool"}]}]},
        {"index": 10, "system_generated": True,
         "messages": [{"role": "user", "content": "hidden system entry"}]},
        {"index": 11, "messages": [{"role": "assistant", "content": "New answer"}]},
    ]}
    saved = StoredState(Path("/private/repository"), Path("/private/journal"), sid,
                        "instance", commit, None, 10.0, document)
    return LoadedSession(saved, source, "local-unchecked", "shared selection is unchecked")


def pending(tmp_path, sid=FIRST, value=None):
    handle = PendingSessionLoad(tmp_path / sid / "session.json", sid, None, None, tmp_path / "local")
    if value is not None:
        handle._value = value
        handle._done.set()
    return handle


def test_fast_first_view_projects_complete_entries_and_cached_view_is_immediate(tmp_path, monkeypatch):
    component, observer, _, _ = opened(tmp_path)
    handle = pending(tmp_path)
    calls = []
    def request(locator, **kwargs):
        calls.append(kwargs)
        assert kwargs["timeout_s"] == 0
        handle._value = loaded()
        handle._done.set()
        return handle
    monkeypatch.setattr(session_load, "load_session", request)
    view = component.render_session(observer, ALIAS)
    assert view.path.endswith("/commits/first")
    for text in ("USER [entry 8]", "ASSISTANT [entry 11]", "Source Commit: first",
                 "Source Instance: instance", "Source Format: journal", "Freshness: local-unchecked",
                 "not journal file lines", "shared selection is unchecked"):
        assert text in view.text
    assert view.text.index("New question") < view.text.index("Old question")
    assert "tool preamble" not in view.text and "hidden system entry" not in view.text
    monkeypatch.setattr(component, "_observe", lambda *args: pytest.fail("cached read waited"))
    assert component.render_session(observer, ALIAS).text == view.text
    assert len(calls) == 1


def test_slow_first_view_is_bounded_and_does_not_block_sessions_or_catalog(tmp_path, monkeypatch):
    component, observer, root, now = opened(tmp_path, [row(), row(SECOND)])
    release, calls = threading.Event(), []
    def read(job):
        calls.append(job.session_id)
        assert threading.current_thread() is not threading.main_thread()
        if job.session_id == FIRST:
            assert release.wait(2)
        return loaded(job.session_id)
    monkeypatch.setattr(SessionLoader, "_read", staticmethod(read))
    try:
        started = time.monotonic()
        view = component.render_session(observer, ALIAS)
        elapsed = time.monotonic() - started
        assert .15 <= elapsed < .6
        assert view.path.endswith("/pending")
        assert "Known title" in view.text and "Known description" in view.text
        assert component.render_session(observer, "ses_97568885").path.endswith("/commits/first")
        catalog(root, [row(SECOND)])
        now[0] += 6
        wait_for(lambda: "Sessions: 1" in component.render_index(observer).text)
    finally:
        release.set()
    assert component.render_session(observer, ALIAS).path.endswith("/commits/first")
    assert calls.count(FIRST) == 1


def test_simultaneous_first_reads_share_one_handle_and_finish_within_observation(tmp_path, monkeypatch):
    component, observer, _, _ = opened(tmp_path)
    handle, entered = pending(tmp_path), threading.Event()
    calls = []
    def request(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        return handle
    monkeypatch.setattr(session_load, "load_session", request)
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(component.render_session, observer, ALIAS)
        assert entered.wait(1)
        others = [pool.submit(component.render_session, observer, ALIAS) for _ in range(7)]
        for future in others:
            assert future.result(1).path.endswith("/pending")
        handle._value = loaded()
        handle._done.set()
        assert first.result(1).path.endswith("/commits/first")
    assert len(calls) == 1


def test_projection_cpu_does_not_hold_the_catalog_or_other_session_lock(tmp_path, monkeypatch):
    component, observer, root, now = opened(tmp_path, [row(), row(SECOND)])
    entered, release = threading.Event(), threading.Event()
    project = pm._build_compact_transcript
    def blocked(session, *args, **kwargs):
        if session["session_id"] == FIRST:
            entered.set()
            assert release.wait(2)
        return project(session, *args, **kwargs)
    monkeypatch.setattr(pm, "_build_compact_transcript", blocked)
    monkeypatch.setattr(session_load, "load_session", lambda *a, **kw: loaded(kw["session_id"]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(component.render_session, observer, ALIAS)
        try:
            assert entered.wait(1)
            assert component.render_session(observer, "ses_97568885").path.endswith("/commits/first")
            catalog(root, [row(SECOND)])
            now[0] += 6
            wait_for(lambda: "Sessions: 1" in component.render_index(observer).text)
        finally:
            release.set()
        assert first.result(1).path.endswith("/commits/first")


def test_unattached_context_is_pending_not_missing():
    observer = State(token_budget=1000)
    observer.buffer_manager = BufferManager()
    component = pm.ProjectSessionBuffers()
    component(observer)
    assert "Inventory: pending" in component.render_index(observer).text
    assert "context is not attached" in component.render_session(observer, ALIAS).text
    assert component.render_session(observer, "ses_not-valid") is None


def test_catalog_is_async_filtered_retriable_and_worker_exits_idle(tmp_path, monkeypatch):
    root, entered, release = tmp_path / "project", threading.Event(), threading.Event()
    rows = [row(), row(SECOND), row(CURRENT), dict(row("bbbbbbbb-1234"), kind="rlm-child"),
            row("../invalid"), row()]
    path = catalog(root, rows)
    read, calls = pm.ProjectSessionBuffers._read_catalog_snapshot, []
    def gated(*args):
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(2)
        return read(*args)
    monkeypatch.setattr(pm.ProjectSessionBuffers, "_read_catalog_snapshot", staticmethod(gated))
    monkeypatch.setattr(session_load, "load_session", lambda *a, **kw: pytest.fail("catalog read a document"))
    now = [100.0]
    component, observer = pm.ProjectSessionBuffers(clock=lambda: now[0]), state(root)
    try:
        started = time.monotonic()
        component(observer)
        assert time.monotonic() - started < .2 and entered.wait(1)
        assert "Inventory: pending" in component.render_index(observer).text
    finally:
        release.set()
    wait_for(lambda: "Inventory: ready" in component.render_index(observer).text)
    index = component.render_index(observer).text
    assert "Sessions: 2" in index and CURRENT not in index and "bbbbbbbb" not in index
    assert all(thread is not threading.current_thread() for thread in calls)
    wait_for(lambda: component._worker is None)
    path.write_text("{bad json")
    now[0] += 6
    wait_for(lambda: "Inventory: unavailable" in component.render_index(observer).text)
    assert FIRST in component.render_index(observer).text
    count = len(calls)
    for _ in range(10):
        component.render_index(observer)
    assert len(calls) == count
    path.unlink()
    now[0] += 6
    wait_for(lambda: "Inventory: absent" in component.render_index(observer).text)
    wait_for(lambda: component._worker is None)


def test_over_32_pins_survive_removal_collision_and_byte_cache_eviction(tmp_path, monkeypatch):
    rows = [row()] + [row(f"{i:08x}-1234") for i in range(40)]
    component, observer, root, now = opened(tmp_path, rows)
    monkeypatch.setattr(pm, "SESSION_PROJECTION_CACHE_BYTES", 2500)
    requests = []
    def request(locator, **kwargs):
        requests.append(kwargs)
        return loaded(kwargs["session_id"], commit=kwargs["commit_id"] or "first",
                      source="shared" if kwargs["commit_id"] else "local")
    monkeypatch.setattr(session_load, "load_session", request)
    original = component.render_session(observer, ALIAS)
    for alias in pm._session_buffer_ids(rows).values():
        assert "/commits/" in component.render_session(observer, alias).path
    assert len(component._pins) == 41 and component._text_bytes <= 2500
    assert all(scope[1] != ALIAS for scope in component._texts)
    collision = "344655c1-2222-2222-2222-222222222222"
    catalog(root, [dict(row(collision), title="New collision")])
    now[0] += 6
    index = wait_for(lambda: (v if "New collision" in (v := component.render_index(observer)).text else None))
    assert "Buffer: ses_344655c12" in index.text
    again = component.render_session(observer, ALIAS)
    assert (again.path, again.text) == (original.path, original.text)
    assert requests[-1]["commit_id"] == "first"
    assert component._text_bytes == sum(size for _, size in component._texts.values())


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
    first = component.render_session(observer, ALIAS)
    now[0] += 100
    second = component.render_session(observer, ALIAS)
    assert first.text == second.text and commits == [None, "first"]
    assert not component._texts and not component._requests


def test_pending_and_error_requests_are_bounded_with_explicit_retry(tmp_path, monkeypatch):
    rows = [row(f"{i:08x}-1234") for i in range(40)]
    component, observer, _, now = opened(tmp_path, rows)
    aliases = list(pm._session_buffer_ids(rows).values())
    monkeypatch.setattr(pm, "SESSION_OBSERVE_S", .001)
    handles, calls = {}, []
    def request(*args, **kwargs):
        sid = kwargs["session_id"]
        calls.append(sid)
        return handles.setdefault(sid, pending(tmp_path, sid))
    monkeypatch.setattr(session_load, "load_session", request)
    for alias in aliases[:8]:
        assert component.render_session(observer, alias).path.endswith("/pending")
    assert "capacity" in component.render_session(observer, aliases[8]).text
    assert len(calls) == len(component._requests) == 8
    sid, handle = next(iter(handles.items()))
    handle._value = loaded(sid)
    handle._done.set()
    assert component.render_session(observer, aliases[8]).path.endswith("/pending")
    assert len(calls) == 9 and len(component._requests) == 8
    for handle in handles.values():
        handle._error = OSError("pin capacity buffer is pinned " + "x" * 600)
        handle._done.set()
    def fail(*args, **kwargs):
        calls.append(kwargs["session_id"])
        raise OSError("pin capacity buffer is pinned " + "x" * 600)
    monkeypatch.setattr(session_load, "load_session", fail)
    for alias in aliases[1:]:
        assert component.render_session(observer, alias).path.endswith("/unavailable")
        assert len(component._requests) <= 8
    count = len(calls)
    assert len(component.render_session(observer, aliases[-1]).text) < 400
    assert len(calls) == count
    now[0] += 6
    component.render_session(observer, aliases[-1])
    assert len(calls) == count + 1


def test_project_roots_isolate_identical_aliases(tmp_path, monkeypatch):
    component, observer, _, _ = opened(tmp_path)
    monkeypatch.setattr(session_load, "load_session", lambda locator, **kw:
                        loaded(commit=str(locator.parent.parent.parent.name)))
    first = component.render_session(observer, ALIAS)
    other = tmp_path / "other-project"
    catalog(other, [row()])
    observer._agent_zoo_context["project_dir"] = str(other)
    component(observer)
    wait_for(lambda: "Inventory: ready" in component.render_index(observer).text)
    second = component.render_session(observer, ALIAS)
    assert first.path != second.path and len(component._pins) == 2


@pytest.mark.parametrize("full", [False, True])
def test_real_journal_registration_readonly_and_persistence_restore(tmp_path, full):
    projects.ensure_project("demo")
    root = tmp_path / "home" / "projects" / "demo"
    local = tmp_path / "local"
    source = Session()
    source.state._session_id = FIRST
    source.state.entries = [Entry(index=i, step=i, messages=[dict(role=role, content=text)])
                            for i, (role, text) in enumerate((("user", "Journal question"),
                                                             ("assistant", "Journal answer")))]
    locator = root / "sessions" / FIRST / "session.json"
    writer = SessionPersistence(locator, session_id=FIRST, instance_id="source", local_state_root=local)
    writers = [writer]
    def save(session, persistence):
        receipt = persistence.save(session, exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS)
        assert receipt.local_saved
        assert persistence.wait_for_shared(receipt.revision_id, timeout_s=4).shared_saved
        return receipt.revision_id
    def viewer(document=None):
        session = Session()
        session.state.buffer_manager = BufferManager()
        builder = PipelineBuilder(session).add(core()).add(buffer_editing())
        pm.register_features(builder, session=session, config={"project_memory": {
            "regex_memories": False, "project_sessions": True}})
        session.pipeline = builder.build()
        if document is not None:
            session.load_document(document)
        session._init_state()
        session.state._session_id = CURRENT
        session.state._agent_zoo_context = dict(project_name="demo", project_dir=str(root), local_state_root=str(local))
        component = next(c for c in flatten(session.pipeline) if isinstance(c, pm.ProjectSessionBuffers))
        component(session.state)
        live = next(c for c in flatten(session.pipeline) if isinstance(c, pm.ProjectLiveSessionBuffer))
        live(session.state)
        assert session.state.buffer_manager.resolve_for_read(session.state, "live_sessions").readonly
        wait_for(lambda: "Inventory: ready" in component.render_index(session.state).text)
        return session
    try:
        commit = save(source, writer)
        projects.register_session("demo", FIRST, title="Journal history")
        session = viewer()
        def read(session):
            session.state.step += 1
            return session.state.buffer_manager.resolve_for_read(session.state, ALIAS + ("_full" if full else ""))
        view = wait_for(lambda: (v if "Source Commit:" in (v := read(session)).text else None))
        assert f"Source Commit: {commit}" in view.text and "Source Format: journal" in view.text
        assert "Journal answer" in view.text and not locator.exists()
        assert view.readonly
        with pytest.raises(ValueError, match="readonly"):
            session.state.buffer_manager.require_writable(ALIAS + ("_full" if full else ""))
        target = root / "sessions" / CURRENT / "session.json"
        persistence = SessionPersistence(target, session_id=CURRENT, instance_id="viewer", local_state_root=local)
        writers.append(persistence)
        save(session, persistence)
        restored = session_load.load_session(target, session_id=CURRENT, local_state_root=local, timeout_s=2)
        assert isinstance(restored, LoadedSession)
        reopened = viewer(restored.state.document)
        text = wait_for(lambda: (v.text if "Source Commit:" in (v := read(reopened)).text else None))
        assert "Journal answer" in text and f"Source Commit: {commit}" in text
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
    views = {alias: component.render_session(observer, alias) for alias in aliases}
    assert commits == [None, "revision-1"]
    compact, full = views[ALIAS], views[ALIAS + "_full"]
    assert compact.readonly and full.readonly
    assert compact.path.endswith("/commits/revision-1")
    assert full.path.endswith("/commits/revision-1/full")
    assert f"Full buffer: {ALIAS}_full" in compact.text
    assert f"Compact buffer: {ALIAS}" in full.text
    assert "tool preamble" not in compact.text and "tool preamble" in full.text
    assert full.text.index("Old question") < full.text.index("New question")
    assert compact.text.index("New question") < compact.text.index("Old question")
    assert len(component._pins) == 1 and not component._requests
    assert component._text_bytes == sum(size for _, size in component._texts.values())
    now[0] += 10
    assert component.render_session(observer, ALIAS + "_full").text == full.text
    # Eviction is not refresh for either projection.
    component._texts.clear()
    component._text_bytes = 0
    monkeypatch.setattr(pm, "SESSION_PROJECTION_CACHE_BYTES", 1)
    for alias in aliases:
        assert component.render_session(observer, alias).text == views[alias].text
    assert commits[-2:] == ["revision-1", "revision-1"]
    assert not component._texts and component._text_bytes == 0


def test_registry_issued_aliases_do_not_need_catalog_membership(tmp_path, monkeypatch):
    component, observer, root, _ = opened(tmp_path, rows=[])
    aliases = component.issue_sessions(observer, [row(), row(), dict(row(CURRENT), kind="rlm"),
                                                 {"session_id": "../bad"}, {}])
    assert set(aliases) == {FIRST, CURRENT}
    monkeypatch.setattr(session_load, "load_session", lambda locator, **kw: loaded(kw["session_id"]))
    for sid, alias in aliases.items():
        assert f"Session ID: {sid}" in component.render_session(observer, alias + "_full").text
    assert "Known title" not in component.render_index(observer).text
    collision = "344655c1-2222-2222-2222-222222222222"
    added = component.issue_sessions(observer, [row(collision)])
    assert added[collision] != aliases[FIRST]
    assert component.issue_sessions(observer, [row()])[FIRST] == aliases[FIRST]


@pytest.mark.parametrize("alias", ["ses_344655c1_full_full", "ses_344655c1/full", "ses_short_full"])
def test_full_namespace_rejects_malformed_ids(tmp_path, alias):
    component, observer, _, _ = opened(tmp_path)
    assert component.render_session(observer, alias) is None


def test_simultaneous_variants_share_pending_load(tmp_path, monkeypatch):
    component, observer, _, _ = opened(tmp_path)
    handle, entered = pending(tmp_path), threading.Event()
    calls = []
    def request(locator, **kwargs):
        calls.append(kwargs["commit_id"])
        if kwargs["commit_id"]:
            return loaded(commit=kwargs["commit_id"])
        entered.set()
        return handle
    monkeypatch.setattr(session_load, "load_session", request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(component.render_session, observer, ALIAS + "_full")
        assert entered.wait(2)
        second = pool.submit(component.render_session, observer, ALIAS)
        assert second.result(timeout=2).path.endswith("/pending")
        assert calls == [None] and len(component._requests) == 1
        handle._value = loaded()
        handle._done.set()
        first_view = first.result(timeout=2)
    # The first wait may have timed out on a heavily loaded test host.
    if first_view.path.endswith("/pending"):
        first_view = component.render_session(observer, ALIAS + "_full")
    compact = component.render_session(observer, ALIAS)
    assert "Source Commit: first" in first_view.text and "Source Commit: first" in compact.text
    assert calls == [None, "first"]
    assert not component._requests


def test_registry_navigation_to_full_saved_tool_history(tmp_path, monkeypatch):
    from agent_utils import ToolCall
    from agent_zoo.live_session_registry import LiveSessionRecord, LiveProcessObservation
    from tmux_pilot.fs_store import RecordStore
    from tmux_pilot.process_identity import ProcessIdentity, ObservationState

    projects.ensure_project("demo")
    root = tmp_path / "home" / "projects" / "demo"
    local = tmp_path / "local"
    paths = ["/fixtures/backend-worktrees/topic", "/fixtures/tui-worktrees/topic"]
    args = json.dumps({"command": f"git worktree add {paths[0]} topic; git worktree add {paths[1]} topic"})
    call = ToolCall("make", "terminal_run", args, json.loads(args), result="Fixture checkout prepared")
    source = Session()
    source.state._session_id = FIRST
    source.state.entries = [
        Entry(index=0, step=0, messages=[{"role": "user", "content": "Implement the feature"}]),
        Entry(index=1, step=1, messages=[
            {"role": "assistant", "content": "Preparing checkouts", "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": args}}]},
            {"role": "tool", "tool_call_id": call.id, "content": call.result},
        ], tool_calls=[call]),
        Entry(index=2, step=2, messages=[{"role": "assistant", "content": "Implemented; final answer omits paths"}]),
    ]
    writer = SessionPersistence(root / "sessions" / FIRST / "session.json", session_id=FIRST,
                                instance_id="saved-instance", local_state_root=local)
    try:
        receipt = writer.save(source, exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS)
        assert receipt.local_saved
        assert writer.wait_for_shared(receipt.revision_id, timeout_s=4).shared_saved
        stamp = time.time()
        live_row = LiveSessionRecord(
            session_id=FIRST, instance_id="live-instance", pid=123, status="running",
            websocket_url="ws://127.0.0.1:8123", title="Feature work",
            process_identity=ProcessIdentity("fixture-host", "fixture-boot", "fixture-ns", 123, "fixture-start"),
            process_observation=LiveProcessObservation(ObservationState.ALIVE, "fixture", stamp, 1),
        )
        RecordStore(root / "coordination").put("live", live_row.instance_id, live_row.to_value())
        observer = state(root)
        observer._agent_zoo_context["local_state_root"] = str(local)
        history = pm.ProjectSessionBuffers()
        live = pm.ProjectLiveSessionBuffer(history=history)
        history(observer)
        live(observer)
        def read(alias):
            observer.step += 1
            return observer.buffer_manager.resolve_for_read(observer, alias)
        text = wait_for(lambda: (v.text if "Saved history:" in (v := read("live_sessions")).text else None))
        assert f"{ALIAS} | {ALIAS}_full" in text
        compact = wait_for(lambda: (v if "Source Commit:" in (v := read(ALIAS)).text else None))
        full = wait_for(lambda: (v if "Source Commit:" in (v := read(ALIAS + "_full")).text else None))
        for path in paths:
            assert path not in compact.text and path in full.text
        assert full.text.count(args) == 1
        assert "Source Instance: saved-instance" in full.text and "live-instance" in text
        assert f"Source Commit: {receipt.revision_id}" in compact.text
        assert f"Source Commit: {receipt.revision_id}" in full.text
        lines = full.text.splitlines()
        line = next(i for i, value in enumerate(lines) if "git worktree add" in value)
        assert "LOCAL FUNCTION REQUEST" in "\n".join(lines[max(0, line - 2):line + 2])
        assert read("live_sessions").readonly and full.readonly
    finally:
        writer.close()


def test_registry_only_session_without_saved_history_reports_unavailable(tmp_path):
    component, observer, root, _ = opened(tmp_path, rows=[])
    alias = component.issue_sessions(observer, [row()])[FIRST] + "_full"
    view = wait_for(lambda: (v if v.path.endswith("/unavailable") else None)
                    if (v := component.render_session(observer, alias)) else None)
    assert view.readonly
    assert "No saved entries recorded" not in view.text
    assert not (root / "sessions" / FIRST / "session.json").exists()
