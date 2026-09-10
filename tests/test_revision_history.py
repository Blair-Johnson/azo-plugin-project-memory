from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import pytest
from agent_utils import Entry, Session, State
from agent_utils.checkpoint_revisions import publish_revision
from agent_utils.files.buffer_manager import BufferManager
from agent_zoo import plugins, projects

import project_memory as pm


CURRENT_ID = "aaaaaaaa-0000-0000-0000-000000000000"
FIRST_ID = "344655c1-1111-1111-1111-111111111111"
MISSING_ID = "97568885-2222-2222-2222-222222222222"
RLM_ID = "bbbbbbbb-3333-3333-3333-333333333333"


def _session(session_id: str, answer: str) -> Session:
    session = Session()
    session.state._session_id = session_id
    session.state.entries = [
        Entry(
            messages=[{"role": "user", "content": "What is the durable decision?"}],
            index=0,
            step=0,
        ),
        Entry(
            messages=[{"role": "assistant", "content": answer}],
            index=1,
            step=1,
        ),
    ]
    return session


def _configure_state_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "agent-zoo-home"
    monkeypatch.delenv("AGENT_ZOO_HOME", raising=False)
    monkeypatch.delenv("AGENT_ZOO_STATE_ROOT", raising=False)
    monkeypatch.setattr(projects, "_STATE_ROOT", root)
    return root


def _publish(
    project: str,
    session_id: str,
    answer: str,
    *,
    execution_id: str,
    parent_revision_id: str | None = None,
    expected_preferred_revision_id: str | None = None,
    title: str = "history",
    description: str = "immutable history",
    kind: str = "",
):
    projects.ensure_project(project)
    paths = projects.resolve_session_paths(project, session_id)
    publication = publish_revision(
        _session(session_id, answer),
        Path(paths["session_dir"]),
        session_id=session_id,
        execution_id=execution_id,
        parent_revision_id=parent_revision_id,
        expected_preferred_revision_id=expected_preferred_revision_id,
    )
    projects.register_session(
        project,
        session_id,
        title=title,
        description=description,
        kind=kind,
    )
    return publication, Path(paths["project_dir"]), Path(paths["state_file"])


def _publish_at(project_dir: Path, session_id: str, answer: str, *, execution_id: str):
    repository = project_dir / "sessions" / session_id
    return publish_revision(
        _session(session_id, answer),
        repository,
        session_id=session_id,
        execution_id=execution_id,
    )


def _write_catalog(project_dir: Path, session_id: str, *, title: str, updated_at: float):
    catalog = project_dir / "sessions" / "index.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        json.dumps(
            [
                {
                    "session_id": session_id,
                    "title": title,
                    "description": title,
                    "kind": "",
                    "created_at": updated_at,
                    "updated_at": updated_at,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _state(project_dir: Path, current_session_id: str, local_root: Path) -> State:
    state = State(token_budget=10_000)
    state.step = 1
    state._session_id = current_session_id
    state.buffer_manager = BufferManager()
    state._agent_zoo_context = {
        "project_name": "demo",
        "project_dir": str(project_dir),
        "session_id": current_session_id,
        "project_session_index": str(project_dir / "sessions" / "index.json"),
        "local_state_root": str(local_root),
    }
    return state


def _wait_for(predicate, *, timeout: float = 4.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.01)
    pytest.fail(f"condition did not settle; last={last!r}")


def _index_view(state: State):
    state.step += 1
    return state.buffer_manager.resolve_for_read(state, pm.PROJECT_SESSIONS_BUFFER_ID)


def test_inventory_is_background_metadata_only_and_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    publication, project_dir, _state_file = _publish(
        "demo",
        FIRST_ID,
        "The first immutable answer.",
        execution_id="first-execution",
        title="First history",
    )
    projects.ensure_project("demo")
    projects.register_session("demo", CURRENT_ID, title="Current")
    projects.register_session("demo", MISSING_ID, title="Catalog-only row")
    projects.register_session("demo", RLM_ID, title="RLM child", kind="rlm")

    entered = threading.Event()
    release = threading.Event()
    real_reader = projects._read_observational_json

    def blocked_reader(path):
        entered.set()
        assert release.wait(timeout=2.0)
        return real_reader(path)

    monkeypatch.setattr(projects, "_read_observational_json", blocked_reader)
    state = _state(project_dir, CURRENT_ID, tmp_path / "local")
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)
    component(state)
    assert entered.wait(timeout=2.0)

    pending = _index_view(state)
    assert "Inventory: pending" in pending.text
    assert "Sessions: 0" in pending.text
    assert publication.revision.revision_id not in pending.text

    release.set()
    ready = _wait_for(
        lambda: (
            view
            if "Inventory: ready" in (view := _index_view(state)).text
            else None
        )
    )
    assert FIRST_ID in ready.text
    assert MISSING_ID in ready.text
    assert CURRENT_ID not in ready.text
    assert RLM_ID not in ready.text
    assert "Catalog-only row" in ready.text
    assert "Source Transcript" not in ready.text


def test_revision_projection_uses_real_au_payload_and_pins_preferred_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    first, project_dir, state_file = _publish(
        "demo",
        FIRST_ID,
        "The first immutable answer.",
        execution_id="first-execution",
        title="Pinned history",
    )
    state = _state(project_dir, CURRENT_ID, tmp_path / "local")
    now = [1_700_000_000.0]
    component = pm.ProjectSessionBuffers(clock=lambda: now[0])
    component(state)
    _wait_for(lambda: "Inventory: ready" in _index_view(state).text)

    def loaded_first():
        view = component.render_session(state, "ses_344655c1")
        return view if view and "/revisions/" in view.path else None

    first_view = _wait_for(loaded_first)
    assert first.revision.revision_id in first_view.path
    assert "The first immutable answer." in first_view.text
    assert f"Source Revision: {first.revision.revision_id}" in first_view.text
    assert "USER [entry 0, lines " in first_view.text
    assert "ASSISTANT [entry 1, lines " in first_view.text

    local_repo = __import__(
        "agent_zoo.checkpoint_sync", fromlist=["local_repository"]
    ).local_repository(state_file, local_state_root=tmp_path / "local")
    assert str(local_repo / "revisions" / first.revision.revision_id / "session.json") in first_view.text

    projects.register_session(
        "demo",
        FIRST_ID,
        title="Changed catalog title",
        description="changed catalog description",
    )
    now[0] += 6.0
    component(state)
    _wait_for(lambda: "Inventory: ready" in _index_view(state).text)
    metadata_changed = component.render_session(state, "ses_344655c1")
    assert metadata_changed is not None
    assert "Title: Pinned history" in metadata_changed.text
    assert "Changed catalog title" not in metadata_changed.text
    assert "immutable history" in metadata_changed.text
    assert "changed catalog description" not in metadata_changed.text

    second, _project_dir, _state_file = _publish(
        "demo",
        FIRST_ID,
        "The newer preferred answer.",
        execution_id="second-execution",
        parent_revision_id=first.revision.revision_id,
        expected_preferred_revision_id=first.revision.revision_id,
        title="Pinned history",
    )
    assert second.preferred == "updated"

    pinned_again = component.render_session(state, "ses_344655c1")
    assert pinned_again is not None
    assert first.revision.revision_id in pinned_again.path
    assert second.revision.revision_id not in pinned_again.path
    assert "The first immutable answer." in pinned_again.text
    assert "The newer preferred answer." not in pinned_again.text


def test_inventory_unavailability_preserves_last_good_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    _publication, project_dir, _state_file = _publish(
        "demo",
        FIRST_ID,
        "Last good answer.",
        execution_id="first-execution",
        title="Last good",
    )
    state = _state(project_dir, CURRENT_ID, tmp_path / "local")
    now = [100.0]
    component = pm.ProjectSessionBuffers(clock=lambda: now[0])
    component(state)
    _wait_for(lambda: "Inventory: ready" in _index_view(state).text)

    def unavailable_reader(*_args, **_kwargs):
        raise OSError("shared catalog unavailable")

    monkeypatch.setattr(projects, "_read_observational_json", unavailable_reader)
    now[0] = 106.0
    component(state)
    unavailable = _wait_for(
        lambda: (
            view
            if "Inventory: unavailable" in (view := _index_view(state)).text
            else None
        )
    )
    assert "last good metadata snapshot" in unavailable.text
    assert FIRST_ID in unavailable.text
    assert "Last good" in unavailable.text


def test_installed_plugin_loader_and_state_callbacks_are_persistence_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plugin_root = tmp_path / "plugins"
    common = plugin_root / "common"
    common.mkdir(parents=True)
    (common / "project_memory.py").symlink_to(
        Path(__file__).resolve().parents[1] / "src" / "project_memory.py"
    )
    monkeypatch.setattr(plugins, "_BASE_DIR", plugin_root)

    class Builder:
        def __init__(self):
            self.features = []

        def add(self, feature):
            self.features.append(feature)

    builder = Builder()
    plugins.apply_pipeline_plugins(
        builder,
        pipeline_name="default",
        session=Session(),
        config={"project_memory": {"enabled": True, "project_sessions": {"enabled": True}}},
    )
    feature = next(item for item in builder.features if item.name == "project_memory")
    assert any(type(item).__name__ == "ProjectSessionBuffers" for item in feature.components)

    _configure_state_root(monkeypatch, tmp_path / "state")
    state = _state(tmp_path / "state" / "projects" / "demo", CURRENT_ID, tmp_path / "local")
    state._agent_zoo_context["project_name"] = "never-created"
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)
    component(state)
    session = Session()
    session.pipeline = [component]
    session.state = state
    output = tmp_path / "session.json"
    session.save(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert pm.PROJECT_SESSION_RENDER_ATTR not in document.get("attrs", {})
    assert pm.PROJECT_SESSION_INDEX_RENDER_ATTR not in document.get("attrs", {})


def test_transient_revision_unavailability_retries_on_a_later_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    _publication, project_dir, _state_file = _publish(
        "demo",
        FIRST_ID,
        "Eventually available answer.",
        execution_id="retry-execution",
    )
    state = _state(project_dir, CURRENT_ID, tmp_path / "local")
    now = [100.0]
    component = pm.ProjectSessionBuffers(clock=lambda: now[0])
    component(state)
    _wait_for(lambda: "Inventory: ready" in _index_view(state).text)

    from agent_zoo import checkpoint_sync

    real_load = checkpoint_sync.load_checkpoint
    calls = []

    def flaky_load(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary revision read failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_sync, "load_checkpoint", flaky_load)
    failed = _wait_for(
        lambda: (
            view
            if "unavailable" in (view := component.render_session(state, "ses_344655c1")).text
            else None
        )
    )
    assert "temporary revision read failure" in failed.text

    now[0] += pm.SESSION_LOAD_RETRY_S + 1.0
    loaded = _wait_for(
        lambda: (
            view
            if "/revisions/" in (view := component.render_session(state, "ses_344655c1")).path
            else None
        )
    )
    assert len(calls) >= 2
    assert "Eventually available answer." in loaded.text


def test_project_switch_scopes_results_and_reads_exact_catalog_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    root_a = tmp_path / "project-a" / "projects" / "demo"
    root_b = tmp_path / "project-b" / "projects" / "demo"
    first = _publish_at(root_a, FIRST_ID, "Answer from project A.", execution_id="a")
    second = _publish_at(root_b, FIRST_ID, "Answer from project B.", execution_id="b")
    _write_catalog(root_a, FIRST_ID, title="Project A", updated_at=1.0)
    _write_catalog(root_b, FIRST_ID, title="Project B", updated_at=2.0)

    state = _state(root_a, CURRENT_ID, tmp_path / "local")
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)
    component(state)
    _wait_for(lambda: "Project A" in _index_view(state).text)
    from_a = _wait_for(
        lambda: (
            view
            if "/revisions/" in (view := component.render_session(state, "ses_344655c1")).path
            else None
        )
    )
    assert first.revision.revision_id in from_a.path
    assert "Answer from project A." in from_a.text

    state._agent_zoo_context["project_dir"] = str(root_b)
    state._agent_zoo_context["project_session_index"] = str(root_b / "sessions" / "index.json")
    component(state)
    _wait_for(lambda: "Project B" in _index_view(state).text)
    from_b = _wait_for(
        lambda: (
            view
            if "/revisions/" in (view := component.render_session(state, "ses_344655c1")).path
            else None
        )
    )
    assert second.revision.revision_id in from_b.path
    assert "Answer from project B." in from_b.text
    assert "Answer from project A." not in from_b.text


def test_pin_capacity_refuses_new_buffer_without_retargeting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _configure_state_root(monkeypatch, tmp_path)
    _publication, project_dir, _state_file = _publish(
        "demo",
        FIRST_ID,
        "Capacity answer.",
        execution_id="capacity-execution",
    )
    state = _state(project_dir, CURRENT_ID, tmp_path / "local")
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)
    component(state)
    _wait_for(lambda: "Inventory: ready" in _index_view(state).text)
    project_key = component._project_key("demo", project_dir)
    for index in range(pm.SESSION_BUFFER_RECORD_LIMIT):
        component._pinned_revisions[(project_key, f"ses_fake{index:04d}")] = {
            "revision_id": f"revision-{index}",
            "session_id": f"session-{index}",
        }

    refused = component.render_session(state, "ses_344655c1")
    assert refused is not None
    assert "pin capacity" in refused.text
    assert (project_key, "ses_344655c1") not in component._pinned_revisions


def test_registration_defers_history_discovery_until_context_exists():
    state = State(token_budget=10_000)
    state.buffer_manager = BufferManager()
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)

    component(state)
    index = state.buffer_manager.resolve_for_read(state, pm.PROJECT_SESSIONS_BUFFER_ID)
    session = state.buffer_manager.resolve_for_read(state, "ses_344655c1")

    assert "Inventory: pending" in index.text
    assert "context is not attached" in index.text
    assert "pending" in session.text
    assert "context is not attached" in session.text
