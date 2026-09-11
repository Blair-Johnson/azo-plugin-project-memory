"""C2 regressions against real current AZ/AU immutable checkpoint APIs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils import Session
from agent_utils.checkpoint_revisions import select_revision
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes

import project_memory as pm
from test_revision_history import (
    CURRENT_ID, FIRST_ID, _configure_state_root, _index_view, _publish,
    _session, _state, _wait_for,
)

BUFFER_ID = "ses_344655c1"


def _open_history(project_dir, local_root):
    state = _state(project_dir, CURRENT_ID, local_root)
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)
    component(state)
    index = _wait_for(lambda: (view if "Inventory: ready" in
                             (view := _index_view(state)).text else None))
    assert FIRST_ID in index.text
    return state, component


def _settled(state, component):
    return _wait_for(lambda: (view if "/pending" not in
                             (view := component.render_session(state, BUFFER_ID)).path else None))


@pytest.mark.parametrize("preserve_legacy", [False, True])
def test_immutable_only_and_preserved_legacy_load_latest_selected(
    tmp_path, monkeypatch, preserve_legacy,
):
    _configure_state_root(monkeypatch, tmp_path)
    first, project_dir, locator = _publish(
        "demo", FIRST_ID, "Older immutable answer.", execution_id="first",
    )
    assert not locator.exists()
    if preserve_legacy:
        _session(FIRST_ID, "OBSOLETE LOOSEFILE MUST NEVER APPEAR").save(locator)
    before = locator.read_bytes() if preserve_legacy else None
    second, _, _ = _publish(
        "demo", FIRST_ID, "Latest selected immutable answer.", execution_id="second",
        parent_revision_id=first.revision.revision_id,
        expected_preferred_revision_id=first.revision.revision_id,
    )
    state, component = _open_history(project_dir, tmp_path / "local")
    view = _settled(state, component)
    assert second.revision.revision_id in view.path
    assert "Latest selected immutable answer." in view.text
    assert "Older immutable answer." not in view.text
    assert "OBSOLETE LOOSEFILE" not in view.text
    assert f"Source Revision: {second.revision.revision_id}" in view.text
    assert "/revisions/" in view.text
    assert (locator.read_bytes() if preserve_legacy else None) == before
    assert locator.exists() == preserve_legacy


def test_preferred_selection_is_not_replaced_by_newest_timestamp(tmp_path, monkeypatch):
    _configure_state_root(monkeypatch, tmp_path)
    first, project_dir, locator = _publish(
        "demo", FIRST_ID, "Deliberately selected older answer.", execution_id="first",
    )
    second, _, _ = _publish(
        "demo", FIRST_ID, "Newer but unselected answer.", execution_id="second",
        parent_revision_id=first.revision.revision_id,
        expected_preferred_revision_id=first.revision.revision_id,
    )
    selection = select_revision(
        locator.parent, first.revision.revision_id,
        expected_preferred_revision_id=second.revision.revision_id,
    )
    assert selection.preferred == "updated"
    state, component = _open_history(project_dir, tmp_path / "local")
    view = _settled(state, component)
    assert first.revision.revision_id in view.path
    assert "Deliberately selected older answer." in view.text
    assert "Newer but unselected answer." not in view.text


def test_invalid_preferred_falls_back_to_latest_valid_immutable_revision(tmp_path, monkeypatch):
    _configure_state_root(monkeypatch, tmp_path)
    first, project_dir, locator = _publish(
        "demo", FIRST_ID, "Latest valid answer.", execution_id="first",
    )
    second, _, _ = _publish(
        "demo", FIRST_ID, "CORRUPT PREFERRED ANSWER", execution_id="second",
        parent_revision_id=first.revision.revision_id,
        expected_preferred_revision_id=first.revision.revision_id,
    )
    (locator.parent / "revisions" / second.revision.revision_id / "session.json").write_text("{}")
    _session(FIRST_ID, "OBSOLETE LOOSEFILE ANSWER").save(locator)
    state, component = _open_history(project_dir, tmp_path / "local")
    view = _settled(state, component)
    assert first.revision.revision_id in view.path
    assert "Latest valid answer." in view.text
    assert "CORRUPT PREFERRED" not in view.text
    assert "OBSOLETE LOOSEFILE" not in view.text


@pytest.mark.parametrize("source_failure", ["corrupt", "missing"])
def test_no_valid_revision_is_unavailable_not_legacy_or_empty_success(
    tmp_path, monkeypatch, source_failure,
):
    _configure_state_root(monkeypatch, tmp_path)
    published, project_dir, locator = _publish(
        "demo", FIRST_ID, "Unusable immutable answer.", execution_id="first",
    )
    payload = locator.parent / "revisions" / published.revision.revision_id / "session.json"
    if source_failure == "corrupt":
        payload.write_text("{}")
    else:
        payload.unlink()
    _session(FIRST_ID, "OBSOLETE LOOSEFILE ANSWER").save(locator)
    before = locator.read_bytes()
    state, component = _open_history(project_dir, tmp_path / "local")
    view = _settled(state, component)
    assert view.path.endswith("/unavailable")
    assert "No complete checkpoint revisions" in view.text
    assert "OBSOLETE LOOSEFILE" not in view.text
    assert "Source Revision:" not in view.text
    assert not component._pinned_revisions
    assert locator.read_bytes() == before


def test_pin_is_exact_until_restart_and_local_cache_is_explicit(tmp_path, monkeypatch):
    _configure_state_root(monkeypatch, tmp_path)
    first, project_dir, _ = _publish(
        "demo", FIRST_ID, "Original cached answer.", execution_id="first",
    )
    state, component = _open_history(project_dir, tmp_path / "local")
    original = _settled(state, component)
    second, _, _ = _publish(
        "demo", FIRST_ID, "New shared answer.", execution_id="second",
        parent_revision_id=first.revision.revision_id,
        expected_preferred_revision_id=first.revision.revision_id,
    )
    pinned = _settled(state, component)
    assert pinned.path == original.path
    assert pinned.text == original.text
    assert "Pinned" in pinned.text
    assert "backend restart" in pinned.text

    # Restart releases in-memory pins, NOT the backend's durable local spool.
    restarted_state, restarted = _open_history(project_dir, tmp_path / "local")
    cached = _settled(restarted_state, restarted)
    assert first.revision.revision_id in cached.path
    assert "Original cached answer." in cached.text
    assert "loaded local cached revision; shared storage was not checked" in cached.text
    assert "Checkpoint source: local" in cached.text

    # A genuinely uncached reader selects the current shared preferred checkpoint.
    fresh_state, fresh = _open_history(project_dir, tmp_path / "fresh-local")
    latest = _settled(fresh_state, fresh)
    assert second.revision.revision_id in latest.path
    assert "New shared answer." in latest.text
    assert "Checkpoint source: shared" in latest.text


@pytest.mark.parametrize("regex_enabled,sessions_enabled", [
    (False, True), (True, False), (False, False), (True, True),
])
@pytest.mark.parametrize("nested", [False, True])
def test_independent_feature_toggles_match_installed_contract(
    monkeypatch, regex_enabled, sessions_enabled, nested,
):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    value = lambda enabled: {"enabled": enabled} if nested else enabled
    pm.register_features(
        SimpleNamespace(add=features.append), session=Session(),
        config={"project_memory": {
            "regex_memories": value(regex_enabled),
            "project_sessions": value(sessions_enabled),
        }},
    )
    if not regex_enabled and not sessions_enabled:
        assert features == []
        return
    assert len(features) == 1
    types = {type(component) for component in features[0].components}
    assert (pm.ProjectSessionBuffers in types) == sessions_enabled
    assert (pm.ProjectMemoryStore in types) == regex_enabled
    assert (pm.ProjectMemoryRecall in types) == regex_enabled
    assert (pm.RecordMemory in types) == regex_enabled
    prompt = next(component for component in features[0].components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    if sessions_enabled and not regex_enabled:
        assert types == {pm.ProjectSessionBuffers, pm.ProjectMemorySystemPrompt}
        assert prompt.text == pm.SESSION_SYSTEM_PROMPT
    elif regex_enabled and not sessions_enabled:
        assert prompt.text == pm.CORE_SYSTEM_PROMPT


@pytest.mark.parametrize("builder,modes", [
    (pipelines.build_default_pipeline, default_modes),
    (pipelines.build_readonly_rlm_pipeline, rlm_modes),
])
def test_current_pipeline_builds_history_only_without_regex_tools(builder, modes, monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    def install(builder, *, session, config, **kwargs):
        pm.register_features(builder, session=session, config=config)
    monkeypatch.setattr(pipelines, "apply_pipeline_plugins", install)
    with Session(system_prompt="test") as session:
        pipeline = builder(
            session,
            config={"project_memory": {
                "regex_memories": {"enabled": False},
                "project_sessions": {"enabled": True},
            }},
            skill_paths=[], scheduler_type="local", max_idle=0,
            mode_defs=modes(), initial_mode="execution", terminal_backend="headless",
            tmux_tools_enabled=False,
        )
    types = {type(component) for component in pipeline}
    assert pm.ProjectSessionBuffers in types
    assert pm.ProjectMemoryStore not in types
    assert pm.ProjectMemoryRecall not in types
    assert not any(isinstance(component, pm.ProjectMemoryTool) for component in pipeline)
