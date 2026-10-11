"""Independent feature toggles and real pipeline composition contracts."""
from types import SimpleNamespace

import pytest
from agent_utils import Session, State
from agent_utils.files.buffer_manager import BufferManager
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes

import project_memory as pm


MEMORY_TOOL_NAMES = {"record_memory", "update_memory", "suppress_memory", "delete_memory"}


@pytest.mark.parametrize("sessions_enabled,live_enabled", [
    (False, True), (True, False), (False, False), (True, True),
])
@pytest.mark.parametrize("nested", [False, True])
def test_independent_feature_toggles_match_installed_contract(
    monkeypatch, sessions_enabled, live_enabled, nested,
):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    value = lambda enabled: {"enabled": enabled} if nested else enabled
    pm.register_features(
        SimpleNamespace(add=features.append), session=None,
        config={"project_memory": {
            "project_sessions": value(sessions_enabled),
            "live_sessions": value(live_enabled),
        }},
    )
    if not any((sessions_enabled, live_enabled)):
        assert features == []
        return
    assert len(features) == 1
    components = features[0].components
    types = {type(component) for component in components}
    assert (pm.ProjectSessionBuffers in types) == sessions_enabled
    assert (pm.ProjectLiveSessionBuffer in types) == live_enabled
    assert MEMORY_TOOL_NAMES.isdisjoint(getattr(component, "name", None) for component in components)
    prompt = next(component for component in components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    assert (pm.SESSION_SYSTEM_PROMPT in prompt.text) == sessions_enabled
    assert (pm.LIVE_SESSION_SYSTEM_PROMPT in prompt.text) == live_enabled
    if live_enabled:
        live = next(c for c in components if isinstance(c, pm.ProjectLiveSessionBuffer))
        # History-only navigation is optional, not a dependency of discovery.
        assert (live.history is not None) == sessions_enabled


@pytest.mark.parametrize("builder,modes", [
    (pipelines.build_default_pipeline, default_modes),
    (pipelines.build_readonly_rlm_pipeline, rlm_modes),
])
def test_current_pipeline_builds_history_only_without_memory_tools(builder, modes, monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    def install(builder, *, session, config, **kwargs):
        pm.register_features(builder, session=session, config=config)
    monkeypatch.setattr(pipelines, "apply_pipeline_plugins", install)
    with Session(system_prompt="test") as session:
        pipeline = builder(
            session,
            config={"project_memory": {
                "project_sessions": {"enabled": True},
                "live_sessions": {"enabled": False},
            }},
            skill_paths=[], scheduler_type="local", max_idle=0,
            mode_defs=modes(), initial_mode="execution", terminal_backend="headless",
            tmux_tools_enabled=False,
        )
    types = {type(component) for component in pipeline}
    assert pm.ProjectSessionBuffers in types
    assert pm.ProjectLiveSessionBuffer not in types
    assert MEMORY_TOOL_NAMES.isdisjoint(getattr(component, "name", None) for component in pipeline)


@pytest.mark.parametrize("builder,modes", [
    (pipelines.build_default_pipeline, default_modes),
    (pipelines.build_readonly_rlm_pipeline, rlm_modes),
])
def test_pipeline_builds_live_registry_only(builder, modes, monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    def install(builder, *, session, config, **kwargs):
        pm.register_features(builder, session=session, config=config)
    monkeypatch.setattr(pipelines, "apply_pipeline_plugins", install)
    with Session(system_prompt="test") as session:
        pipeline = builder(
            session, config={"project_memory": {
                "project_sessions": False, "live_sessions": True,
            }}, skill_paths=[], scheduler_type="local", max_idle=0,
            mode_defs=modes(), initial_mode="execution", terminal_backend="headless",
            tmux_tools_enabled=False,
        )
    types = {type(component) for component in pipeline}
    assert pm.ProjectLiveSessionBuffer in types
    assert pm.ProjectSessionBuffers not in types
    assert MEMORY_TOOL_NAMES.isdisjoint(getattr(component, "name", None) for component in pipeline)


@pytest.mark.parametrize("enabled", [False, "false"])
def test_master_disable_is_preserved(monkeypatch, enabled):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None,
                         config={"project_memory": {"enabled": enabled}})
    assert features == []


@pytest.mark.parametrize("origin", ["installed", "launch"])
def test_legacy_memory_config_cannot_restore_tools_or_buffer(tmp_path, monkeypatch, origin):
    legacy = {"project_memory": {
        "regex_memories": {"enabled": True},
        "common_memories": {
            "legacy": {"enabled": True, "content": "Obsolete guidance.", "trigger": ".*"},
        },
    }}
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config",
                        lambda: legacy if origin == "installed" else {})
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None,
                         config=legacy if origin == "launch" else {})

    assert len(features) == 1
    components = features[0].components
    assert {type(component) for component in components} == {
        pm.ProjectSessionBuffers, pm.ProjectLiveSessionBuffer, pm.ProjectMemorySystemPrompt,
    }
    assert MEMORY_TOOL_NAMES.isdisjoint(getattr(component, "name", None) for component in components)
    state = State(token_budget=10_000)
    state.entries = []
    state.buffer_manager = BufferManager()
    state._agent_zoo_context = {"project_name": "demo", "project_dir": str(tmp_path / "project")}
    for component in components:
        component(state)
    for buffer_id in ("project_sessions", "live_sessions"):
        assert state.buffer_manager.resolve_for_read(state, buffer_id).readonly
    with pytest.raises(KeyError, match="project_memory"):
        state.buffer_manager.resolve_for_read(state, "project_memory")
