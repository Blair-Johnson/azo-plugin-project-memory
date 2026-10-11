"""Independent feature toggles and real pipeline composition contracts."""
from types import SimpleNamespace

import pytest
from agent_utils import Session
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes

import project_memory as pm


@pytest.mark.parametrize("regex_enabled,sessions_enabled", [
    (False, True), (True, False), (False, False), (True, True),
])
@pytest.mark.parametrize("live_enabled", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_independent_feature_toggles_match_installed_contract(
    monkeypatch, regex_enabled, sessions_enabled, live_enabled, nested,
):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    value = lambda enabled: {"enabled": enabled} if nested else enabled
    pm.register_features(
        SimpleNamespace(add=features.append), session=Session(),
        config={"project_memory": {
            "regex_memories": value(regex_enabled),
            "project_sessions": value(sessions_enabled),
            "live_sessions": value(live_enabled),
        }},
    )
    if not any((regex_enabled, sessions_enabled, live_enabled)):
        assert features == []
        return
    assert len(features) == 1
    types = {type(component) for component in features[0].components}
    assert (pm.ProjectSessionBuffers in types) == sessions_enabled
    assert (pm.ProjectLiveSessionBuffer in types) == live_enabled
    assert (pm.ProjectMemoryStore in types) == regex_enabled
    assert (pm.ProjectMemoryRecall in types) == regex_enabled
    assert (pm.RecordMemory in types) == regex_enabled
    prompt = next(component for component in features[0].components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    assert (pm.CORE_SYSTEM_PROMPT in prompt.text) == regex_enabled
    assert (pm.SESSION_SYSTEM_PROMPT in prompt.text) == sessions_enabled
    assert (pm.LIVE_SESSION_SYSTEM_PROMPT in prompt.text) == live_enabled
    if live_enabled:
        live = next(c for c in features[0].components if isinstance(c, pm.ProjectLiveSessionBuffer))
        # History-only navigation is optional, not a dependency of discovery.
        assert (live.history is not None) == sessions_enabled


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
                "regex_memories": False, "project_sessions": False, "live_sessions": True,
            }}, skill_paths=[], scheduler_type="local", max_idle=0,
            mode_defs=modes(), initial_mode="execution", terminal_backend="headless",
            tmux_tools_enabled=False,
        )
    types = {type(component) for component in pipeline}
    assert pm.ProjectLiveSessionBuffer in types
    assert pm.ProjectSessionBuffers not in types
    assert pm.ProjectMemoryStore not in types


def test_master_disable_and_explicit_prompt_are_preserved(monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    builder = SimpleNamespace(add=features.append)
    pm.register_features(builder, session=Session(), config={"project_memory": {"enabled": False}})
    assert not features
    pm.register_features(builder, session=Session(), config={"project_memory": {
        "regex_memories": False, "project_sessions": False, "live_sessions": True,
        "system_prompt": "custom instructions",
    }})
    prompt = next(c for c in features[0].components if isinstance(c, pm.ProjectMemorySystemPrompt))
    assert prompt.text == "custom instructions"
