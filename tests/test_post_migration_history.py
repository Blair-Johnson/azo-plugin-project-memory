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
